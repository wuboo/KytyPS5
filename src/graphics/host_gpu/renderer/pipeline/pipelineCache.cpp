#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "graphics/host_gpu/renderer/pipeline/matMemoShadow.h"
#include "graphics/host_gpu/renderer/syncCensus.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/perVertexPrototype.h"
#include "graphics/host_gpu/renderer/perVertexVsStoreRuntime.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/cpSampler.h"
#include "graphics/shader/recompiler/ir/passes/srtWalkPaths.h"
#include "kernel/specPageCache.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/presentation/srtFast.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/specCond.h"
#include "graphics/shader/recompiler/ir/passes/specShadow.h"
#include "graphics/shader/recompiler/rtMiss.h"
#include "graphics/shader/nullDescDiag.h"
#include "graphics/shader/recompiler/srtReader.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_set>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

// Bumped by hand when the emulator-side layout of the cache file changes. The blob itself is the
// driver's vkGetPipelineCacheData output, which the driver validates and keys by shader content,
// so a rebuilt emulator keeps its warm cache; entries for changed shaders simply do not hit.
constexpr uint32_t kPipelineCacheFormat = 1;

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	return fmt::format("KytyPC3:{}:{:08x}:{:08x}:{:08x}:{}\n", kPipelineCacheFormat,
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

const char* SyncStageName(ShaderType stage) {
	switch (stage) {
		case ShaderType::Vertex: return "vs";
		case ShaderType::Mesh: return "ms";
		case ShaderType::Local: return "ls";
		case ShaderType::TessellationControl: return "hs";
		case ShaderType::TessellationEvaluation: return "ds";
		case ShaderType::Pixel: return "ps";
		case ShaderType::Compute: return "cs";
		default: return "?";
	}
}

// ---- S406 (KYTY_DIAG_MATMEMO=1): shadow memo after the walk, see matMemoShadow.h ----
bool MatMemoEnabled() {
	static const bool on = FrameTiming::DiagFlagOn(std::getenv("KYTY_DIAG_MATMEMO"));
	return on;
}

uint64_t HashDescriptors(uint64_t hash, const std::vector<ShaderRecompiler::IR::DescriptorValue>& values, bool mask_base48) {
	hash = MatMemo::Mix(hash, values.size());
	for (const auto& value: values) {
		auto dwords = value.dwords;
		if (mask_base48) {
			dwords[0] = 0;
			dwords[1] &= 0xFFFF0000u;
		}
		hash = MatMemo::Mix(hash, value.dword_count);
		hash = MatMemo::HashWords(hash, dwords);
	}
	return hash;
}

uint64_t HashSpecialization(uint64_t hash, const ShaderRecompiler::IR::ResourceSpecialization& spec) {
	for (const auto& b: spec.buffers) {
		hash = MatMemo::Mix(hash, b.packed_stride);
		hash = MatMemo::Mix(hash, static_cast<uint64_t>(b.descriptor_format));
		hash = MatMemo::Mix(hash, b.descriptor_swizzle);
		hash = MatMemo::Mix(hash, (b.zero_stride_oob ? 1u : 0u) | (b.scalar_empty ? 2u : 0u));
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(b.indirect_root) << 32) | b.indirect_mapping_offset);
		hash = MatMemo::Mix(hash, b.indirect_search_iterations);
	}
	hash = MatMemo::Mix(hash, 0x1111);
	for (const auto& i: spec.images) {
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(i.numeric_class) << 32) | static_cast<uint64_t>(i.dimension));
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(i.mip_count) << 32) | static_cast<uint64_t>(i.conversion_format));
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(i.shader_swizzle) << 32) | i.indirect_root);
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(i.indirect_mapping_offset) << 32) | i.indirect_search_iterations);
		hash = MatMemo::Mix(hash, (i.cube ? 1u : 0u) | (i.fmask ? 2u : 0u));
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(i.min_lod_resinfo.shift) << 32) | i.min_lod_resinfo.levels);
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(i.min_lod_resinfo.extent[0]) << 32) | i.min_lod_resinfo.extent[1]);
		hash = MatMemo::Mix(hash, i.min_lod_resinfo.extent[2]);
	}
	hash = MatMemo::Mix(hash, 0x2222);
	for (const auto& s: spec.samplers) {
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(s.indirect_root) << 32) | s.indirect_mapping_offset);
		hash = MatMemo::Mix(hash, s.indirect_search_iterations);
	}
	return hash;
}

// Output of the part after the walk: snapshot (buffers, images, samplers, the user_data words in `used_mask`, flattened SRT, uniform
// fill), specialization and the chosen permutation (UINT32_MAX = none yet).
MatMemo::Output HashMaterializationOutput(const ShaderRecompiler::IR::ResourceSnapshot& snapshot,
                                          const ShaderRecompiler::IR::ResourceSpecialization& spec, uint64_t used_mask,
                                          uint32_t permutation) {
	MatMemo::Output out;
	for (int masked = 0; masked < 2; masked++) {
		uint64_t hash = 0x6f757470757401ull;
		hash          = HashDescriptors(hash, snapshot.buffers, masked != 0);
		hash          = HashDescriptors(hash, snapshot.images, false);
		hash          = HashDescriptors(hash, snapshot.samplers, false);
		for (size_t i = 0; i < snapshot.user_data.size() && i < 64; i++) {
			if (((used_mask >> i) & 1u) != 0) hash = MatMemo::Mix(hash, (static_cast<uint64_t>(i) << 32) | snapshot.user_data[i]);
		}
		hash = MatMemo::HashWords(hash, snapshot.flattened_srt);
		const auto& fill = snapshot.uniform_fill;
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(fill.kind) << 32) | fill.resource);
		hash = MatMemo::Mix(hash, (static_cast<uint64_t>(fill.words) << 32) | fill.value);
		hash = MatMemo::Mix(hash, fill.bounded_elements);
		hash = HashSpecialization(hash, spec);
		hash = MatMemo::Mix(hash, permutation);
		(masked != 0 ? out.masked : out.full) = hash;
	}
	return out;
}

struct MatMemoLog {
	MatMemo::Report report;
	uint64_t        last_bucket = 0;
};
MatMemoLog& MatMemoLogState() {
	static MatMemoLog state;
	return state;
}

void NoteMatMemoResult(uint64_t program_hash, uint32_t stage, const MatMemo::Result& result) {
	using FrameTiming::Add;
	using FrameTiming::MatMemoCounter;
	using FrameTiming::MatMemoMetric;
	Add(MatMemoCounter(MatMemoMetric::N, stage));
	if (result.in_hit) Add(MatMemoCounter(MatMemoMetric::InHit, stage));
	if (result.unsafe) Add(MatMemoCounter(MatMemoMetric::Unsafe, stage));
	if (result.out_hit) Add(MatMemoCounter(MatMemoMetric::OutHit, stage));
	if (result.rebase) Add(MatMemoCounter(MatMemoMetric::Rebase, stage));
	if (result.unsafe) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 50) {
			LOGF("[matmemo-unsafe] program=0x%016" PRIx64 " stage=%s\n", program_hash, stage == 0 ? "vs" : (stage == 1 ? "ps" : "cs"));
		}
	}
	const auto window = CpSampler::Window();
	const auto flips  = FrameTiming::GuestFlips();
	auto&      state  = MatMemoLogState();
	if (flips >= window.from && flips <= window.to) {
		state.report.Add(program_hash, stage, result);
		if (flips / 100 != state.last_bucket) {
			if (state.last_bucket != 0) {
				for (const auto& row: state.report.TopMisses(10)) {
					LOGF("[matmemo] flips<=%" PRIu64 " program=0x%016" PRIx64 " stage=%s n=%" PRIu64 " in_hit=%" PRIu64 " miss=%" PRIu64
					     " unsafe=%" PRIu64 "\n",
					     flips, row.hash, row.stage == 0 ? "vs" : (row.stage == 1 ? "ps" : "cs"), row.n, row.in_hit, row.Misses(),
					     row.unsafe);
				}
				state.report.Clear();
			}
			state.last_bucket = flips / 100;
		}
	}
}

// ---- S406 (KYTY_DIAG_SYNC=1): shadow of the decision when the spec reads see stale bytes of GPU-written memory ----
namespace SpecShadow = ShaderRecompiler::IR::SpecShadow;

bool ShadowBacking(uint64_t address, void* data, uint64_t size) {
	return Libs::LibKernel::Memory::TryReadBacking(address, data, size);
}
bool ShadowDirty(uint64_t address, uint64_t size) {
	return Libs::LibKernel::Memory::QueryShaderGpuDirty(address, size);
}

struct SpecShadowRun {
	bool                      active = false; // a spec read hit GPU-dirty bytes: stale and superset were run, compare after the real run
	SpecShadow::Outcome       stale;
	SpecShadow::Outcome       superset;
	SpecShadow::ShadowReader  stale_reader;
	SpecShadow::ShadowReader  super_reader;
};

// Runs the stale shadow (every read returns the bytes in memory now, no drain). Only when it saw a GPU-dirty read the superset
// shadow follows; otherwise the real run will not drain either and there is nothing to compare.
void RunSpecShadowBefore(const ShaderRecompiler::IR::ResourcePlan& plan, const ShaderRecompiler::IR::SrtRuntime& runtime,
                         SpecShadowRun& run) {
	if (!plan.requires_specialization_memory && plan.source_reads.empty() && !plan.capture_specialization_reads) return;
	run.stale_reader = {.mode = SpecShadow::ReadMode::Stale, .backing = ShadowBacking, .dirty = ShadowDirty};
	auto shadow_runtime                       = runtime;
	shadow_runtime.read_specialization_memory = SpecShadow::ShadowRead;
	shadow_runtime.fast_mode                  = ShaderRecompiler::IR::SrtFastMode::Off; // S425: diag copy, no memo
	shadow_runtime.userdata                   = &run.stale_reader;
	SpecShadow::Materialize(plan, shadow_runtime, run.stale);
	if (run.stale_reader.DirtyTotal() == 0) return;
	run.super_reader                          = {.mode = SpecShadow::ReadMode::Superset, .backing = ShadowBacking, .dirty = ShadowDirty};
	shadow_runtime.userdata                   = &run.super_reader;
	SpecShadow::Materialize(plan, shadow_runtime, run.superset);
	run.active = true;
}

bool ShadowReadable(uint64_t address) {
	uint32_t word = 0;
	return Libs::LibKernel::Memory::TryReadBacking(address, &word, sizeof(word));
}

// S408: what the superset adds over the fresh run (kinds, written, addresses outside guest memory, new drains of other purposes).
void NoteSpecSuperExtra(const ShaderRecompiler::IR::ResourcePlan& plan, const SpecShadowRun& run, const SpecShadow::Outcome& fresh,
                        bool super_perm_same, uint64_t program_hash, const char* stage) {
	using C = FrameTiming::Counter;
	if (run.superset.ok) {
		if (super_perm_same) FrameTiming::Add(C::SpecSuperPermSameN);
		if (run.superset.specialization == fresh.specialization) FrameTiming::Add(C::SpecSuperSpecEqN);
	}
	FrameTiming::Add(C::SpecSuperDirtyNoncondN,
	                 SpecShadow::DirtyNonCondAdded(run.super_reader.DirtyNonCond(), run.stale_reader.DirtyNonCond()));
	const auto extra = SpecShadow::ClassifyExtra(plan, run.superset, fresh, ShadowReadable);
	FrameTiming::Add(C::SpecSuperExtraBufferN, extra.buffers);
	FrameTiming::Add(C::SpecSuperExtraImageN, extra.images);
	FrameTiming::Add(C::SpecSuperExtraSamplerN, extra.samplers);
	FrameTiming::Add(C::SpecSuperExtraWrittenN, extra.written);
	FrameTiming::Add(C::SpecSuperExtraBadAddrN, extra.bad_addr);
	static std::atomic<uint32_t> logged {0};
	for (const auto& item: extra.list) {
		if (logged.fetch_add(1, std::memory_order_relaxed) >= 50) return;
		const uint32_t size = item.kind == 'b' && item.index < run.superset.snapshot.buffers.size()
		                          ? run.superset.snapshot.buffers[item.index].dwords[2]
		                          : 0u;
		LOGF("[spec-super-extra] flip=%" PRIu64 " program=0x%016" PRIx64 " stage=%s kind=%c source=%u index=%u addr=0x%" PRIx64
		     " size=%u written=%d bad_addr=%d\n",
		     FrameTiming::GuestFlips(), program_hash, stage, item.kind, item.source, item.index, item.address, size, item.written ? 1 : 0,
		     item.bad ? 1 : 0);
	}
}

void NoteSpecShadow(const ShaderRecompiler::IR::ResourcePlan& plan, const SpecShadowRun& run, const SpecShadow::Outcome& fresh,
                    bool perm_same, bool super_perm_same, uint64_t program_hash, const char* stage) {
	using C = FrameTiming::Counter;
	NoteSpecSuperExtra(plan, run, fresh, super_perm_same, program_hash, stage);
	const auto verdict = SpecShadow::Compare(run.stale, run.superset, fresh);
	FrameTiming::Add(verdict.StaleEqual() ? C::SpecDecStaleEqN : C::SpecDecStaleNeN);
	FrameTiming::Add(verdict.super_ok ? C::SpecDecSuperOkN : C::SpecDecSuperFailN);
	FrameTiming::Add(C::SpecSuperExtraSourcesN, verdict.extra_sources);
	FrameTiming::Add(C::SpecSuperNewImagesN, verdict.new_images);
	if (perm_same) FrameTiming::Add(C::SpecPermSameN);
	if (verdict.StaleEqual() && verdict.super_ok && verdict.extra_sources == 0 && verdict.new_images == 0) return;
	static std::atomic<uint32_t> logged {0};
	if (logged.fetch_add(1, std::memory_order_relaxed) >= 200) return;
	char dirty[96];
	std::snprintf(dirty, sizeof(dirty), "cond=%u,slot=%u,desc=%u,fill=%u,wg=%u,call=%u,other=%u",
	              run.stale_reader.dirty_reads[1], run.stale_reader.dirty_reads[2], run.stale_reader.dirty_reads[3],
	              run.stale_reader.dirty_reads[4], run.stale_reader.dirty_reads[5], run.stale_reader.dirty_reads[6],
	              run.stale_reader.dirty_reads[0]);
	const auto diff = SpecShadow::FirstWordDiff(verdict.stale_part, run.stale, fresh);
	LOGF("[spec-dec] flip=%" PRIu64 " program=0x%016" PRIx64 " stage=%s stale_part=%s stale_word=%u:%08x/%08x super_ok=%d "
	     "extra_sources=%u new_images=%u perm_same=%d dirty_reads(%s)\n",
	     FrameTiming::GuestFlips(), program_hash, stage, SpecShadow::PartName(verdict.stale_part), diff.index, diff.stale, diff.fresh,
	     verdict.super_ok ? 1 : 0, verdict.extra_sources, verdict.new_images, perm_same ? 1 : 0, dirty);
}

// S425: one per Get(): the SRT pass of a draw stage (KYTY_SRT_FAST).
struct ShaderReadContext {
	ShaderRecompiler::IR::SrtFastMode    mode = ShaderRecompiler::IR::SrtFastMode::Off;
	Libs::LibKernel::Memory::SrtReadPass pass;
	uint32_t                             page_mismatch = 0;
};

// S430 DIAG: the sampler / phases switches turn the walker's path counters on; Drain moves them into FrameTiming once per Get().
bool SpecDiagOn() {
	static const bool on = [] {
		const bool diag = CpSampler::Enabled() || CpPhases::Enabled();
		const auto fast = Libs::LibKernel::Memory::SpecFastEnvMode();
		LOGF("[spec-read-fast] mode=%s\n", fast == Libs::LibKernel::Memory::SpecFastMode::On ? "1"
		                                    : fast == Libs::LibKernel::Memory::SpecFastMode::Shadow ? "shadow" : "0");
		if (diag) ShaderRecompiler::IR::SrtWalkPaths::Active().store(true, std::memory_order_relaxed);
		return diag;
	}();
	return on;
}

void DrainSrtWalkPaths() {
	if (!SpecDiagOn()) return;
	using Path = ShaderRecompiler::IR::SrtWalkPaths::Path;
	using C    = FrameTiming::Counter;
	const auto cells = ShaderRecompiler::IR::SrtWalkPaths::Drain();
	const auto at    = [&](Path path) { return cells[static_cast<size_t>(path)]; };
	FrameTiming::Add(C::SrtWalkTreeN, at(Path::Tree));
	FrameTiming::Add(C::SrtWalkTreeFailN, at(Path::TreeFail));
	FrameTiming::Add(C::SrtWalkCaptureN, at(Path::Capture));
	FrameTiming::Add(C::SrtWalkSlotJitN, at(Path::SlotJit));
	FrameTiming::Add(C::SrtWalkInterpN, at(Path::InterpWalk));
	FrameTiming::Add(C::SrtInterpSlotsN, at(Path::InterpSlots));
}

// S430 (KYTY_SPEC_READ_FAST): tallies of the CP thread, added to FrameTiming once per Get() (NoteSrtMemo).
struct SpecFastTally {
	uint64_t hit = 0, miss = 0, stale = 0, mismatch = 0;
};
SpecFastTally& SpecFastCounts() {
	static SpecFastTally tally;
	return tally;
}

void FlushSpecFastCounts() {
	auto& tally = SpecFastCounts();
	using C     = FrameTiming::Counter;
	if (tally.hit != 0) FrameTiming::Add(C::SpecFastHitN, std::exchange(tally.hit, 0));
	if (tally.miss != 0) FrameTiming::Add(C::SpecFastMissN, std::exchange(tally.miss, 0));
	if (tally.stale != 0) FrameTiming::Add(C::SpecFastStaleN, std::exchange(tally.stale, 0));
	if (tally.mismatch != 0) FrameTiming::Add(C::SpecFastMismatchN, std::exchange(tally.mismatch, 0));
}

// 1: the page cache. shadow: the cache first (so it cannot profit from a drain the full read does), then the full read, whose result is returned;
// stale = the cache answered from an entry although the range is GPU-dirty / in flight now (must stay 0), mismatch = different ok or bytes.
bool ReadSpecFast(Libs::LibKernel::Memory::SpecFastMode mode, uint64_t address, std::span<uint32_t> values) {
	namespace Memory = Libs::LibKernel::Memory;
	auto& tally      = SpecFastCounts();
	bool  hit        = false;
	if (mode == Memory::SpecFastMode::On) {
		const bool ok = Memory::TryReadBufferBackingFast(address, values.data(), values.size_bytes(), &hit);
		++(hit ? tally.hit : tally.miss);
		return ok;
	}
	std::array<uint32_t, 32> fast_small;
	std::vector<uint32_t>    fast_large;
	std::span<uint32_t>      fast = values.size() <= fast_small.size() ? std::span(fast_small).first(values.size())
	                                                                    : (fast_large.resize(values.size()), std::span(fast_large));
	const bool fast_ok = Memory::TryReadBufferBackingFast(address, fast.data(), fast.size_bytes(), &hit);
	const bool stale   = hit && Memory::QueryShaderGpuDirty(address, values.size_bytes());
	const bool ok      = Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
	const bool differs = fast_ok != ok || (ok && !std::ranges::equal(fast, values));
	++(hit ? tally.hit : tally.miss);
	static std::atomic_uint logged_stale = 0, logged_mismatch = 0;
	if (stale) {
		++tally.stale;
		if (logged_stale.fetch_add(1) < 20) PipelineCacheLog("[spec-fast-stale] addr={:016x} size={}", address, values.size_bytes());
	}
	if (differs) {
		++tally.mismatch;
		if (logged_mismatch.fetch_add(1) < 20) {
			PipelineCacheLog("[spec-fast-mismatch] addr={:016x} size={} hit={} fast_ok={} ok={}", address, values.size_bytes(), hit, fast_ok, ok);
		}
	}
	return ok;
}

bool ReadShaderGuestMemory(void* userdata, uint64_t address, std::span<uint32_t> values) {
	KYTY_CP_SITE(ShSpecRead); // S430
	if (SpecDiagOn() && !values.empty()) {
		FrameTiming::Add(FrameTiming::Counter::SpecReadN);
		if (((address ^ (address + values.size_bytes() - 1)) & ~uint64_t {4095}) != 0) FrameTiming::Add(FrameTiming::Counter::SpecReadXpageN);
	}
	// S408 (KYTY_SPEC_COND=superset): a condition read of GPU-dirty bytes on the hit / refresh path fails without draining; the
	// walker then follows both successors. See specCond.h. Mode 0 never asks for the dirty state.
	const auto cond_mode = ShaderRecompiler::IR::SpecCond::EnvMode();
	if (cond_mode != ShaderRecompiler::IR::SpecCond::Mode::Off && !values.empty()) {
		using C            = FrameTiming::Counter;
		const auto action = ShaderRecompiler::IR::SpecCond::Decide(
		    cond_mode, SpecShadow::CurrentPurpose(), ShaderRecompiler::IR::SpecCond::IsMissPath(SyncCensus::CurrentReaderRef().path),
		    [&] {
			    // S430: a page remembered as quiet is not GPU-dirty (KYTY_SPEC_READ_FAST=1).
			    if (Libs::LibKernel::Memory::SpecFastEnvMode() == Libs::LibKernel::Memory::SpecFastMode::On &&
			        Libs::LibKernel::Memory::SpecPageQuietCached(address, values.size_bytes())) {
				    return false;
			    }
			    return Libs::LibKernel::Memory::QueryShaderGpuDirty(address, values.size_bytes());
		    });
		if (action == ShaderRecompiler::IR::SpecCond::Action::Fail) {
			FrameTiming::Add(C::SpecCondSupersetN);
			return false;
		}
		if (action == ShaderRecompiler::IR::SpecCond::Action::SkipMiss) FrameTiming::Add(C::SpecCondSupersetSkipMissN);
	}
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	const SyncCensus::SiteScope site_tag(SyncCensus::Site::Spec);
	if (values.empty()) {
		return false;
	}
	auto* context = static_cast<ShaderReadContext*>(userdata);
	if (context == nullptr || context->mode == ShaderRecompiler::IR::SrtFastMode::Off) {
		if (const auto fast = Libs::LibKernel::Memory::SpecFastEnvMode(); fast != Libs::LibKernel::Memory::SpecFastMode::Off) {
			return ReadSpecFast(fast, address, values);
		}
		return Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
	}
	if (context->mode == ShaderRecompiler::IR::SrtFastMode::On) {
		return Libs::LibKernel::Memory::TryReadBufferBackingPass(address, values.data(), values.size_bytes(), &context->pass);
	}
	// Shadow: the page-memoized read first (so it cannot profit from a sync the full read does), the full one is the result.
	std::array<uint32_t, 32> fast_small;
	std::vector<uint32_t>    fast_large;
	std::span<uint32_t>      fast = values.size() <= fast_small.size() ? std::span(fast_small).first(values.size())
	                                                                    : (fast_large.resize(values.size()), std::span(fast_large));
	const bool fast_ok = Libs::LibKernel::Memory::TryReadBufferBackingPass(address, fast.data(), fast.size_bytes(), &context->pass);
	const bool ok      = Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
	if (fast_ok != ok || (ok && !std::ranges::equal(fast, values))) {
		++context->page_mismatch;
	}
	return ok;
}

// S425: counters of one MaterializeResources and, in shadow mode, the first 16 mismatches (hash, stage, flat slot).
void NoteSrtMemo(const ShaderRecompiler::IR::ResourceSnapshot& snapshot, const ShaderReadContext& context, uint64_t hash,
                 ShaderType stage) {
	using ShaderRecompiler::IR::SrtMemoOutcome;
	DrainSrtWalkPaths();
	FlushSpecFastCounts();
	if (context.mode == ShaderRecompiler::IR::SrtFastMode::Off) {
		return;
	}
	switch (snapshot.srt_outcome) {
		case SrtMemoOutcome::Hit: FrameTiming::Add(FrameTiming::Counter::SrtMemoHit); break;
		case SrtMemoOutcome::Miss: FrameTiming::Add(FrameTiming::Counter::SrtMemoMiss); break;
		case SrtMemoOutcome::Mismatch: FrameTiming::Add(FrameTiming::Counter::SrtMemoMismatch); break;
		case SrtMemoOutcome::None: break;
	}
	FrameTiming::Add(FrameTiming::Counter::SrtPagesChecked, context.pass.pages_checked);
	FrameTiming::Add(FrameTiming::Counter::SrtPageHit, context.pass.page_hits);
	if (context.page_mismatch != 0) {
		FrameTiming::Add(FrameTiming::Counter::SrtMemoMismatch, context.page_mismatch);
	}
	static std::atomic_uint logged = 0;
	if ((snapshot.srt_outcome == SrtMemoOutcome::Mismatch || context.page_mismatch != 0) && logged.fetch_add(1) < 16) {
		PipelineCacheLog("[srt-shadow] mismatch hash={:016x} stage={} slot={} {}", hash, static_cast<int>(stage),
		                 snapshot.srt_mismatch, context.page_mismatch != 0 ? "page-read" : "memo");
	}
}

bool ReadBackingForShader(void*, uint64_t address, void* data, uint64_t size, bool* gpu_dirty) {
	if (gpu_dirty != nullptr) {
		*gpu_dirty = NullDesc::Enabled() && Libs::LibKernel::Memory::QueryShaderGpuDirty(address, size);
	}
	const SyncCensus::SiteScope site_tag(SyncCensus::Site::Spec);
	return Libs::LibKernel::Memory::TryReadBufferBacking(address, data, size);
}

// checked mode: a miss is zeros, never a failed read. MaterializeResources EXIT_IF's on false.
bool ReadShaderGuestMemoryOrZero(void*, uint64_t address, std::span<uint32_t> values) {
	return SrtReader::ReadOrZero(address, values, ReadBackingForShader, nullptr);
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			ShaderRecompiler::IR::SpecShadow::PurposeScope purpose(ShaderRecompiler::IR::SpecShadow::Purpose::Call); // S406
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		// S406 (KYTY_DIAG_MATMEMO=1): what a memo keyed by the input of the part after the walk would have done.
		void NoteMatMemo(const ShaderRecompiler::IR::SrtRuntime& runtime, uint32_t push_data_cursor, uint32_t permutation) {
			if (!m_matmemo_mask_valid) {
				m_matmemo_plan_mask  = ShaderRecompiler::IR::PlanUsedUserDataMask(resource_plan);
				m_matmemo_mask_valid = true;
			}
			uint64_t mask = m_matmemo_plan_mask;
			for (const auto& candidate: permutations) {
				for (const auto reg: candidate.program.info.user_data_registers) {
					if (reg >= candidate.program.user_data_base && reg - candidate.program.user_data_base < 64u) {
						mask |= uint64_t {1} << (reg - candidate.program.user_data_base);
					}
				}
			}
			MatMemo::InputParts parts;
			parts.user_data        = runtime.user_data;
			parts.used_mask        = mask;
			parts.flat             = resources.flattened_srt;
			parts.workgroup_counts = runtime.workgroup_counts;
			parts.min_lod_remap    = runtime.min_lod_remap;
			parts.push_cursor      = push_data_cursor;
			const auto  result = m_matmemo.Note(MatMemo::InputKey(parts),
			                                    HashMaterializationOutput(resources, specialization, mask, permutation));
			const auto stage_index = resource_plan.stage == ShaderType::Pixel ? 1u : (resource_plan.stage == ShaderType::Compute ? 2u : 0u);
			NoteMatMemoResult(resource_plan.shader_hash, stage_index, result);
		}

		template <bool HasCalls>
		Permutation* FindPermutation(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                             uint32_t push_data_cursor) {
			SpecShadowRun shadow;
			if (SyncCensus::Enabled()) RunSpecShadowBefore(resource_plan, runtime, shadow); // S406: before the real reads drain
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    resource_plan, runtime, resources, specialization));
			KYTY_CP_SITE(ShPermSearch);
			Permutation* found = nullptr;
			uint32_t     found_index = UINT32_MAX;
			uint32_t     index = 0;
			for (auto& candidate: permutations) {
				const auto here = index++;
				const auto& layout = candidate.program.bindings;
				if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
				        push_data_cursor, layout.ShaderDataDwords()) ||
				    candidate.specialization != specialization) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
				}
				found       = &candidate;
				found_index = here;
				break;
			}
			if (MatMemoEnabled()) NoteMatMemo(runtime, push_data_cursor, found_index);
			if (shadow.active) {
				SpecShadow::Outcome fresh;
				fresh.ok             = true;
				fresh.snapshot       = resources;
				fresh.specialization = specialization;
				fresh.active         = resource_plan.active_sources;
				// Would the stale / the superset specialization have picked the same permutation?
				const auto same_permutation = [&](const SpecShadow::Outcome& other) {
					bool same = other.ok && other.specialization == specialization;
					if (other.ok && !same && found != nullptr) {
						uint32_t at = 0;
						for (auto& candidate: permutations) {
							const auto& layout = candidate.program.bindings;
							if (layout.push_data_start_dword == ShaderRecompiler::IR::PushData::StartFor(
							        push_data_cursor, layout.ShaderDataDwords()) &&
							    candidate.specialization == other.specialization) {
								same = at == found_index;
								break;
							}
							at++;
						}
					}
					return same;
				};
				const bool perm_same       = same_permutation(shadow.stale);
				const bool super_perm_same = same_permutation(shadow.superset);
				const auto stage_name = resource_plan.stage == ShaderType::Pixel ? "ps" : (resource_plan.stage == ShaderType::Compute ? "cs" : "vs");
				NoteSpecShadow(resource_plan, shadow, fresh, perm_same, super_perm_same, resource_plan.shader_hash, stage_name);
			}
			return found;
		}

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
		MatMemo::Shadow                             m_matmemo;          // S406
		uint64_t                                    m_matmemo_plan_mask = 0;
		bool                                        m_matmemo_mask_valid = false;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.maintenance8_enabled)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		const ShaderProgram handle {.id = ++next_shader_id, .module = module};
		// The per-vertex capture and VS-store variants re-emit the program, and EmitProgram rewrites
		// program.bindings (push start, layout); the permutation keeps the bindings of its own emission.
		RememberPerVertexVariants(handle, result, options, graphics.subgroup_size, push_data_start_dword);
		return {
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = handle,
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		std::optional<CpSampler::SiteScope> static_key_site;
		if (CpSampler::Active().load(std::memory_order_relaxed)) static_key_site.emplace(CpSampler::SiteId(CpSampler::Site::ShStaticKeyFind));
		BuildStageStaticKey(input_info, lookup_key.static_state);
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			// Only when on, so the keys of runs without KYTY_RT_MISS stay as they were.
			if (ShaderRecompiler::RtMissEnabled()) lookup_key.static_state.push_back(0x52544d53u);
		}
		auto                                         entry = programs.find(lookup_key);
		static_key_site.reset();
		// S404: the reader is named for the whole lookup (hit, call-source refresh, compile), not only the compile.
		const SyncCensus::ReaderScope reader_tag(
		    params.hash, SyncStageName(stage),
		    entry == programs.end() ? "miss" : (entry->second.call_source ? "refresh" : "hit"));
		// S425: programs with a retained call source re-extract / edit their plan in place (RefreshCallSource), so the memo
		// (keyed by the plan address) is off for them.
		ShaderReadContext                            read_context {
		    .mode = (entry != programs.end() && entry->second.call_source) ? ShaderRecompiler::IR::SrtFastMode::Off : SrtFast::Mode()};
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .userdata                   = &read_context,
		    .read_specialization_memory = ReadShaderGuestMemory,
		    .min_lod_remap              = graphics.min_lod_remap,
		    .fast_mode                  = read_context.mode,
		};
		// RefreshCallSource feeds this same runtime to RefreshShaderSource, which walks through
		// CleanRuntime. That replaces read_memory with read_specialization_memory, so call-target
		// fetches stay on ReadShaderGuestMemory and are not zero-filled here.
		SrtReader::Apply(runtime, ReadShaderGuestMemoryOrZero);
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
		}
		bool source_changed = false;
		if (entry != programs.end()) {
			auto& cached = entry->second;
			Permutation* permutation = nullptr;
			if (cached.call_source) {
				{
					KYTY_CP_SITE(ShRefreshCallSource);
					source_changed = cached.RefreshCallSource(runtime);
				}
				if (!source_changed) permutation = cached.FindPermutation<true>(runtime, push_data_cursor);
			} else {
				permutation = cached.FindPermutation<false>(runtime, push_data_cursor);
				NoteSrtMemo(cached.resources, read_context, params.hash, stage);
			}
			if (permutation != nullptr) {
				KYTY_CP_SITE(ShProgramsRest);
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		SyncCensus::SetReaderPath("miss");
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.maintenance8_enabled = graphics.maintenance8_enabled;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		ShaderRecompiler::ShaderSource source;
		if (entry == programs.end()) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		auto* current = entry == programs.end() ? &source : entry->second.call_source.get();
		ShaderRecompiler::TranslateResult translated;
		if (current != nullptr) {
			if (current->call) {
				if (entry == programs.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
				translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
				translated.program.source_reads = current->reads;
			} else {
				translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
			}
		} else {
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		if (entry == programs.end()) {
			auto retained = source.call ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program), std::move(retained)).first;
			if (entry->second.call_source) read_context.mode = runtime.fast_mode = ShaderRecompiler::IR::SrtFastMode::Off;
			read_context.pass = {};
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
			NoteSrtMemo(entry->second.resources, read_context, params.hash, stage);
		} else if (source_changed) {
			entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			entry->second.m_matmemo_mask_valid = false;
			if (const auto* permutation = entry->second.FindPermutation<true>(runtime, push_data_cursor)) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor,
		    entry->second.FunctionCode()));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	explicit ProgramCache(GraphicContext& graphics): graphics(graphics) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	GraphicContext&                                             graphics;
	uint64_t                                                    next_shader_id = 0;
};

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics)),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	// S408: stale reads of the condition word are only allowed with a verification of the used bytes after GPU-done (plan-chainA 1b),
	// which is not built. Not a silent 0.
	if (ShaderRecompiler::IR::SpecCond::EnvMode() == ShaderRecompiler::IR::SpecCond::Mode::Stale) {
		EXIT("KYTY_SPEC_COND=stale is not implemented (it needs the GPU-done verification of the stale bytes); use 0 or superset\n");
	}
	InitializeDriverCache();
}

PipelineCache::~PipelineCache() {
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
		m_driver_cache = nullptr;
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) && file_size <= 256 * 1024 * 1024) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver, format, or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_driver_cache_dirty = true;
	if (std::chrono::steady_clock::now() >= m_next_driver_cache_save) {
		Save();
	}
}

void PipelineCache::Save() {
	if (m_driver_cache == nullptr || !m_driver_cache_dirty) {
		return;
	}
	m_next_driver_cache_save = std::chrono::steady_clock::now() + DriverCacheSaveInterval;

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 || size > 256 * 1024 * 1024) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 || size > 256 * 1024 * 1024) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	static std::atomic<uint64_t> s_save_counter = 0;
	const auto                   temp_path      = std::filesystem::path(
	    m_driver_cache_path.string() +
	    fmt::format(".tmp.{}.{}", Common::Thread::GetProcessId(), s_save_counter++));
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		std::error_code remove_error;
		std::filesystem::remove(temp_path, remove_error);
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
	m_driver_cache_dirty = false;
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info,
    bool single_sample) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	const bool object_primitive_id = context.HasObjectPrimitiveIdOutput();
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active && !m_graphics.mesh_shader_enabled) {
		// Bench bring-up: this host's Vulkan driver does not expose VK_EXT_mesh_shader (MoltenVK
		// on this Mac reports no mesh shader support). Skip whatever needs it instead of aborting
		// the whole process -- PrepareDrawRenderState() treats an empty GraphicsPrograms as "skip
		// this draw", so the rest of the frame/scene keeps rendering.
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("PipelineCache: skipping mesh-shader draw, host has no mesh shader support\n");
		}
		return {};
	}
	if (mesh_active) {
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
		// Bench bring-up: MoltenVK splits threadgroup memory evenly between shared memory and
		// staged mesh outputs, so the reported shared limit is conservative. Let the driver
		// decide whether the pipeline fits.
		if (mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			// Once per shader. Printing this on every mesh draw wrote about 126k lines in a
			// 300 s ship run, and the log itself was a large part of the cost.
			const uint64_t hash = vertex_params[0].hash;
			static std::mutex seen_mu;
			static std::unordered_set<uint64_t> seen;
			bool first = false;
			{
				const std::lock_guard lock(seen_mu);
				first = seen.insert(hash).second;
			}
			if (first) {
				LOGF("mesh shader LDS above reported host limit: hash=0x%016" PRIx64
				     " LDS=%u bytes limit=%u\n",
				     hash, mesh.lds_size_dwords * 4u, limits.maxMeshSharedMemorySize);
			}
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
		pixel_info.ps_single_sample = single_sample;
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id);
	GraphicsPrograms  result;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	auto              program = m_program_cache->Get(params, input_info, push_data_cursor);
	return program;
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	const auto compile_begin = std::chrono::steady_clock::now();
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, static_params, m_driver_cache);
	}
	FrameTiming::Add(FrameTiming::Counter::PipelinesCreated);
	FrameTiming::Add(FrameTiming::Counter::PipelineCompileNs,
	                 static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                           std::chrono::steady_clock::now() - compile_begin)
	                                           .count()));
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	CheckpointDriverCache();

	static auto last_save = std::chrono::steady_clock::now();
	const auto  now       = std::chrono::steady_clock::now();
	if (now - last_save >= std::chrono::milliseconds(500)) {
		last_save = now;
		Save();
	}

	return *iter->second;
}

PipelineCache::Pipeline& PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                                           const ShaderProgram& compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	const auto compile_begin = std::chrono::steady_clock::now();
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                       m_driver_cache);
	}
	FrameTiming::Add(FrameTiming::Counter::PipelinesCreated);
	FrameTiming::Add(FrameTiming::Counter::PipelineCompileNs,
	                 static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                           std::chrono::steady_clock::now() - compile_begin)
	                                           .count()));

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	CheckpointDriverCache();

	static auto last_save = std::chrono::steady_clock::now();
	const auto  now       = std::chrono::steady_clock::now();
	if (now - last_save >= std::chrono::milliseconds(500)) {
		last_save = now;
		Save();
	}

	return *iter->second;
}
} // namespace Libs::Graphics
