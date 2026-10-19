#include "graphics/host_gpu/renderer/perVertexVsStoreRuntime.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/perVertexPrototype.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/permutationRecord.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <cstring>
#include <map>
#include <spirv-tools/libspirv.hpp>

namespace Libs::Graphics {
namespace {

// 4M vec4 = 64 MiB per slab; a frame of the game needs a few slabs at most (counted in pv_fallback_ring).
constexpr uint64_t kSlabVec4s = uint64_t {1} << 22u;
constexpr uint32_t kMaxSlabs  = 8;
// S388: guard counters, one 16-byte slot (256 B apart, the storage offset alignment) per flip modulo kGuardSlots.
constexpr uint32_t kGuardSlots  = 4;
constexpr uint64_t kGuardStride = 256;
constexpr uint64_t kPvGuardBytes = 16;

struct StoreShader {
	ShaderType            stage = ShaderType::Unknown;
	std::vector<uint32_t> words;       // empty: emission or validation failed
	uint32_t              mask = 0;    // host locations the stage stores (VS) / reads per vertex (PS)
	uint32_t              corners = 0; // PS: corners (P0/P10/P20) it reads, bit k = corner k
	uint64_t              hash = 0;    // guest shader hash (stable between runs)
	uint32_t              compact_spec_id = 0; // S409 VS: id of the OpSpecConstant that holds the PS mask (its literal is patched per pair)
};
std::map<uint64_t, StoreShader>                                    s_shaders;
std::map<std::pair<uint64_t, uint64_t>, PvVsStorePrograms>         s_pairs;
// S409: the VS-store VS specialized with a PS mask, one module per (VS, mask).
std::map<std::pair<uint64_t, uint32_t>, vk::ShaderModule>          s_compact_vs;
uint32_t                                                           s_compact_pair_lines = 0;
vk::DescriptorSetLayout                                            s_set_layout = nullptr;
uint32_t                                                           s_pairs_admitted = 0;
uint32_t                                                           s_pairs_candidates = 0;
uint32_t                                                           s_pair_lines = 0;
// S389: the pair-use lines (`[pv-pairs]` index lines and first-use lines) share this budget.
constexpr uint32_t kMaxPairLines = 200;
constexpr uint32_t kMaxCompactLines = 100; // S409: `[pv-compact]` lines

} // namespace

void FillPvCompactPixelInfo(ShaderRecompiler::Spirv::PvStoreInfo& info, uint32_t ps_mask, uint32_t vs_mask) {
#if defined(KYTY_PVVS_MUTANT_COMPACT_PS_USES_VS_SLOT)
	// the PS ranks its locations in the mask of the VS (and so reads other slots than the VS wrote)
	const uint32_t slot_mask = vs_mask | ps_mask;
#else
	(void)vs_mask;
	const uint32_t slot_mask = ps_mask;
#endif
	info.record_stride_vec4 = PvCompactStride(slot_mask);
	info.parameter_slots.clear();
	for (uint32_t location = 0; location < kPvVsStoreInstanceLocation; ++location) {
		if (((ps_mask >> location) & 1u) != 0) info.parameter_slots[location] = PvCompactSlot(slot_mask, location);
	}
}

std::vector<uint32_t> EmitPvVsStoreVariant(ShaderRecompiler::IR::Program&          program,
                                           const ShaderRecompiler::CompileOptions& options,
                                           const ShaderRecompiler::Spirv::PvStoreInfo& info,
                                           uint32_t push_data_start_dword, bool maintenance8_enabled) {
#if defined(KYTY_PVPUSH_MUTANT_ZERO)
	push_data_start_dword = 0;
	maintenance8_enabled  = false;
#endif
	return ShaderRecompiler::Spirv::EmitProgram(program, options.input_info, push_data_start_dword,
	                                            maintenance8_enabled, false, nullptr, &info);
}

void RememberPvVsStoreShader(const ShaderProgram& program, ShaderRecompiler::CompileResult& result,
                             const ShaderRecompiler::CompileOptions& options,
                             uint32_t push_data_start_dword, bool maintenance8_enabled) {
	if (!PerVertexVsStoreEnabled()) return;
	(void)PerVertexVsStoreGpuIdxMode(); // read and logged once, at the first compile
	(void)PerVertexVsStoreInstanced();
	(void)PerVertexVsStorePairs();
	(void)PerVertexVsStoreCompact();
	using Role = ShaderRecompiler::Spirv::PvStoreInfo::Role;
	const bool vertex = options.stage == ShaderType::Vertex && options.input_info.vertex != nullptr;
	const bool pixel  = options.stage == ShaderType::Pixel && options.input_info.pixel != nullptr &&
	                   HasPerVertexPrototypeInput(result.spirv);
	if (!vertex && !pixel) return;
	StoreShader entry;
	entry.stage = options.stage;
	entry.hash  = options.shader_hash;
	ShaderRecompiler::Spirv::PvStoreInfo info;
	info.role          = vertex ? Role::Vertex : Role::Pixel;
	info.guard         = PerVertexVsStoreGuard();
	info.coherent      = true; // Coherent cap + a device barrier after the VS stores (the probe of RESULT-262 needs them)
	info.location_mask = &entry.mask;
	info.corner_mask   = &entry.corners;
	const bool compact = PerVertexVsStoreCompact();
	if (compact && vertex) {
		info.compact_spec    = true;
		info.compact_mask_id = &entry.compact_spec_id;
	}
	auto words         = EmitPvVsStoreVariant(result.program, options, info, push_data_start_dword, maintenance8_enabled);
	if (compact && pixel) {
		// The first emission (dynamic stride) found the locations the PS reads; the PS of a compact record knows them at compile time.
		FillPvCompactPixelInfo(info, entry.mask, entry.mask);
		words = EmitPvVsStoreVariant(result.program, options, info, push_data_start_dword, maintenance8_enabled);
	}
	NotePvPushVariant(vertex ? "vsstore-vs" : "vsstore-ps", options.shader_hash, result.program.bindings);
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
	tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
		LOGF("PvVsStore: %s 0x%016llx SPIR-V word %zu: %s\n", vertex ? "VS" : "PS",
		     static_cast<unsigned long long>(program.id), position.index, message);
	});
	// S437 KYTY_PV_SKIP_VALIDATE=1 (bench only, default 0): the variant is our own emitter output; Validate (the larger part of the cost of a
	// permutation) is left to the driver at the first draw. Independent of KYTY_PV_LAYOUT_FAST (S432 coupled them). shadow counts a rejection (must be 0).
	const auto layout_mode     = PerVertexLayoutFastMode();
	static const bool should_validate = PvShouldValidate(layout_mode, std::getenv("KYTY_PV_SKIP_VALIDATE"));
	const bool validated       = !should_validate || tools.Validate(words);
	if (!should_validate) {
		FrameTiming::Add(FrameTiming::Counter::PvValidateSkippedN);
	} else if (layout_mode == PvLayoutFastMode::Shadow && !validated) {
		FrameTiming::Add(FrameTiming::Counter::PvValidateNeN);
	}
	if (validated) {
		entry.words = std::move(words);
	} else {
		LOGF("PvVsStore: %s 0x%016llx variant rejected, its draws keep the prototype\n", vertex ? "VS" : "PS",
		     static_cast<unsigned long long>(program.id));
	}
	s_shaders.emplace(program.id, std::move(entry));
}

void RememberPerVertexVariants(const ShaderProgram& program, ShaderRecompiler::CompileResult& result,
                               const ShaderRecompiler::CompileOptions& options, uint32_t host_subgroup_size,
                               uint32_t push_data_start_dword) {
	const uint32_t start = PerVertexReemitPushStart(push_data_start_dword);
	const bool     m8    = PerVertexReemitMaintenance8(options.maintenance8_enabled);
	// Each re-emission rewrites program.bindings; the first must not change the input of the second.
	const auto emitted_bindings = result.program.bindings;
	// A variant that does not read its shader data where the regular program does would see wrong dwords.
	const bool check = PerVertexPushStartFix();
	RememberPerVertexPrototypeShader(program, result, options, host_subgroup_size, start, m8);
	EXIT_IF(check && !PvVariantPushMatches(emitted_bindings, result.program.bindings));
	result.program.bindings = emitted_bindings;
	RememberPvVsStoreShader(program, result, options, start, m8);
	EXIT_IF(check && !PvVariantPushMatches(emitted_bindings, result.program.bindings));
	EXIT_IF(result.program.bindings.descriptor_mask != emitted_bindings.descriptor_mask);
	result.program.bindings = emitted_bindings;
}

ProgramDisk::PvRecord ExportPerVertexVariants(uint64_t program_id) {
	ProgramDisk::PvRecord record;
	ProgramDisk::PvOriginalRecord original;
	if (ExportPerVertexPrototypeVariant(program_id, &original)) record.original = std::move(original);
	if (const auto it = s_shaders.find(program_id); it != s_shaders.end()) {
		const auto& shader = it->second;
		record.store       = ProgramDisk::PvStoreRecord {static_cast<uint32_t>(shader.stage), shader.words, shader.mask, shader.corners, shader.hash,
		                                                 shader.compact_spec_id};
	}
	return record;
}

void ImportPerVertexVariants(uint64_t program_id, const ProgramDisk::PvRecord& record, std::span<const uint32_t> spirv) {
	if (record.original) ImportPerVertexPrototypeVariant(program_id, *record.original, spirv);
#ifndef KYTY_PROGDISK_MUTANT_PV_IMPORT_DROPS_STORE
	if (record.store) {
		const auto& stored = *record.store;
		s_shaders.emplace(program_id, StoreShader {static_cast<ShaderType>(stored.stage), stored.words, stored.mask, stored.corners, stored.hash,
		                                           stored.compact_spec_id});
	}
#endif
}

vk::DescriptorSetLayout PvVsStoreSetLayout(GraphicContext& graphics) {
	if (s_set_layout != nullptr) return s_set_layout;
	// S388: binding 2 (the guard's counter) exists only with KYTY_PV_VSSTORE_GUARD=1.
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1,
	     vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment},
	    {kPvGuardBinding, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex}};
	vk::DescriptorSetLayoutCreateInfo create {};
	create.bindingCount = PerVertexVsStoreGuard() ? 3 : 2;
	create.pBindings    = bindings;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &s_set_layout) != vk::Result::eSuccess);
	return s_set_layout;
}

PvVsStoreRuntime::GuardSlot PvVsStoreRuntime::AcquireGuard(uint64_t flip) {
	if (!m_guard) {
		// The counters are read on the CPU (Invalidate below): host memory, MemoryUsage::Download.
		m_guard = std::make_unique<Buffer>(m_graphics, m_scheduler,
		                                   kPvGuardMemory == PvGuardMemory::Download ? MemoryUsage::Download : MemoryUsage::Stream,
		                                   0, AllFlags, uint64_t {kGuardSlots} * kGuardStride);
		EXIT_IF(m_guard->Mapped().empty());
		std::memset(m_guard->Mapped().data(), 0, m_guard->Mapped().size());
		m_guard->Flush(0, m_guard->Size());
		m_guard_flip = flip;
	}
	auto* mapped = m_guard->Mapped().data();
	const auto zero = [&](uint64_t slot) {
		std::memset(mapped + slot * kGuardStride, 0, kPvGuardBytes);
		m_guard->Flush(slot * kGuardStride, kPvGuardBytes);
	};
	if (flip > m_guard_flip + kGuardSlots) {
		// A long gap: the draws of the missed flips are long finished and not reported.
		for (uint32_t slot = 0; slot < kGuardSlots; ++slot) zero(slot);
		m_guard_flip = flip;
	}
	// Flip g starts: report what the draws of flip g-2 counted (the GPU has had two flips to finish them; no wait),
	// then clear the slot flip g will use. The counter is read, never waited for.
	while (m_guard_flip < flip) {
		++m_guard_flip;
		if (m_guard_flip >= 2) {
			const uint64_t slot = (m_guard_flip - 2) % kGuardSlots;
			m_guard->Invalidate(slot * kGuardStride, kPvGuardBytes);
			uint32_t count = 0;
			std::memcpy(&count, mapped + slot * kGuardStride, sizeof(count));
			if (count != 0) FrameTiming::Add(FrameTiming::Counter::PvVsStoreOob, count);
		}
		zero(m_guard_flip % kGuardSlots);
	}
	return {m_guard->Handle(), (flip % kGuardSlots) * kGuardStride, kPvGuardBytes};
}

void LogPvPairFirstUse(const PvVsStorePrograms& programs, uint32_t topology, bool indexed, uint32_t instance_count) {
	if (programs.first_use_logged || s_pair_lines >= kMaxPairLines) return;
	programs.first_use_logged = true;
	s_pair_lines++;
	LOGF("[pv-pairs] first use #%u VS=0x%016llx PS=0x%016llx stride=%u slots_vs=0x%x slots_ps=0x%x "
	     "corners=P0:%u/P10:%u/P20:%u primitive_id=%d topology=%u indexed=%d instanced=%d (instances=%u)\n",
	     programs.index, static_cast<unsigned long long>(programs.vs_hash),
	     static_cast<unsigned long long>(programs.ps_hash), programs.stride_vec4, programs.vs_mask, programs.ps_mask,
	     programs.corners & 1u, (programs.corners >> 1u) & 1u, (programs.corners >> 2u) & 1u,
	     programs.corners != 0 ? 1 : 0, topology, indexed ? 1 : 0, instance_count > 1 ? 1 : 0, instance_count);
}

uint32_t PvVsStorePushOffset() {
	return ShaderRecompiler::IR::NativePushConstantSize;
}

const PvVsStorePrograms* GetPvVsStorePrograms(GraphicContext& graphics, const PipelineCache::GraphicsPrograms& programs) {
	if (!PerVertexVsStoreEnabled() || programs.VertexStageCount() != 1) return nullptr;
	const auto key = std::pair {programs.vertex[0].id, programs.pixel.id};
	if (const auto it = s_pairs.find(key); it != s_pairs.end()) {
		return it->second.stride_vec4 == UINT32_MAX ? nullptr : &it->second;
	}
	PvVsStorePrograms value;
	const auto        vs = s_shaders.find(key.first);
	const auto        ps = s_shaders.find(key.second);
	if (vs == s_shaders.end() || ps == s_shaders.end() || vs->second.words.empty() || ps->second.words.empty()) {
		value.stride_vec4 = UINT32_MAX; // negative cache: not a candidate
		s_pairs.emplace(key, value);
		return nullptr;
	}
	value.verdict = ClassifyPvVsStoreInterface(vs->second.mask, ps->second.mask);
	if (value.verdict == PvVsStoreInterfaceVerdict::Use) {
		const uint32_t index   = s_pairs_candidates++;
		const bool     allowed = PvVsStorePairAllowed(PerVertexVsStorePairs(), key.first, vs->second.hash, key.second,
		                                              ps->second.hash, index);
		value.index   = index;
		value.vs_hash = vs->second.hash;
		value.ps_hash = ps->second.hash;
		value.vs_mask = vs->second.mask;
		value.ps_mask = ps->second.mask;
		value.corners = ps->second.corners;
		if (s_pair_lines++ < kMaxPairLines) {
			LOGF("[pv-pairs] #%u VS=%llu/0x%016llx PS=%llu/0x%016llx stride=%u vec4 %s\n", index,
			     static_cast<unsigned long long>(key.first), static_cast<unsigned long long>(vs->second.hash),
			     static_cast<unsigned long long>(key.second), static_cast<unsigned long long>(ps->second.hash),
			     PvVsStoreStrideFor(vs->second.mask), allowed ? "admitted" : "skipped (KYTY_PV_VSSTORE_PAIRS)");
			if (PerVertexVsStoreCompact()) LOGF("[pv-pairs] #%u stride_compact=%u\n", index, PvCompactStride(ps->second.mask));
		}
		if (!allowed) {
			value.stride_vec4 = UINT32_MAX; // negative cache: this pair stays on the prototype
			s_pairs.emplace(key, value);
			return nullptr;
		}
		s_pairs_admitted++;
		value.stride_vec4               = PvVsStoreStrideFor(vs->second.mask);
		value.graphics                  = programs;
		value.graphics.vertex[0].module = nullptr;
		if (PerVertexVsStoreCompact()) {
			// S409: the record holds the locations of the PS mask; the VS is the same module with that mask as its specialization literal.
			const uint32_t ps_mask = ps->second.mask;
			const auto     module_key = std::pair {key.first, ps_mask};
			auto           cached     = s_compact_vs.find(module_key);
			if (cached == s_compact_vs.end()) {
				auto words = vs->second.words;
				EXIT_IF(vs->second.compact_spec_id == 0 || !PvCompactPatchMask(words, vs->second.compact_spec_id, ps_mask));
				cached = s_compact_vs.emplace(module_key, CompileSPV(words, graphics.device)).first;
			}
			value.stride_vec4               = PvCompactStride(ps_mask);
			value.graphics.vertex[0].module = cached->second;
			value.compact                   = true;
			// one module per (VS, mask): the id tells the pipelines of one VS with different masks apart
			value.graphics.vertex[0].id |= (1ull << 60) | (uint64_t {ps_mask} << 29);
			if (s_compact_pair_lines++ < kMaxCompactLines) {
				LOGF("[pv-compact] #%u VS=0x%016llx PS=0x%016llx vs_mask=0x%x ps_mask=0x%x stride %u -> %u\n", index,
				     static_cast<unsigned long long>(vs->second.hash), static_cast<unsigned long long>(ps->second.hash),
				     vs->second.mask, ps_mask, PvVsStoreStrideFor(vs->second.mask), value.stride_vec4);
			}
		} else {
			value.graphics.vertex[0].module = CompileSPV(vs->second.words, graphics.device);
		}
		value.graphics.pixel.module     = CompileSPV(ps->second.words, graphics.device);
		value.graphics.vertex[0].id |= 1ull << 61;
		value.graphics.pixel.id |= 1ull << 61;
		value.graphics.pv_store_layout = PvVsStoreSetLayout(graphics);
	}
	return &s_pairs.emplace(key, std::move(value)).first->second;
}

PvVsStoreRuntime::PvVsStoreRuntime(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler), m_slabs(kSlabVec4s, kMaxSlabs) {}

PvVsStoreRuntime::~PvVsStoreRuntime() = default;

uint64_t PvVsStoreRuntime::SlabVec4s() {
	return kSlabVec4s;
}

PvVsStoreRuntime::Region PvVsStoreRuntime::Place(uint64_t records, uint32_t stride_vec4) {
	Region region;
	if (stride_vec4 == 0 || records == 0) return region;
	// The shaders address record * stride + slot, so a region starts at a multiple of the stride.
	const uint64_t vec4s     = records * stride_vec4 + stride_vec4;
	const auto     placement = m_slabs.Place(vec4s, m_scheduler.CurrentTick(),
	                                         [&](uint64_t tick) { return m_scheduler.IsFree(tick); });
	if (!placement.ok) return region;
	if (placement.created) {
		EXIT_IF(placement.slab != m_buffers.size());
		m_buffers.push_back(std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0, AllFlags,
		                                             kSlabVec4s * 16));
	}
	const uint64_t aligned = (uint64_t {placement.base} + stride_vec4 - 1) / stride_vec4 * stride_vec4;
	region.ok          = true;
	region.slab        = placement.slab;
	region.base_record = static_cast<uint32_t>(aligned / stride_vec4);
	region.buffer      = m_buffers.at(placement.slab)->Handle();
	region.buffer_size = m_buffers.at(placement.slab)->Size();
	return region;
}

void PvVsStoreRuntime::Touch(const Region& region) {
	m_slabs.Touch(region.slab, m_scheduler.CurrentTick());
}

} // namespace Libs::Graphics
