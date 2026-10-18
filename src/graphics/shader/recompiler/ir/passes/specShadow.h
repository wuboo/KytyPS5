#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SPECSHADOW_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SPECSHADOW_H_

// S406 (KYTY_DIAG_SYNC=1, DIAG, default off): what the readers of the specialization memory are for, and what the materialization
// decides when those reads see stale bytes of GPU-written memory.
//
// Purpose: a thread-local tag, set by the code that evaluates values on the clean (strict) reader, so the readback that drains the
// queue can say what it was for. Innermost scope wins, leaving a scope restores the outer tag.
//   cond   the condition of a control-flow block of the SRT plan (chooses the reachable blocks, so the active descriptor sources)
//   slot   a clean flattened-SRT slot
//   desc   descriptor evaluation (buffers / images / samplers), including indirect tables
//   fill   uniform_fill parameters (input of the decision to replace a dispatch by a fill)
//   wg     workgroup counts of an indirect dispatch
//   call   refresh of the call source (RefreshShaderSource)
//
// Shadow runs: MaterializeResources on scratch copies, with a reader that never drains:
//   stale     every read returns the bytes in guest memory now (the bytes before the drain)
//   superset  as stale, except that a read of GPU-dirty bytes for purpose `cond` fails, which the walker answers by following both
//             sides of the branch (the active sources of both are materialized, as on the real GPU)
// Compare() then sets them against the fresh materialization (the real one, after the drain).

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR::SpecShadow {

enum class Purpose : uint8_t { Other, Cond, Slot, Desc, Fill, Wg, Call, Count };
inline constexpr size_t kPurposeCount = static_cast<size_t>(Purpose::Count);

constexpr const char* PurposeName(Purpose purpose) {
	switch (purpose) {
		case Purpose::Cond: return "cond";
		case Purpose::Slot: return "slot";
		case Purpose::Desc: return "desc";
		case Purpose::Fill: return "fill";
		case Purpose::Wg: return "wg";
		case Purpose::Call: return "call";
		default: return "other";
	}
}

inline Purpose& CurrentPurposeRef() {
	static thread_local Purpose purpose = Purpose::Other;
	return purpose;
}
inline Purpose CurrentPurpose() {
	return CurrentPurposeRef();
}

class PurposeScope {
public:
	explicit PurposeScope(Purpose purpose): m_previous(CurrentPurposeRef()) {
#if defined(KYTY_SPECSHADOW_MUTANT_TAG_IGNORED)
		(void)purpose;
#else
		CurrentPurposeRef() = purpose;
#endif
	}
	~PurposeScope() { CurrentPurposeRef() = m_previous; }
	PurposeScope(const PurposeScope&)            = delete;
	PurposeScope& operator=(const PurposeScope&) = delete;

private:
	Purpose m_previous;
};

// True while a shadow run is in progress on this thread: a failure of a shadow run is a result, not a message.
inline bool& ShadowRunningRef() {
	static thread_local bool running = false;
	return running;
}
inline bool ShadowRunning() {
	return ShadowRunningRef();
}

// ---- shadow reader ----
enum class ReadMode : uint8_t { Stale, Superset };

struct ShadowReader {
	ReadMode mode = ReadMode::Stale;
	// Bytes in guest memory now (no drain). The real reader is TryReadBufferBacking; this one is TryReadBacking.
	bool (*backing)(uint64_t address, void* data, uint64_t size) = nullptr;
	// True when the range holds GPU-written bytes the CPU has not seen yet (QueryShaderGpuDirty).
	bool     (*dirty)(uint64_t address, uint64_t size) = nullptr;
	uint32_t reads                         = 0;
	uint32_t dirty_reads[kPurposeCount]    = {};
	[[nodiscard]] uint32_t DirtyTotal() const {
		uint32_t sum = 0;
		for (const auto n: dirty_reads) sum += n;
		return sum;
	}
	// S408: dirty reads for every purpose but the condition (they drain in the real superset mode too).
	[[nodiscard]] uint32_t DirtyNonCond() const { return DirtyTotal() - dirty_reads[static_cast<size_t>(Purpose::Cond)]; }
};

// S408: dirty non-condition reads the superset run makes and the stale run does not (new drains in the blocks the superset adds).
constexpr uint32_t DirtyNonCondAdded(uint32_t super_noncond, uint32_t stale_noncond) {
	return super_noncond > stale_noncond ? super_noncond - stale_noncond : 0u;
}

// SrtMemoryReader for the shadow runtimes; `userdata` is a ShadowReader.
inline bool ShadowRead(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& reader = *static_cast<ShadowReader*>(userdata);
	if (values.empty() || reader.backing == nullptr) return false;
	reader.reads++;
	const bool is_dirty = reader.dirty != nullptr && reader.dirty(address, values.size_bytes());
	if (is_dirty) {
		const auto purpose = CurrentPurpose();
		reader.dirty_reads[static_cast<size_t>(purpose)]++;
#if defined(KYTY_SPECSHADOW_MUTANT_SUPERSET_READS_STALE)
		(void)purpose;
#else
		// The value is unknown before the GPU finishes: the walker follows both sides of a branch it cannot decide.
		if (reader.mode == ReadMode::Superset && purpose == Purpose::Cond) return false;
#endif
	}
	return reader.backing(address, values.data(), values.size_bytes());
}

// ---- outcome of one materialization and the comparison ----
struct Outcome {
	bool                   ok = false; // MaterializeResources returned true
	ResourceSnapshot       snapshot;
	ResourceSpecialization specialization;
	std::vector<uint8_t>   active; // plan.active_sources after the run (empty: no control flow)
};

// Runs MaterializeResources into the scratch outcome; the plan's own scratch (active_sources, selector_keys, ...) is recomputed
// by every run, so the run that follows sees nothing of this one.
inline void Materialize(const ResourcePlan& plan, const SrtRuntime& runtime, Outcome& out) {
	ShadowRunningRef() = true;
	out.ok             = MaterializeResources(plan, runtime, out.snapshot, out.specialization);
	ShadowRunningRef() = false;
	out.active         = plan.active_sources;
}

enum class Part : uint8_t { None, Failed, Specialization, Active, Flat, Buffers, Images, Samplers, UniformFill };

constexpr const char* PartName(Part part) {
	switch (part) {
		case Part::Failed: return "failed";
		case Part::Specialization: return "specialization";
		case Part::Active: return "active";
		case Part::Flat: return "flat";
		case Part::Buffers: return "buffers";
		case Part::Images: return "images";
		case Part::Samplers: return "samplers";
		case Part::UniformFill: return "uniform_fill";
		default: return "none";
	}
}

// First part in which `other` differs from `fresh` (None: identical).
inline Part FirstDifference(const Outcome& other, const Outcome& fresh) {
	if (!other.ok) return Part::Failed;
	if (other.specialization != fresh.specialization) return Part::Specialization;
	if (other.active != fresh.active) return Part::Active;
	if (other.snapshot.flattened_srt != fresh.snapshot.flattened_srt) return Part::Flat;
	if (other.snapshot.buffers != fresh.snapshot.buffers) return Part::Buffers;
	if (other.snapshot.images != fresh.snapshot.images) return Part::Images;
	if (other.snapshot.samplers != fresh.snapshot.samplers) return Part::Samplers;
	if (!(other.snapshot.uniform_fill == fresh.snapshot.uniform_fill)) return Part::UniformFill;
	return Part::None;
}

struct Verdict {
	Part     stale_part     = Part::None; // stale vs fresh
	bool     super_ok       = false;
	uint32_t extra_sources  = 0;  // sources active in the superset and not in the fresh run
	uint32_t new_images     = 0;  // non-null image descriptors of the superset that the fresh run does not have
	[[nodiscard]] bool StaleEqual() const { return stale_part == Part::None; }
};

inline bool NullDescriptor(const DescriptorValue& value) {
	for (uint32_t i = 0; i < value.dword_count && i < value.dwords.size(); i++) {
		if (value.dwords[i] != 0) return false;
	}
	return true;
}

inline Verdict Compare(const Outcome& stale, const Outcome& superset, const Outcome& fresh) {
	Verdict verdict;
#if defined(KYTY_SPECSHADOW_MUTANT_SHADOW_FRESH)
	// Mutant: the stale shadow is taken from the fresh run, so the comparison can never see a difference.
	verdict.stale_part = FirstDifference(fresh, fresh);
#else
	verdict.stale_part = FirstDifference(stale, fresh);
#endif
	verdict.super_ok = superset.ok;
	if (superset.ok) {
		for (size_t i = 0; i < superset.active.size(); i++) {
			const bool in_fresh = i < fresh.active.size() ? fresh.active[i] != 0 : true;
			if (superset.active[i] != 0 && !in_fresh) verdict.extra_sources++;
		}
		for (const auto& image: superset.snapshot.images) {
			if (NullDescriptor(image)) continue;
			bool found = false;
			for (const auto& other: fresh.snapshot.images) {
				if (other == image) {
					found = true;
					break;
				}
			}
			if (!found) verdict.new_images++;
		}
	}
	return verdict;
}

// ---- S408: what the sources added by the superset are ----
// A source is "extra" when the superset run activates it and the fresh run does not. The kinds are the resources that use it.
struct ExtraSource {
	uint32_t source  = 0;
	char     kind    = '?'; // b buffer, i image, s sampler
	uint32_t index   = 0;   // resource index in plan.info
	bool     written = false;
	uint64_t address = 0;   // base of the V# / T# (0: null descriptor or a sampler)
	bool     bad     = false; // not null, and below guest memory or not backed
};
struct ExtraSummary {
	uint32_t                 buffers = 0, images = 0, samplers = 0, written = 0, bad_addr = 0;
	std::vector<ExtraSource> list;
};

inline constexpr uint64_t kGuestMemoryMin = 0x40000;

// `readable(address)`: the address is backed guest memory (may be null: then only the lower bound is checked).
inline ExtraSummary ClassifyExtra(const ResourcePlan& plan, const Outcome& superset, const Outcome& fresh, bool (*readable)(uint64_t)) {
	ExtraSummary summary;
	if (!superset.ok) return summary;
	const auto extra = [&](uint32_t source) {
		const bool in_super = source < superset.active.size() && superset.active[source] != 0;
		const bool in_fresh = source < fresh.active.size() ? fresh.active[source] != 0 : true;
		return in_super && !in_fresh;
	};
	const auto bad_address = [&](uint64_t address) {
		if (address == 0) return false;
		return address < kGuestMemoryMin || (readable != nullptr && !readable(address));
	};
	const auto add = [&](ExtraSource item) {
		item.bad = bad_address(item.address);
		if (item.kind == 'b') summary.buffers++;
		if (item.kind == 'i') summary.images++;
		if (item.kind == 's') summary.samplers++;
		if (item.written) summary.written++;
		if (item.bad) summary.bad_addr++;
		summary.list.push_back(item);
	};
	for (uint32_t i = 0; i < plan.info.buffers.size() && i < superset.snapshot.buffers.size(); i++) {
		const auto& resource = plan.info.buffers[i];
		if (!extra(resource.source)) continue;
		const auto& dwords = superset.snapshot.buffers[i].dwords;
		add({resource.source, 'b', i, resource.written, uint64_t {dwords[0]} | (uint64_t {dwords[1] & 0xffffu} << 32u), false});
	}
	for (uint32_t i = 0; i < plan.info.images.size() && i < superset.snapshot.images.size(); i++) {
		const auto& resource = plan.info.images[i];
		if (!extra(resource.source)) continue;
		const auto& dwords = superset.snapshot.images[i].dwords;
		add({resource.source, 'i', i, resource.written, ((uint64_t {dwords[0]} | (uint64_t {dwords[1]} << 32u)) & 0xffffffffffull) << 8u, false});
	}
	for (uint32_t i = 0; i < plan.info.samplers.size(); i++) {
		if (!extra(plan.info.samplers[i].source)) continue;
		add({plan.info.samplers[i].source, 's', i, false, 0, false});
	}
	return summary;
}

// First differing dword of the stale and the fresh outcome in `part`, for the log: index, stale word, fresh word.
struct WordDiff {
	bool     valid = false;
	uint32_t index = 0;
	uint32_t stale = 0;
	uint32_t fresh = 0;
};
inline WordDiff FirstWordDiff(Part part, const Outcome& stale, const Outcome& fresh) {
	WordDiff diff;
	const auto scan = [&](const std::vector<uint32_t>& a, const std::vector<uint32_t>& b) {
		for (size_t i = 0; i < a.size() && i < b.size(); i++) {
			if (a[i] != b[i]) {
				diff = {true, static_cast<uint32_t>(i), a[i], b[i]};
				return;
			}
		}
	};
	const auto scan_descriptors = [&](const std::vector<DescriptorValue>& a, const std::vector<DescriptorValue>& b) {
		for (size_t i = 0; i < a.size() && i < b.size(); i++) {
			for (size_t w = 0; w < a[i].dwords.size(); w++) {
				if (a[i].dwords[w] != b[i].dwords[w]) {
					diff = {true, static_cast<uint32_t>(i * 8 + w), a[i].dwords[w], b[i].dwords[w]};
					return;
				}
			}
		}
	};
	switch (part) {
		case Part::Flat: scan(stale.snapshot.flattened_srt, fresh.snapshot.flattened_srt); break;
		case Part::Buffers: scan_descriptors(stale.snapshot.buffers, fresh.snapshot.buffers); break;
		case Part::Images: scan_descriptors(stale.snapshot.images, fresh.snapshot.images); break;
		case Part::Samplers: scan_descriptors(stale.snapshot.samplers, fresh.snapshot.samplers); break;
		case Part::Active:
			for (size_t i = 0; i < stale.active.size() && i < fresh.active.size(); i++) {
				if (stale.active[i] != fresh.active[i]) {
					diff = {true, static_cast<uint32_t>(i), stale.active[i], fresh.active[i]};
					break;
				}
			}
			break;
		case Part::UniformFill:
			diff = {true, 0, stale.snapshot.uniform_fill.value, fresh.snapshot.uniform_fill.value};
			break;
		default: break;
	}
	return diff;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::SpecShadow

#endif
