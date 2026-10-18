#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PVINDIRECT_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PVINDIRECT_H_

// S408: KYTY_PV_INDIRECT=0|1|shadow (unset = 0). A VS-store draw (KYTY_PV_VSSTORE) whose indexed indirect args are written by the GPU is
// recorded as vkCmdDrawIndexedIndirect from a PREPARED copy of the args, so the CPU does not read the args (the read drains the queue;
// S401/S405: draw A, ~0.4 ms, one of the two drains of the chain). The VS-store shaders do not change: a record is
//   base + (instance - first_instance) * span + (vertex_index - first)
// so a region for `cap` instances (span * cap records) is enough for any draw with at most `cap` instances. A small compute shader
// ("prepare") copies the args the guest wrote into a host ring, with
//   - instanceCount clamped to cap, and
//   - the whole draw zeroed (indexCount 0) when index_count / first_index / vertex_offset / first_instance are not the values the
//     CPU learned (those four are baked into the CPU side plan and the push constants).
// A status word per flip (mismatch / overflow / max instance) is read two flips later without waiting; it drives the cooldown (back to
// the CPU read for kCooldownFlips and learn again) and the growth of cap.
//   0       the old path, bit for bit
//   shadow  the old path (CPU reads the args); everything of the new path but the draw: learn, decision, prepare compute recorded next
//           to the draw, the dry plan with cap; the prepared args are compared with what the CPU read in the same flip
//   1       a learned entry in state `use` draws from the prepared args; every other case takes the old path and learns
// Pure part; tests/PvIndirectTests.cpp pins it (the GLSL of pv_indirect_prepare.comp has the same arithmetic as PrepareArgs).

#include "graphics/host_gpu/renderer/indirectGpu.h"
#include "graphics/host_gpu/renderer/perVertexVsStore.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::PvIndirect {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_PV_INDIRECT"));
	return mode;
}

inline constexpr uint32_t kCooldownFlips = 64;
inline constexpr uint32_t kDefaultCap    = 8192;
inline constexpr uint32_t kMaxCap        = uint32_t {1} << 20;
inline constexpr uint32_t kMaxEntries    = 8; // status slots per flip; more args addresses stay on the old path

inline uint32_t ParseCap(const char* text) {
	if (text == nullptr) return kDefaultCap;
	char*      end   = nullptr;
	const auto value = std::strtoul(text, &end, 10);
	if (end == text || *end != '\0' || value == 0 || value > kMaxCap) return kDefaultCap;
	return static_cast<uint32_t>(value);
}
inline uint32_t EnvCap() {
	static const uint32_t cap = ParseCap(std::getenv("KYTY_PV_INDIRECT_CAP"));
	return cap;
}

constexpr uint32_t NextPow2(uint32_t value) {
	if (value <= 1) return 1;
	uint32_t result = 1;
	while (result < value && result < (uint32_t {1} << 31u)) result <<= 1u;
	return result;
}

// vkDrawIndexedIndirectCommand = the guest's DrawIndexedIndirectArgs: dword order.
struct Args {
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;
	uint32_t first_index    = 0;
	uint32_t vertex_offset  = 0; // bits of the signed offset
	uint32_t first_instance = 0;
	bool     operator==(const Args&) const = default;
};

// The four values a prepared draw keeps from the guest args.
struct Fields {
	uint32_t index_count    = 0;
	uint32_t first_index    = 0;
	uint32_t vertex_offset  = 0;
	uint32_t first_instance = 0;
	bool     operator==(const Fields&) const = default;
};
constexpr Fields FieldsOf(const Args& args) {
	return {args.index_count, args.first_index, args.vertex_offset, args.first_instance};
}

// ---- the arithmetic of pv_indirect_prepare.comp ----
struct Status {
	uint32_t mismatch     = 0; // draws whose fields are not the expected ones
	uint32_t overflow     = 0; // draws with more instances than cap
	uint32_t max_instance = 0; // largest instance_count of a draw whose fields matched
	uint32_t draws        = 0;
	bool     operator==(const Status&) const = default;
};
struct Prepared {
	std::array<uint32_t, 5> out {};
	Status                  status;
	bool                    operator==(const Prepared&) const = default;
};

constexpr Prepared PrepareArgs(const Args& in, const Fields& expected, uint32_t cap) {
	Prepared result;
	result.status.draws = 1;
#if defined(KYTY_PVINDIRECT_MUTANT_IGNORE_FIELDS)
	const bool match = true;
#else
	const bool match = FieldsOf(in) == expected;
#endif
	if (!match) {
		result.status.mismatch = 1;
		return result; // indexCount 0: the draw draws nothing
	}
#if defined(KYTY_PVINDIRECT_MUTANT_CLAMP_OFF)
	const uint32_t instances = in.instance_count;
#else
	const uint32_t instances = in.instance_count < cap ? in.instance_count : cap;
#endif
	result.out                = {expected.index_count, instances, expected.first_index, expected.vertex_offset, 0u};
	result.status.overflow     = in.instance_count > cap ? 1u : 0u;
	result.status.max_instance = in.instance_count;
	return result;
}

// ---- what the CPU keeps per args address ----
struct Entry {
	bool     learned        = false;
	bool     off            = false; // never again: more instances than a region can hold
	Fields   fields;
	uint32_t cap            = 0;
	uint32_t max_seen       = 0;
	uint64_t cooldown_until = 0; // flips < this take the old path
	uint32_t id             = 0; // status slot
};

constexpr uint32_t InitialCap(uint32_t max_seen, uint32_t env_cap) {
	const uint32_t grown = NextPow2(max_seen > (uint32_t {1} << 30u) ? (uint32_t {1} << 30u) : max_seen * 2u);
	return env_cap > grown ? env_cap : grown;
}

// Args the prepared draw can stand in for: the first index and instance are the baked zero, the draw fits the bound index range.
constexpr bool Preparable(const Args& args, uint32_t index_buffer_size) {
	return args.index_count != 0 && args.first_index == 0 && args.first_instance == 0 && index_buffer_size != 0 &&
	       args.index_count <= index_buffer_size;
}

// The CPU read the args of the draw (old path): learn their fields and size. False: these args cannot be prepared; the entry is not
// learned (it keeps taking the old path, which keeps looking at the args).
inline bool Learn(Entry& entry, const Args& args, uint32_t index_buffer_size, uint32_t env_cap) {
	if (!Preparable(args, index_buffer_size)) {
		entry.learned = false;
		return false;
	}
	entry.fields   = FieldsOf(args);
	entry.learned  = true;
	entry.max_seen = std::max(entry.max_seen, args.instance_count);
	entry.cap      = std::max(entry.cap, InitialCap(entry.max_seen, env_cap));
	return true;
}

enum class Verdict : uint8_t { Learn, Cooldown, Use, Off };
constexpr Verdict Decide(const Entry& entry, uint64_t flip) {
	if (entry.off) return Verdict::Off;
	if (!entry.learned) return Verdict::Learn;
	if (flip < entry.cooldown_until) return Verdict::Cooldown;
	return Verdict::Use;
}

// The status of the draws of one flip, read two flips later.
struct StatusEffect {
	bool cooldown = false;
	bool grew     = false;
	bool off      = false;
};
inline StatusEffect OnStatus(Entry& entry, const Status& status, uint64_t flip, uint32_t cap_limit) {
	StatusEffect effect;
	if (status.mismatch != 0) {
#if !defined(KYTY_PVINDIRECT_MUTANT_NO_COOLDOWN)
		entry.cooldown_until = flip + kCooldownFlips;
#endif
		effect.cooldown = true;
	}
	entry.max_seen = std::max(entry.max_seen, status.max_instance);
	if (status.overflow != 0) {
		if (status.max_instance > cap_limit) {
			entry.off  = true;
			effect.off = true;
		} else {
#if !defined(KYTY_PVINDIRECT_MUTANT_CAP_NO_GROW)
			const uint32_t wanted = std::min(NextPow2(status.max_instance * 2u), cap_limit);
			if (wanted > entry.cap) {
				entry.cap  = wanted;
				effect.grew = true;
			}
#endif
		}
	}
	return effect;
}

// Largest cap whose region (span * cap records of stride vec4s, plus the alignment record) fits a slab.
constexpr uint32_t CapLimit(uint64_t slab_vec4s, uint32_t span, uint32_t stride_vec4) {
	if (span == 0 || stride_vec4 == 0) return 0;
	const uint64_t records = slab_vec4s / stride_vec4;
	const uint64_t limit   = records > 1 ? (records - 1) / span : 0;
	return static_cast<uint32_t>(std::min<uint64_t>(limit, kMaxCap));
}

// The plan made with `cap` instances (no wait for the args) must be the plan of the real draw: same span and first vertex, and the real
// records fit the region.
constexpr bool PlanEq(const PvVsStorePlan& cap_plan, const PvVsStorePlan& real_plan) {
	return cap_plan.Use() && real_plan.Use() && cap_plan.span == real_plan.span && cap_plan.first == real_plan.first &&
	       real_plan.records <= cap_plan.records;
}

// What the draw gets from its host path, besides the VS-store: the reasons of IndirectGpu::QualifyHost that remain once the VS-store and
// the prototype (which the declined draw never reaches) are taken away.
inline IndirectGpu::Reason ResidualReason(IndirectGpu::HostPath host) {
	host.pv_store_possible = false;
	host.proto_possible    = false;
	return IndirectGpu::QualifyHost(host);
}

// Why a draw that could have used the prepared args went to the old path (columns pv_ind_decline_<reason>_n).
enum class Decline : uint8_t { Learn, Cooldown, Off, Residual, Range, Ring, IndexTail, TooLarge, Other, Count };
constexpr const char* DeclineName(Decline reason) {
	switch (reason) {
		case Decline::Learn: return "learn";
		case Decline::Cooldown: return "cooldown";
		case Decline::Off: return "off";
		case Decline::Residual: return "residual";
		case Decline::Range: return "range";
		case Decline::Ring: return "ring";
		case Decline::IndexTail: return "index_tail";
		case Decline::TooLarge: return "too_large";
		default: return "other";
	}
}

// Order of the commands of a prepared draw (the production recorder and the tests share it): the pass is closed first when open,
// the compute runs, and its output is made visible to the indirect read before the draw.
template <class Recorder>
inline bool RecordPrepare(Recorder& rec, bool pass_open) {
	if (pass_open) rec.EndPass();
	rec.BindCompute();
	rec.Push();
	rec.Dispatch();
	rec.BarrierToIndirect();
	return pass_open;
}

} // namespace Libs::Graphics::PvIndirect

#endif
