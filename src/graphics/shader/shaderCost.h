#ifndef KYTY_GRAPHICS_SHADER_SHADERCOST_H_
#define KYTY_GRAPHICS_SHADER_SHADERCOST_H_

// S433 (S441, diagnostics, no change of behaviour): helpers behind the first-use cost counters of shaders.
//
// Lead: the time from the game registering a shader (ShaderMapUserData, the first time we see it) to the first draw/dispatch that looks it up
// (ShaderGetMappedData). It tells how much a background pre-warm could get ahead of the command processor (plan-hitch S446).
// Miss: which of program miss / permutation miss / call program a lookup that did not return a cached permutation counts as.
//
// Mutants (tests/ShaderCostTests.cpp): KYTY_SHADERCOST_MUTANT_{COUNT_ON_HIT, LEAD_EVERY_USE, REMAP_RESETS_USED}.

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace Libs::Graphics::ShaderCost {

// Upper edges in ms; the last bucket is open: [0,1) [1,5) [5,20) [20,50) [50,100) [100,250) [250,1000) [1000,..).
inline constexpr std::array<uint64_t, 7> kLeadEdgesMs = {1, 5, 20, 50, 100, 250, 1000};
inline constexpr size_t                  kLeadBuckets = kLeadEdgesMs.size() + 1;

[[nodiscard]] inline size_t LeadBucket(uint64_t lead_ns) {
	const uint64_t ms = lead_ns / 1'000'000ull;
	size_t         i  = 0;
	while (i < kLeadEdgesMs.size() && ms >= kLeadEdgesMs[i]) {
		i++;
	}
	return i;
}

struct LeadState {
	uint64_t map_ns = 0;
	bool     used   = false;
};

// A shader registered again with the same code keeps its first-seen time and its used flag (the game re-registers); other code starts over.
[[nodiscard]] inline LeadState LeadOnMapped(const LeadState* previous, uint64_t previous_hash, uint64_t new_hash, uint64_t now_ns) {
	if (previous != nullptr && previous_hash == new_hash) {
#ifdef KYTY_SHADERCOST_MUTANT_REMAP_RESETS_USED
		return LeadState {previous->map_ns, false};
#else
		return *previous;
#endif
	}
	return LeadState {now_ns, false};
}

// The lead in ns on the first use after registering, nullopt on every later use.
[[nodiscard]] inline std::optional<uint64_t> LeadOnUse(LeadState& state, uint64_t now_ns) {
#ifndef KYTY_SHADERCOST_MUTANT_LEAD_EVERY_USE
	if (state.used) {
		return std::nullopt;
	}
#endif
	state.used = true;
	return now_ns >= state.map_ns ? now_ns - state.map_ns : 0;
}

struct MissCount {
	uint32_t program_miss = 0; // the program key was not in the cache: Prepare + Translate + Extract + Materialize
	uint32_t perm_miss    = 0; // no permutation for this specialization / push start: Compile (+ PV variants)
	uint32_t call_program = 0; // the miss is a program with a retained call source (its IR depends on guest memory)
};

// program_found: the ProgramKey was in the cache; permutation_found: a permutation matched (a cache hit).
[[nodiscard]] inline MissCount ClassifyLookup(bool program_found, bool permutation_found, bool call_source) {
	MissCount count;
#ifdef KYTY_SHADERCOST_MUTANT_COUNT_ON_HIT
	(void)permutation_found;
	count.program_miss = program_found ? 0u : 1u;
	count.perm_miss    = 1;
#else
	if (permutation_found) {
		return count;
	}
	count.program_miss = program_found ? 0u : 1u;
	count.perm_miss    = 1;
#endif
	count.call_program = call_source ? 1u : 0u;
	return count;
}

} // namespace Libs::Graphics::ShaderCost

#endif
