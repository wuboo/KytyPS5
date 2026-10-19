#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_RTMISS_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_RTMISS_H_

#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler {

// KYTY_RT_MISS=1 (diagnostics): every BvhIntersect answers "no hit" after it has done its work (the
// answer is chosen at the end), so a traversal ends at its first node but the cost stays.
// KYTY_RT_MISS=2 (S446, diagnostics, ablation of the cost): BvhIntersect returns the "no hit" answer at
// its first instruction, with no descriptor decoding and no loads from the structure.
// Read per shader compile (cheap, tests can switch it); the value is part of the in-memory program cache
// key, so a change gives a new permutation.
inline int RtMissMode() {
#ifdef KYTY_RTMISS_MUTANT_IGNORED
	return 0;
#else
	const char* value = std::getenv("KYTY_RT_MISS");
	if (value == nullptr || value[0] == '\0' || value[0] == '0') return 0;
	return value[0] == '2' ? 2 : 1;
#endif
}

// True when BvhIntersect must answer before doing any work (mode 2). Mutant RTMISS2_AFTER_LOOP: the answer
// comes only at the end, like mode 1, so the cost is still paid.
inline bool RtMissEarlyReturn() {
#ifdef KYTY_RTMISS2_MUTANT_AFTER_LOOP
	return false;
#else
	return RtMissMode() == 2;
#endif
}

inline bool RtMissEnabled() {
	return RtMissMode() != 0;
}

} // namespace Libs::Graphics::ShaderRecompiler

#endif
