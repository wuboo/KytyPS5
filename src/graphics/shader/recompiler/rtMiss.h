#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_RTMISS_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_RTMISS_H_

#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler {

// KYTY_RT_MISS=1 (diagnostics): every BvhIntersect answers "no hit" without reading the structure, so
// a traversal ends at its first node. Read per shader compile (cheap, tests can switch it); the value
// is part of the in-memory program cache key, so a change gives a new permutation.
inline bool RtMissEnabled() {
#ifdef KYTY_RTMISS_MUTANT_IGNORED
	return false;
#else
	const char* value = std::getenv("KYTY_RT_MISS");
	return value != nullptr && value[0] != '\0' && value[0] != '0';
#endif
}

} // namespace Libs::Graphics::ShaderRecompiler

#endif
