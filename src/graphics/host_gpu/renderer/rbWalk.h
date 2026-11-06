#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RBWALK_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RBWALK_H_

// S502 (KYTY_RB_WALK=0|1|shadow, unset = 0): the upstream SRT walker loads guest memory with
// mov eax,[r10] (SrtWalker::LoadGuest). A 4 KiB page that holds any GPU-written byte is
// protected, so the load faults and ReadMemory drains the queue even when this dword is not
// GPU-dirty and no copy of it is in flight (false sharing; the same rule as KYTY_CP_READ_BACKING).
//   0       the fault, as before
//   1       a walker load whose 4 bytes are known clean is completed from the backing (no drain)
//   shadow  the fault still runs; the backing dword is compared with the drained bytes (ne = 0)
// A query that cannot answer (lock busy, range not tracked) stays on the fault. Dirty or
// in-flight bytes stay on the fault: the backing would be stale.
// Mutants (tests/RbWalkTests.cpp): KYTY_RBWALK_MUTANT_SERVE_DIRTY, _SERVE_INFLIGHT, _SERVE_UNKNOWN, _SHADOW_SKIPS_FAULT.

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::RbWalk {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_RB_WALK"));
	return mode;
}

enum class Source : uint8_t {
	Fault,   // the page fault drains, as before
	Backing, // complete mov eax from the backing
	Shadow,  // the fault runs; the caller compares the backing dword with the drained bytes
};

// dirty / inflight: the 4 bytes the load reads. unknown: the tracker could not be queried.
constexpr Source Decide(Mode mode, bool dirty, bool inflight, bool unknown) {
	if (mode == Mode::Off) return Source::Fault;
#if defined(KYTY_RBWALK_MUTANT_SERVE_DIRTY)
	dirty = false;
#endif
#if defined(KYTY_RBWALK_MUTANT_SERVE_INFLIGHT)
	inflight = false;
#endif
#if defined(KYTY_RBWALK_MUTANT_SERVE_UNKNOWN)
	unknown = false;
#endif
	if (dirty || inflight || unknown) return Source::Fault;
#if defined(KYTY_RBWALK_MUTANT_SHADOW_SKIPS_FAULT)
	return Source::Backing;
#else
	return mode == Mode::Shadow ? Source::Shadow : Source::Backing;
#endif
}

} // namespace Libs::Graphics::RbWalk

#endif
