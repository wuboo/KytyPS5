#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_CPWALKQUERY_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_CPWALKQUERY_H_

// S608 (KYTY_CP_WALK_QUERY=0|1, unset = 0).
// The CP's clean-walk question on a walker dword (runtimeLinker.cpp: the KYTY_WALK_NOFAULT reader before the mov, TryServeCleanWalk in the
// fault handler, the K-rf queries of a dispatch walker) goes through Memory::QueryGuestFaultBytes, which takes each lock of the byte tracker
// with ONE try_lock (S428: written for a signal handler of a guest thread, which must never wait for the CP). The lock is only contended by
// readers: the CP is the only thread that changes the GPU-dirty set. With KYTY_WORKER_CLEAN_WALK=1 (S607) three pipeline workers ask the
// same tracker ~7000 times per flip with a spin, so the CP's single try fails: the NOFAULT reader answers "busy" (ordinary mov, walk_nf_ordinary
// 6 -> 26/flip in S607 t), the load faults on the protected page, the handler's try fails again often enough (~1/flip) that the clean walk
// decides Drain (CleanWalk::Decide: busy -> drain), and the drain is a synchronous GPU readback of the page (rf_gpu_walker 0 -> 1.07/flip at
// 4.6 ms each = the whole +4.6 ms of pipe_cp_gfp_ns and +4.7 ms of cp_wait_gpu in S607 t).
// 1: on the GPU thread the question is answered from the CP's own view: GPU-dirty from the set without its lock (the CP is the set's only writer,
// gpuDirtyRanges.h; the same read HasGpuDirtyBytes and the S607 join do), in flight with a bounded spin on InflightRanges' lock (its Remove runs
// on the publish thread). The answer is the one a successful try_lock gives at the same moment; it is never "busy" because a reader holds a lock.
// Other threads (guest handlers, workers) keep their path. Not codegen.
//
// Mutants (tests/CpWalkQueryTests.cpp, WILL_FAIL): KYTY_CPWQ_MUTANT_MODE_IGNORED (the CP view without the switch), _ANY_THREAD (the CP view off
// the GPU thread), _TRY_LOCK (the dirty part takes one try_lock as before), _NO_INFLIGHT (copies in flight are not asked), _NO_DIRTY (the
// GPU-dirty set is not asked), _INFLIGHT_BLOCKS (the in-flight lock is taken without a bound).

#include "graphics/host_gpu/renderer/cache/guestWait.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::CpWalkQuery {

enum class Mode : uint8_t { Off, On };

inline Mode Parse(const char* text) {
	if (text != nullptr && std::strcmp(text, "1") == 0) return Mode::On;
	return Mode::Off;
}

inline Mode EnvMode() {
	static const Mode mode = Parse(std::getenv("KYTY_CP_WALK_QUERY"));
	return mode;
}

inline const char* ModeName(Mode mode) {
	return mode == Mode::On ? "1" : "0";
}

// Tries of the spin on the in-flight lock (GuestWait::SpinTryLock: a pause between tries, a yield every 64).
constexpr int kInflightSpin = GuestWait::kSpinTries;

// Whether this thread's query takes the CP view.
constexpr bool UseCpView(Mode mode, bool gpu_thread) {
#if defined(KYTY_CPWQ_MUTANT_MODE_IGNORED)
	mode = Mode::On;
#endif
#if defined(KYTY_CPWQ_MUTANT_ANY_THREAD)
	gpu_thread = true;
#endif
	return mode == Mode::On && gpu_thread;
}

// The CP view (GPU thread only): Dirty = GpuDirtyRanges, Inflight = InflightRanges (templates for the tests). False (busy) only when the spin on
// the in-flight lock gave up; *dirty and *inflight are then unspecified.
template <typename Dirty, typename Inflight>
bool Query(const Dirty& dirty_set, const Inflight& inflight_set, uint64_t vaddr, uint64_t size, bool* dirty, bool* inflight,
           int tries = kInflightSpin) {
#if defined(KYTY_CPWQ_MUTANT_TRY_LOCK)
	if (!dirty_set.TryIntersects(vaddr, size, dirty)) return false;
#elif defined(KYTY_CPWQ_MUTANT_NO_DIRTY)
	(void)dirty_set;
	*dirty = false;
#else
	// Dirty first, in flight second (BeginDownloadCopy adds to in-flight before it removes from dirty; on the CP both happen on this thread).
	*dirty = dirty_set.Intersects(vaddr, size);
#endif
#if defined(KYTY_CPWQ_MUTANT_NO_INFLIGHT)
	(void)inflight_set;
	(void)tries;
	*inflight = false;
	return true;
#elif defined(KYTY_CPWQ_MUTANT_INFLIGHT_BLOCKS)
	(void)tries;
	*inflight = inflight_set.Intersects(vaddr, size);
	return true;
#else
	return inflight_set.TryIntersectsSpin(vaddr, size, inflight, nullptr, tries);
#endif
}

} // namespace Libs::Graphics::CpWalkQuery

#endif
