#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDADEFER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDADEFER_H_

// S601 (KYTY_BDA_DEFER=0|1, default 0): port of KytyPS5#1933 by MrBigBritches ("gpu: hold game-thread BDA events until the command
// processor resumes", f84350b41 + d6d22a51a).
//
// Before every draw or dispatch with DMA (BDA) access, RenderContext::PrepareBda synchronizes the pages queued since the previous one.
// A write fault on a game thread queues its page at once, so a page the game keeps writing for a later frame is uploaded, write-protected
// and faults again at nearly every BDA draw of the frame. With the switch, BufferCache::InvalidateMemory on any thread but the GPU thread
// puts its event into a held-back set. The CP promotes the held-back set into the one TakeBdaSync returns:
// - at the start of every submission slice (GuestGpu::Process);
// - after it read a guest value that a game thread may have written after the data it guards: a passing WAIT_REG_MEM, COND_EXEC, BRANCH,
//   a valid REWIND, SET_PREDICATION on a memory value (op 3), a MEM_SEMAPHORE wait that consumes a count.
// The game thread queues the event of its data (fault handler) before it writes the guard, and the promotion reads the flag below after
// the guard (acquire), so it cannot miss the event.
// With the CP front thread (KYTY_CP_FRONT_THREAD) the front evaluates some of these packets before the recorder reaches the draws in front
// of them. A promotion that comes earlier than its packet only synchronizes more, which is what happens without the switch.
// GPU-thread events (CP writes, buffer creation, the write of a GPU-written page) and every other QueueBdaSync caller stay immediate.
//
// Off: Defer() is false, the held-back set stays empty and every promotion returns at the flag.
//
// Mutants (tests/BdaDeferTests.cpp, WILL_FAIL): PROMOTE_DROPS (a promotion clears the held-back set without moving it),
// GPU_THREAD_DEFERRED (the GPU thread's events are held back too), NO_FLAG (a held-back event does not raise the flag, so the promotion's
// early return leaves it behind).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace Libs::Graphics::BdaDefer {

inline bool Parse(const char* text) {
	return text != nullptr && text[0] == '1' && text[1] == 0;
}

// KYTY_BDA_DEFER (read once).
inline bool EnvOn() {
	static const bool on = Parse(std::getenv("KYTY_BDA_DEFER"));
	return on;
}

// Whether an invalidation is held back: only with the switch and only off the GPU thread.
inline bool Defer(bool on, bool gpu_thread) {
#ifdef KYTY_BDADEFER_MUTANT_GPU_THREAD_DEFERRED
	(void)gpu_thread;
	return on;
#else
	return on && !gpu_thread;
#endif
}

// Caller holds the lock of both sets.
template <typename Set>
void Queue(Set& immediate, Set& held, std::atomic<bool>& has_held, uint64_t begin, uint64_t size, bool deferred) {
	if (!deferred) {
		immediate.Add(begin, size);
		return;
	}
	held.Add(begin, size);
#ifndef KYTY_BDADEFER_MUTANT_NO_FLAG
	has_held.store(true, std::memory_order_release);
#endif
}

// Lock-free check before the lock: false while nothing was held back since the last promotion.
inline bool MayHaveHeld(const std::atomic<bool>& has_held) {
	return has_held.load(std::memory_order_acquire);
}

// Caller holds the lock. Moves the held-back set into the immediate one; returns whether it held anything.
template <typename Set>
bool Promote(Set& immediate, Set& held, std::atomic<bool>& has_held) {
	has_held.store(false, std::memory_order_relaxed);
	if (held.Empty()) {
		return false;
	}
#ifndef KYTY_BDADEFER_MUTANT_PROMOTE_DROPS
	// Every BDA draw takes the immediate set, so it is usually empty: then the held-back set moves whole.
	if (immediate.Empty()) {
		std::swap(immediate, held);
	} else {
		held.ForEach([&immediate](uint64_t first, uint64_t last) { immediate.Add(first, last - first); });
	}
#endif
	held = Set {};
	return true;
}

} // namespace Libs::Graphics::BdaDefer

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDADEFER_H_ */
