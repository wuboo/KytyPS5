#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GUESTWAIT_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GUESTWAIT_H_

// S585 (KYTY_GUEST_WAIT=0|clean|1): a guest read fault on a protected page waits only for the bytes it reads.
// S582/S583: with K-rf v3 the desert's guest threads wait 8-13 ms/flip on pages whose own bytes are clean (false sharing with a GPU buffer,
// 0x55c008x18, 0x50740a590) or on a newer eager tick of the page while the copy of their own bytes has already landed (flip 6301).
//  - clean: the guest read emulator (guestReadEmulator.h) is on (unless KYTY_GF_EMULATE / KYTY_LAND_SYNC_GUEST already choose a mode, e.g.
//    shadow), and two of its S495 fallbacks are removed: the buffer cache state is queried with a bounded spin on try_lock instead of one
//    try (LockBusy, 180-272 per landing), and on Windows a VEX.128 load whose saved context has the AVX component in its init state (the upper
//    YMM halves are all zero, XSTATE_BV bit clear) is served (VexNoAvx, 105-228); the zeroing of bits 128..255 is then already true.
//    Clean bytes (not GPU-dirty, no copy in flight) are served from the backing without any Wait(tick) (MayEmulate).
//  - 1: clean + the byte wait (DecideBytes): bytes that are not GPU-dirty but whose GPU->host copy is in flight wait for the tick of THAT
//    copy (its GPU work and its publish), then the load is emulated; the page's eager tick (a newer download of other bytes) is not waited
//    for, and no SendCommandSync round trip through the CP. GPU-dirty bytes, copies not submitted yet and every other case keep the
//    ordinary drain exactly as today.
// Unset / 0 = 1606ec74 bit for bit.
//
// KYTY_DIAG_GUEST_STALE=1|0x<address> (S583 U5, log only, never serves the old value): a guest read fault on GPU-dirty or in-flight bytes logs
// the backing before the ordinary drain (stale) and after it (fresh); diag_gs_n / diag_gs_ne_n / diag_gs_inflight_n / diag_gs_unsettled_n.
// 0x<address>: lines only for accesses containing that address (the counters count every address). <= 300 lines outside
// KYTY_DIAG_GF_WINDOW, <= 3000 inside it.
//
// Pure part; tests/GuestWaitTests.cpp pins it. Mutants (WILL_FAIL): KYTY_GUESTWAIT_MUTANT_INFLIGHT_SERVES (bytes in flight served at once),
// _CLEAN_WAITS_PAGE (clean bytes wait for the page tick), _UNDONE_SERVES (a copy whose tick is not done is served), _WAITS_NEWEST (the wait
// is for the page's newest eager tick, as today), _BUSY_AS_CLEAN (an exhausted spin answers "clean": a write in the hand-over is lost).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Libs::Graphics::GuestWait {

enum class Mode : uint8_t { Off, Clean, Bytes };

inline Mode Parse(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::Bytes;
	if (std::strcmp(text, "clean") == 0) return Mode::Clean;
	return Mode::Off;
}
inline Mode EnvMode() {
	static const Mode mode = Parse(std::getenv("KYTY_GUEST_WAIT"));
	return mode;
}
constexpr bool CleanOn(Mode mode) { return mode != Mode::Off; }
constexpr bool BytesOn(Mode mode) { return mode == Mode::Bytes; }
constexpr const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::Clean: return "clean";
		case Mode::Bytes: return "1";
		default: return "0";
	}
}

// GuestReadEmulator::Mode as a number (Off=0, On=1, Shadow=2): the mode KYTY_GF_EMULATE / KYTY_LAND_SYNC_GUEST resolved to wins when it is not
// Off; otherwise KYTY_GUEST_WAIT turns the emulator on.
constexpr uint8_t ResolveEmu(uint8_t resolved, Mode mode) {
	if (resolved != 0) return resolved;
	return CleanOn(mode) ? 1 : 0;
}

// --- bounded spin on try_lock (a signal handler never blocks on the CP's lock, but a lock held for microseconds is waited out) ---
constexpr int kSpinTries = 2048; // ~50-200 us: pause between tries, a yield every 64

inline void CpuRelax() {
#if defined(__x86_64__) || defined(_M_X64)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

// True when the lock was taken (the caller unlocks). *spins = failed tries before it (0 = free at once).
template <typename Mutex>
bool SpinTryLock(Mutex& mutex, int tries, int* spins) {
	for (int i = 0; i < tries; ++i) {
		if (mutex.try_lock()) {
			if (spins != nullptr) *spins = i;
			return true;
		}
		if ((i & 63) == 63) {
			std::this_thread::yield();
		} else {
			CpuRelax();
		}
	}
	if (spins != nullptr) *spins = tries;
	return false;
}

// Per-thread statistics of the spin queries of one fault (read by the caller for its counters).
struct SpinStats {
	uint32_t spun = 0; // queries that did not get the lock at the first try
	uint32_t busy = 0; // queries that gave up
};
inline SpinStats& ThreadSpin() {
	static thread_local SpinStats stats;
	return stats;
}

// --- (b) the byte wait ---
struct ByteState {
	bool     dirty         = false; // the bytes are GPU-dirty (a recorded writer not handed to a copy)
	bool     inflight      = false; // a GPU->host copy of the bytes is recorded and not published
	uint64_t copy_tick     = 0;     // newest tick of the in-flight copies of the bytes (0 = unknown)
	uint64_t page_tick     = 0;     // the page's eager download tick (m_eager_pages), 0 = none: what today's path waits for
	uint64_t current_tick  = 0;     // CommandScheduler::CurrentTick (ticks below it are submitted)
	uint64_t known_gpu_tick = 0;    // MasterSemaphore::KnownGpuTick (done)
};

enum class ByteAction : uint8_t {
	Serve,    // the backing holds the bytes: emulate now, no wait
	Wait,     // wait for `tick` (GPU + its priority publish), then query again and emulate if the bytes are clean
	Ordinary, // today's path (drain / eager wait / SendCommandSync)
};
struct ByteDecision {
	ByteAction action = ByteAction::Ordinary;
	uint64_t   tick   = 0;
};

constexpr ByteDecision DecideBytes(const ByteState& s) {
	if (s.dirty) {
		return {ByteAction::Ordinary, 0}; // GPU-dirty bytes wait as today (their writer may not even be submitted)
	}
	if (!s.inflight) {
#if defined(KYTY_GUESTWAIT_MUTANT_CLEAN_WAITS_PAGE)
		if (s.page_tick != 0) return {ByteAction::Wait, s.page_tick};
#endif
		return {ByteAction::Serve, 0}; // clean: another download of the page is not ours
	}
#if defined(KYTY_GUESTWAIT_MUTANT_INFLIGHT_SERVES)
	return {ByteAction::Serve, 0};
#endif
	if (s.copy_tick == 0 || s.copy_tick >= s.current_tick) {
		return {ByteAction::Ordinary, 0}; // the copy is not submitted yet (or its tick is unknown): today's path
	}
#if defined(KYTY_GUESTWAIT_MUTANT_UNDONE_SERVES)
	if (s.copy_tick > s.known_gpu_tick) return {ByteAction::Serve, 0};
#endif
#if defined(KYTY_GUESTWAIT_MUTANT_WAITS_NEWEST)
	if (s.page_tick > s.copy_tick) return {ByteAction::Wait, s.page_tick};
#endif
	return {ByteAction::Wait, s.copy_tick}; // this copy, not a newer download of the page (S583 class 2); not done = the wait blocks
}

// --- runtimeLinker -> RenderContext::HandleFault: the access the faulting instruction makes (the handler only knows the fault byte) ---
struct AccessHint {
	uint64_t address = 0;
	uint8_t  size    = 0;
	bool     valid   = false;
};
inline AccessHint& ThreadHint() {
	static thread_local AccessHint hint;
	return hint;
}

// --- KYTY_DIAG_GUEST_STALE ---
struct StaleEnv {
	bool     on     = false;
	uint64_t filter = 0; // 0 = every address
};
inline StaleEnv ParseStale(const char* text) {
	StaleEnv env;
	if (text == nullptr || text[0] == '\0' || std::strcmp(text, "0") == 0) return env;
	if (text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
		char*      end   = nullptr;
		const auto value = std::strtoull(text + 2, &end, 16);
		if (end == text + 2 || *end != '\0' || value == 0) return env;
		env.on     = true;
		env.filter = value;
		return env;
	}
	env.on = std::strcmp(text, "1") == 0;
	return env;
}
inline const StaleEnv& Stale() {
	static const StaleEnv env = ParseStale(std::getenv("KYTY_DIAG_GUEST_STALE"));
	return env;
}
// The bytes the DIAG compares: the decoded access when the hint contains the fault byte, else the aligned dword of it (clipped to its page).
inline void StaleRange(const AccessHint& hint, uint64_t fault, uint64_t* address, uint64_t* size) {
	if (hint.valid && hint.size != 0 && hint.size <= 16 && fault >= hint.address && fault < hint.address + hint.size) {
		*address = hint.address;
		*size    = hint.size;
		return;
	}
	*address = fault & ~uint64_t {3};
	*size    = 4;
}
constexpr bool StaleLineWanted(const StaleEnv& env, uint64_t address, uint64_t size) {
	return env.filter == 0 || (env.filter >= address && env.filter < address + size);
}
constexpr bool StaleShouldLog(uint64_t outside, uint64_t inside, bool in_window) {
	return in_window ? inside < 3000 : outside < 300;
}

} // namespace Libs::Graphics::GuestWait

#endif
