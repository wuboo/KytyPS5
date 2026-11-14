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

// ==== S587 KYTY_GW_PAGE=0|1|<limit> (only with the guest read emulator on, e.g. KYTY_GUEST_WAIT=clean|1) ======================================
// S586 (KRF3 + KYTY_GUEST_WAIT=1): garden +13.2 / +11.4 ms cp_frame, desert hitches > 33 ms 3 -> 163. An emulated load is served in the
// exception handler and returns (runtimeLinker.cpp TryGuestReadEmulation): it never reaches BufferCache::ReadMemory, so
//  (1) the page is never downloaded and never released: it stays protected while K-rf v3 defers the walker's GPU-written dword on it, and
//      every later read faults again (garden: 375 emulated loads/flip, the CP's clean walks 1015 -> 8030/flip, pipe worker bails 238 -> 1426);
//  (2) NoteReadbackFault is never called, so the page never becomes an eager ("hot") page whose GPU writes are downloaded at the next
//      submit (rb_eager_downloads 24 -> 15/flip);
//  (3) RenderContext::HandleFault is never reached, so the K-rf v3 guest page guard sees no guest read (krf3_guest_page_n 0 for the whole
//      desert run d1k vs 0.1-0.4/flip in d1x) and never bans the desert page: 3 defers/flip, gpu_ts_busy 20 ms instead of 11.
// KYTY_GW_PAGE gives the emulator the three side effects of the ordinary path, per guest page and guest flip (PageFlipCounts):
//  - the first emulated load of a page in a flip (EmulateFirst) posts the page to the CP (EmuPages mailbox; BufferCache::DrainGuestEmuPages
//    in the pre-submit hook calls NoteReadbackFault, and a hot page with GPU-dirty bytes and no download in flight is queued for the eager
//    download at once), and counts one guard hit (GuestPageGuard::NoteGuestRead) - once per flip, what the ordinary path counts: its first
//    fault drains the page, so later reads of that flip do not fault;
//  - a load beyond `limit` faults on one page in one flip takes the ordinary path (drain + release of the page).
// Unset / 0 = bdae37c9 bit for bit. 1 = limit 32; <n> (2..65535) = limit n.
// Mutants (WILL_FAIL, tests/GuestWaitPageTests.cpp): KYTY_GWPAGE_MUTANT_NO_LIMIT (never the ordinary path), _NO_FEED (no post to the CP),
// _NO_GUARD_HIT (the guard is not told), _HIT_EVERY (every emulated load is a guard hit).
struct PageEnv {
	bool     on    = false;
	uint32_t limit = 32;
};
inline PageEnv ParsePage(const char* text) {
	PageEnv env;
	if (text == nullptr || text[0] == '\0' || std::strcmp(text, "0") == 0) return env;
	if (std::strcmp(text, "1") == 0) {
		env.on = true;
		return env;
	}
	char*      end   = nullptr;
	const auto value = std::strtoul(text, &end, 10);
	if (end == text || *end != '\0' || value < 2 || value > 65535) return env;
	env.on    = true;
	env.limit = static_cast<uint32_t>(value);
	return env;
}
inline const PageEnv& PageCare() {
	static const PageEnv env = ParsePage(std::getenv("KYTY_GW_PAGE"));
	return env;
}

enum class PageAction : uint8_t {
	Emulate,      // emulate as S585
	EmulateFirst, // emulate; the first load of the page in this flip: post the page to the CP and count a guard hit
	Ordinary,     // over the limit: the ordinary path (it downloads the page and releases it)
};
// count = loads of the page in this flip including this one (1-based); 0 = not tracked (the table is full): as S585.
constexpr PageAction DecidePage(uint32_t count, uint32_t limit) {
	if (count == 0) return PageAction::Emulate;
#if !defined(KYTY_GWPAGE_MUTANT_NO_LIMIT)
	if (count > limit) return PageAction::Ordinary;
#endif
	return count == 1 ? PageAction::EmulateFirst : PageAction::Emulate;
}
struct PageEffects {
	bool feed      = false; // post the page to the CP (hot page / eager download)
	bool guard_hit = false; // GuestPageGuard::NoteGuestRead
};
constexpr PageEffects EffectsOf(PageAction action) {
	PageEffects e;
	const bool  first = action == PageAction::EmulateFirst;
#if !defined(KYTY_GWPAGE_MUTANT_NO_FEED)
	e.feed = first;
#endif
#if defined(KYTY_GWPAGE_MUTANT_HIT_EVERY)
	e.guard_hit = action != PageAction::Ordinary;
#elif !defined(KYTY_GWPAGE_MUTANT_NO_GUARD_HIT)
	e.guard_hit = first;
#endif
	return e;
}

// Loads per 4 KiB guest page within one guest flip: open addressing over atomics, cleared when the flip number changes (as gf_emu_page_max).
// Any thread. Note() returns the count including this load, 0 when the table is full. A clear racing with a note can lose a few counts
// at the flip boundary: the decision then errs on the S585 side (emulate).
template <size_t N = 256>
class PageFlipCounts {
	static_assert((N & (N - 1)) == 0, "power of two");

public:
	uint32_t Note(uint64_t address, uint64_t flip) {
		const uint64_t key = (address >> 12) + 1; // 0 = free
		uint64_t       seen = m_flip.load(std::memory_order_acquire);
		if (seen != flip + 1) {
			if (m_flip.compare_exchange_strong(seen, flip + 1, std::memory_order_acq_rel)) {
				for (size_t i = 0; i < N; ++i) {
					m_counts[i].store(0, std::memory_order_relaxed);
					m_pages[i].store(0, std::memory_order_relaxed);
				}
			}
		}
		size_t slot = static_cast<size_t>((key * 0x9e3779b97f4a7c15ull) >> 56) & (N - 1);
		for (size_t probe = 0; probe < N; ++probe, slot = (slot + 1) & (N - 1)) {
			uint64_t known = m_pages[slot].load(std::memory_order_acquire);
			if (known == 0 && m_pages[slot].compare_exchange_strong(known, key, std::memory_order_acq_rel)) {
				known = key;
			}
			if (known == key) {
				return m_counts[slot].fetch_add(1, std::memory_order_relaxed) + 1u;
			}
		}
		return 0;
	}

private:
	std::atomic<uint64_t> m_flip {0}; // flip + 1 (0 = never)
	std::atomic<uint64_t> m_pages[N] {};
	std::atomic<uint32_t> m_counts[N] {};
};
inline PageFlipCounts<>& EmuPageCounts() {
	static PageFlipCounts<> counts;
	return counts;
}

// Guest threads post pages (lock-free, bounded), the CP thread drains them. Push returns false when every slot is taken (the post is
// dropped: gw_page_feed_drop_n; the page still gets the limit's ordinary path).
template <size_t N = 64>
class PageMailbox {
public:
	bool Push(uint64_t page) {
		if (page == 0) return false;
		size_t start = static_cast<size_t>((page >> 12) * 0x9e3779b97f4a7c15ull >> 58) % N;
		for (size_t i = 0; i < N; ++i) {
			auto&    slot  = m_slots[(start + i) % N];
			uint64_t empty = 0;
			if (slot.load(std::memory_order_relaxed) == page) return true; // already posted
			if (slot.compare_exchange_strong(empty, page, std::memory_order_acq_rel)) return true;
		}
		return false;
	}
	template <typename F>
	size_t Drain(F&& f) {
		size_t n = 0;
		for (auto& slot: m_slots) {
			if (slot.load(std::memory_order_relaxed) == 0) continue;
			const uint64_t page = slot.exchange(0, std::memory_order_acq_rel);
			if (page != 0) {
				f(page);
				++n;
			}
		}
		return n;
	}

private:
	std::atomic<uint64_t> m_slots[N] {};
};
inline PageMailbox<>& EmuPages() {
	static PageMailbox<> mailbox;
	return mailbox;
}

} // namespace Libs::Graphics::GuestWait

#endif
