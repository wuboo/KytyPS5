// S585 (KYTY_GUEST_WAIT, guestWait.h): a guest read fault waits only for the bytes it reads.
//  - DecideBytes on the S583 classes: GPU-dirty bytes keep today's path; clean bytes are served without any wait even when the page has an
//    eager download in flight (false sharing, 0x55c008x18); bytes in flight wait for the tick of their own copy, not for a newer download of
//    the page (flip 6301), and a copy that is not done is waited for (never served); a copy not submitted yet keeps today's path.
//  - a model of one guest float next to a GPU-written neighbour on one page (writes, eager flushes at submit, GPU progress, publishes, guest
//    reads): every value a guest read gets equals the value of the newest recorded writer (what the drain would give), clean reads never wait,
//    every wait is for the bytes' own copy;
//  - guest threads against the CP: the bounded spin of the state queries never answers "clean" for bytes in the dirty -> in-flight hand-over
//    or before their publish (no write is lost), and a lock held for a short while is waited out;
//  - the copy ticks of InflightRanges, the env parsers (KYTY_GUEST_WAIT, KYTY_DIAG_GUEST_STALE, KYTY_PRESENT_MODE), the YMM init-state rule.
// Mutants (WILL_FAIL): KYTY_GUESTWAIT_MUTANT_{INFLIGHT_SERVES, CLEAN_WAITS_PAGE, UNDONE_SERVES, WAITS_NEWEST, BUSY_AS_CLEAN},
// KYTY_GUESTREAD_MUTANT_VEX_MASK_GUESS.

#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"
#include "graphics/host_gpu/renderer/cache/guestWait.h"
#include "graphics/host_gpu/renderer/cache/inflightRanges.h"
#include "graphics/presentation/window/presentModeEnv.h"
#include "loader/guestReadEmulator.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <random>
#include <thread>
#include <vector>

using namespace Libs::Graphics;
using GuestWait::ByteAction;
using GuestWait::ByteState;
using GuestWait::DecideBytes;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

static ByteState State(bool dirty, bool inflight, uint64_t copy_tick, uint64_t page_tick, uint64_t current, uint64_t known) {
	ByteState s;
	s.dirty          = dirty;
	s.inflight       = inflight;
	s.copy_tick      = copy_tick;
	s.page_tick      = page_tick;
	s.current_tick   = current;
	s.known_gpu_tick = known;
	return s;
}

static void TestDecision() {
	// GPU-dirty bytes (S583 class 1: the writer in the open batch): today's path.
	Check(DecideBytes(State(true, false, 0, 0, 10, 9)).action == ByteAction::Ordinary, "dirty: ordinary");
	Check(DecideBytes(State(true, true, 5, 7, 10, 3)).action == ByteAction::Ordinary, "dirty and in flight: ordinary");
	// Clean bytes on a page with an eager download of other bytes (false sharing): served, no wait.
	{
		const auto d = DecideBytes(State(false, false, 0, 9, 10, 4));
		Check(d.action == ByteAction::Serve && d.tick == 0, "clean bytes, page tick in flight: served without a wait");
	}
	Check(DecideBytes(State(false, false, 0, 0, 10, 4)).action == ByteAction::Serve, "clean bytes, no eager page: served");
	// Bytes in flight whose copy's tick is not done: the wait is for that tick (never a serve of the stale backing).
	{
		const auto d = DecideBytes(State(false, true, 5, 5, 8, 3));
		Check(d.action == ByteAction::Wait && d.tick == 5, "in flight, copy not done: waits for the copy's tick");
	}
	// S583 class 2 (flip 6301): the copy of these bytes is older than the page's eager tick: wait for the copy only.
	{
		const auto d = DecideBytes(State(false, true, 5, 7, 8, 6));
		Check(d.action == ByteAction::Wait && d.tick == 5, "in flight, newer page tick: the wait is for the bytes' own copy");
	}
	{
		const auto d = DecideBytes(State(false, true, 5, 9, 10, 2));
		Check(d.action == ByteAction::Wait && d.tick == 5, "in flight, not done, newer page tick: still the own copy");
	}
	// GPU done, publish pending: a wait (only the priority publish is left).
	Check(DecideBytes(State(false, true, 5, 5, 8, 6)).action == ByteAction::Wait, "in flight, GPU done: waits for the publish");
	// A copy not submitted yet (tick == CurrentTick) or with an unknown tick: today's path.
	Check(DecideBytes(State(false, true, 8, 8, 8, 6)).action == ByteAction::Ordinary, "copy not submitted: ordinary");
	Check(DecideBytes(State(false, true, 0, 0, 8, 6)).action == ByteAction::Ordinary, "copy tick unknown: ordinary");
}

// One guest float B and a GPU-written neighbour N on one tracker page.
struct PageModel {
	struct Copy {
		uint64_t tick;
		uint32_t value;
		bool     ours; // a copy of B (else of N)
	};
	uint64_t         current   = 1; // CurrentTick: the recording submission
	uint64_t         known     = 0; // KnownGpuTick
	uint32_t         gpu_value = 0; // B after every recorded write (what the drain gives)
	uint32_t         backing   = 0;
	bool             dirty     = false;
	bool             n_dirty   = false;
	uint64_t         page_tick = 0; // m_eager_pages[page].tick
	uint32_t         next      = 1;
	std::deque<Copy> copies;

	void WriteB() {
		gpu_value = next++;
		dirty     = true;
	}
	void WriteN() { n_dirty = true; }
	// The eager flush in the hook before a submit, then the submit.
	void FlushSubmit() {
		bool any = false;
		if (dirty) {
			copies.push_back({current, gpu_value, true});
			dirty = false;
			any   = true;
		}
		if (n_dirty) {
			copies.push_back({current, 0, false});
			n_dirty = false;
			any     = true;
		}
		if (any) page_tick = current;
		current++;
	}
	void Submit() { current++; }
	// GPU progress up to `tick` (<= current - 1) and the publishes of the copies done by then (in tick order, as priority operations).
	void RunTo(uint64_t tick) {
		if (tick > current - 1) tick = current - 1;
		if (tick > known) known = tick;
		while (!copies.empty() && copies.front().tick <= known) {
			const auto copy = copies.front();
			copies.pop_front();
			if (copy.ours) backing = copy.value;
			if (page_tick == copy.tick) {
				bool more = false;
				for (const auto& c: copies) more |= c.tick == copy.tick;
				if (!more) page_tick = 0;
			}
		}
	}
	[[nodiscard]] bool Inflight() const {
		for (const auto& c: copies) {
			if (c.ours) return true;
		}
		return false;
	}
	[[nodiscard]] uint64_t CopyTick() const {
		uint64_t t = 0;
		for (const auto& c: copies) {
			if (c.ours && c.tick > t) t = c.tick;
		}
		return t;
	}
	// Today's ordinary path for B: the drain (submit what is recorded, wait for all of it).
	uint32_t Drain() {
		if (dirty) FlushSubmit();
		RunTo(current - 1);
		return backing;
	}
};

static void TestModel() {
	std::mt19937 rng(585);
	uint64_t     reads = 0, served = 0, served_after_wait = 0, waits = 0, waits_blocking = 0, ordinary = 0, clean_waits = 0,
	         wrong = 0, not_own = 0, older = 0, refused = 0;
	for (int run = 0; run < 400; run++) {
		PageModel m;
		for (int step = 0; step < 400; step++) {
			const auto r = rng() % 100;
			if (r < 12) {
				m.WriteB();
			} else if (r < 40) {
				m.WriteN();
			} else if (r < 55) {
				m.FlushSubmit();
			} else if (r < 60) {
				m.Submit();
			} else if (r < 75) {
				m.RunTo(m.known + 1 + rng() % 2);
			} else {
				reads++;
				const auto state = State(m.dirty, m.Inflight(), m.CopyTick(), m.page_tick, m.current, m.known);
				const auto d     = DecideBytes(state);
				uint32_t   value = 0;
				if (d.action == ByteAction::Serve) {
					served++;
					value = m.backing;
				} else if (d.action == ByteAction::Wait) {
					waits++;
					if (!state.dirty && !state.inflight) clean_waits++;
					if (state.inflight && d.tick != state.copy_tick) not_own++;
					if (state.page_tick > d.tick) older++;
					if (d.tick > m.known) waits_blocking++;
					m.RunTo(d.tick); // Wait(tick) + WaitPriorityOperations(tick)
					if (rng() % 4 == 0) m.WriteB(); // the CP records a new writer of the bytes during the wait
					// Query again: served only when the bytes are clean now (the CP may have recorded a new writer meanwhile).
					if (!m.dirty && !m.Inflight()) {
						served_after_wait++;
						value = m.backing;
					} else {
						refused++;
						ordinary++;
						value = m.Drain();
					}
				} else {
					ordinary++;
					value = m.Drain();
				}
				if (value != m.gpu_value) wrong++;
			}
		}
	}
	std::printf("model: reads=%llu served=%llu waits=%llu (blocking %llu, older than the page tick %llu) served_after_wait=%llu refused=%llu ordinary=%llu\n",
	            static_cast<unsigned long long>(reads), static_cast<unsigned long long>(served), static_cast<unsigned long long>(waits),
	            static_cast<unsigned long long>(waits_blocking), static_cast<unsigned long long>(older),
	            static_cast<unsigned long long>(served_after_wait), static_cast<unsigned long long>(refused),
	            static_cast<unsigned long long>(ordinary));
	Check(reads > 10000 && served > 1000 && waits > 1000 && waits_blocking > 100 && older > 50 && refused > 100, "model: every class occurs");
	Check(wrong == 0, "model: every guest read gets the newest recorded writer's value (bytes in flight are never served stale)");
	Check(clean_waits == 0, "model: clean bytes never wait (another download of the page is not theirs)");
	Check(not_own == 0, "model: a wait is for the bytes' own copy, not the page's newest eager tick");
}

// Guest threads against the CP: the hand-over and the publish under a lock held for a while; a spin that gives up must not answer "clean".
static void TestThreadsNoLostWrite() {
	GpuDirtyRanges        d;
	InflightRanges        inflight;
	constexpr uint64_t    R = 0x555f41dd0, N = 4;
	std::atomic<uint64_t> gpu {0}, backing {0};
	std::atomic<bool>     stop {false};
	std::atomic<uint64_t> lost {0}, clean {0}, busy {0}, dirty_seen {0};
	auto                  reader = [&] {
        while (!stop.load(std::memory_order_acquire)) {
            const auto g     = gpu.load(std::memory_order_acquire);
            bool       dirty = false;
            if (!d.TryIntersectsSpin(R, N, &dirty, 64)) {
                busy++;
                continue;
            }
            if (dirty) {
                dirty_seen++;
                continue;
            }
            bool     in   = false;
            uint64_t tick = 0;
            if (!inflight.TryIntersectsSpin(R, N, &in, &tick, 64)) {
                busy++;
                continue;
            }
            if (in) continue;
            const auto b = backing.load(std::memory_order_acquire);
            clean++;
            if (b < g) lost++;
        }
	};
	std::vector<std::thread> readers;
	for (int i = 0; i < 3; i++) readers.emplace_back(reader);
	for (uint64_t k = 1; k <= 3000; k++) {
		d.Add(R, N); // the writer is recorded (dirty) before the GPU can write the bytes
		gpu.store(k, std::memory_order_release);
		d.BeginDownloadCopy(
		    inflight, R, N, [](uint64_t, uint64_t) {},
		    [&] {
			    if (k % 8 == 0) { // the CP holds both locks for a while in the hand-over window (dirty and in flight)
				    std::lock_guard lock(inflight.MutexForTest());
				    std::this_thread::sleep_for(std::chrono::microseconds(200));
			    }
		    },
		    k);
		if (k % 16 == 0) {
			std::lock_guard lock(inflight.MutexForTest()); // the publish thread is busy with another copy
			std::this_thread::sleep_for(std::chrono::microseconds(200));
		}
		backing.store(k, std::memory_order_release); // the deferred publish: the backing, then the in-flight entry goes
		inflight.Remove(R, N);
	}
	stop.store(true, std::memory_order_release);
	for (auto& t: readers) t.join();
	std::printf("threads: clean=%llu dirty=%llu busy=%llu lost=%llu\n", static_cast<unsigned long long>(clean.load()),
	            static_cast<unsigned long long>(dirty_seen.load()), static_cast<unsigned long long>(busy.load()),
	            static_cast<unsigned long long>(lost.load()));
	Check(clean.load() > 0, "threads: the readers saw clean bytes (the test exercises the published state)");
	Check(lost.load() == 0, "threads: a clean answer never precedes the publish of a recorded write (busy is never clean)");
}

static void TestSpinWaitsOut() {
	// A lock held for 5 ms is waited out by a spin long enough.
	{
		GpuDirtyRanges d;
		d.Add(0x5000, 0x10);
		std::atomic<bool> held {false};
		std::thread       holder([&] {
            std::lock_guard lock(d.MutexForTest());
            held = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        });
		while (!held) std::this_thread::yield();
		GuestWait::ThreadSpin() = {};
		bool       dirty    = false;
		const bool answered = d.TryIntersectsSpin(0x5000, 0x10, &dirty, 1 << 24);
		holder.join();
		Check(answered && dirty, "spin: a short-held lock is waited out, the answer is the real one");
		Check(GuestWait::ThreadSpin().spun == 1 && GuestWait::ThreadSpin().busy == 0, "spin: counted as spun, not busy");
	}
	// The default spin gives up on a lock held for 300 ms, long before it is released: busy, not an answer.
	{
		GpuDirtyRanges d;
		d.Add(0x5000, 0x10);
		std::atomic<bool> held {false};
		std::thread       holder([&] {
            std::lock_guard lock(d.MutexForTest());
            held = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        });
		while (!held) std::this_thread::yield();
		GuestWait::ThreadSpin() = {};
		bool       dirty    = true;
		const auto begin    = std::chrono::steady_clock::now();
		const bool answered = d.TryIntersectsSpin(0x5000, 0x10, &dirty);
		const auto waited   = std::chrono::steady_clock::now() - begin;
		holder.join();
		Check(!answered, "spin: gives up on a long-held lock (busy)");
		Check(waited < std::chrono::milliseconds(100), "spin: bounded");
		Check(GuestWait::ThreadSpin().busy == 1, "spin: counted as busy");
	}
}

static void TestInflightTicks() {
	InflightRanges r;
	r.Add(0x1000, 0x100, 5);
	r.Add(0x1080, 0x100, 7);
	bool     in   = false;
	uint64_t tick = 0;
	Check(r.TryIntersectsSpin(0x1000, 4, &in, &tick) && in && tick == 5, "ticks: the copy of the bytes");
	Check(r.TryIntersectsSpin(0x1090, 4, &in, &tick) && in && tick == 7, "ticks: the newest overlapping copy");
	Check(r.TryIntersectsSpin(0x2000, 4, &in, &tick) && !in && tick == 0, "ticks: nothing in flight");
	// Two copies of one range: the publish of the older one leaves the newer tick.
	r.Add(0x3000, 0x10, 3);
	r.Add(0x3000, 0x10, 9);
	r.Remove(0x3000, 0x10);
	Check(r.TryIntersectsSpin(0x3000, 4, &in, &tick) && in && tick == 9, "ticks: the older copy publishes first");
	r.Add(0x3000, 0x10); // unknown tick (0): the whole answer is unknown
	Check(r.TryIntersectsSpin(0x3000, 4, &in, &tick) && in && tick == 0, "ticks: an unknown copy tick makes the answer unknown");
	// BeginDownloadCopy carries the tick.
	GpuDirtyRanges d;
	InflightRanges inflight;
	d.Add(0x8000, 0x40);
	d.BeginDownloadCopyAt(inflight, 11, 0x8000, 0x40, [](uint64_t, uint64_t) {});
	Check(inflight.TryIntersectsSpin(0x8010, 4, &in, &tick) && in && tick == 11, "ticks: BeginDownloadCopyAt records the submission");
}

static void TestParsers() {
	Check(GuestWait::Parse(nullptr) == GuestWait::Mode::Off && GuestWait::Parse("0") == GuestWait::Mode::Off &&
	          GuestWait::Parse("x") == GuestWait::Mode::Off,
	      "KYTY_GUEST_WAIT: unset / 0 / junk = off");
	Check(GuestWait::Parse("clean") == GuestWait::Mode::Clean && GuestWait::Parse("1") == GuestWait::Mode::Bytes, "KYTY_GUEST_WAIT: clean, 1");
	Check(GuestWait::ResolveEmu(0, GuestWait::Mode::Off) == 0, "emulator: off stays off");
	Check(GuestWait::ResolveEmu(0, GuestWait::Mode::Clean) == 1 && GuestWait::ResolveEmu(0, GuestWait::Mode::Bytes) == 1, "emulator: on");
	Check(GuestWait::ResolveEmu(2, GuestWait::Mode::Bytes) == 2, "emulator: a shadow from KYTY_LAND_SYNC_GUEST / KYTY_GF_EMULATE wins");
	{
		const auto e = GuestWait::ParseStale("1");
		Check(e.on && e.filter == 0, "stale: 1");
		const auto f = GuestWait::ParseStale("0x555f41dd0");
		Check(f.on && f.filter == 0x555f41dd0, "stale: address filter");
		Check(!GuestWait::ParseStale("0").on && !GuestWait::ParseStale(nullptr).on && !GuestWait::ParseStale("0x").on &&
		          !GuestWait::ParseStale("yes").on,
		      "stale: off");
		Check(GuestWait::StaleLineWanted(f, 0x555f41dd0, 4) && !GuestWait::StaleLineWanted(f, 0x55c008018, 8), "stale: the filter");
		Check(GuestWait::StaleShouldLog(299, 0, false) && !GuestWait::StaleShouldLog(300, 0, false) && GuestWait::StaleShouldLog(300, 2999, true) &&
		          !GuestWait::StaleShouldLog(0, 3000, true),
		      "stale: line limits");
		uint64_t a = 0, n = 0;
		GuestWait::StaleRange({0x555f41dd0, 4, true}, 0x555f41dd2, &a, &n);
		Check(a == 0x555f41dd0 && n == 4, "stale: the decoded access");
		GuestWait::StaleRange({0x1000, 4, true}, 0x555f41dd3, &a, &n);
		Check(a == 0x555f41dd0 && n == 4, "stale: a hint of another access is ignored (the aligned dword)");
		GuestWait::StaleRange({}, 0x55c00801b, &a, &n);
		Check(a == 0x55c008018 && n == 4, "stale: no hint, the aligned dword");
	}
	{
		enum class Mode { Fifo, Mailbox, Immediate };
		using PresentModeEnv::Override;
		Check(PresentModeEnv::Parse(nullptr) == Override::None && PresentModeEnv::Parse("") == Override::None &&
		          PresentModeEnv::Parse("vsync") == Override::None,
		      "present: unset / junk = the flag");
		Check(PresentModeEnv::Parse("fifo") == Override::Fifo && PresentModeEnv::Parse("Mailbox") == Override::Mailbox &&
		          PresentModeEnv::Parse("immediate") == Override::Immediate,
		      "present: values");
		Check(PresentModeEnv::Resolve(Mode::Immediate, Override::None) == Mode::Immediate, "present: no env keeps --present-mode");
		Check(PresentModeEnv::Resolve(Mode::Immediate, Override::Fifo) == Mode::Fifo, "present: the env wins");
	}
}

static void TestYmmh() {
	using Loader::GuestReadEmulator::ClassifyYmmh;
	using Loader::GuestReadEmulator::YmmhState;
	// (xstate, mask_ok, avx_bit, located, allow, hw_ok)
	Check(ClassifyYmmh(true, true, true, true, false, false) == YmmhState::Present, "ymm: AVX bit set, area present");
	Check(ClassifyYmmh(true, true, false, true, true, false) == YmmhState::InitZero, "ymm: AVX bit clear in a context with the area: init state");
	Check(ClassifyYmmh(true, true, false, true, false, false) == YmmhState::Unknown, "ymm: init state not allowed (KYTY_GUEST_WAIT off)");
	Check(ClassifyYmmh(true, true, false, false, true, true) == YmmhState::Unknown, "ymm: no AVX area in an extended context: unknown");
	Check(ClassifyYmmh(true, false, false, true, true, true) == YmmhState::Unknown, "ymm: the mask cannot be read: unknown (a guess)");
	Check(ClassifyYmmh(false, false, false, false, true, true) == YmmhState::Hardware, "ymm: no extended area: the hardware register");
	Check(ClassifyYmmh(false, false, false, false, true, false) == YmmhState::Unknown, "ymm: no extended area, no hardware stub: unknown");
	Check(ClassifyYmmh(false, false, false, false, false, true) == YmmhState::Unknown, "ymm: no extended area, not allowed: unknown");
}

int main() {
	TestDecision();
	TestModel();
	TestThreadsNoLostWrite();
	TestSpinWaitsOut();
	TestInflightTicks();
	TestParsers();
	TestYmmh();
	if (g_failures != 0) {
		std::fprintf(stderr, "GuestWaitTests: %d failures\n", g_failures);
		return 1;
	}
	std::printf("GuestWaitTests: all passed\n");
	return 0;
}
