// S605 (KYTY_UPLOAD_ASYNC): the deferred upload copies (uploadAsync.h Engine). Every check ends in exit(1) when it fails (no crash), so each mutant
// (CMake upload_async_mutant_*, WILL_FAIL) is caught by a failed check:
//   NO_WRITE_JOIN  - a write does not wait for the copies under it: the oracle sees bytes written after the snapshot;
//   JOIN_NO_WAIT   - a join returns while a copy it needs is running: the slow copy is not done after JoinAll;
//   OVERLAP_END    - the overlap test misses a write on the last byte of a copy: the boundary write lands under a pending copy;
//   SLOT_REUSE     - a ring slot is refilled without checking that its copy has run: copies queued behind a slow one are lost;
//   LABEL_NO_JOIN  - RELEASE_MEM counts as a packet that writes no guest memory;
//   SHADOW_BLIND   - the shadow compare never reports a difference;
//   FAIL_NO_ZERO   - a read the backing no longer serves leaves the destination as it was;
//   GUEST_HELPS    - a thread other than the producer runs copies in its join (a guest thread the game can suspend would then hold a copy).

#include "graphics/host_gpu/renderer/cache/uploadAsync.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::UploadAsync;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "UploadAsyncTests: failed: %s\n", text);
		std::exit(1);
	}
}

// The guest memory of the tests: "addresses" are offsets into g_mem. g_slow: a read that starts inside [g_slow_begin, g_slow_end) sleeps first,
// so a copy is still running when the test looks (the mutants that only show under a race become deterministic).
constexpr uint64_t           kMem = 1u << 20u;
std::vector<uint8_t>         g_mem(kMem);
std::atomic<uint64_t>        g_slow_begin {0}, g_slow_end {0};
std::atomic<int>             g_slow_ms {0};
std::atomic<uint64_t>        g_fail_begin {0}, g_fail_end {0};
std::atomic<uint64_t>        g_reads {0};

bool TestRead(uint64_t src, void* dst, uint64_t size) {
	g_reads.fetch_add(1, std::memory_order_relaxed);
	if (src >= g_fail_begin.load() && src < g_fail_end.load()) return false;
	if (src >= g_slow_begin.load() && src < g_slow_end.load() && g_slow_ms.load() > 0) {
		std::this_thread::sleep_for(std::chrono::milliseconds(g_slow_ms.load()));
	}
	if (src + size > kMem) return false;
	// relaxed byte loads: the oracle's writer thread may store concurrently into bytes no pending copy reads
	auto* out = static_cast<uint8_t*>(dst);
	for (uint64_t i = 0; i < size; i++) out[i] = std::atomic_ref<uint8_t>(g_mem[src + i]).load(std::memory_order_relaxed);
	return true;
}

void ResetSlow() {
	g_slow_ms = 0;
	g_slow_begin = 0;
	g_slow_end = 0;
	g_fail_begin = 0;
	g_fail_end = 0;
}

void Fill(uint64_t at, uint64_t size, uint8_t seed) {
	for (uint64_t i = 0; i < size; i++) std::atomic_ref<uint8_t>(g_mem[at + i]).store(static_cast<uint8_t>(seed + i * 7u), std::memory_order_relaxed);
}

void ModeTests() {
	Check(ParseMode(nullptr) == Mode::Off, "unset = off");
	Check(ParseMode("0") == Mode::Off && ParseMode("") == Mode::Off && ParseMode("yes") == Mode::Off, "anything else = off");
	Check(ParseMode("1") == Mode::On, "1 = on");
	Check(ParseMode("shadow") == Mode::Shadow, "shadow");
	Check(std::strcmp(ModeName(Mode::Shadow), "shadow") == 0 && std::strcmp(ModeName(Mode::On), "1") == 0 && std::strcmp(ModeName(Mode::Off), "0") == 0, "names");
}

void HandlerTests() {
	using namespace Libs::Graphics::Pm4;
	for (const uint32_t op: {IT_WRITE_DATA, IT_RELEASE_MEM, IT_EVENT_WRITE, IT_DMA_DATA, IT_COPY_DATA, IT_DRAW_INDIRECT, IT_DRAW_INDEX_INDIRECT,
	                         IT_MEM_SEMAPHORE, IT_COND_EXEC, IT_SET_PREDICATION, 0xffu}) {
		Check(HandlerMayWriteGuest(op, 0), "a signal, a CPU write, an indirect draw or an unknown packet joins");
	}
	Check(HandlerMayWriteGuest(IT_NOP, R_FLIP), "a flip joins");
	Check(HandlerMayWriteGuest(IT_NOP, R_NUM - 1u), "an unknown NOP joins");
	for (const uint32_t op: {IT_SET_CONTEXT_REG, IT_SET_SH_REG, IT_SET_UCONFIG_REG, IT_SET_CONTEXT_REG_INDIRECT, IT_INDEX_TYPE, IT_NUM_INSTANCES,
	                         IT_DRAW_INDEX_2, IT_DRAW_INDEX_AUTO, IT_DRAW_INDEX_OFFSET_2, IT_DISPATCH_DIRECT, IT_DISPATCH_INDIRECT, IT_WAIT_REG_MEM,
	                         IT_WAIT_REG_MEM_64, IT_ACQUIRE_MEM, IT_INDIRECT_BUFFER}) {
		Check(!HandlerMayWriteGuest(op, 0), "register writes, draws, dispatches, waits run without a join");
	}
	Check(!HandlerMayWriteGuest(IT_NOP, R_ZERO) && !HandlerMayWriteGuest(IT_NOP, R_PUSH_MARKER), "markers run without a join");
}

void OverlapTests() {
	Check(Engine::Overlaps(100, 10, 109, 1), "the last byte overlaps");
	Check(!Engine::Overlaps(100, 10, 110, 1), "one past the end does not");
	Check(Engine::Overlaps(100, 10, 99, 2), "the first byte overlaps");
	Check(!Engine::Overlaps(100, 10, 90, 10), "ends at the start: no overlap");
	Check(Engine::Overlaps(100, 10, 0, 1000), "covering range");
}

void BasicTests() {
	ResetSlow();
	Fill(0, 300000, 3);
	Engine e(&TestRead, false, 64);
	Engine::SetProducerThread();
	e.Start();
	std::vector<uint8_t> dst(300000, 0xee);
	e.Copy(0, dst.data(), 300000, Kind::Stage); // 5 pieces of 64 KiB
	e.Copy(1000, dst.data() + 200, 10, Kind::Stream);
	e.JoinAll(Reason::Submit);
	Check(e.Idle(), "idle after a join");
	Check(std::memcmp(dst.data(), g_mem.data(), 200) == 0, "first piece");
	Check(std::memcmp(dst.data() + 200, g_mem.data() + 1000, 10) == 0, "the later copy over the earlier one's bytes");
	Check(std::memcmp(dst.data() + 210, g_mem.data() + 210, 300000 - 210) == 0, "every piece of a big copy");
	auto& s = e.GetStats();
	Check(s.stage_n.load() == 1 && s.stream_n.load() == 1 && s.bytes.load() == 300010, "counters count copies, not pieces");
	Check(s.fail_n.load() == 0 && s.ne_n.load() == 0, "no failures");

	// A join returns only when a running copy is done (JOIN_NO_WAIT).
	Fill(500000, 4096, 9);
	g_slow_begin = 500000;
	g_slow_end   = 500001;
	g_slow_ms    = 30;
	std::vector<uint8_t> slow(4096, 0);
	e.Copy(500000, slow.data(), 4096, Kind::Stage);
	std::this_thread::sleep_for(std::chrono::milliseconds(5)); // the helper has claimed it and sleeps inside the read
	e.JoinAll(Reason::Packet);
	Check(std::memcmp(slow.data(), g_mem.data() + 500000, 4096) == 0, "JoinAll waits for the copy the helper is running");
	ResetSlow();

	// A failed read is zero-filled and counted (FAIL_NO_ZERO).
	g_fail_begin = 600000;
	g_fail_end   = 600001;
	std::vector<uint8_t> failed(512, 0xab);
	e.Copy(600000, failed.data(), 512, Kind::Stream);
	e.JoinAll(Reason::Submit);
	bool zero = true;
	for (auto b: failed) zero = zero && b == 0;
	Check(zero, "a read the backing no longer serves gives zeros (like a read that races an unmap)");
	Check(e.GetStats().fail_n.load() == 1, "and is counted");
	ResetSlow();
	e.Stop();
}

// The ring is refilled only behind finished copies (SLOT_REUSE): capacity 4, one slow copy, then many quick ones.
void RingTests() {
	ResetSlow();
	Fill(0, 65536, 11);
	Engine e(&TestRead, false, 4);
	Engine::SetProducerThread();
	e.Start();
	std::vector<uint8_t> dst(64 * 64, 0);
	g_slow_begin = 0;
	g_slow_end   = 1;
	g_slow_ms    = 40;
	e.Copy(0, dst.data(), 64, Kind::Stage); // seq 0, slow
	std::this_thread::sleep_for(std::chrono::milliseconds(5)); // the helper is inside it
	for (uint64_t i = 1; i < 64; i++) e.Copy(i * 64, dst.data() + i * 64, 64, Kind::Stream);
	e.JoinAll(Reason::Submit);
	Check(std::memcmp(dst.data(), g_mem.data(), 64 * 64) == 0, "every copy of a wrapped ring lands where it was asked to");
	Check(e.GetStats().join_n[static_cast<size_t>(Reason::Full)].load() >= 1, "a full ring joins");
	ResetSlow();
	e.Stop();
}

// A write joins exactly the copies under it (NO_WRITE_JOIN, OVERLAP_END).
void WriteJoinTests() {
	ResetSlow();
	Engine e(&TestRead, false, 64);
	Engine::SetProducerThread();
	e.Start();
	Fill(700000, 1024, 21);
	std::vector<uint8_t> snap(g_mem.begin() + 700000, g_mem.begin() + 700000 + 1024);
	std::vector<uint8_t> dst(1024, 0);
	g_slow_begin = 700000;
	g_slow_end   = 700001;
	g_slow_ms    = 30;
	e.Copy(700000, dst.data(), 1024, Kind::Stage);
	std::this_thread::sleep_for(std::chrono::milliseconds(3));
	// unrelated write: returns without waiting for the slow copy
	const auto t0 = NowNs();
	e.JoinOverlap(700000 + 1024, 16);
	Check(NowNs() - t0 < 20'000'000, "a write next to a pending copy does not wait for it");
	// a write on the copy's last byte waits, then lands
	e.JoinOverlap(700000 + 1023, 1);
	std::atomic_ref<uint8_t>(g_mem[700000 + 1023]).store(0x5a);
	e.JoinAll(Reason::Submit);
	Check(std::memcmp(dst.data(), snap.data(), 1024) == 0, "the copy has the bytes of before the write on its last byte");
	ResetSlow();
	e.Stop();
}

// The oracle: the producer (the recorder) consumes a page (lock = write protection), takes the reference snapshot, hands the copy over and releases the
// page; a writer thread (the guest) takes a page's protection the way a write fault does: it calls the Write join, then stores. After the joins every
// destination must hold its snapshot: a deferred copy never sees a store ordered after its consume.
void OracleTests(uint32_t rounds) {
	ResetSlow();
	constexpr uint64_t kPage  = 4096;
	constexpr uint64_t kPages = 64;
	constexpr uint64_t kBase  = 128 * 1024;
	std::array<std::mutex, kPages> page_lock;
	Engine                          e(&TestRead, false, 256);
	Engine::SetProducerThread();
	e.Start();
	std::atomic<bool> stop {false};
	std::atomic<uint64_t> writes {0};
	std::thread writer([&] {
		std::mt19937_64 rng(77);
		while (!stop.load(std::memory_order_relaxed)) {
			const uint64_t page = rng() % kPages;
			const uint64_t off  = rng() % kPage;
			const uint64_t len  = 1 + rng() % (kPage - off);
			std::lock_guard lock(page_lock[page]);
			const auto at = kBase + page * kPage + off;
			e.JoinOverlap(at, len); // BufferCache::InvalidateMemory from the write fault
			Fill(at, len, static_cast<uint8_t>(rng()));
			writes.fetch_add(1, std::memory_order_relaxed);
		}
	});
	std::mt19937_64 rng(5);
	uint64_t        checked = 0;
	for (uint32_t round = 0; round < rounds; round++) {
		struct Pending {
			uint64_t             at = 0, size = 0;
			std::vector<uint8_t> want, got;
		};
		std::vector<Pending> pend(24);
		for (auto& p: pend) {
			const uint64_t first = rng() % kPages;
			const uint64_t count = 1 + rng() % 3;
			const uint64_t last  = std::min<uint64_t>(kPages - 1, first + count - 1);
			for (auto pg = first; pg <= last; pg++) page_lock[pg].lock();
			p.at   = kBase + first * kPage + rng() % 512;
			p.size = (last - first + 1) * kPage - (p.at - kBase - first * kPage) - rng() % 512;
			p.want.assign(g_mem.begin() + static_cast<int64_t>(p.at), g_mem.begin() + static_cast<int64_t>(p.at + p.size));
			p.got.assign(p.size, 0);
			if (round % 3 == 0) {
				g_slow_begin = p.at;
				g_slow_end   = p.at + 1;
				g_slow_ms    = 1;
			}
			e.Copy(p.at, p.got.data(), p.size, (rng() & 1u) != 0 ? Kind::Stage : Kind::Stream);
			for (auto pg = first; pg <= last; pg++) page_lock[pg].unlock();
			if ((rng() % 8) == 0) e.JoinAll(Reason::Packet);
		}
		e.JoinAll(Reason::Submit);
		ResetSlow();
		for (auto& p: pend) {
			Check(p.got == p.want, "oracle: a deferred copy holds the bytes of its consume, whatever the guest wrote after it");
			checked++;
		}
	}
	stop = true;
	writer.join();
	e.Stop();
	std::printf("UploadAsyncTests: oracle %u rounds, %llu copies checked, %llu guest writes, %llu write joins\n", rounds,
	            static_cast<unsigned long long>(checked), static_cast<unsigned long long>(writes.load()),
	            static_cast<unsigned long long>(e.GetStats().join_n[static_cast<size_t>(Reason::Write)].load()));
	Check(writes.load() > 0, "the writer ran");
}

// shadow: the recorder's copy is the reference; the helper's read at its later time must equal it. A write without its join is reported.
void ShadowTests() {
	ResetSlow();
	Engine e(&TestRead, true, 64, 1u << 16u);
	Engine::SetProducerThread();
	e.Start();
	Fill(800000, 8192, 31);
	// agreeing copies
	for (int i = 0; i < 16; i++) {
		auto* scratch = e.ShadowScratch(512);
		Check(scratch != nullptr, "scratch space");
		std::memcpy(scratch, g_mem.data() + 800000 + i * 512, 512);
		e.Verify(800000 + i * 512, scratch, 512, Kind::Stream);
	}
	e.JoinAll(Reason::Submit);
	Check(e.GetStats().ne_n.load() == 0, "shadow: no difference when nothing writes");
	// arena wraps (64 KiB arena, 200 x 1 KiB) through Full joins
	for (int i = 0; i < 200; i++) {
		auto* scratch = e.ShadowScratch(1024);
		Check(scratch != nullptr, "scratch space after a wrap");
		std::memcpy(scratch, g_mem.data() + 800000, 1024);
		e.Verify(800000, scratch, 1024, Kind::Stage);
	}
	e.JoinAll(Reason::Submit);
	Check(e.GetStats().ne_n.load() == 0, "shadow: the arena is reused only behind finished compares");
	// a write that skips its join is caught (the helper is held inside the read until the write has landed)
	g_slow_begin = 801000;
	g_slow_end   = 801001;
	g_slow_ms    = 20;
	auto* scratch = e.ShadowScratch(256);
	std::memcpy(scratch, g_mem.data() + 801000, 256);
	e.Verify(801000, scratch, 256, Kind::Stage);
	std::this_thread::sleep_for(std::chrono::milliseconds(3));
	Fill(801000, 256, 99); // no JoinOverlap: the missing join shadow exists to find
	e.JoinAll(Reason::Submit);
	Check(e.GetStats().ne_n.load() == 1, "shadow: a store under a pending copy is reported");
	Check(e.m_ne_first[0].load() == 801000, "with its address");
	ResetSlow();
	e.Stop();
}

// Joins from other threads (a guest's write fault, the front) while the producer keeps handing copies over.
void CrossThreadTests() {
	ResetSlow();
	Fill(900000, 65536, 41);
	Engine e(&TestRead, false, 128);
	Engine::SetProducerThread();
	e.Start();
	std::atomic<bool> stop {false};
	std::thread joiner([&] {
		std::mt19937_64 rng(9);
		while (!stop.load()) {
			e.JoinOverlap(900000 + rng() % 65536, 64);
			if ((rng() & 15u) == 0) e.JoinAll(Reason::Packet);
		}
	});
	std::vector<uint8_t> dst(65536, 0);
	for (int r = 0; r < 200; r++) {
		for (uint64_t i = 0; i < 64; i++) e.Copy(900000 + i * 1024, dst.data() + i * 1024, 1024, Kind::Stream);
		e.JoinAll(Reason::Submit);
		Check(std::memcmp(dst.data(), g_mem.data() + 900000, 65536) == 0, "copies with joins on another thread");
		std::memset(dst.data(), 0, dst.size());
	}
	stop = true;
	joiner.join();
	Check(e.GetStats().other_join_ns.load() > 0 || e.GetStats().help_n.load() > 0, "the other thread joined");
	e.Stop();
}

// Only the producer and the upload thread run copies; another thread's join waits (GUEST_HELPS). No upload thread here: the copy waits for the producer.
void GuestWaitsTests() {
	ResetSlow();
	Fill(950000, 4096, 51);
	Engine e(&TestRead, false, 16);
	Engine::SetProducerThread();
	std::vector<uint8_t> dst(4096, 0);
	e.Copy(950000, dst.data(), 4096, Kind::Stage);
	std::atomic<bool> guest_done {false};
	std::atomic<bool> copied_by_guest {false};
	std::thread guest([&] {
		e.JoinOverlap(950000 + 100, 4); // a write fault of a guest thread
		copied_by_guest = std::memcmp(dst.data(), g_mem.data() + 950000, 4096) == 0 && !guest_done.load();
		guest_done = true;
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(30));
	const bool guest_ran_copy = guest_done.load() || std::memcmp(dst.data(), g_mem.data() + 950000, 4096) == 0;
	e.JoinAll(Reason::Submit); // the producer does it
	guest.join();
	Check(!guest_ran_copy, "a guest thread's join does not run the copy itself");
	Check(guest_done.load() && std::memcmp(dst.data(), g_mem.data() + 950000, 4096) == 0, "it returns once the producer has done it");
	(void)copied_by_guest;
}

} // namespace

int main() {
	ModeTests();
	HandlerTests();
	OverlapTests();
	BasicTests();
	RingTests();
	WriteJoinTests();
	ShadowTests();
	CrossThreadTests();
	GuestWaitsTests();
	OracleTests(400);
	std::printf("UploadAsyncTests: OK\n");
	return 0;
}
