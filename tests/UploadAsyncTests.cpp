// S605 (KYTY_UPLOAD_ASYNC): the deferred upload copies (uploadAsync.h Engine). Every check ends in exit(1) when it fails (no crash), so each mutant
// (CMake upload_async_mutant_*, WILL_FAIL) is caught by a failed check:
//   NO_WRITE_JOIN  - a write does not wait for the copies under it: the oracle sees bytes written after the snapshot;
//   JOIN_NO_WAIT   - a join returns while a copy it needs is running: the slow copy is not done after JoinAll;
//   OVERLAP_END    - the overlap test misses a write on the last byte of a copy: the boundary write lands under a pending copy;
//   SLOT_REUSE     - a ring slot is refilled without checking that its copy has run: copies queued behind a slow one are lost;
//   LABEL_NO_JOIN  - RELEASE_MEM counts as a packet that writes no guest memory;
//   SHADOW_BLIND   - the shadow compare never reports a difference;
//   FAIL_NO_ZERO   - a read the backing no longer serves leaves the destination as it was;
//   GUEST_HELPS    - a thread other than the producer runs copies in its join (a guest thread the game can suspend would then hold a copy);
//   STAGE_AS_RACE  - a shadow difference on a protected (stage) source is counted as a stream race instead of an error;
//   SHADOW_LATE_PUBLISH - the S605 shadow: the verify is handed over after the recorder's copy, so a write that faults (joins) right after the
//                    recorder's copy finds nothing to wait for and is reported although mode 1 would have ordered it;
//   REDIRTY_AFTER  - a stage difference whose page's write hook ran after the hand-over (a join that did not hold) is excused as a re-dirty;
//   SHADOW_PER_COPY - a shadow batch hands each verify over just before its own copy: a write that faults on a later range while an earlier one
//                    is copied finds nothing to join (mode 1 hands the whole batch over before any copy).

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

std::mutex           g_ne_mutex;
std::vector<NeInfo>  g_ne_seen;
void NoteNe(const NeInfo& info) {
	std::lock_guard lock(g_ne_mutex);
	g_ne_seen.push_back(info);
}

// shadow: the recorder's copy is the reference; the helper's read at its later time must equal it. A write without its join is reported: as an
// error on a protected (stage) source, as a race on a writable (stream) source, with its address, first byte and age either way (S605b).
void ShadowTests() {
	ResetSlow();
	Engine e(&TestRead, true, 64, 1u << 16u);
	Engine::SetProducerThread();
	e.SetNeObserver(&NoteNe);
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
	const uint8_t before = g_mem[801000 + 10];
	for (uint64_t i = 10; i < 14; i++) std::atomic_ref<uint8_t>(g_mem[801000 + i]).store(static_cast<uint8_t>(g_mem[801000 + i] ^ 0x5au));
	// no JoinOverlap: the missing join shadow exists to find
	e.JoinAll(Reason::Submit);
	Check(e.GetStats().ne_n.load() == 1, "shadow: a store under a pending copy of a protected source is an error");
	Check(e.GetStats().ne_race_n.load() == 0, "and not a race");
	Check(e.m_ne_first[0].load() == 801000, "with its address");
	{
		std::lock_guard lock(g_ne_mutex);
		Check(g_ne_seen.size() == 1, "the observer sees the difference");
		const auto& n = g_ne_seen[0];
		Check(n.src == 801000 && n.size == 256 && n.kind == Kind::Stage, "address, size, path");
		Check(n.first_off == 10 && n.diff_bytes == 4, "first differing byte and count");
		Check(n.recorded == before && n.reread == static_cast<uint8_t>(before ^ 0x5au), "the two bytes");
		Check(n.age_ns >= 10'000'000, "age: the helper read it after the store (>= the 20 ms hold minus the wait)");
		Check(n.last_join == 0xff || n.last_join <= 3, "last join reason");
	}
	// the same unsynchronized store on a writable (stream) source: a race, counted apart, not an error
	g_slow_begin = 802000;
	g_slow_end   = 802001;
	scratch      = e.ShadowScratch(128);
	std::memcpy(scratch, g_mem.data() + 802000, 128);
	e.Verify(802000, scratch, 128, Kind::Stream);
	std::this_thread::sleep_for(std::chrono::milliseconds(3));
	std::atomic_ref<uint8_t>(g_mem[802000 + 127]).store(static_cast<uint8_t>(g_mem[802000 + 127] + 1u));
	e.JoinAll(Reason::Submit);
	Check(e.GetStats().ne_race_n.load() == 1, "shadow: a store on a writable source is a race");
	Check(e.GetStats().ne_n.load() == 1, "and does not count as an error");
	Check(e.GetStats().ne_bytes.load() == 5, "differing bytes of both");
	{
		std::lock_guard lock(g_ne_mutex);
		Check(g_ne_seen.size() == 2 && g_ne_seen[1].kind == Kind::Stream && g_ne_seen[1].first_off == 127 && g_ne_seen[1].index == 1,
		      "the observer sees the race with its path");
	}
	// the same store through its Write join: no difference at all, on either path
	for (const auto kind: {Kind::Stage, Kind::Stream}) {
		g_slow_begin = 803000;
		g_slow_end   = 803001;
		scratch      = e.ShadowScratch(64);
		std::memcpy(scratch, g_mem.data() + 803000, 64);
		e.Verify(803000, scratch, 64, kind);
		std::this_thread::sleep_for(std::chrono::milliseconds(3));
		e.JoinOverlap(803000 + 5, 1);
		std::atomic_ref<uint8_t>(g_mem[803000 + 5]).store(static_cast<uint8_t>(g_mem[803000 + 5] + 1u));
		e.JoinAll(Reason::Submit);
	}
	Check(e.GetStats().ne_n.load() == 1 && e.GetStats().ne_race_n.load() == 1, "a joined store is no difference");
	ResetSlow();
	e.Stop();
}

// S605b: shadow at the moment of mode 1. The recorder hands the verify over, copies, and only then marks it ready; a guest write that faults
// (Write join) after the hand-over waits until the helper has re-read, exactly as it would wait for the deferred copy in mode 1: no difference.
// Under SHADOW_LATE_PUBLISH (S605) the write lands between the recorder's copy and the hand-over and is reported.
void ShadowMomentTests() {
	ResetSlow();
	Engine e(&TestRead, true, 64, 1u << 16u);
	Engine::SetProducerThread();
	e.Start();
	for (int round = 0; round < 4; round++) {
		const uint64_t at = 850000 + static_cast<uint64_t>(round) * 4096;
		Fill(at, 4096, static_cast<uint8_t>(61 + round));
		auto* scratch = e.ShadowScratch(4096);
		Check(scratch != nullptr, "scratch");
		const auto token = e.VerifyBegin(at, scratch, 4096, round % 2 == 0 ? Kind::Stage : Kind::Stream);
		std::memcpy(scratch, g_mem.data() + at, 4096); // the recorder's copy
		std::atomic<bool> landed {false};
		std::thread guest([&] {
			e.JoinOverlap(at + 100, 8); // the write fault of a page the consume protected
			for (uint64_t i = 100; i < 108; i++) std::atomic_ref<uint8_t>(g_mem[at + i]).store(static_cast<uint8_t>(g_mem[at + i] ^ 0xffu));
			landed = true;
		});
		std::this_thread::sleep_for(std::chrono::milliseconds(15)); // the guest is either blocked in its join (right) or has written (S605)
		const bool early = landed.load();
		e.VerifyReady(token);
		guest.join();
		e.JoinAll(Reason::Submit);
		Check(!early, "a write after the hand-over waits for the verify (as for the deferred copy)");
	}
	Check(e.GetStats().ne_n.load() == 0 && e.GetStats().ne_race_n.load() == 0, "shadow at the moment of mode 1: no difference for an ordered write");
	// the helper does not compare before the recorder's bytes are in (ready)
	Fill(870000, 512, 77);
	auto* scratch = e.ShadowScratch(512);
	std::memset(scratch, 0, 512);
	const auto token = e.VerifyBegin(870000, scratch, 512, Kind::Stage);
	std::this_thread::sleep_for(std::chrono::milliseconds(5)); // the helper has claimed it and waits
	std::memcpy(scratch, g_mem.data() + 870000, 512);
	e.VerifyReady(token);
	e.JoinAll(Reason::Submit);
	Check(e.GetStats().ne_n.load() == 0, "the helper waits for the recorder's copy before it compares");
	e.Stop();
}

// S605b: a batch (ForEachUploadRange consumed every range first). While the recorder copies range 0, a guest write faults on range 2: it must wait
// for range 2's verify, which mode 1 would have handed over already. Also: the destinations get the recorder's bytes; ineligible pieces go straight.
void ShadowBatchTests() {
	ResetSlow();
	Engine e(&TestRead, true, 64, 1u << 16u);
	Engine::SetProducerThread();
	e.Start();
	constexpr uint64_t base = 880000;
	Fill(base, 4 * 4096, 91);
	std::vector<uint8_t> want(g_mem.begin() + base, g_mem.begin() + base + 4 * 4096);
	std::vector<uint8_t> dst(4 * 4096, 0);
	BatchPiece pieces[4] = {{base, dst.data(), 4096, true}, {base + 4096, dst.data() + 4096, 4096, true}, {base + 8192, dst.data() + 8192, 4096, true},
	                        {base + 12288, dst.data() + 12288, 4096, false}};
	std::atomic<bool> landed {false};
	bool              early = false;
	std::thread       guest;
	ShadowCopyBatch(e, pieces, 4, Kind::Stage, [&](uint64_t src, uint8_t* out, uint64_t size) {
		std::memcpy(out, g_mem.data() + src, size); // the recorder's copy
		if (src == base) {                          // while range 0 is being copied, the guest writes range 2 (a write fault: Write join first)
			guest = std::thread([&] {
				e.JoinOverlap(base + 8192 + 40, 4);
				for (uint64_t i = 40; i < 44; i++) std::atomic_ref<uint8_t>(g_mem[base + 8192 + i]).store(static_cast<uint8_t>(g_mem[base + 8192 + i] ^ 0x33u));
				landed = true;
			});
			std::this_thread::sleep_for(std::chrono::milliseconds(15));
			early = landed.load();
		}
	});
	guest.join();
	e.JoinAll(Reason::Submit);
	Check(!early, "a write on a later range of the batch waits for its verify");
	Check(e.GetStats().ne_n.load() == 0, "shadow batch at the moment of mode 1: no difference");
	Check(std::memcmp(dst.data(), want.data(), dst.size()) == 0, "every destination holds the recorder's bytes (ineligible piece copied straight)");
	Check(e.GetStats().stage_n.load() == 3, "three verifies, the ineligible piece is not one");
	e.Stop();
}

// S605b: a stage difference is excused (ne_redirty_n) only when the page's write hook ran before the hand-over: the page faulted before there was
// anything to join and its store raced the copy (the page is dirty again, as in mode 0). A hook after the hand-over that did not hold the store
// (simulated: NoteWriteHook without the join) and a store with no hook at all are errors (ne_n).
void ShadowVerdictTests() {
	ResetSlow();
	Engine e(&TestRead, true, 64, 1u << 16u);
	Engine::SetProducerThread();
	e.Start();
	const auto run = [&](uint64_t at, int how) {
		Fill(at, 4096, static_cast<uint8_t>(at >> 12u));
		if (how == 0) e.NoteWriteHook(at + 7, 1); // the fault before the hand-over
		std::this_thread::sleep_for(std::chrono::microseconds(50));
		g_slow_begin = at;
		g_slow_end   = at + 1;
		g_slow_ms    = 10;
		auto*      scratch = e.ShadowScratch(4096);
		const auto token   = e.VerifyBegin(at, scratch, 4096, Kind::Stage);
		std::memcpy(scratch, g_mem.data() + at, 4096);
		e.VerifyReady(token);
		std::this_thread::sleep_for(std::chrono::milliseconds(2)); // the helper holds the piece inside its read
		if (how == 1) e.NoteWriteHook(at + 7, 1);                   // a hook after the hand-over whose join did not hold
		std::atomic_ref<uint8_t>(g_mem[at + 7]).store(static_cast<uint8_t>(g_mem[at + 7] + 1u));
		e.JoinAll(Reason::Submit);
		ResetSlow();
	};
	run(890000 & ~uint64_t {4095}, 0);
	Check(e.GetStats().ne_redirty_n.load() == 1 && e.GetStats().ne_n.load() == 0, "a fault before the hand-over: a re-dirty, not an error");
	run((890000 & ~uint64_t {4095}) + 8192, 1);
	Check(e.GetStats().ne_n.load() == 1 && e.GetStats().ne_redirty_n.load() == 1, "a hook after the hand-over that did not hold the store: an error");
	run((890000 & ~uint64_t {4095}) + 16384, 2);
	Check(e.GetStats().ne_n.load() == 2 && e.GetStats().ne_redirty_n.load() == 1, "a store no hook saw: an error");
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
	ShadowMomentTests();
	ShadowBatchTests();
	ShadowVerdictTests();
	OracleTests(400);
	std::printf("UploadAsyncTests: OK\n");
	return 0;
}
