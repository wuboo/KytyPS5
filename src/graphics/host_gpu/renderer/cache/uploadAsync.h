#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADASYNC_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADASYNC_H_

// S605 (KYTY_UPLOAD_ASYNC=0|1|shadow, default 0): the byte copies of the buffer uploads (guest backing -> mapped staging / stream buffer) leave the
// recorder (Thread_Gpu) for one helper thread. The recorder still makes every decision: it consumes the dirty pages (which write-protects them),
// reserves the staging / stream space and records the copyBuffer; only the memmove is deferred. vkCmdCopyBuffer does not need the bytes when it
// is recorded, only when the GPU runs it, so every copy is done before the command buffer that reads it reaches vkQueueSubmit.
//
// Join points (each one waits for, and helps with, the copies it needs):
//   Submit   - CommandScheduler::Submit of the renderer, after End(): every copy (the command buffer that reads them goes to the queue next).
//   Packet   - before a PM4 handler that may write guest memory (HandlerMayWriteGuest: WRITE_DATA, RELEASE_MEM, EVENT_WRITE, DMA_DATA, COPY_DATA, flip,
//              every drain-class packet): every copy. A label the CP writes is what lets the guest reuse the memory a copy reads, and a copy from an
//              unprotected page (the stream path, a hot page) is not ordered by a fault.
//   Write    - BufferCache::InvalidateMemory (every write fault on a tracked page, file reads into guest memory, DMA pre-invalidation, unmap) and every
//              write into the guest backing (Memory::TryWriteBacking: labels beside GPU bytes, downloads): the copies that overlap the range. A
//              consumed page stays write-protected until such a call, so a write never lands under a copy that has not run yet - the same snapshot
//              the recorder's own memmove took right after the consume.
//   Full     - the ring (or the shadow arena) is full.
// Only ranges that the backing's page table shows fully backed are deferred; the helper reads them with the lock-free backing read
// (TryReadNoLock), which never faults and never takes a lock, so a join can never wait on a thread that waits on the joiner. Anything else
// (not backed, non-coherent memory, a producer other than the recorder) is copied by the recorder as before.
//
// shadow: the recorder copies as before (through a scratch arena: the GPU sees exactly the off bytes); the helper reads the same range again at
// its (later) time, with all join points active, and compares. S605b: the verify is handed over before the recorder's copy, at the point where mode 1
// hands the copy over (S605 published it after the copy and a 64 KiB memcpy into staging: a guest write that faulted in that window found nothing
// to join, landed, and the re-read reported it although the page had been re-dirtied correctly). The difference is classified by the protection of
// the source at record time:
//   Stage  - the consume write-protected the pages before the copy, so every later store faults into a Write join: a difference is a store that
//            bypassed the joins (upas_ne_n, must be 0);
//   Stream - ObtainBuffer's stream path copies CPU-dirty pages, which are writable: a guest store lands there with no fault and no join, in mode 0
//            too (it is only a question of when the copy runs). The GPU reads such a buffer when it executes the draw, after the submit, so a store
//            between the record and the deferred copy is one the hardware shows too; an ordered writer (a CP label, a download, a DMA, a file read)
//            still joins first. These differences are counted apart (upas_ne_race_n) and are not an error.
// Every difference goes to an observer (every error up to 64, the first 32 others, then every 256th) with its address, path, first differing byte, bytes, age.
//
// Mutants (WILL_FAIL, tests/UploadAsyncTests.cpp, each ends in exit(1)): KYTY_UPAS_MUTANT_NO_WRITE_JOIN (a write does not wait for the copies
// under it), KYTY_UPAS_MUTANT_JOIN_NO_WAIT (a join returns while a copy it needs is still running), KYTY_UPAS_MUTANT_OVERLAP_END (the overlap test
// misses a write on the last byte of a copy), KYTY_UPAS_MUTANT_SLOT_REUSE (a ring slot is refilled without checking that its copy has run), KYTY_UPAS_MUTANT_LABEL_NO_JOIN
// (RELEASE_MEM counts as a packet that writes no guest memory), KYTY_UPAS_MUTANT_SHADOW_BLIND (the shadow compare never reports),
// KYTY_UPAS_MUTANT_FAIL_NO_ZERO (a failed read leaves the destination as it was), KYTY_UPAS_MUTANT_GUEST_HELPS (a thread other than the producer runs copies
// in its join), KYTY_UPAS_MUTANT_STAGE_AS_RACE (a difference on a protected source is counted as a stream race), KYTY_UPAS_MUTANT_REDIRTY_AFTER (a
// stage difference whose page's write hook ran after the hand-over is excused as a re-dirty), KYTY_UPAS_MUTANT_SHADOW_LATE_PUBLISH
// (the S605 shadow: the verify is handed over after the recorder's copy, so a write that faults in between finds nothing to join and is reported).

#include "graphics/guest_gpu/pm4.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>
#if defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics::UploadAsync {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::On: return "1";
		case Mode::Shadow: return "shadow";
		default: return "0";
	}
}

// KYTY_UPLOAD_ASYNC (read once). Off: the old path (the copies stay on the recorder), byte for byte.
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_UPLOAD_ASYNC"));
	return mode;
}

enum class Kind : uint8_t { Stage, Stream };
enum class Reason : uint8_t { Submit, Packet, Write, Full, Count };

inline const char* KindName(Kind kind) { return kind == Kind::Stage ? "stage" : "stream"; }
inline const char* ReasonName(uint8_t reason) {
	switch (reason) {
		case 0: return "submit";
		case 1: return "packet";
		case 2: return "write";
		case 3: return "full";
		default: return "none";
	}
}

// S605b: one shadow difference (the observer's argument).
struct NeInfo {
	uint64_t src        = 0;     // guest address of the piece
	uint64_t size       = 0;
	uint64_t first_off  = 0;     // first differing byte, from src
	uint64_t diff_bytes = 0;     // differing bytes in the piece
	int64_t  age_ns     = 0;     // from the recorder's copy to the helper's read
	uint64_t pending    = 0;     // copies handed over and not done when the helper read it
	uint64_t index      = 0;     // differences so far (all kinds)
	Kind     kind       = Kind::Stage;
	uint8_t  last_join  = 0xff;  // Reason of the last join that waited (0xff: none yet)
	uint8_t  recorded   = 0;     // the recorder's byte at first_off
	uint8_t  reread     = 0;     // the helper's byte there
	int64_t  hook_dt_ns = 0;     // last write hook on the differing page minus the hand-over (shadow; INT64_MIN: none remembered)
	uint8_t  verdict    = 0;     // 0 error (counted in ne_n), 1 redirty (before the hand-over), 2 stream race
};
using NeObserver = void (*)(const NeInfo& info);

// Does the handler of this packet possibly store into guest memory on the CP (the label of a signal, a CPU-side DMA, a flip's state, an unknown
// packet)? The list of the packets that do not is the CP pipeline's Inline / Draw / Dispatch / Wait classes (cpPipe.h Classify) minus nothing:
// register writes, index state, draws and dispatches (their buffer and image work goes through the caches, whose guest writes are downloads
// through Memory::TryWriteBacking, a Write join), waits (they read). Everything else joins.
inline bool HandlerMayWriteGuest(uint32_t opcode, uint32_t r) {
	using namespace Pm4;
	switch (opcode) {
		case IT_SET_BASE:
		case IT_CLEAR_STATE:
		case IT_INDEX_BASE:
		case IT_INDEX_BUFFER_SIZE:
		case IT_INDEX_TYPE:
		case IT_NUM_INSTANCES:
		case IT_SET_CONTEXT_REG:
		case IT_SET_SH_REG:
		case IT_SET_UCONFIG_REG:
		case IT_SET_UCONFIG_REG_INDEX:
		case IT_SET_SH_REG_INDIRECT:
		case IT_SET_UCONFIG_REG_INDIRECT:
		case IT_SET_CONTEXT_REG_INDIRECT:
		case IT_PFP_SYNC_ME:
		case IT_ACQUIRE_MEM:
		case IT_INDIRECT_BUFFER:
		case IT_DRAW_INDEX_2:
		case IT_DRAW_INDEX_AUTO:
		case IT_DRAW_INDEX_OFFSET_2:
		case IT_DISPATCH_DIRECT:
		case IT_DISPATCH_INDIRECT:
		case IT_WAIT_REG_MEM:
		case IT_WAIT_REG_MEM_64: return false;
#ifdef KYTY_UPAS_MUTANT_LABEL_NO_JOIN
		case IT_RELEASE_MEM: return false;
#endif
		case IT_NOP:
			switch (r) {
				case R_ZERO:
				case R_PUSH_MARKER:
				case R_POP_MARKER:
				case R_CONTEXT_STATE:
				case R_DISPATCH_RESET: return false;
				default: return true; // R_FLIP and the rest
			}
		default: return true;
	}
}

inline void CpuRelax() {
#if defined(_MSC_VER) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

inline int64_t NowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// The helper's read of guest bytes: true when every byte came from the backing; false = not (or no longer) backed, nothing copied.
using ReadFn = bool (*)(uint64_t src, void* dst, uint64_t size);

// Counters (relaxed atomics; the owner drains them with Take()). Producer side and helper side on separate cache lines.
struct Stats {
	alignas(64) std::atomic<uint64_t> stage_n {0}; // copies (pieces) handed over, staging path
	std::atomic<uint64_t> stream_n {0};            // stream path
	std::atomic<uint64_t> bytes {0};               // bytes handed over
	std::atomic<uint64_t> join_n[static_cast<size_t>(Reason::Count)] {}; // joins that had to wait for or do a copy
	std::atomic<uint64_t> join_ns {0};             // time in those joins on the producer thread (help + wait)
	std::atomic<uint64_t> other_join_ns {0};       // time in joins on other threads (a guest's write fault, the front)
	alignas(64) std::atomic<uint64_t> worker_ns {0}; // helper thread: time copying (shadow: reading + comparing)
	std::atomic<uint64_t> help_n {0};              // copies the producer did itself in a join
	std::atomic<uint64_t> ne_n {0};                // shadow: differences on a protected source (stage path) the joins should have ordered: the page's last
	                                               // write hook ran after the hand-over (a join that did not hold) or there was none (a write no hook sees)
	std::atomic<uint64_t> ne_redirty_n {0};        // shadow: stage differences whose page faulted (write hook) before the hand-over: the store raced the copy
	                                               // itself, as in mode 0, and the page is dirty again (uploaded once more)
	std::atomic<uint64_t> ne_race_n {0};           // shadow: differences on a writable source (stream path): an unsynchronized guest store
	std::atomic<uint64_t> ne_bytes {0};            // shadow: differing bytes (both kinds)
	std::atomic<uint64_t> fail_n {0};              // reads the backing no longer served (zero-filled, like a read that races an unmap)
};

class Engine {
public:
	static constexpr uint64_t kChunk = 64 * 1024; // a deferred copy is split into pieces, so a joiner can help with a big one

	// capacity: ring slots (power of two); arena: shadow scratch bytes (0 when not shadow).
	Engine(ReadFn read, bool shadow, size_t capacity = size_t {1} << 14u, size_t arena = size_t {64} << 20u)
	    : m_read(read), m_shadow(shadow), m_mask(capacity - 1), m_items(new Item[capacity]) {
		if ((capacity & (capacity - 1)) != 0 || capacity < 2) std::abort();
		if (shadow) m_arena.resize(arena);
	}
	~Engine() { Stop(); }
	Engine(const Engine&)            = delete;
	Engine& operator=(const Engine&) = delete;

	void Start(void (*thread_setup)() = nullptr) {
		if (m_thread.joinable()) return;
		m_stop.store(false, std::memory_order_relaxed);
		m_thread = std::thread([this, thread_setup] {
			if (thread_setup != nullptr) thread_setup();
			WorkerLoop();
		});
	}
	void Stop() {
		if (!m_thread.joinable()) return;
		JoinAll(Reason::Submit);
		{
			std::lock_guard lock(m_sleep_mutex);
			m_stop.store(true, std::memory_order_seq_cst);
		}
		m_sleep_cv.notify_all();
		m_thread.join();
	}
	[[nodiscard]] bool Running() const { return m_thread.joinable(); }
	[[nodiscard]] bool Shadow() const { return m_shadow; }
	void               SetNeObserver(NeObserver observer) { m_ne_observer.store(observer, std::memory_order_release); }
	[[nodiscard]] Stats& GetStats() { return m_stats; }

	// Producer (one thread). Mode 1: the helper copies `size` bytes of guest `src` into `dst` before the next join that needs them.
	void Copy(uint64_t src, uint8_t* dst, uint64_t size, Kind kind) {
		(kind == Kind::Stage ? m_stats.stage_n : m_stats.stream_n).fetch_add(1, std::memory_order_relaxed);
		m_stats.bytes.fetch_add(size, std::memory_order_relaxed);
		for (uint64_t done = 0; done < size;) {
			const auto piece = size - done < kChunk ? size - done : kChunk;
			Publish(src + done, dst + done, nullptr, piece, kind);
			done += piece;
		}
	}
	// Producer, shadow: scratch space for the recorder's own copy of `size` bytes (valid until the verify item made from it has run).
	[[nodiscard]] uint8_t* ShadowScratch(uint64_t size) {
		if (size > m_arena.size()) return nullptr;
		if (m_arena.size() - m_arena_used < size) {
			JoinAll(Reason::Full);
		}
		if (Idle()) m_arena_used = 0; // nothing pending reads the arena
		if (m_arena.size() - m_arena_used < size) return nullptr;
		auto* p = m_arena.data() + m_arena_used;
		m_arena_used += (size + 15u) & ~uint64_t {15};
		if (m_arena_used > m_arena.size()) m_arena_used = m_arena.size();
		return p;
	}
	// Producer, shadow (S605b): the verify of `size` bytes of `src` against `expected` (ShadowScratch memory) is handed over BEFORE the recorder copies
	// into `expected`, i.e. at the moment mode 1 would hand the copy over: a write after this point joins it (waits until the helper has re-read), as
	// in mode 1; a write before it is in both reads. The helper waits for VerifyReady before it compares. Returns the token for VerifyReady.
	struct VerifyToken {
		uint64_t first = 0, last = 0;
	};
	[[nodiscard]] VerifyToken VerifyBegin(uint64_t src, const uint8_t* expected, uint64_t size, Kind kind) {
		(kind == Kind::Stage ? m_stats.stage_n : m_stats.stream_n).fetch_add(1, std::memory_order_relaxed);
		m_stats.bytes.fetch_add(size, std::memory_order_relaxed);
		const uint64_t pieces = (size + kChunk - 1) / kChunk;
		// Room for every piece first: a full-ring join after the first piece would wait for a verify that is not ready yet.
		const auto first = m_published.load(std::memory_order_relaxed);
		if (pieces > m_mask + 1 || first + pieces > m_mask + 1 + m_watermark.load(std::memory_order_acquire)) JoinAll(Reason::Full);
		VerifyToken token {first, first};
#ifdef KYTY_UPAS_MUTANT_SHADOW_LATE_PUBLISH
		m_late = {src, expected, size, kind};
		return token;
#endif
		for (uint64_t done = 0; done < size;) {
			const auto piece = size - done < kChunk ? size - done : kChunk;
			Publish(src + done, nullptr, expected + done, piece, kind, false);
			done += piece;
		}
		token.last = m_published.load(std::memory_order_relaxed);
		return token;
	}
	// Producer, shadow: the recorder's bytes are in `expected`; the helper may compare.
	void VerifyReady(VerifyToken token) {
#ifdef KYTY_UPAS_MUTANT_SHADOW_LATE_PUBLISH
		(void)token; // S605 behaviour: the verify appears only after the recorder's copy
		for (uint64_t done = 0; done < m_late.size;) {
			const auto piece = m_late.size - done < kChunk ? m_late.size - done : kChunk;
			Publish(m_late.src + done, nullptr, m_late.expected + done, piece, m_late.kind, true);
			done += piece;
		}
		return;
#endif
		for (auto seq = token.first; seq < token.last; ++seq) m_items[seq & m_mask].ready.store(true, std::memory_order_release);
	}
	// Producer, shadow: VerifyBegin + VerifyReady with the bytes already in `expected` (tests).
	void Verify(uint64_t src, const uint8_t* expected, uint64_t size, Kind kind) { VerifyReady(VerifyBegin(src, expected, size, kind)); }
	// Producer, shadow: makes room in the ring for `pieces` pieces, so that the VerifyBegin calls of one batch never join (they would wait for the
	// verifies of the same batch, which are not ready yet). False when the batch does not fit at all.
	[[nodiscard]] bool ReserveVerify(uint64_t pieces) {
		if (pieces > m_mask + 1) return false;
		if (m_published.load(std::memory_order_relaxed) + pieces > m_mask + 1 + m_watermark.load(std::memory_order_acquire)) JoinAll(Reason::Full);
		return true;
	}
	[[nodiscard]] static uint64_t Pieces(uint64_t size) { return (size + kChunk - 1) / kChunk; }

	// Nothing handed over is still waiting or running (two loads).
	[[nodiscard]] bool Idle() const {
		return m_watermark.load(std::memory_order_acquire) == m_published.load(std::memory_order_acquire);
	}

	// Any thread: every copy handed over so far is done when this returns. Only the producer helps with the copies; another thread (a guest's
	// write fault) only waits, so no thread the guest can suspend ever holds a copy somebody waits for.
	void JoinAll(Reason reason) {
		if (Idle()) return;
		const auto t0     = NowNs();
		const auto target = m_published.load(std::memory_order_acquire);
		while (MayHelp() && RunOne(target, true)) {
		}
		for (auto seq = m_watermark.load(std::memory_order_acquire); seq < target; ++seq) {
			WaitDone(seq);
		}
		AdvanceWatermark(target);
		NoteJoin(reason, t0);
	}

	// Shadow: remembers the last two times a write hook ran for each 4 KiB page (direct-mapped, 65536 entries, time in ns modulo 2^36 = 68 s; a write
	// of more than 64 pages is noted for its first and last 32), so a difference can tell a fault before the hand-over (the store raced the copy,
	// as in mode 0) from one the join missed.
	void NoteWriteHook(uint64_t vaddr, uint64_t size) {
		if (!m_shadow || size == 0) return;
		const auto first = vaddr >> 12u, last = (vaddr + size - 1) >> 12u;
		const auto now   = static_cast<uint64_t>(NowNs()) & kHookTimeMask;
		for (auto page = first; page <= last; ++page) {
			if (last - first > 64 && page == first + 32) page = last - 31;
			const auto slot = page & (kHooks - 1);
			const auto old  = m_hooks[slot].exchange(((page & kHookPageMask) << kHookTimeBits) | now, std::memory_order_relaxed);
			if (old != 0 && (old >> kHookTimeBits) == (page & kHookPageMask)) m_hooks_prev[slot].store(old, std::memory_order_relaxed);
		}
	}
	// Shadow: did a write hook of `page` run before `t_ns` (within 34 s)? *dt_ns: the newest remembered hook minus t_ns (INT64_MIN: none).
	[[nodiscard]] bool HookBefore(uint64_t page, int64_t t_ns, int64_t* dt_ns) const {
		*dt_ns       = INT64_MIN;
		bool before  = false;
		const auto t = static_cast<uint64_t>(t_ns) & kHookTimeMask;
		for (const auto* table: {&m_hooks, &m_hooks_prev}) {
			const auto v = (*table)[page & (kHooks - 1)].load(std::memory_order_relaxed);
			if (v == 0 || (v >> kHookTimeBits) != (page & kHookPageMask)) continue;
			constexpr uint64_t half = (kHookTimeMask + 1) / 2;
			const auto         dt   = static_cast<int64_t>(((v & kHookTimeMask) - t + half) & kHookTimeMask) - static_cast<int64_t>(half);
			if (*dt_ns == INT64_MIN) *dt_ns = dt;
			if (dt < 0) before = true;
		}
		return before;
	}

	// Any thread: before a write of [vaddr, vaddr + size) lands, the copies that read any of these bytes are done.
	void JoinOverlap(uint64_t vaddr, uint64_t size) {
		NoteWriteHook(vaddr, size);
#ifdef KYTY_UPAS_MUTANT_NO_WRITE_JOIN
		(void)vaddr;
		(void)size;
		return;
#endif
		if (size == 0 || Idle()) return;
		const auto target = m_published.load(std::memory_order_acquire);
		auto       first  = m_watermark.load(std::memory_order_acquire);
		if (target - first > kMaxScan) { // far behind: no point in a long scan
			JoinAll(Reason::Write);
			return;
		}
		uint64_t last = 0; // one past the newest overlapping copy that is not done
		for (auto seq = first; seq < target; ++seq) {
			const auto& it = m_items[seq & m_mask];
			if (it.done_tag.load(std::memory_order_acquire) > seq) continue; // done (or the slot was reused, which needs it done)
			const auto src   = it.src.load(std::memory_order_relaxed);
			const auto bytes = it.size.load(std::memory_order_relaxed);
			if (Overlaps(src, bytes, vaddr, size)) last = seq + 1;
		}
		if (last == 0) return;
		const auto t0 = NowNs();
		while (MayHelp() && RunOne(last, true)) {
		}
		for (auto seq = first; seq < last; ++seq) {
			const auto& it = m_items[seq & m_mask];
			if (it.done_tag.load(std::memory_order_acquire) > seq) continue;
			if (Overlaps(it.src.load(std::memory_order_relaxed), it.size.load(std::memory_order_relaxed), vaddr, size)) WaitDone(seq);
		}
		NoteJoin(Reason::Write, t0);
	}

	// The calling thread is the producer (the one that calls Copy / Verify): its joins go to join_ns, the others' to other_join_ns.
	static void SetProducerThread() { ProducerTls() = true; }
	[[nodiscard]] static bool IsProducerThread() { return ProducerTls(); }

	[[nodiscard]] static bool Overlaps(uint64_t a, uint64_t a_size, uint64_t b, uint64_t b_size) {
#ifdef KYTY_UPAS_MUTANT_OVERLAP_END
		return a < b + b_size && b < a + a_size - 1;
#else
		return a < b + b_size && b < a + a_size;
#endif
	}

private:
	static constexpr uint64_t kMaxScan = 512;

	static bool& ProducerTls() {
		static thread_local bool producer = false;
		return producer;
	}
	static bool MayHelp() {
#ifdef KYTY_UPAS_MUTANT_GUEST_HELPS
		return true;
#else
		return ProducerTls();
#endif
	}

	struct Item {
		std::atomic<uint64_t> done_tag {0}; // seq + 1 once the copy of `seq` (the item in this slot) is done
		std::atomic<uint64_t> src {0};      // atomics: a write's overlap scan may read a slot while the producer refills it (after it was done)
		std::atomic<uint64_t> size {0};
		uint8_t*              dst      = nullptr; // mode 1
		const uint8_t*        expected = nullptr; // shadow
		int64_t               t_pub    = 0;       // shadow: when the verify was handed over (before the recorder's copy)
		Kind                  kind     = Kind::Stage;
		std::atomic<bool>     ready {true};       // shadow: the recorder's bytes are in `expected`
	};
#ifdef KYTY_UPAS_MUTANT_SHADOW_LATE_PUBLISH
	struct Late {
		uint64_t       src      = 0;
		const uint8_t* expected = nullptr;
		uint64_t       size     = 0;
		Kind           kind     = Kind::Stage;
	} m_late;
#endif

	void Publish(uint64_t src, uint8_t* dst, const uint8_t* expected, uint64_t size, Kind kind, bool ready = true) {
		const auto seq  = m_published.load(std::memory_order_relaxed);
		auto&      slot = m_items[seq & m_mask];
#ifndef KYTY_UPAS_MUTANT_SLOT_REUSE
		if (seq > m_mask && slot.done_tag.load(std::memory_order_acquire) <= seq - (m_mask + 1)) {
			JoinAll(Reason::Full); // the slot's previous copy has not run yet
		}
#endif
		slot.src.store(src, std::memory_order_relaxed);
		slot.size.store(size, std::memory_order_relaxed);
		slot.dst      = dst;
		slot.expected = expected;
		slot.kind     = kind;
		slot.t_pub    = expected != nullptr ? NowNs() : 0;
		slot.ready.store(ready, std::memory_order_relaxed); // published by the seq_cst store below
		m_published.store(seq + 1, std::memory_order_seq_cst);
		if (m_sleeping.load(std::memory_order_seq_cst)) {
			std::lock_guard lock(m_sleep_mutex);
			m_sleep_cv.notify_one();
		}
	}

	// Claims and runs the oldest copy below `limit`; false when there is none left to claim.
	bool RunOne(uint64_t limit, bool helper) {
		auto seq = m_next.load(std::memory_order_acquire);
		for (;;) {
			if (seq >= limit || seq >= m_published.load(std::memory_order_acquire)) return false;
			if (m_next.compare_exchange_weak(seq, seq + 1, std::memory_order_acq_rel, std::memory_order_acquire)) break;
		}
		auto&      it    = m_items[seq & m_mask];
		const auto src   = it.src.load(std::memory_order_relaxed);
		const auto size  = it.size.load(std::memory_order_relaxed);
		auto*      dst   = it.dst;
		const auto* exp  = it.expected;
		if (exp == nullptr) {
			if (!m_read(src, dst, size)) {
#ifndef KYTY_UPAS_MUTANT_FAIL_NO_ZERO
				std::memset(dst, 0, size);
#endif
				m_stats.fail_n.fetch_add(1, std::memory_order_relaxed);
			}
		} else {
			// The recorder copies right after VerifyBegin (no lock, no fault in between), so this wait is its copy time.
			for (uint32_t spin = 0; !it.ready.load(std::memory_order_acquire); ++spin) {
				if (spin < 4096) {
					CpuRelax();
				} else {
					std::this_thread::yield();
				}
			}
			uint8_t  tmp[4096];
			uint64_t diff = 0, first = UINT64_MAX;
			uint8_t  rec_byte = 0, new_byte = 0;
			for (uint64_t off = 0; off < size; off += sizeof(tmp)) {
				const auto n = size - off < sizeof(tmp) ? size - off : sizeof(tmp);
				if (!m_read(src + off, tmp, n)) {
					m_stats.fail_n.fetch_add(1, std::memory_order_relaxed);
					std::memset(tmp, 0, n);
				}
				if (std::memcmp(tmp, exp + off, n) == 0) continue;
				for (uint64_t i = 0; i < n; i++) {
					if (tmp[i] == exp[off + i]) continue;
					if (first == UINT64_MAX) {
						first    = off + i;
						rec_byte = exp[off + i];
						new_byte = tmp[i];
					}
					diff++;
				}
			}
#ifndef KYTY_UPAS_MUTANT_SHADOW_BLIND
			if (diff != 0) {
				const auto kind = it.kind;
#ifdef KYTY_UPAS_MUTANT_STAGE_AS_RACE
				const bool race = true;
#else
				const bool race = kind == Kind::Stream;
#endif
				// Stage: a write hook on the differing page before the hand-over = the page faulted before the copy was handed over (there was nothing
				// to join), the store raced the copy itself and the page is dirty again: the same in mode 0. After it, or none: an error.
				int64_t    hook_dt   = INT64_MIN;
				const bool before    = HookBefore((src + first) >> 12u, it.t_pub, &hook_dt);
#ifdef KYTY_UPAS_MUTANT_REDIRTY_AFTER
				const bool redirty = !race && hook_dt != INT64_MIN;
#else
				const bool redirty = !race && before;
#endif
				(race ? m_stats.ne_race_n : (redirty ? m_stats.ne_redirty_n : m_stats.ne_n)).fetch_add(1, std::memory_order_relaxed);
				m_stats.ne_bytes.fetch_add(diff, std::memory_order_relaxed);
				const auto index = m_ne_index.fetch_add(1, std::memory_order_relaxed);
				if (!race && !redirty) {
					const auto stage_index = m_ne_stage_index.fetch_add(1, std::memory_order_relaxed);
					if (stage_index < 8) m_ne_first[stage_index].store(src, std::memory_order_relaxed);
				}
				const auto observer = m_ne_observer.load(std::memory_order_acquire);
				const bool error    = !race && !redirty;
				const auto err_seen = error ? m_ne_error_logged.fetch_add(1, std::memory_order_relaxed) : uint64_t {0};
				if (observer != nullptr && ((error && err_seen < 64) || index < 32 || (index & 255u) == 0)) {
					NeInfo info;
					info.src        = src;
					info.size       = size;
					info.first_off  = first;
					info.diff_bytes = diff;
					info.age_ns     = NowNs() - it.t_pub;
					info.pending    = m_published.load(std::memory_order_relaxed) - m_watermark.load(std::memory_order_relaxed);
					info.index      = index;
					info.kind       = kind;
					info.last_join  = m_last_join.load(std::memory_order_relaxed);
					info.recorded   = rec_byte;
					info.reread     = new_byte;
					info.hook_dt_ns = hook_dt;
					info.verdict    = race ? 2 : (redirty ? 1 : 0);
					observer(info);
				}
			}
#endif
		}
		if (helper) m_stats.help_n.fetch_add(1, std::memory_order_relaxed);
		it.done_tag.store(seq + 1, std::memory_order_release);
		AdvanceWatermark(seq + 1);
		return true;
	}

	void WaitDone(uint64_t seq) const {
#ifdef KYTY_UPAS_MUTANT_JOIN_NO_WAIT
		(void)seq;
		return;
#endif
		const auto& it = m_items[seq & m_mask];
		for (uint32_t spin = 0; it.done_tag.load(std::memory_order_acquire) <= seq; ++spin) {
			if (spin < 4096) {
				CpuRelax();
			} else {
				std::this_thread::yield();
			}
		}
	}

	// Moves the watermark (every copy below it is done) forward over the done slots, at most to `limit`.
	void AdvanceWatermark(uint64_t limit) {
		auto w = m_watermark.load(std::memory_order_acquire);
		while (w < limit && m_items[w & m_mask].done_tag.load(std::memory_order_acquire) > w) {
			if (m_watermark.compare_exchange_weak(w, w + 1, std::memory_order_acq_rel, std::memory_order_acquire)) ++w;
		}
	}

	void NoteJoin(Reason reason, int64_t t0) {
		const auto ns = static_cast<uint64_t>(NowNs() - t0);
		m_last_join.store(static_cast<uint8_t>(reason), std::memory_order_relaxed);
		m_stats.join_n[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
		(ProducerTls() ? m_stats.join_ns : m_stats.other_join_ns).fetch_add(ns, std::memory_order_relaxed);
	}

	void WorkerLoop() {
		constexpr int64_t kSpinNs = 200'000; // busy-waits this long after the last copy before it sleeps (copies come every few microseconds in a frame)
		int64_t           idle_since = 0;
		int64_t           batch_t0   = 0;
		for (;;) {
			if (RunOne(UINT64_MAX, false)) {
				if (batch_t0 == 0) batch_t0 = NowNs();
				idle_since = 0;
				continue;
			}
			const auto now = NowNs();
			if (batch_t0 != 0) {
				m_stats.worker_ns.fetch_add(static_cast<uint64_t>(now - batch_t0), std::memory_order_relaxed);
				batch_t0 = 0;
			}
			if (m_stop.load(std::memory_order_acquire)) return;
			if (idle_since == 0) idle_since = now;
			if (now - idle_since < kSpinNs) {
				for (int i = 0; i < 32; i++) CpuRelax();
				continue;
			}
			std::unique_lock lock(m_sleep_mutex);
			m_sleeping.store(true, std::memory_order_seq_cst);
			m_sleep_cv.wait_for(lock, std::chrono::milliseconds(2), [this] {
				return m_stop.load(std::memory_order_seq_cst) ||
				       m_next.load(std::memory_order_seq_cst) < m_published.load(std::memory_order_seq_cst);
			});
			m_sleeping.store(false, std::memory_order_seq_cst);
			idle_since = 0;
		}
	}

public:
	// Shadow: the guest addresses of the first differing copies on a protected source (diagnostics).
	std::atomic<uint64_t> m_ne_first[8] {};

private:
	ReadFn                    m_read;
	bool                      m_shadow;
	uint64_t                  m_mask;
	std::unique_ptr<Item[]>   m_items;
	std::vector<uint8_t>      m_arena;
	uint64_t                  m_arena_used = 0;
	alignas(64) std::atomic<uint64_t> m_published {0}; // producer: copies handed over
	alignas(64) std::atomic<uint64_t> m_next {0};      // copies claimed (helper, joiners)
	alignas(64) std::atomic<uint64_t> m_watermark {0}; // every copy below it is done
	alignas(64) std::atomic<bool> m_sleeping {false};
	std::atomic<bool>             m_stop {false};
	std::atomic<NeObserver>       m_ne_observer {nullptr};
	std::atomic<uint64_t>         m_ne_index {0};
	std::atomic<uint64_t>         m_ne_stage_index {0};
	std::atomic<uint64_t>         m_ne_error_logged {0};
	static constexpr uint64_t     kHooks        = 65536;
	static constexpr uint32_t     kHookTimeBits = 36;
	static constexpr uint64_t     kHookTimeMask = (uint64_t {1} << kHookTimeBits) - 1;
	static constexpr uint64_t     kHookPageMask = (uint64_t {1} << (64 - kHookTimeBits)) - 1;
	using HookTable = std::unique_ptr<std::atomic<uint64_t>[]>;
	HookTable                     m_hooks {new std::atomic<uint64_t>[kHooks] {}};
	HookTable                     m_hooks_prev {new std::atomic<uint64_t>[kHooks] {}};
	std::atomic<uint8_t>          m_last_join {0xff};
	std::mutex                    m_sleep_mutex;
	std::condition_variable       m_sleep_cv;
	Stats                         m_stats;
	std::thread                   m_thread;
};

// S605b: one upload batch in shadow mode (ForEachUploadRange consumes every range of the batch first, then the copies run one after another).
// Mode 1 hands all of them over within nanoseconds, before any byte is copied, so the shadow hands all verifies over first, then the recorder copies
// (each eligible piece through scratch, the rest straight), then the verifies become ready and the scratch goes to its destination. Without the
// batch order (KYTY_UPAS_MUTANT_SHADOW_PER_COPY) a write that faults while an earlier range is copied finds nothing to join for a later one.
struct BatchPiece {
	uint64_t src      = 0;
	uint8_t* dst      = nullptr;
	uint64_t size     = 0;
	bool     eligible = false; // deferred in mode 1 (backed, coherent, on the recorder)
};
template <typename SyncCopy>
void ShadowCopyBatch(Engine& e, BatchPiece* pieces, size_t n, Kind kind, SyncCopy&& sync_copy) {
	uint64_t bytes = 0, ring = 0;
	for (size_t i = 0; i < n; i++) {
		if (!pieces[i].eligible) continue;
		bytes += (pieces[i].size + 15u) & ~uint64_t {15};
		ring += Engine::Pieces(pieces[i].size);
	}
	uint8_t* scratch = bytes != 0 ? e.ShadowScratch(bytes) : nullptr;
	if (scratch == nullptr || !e.ReserveVerify(ring)) {
		for (size_t i = 0; i < n; i++) sync_copy(pieces[i].src, pieces[i].dst, pieces[i].size);
		return;
	}
	constexpr size_t kInline = 64;
	Engine::VerifyToken inline_tokens[kInline];
	std::vector<Engine::VerifyToken> more;
	if (n > kInline) more.resize(n);
	auto* tokens = n > kInline ? more.data() : inline_tokens;
	uint64_t off = 0;
	std::vector<uint64_t> offsets_more;
	uint64_t inline_offsets[kInline];
	if (n > kInline) offsets_more.resize(n);
	auto* offsets = n > kInline ? offsets_more.data() : inline_offsets;
	for (size_t i = 0; i < n; i++) {
		offsets[i] = off;
		if (!pieces[i].eligible) continue;
#ifndef KYTY_UPAS_MUTANT_SHADOW_PER_COPY
		tokens[i] = e.VerifyBegin(pieces[i].src, scratch + off, pieces[i].size, kind);
#endif
		off += (pieces[i].size + 15u) & ~uint64_t {15};
	}
	for (size_t i = 0; i < n; i++) {
		if (!pieces[i].eligible) {
			sync_copy(pieces[i].src, pieces[i].dst, pieces[i].size);
			continue;
		}
#ifdef KYTY_UPAS_MUTANT_SHADOW_PER_COPY
		tokens[i] = e.VerifyBegin(pieces[i].src, scratch + offsets[i], pieces[i].size, kind);
#endif
		sync_copy(pieces[i].src, scratch + offsets[i], pieces[i].size);
	}
	for (size_t i = 0; i < n; i++) {
		if (!pieces[i].eligible) continue;
		e.VerifyReady(tokens[i]);
		std::memcpy(pieces[i].dst, scratch + offsets[i], static_cast<size_t>(pieces[i].size));
	}
}

// The renderer's engine (nullptr until the buffer cache starts it; never with KYTY_UPLOAD_ASYNC=0).
inline std::atomic<Engine*>& Global() {
	static std::atomic<Engine*> engine {nullptr};
	return engine;
}

// The join hooks of the other files: one relaxed load when the switch is off.
inline void BeforeHandler(uint32_t opcode, uint32_t r) {
	auto* e = Global().load(std::memory_order_acquire);
	if (e == nullptr || !HandlerMayWriteGuest(opcode, r)) return;
	e->JoinAll(Reason::Packet);
}
inline void BeforeCpuWrite(uint64_t vaddr, uint64_t size) {
	auto* e = Global().load(std::memory_order_acquire);
	if (e == nullptr) return;
	e->JoinOverlap(vaddr, size);
}

} // namespace Libs::Graphics::UploadAsync

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_UPLOADASYNC_H_
