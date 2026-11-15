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
// its (later) time, with all join points active, and compares: upas_ne_n counts the copies the deferral would have changed (must be 0).
//
// Mutants (WILL_FAIL, tests/UploadAsyncTests.cpp, each ends in exit(1)): KYTY_UPAS_MUTANT_NO_WRITE_JOIN (a write does not wait for the copies
// under it), KYTY_UPAS_MUTANT_JOIN_NO_WAIT (a join returns while a copy it needs is still running), KYTY_UPAS_MUTANT_OVERLAP_END (the overlap test
// misses a write on the last byte of a copy), KYTY_UPAS_MUTANT_SLOT_REUSE (a ring slot is refilled without checking that its copy has run), KYTY_UPAS_MUTANT_LABEL_NO_JOIN
// (RELEASE_MEM counts as a packet that writes no guest memory), KYTY_UPAS_MUTANT_SHADOW_BLIND (the shadow compare never reports),
// KYTY_UPAS_MUTANT_FAIL_NO_ZERO (a failed read leaves the destination as it was), KYTY_UPAS_MUTANT_GUEST_HELPS (a thread other than the producer runs copies
// in its join).

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
	std::atomic<uint64_t> ne_n {0};                // shadow: copies whose bytes the deferral would have changed
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
	[[nodiscard]] Stats& GetStats() { return m_stats; }

	// Producer (one thread). Mode 1: the helper copies `size` bytes of guest `src` into `dst` before the next join that needs them.
	void Copy(uint64_t src, uint8_t* dst, uint64_t size, Kind kind) {
		(kind == Kind::Stage ? m_stats.stage_n : m_stats.stream_n).fetch_add(1, std::memory_order_relaxed);
		m_stats.bytes.fetch_add(size, std::memory_order_relaxed);
		for (uint64_t done = 0; done < size;) {
			const auto piece = size - done < kChunk ? size - done : kChunk;
			Publish(src + done, dst + done, nullptr, piece);
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
	// Producer, shadow: the helper reads `src` again and compares it with `expected` (the recorder's bytes, in ShadowScratch memory).
	void Verify(uint64_t src, const uint8_t* expected, uint64_t size, Kind kind) {
		(kind == Kind::Stage ? m_stats.stage_n : m_stats.stream_n).fetch_add(1, std::memory_order_relaxed);
		m_stats.bytes.fetch_add(size, std::memory_order_relaxed);
		for (uint64_t done = 0; done < size;) {
			const auto piece = size - done < kChunk ? size - done : kChunk;
			Publish(src + done, nullptr, expected + done, piece);
			done += piece;
		}
	}

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

	// Any thread: before a write of [vaddr, vaddr + size) lands, the copies that read any of these bytes are done.
	void JoinOverlap(uint64_t vaddr, uint64_t size) {
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
	};

	void Publish(uint64_t src, uint8_t* dst, const uint8_t* expected, uint64_t size) {
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
			uint8_t tmp[4096];
			bool    same = true;
			for (uint64_t off = 0; off < size && same; off += sizeof(tmp)) {
				const auto n = size - off < sizeof(tmp) ? size - off : sizeof(tmp);
				if (!m_read(src + off, tmp, n)) {
					m_stats.fail_n.fetch_add(1, std::memory_order_relaxed);
					std::memset(tmp, 0, n);
				}
				same = std::memcmp(tmp, exp + off, n) == 0;
			}
#ifndef KYTY_UPAS_MUTANT_SHADOW_BLIND
			if (!same) {
				const auto n = m_stats.ne_n.fetch_add(1, std::memory_order_relaxed);
				if (n < 8) {
					m_ne_first[n].store(src, std::memory_order_relaxed);
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
	// Shadow: the guest addresses of the first differing copies (diagnostics).
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
	std::mutex                    m_sleep_mutex;
	std::condition_variable       m_sleep_cv;
	Stats                         m_stats;
	std::thread                   m_thread;
};

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
