#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMDEDUP_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMDEDUP_H_

// S614 (KYTY_STREAM_DEDUP=0|1|strict|shadow|shadow_strict, default 0): a stream copy that repeats an earlier one reuses its stream offset.
//
// The stream path of BufferCache::ObtainBuffer copies a small (<= one 16 KiB caching page) CPU-modified, not GPU-modified range into the stream
// ring at every binding. S613 (KYTY_STREAM_PROBE=1, garden): 12 964 copies per flip, 10 817 (83 %) of them the same range as an earlier copy in the
// same command buffer with the same bytes (no repeat with other bytes at all). With the switch on, such a repeat binds the earlier copy's offset:
// no Map, no copy (no hand-over to the upload thread, no memmove there), no Commit.
//
// When is the earlier copy what the GPU would get from a new copy? Off, the GPU gets the guest bytes at the time of the new copy (its deferred
// memmove with KYTY_UPLOAD_ASYNC=1; every ordered writer joins that copy before it writes). The earlier copy holds the bytes at its own time. So the
// reuse is exact when (a) the earlier copy's region still holds them when the GPU reads it, and (b) no store that the new copy must see happened
// in between. A store the new copy must see is one ordered before it: a store of the CP itself (earlier in the stream), or a guest store that the
// CP has been told about since (a wait it read, a submission). The key is (address, size) in a direct-mapped table; a hit needs all of:
//   tick       - the same scheduler tick (command buffer). The ring keeps a region of a tick until the GPU is past it: overwriting it needs
//                Scheduler().Wait(tick), which submits the tick first; ObtainBuffer maps with allow_wait = false. (a)
//   page gens  - no store on the range's pages since: the CP's own CPU stores, at the handler, with the extent the S609 classifier knows
//                (CpPipe::Checked::ClassifyItem: WRITE_DATA, DMA_DATA, COPY_DATA, the RELEASE_MEM label, the synthetic occlusion dump of
//                EVENT_WRITE 0x39; a cache-flush EVENT_WRITE stores nothing), and every CPU write hook (BufferCache::InvalidateMemory: write
//                faults, file reads, DMA pre-invalidation, unmap; Memory::TryWriteBacking: labels beside GPU bytes, downloads), bumped before the
//                write lands, any thread. 16 KiB pages hashed into kGenSlots counters (an alias only costs a hit).
//   rec_epoch  - no recorder event since: a handler that may write guest memory with no known extent (a flip, every drain-class packet, any
//                ordered packet when the host occlusion queries defer their dumps, or before the CP said whether they do), a wait
//                (WAIT_REG_MEM: a label the guest stores after its data), every Thread_Gpu loop iteration (a new submission or command).
//                Plain counter: only Thread_Gpu bumps and reads it.
//   epoch      - the same events on another thread: a WAIT_REG_MEM the CP front evaluates (KYTY_CP_FRONT_THREAD: at parse time, bumped again
//                after the label is read), a guest submission or command (GuestGpu::Enqueue / SendCommand, on the guest thread, before it is
//                queued), a CPU write of more than kMaxPages pages.
// A CP signal (a label, an interrupt) orders nothing the CP reads later by itself: the guest's stores after it reach the CP's later reads only
// through a wait or a submission, which bump. What stays out, as in S605 (uploadAsync.h, Kind::Stream): a guest store ordered with nothing the CP
// sees (it lands on a CPU-dirty page, which is writable, so nothing is told); the hardware shows such a store or not depending on when the GPU
// reads. S613 counted 0 repeats with other bytes in 10,8 M. The GPU-modified and CPU-modified checks of the stream path run as before at every
// binding (a GPU write in between takes the range off the stream path). A hit never refreshes the entry: the region holds the bytes of the copy
// that made it. strict (and shadow_strict): every handler that may write guest memory bumps rec_epoch (the S605 join list), no extents used.
//
// shadow: the copy is made every time (the GPU sees exactly the off bytes), and every stream copy hashes the guest bytes (XXH3 of a private
// CopySparseMemory) at the point where mode 1 would decide. A would-be hit compares with the hash of the copy it would reuse: a difference is
// stream_dedup_ne_n (must be 0). A repeat in the same tick that the rule refused is counted as stream_dedup_lost_n (same bytes: the rule was too
// careful) or stream_dedup_caught_n (other bytes: the rule mattered). The shadow keeps the entry of the copy mode 1 would reuse.
//
// frames.csv (flushed at every submit, GPU thread): stream_dedup_look_n (stream-path bindings seen), _hit_n, _hit_bytes (1: reused; shadow: would
// reuse), _new_n, _miss_tick_n, _miss_epoch_n, _miss_page_n, _skip_n (not Thread_Gpu, or more than two pages), _ne_n, _lost_n, _caught_n (shadow),
// events: _bump_pkt_n (handlers without a known extent), _bump_wait_n, _bump_sub_n, _bump_write_n (write hooks), _bump_big_n, _bump_cpw_n (CP
// stores with a known extent), _bump_nostore_n (ordered packets that store nothing).
//
// Pure part (tests/StreamDedupTests.cpp: an oracle of a CP stream of real PM4 payloads with ordered writes, waits, submissions, write hooks and ticks
// against copies made every time, both rules). Mutants (WILL_FAIL): KYTY_STREAMDEDUP_MUTANT_TICK_IGNORED, _EPOCH_IGNORED (the global epoch),
// _REC_EPOCH_IGNORED, _LAST_PAGE_IGNORED (only the first page's generation), _SIZE_IGNORED, _WAIT_NO_BUMP, _LABEL_NO_BUMP (RELEASE_MEM bumps
// nothing), _KNOWN_AS_NONE (a CP store of known extent bumps nothing), _PRECISE_ALWAYS (extents used with host occlusion queries or before the CP
// said), _SUBMIT_NO_BUMP, _BIG_NO_BUMP (a write of many pages bumps nothing), _SHADOW_BLIND, _SHADOW_REPLACES (a would-be hit replaces the
// entry), _DEFAULT_ON.

#include "graphics/guest_gpu/command_processor/cpPipeChecked.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/cache/uploadAsync.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::StreamDedup {

enum class Mode : uint8_t { Off = 0, On = 1, Shadow = 2 };

struct Config {
	Mode mode   = Mode::Off;
	bool strict = false; // every handler that may write guest memory bumps rec_epoch (no extents)
};

[[nodiscard]] inline Config ParseConfig(const char* text) {
#ifdef KYTY_STREAMDEDUP_MUTANT_DEFAULT_ON
	if (text == nullptr) return {Mode::On, false};
#endif
	if (text == nullptr) return {};
	if (std::strcmp(text, "1") == 0 || std::strcmp(text, "on") == 0) return {Mode::On, false};
	if (std::strcmp(text, "strict") == 0) return {Mode::On, true};
	if (std::strcmp(text, "shadow") == 0) return {Mode::Shadow, false};
	if (std::strcmp(text, "shadow_strict") == 0) return {Mode::Shadow, true};
	return {};
}
[[nodiscard]] inline Mode ParseMode(const char* text) {
	return ParseConfig(text).mode;
}

[[nodiscard]] inline const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::On: return "on";
		case Mode::Shadow: return "shadow";
		default: return "off";
	}
}

// The first call logs `Environment: KYTY_STREAM_DEDUP=<value> mode=<off|on|shadow> rule=<pages|strict>`.
inline Config EnvConfig() {
	static const Config config = [] {
		const char*  value = std::getenv("KYTY_STREAM_DEDUP");
		const Config c     = ParseConfig(value);
		std::fprintf(stderr, "Environment: KYTY_STREAM_DEDUP=%s mode=%s rule=%s\n", value != nullptr ? value : "", ModeName(c.mode),
		             c.strict ? "strict" : "pages");
		std::fflush(stderr);
		return c;
	}();
	return config;
}
inline Mode EnvMode() {
	return EnvConfig().mode;
}

inline constexpr uint32_t kPageBits = 14;   // BufferCache::CACHING_PAGEBITS
inline constexpr uint64_t kMaxPages = 16;   // a CPU write of more pages bumps the global epoch
inline constexpr size_t   kGenSlots = 4096; // page generation counters

enum Bump : uint8_t { BumpPkt, BumpWait, BumpSub, BumpWrite, BumpBig, BumpCpw, BumpNoStore, BumpKinds };

// Process-wide state. Hooks are a relaxed load when the switch is off (g_active stays 0 until the buffer cache activates it).
inline std::atomic<uint8_t> g_active {0};
inline std::atomic<uint8_t> g_strict {0};
inline std::atomic<uint8_t> g_occlusion {0}; // 0: the CP has not said, 1: no host occlusion queries, 2: host occlusion queries (dumps deferred)
alignas(64) inline std::atomic<uint64_t> g_epoch {0};
alignas(64) inline std::atomic<uint32_t> g_page_gen[kGenSlots] {};
alignas(64) inline std::atomic<uint64_t> g_bump_n[BumpKinds] {};
alignas(64) inline uint64_t g_rec_epoch = 0; // Thread_Gpu only
inline thread_local bool t_recorder = false;

[[nodiscard]] inline Mode Active() {
	return static_cast<Mode>(g_active.load(std::memory_order_relaxed));
}
inline void Activate(Mode mode, bool strict = false) {
	g_strict.store(strict ? 1 : 0, std::memory_order_relaxed);
	g_active.store(static_cast<uint8_t>(mode), std::memory_order_release);
}
// CommandProcessor (pipe set-up): whether the host occlusion queries defer the dumps (then no ordered store has a known extent).
inline void NoteHostOcclusion(bool host_occlusion) {
	g_occlusion.store(host_occlusion ? 2 : 1, std::memory_order_relaxed);
}
// The extents of the CP's stores are used: rule pages, and the CP said the host occlusion queries are off.
[[nodiscard]] inline bool Precise() {
#ifdef KYTY_STREAMDEDUP_MUTANT_PRECISE_ALWAYS
	return g_strict.load(std::memory_order_relaxed) == 0;
#else
	return g_strict.load(std::memory_order_relaxed) == 0 && g_occlusion.load(std::memory_order_relaxed) == 1;
#endif
}
// Thread_Gpu (GuestGpu::ThreadRun): the thread that owns the table and rec_epoch.
inline void BindRecorderThread() {
	t_recorder = true;
}
[[nodiscard]] inline bool OnRecorder() {
	return t_recorder;
}

[[nodiscard]] inline size_t GenSlot(uint64_t page) {
	return static_cast<size_t>((page * 0x9E3779B97F4A7C15ull) >> (64u - 12u)) & (kGenSlots - 1);
}

inline void BumpEpoch(Bump kind) {
	g_epoch.fetch_add(1, std::memory_order_seq_cst);
	g_bump_n[kind].fetch_add(1, std::memory_order_relaxed);
}
// A recorder event: the plain counter on Thread_Gpu, the global epoch elsewhere.
inline void BumpRecorder(Bump kind) {
	if (t_recorder) {
		g_rec_epoch++;
		g_bump_n[kind].fetch_add(1, std::memory_order_relaxed);
	} else {
		BumpEpoch(kind);
	}
}

// What a handler means for the repeats: nothing, a store of known extent (its pages), or an event (rec_epoch).
enum class Act : uint8_t { None, Pages, Epoch };
struct HandlerAct {
	Act      act     = Act::None;
	Bump     kind    = BumpKinds; // the counter (BumpKinds: none)
	uint64_t address = 0;
	uint64_t size    = 0;
};

// payload = the dwords after the header, packet_dw = the packet with its header. precise: Precise().
[[nodiscard]] inline HandlerAct ClassifyHandler(uint32_t opcode, uint32_t r, const uint32_t* payload, uint32_t packet_dw, bool async_queue, bool precise) {
	if (opcode == Pm4::IT_WAIT_REG_MEM || opcode == Pm4::IT_WAIT_REG_MEM_64) {
#ifdef KYTY_STREAMDEDUP_MUTANT_WAIT_NO_BUMP
		return {};
#else
		return {Act::Epoch, BumpWait};
#endif
	}
	if (!UploadAsync::HandlerMayWriteGuest(opcode, r)) return {}; // registers, index state, draws, dispatches, cache syncs, markers
#ifdef KYTY_STREAMDEDUP_MUTANT_LABEL_NO_BUMP
	if (opcode == Pm4::IT_RELEASE_MEM) return {};
#endif
	if (precise) {
		switch (opcode) {
			case Pm4::IT_WRITE_DATA:
			case Pm4::IT_DMA_DATA:
			case Pm4::IT_COPY_DATA:
			case Pm4::IT_EVENT_WRITE:
			case Pm4::IT_RELEASE_MEM: {
				const auto w = CpPipe::Checked::ClassifyItem(CpPipe::OpClass::Ordered, opcode, r, payload, packet_dw, async_queue, false);
				if (w.kind == CpPipe::Checked::Kind::None) return {Act::None, BumpNoStore};
#ifdef KYTY_STREAMDEDUP_MUTANT_KNOWN_AS_NONE
				if (w.kind == CpPipe::Checked::Kind::Known) return {Act::None, BumpCpw};
#endif
				if (w.kind == CpPipe::Checked::Kind::Known) return {Act::Pages, BumpCpw, w.address, w.size};
				return {Act::Epoch, BumpPkt};
			}
			default: break;
		}
	}
	return {Act::Epoch, BumpPkt};
}

// Before a PM4 handler runs (every place that calls UploadAsync::BeforeHandler). header = the packet's first dword, payload = the dwords after it,
// max_dw = what the buffer holds from the header on (the packet is KYTY_PM4_LEN(header) dwords, at most max_dw). The callers pass these as they are:
// no comparison or conditional in their bodies (they are hot and keep their PGO hash).
inline void BeforeHandler(uint32_t opcode, uint32_t header, const uint32_t* payload, uint32_t max_dw, bool async_queue);

// CommandProcessor::WaitRegMem, after it read the label (the front evaluates waits on its own thread while the recorder copies).
inline void AfterWaitRead() {
	if (g_active.load(std::memory_order_relaxed) == 0) return;
#ifndef KYTY_STREAMDEDUP_MUTANT_WAIT_NO_BUMP
	BumpRecorder(BumpWait);
#endif
}
// A guest submission or command, before it is queued (guest thread), and every Thread_Gpu loop iteration.
inline void NoteSubmit() {
	if (g_active.load(std::memory_order_relaxed) == 0) return;
#ifndef KYTY_STREAMDEDUP_MUTANT_SUBMIT_NO_BUMP
	BumpRecorder(BumpSub);
#endif
}
// Any thread, before a CPU write of [vaddr, vaddr + size) lands.
inline void NoteCpuWrite(uint64_t vaddr, uint64_t size, Bump kind = BumpWrite) {
	if (g_active.load(std::memory_order_relaxed) == 0 || size == 0) return;
	const uint64_t first = vaddr >> kPageBits;
	const uint64_t last  = (size > UINT64_MAX - vaddr ? UINT64_MAX : vaddr + size - 1) >> kPageBits;
	if (last - first >= kMaxPages) {
#ifndef KYTY_STREAMDEDUP_MUTANT_BIG_NO_BUMP
		BumpEpoch(BumpBig);
#endif
		return;
	}
	for (uint64_t page = first;; page++) {
		g_page_gen[GenSlot(page)].fetch_add(1, std::memory_order_relaxed);
		if (page == last) break;
	}
	g_bump_n[kind].fetch_add(1, std::memory_order_relaxed);
	std::atomic_thread_fence(std::memory_order_seq_cst); // the generations are visible before the write lands
}

inline void BeforeHandler(uint32_t opcode, uint32_t header, const uint32_t* payload, uint32_t max_dw, bool async_queue) {
	if (g_active.load(std::memory_order_relaxed) == 0) return;
	const uint32_t   r         = opcode == Pm4::IT_NOP ? KYTY_PM4_R(header) : 0u;
	const uint32_t   len       = KYTY_PM4_LEN(header);
	const uint32_t   packet_dw = len < max_dw ? len : max_dw;
	const HandlerAct a         = ClassifyHandler(opcode, r, payload, packet_dw, async_queue, Precise());
	switch (a.act) {
		case Act::None:
			if (a.kind != BumpKinds) g_bump_n[a.kind].fetch_add(1, std::memory_order_relaxed);
			return;
		case Act::Pages: NoteCpuWrite(a.address, a.size, a.kind); return;
		case Act::Epoch: BumpRecorder(a.kind); return;
	}
}

// What a copy depended on when it was made (or what a repeat sees now).
struct Snapshot {
	uint64_t tick      = 0;
	uint64_t rec_epoch = 0;
	uint64_t epoch     = 0;
	uint32_t gen0      = 0; // first page
	uint32_t gen1      = 0; // last page (= first when the range is on one page)
};

// false: the range covers more than two pages (never deduplicated).
[[nodiscard]] inline bool TakeSnapshot(uint64_t vaddr, uint64_t size, uint64_t tick, Snapshot& s) {
	if (size == 0 || size > UINT64_MAX - vaddr) return false;
	const uint64_t first = vaddr >> kPageBits;
	const uint64_t last  = (vaddr + size - 1) >> kPageBits;
	if (last - first > 1) return false;
	s.tick      = tick;
	s.rec_epoch = g_rec_epoch;
	s.epoch     = g_epoch.load(std::memory_order_acquire);
	s.gen0      = g_page_gen[GenSlot(first)].load(std::memory_order_acquire);
	s.gen1      = g_page_gen[GenSlot(last)].load(std::memory_order_acquire);
	return true;
}

enum class Verdict : uint8_t { New, Hit, MissTick, MissEpoch, MissPage };

[[nodiscard]] inline Verdict Compare(const Snapshot& then, const Snapshot& now) {
#ifndef KYTY_STREAMDEDUP_MUTANT_TICK_IGNORED
	if (then.tick != now.tick) return Verdict::MissTick;
#endif
#ifndef KYTY_STREAMDEDUP_MUTANT_REC_EPOCH_IGNORED
	if (then.rec_epoch != now.rec_epoch) return Verdict::MissEpoch;
#endif
#ifndef KYTY_STREAMDEDUP_MUTANT_EPOCH_IGNORED
	if (then.epoch != now.epoch) return Verdict::MissEpoch;
#endif
	if (then.gen0 != now.gen0) return Verdict::MissPage;
#ifndef KYTY_STREAMDEDUP_MUTANT_LAST_PAGE_IGNORED
	if (then.gen1 != now.gen1) return Verdict::MissPage;
#endif
	return Verdict::Hit;
}

struct Stats {
	uint64_t look_n = 0, hit_n = 0, hit_bytes = 0, new_n = 0, miss_tick_n = 0, miss_epoch_n = 0, miss_page_n = 0, skip_n = 0;
	uint64_t ne_n = 0, lost_n = 0, caught_n = 0;
};

// The recorder's table. BeforeMap decides before the stream Map; BeforeCopy runs after a successful Map, before the guest bytes are copied.
template <size_t Slots>
class Table {
	static_assert(Slots != 0 && (Slots & (Slots - 1)) == 0, "Slots must be a power of two");

public:
	[[nodiscard]] static size_t SlotOf(uint64_t address, uint64_t size) {
#ifdef KYTY_STREAMDEDUP_MUTANT_SIZE_IGNORED
		(void)size;
		const uint64_t key = address;
#else
		const uint64_t key = address ^ (size * 0x9E3779B97F4A7C15ull);
#endif
		return static_cast<size_t>((key ^ (key >> 17u) ^ (key >> 31u)) & (Slots - 1));
	}

	// Mode::On: true = reuse *offset (no Map, no copy). Mode::Shadow: always false (the copy is made); the verdict waits for BeforeCopy.
	// `recorder`: the calling thread is Thread_Gpu (the table and rec_epoch are its own); otherwise nothing is looked up or stored.
	bool BeforeMap(uint64_t vaddr, uint64_t size, uint64_t tick, Mode mode, bool recorder, uint64_t* offset) {
		m_pending.valid = false;
		m_stats.look_n++;
		Snapshot now;
		if (mode == Mode::Off || !recorder || !TakeSnapshot(vaddr, size, tick, now)) {
			m_stats.skip_n++;
			return false;
		}
		const size_t slot  = SlotOf(vaddr, size);
		Entry&       e     = m_entries[slot];
#ifdef KYTY_STREAMDEDUP_MUTANT_SIZE_IGNORED
		const bool   known = e.used && e.vaddr == vaddr;
#else
		const bool   known = e.used && e.vaddr == vaddr && e.size == size;
#endif
		const Verdict v    = known ? Compare(e.snap, now) : Verdict::New;
		m_pending          = {true, v, slot, vaddr, size, now};
		if (mode == Mode::On) {
			Count(v, size);
			if (v == Verdict::Hit) {
				m_pending.valid = false;
				*offset         = e.offset;
				return true;
			}
		}
		return false;
	}

	// After the Map of a copy that BeforeMap did not reuse. `hash`: the guest bytes (shadow only).
	void BeforeCopy(uint64_t offset, Mode mode, uint64_t hash = 0) {
		if (!m_pending.valid) return;
		m_pending.valid = false;
		Entry& e        = m_entries[m_pending.slot];
		const Verdict v = m_pending.verdict;
		if (mode == Mode::Shadow) {
			Count(v, m_pending.size);
			if (v == Verdict::Hit) {
#ifndef KYTY_STREAMDEDUP_MUTANT_SHADOW_BLIND
				if (e.hash != hash) m_stats.ne_n++;
#endif
#ifndef KYTY_STREAMDEDUP_MUTANT_SHADOW_REPLACES
				return; // mode 1 would bind the entry's copy: keep it
#endif
			} else if (v == Verdict::MissEpoch || v == Verdict::MissPage) {
				if (e.hash == hash) {
					m_stats.lost_n++;
				} else {
					m_stats.caught_n++;
				}
			}
		}
		e = {true, m_pending.vaddr, m_pending.size, m_pending.now, offset, hash};
	}

	[[nodiscard]] Stats&       GetStats() { return m_stats; }
	[[nodiscard]] const Stats& GetStats() const { return m_stats; }

private:
	void Count(Verdict v, uint64_t size) {
		switch (v) {
			case Verdict::Hit:
				m_stats.hit_n++;
				m_stats.hit_bytes += size;
				break;
			case Verdict::New: m_stats.new_n++; break;
			case Verdict::MissTick: m_stats.miss_tick_n++; break;
			case Verdict::MissEpoch: m_stats.miss_epoch_n++; break;
			case Verdict::MissPage: m_stats.miss_page_n++; break;
		}
	}

	struct alignas(64) Entry {
		bool     used   = false;
		uint64_t vaddr  = 0;
		uint64_t size   = 0;
		Snapshot snap {};
		uint64_t offset = 0;
		uint64_t hash   = 0;
	};
	struct Pending {
		bool     valid   = false;
		Verdict  verdict = Verdict::New;
		size_t   slot    = 0;
		uint64_t vaddr   = 0;
		uint64_t size    = 0;
		Snapshot now {};
	};
	Entry   m_entries[Slots] {};
	Pending m_pending {};
	Stats   m_stats {};
};

// The buffer cache's instance (forward-declared in bufferCache.h).
struct State: Table<4096> {};

} // namespace Libs::Graphics::StreamDedup

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMDEDUP_H_ */
