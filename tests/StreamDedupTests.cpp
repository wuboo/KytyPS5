// S614 (KYTY_STREAM_DEDUP, streamDedup.h): a repeated stream copy reuses the earlier copy's offset. Every check ends in exit(1) when it fails, so
// each mutant (CMake stream_dedup_mutant_*, WILL_FAIL) is caught by a failed check.
//
// Mode table; the packet table (ClassifyHandler on real PM4 payloads: which handlers store on known pages, store nothing, or are an event, in both
// rules); hand cases (a repeat in the same tick hits and binds the same offset; a new tick, a CP store on the range's first or last page, a label on
// it, a flip, a wait on the recorder and on another thread, a submission from a guest thread, a write hook, a write of many pages: each makes the
// repeat copy again; an inline packet, a CP store or a label on another page, a cache-flush EVENT_WRITE, a write of 16 pages elsewhere do not; with
// host occlusion queries or before the CP said, a label anywhere is an event; another size at the same address is another range; another thread
// never uses the table; a range on more than two pages is never looked up; shadow: a would-be hit with other bytes is reported, against the copy
// mode 1 would bind).
// Oracle: a CP stream of random bindings of 24 ranges (one page, two pages, the same address with other sizes), inline packets, ordered packets as
// real PM4 payloads whose handler stores what ClassifyItem says (WRITE_DATA, RELEASE_MEM, EVENT_WRITE 0x39 and cache flushes, DMA_DATA, COPY_DATA)
// or anything (flip, MEM_SEMAPHORE), write hooks (fault, backing, big), guest stores that a wait or a submission orders, and ticks (the ring region
// of a tick is reused by the next one), under both rules (pages, strict). Every binding's bytes at the end of its tick, read from the offset the
// table gave, must equal a copy made at the binding; the hits must equal an exact model (no slot or generation aliasing in the set); shadow over the
// same script decides the same, reports no difference, and catches the repeats the rule refused with other bytes.
// Threads: guest threads bump pages, submit and run front-thread waits while the recorder looks up; a write hook that happens-before a lookup
// always makes it miss (TSan on the Mac).
//   TICK_IGNORED, EPOCH_IGNORED, REC_EPOCH_IGNORED, LAST_PAGE_IGNORED, SIZE_IGNORED, WAIT_NO_BUMP, LABEL_NO_BUMP, KNOWN_AS_NONE, PRECISE_ALWAYS,
//   SUBMIT_NO_BUMP, BIG_NO_BUMP, SHADOW_BLIND, SHADOW_REPLACES, DEFAULT_ON: caught.

#include "graphics/host_gpu/renderer/cache/streamDedup.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <tuple>
#include <vector>

namespace {

namespace SD = Libs::Graphics::StreamDedup;
namespace P  = Libs::Graphics::Pm4;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "StreamDedupTests: failed: %s\n", text);
		std::fflush(stderr);
		std::exit(1);
	}
}

void ResetGlobals(SD::Mode mode, bool strict = false, int occlusion = 1) {
	SD::g_epoch.store(0);
	SD::g_rec_epoch = 0;
	for (auto& g: SD::g_page_gen) g.store(0);
	for (auto& b: SD::g_bump_n) b.store(0);
	SD::Activate(mode, strict);
	SD::g_occlusion.store(0);
	if (occlusion != 0) SD::NoteHostOcclusion(occlusion == 2);
	SD::BindRecorderThread();
}

// PM4 packets (header + payload) as the handlers get them; Run() calls the hook as the CP does, before the handler.
struct Pkt {
	uint32_t              opcode = 0, r = 0;
	std::vector<uint32_t> dw; // [0] = header
	uint64_t              store_addr = 0, store_size = 0; // what the handler stores (size 0: nothing; UINT64_MAX: anything, no known extent)
};
uint32_t Header(uint32_t opcode, uint32_t packet_dw, uint32_t r = 0) {
	return (3u << 30u) | ((packet_dw - 2u) << 16u) | (opcode << 8u) | (r << 2u);
}
Pkt WriteData(uint64_t addr, uint32_t ndw) {
	Pkt p {P::IT_WRITE_DATA, 0, {Header(P::IT_WRITE_DATA, 4 + ndw), 0u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u)}, addr, 4ull * ndw};
	for (uint32_t i = 0; i < ndw; i++) p.dw.push_back(0x1000u + i);
	return p;
}
Pkt ReleaseMem(uint64_t addr) { // data_sel 2: a 64-bit label
	return {P::IT_RELEASE_MEM, 0, {Header(P::IT_RELEASE_MEM, 8), 0x14u, 2u << 29u, static_cast<uint32_t>(addr) & ~3u, static_cast<uint32_t>(addr >> 32u), 7u, 0u, 0u}, addr, 8};
}
Pkt EventFlush() {
	return {P::IT_EVENT_WRITE, 0, {Header(P::IT_EVENT_WRITE, 2), 0x16u}, 0, 0};
}
Pkt EventOcclusion(uint64_t addr) {
	return {P::IT_EVENT_WRITE, 0, {Header(P::IT_EVENT_WRITE, 4), 0x39u, static_cast<uint32_t>(addr), static_cast<uint32_t>(addr >> 32u)}, addr,
	        Libs::Graphics::CpPipe::Checked::kOcclusionDumpBytes};
}
Pkt DmaData(uint64_t dst, uint32_t n) {
	return {P::IT_DMA_DATA, 0, {Header(P::IT_DMA_DATA, 7), 0u, 0u, 0u, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u), n}, dst, n};
}
Pkt CopyData(uint64_t dst, bool wide) {
	return {P::IT_COPY_DATA, 0, {Header(P::IT_COPY_DATA, 6), wide ? 1u << 16u : 0u, 0u, 0u, static_cast<uint32_t>(dst), static_cast<uint32_t>(dst >> 32u)}, dst,
	        wide ? 8u : 4u};
}
Pkt Flip() {
	return {P::IT_NOP, P::R_FLIP, {Header(P::IT_NOP, 3, P::R_FLIP), 0u, 0u}, 0, UINT64_MAX};
}
Pkt Semaphore() {
	return {P::IT_MEM_SEMAPHORE, 0, {Header(P::IT_MEM_SEMAPHORE, 4), 0u, 0u, 0u}, 0, UINT64_MAX};
}
Pkt Inline(uint32_t opcode, uint32_t r = 0) {
	return {opcode, r, {Header(opcode, 3, r), 0u, 0u}, 0, 0};
}
void Run(const Pkt& p) {
	SD::BeforeHandler(p.opcode, p.dw[0], p.dw.data() + 1, static_cast<uint32_t>(p.dw.size()) + 16u, false); // the buffer goes on after the packet
}
SD::HandlerAct Classify(const Pkt& p, bool precise) {
	return SD::ClassifyHandler(p.opcode, p.r, p.dw.data() + 1, static_cast<uint32_t>(p.dw.size()), false, precise);
}

constexpr uint64_t kPage  = uint64_t {1} << SD::kPageBits;
constexpr uint64_t kBase  = uint64_t {0x2000} << SD::kPageBits; // page-aligned
constexpr uint64_t kPages = 40;

uint64_t Fnv(const uint8_t* p, uint64_t n) {
	uint64_t h = 1469598103934665603ull, w = 0;
	uint64_t i = 0;
	for (; i + 8 <= n; i += 8) {
		std::memcpy(&w, p + i, 8);
		h = (h ^ w) * 1099511628211ull;
	}
	for (; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
	return h ^ n;
}

using Table = SD::Table<4096>;

void ModeTable() {
	Check(SD::ParseMode(nullptr) == SD::Mode::Off, "unset -> off");
	Check(SD::ParseMode("") == SD::Mode::Off, "empty -> off");
	Check(SD::ParseMode("0") == SD::Mode::Off, "0 -> off");
	Check(SD::ParseMode("yes") == SD::Mode::Off, "unknown -> off");
	Check(SD::ParseMode("1") == SD::Mode::On && SD::ParseMode("on") == SD::Mode::On, "1/on -> on");
	Check(SD::ParseMode("shadow") == SD::Mode::Shadow, "shadow");
	Check(!SD::ParseConfig("1").strict && SD::ParseConfig("strict").mode == SD::Mode::On && SD::ParseConfig("strict").strict &&
	          SD::ParseConfig("shadow_strict").mode == SD::Mode::Shadow && SD::ParseConfig("shadow_strict").strict && !SD::ParseConfig("shadow").strict,
	      "strict variants");
	Check(std::strcmp(SD::ModeName(SD::Mode::Off), "off") == 0 && std::strcmp(SD::ModeName(SD::Mode::On), "on") == 0 &&
	          std::strcmp(SD::ModeName(SD::Mode::Shadow), "shadow") == 0,
	      "mode names");
}

void PacketTable() {
	const uint64_t a = kBase + 0x1230;
	for (bool precise: {true, false}) {
		for (const Pkt& p: {Inline(P::IT_SET_SH_REG), Inline(P::IT_SET_CONTEXT_REG), Inline(P::IT_SET_UCONFIG_REG), Inline(P::IT_INDEX_TYPE),
		                    Inline(P::IT_NUM_INSTANCES), Inline(P::IT_DRAW_INDEX_2), Inline(P::IT_DRAW_INDEX_AUTO), Inline(P::IT_DRAW_INDEX_OFFSET_2),
		                    Inline(P::IT_DISPATCH_DIRECT), Inline(P::IT_DISPATCH_INDIRECT), Inline(P::IT_ACQUIRE_MEM), Inline(P::IT_PFP_SYNC_ME),
		                    Inline(P::IT_INDIRECT_BUFFER), Inline(P::IT_NOP, P::R_ZERO), Inline(P::IT_NOP, P::R_PUSH_MARKER), Inline(P::IT_NOP, P::R_POP_MARKER)}) {
			const auto c = Classify(p, precise);
			Check(c.act == SD::Act::None && c.kind == SD::BumpKinds, "register / draw / dispatch / cache / marker packets: nothing");
		}
		for (const Pkt& p: {Inline(P::IT_WAIT_REG_MEM), Inline(P::IT_WAIT_REG_MEM_64)}) {
			const auto c = Classify(p, precise);
			Check(c.act == SD::Act::Epoch && c.kind == SD::BumpWait, "a wait is an event");
		}
		for (const Pkt& p: {Flip(), Semaphore(), Inline(P::IT_DRAW_INDIRECT)}) {
			Check(Classify(p, precise).act == SD::Act::Epoch, "a flip / a drain-class packet: an event");
		}
	}
	for (const Pkt& p: {WriteData(a, 3), ReleaseMem(a + 8), EventOcclusion(a), DmaData(a, 100), CopyData(a, true), CopyData(a + 4, false)}) {
		const auto c = Classify(p, true);
		Check(c.act == SD::Act::Pages && c.address == p.store_addr && c.size == p.store_size && c.kind == SD::BumpCpw, "a CP store of known extent: its pages");
		Check(Classify(p, false).act == SD::Act::Epoch, "strict: an event");
	}
	Check(Classify(EventFlush(), true).act == SD::Act::None && Classify(EventFlush(), true).kind == SD::BumpNoStore, "a cache-flush EVENT_WRITE stores nothing");
	Check(Classify(EventFlush(), false).act == SD::Act::Epoch, "strict: a cache flush is an event");
	Pkt bad = WriteData(a, 1);
	bad.dw.resize(4); // shorter than a WRITE_DATA can be
	Check(Classify(bad, true).act == SD::Act::Epoch, "an ordered packet of unknown extent: an event");
	// Precise(): only with the pages rule and after the CP said there are no host occlusion queries.
	ResetGlobals(SD::Mode::On, false, 1);
	Check(SD::Precise(), "pages, no host occlusion: extents");
	ResetGlobals(SD::Mode::On, false, 2);
	Check(!SD::Precise(), "host occlusion queries: no extents");
	ResetGlobals(SD::Mode::On, false, 0);
	Check(!SD::Precise(), "the CP has not said: no extents");
	ResetGlobals(SD::Mode::On, true, 1);
	Check(!SD::Precise(), "strict: no extents");
}

// One binding through the table as BufferCache::StreamMapDedup / StreamDedupBeforeCopy do it. Returns the offset; *hit set when reused.
struct Sim {
	std::unique_ptr<Table> table = std::make_unique<Table>();
	uint64_t               tick  = 1;
	uint64_t               next  = 0; // the next free ring offset of this tick
	SD::Mode               mode  = SD::Mode::On;

	uint64_t Bind(uint64_t vaddr, uint64_t size, uint64_t hash, bool* hit, bool recorder = true) {
		uint64_t off = 0;
		*hit         = table->BeforeMap(vaddr, size, tick, mode, recorder, &off);
		if (*hit) return off;
		off = next;
		next += (size + 255u) & ~uint64_t {255};
		table->BeforeCopy(off, mode, hash);
		return off;
	}
	void NewTick() {
		tick++;
		next = 0;
	}
};

void OffThread(void (*f)()) {
	SD::t_recorder = false;
	f();
	SD::t_recorder = true;
}

void HandCases() {
	ResetGlobals(SD::Mode::On);
	Sim  s;
	bool hit = false;
	const uint64_t a = kBase + 0x100, b = kBase + kPage - 0x80; // b crosses into the second page
	const uint64_t o1 = s.Bind(a, 256, 1, &hit);
	Check(!hit, "first copy");
	Check(s.Bind(a, 256, 1, &hit) == o1 && hit, "a repeat in the same tick binds the same offset");
	Check(s.table->GetStats().hit_n == 1 && s.table->GetStats().hit_bytes == 256 && s.table->GetStats().new_n == 1, "hit counted");
	(void)s.Bind(a, 128, 1, &hit);
	Check(!hit, "another size at the same address is another range");
	(void)s.Bind(a, 256, 1, &hit); // the 128-byte range may have taken the slot: copy again then
	(void)s.Bind(a, 256, 1, &hit);
	Check(hit, "the 256-byte range is known again");

	Run(Inline(P::IT_SET_SH_REG));
	Run(Inline(P::IT_DRAW_INDEX_2));
	Run(Inline(P::IT_ACQUIRE_MEM));
	Run(Inline(P::IT_NOP, P::R_PUSH_MARKER));
	Run(EventFlush());
	(void)s.Bind(a, 256, 1, &hit);
	Check(hit, "inline packets and a cache flush keep the copy");
	Run(WriteData(kBase + 6 * kPage, 4));
	Run(ReleaseMem(kBase + 7 * kPage + 64));
	Run(EventOcclusion(kBase + 9 * kPage));
	Run(DmaData(kBase + 10 * kPage, 4096));
	Run(CopyData(kBase + 11 * kPage, true));
	(void)s.Bind(a, 256, 1, &hit);
	Check(hit, "CP stores on other pages keep the copy");
	Check(SD::g_bump_n[SD::BumpCpw].load() == 5 && SD::g_bump_n[SD::BumpNoStore].load() == 1, "CP stores counted");

	Run(WriteData(a + 0x2000, 1)); // a's page, not a's bytes: the page decides
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit && s.table->GetStats().miss_page_n == 1, "WRITE_DATA on a's page: copy again");
	(void)s.Bind(a, 256, 1, &hit);
	Check(hit, "and the new copy is reused");
	Run(ReleaseMem(a + 64));
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "a RELEASE_MEM label on a: copy again");
	(void)s.Bind(a, 256, 1, &hit);
	Run(DmaData(a - 64, 128));
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "DMA_DATA over a: copy again");
	(void)s.Bind(a, 256, 1, &hit);
	Run(Flip());
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit && s.table->GetStats().miss_epoch_n == 1, "a flip (no known extent): copy again");
	(void)s.Bind(a, 256, 1, &hit);
	Run(Semaphore());
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "a drain-class packet: copy again");
	Run(Inline(P::IT_WAIT_REG_MEM));
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "a wait on the recorder: copy again");
	OffThread([] { SD::AfterWaitRead(); });
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "a wait the front read: copy again");
	OffThread([] { SD::NoteSubmit(); });
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "a guest submission: copy again");
	SD::NoteSubmit(); // a Thread_Gpu loop iteration
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "a new loop iteration: copy again");
	OffThread([] { Run(Flip()); });
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "an event on another thread: copy again");
	OffThread([] { Run(WriteData(kBase + 0x180, 1)); });
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit, "a CP store on another thread (its pages): copy again");

	// Without the extents (host occlusion queries, or before the CP said): a label anywhere is an event.
	for (int occlusion: {2, 0}) {
		SD::g_occlusion.store(0);
		if (occlusion != 0) SD::NoteHostOcclusion(true);
		(void)s.Bind(a, 256, 1, &hit);
		(void)s.Bind(a, 256, 1, &hit);
		Check(hit, "a reused");
		Run(ReleaseMem(kBase + 7 * kPage + 64));
		(void)s.Bind(a, 256, 1, &hit);
		Check(!hit, "no extents: a label elsewhere is an event");
	}
	SD::NoteHostOcclusion(false);

	// Pages: b is on pages 0 and 1 of kBase.
	(void)s.Bind(b, 256, 2, &hit);
	(void)s.Bind(b, 256, 2, &hit);
	Check(hit, "b reused");
	SD::NoteCpuWrite(kBase + 5 * kPage, 64);
	(void)s.Bind(b, 256, 2, &hit);
	Check(hit, "a write on another page keeps b");
	SD::NoteCpuWrite(kBase + kPage + 16, 4);
	(void)s.Bind(b, 256, 2, &hit);
	Check(!hit && s.table->GetStats().miss_page_n >= 1, "a write on b's last page: copy again");
	(void)s.Bind(b, 256, 2, &hit);
	SD::NoteCpuWrite(kBase + kPage - 8, 4);
	(void)s.Bind(b, 256, 2, &hit);
	Check(!hit, "a write on b's first page: copy again");
	(void)s.Bind(b, 256, 2, &hit);
	SD::NoteCpuWrite(kBase + 8 * kPage, 16 * kPage); // 16 pages: page by page
	(void)s.Bind(b, 256, 2, &hit);
	Check(hit, "16 pages elsewhere keep b");
	SD::NoteCpuWrite(kBase + 8 * kPage, 16 * kPage + 1); // 17 pages: the global epoch
	(void)s.Bind(b, 256, 2, &hit);
	Check(!hit, "a write of many pages anywhere: copy again");
	Check(SD::g_bump_n[SD::BumpBig].load() == 1 && SD::g_bump_n[SD::BumpWrite].load() == 4, "write bumps counted");

	(void)s.Bind(a, 256, 1, &hit);
	(void)s.Bind(a, 256, 1, &hit);
	Check(hit, "a reused");
	s.NewTick();
	(void)s.Bind(a, 256, 1, &hit);
	Check(!hit && s.table->GetStats().miss_tick_n == 1, "a new tick: copy again");

	const auto skip0 = s.table->GetStats().skip_n;
	(void)s.Bind(a, 256, 1, &hit, false);
	Check(!hit && s.table->GetStats().skip_n == skip0 + 1, "another thread: no lookup");
	(void)s.Bind(a, 256, 1, &hit);
	Check(hit, "another thread stored nothing (the recorder's copy is still the one)");
	(void)s.Bind(kBase + 3 * kPage + 64, 2 * kPage, 3, &hit);
	(void)s.Bind(kBase + 3 * kPage + 64, 2 * kPage, 3, &hit);
	Check(!hit && s.table->GetStats().skip_n == skip0 + 3, "three pages: never looked up");

	// Off: nothing at all.
	Sim off;
	off.mode = SD::Mode::Off;
	(void)off.Bind(a, 256, 1, &hit);
	(void)off.Bind(a, 256, 1, &hit);
	Check(!hit, "off never reuses");
}

void ShadowHandCases() {
	ResetGlobals(SD::Mode::Shadow);
	Sim s;
	s.mode   = SD::Mode::Shadow;
	bool hit = false;
	const uint64_t a = kBase + 0x400;
	const uint64_t o1 = s.Bind(a, 512, 11, &hit);
	const uint64_t o2 = s.Bind(a, 512, 11, &hit);
	Check(!hit && o2 != o1, "shadow copies every time");
	Check(s.table->GetStats().hit_n == 1 && s.table->GetStats().ne_n == 0, "a would-be hit with the same bytes");
	(void)s.Bind(a, 512, 12, &hit); // a store nothing was told about
	Check(s.table->GetStats().ne_n == 1, "a would-be hit with other bytes is reported");
	(void)s.Bind(a, 512, 12, &hit);
	Check(s.table->GetStats().ne_n == 2, "compared with the copy mode 1 would bind (the first), not the last");
	Run(WriteData(a + 16, 2));
	(void)s.Bind(a, 512, 13, &hit);
	Check(s.table->GetStats().caught_n == 1 && s.table->GetStats().ne_n == 2, "a refused repeat with other bytes");
	Run(Flip());
	(void)s.Bind(a, 512, 13, &hit);
	Check(s.table->GetStats().lost_n == 1, "a refused repeat with the same bytes");
}

// The oracle: random CP streams.
struct Range {
	uint64_t vaddr, size;
};

std::vector<Range> MakeRanges(std::mt19937_64& rng) {
	std::vector<Range> out;
	std::set<size_t>   slots;
	const auto add = [&](uint64_t v, uint64_t n) {
		if (n == 0 || n > kPage || v < kBase || v + n > kBase + 8 * kPage) return;
		if (!slots.insert(Table::SlotOf(v, n)).second) return;
		out.push_back({v, n});
	};
	add(kBase + kPage - 128, 256);          // two pages
	add(kBase + 2 * kPage - 64, kPage);     // two pages, a full page long
	add(kBase + 3 * kPage + 512, 1024);     // same address, three sizes
	add(kBase + 3 * kPage + 512, 512);
	add(kBase + 3 * kPage + 512, 64);
	add(kBase + 5 * kPage, kPage);          // a whole page
	const uint64_t sizes[] = {64, 256, 1024, 4096};
	while (out.size() < 24) {
		const uint64_t n = sizes[rng() % 4];
		const uint64_t v = kBase + (rng() % (8 * kPage - n)) / 64 * 64;
		add(v, n);
	}
	return out;
}

struct Oracle {
	std::vector<uint8_t> mem = std::vector<uint8_t>(kPages * kPage);
	std::mt19937_64      rng;
	Sim                  sim;
	std::vector<Range>   ranges;
	std::vector<uint8_t> ring = std::vector<uint8_t>(64u << 20);
	struct Bound {
		uint64_t             off;
		std::vector<uint8_t> bytes;
	};
	std::vector<Bound> bound;
	// The exact model: per range, the state the last copy saw.
	uint64_t events = 0;
	uint64_t pw[kPages] {};
	struct Last {
		bool     valid = false;
		uint64_t tick = 0, events = 0, pw0 = 0, pw1 = 0;
	};
	std::map<std::pair<uint64_t, uint64_t>, Last> last;
	uint64_t ideal = 0, hits = 0, binds = 0;
	bool     exact  = true;
	bool     strict = false;

	explicit Oracle(uint64_t seed, SD::Mode mode, bool strict_rule): rng(seed), strict(strict_rule) {
		for (auto& b: mem) b = static_cast<uint8_t>(rng());
		sim.mode = mode;
		ranges   = MakeRanges(rng);
		std::set<size_t> gens;
		for (uint64_t p = 0; p < kPages; p++) gens.insert(SD::GenSlot((kBase >> SD::kPageBits) + p));
		exact = gens.size() == kPages && ranges.size() == 24;
	}
	uint8_t* At(uint64_t vaddr) { return mem.data() + (vaddr - kBase); }
	uint64_t PageOf(uint64_t vaddr) const { return (vaddr - kBase) >> SD::kPageBits; }

	void Store(uint64_t vaddr, uint64_t n) {
		for (uint64_t i = 0; i < n; i++) At(vaddr)[i] = static_cast<uint8_t>(rng());
	}
	// A small store that often lands inside a bound range (and on either of its pages).
	std::pair<uint64_t, uint64_t> SmallTarget() {
		if (rng() % 10 < 8) {
			const auto& r = ranges[rng() % ranges.size()];
			const uint64_t n = 1 + rng() % 8;
			const uint64_t v = r.vaddr + rng() % (r.size - std::min<uint64_t>(r.size, n) + 1);
			return {v, std::min<uint64_t>(n, kBase + kPages * kPage - v)};
		}
		const uint64_t v = kBase + rng() % (8 * kPage - 64);
		return {v, 1 + rng() % 64};
	}
	void ModelEvent() { events++; }
	// A random ordered packet; its store often lands on a bound range.
	Pkt Ordered() {
		const auto [v, n] = SmallTarget();
		switch (rng() % 9) {
			case 0: return WriteData(v & ~uint64_t {3}, 1 + static_cast<uint32_t>(rng() % 4));
			case 1:
			case 2: return ReleaseMem(v & ~uint64_t {7});
			case 3: return EventOcclusion(v & ~uint64_t {7});
			case 4: return EventFlush();
			case 5: return DmaData(v, 1 + static_cast<uint32_t>(rng() % 64));
			case 6: return CopyData(v & ~uint64_t {7}, rng() % 2 == 0);
			case 7: return Flip();
			default: return Semaphore();
		}
	}
	void ModelPages(uint64_t v, uint64_t n) {
		for (uint64_t p = PageOf(v); p <= PageOf(v + n - 1); p++) pw[p]++;
	}

	void Bind() {
		const auto& r   = ranges[rng() % ranges.size()];
		auto&       l   = last[{r.vaddr, r.size}];
		const uint64_t p0 = PageOf(r.vaddr), p1 = PageOf(r.vaddr + r.size - 1);
		const bool  want = l.valid && l.tick == sim.tick && l.events == events && l.pw0 == pw[p0] && l.pw1 == pw[p1];
		std::vector<uint8_t> bytes(At(r.vaddr), At(r.vaddr) + r.size);
		bool        hit = false;
		const uint64_t off = sim.Bind(r.vaddr, r.size, Fnv(bytes.data(), r.size), &hit);
		binds++;
		if (want) ideal++;
		if (sim.mode == SD::Mode::Shadow) {
			Check(!hit, "shadow never reuses");
		}
		if (hit) {
			hits++;
		} else {
			Check(off + r.size <= ring.size(), "ring");
			std::memcpy(ring.data() + off, bytes.data(), r.size);
			l = {true, sim.tick, events, pw[p0], pw[p1]};
		}
		bound.push_back({off, std::move(bytes)});
	}
	void EndTick() {
		for (const auto& b: bound) {
			Check(b.off + b.bytes.size() <= sim.next, "a binding reads a region of its own tick");
			Check(std::memcmp(ring.data() + b.off, b.bytes.data(), b.bytes.size()) == 0, "the GPU reads what a copy at the binding held");
		}
		bound.clear();
		std::memset(ring.data(), 0xcd, sim.next); // the GPU is past this tick: its region may hold anything now
		sim.NewTick(); // the next tick writes the same ring region again
	}

	void Step() {
		const unsigned k = static_cast<unsigned>(rng() % 100);
		if (k < 75) {
			Bind();
		} else if (k < 82) {
			const uint32_t ops[] = {P::IT_SET_SH_REG, P::IT_SET_CONTEXT_REG, P::IT_DRAW_INDEX_2, P::IT_DISPATCH_DIRECT, P::IT_ACQUIRE_MEM};
			Run(Inline(ops[rng() % 5]));
		} else if (k < 87) { // an ordered CP packet, in stream order: its handler stores what the packet says (a flip / a semaphore: anything)
			const Pkt pk = Ordered();
			Run(pk);
			if (pk.store_size == UINT64_MAX) {
				ModelEvent();
				const auto [v, n] = SmallTarget();
				Store(v, n);
			} else {
				if (strict) {
					ModelEvent();
				} else if (pk.store_size != 0) {
					ModelPages(pk.store_addr, pk.store_size);
				}
				if (pk.store_size != 0) Store(pk.store_addr, pk.store_size);
			}
		} else if (k < 89) { // a write hook (fault, backing write) before its store; any thread
			const auto [v, n] = SmallTarget();
			SD::t_recorder = rng() % 2 == 0;
			SD::NoteCpuWrite(v, n);
			SD::t_recorder = true;
			ModelPages(v, n);
			Store(v, n);
		} else if (k < 90) { // a big write (file read, unmap)
			const uint64_t v = kBase + (rng() % 4) * kPage + rng() % 1024;
			const uint64_t n = (17 + rng() % 4) * kPage;
			SD::NoteCpuWrite(v, n);
			ModelEvent();
			const uint64_t span = std::min<uint64_t>(n, kBase + kPages * kPage - v);
			for (int i = 0; i < 64; i++) Store(v + rng() % span, 1); // a few bytes anywhere in it (the hook covers all of it)
		} else if (k < 93) { // the guest stores on a writable page, then a wait reads its label: on the recorder or on the front
			const auto [v, n] = SmallTarget();
			Store(v, n);
			if (rng() % 2 == 0) {
				Run(Inline(P::IT_WAIT_REG_MEM));
				SD::AfterWaitRead();
			} else {
				SD::t_recorder = false;
				Run(Inline(rng() % 2 == 0 ? P::IT_WAIT_REG_MEM : P::IT_WAIT_REG_MEM_64));
				SD::AfterWaitRead();
				SD::t_recorder = true;
			}
			ModelEvent();
		} else if (k < 96) { // the guest stores, then submits (guest thread) / a new loop iteration
			const auto [v, n] = SmallTarget();
			Store(v, n);
			SD::t_recorder = rng() % 2 == 0;
			SD::NoteSubmit();
			SD::t_recorder = true;
			ModelEvent();
		} else {
			EndTick();
		}
	}
};

void OracleRun(uint64_t seed, int steps, bool strict) {
	ResetGlobals(SD::Mode::On, strict);
	Oracle on(seed, SD::Mode::On, strict);
	for (int i = 0; i < steps; i++) on.Step();
	on.EndTick();
	Check(on.hits <= on.ideal, "never more hits than the exact model");
	if (on.exact) Check(on.hits == on.ideal, "no aliasing: exactly the hits of the exact model");
	Check(on.hits * 10 > on.binds, "the stream reuses a fair share");
	const auto& st = on.sim.table->GetStats();
	Check(st.hit_n == on.hits && st.look_n == on.binds && st.ne_n == 0, "stats of mode 1");
	Check(st.miss_epoch_n > 0 && st.miss_page_n > 0 && st.miss_tick_n > 0, "every kind of miss happens");
	Check(strict == (SD::g_bump_n[SD::BumpCpw].load() == 0), "CP stores of known extent: pages (rule pages) / events (strict)");

	ResetGlobals(SD::Mode::Shadow, strict);
	Oracle sh(seed, SD::Mode::Shadow, strict);
	for (int i = 0; i < steps; i++) sh.Step();
	sh.EndTick();
	const auto& ss = sh.sim.table->GetStats();
	Check(ss.hit_n == on.hits, "shadow decides as mode 1");
	Check(ss.ne_n == 0, "shadow: no difference on a stream without unordered stores");
	Check(ss.caught_n > 0, "shadow: the rule refused repeats with other bytes");
	std::fprintf(stderr, "StreamDedupTests: seed %llu rule %s: binds %llu hits %llu (ideal %llu, exact %d) caught %llu lost %llu\n",
	             static_cast<unsigned long long>(seed), strict ? "strict" : "pages", static_cast<unsigned long long>(on.binds),
	             static_cast<unsigned long long>(on.hits), static_cast<unsigned long long>(on.ideal), on.exact ? 1 : 0,
	             static_cast<unsigned long long>(ss.caught_n), static_cast<unsigned long long>(ss.lost_n));
}

// Threads: a write hook that happens-before a lookup makes it miss; noise from guest threads (other pages, submissions, front waits).
void Threads() {
	ResetGlobals(SD::Mode::On);
	Sim                   s;
	std::atomic<int64_t>  request {-1};
	std::atomic<uint64_t> ack {0};
	std::atomic<bool>     stop {false};
	std::thread           writer([&] {
		uint64_t done = 0;
		while (!stop.load(std::memory_order_acquire)) {
			const int64_t p = request.load(std::memory_order_acquire);
			if (p < 0) continue;
			request.store(-1, std::memory_order_relaxed);
			SD::NoteCpuWrite(kBase + static_cast<uint64_t>(p) * kPage + 100, 8);
			ack.store(++done, std::memory_order_release);
		}
	});
	std::vector<std::thread> noise;
	for (int t = 0; t < 3; t++) {
		noise.emplace_back([&, t] {
			std::mt19937_64 rng(77 + t);
			while (!stop.load(std::memory_order_acquire)) {
				switch (rng() % 3) {
					case 0: SD::NoteCpuWrite(kBase + (20 + rng() % 16) * kPage, 64); break; // pages the recorder never binds
					case 1: Run(Inline(P::IT_SET_SH_REG)); break;
					default: SD::NoteCpuWrite(kBase + (20 + rng() % 4) * kPage, 3 * kPage); break;
				}
			}
		});
	}
	std::mt19937_64 rng(5);
	uint64_t        expect = 0, hits = 0;
	for (int i = 0; i < 1000; i++) {
		const uint64_t p = rng() % 8;
		const uint64_t v = kBase + p * kPage + 64;
		bool           hit = false;
		(void)s.Bind(v, 512, 0, &hit);
		(void)s.Bind(v, 512, 0, &hit);
		hits += hit ? 1 : 0;
		request.store(static_cast<int64_t>(p), std::memory_order_release);
		while (ack.load(std::memory_order_acquire) != expect + 1) {
		}
		expect++;
		(void)s.Bind(v, 512, 0, &hit);
		Check(!hit, "a write hook that happened before the lookup: copy again");
	}
	stop.store(true, std::memory_order_release);
	writer.join();
	for (auto& t: noise) t.join();
	Check(hits > 750, "without a write in between the repeat is reused");
}

} // namespace

int main() {
#ifndef KYTY_STREAMDEDUP_ORACLE_ONLY
	ModeTable();
	PacketTable();
	HandCases();
	ShadowHandCases();
#endif
	for (uint64_t seed: {1ull, 2ull, 3ull, 4ull}) {
		OracleRun(seed, 200000, false);
		OracleRun(seed, 200000, true);
	}
	Threads();
	SD::Activate(SD::Mode::Off);
	std::fprintf(stderr, "StreamDedupTests: ok\n");
	return 0;
}
