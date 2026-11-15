// S609 (KYTY_CP_PIPE_SIGNALS=checked): the pure part of the checked policy (cpPipeChecked.h). Every check ends in exit(1) when it fails (a watchdog
// thread exits with 1 when the oracle does not finish), so each mutant (CMake cp_pipe_checked_mutant_*, WILL_FAIL) is caught by a failed check:
//   NO_CAPTURE           - a job carries no ranges (the tracker table; the oracle: a store of an item still in the FIFO lands on a page a used job read);
//   UNKNOWN_NO_GATE      - an item of unknown extent holds nothing (the tracker table; the oracle: the flip's store is not ordered);
//   JOIN_IGNORES_WRITES  - Overlaps is always false (the overlap table; the oracle);
//   PRUNE_NEXT           - the ranges of item done + 1 are dropped as if it had run (the tracker table; the oracle);
//   RELEASE_NONE         - RELEASE_MEM writes nothing (the classification table; the oracle: label stores);
//   OCCLUSION_SMALL      - the synthetic occlusion dump is one qword, not 16 DB pairs (the classification table; the oracle: dumps across a page end);
//   HOST_OCCLUSION_KNOWN - with host occlusion queries an ordered item still has its own extent (the classification table);
//   LAST_PAGE            - a range's last page is not checked (the overlap table; the oracle: stores across a page end);
//   DMA_NONE             - DMA_DATA writes nothing (the classification table; the oracle).
// The oracle runs the protocol of cpPipe.cpp on threads: a front that builds real PM4 payloads, classifies them (ClassifyItem), captures the ranges of a
// draw's job and holds it behind an item of unknown extent (Tracker), a recorder that executes the items in order (CPU stores into a page-versioned memory,
// dispatches that mark pages GPU-dirty, downloads that publish them with a write log), releases held jobs after their gate and joins every draw (dirty pages,
// the write log, Overlaps), and three workers that read the pages of a job at any time after it was handed out. A job the join uses must have seen every
// page at the version the recorder sees at the draw (what the CP's own walk would read).
// -DKYTY_CPCHK_ORACLE_ONLY builds the oracle alone (a control of what it catches without the tables).

#include "graphics/guest_gpu/command_processor/cpPipeChecked.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CpPipe;
namespace K = Libs::Graphics::CpPipe::Checked;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpPipeCheckedTests: failed: %s\n", text);
		std::fflush(stderr);
		std::exit(1);
	}
}

// --- PM4 payloads (the dwords after the header) ---
struct Packet {
	OpClass               cls    = OpClass::Ordered;
	uint32_t              opcode = 0;
	uint32_t              r      = 0;
	std::vector<uint32_t> payload;
	uint32_t              packet_dw = 0; // with the header
};

uint32_t Lo(uint64_t v) { return static_cast<uint32_t>(v); }
uint32_t Hi(uint64_t v) { return static_cast<uint32_t>(v >> 32u); }

Packet WriteData(uint64_t address, uint32_t dwords, bool one_address) {
	Packet p {OpClass::Ordered, Pm4::IT_WRITE_DATA, 0, {one_address ? (1u << 16u) : 0u, Lo(address), Hi(address)}, 0};
	for (uint32_t i = 0; i < dwords; i++) p.payload.push_back(0x1000u + i);
	p.packet_dw = static_cast<uint32_t>(p.payload.size()) + 1u;
	return p;
}
Packet DmaData(uint64_t dst, uint32_t bytes) {
	return {OpClass::Ordered, Pm4::IT_DMA_DATA, 0, {0u, 0x1000u, 0u, Lo(dst), Hi(dst), bytes & 0x03ffffffu}, 7u};
}
Packet CopyData(uint64_t dst, bool wide) {
	return {OpClass::Ordered, Pm4::IT_COPY_DATA, 0, {wide ? (1u << 16u) : 0u, 0x2000u, 0u, Lo(dst), Hi(dst)}, 6u};
}
Packet ReleaseMem(uint64_t address, uint32_t data_sel, uint32_t interrupt) {
	return {OpClass::Ordered, Pm4::IT_RELEASE_MEM, 0, {0u, (data_sel << 29u) | (interrupt << 24u), Lo(address), Hi(address), 0x55u, 0u, 0u}, 8u};
}
Packet OcclusionDump(uint64_t address) {
	return {OpClass::Ordered, Pm4::IT_EVENT_WRITE, 0, {0x39u | (1u << 8u), Lo(address), Hi(address)}, 4u};
}
Packet CacheEvent() { return {OpClass::Ordered, Pm4::IT_EVENT_WRITE, 0, {0x16u}, 2u}; }
Packet Flip() { return {OpClass::Ordered, Pm4::IT_NOP, Pm4::R_FLIP, {0u, 0u, 0u, 0u, 0u}, 6u}; }

K::ItemWrite Classify(const Packet& p, bool host_occlusion = false) {
	return K::ClassifyItem(p.cls, p.opcode, p.r, p.payload.data(), p.packet_dw, false, host_occlusion);
}

bool Is(const K::ItemWrite& w, K::Kind kind, uint64_t address, uint64_t size) {
	return w.kind == kind && (kind != K::Kind::Known || (w.address == address && w.size == size));
}

// --- tables ---
void PolicyTests() {
	Check(ParseSignalPolicy("checked") == SignalPolicy::Checked && ParseSignalPolicy("checked_shadow") == SignalPolicy::Checked, "checked parses");
	Check(ParseSignalPolicy(nullptr) == SignalPolicy::Fifo && ParseSignalPolicy("checkedx") == SignalPolicy::Fifo, "default and unknown text are fifo");
	Check(ParseCheckedShadow("checked_shadow") && !ParseCheckedShadow("checked") && !ParseCheckedShadow(nullptr), "the shadow flag");
	Check(std::string_view(SignalPolicyName(SignalPolicy::Checked, false)) == "checked" && std::string_view(SignalPolicyName(SignalPolicy::Checked, true)) == "checked_shadow",
	      "names");
	GateTracker gates;
	gates.NoteItem(OpClass::Dispatch, Pm4::IT_DISPATCH_DIRECT, 4);
	Check(gates.GateFor(SignalPolicy::Checked, 3) == 4 && gates.GateFor(SignalPolicy::Fifo, 3) == 4, "GateTracker answers checked as fifo");
}

void ClassifyTests() {
	constexpr uint64_t a = 0x10002ff0ull;
	Check(Is(Classify(WriteData(a, 3, false)), K::Kind::Known, a, 12), "WRITE_DATA: its dwords");
	Check(Is(Classify(WriteData(a, 3, true)), K::Kind::Known, a, 4), "WRITE_DATA to one address: one dword");
	Check(Is(Classify(DmaData(a, 0x2400)), K::Kind::Known, a, 0x2400), "DMA_DATA: its destination");
	Check(Classify(DmaData(a, 0)).kind == K::Kind::None, "DMA_DATA of 0 bytes: nothing");
	Check(Is(Classify(CopyData(a, false)), K::Kind::Known, a, 4) && Is(Classify(CopyData(a, true)), K::Kind::Known, a, 8), "COPY_DATA: 4 or 8 bytes");
	Check(Is(Classify(ReleaseMem(a, 1, 0)), K::Kind::Known, a, 4), "RELEASE_MEM 32-bit label");
	Check(Is(Classify(ReleaseMem(a, 2, 2)), K::Kind::Known, a, 8), "RELEASE_MEM 64-bit label with an interrupt");
	Check(Classify(ReleaseMem(a, 0, 2)).kind == K::Kind::None, "RELEASE_MEM without data: nothing");
	Check(Classify(ReleaseMem(a, 1, 1)).kind == K::Kind::None, "RELEASE_MEM interrupt only on the graphics queue: nothing");
	Check(Is(Classify(OcclusionDump(0x10003f80ull)), K::Kind::Known, 0x10003f80ull, 256), "occlusion dump: 16 DB pairs");
	Check(Classify(CacheEvent()).kind == K::Kind::None, "a cache event: nothing");
	Packet other = OcclusionDump(a);
	other.payload[0] = 0x0au;
	Check(Classify(other).kind == K::Kind::Unknown, "another event with an address: unknown");
	Check(Classify(Flip()).kind == K::Kind::Unknown, "a flip: unknown");
	Check(Classify(ReleaseMem(a, 1, 0), true).kind == K::Kind::Unknown && Classify(WriteData(a, 1, false), true).kind == K::Kind::Unknown,
	      "host occlusion queries: every ordered item may publish a queued dump");
	Packet short_write = WriteData(a, 1, false);
	short_write.packet_dw = 4;
	Check(Classify(short_write).kind == K::Kind::Unknown, "a truncated WRITE_DATA: unknown");
	Packet draw {OpClass::Draw, Pm4::IT_DRAW_INDEX_AUTO, 0, {3u, 0u}, 3u};
	Packet dispatch {OpClass::Dispatch, Pm4::IT_DISPATCH_DIRECT, 0, {1u, 1u, 1u, 0u}, 5u};
	Check(Classify(draw).kind == K::Kind::None && Classify(dispatch).kind == K::Kind::None, "draws and dispatches store nothing from the CPU");
}

void TrackerTests() {
	K::Tracker t;
	std::vector<WriteRange> out;
	t.Note(3, {K::Kind::Known, 0x1000, 4});
	t.Note(4, {K::Kind::None, 0, 0});
	t.Note(5, {K::Kind::Unknown, 0, 0});
	t.Note(7, {K::Kind::Known, 0x5000, 8});
	Check(t.Capture(2, &out) == 5, "held behind the unknown item 5");
	Check(out.size() == 2 && out[0].address == 0x1000 && out[0].size == 0x1000 && out[1].address == 0x5000 && out[1].size == 0x1000,
	      "the pages of items 3 and 7");
	Check(t.Capture(3, &out) == 5 && out.size() == 1 && out[0].address == 0x5000, "item 3 ran: its range is gone, item 7 stays");
	Check(t.Capture(5, &out) == 0 && out.size() == 1, "the unknown item ran: free, item 7 still in the FIFO");
	Check(t.Capture(7, &out) == 0 && out.empty() && t.Pending() == 0, "everything ran");
	// ranges are merged by pages: the same label again, a neighbour page, a range across two pages, one far away
	t.Note(10, {K::Kind::Known, 0x8ff8, 8});
	t.Note(11, {K::Kind::Known, 0x8ff8, 8});
	t.Note(12, {K::Kind::Known, 0x9ffc, 8}); // pages 9 and a
	t.Note(13, {K::Kind::Known, 0x20000, 4});
	t.Note(14, {K::Kind::Known, 0x7000, 0}); // empty
	Check(t.Capture(9, &out) == 0 && out.size() == 2 && out[0].address == 0x8000 && out[0].size == 0x3000 && out[1].address == 0x20000 && out[1].size == 0x1000,
	      "merged pages 8..a and page 0x20");
	Check(t.Capture(12, &out) == 0 && out.size() == 1 && out[0].address == 0x20000, "after item 12 only page 0x20");
	// many items: the compaction keeps the order and the newest ranges
	for (uint64_t seq = 20; seq < 1000; seq++) t.Note(seq, {K::Kind::Known, seq * 0x1000u, 4});
	Check(t.Capture(700, &out) == 0 && out.size() == 1 && out[0].address == 701u * 0x1000u && out[0].size == 299u * 0x1000u, "compacted, merged ranges");
	Check(t.Capture(998, &out) == 0 && out.size() == 1 && out[0].address == 999u * 0x1000u && out[0].size == 0x1000u, "the last range");
}

void OverlapTests() {
	const std::vector<uint64_t> pages {0x2000, 0x5000, 0x9000};
	Check(K::Overlaps({{0x2ffc, 4}}, pages), "a dword in a page");
	Check(!K::Overlaps({{0x3000, 0x2000}}, pages), "pages 3 and 4: none read");
	Check(K::Overlaps({{0x1ff8, 16}}, pages), "a range from page 1 into page 2");
	Check(K::Overlaps({{0x3ff8, 0x1010}}, pages), "a range from page 3 to page 5");
	Check(!K::Overlaps({{0x5000, 0}}, pages), "an empty range");
	Check(!K::Overlaps({{0x2000, 4}}, {}), "no pages");
	Check(!K::Overlaps({}, pages), "no writes");
	Check(!K::Overlaps({{0xa000, 4}, {0x1000, 0x1000}}, pages), "below and above");
	Check(K::Overlaps({{UINT64_MAX - 8u, 64}}, {UINT64_MAX & ~uint64_t {4095}}), "a range at the top of the address space");
}

// --- oracle ---
constexpr uint64_t kBase  = 0x10000000ull;
constexpr uint32_t kPages = 64;
constexpr uint64_t kPage  = 4096;
constexpr size_t   kDepth = 16;

enum class Op : uint8_t { Draw, Store, Unknown, Dispatch, Download, Nothing };

struct Job {
	std::atomic<uint8_t>    state {0}; // 0 idle, 1 held, 2 queued, 3 done
	uint64_t                gate = 0;
	std::vector<uint64_t>   pages;     // sorted
	std::vector<WriteRange> cpu_writes;
	bool                    early = false;
	// worker
	bool                  answered  = false;
	uint64_t              write_seq = 0;
	std::vector<uint64_t> seen;
};

struct Item {
	Op       op = Op::Nothing;
	Packet   packet;
	uint64_t store_address = 0; // the bytes the recorder stores (Store: the packet's real extent; Unknown: one page)
	uint64_t store_size    = 0;
	uint32_t page          = 0; // Dispatch / Download
	Job      job;
};

struct Memory {
	std::array<std::atomic<uint64_t>, kPages> version {};
	std::array<std::atomic<bool>, kPages>     dirty {};
	std::array<std::atomic<uint64_t>, kPages> published {}; // the write log: the newest publish of the page
	std::atomic<uint64_t>                     log_seq {0};
};

uint32_t PageIndex(uint64_t page_address) { return static_cast<uint32_t>((page_address - kBase) / kPage); }

void StoreBytes(Memory& m, uint64_t address, uint64_t size) {
	for (uint64_t page = address & ~(kPage - 1u); page < address + size; page += kPage) {
		const uint32_t i = PageIndex(page);
		if (i < kPages) m.version[i].fetch_add(1, std::memory_order_seq_cst);
	}
}

class WorkQueue {
public:
	void Push(Job* job) {
		{
			std::lock_guard lock(m_mutex);
			m_jobs.push_back(job);
		}
		m_cv.notify_one();
	}
	Job* Pop() {
		std::unique_lock lock(m_mutex);
		m_cv.wait(lock, [this] { return m_stop || !m_jobs.empty(); });
		if (m_jobs.empty()) return nullptr;
		Job* job = m_jobs.front();
		m_jobs.pop_front();
		return job;
	}
	void Stop() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_cv.notify_all();
	}

private:
	std::mutex              m_mutex;
	std::condition_variable m_cv;
	std::deque<Job*>        m_jobs;
	bool                    m_stop = false;
};

struct OracleStats {
	uint64_t draws = 0, used = 0, early = 0, held = 0, refused_cpuw = 0, refused_dirty = 0, refused_log = 0, unanswered = 0, captured = 0;
};

OracleStats RunOracle(uint32_t seed, uint32_t items) {
	Memory                             mem;
	std::vector<std::unique_ptr<Item>> ring(kDepth);
	for (auto& it: ring) it = std::make_unique<Item>();
	std::atomic<uint64_t> tail {0};
	std::atomic<uint64_t> done {0};
	WorkQueue             queue;
	OracleStats           stats;
	std::atomic<bool>     finished {false};

	std::thread watchdog([&finished] {
		for (int i = 0; i < 1200 && !finished.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
		if (!finished.load()) {
			std::fprintf(stderr, "CpPipeCheckedTests: failed: the oracle did not finish (a job nobody runs?)\n");
			std::fflush(stderr);
			std::_Exit(1);
		}
	});

	std::vector<std::thread> workers;
	for (uint32_t w = 0; w < 3; w++) {
		workers.emplace_back([&mem, &queue, seed, w] {
			std::mt19937 rng(seed * 31u + w);
			while (Job* job = queue.Pop()) {
				job->write_seq = mem.log_seq.load(std::memory_order_seq_cst); // before the first read (RunGfpJob)
				job->answered  = true;
				job->seen.clear();
				for (const uint64_t page: job->pages) {
					if ((rng() & 3u) == 0) std::this_thread::yield();
					const uint32_t i = PageIndex(page);
					if (mem.dirty[i].load(std::memory_order_seq_cst)) job->answered = false; // a protected page: the walk faults, the job is not answered
					job->seen.push_back(mem.version[i].load(std::memory_order_seq_cst));
				}
				job->state.store(3, std::memory_order_release);
			}
		});
	}

	// The recorder: items in order, as PipeFrontRun / PipeFrontExecute / PipeFrontRelease / JoinJob.
	std::thread recorder([&] {
		for (uint64_t seq = 1; seq <= items; seq++) {
			while (tail.load(std::memory_order_acquire) < seq) std::this_thread::yield();
			Item& it = *ring[seq % kDepth];
			switch (it.op) {
				case Op::Draw: {
					Job&    job  = it.job;
					uint8_t held = 1;
					if (job.state.load(std::memory_order_acquire) == 1 && job.state.compare_exchange_strong(held, 2)) queue.Push(&job);
					while (job.state.load(std::memory_order_acquire) != 3) std::this_thread::yield();
					stats.draws++;
					bool use = job.answered;
					if (!job.answered) stats.unanswered++;
					if (use) {
						for (const uint64_t page: job.pages) use = use && !mem.dirty[PageIndex(page)].load();
						if (!use) stats.refused_dirty++;
					}
					if (use) {
						for (const uint64_t page: job.pages) use = use && mem.published[PageIndex(page)].load() <= job.write_seq;
						if (!use) stats.refused_log++;
					}
					if (use && K::Overlaps(job.cpu_writes, job.pages)) {
						use = false;
						stats.refused_cpuw++;
					}
					if (use) {
						stats.used++;
						for (size_t k = 0; k < job.pages.size(); k++) {
							if (job.seen[k] != mem.version[PageIndex(job.pages[k])].load()) {
								std::fprintf(stderr, "CpPipeCheckedTests: seed %u item %llu page 0x%llx: worker saw version %llu, the draw sees %llu (early=%d, %zu ranges)\n", seed,
								             static_cast<unsigned long long>(seq), static_cast<unsigned long long>(job.pages[k]), static_cast<unsigned long long>(job.seen[k]),
								             static_cast<unsigned long long>(mem.version[PageIndex(job.pages[k])].load()), job.early ? 1 : 0, job.cpu_writes.size());
								Check(false, "a used job read a page that changed before its draw");
							}
						}
					}
					job.state.store(0, std::memory_order_relaxed);
					break;
				}
				case Op::Store:
				case Op::Unknown: StoreBytes(mem, it.store_address, it.store_size); break;
				case Op::Dispatch: mem.dirty[it.page].store(true, std::memory_order_seq_cst); break;
				case Op::Download:
					if (mem.dirty[it.page].load()) {
						mem.version[it.page].fetch_add(1, std::memory_order_seq_cst); // the GPU's bytes land in the backing
						mem.published[it.page].store(mem.log_seq.fetch_add(1, std::memory_order_seq_cst) + 1u, std::memory_order_seq_cst);
						mem.dirty[it.page].store(false, std::memory_order_seq_cst);
					}
					break;
				case Op::Nothing: break;
			}
			done.store(seq, std::memory_order_release);
			// Release the held jobs whose gate has run (gates grow with the items).
			const uint64_t last = std::min<uint64_t>(tail.load(std::memory_order_acquire), seq + kDepth - 1u);
			for (uint64_t next = seq + 1; next <= last; next++) {
				Item& n = *ring[next % kDepth];
				if (n.op != Op::Draw || n.job.state.load(std::memory_order_acquire) != 1) continue;
				if (n.job.gate > seq) break;
				uint8_t held = 1;
				if (n.job.state.compare_exchange_strong(held, 2)) queue.Push(&n.job);
			}
		}
	});

	// The front: builds the items, classifies them, holds or hands out the jobs.
	std::mt19937 rng(seed);
	K::Tracker   tracker;
	GateTracker  gates;
	auto         page_address = [&rng] { return kBase + (rng() % kPages) * kPage; };
	for (uint64_t seq = 1; seq <= items; seq++) {
		while (done.load(std::memory_order_acquire) + kDepth <= seq) std::this_thread::yield(); // the slot's previous item has run
		Item&          it   = *ring[seq % kDepth];
		const uint32_t roll = rng() % 100u;
		it.op               = Op::Nothing;
		it.packet           = CacheEvent();
		if (roll < 45) {
			it.op = Op::Draw;
			Job& job = it.job;
			job.pages.clear();
			const uint32_t n = 1u + rng() % 3u;
			for (uint32_t k = 0; k < n; k++) job.pages.push_back(page_address());
			std::sort(job.pages.begin(), job.pages.end());
			job.pages.erase(std::unique(job.pages.begin(), job.pages.end()), job.pages.end());
			const uint64_t d         = done.load(std::memory_order_acquire);
			const uint64_t fifo_gate = gates.GateFor(SignalPolicy::Fifo, d + 1u);
			job.gate                 = tracker.Capture(d, &job.cpu_writes);
			job.early                = fifo_gate > job.gate;
			stats.early += job.early ? 1u : 0u;
			stats.captured += job.cpu_writes.size();
			it.packet = {OpClass::Draw, Pm4::IT_DRAW_INDEX_AUTO, 0, {3u, 0u}, 3u};
			if (job.gate != 0) {
				stats.held++;
				job.state.store(1, std::memory_order_release); // the recorder releases it after its gate (or at the draw)
			} else {
				job.state.store(2, std::memory_order_release);
				queue.Push(&job); // before the commit, as PipeFrontEnqueue's SubmitJob
			}
		} else if (roll < 75) {
			// A CPU store with a known extent; near a page end now and then (ranges across pages).
			const uint64_t page = page_address();
			const uint64_t off  = (rng() & 1u) ? kPage - 8u * (1u + rng() % 4u) : (rng() % (kPage / 4u)) * 4u;
			const uint64_t a    = page + off;
			switch (rng() % 5u) {
				case 0: it.packet = WriteData(a, 1u + rng() % 6u, false); break;
				case 1: it.packet = DmaData(a, 4u * (1u + rng() % 1500u)); break;
				case 2: it.packet = CopyData(a, (rng() & 1u) != 0); break;
				case 3: it.packet = ReleaseMem(a & ~uint64_t {7}, 1u + rng() % 2u, rng() % 3u == 1 ? 2u : 0u); break;
				default: it.packet = OcclusionDump(a & ~uint64_t {7}); break;
			}
			it.op = Op::Store;
			// The extent the recorder really stores (graphicsRun.cpp), independent of ClassifyItem.
			switch (it.packet.opcode) {
				case Pm4::IT_WRITE_DATA: it.store_address = a; it.store_size = 4u * (it.packet.packet_dw - 4u); break;
				case Pm4::IT_DMA_DATA: it.store_address = a; it.store_size = it.packet.payload[5]; break;
				case Pm4::IT_COPY_DATA: it.store_address = a; it.store_size = (it.packet.payload[0] & (1u << 16u)) != 0 ? 8u : 4u; break;
				case Pm4::IT_RELEASE_MEM: it.store_address = a & ~uint64_t {7}; it.store_size = ((it.packet.payload[1] >> 29u) & 7u) == 1u ? 4u : 8u; break;
				default: it.store_address = a & ~uint64_t {7}; it.store_size = 256u; break;
			}
		} else if (roll < 79) {
			it.op            = Op::Unknown; // a flip: it stores somewhere
			it.packet        = Flip();
			it.store_address = page_address() + 64u;
			it.store_size    = 8u;
		} else if (roll < 84) {
			it.op     = Op::Dispatch;
			it.page   = rng() % kPages;
			it.packet = {OpClass::Dispatch, Pm4::IT_DISPATCH_DIRECT, 0, {1u, 1u, 1u, 0u}, 5u};
		} else if (roll < 97) {
			it.op   = Op::Download;
			it.page = rng() % kPages;
			it.packet = CacheEvent(); // the publish is not a CPU store of the item (BackingWriteLog)
		}
		tail.store(seq, std::memory_order_release); // Commit
		gates.NoteItem(it.packet.cls, it.packet.opcode, seq);
		tracker.Note(seq, Classify(it.packet));
	}
	recorder.join();
	queue.Stop();
	for (auto& w: workers) w.join();
	finished.store(true);
	watchdog.join();
	return stats;
}

void OracleTests() {
	for (const uint32_t seed: {1u, 7u, 1234u}) {
		const OracleStats s = RunOracle(seed, 60000);
		std::printf("CpPipeCheckedTests: oracle seed %u: draws %llu used %llu early %llu held %llu ranges %llu refused cpu-write %llu dirty %llu log %llu unanswered %llu\n",
		            seed, static_cast<unsigned long long>(s.draws), static_cast<unsigned long long>(s.used), static_cast<unsigned long long>(s.early),
		            static_cast<unsigned long long>(s.held), static_cast<unsigned long long>(s.captured), static_cast<unsigned long long>(s.refused_cpuw),
		            static_cast<unsigned long long>(s.refused_dirty), static_cast<unsigned long long>(s.refused_log), static_cast<unsigned long long>(s.unanswered));
		Check(s.draws > 20000 && s.used > s.draws / 3, "the oracle uses jobs");
		Check(s.early > s.draws / 4, "most jobs start earlier than fifo would let them");
		Check(s.refused_cpuw > 0, "some jobs are refused for a CPU store (the check is exercised)");
	}
}

} // namespace

int main() {
#if !defined(KYTY_CPCHK_ORACLE_ONLY)
	PolicyTests();
	ClassifyTests();
	TrackerTests();
	OverlapTests();
#endif
	OracleTests();
	std::printf("CpPipeCheckedTests: ok\n");
	return 0;
}
