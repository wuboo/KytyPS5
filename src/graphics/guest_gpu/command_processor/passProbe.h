#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSPROBE_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSPROBE_H

// S611 (KYTY_PASS_PROBE=<from>,<to>, default off): a measurement spike for parallel pass recording. The recorder (Thread_Gpu executing the FIFO items,
// cpPipe.cpp PipeFrontExecute) logs every item it executes while the guest flip count is in [from, to): its start, its duration, its class and
// opcode, and the change of a fixed set of KYTY_FRAME_LOG counters over the item (render passes begun, submits, GPU waits, readbacks, uploads, cache
// misses, worker stalls). At `to` the probe computes, for several models of a recorder split across N threads, how long the recorded work would take
// (Makespan below) and logs one line per model; KYTY_PASS_PROBE_FILE=<path> also writes the raw records (WriteRecords) for the offline analysis.
// Nothing else changes: off, the recorder does one load of a flag per item.
//
// The model (what "independent between passes" means here):
//   - an item's work is its duration minus the waits inside it (worker stall, GPU wait, upload join): waits do not shrink with more recorders;
//   - a serial item runs alone, after everything before it and before everything after it (a fork-join barrier): Sync items (the front hands its
//     live state over), and per policy items with a submit/flush, a GPU wait, a readback/download, an upload, a pipeline creation, a render-target
//     or depth memo miss, or (strict policy) every Ordered item (a CPU store or signal in stream order);
//   - the rest is cut into units: a unit starts at an item that begins a render pass (a pass group), after a serial item, and (chunk K > 0) after K
//     draws of the same pass (secondary command buffers inside a pass); units between two serial items are recorded in stream order by N threads,
//     each unit on the thread that is free first (in-order list scheduling); a unit costs its work plus a fixed overhead (begin/end of the secondary
//     buffer, the state the unit re-establishes);
//   - total = sum of the segment makespans + the serial items' work. N = 1 with no overhead is the serial work exactly.
//
// Mutants (WILL_FAIL, tests/PassProbeTests.cpp): KYTY_PASSPROBE_MUTANT_SERIAL_IGNORED (a serial item does not cut the segment),
// KYTY_PASSPROBE_MUTANT_PASS_IGNORED (a pass begin does not start a unit), KYTY_PASSPROBE_MUTANT_NO_OVERHEAD (the unit overhead is dropped),
// KYTY_PASSPROBE_MUTANT_CHUNK_PLUS_ONE (a chunk holds K + 1 draws), KYTY_PASSPROBE_MUTANT_WAIT_KEPT (the waits stay in the work),
// KYTY_PASSPROBE_MUTANT_SYNC_PARALLEL (a Sync item is not serial), KYTY_PASSPROBE_MUTANT_ROUND_ROBIN (units go to the threads in turn),
// KYTY_PASSPROBE_MUTANT_WINDOW_OPEN_END (the probe records at flip `to`).

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics::PassProbe {

// The counters observed over every item (KYTY_FRAME_LOG column names; FrameTiming resolves them, an unknown name stays 0).
inline constexpr std::array<const char*, 24> kColumns = {
    "render_passes",     // 0  passes begun (BeginRendering)
    "cp_flushes",        // 1  scheduler flushes (submits) on the recorder
    "eager_submit_n",    // 2  eager submits
    "flush_draws_n",     // 3  KYTY_FLUSH_DRAWS submits
    "cp_wait_gpu_ns",    // 4  waits for the GPU (wait)
    "pipe_stall_ns",     // 5  waits for a pipeline worker (wait)
    "upas_join_ns",      // 6  waits for the upload thread (wait)
    "readbacks",         // 7  GPU->CPU readbacks
    "readback_downloads",// 8
    "rb_eager_downloads",// 9
    "sync_rf_guest_n",   // 10 the guest waited for a readback
    "upload_sync_n",     // 11 buffer uploads (SynchronizeBuffer)
    "upload_bytes",      // 12
    "pipelines_created", // 13
    "rt_memo_miss_n",    // 14 render target memo misses (texture cache lookups)
    "depth_memo_miss_n", // 15
    "size_memo_miss_n",  // 16
    "tex_slot_memo_miss_n", // 17
    "pipe_fallback_n",   // 18 the recorder computed a draw's programs itself
    "descriptor_sets",   // 19
    "texture_upload_bytes", // 20
    "defer_write_flush_n",  // 21
    "dcc_publish_n",     // 22
    "emit_drain_n",      // 23
};
inline constexpr size_t kCols = kColumns.size();
enum Col : uint32_t {
	ColPasses = 0, ColFlushes = 1, ColEager = 2, ColFlushDraws = 3, ColWaitGpu = 4, ColStall = 5, ColUpasJoin = 6, ColReadbacks = 7, ColRbDownloads = 8,
	ColRbEager = 9, ColSyncRfGuest = 10, ColUploadSync = 11, ColUploadBytes = 12, ColPipelines = 13, ColRtMiss = 14, ColDepthMiss = 15, ColSizeMiss = 16,
	ColTexSlotMiss = 17, ColFallback = 18, ColDescSets = 19, ColTexUploadBytes = 20, ColDeferFlush = 21, ColDccPublish = 22, ColEmitDrain = 23,
};

enum class Kind : uint8_t { Draw = 0, Dispatch = 1, Ordered = 2, Sync = 3 };

// One executed FIFO item.
struct Rec {
	uint64_t t0_ns  = 0;
	uint32_t dt_ns  = 0;
	uint32_t flip   = 0;
	uint16_t opcode = 0;
	uint8_t  kind   = 0;
	uint8_t  pad    = 0;
	uint32_t pad2   = 0;
	std::array<uint32_t, kCols> d {}; // counter deltas over the item (saturated)
};
static_assert(sizeof(Rec) == 24 + 4 * kCols);

// KYTY_PASS_PROBE=<from>,<to>: the flip window; anything else (or from >= to) is off.
struct Window {
	bool     on   = false;
	uint64_t from = 0;
	uint64_t to   = 0;
};
inline Window ParseWindow(const char* text) {
	Window w;
	if (text == nullptr || *text == '\0') return w;
	char*                    end  = nullptr;
	const unsigned long long from = std::strtoull(text, &end, 10);
	if (end == text || *end != ',') return w;
	const char*              second = end + 1;
	const unsigned long long to     = std::strtoull(second, &end, 10);
	if (end == second || *end != '\0' || from >= to) return w;
	w.on   = true;
	w.from = from;
	w.to   = to;
	return w;
}
inline bool InWindow(const Window& w, uint64_t flip) {
#ifdef KYTY_PASSPROBE_MUTANT_WINDOW_OPEN_END
	return w.on && flip >= w.from && flip <= w.to;
#else
	return w.on && flip >= w.from && flip < w.to;
#endif
}

// Which items are serial.
struct Policy {
	bool ordered_serial = false; // strict: every Ordered item (CPU store, signal) runs in stream order
	bool mutation_serial = true; // uploads, pipeline creations, render-target/depth/size memo misses
	const char* name = "lenient";
};
inline constexpr std::array<Policy, 3> kPolicies = {Policy {false, false, "loose"}, Policy {false, true, "lenient"}, Policy {true, true, "strict"}};

struct Item {
	uint64_t work_ns    = 0;
	bool     pass_begin = false;
	bool     serial     = false;
	bool     draw       = false;
};

inline uint64_t WaitNs(const Rec& r) {
	return static_cast<uint64_t>(r.d[ColWaitGpu]) + r.d[ColStall] + r.d[ColUpasJoin];
}

inline bool IsSerial(const Rec& r, const Policy& p) {
	const auto kind = static_cast<Kind>(r.kind);
#ifndef KYTY_PASSPROBE_MUTANT_SYNC_PARALLEL
	if (kind == Kind::Sync) return true;
#endif
	if (p.ordered_serial && kind == Kind::Ordered) return true;
	const auto& d = r.d;
	if (d[ColFlushes] != 0 || d[ColEager] != 0 || d[ColFlushDraws] != 0 || d[ColWaitGpu] != 0 || d[ColReadbacks] != 0 || d[ColRbDownloads] != 0 ||
	    d[ColRbEager] != 0 || d[ColSyncRfGuest] != 0) {
		return true;
	}
	if (p.mutation_serial && (d[ColUploadSync] != 0 || d[ColPipelines] != 0 || d[ColRtMiss] != 0 || d[ColDepthMiss] != 0 || d[ColSizeMiss] != 0)) {
		return true;
	}
	return false;
}

inline Item ToItem(const Rec& r, const Policy& p) {
	Item it;
	const uint64_t wait = WaitNs(r);
#ifdef KYTY_PASSPROBE_MUTANT_WAIT_KEPT
	it.work_ns = r.dt_ns;
#else
	it.work_ns = r.dt_ns > wait ? r.dt_ns - wait : 0;
#endif
	it.pass_begin = r.d[ColPasses] != 0;
	it.serial     = IsSerial(r, p);
	it.draw       = static_cast<Kind>(r.kind) == Kind::Draw;
	return it;
}

inline std::vector<Item> ToItems(const std::vector<Rec>& recs, const Policy& p) {
	std::vector<Item> items;
	items.reserve(recs.size());
	for (const auto& r: recs) items.push_back(ToItem(r, p));
	return items;
}

struct Model {
	uint32_t threads     = 2;
	uint32_t chunk       = 0; // draws per unit inside a pass (0: whole pass groups)
	uint64_t overhead_ns = 0; // per unit
};

struct Result {
	uint64_t total_ns  = 0; // modelled time of the work
	uint64_t serial_ns = 0; // of it: serial items
	uint64_t units     = 0;
	uint64_t segments  = 0;
};

// In-order list scheduling of one segment's units on `threads` recorders: each unit goes to the thread that is free first (ties: the lowest index).
inline uint64_t ScheduleSegment(const std::vector<uint64_t>& units, uint32_t threads, std::vector<uint64_t>& free_at) {
	if (units.empty()) return 0;
	const uint32_t n = threads == 0 ? 1u : threads;
	free_at.assign(n, 0);
	uint32_t turn = 0;
	for (const uint64_t u: units) {
#ifdef KYTY_PASSPROBE_MUTANT_ROUND_ROBIN
		const uint32_t best = turn;
		turn                = (turn + 1) % n;
#else
		(void)turn;
		uint32_t best = 0;
		for (uint32_t t = 1; t < n; t++) {
			if (free_at[t] < free_at[best]) best = t;
		}
#endif
		free_at[best] += u;
	}
	return *std::max_element(free_at.begin(), free_at.end());
}

inline Result Makespan(const std::vector<Item>& items, const Model& m) {
	Result                r;
	std::vector<uint64_t> units;
	std::vector<uint64_t> free_at;
	uint64_t              cur       = 0;
	bool                  open      = false;
	uint32_t              cur_draws = 0;
	const auto            close     = [&] {
        if (open) {
#ifdef KYTY_PASSPROBE_MUTANT_NO_OVERHEAD
            units.push_back(cur);
#else
            units.push_back(cur + m.overhead_ns);
#endif
            r.units++;
        }
        open      = false;
        cur       = 0;
        cur_draws = 0;
	};
	const auto end_segment = [&] {
		close();
		if (!units.empty()) {
			r.total_ns += ScheduleSegment(units, m.threads, free_at);
			r.segments++;
		}
		units.clear();
	};
#ifdef KYTY_PASSPROBE_MUTANT_CHUNK_PLUS_ONE
	const uint32_t chunk = m.chunk != 0 ? m.chunk + 1 : 0;
#else
	const uint32_t chunk = m.chunk;
#endif
	for (const auto& it: items) {
#ifndef KYTY_PASSPROBE_MUTANT_SERIAL_IGNORED
		if (it.serial) {
			end_segment();
			r.total_ns += it.work_ns;
			r.serial_ns += it.work_ns;
			continue;
		}
#endif
#ifndef KYTY_PASSPROBE_MUTANT_PASS_IGNORED
		if (it.pass_begin) close();
#endif
		if (chunk != 0 && it.draw && cur_draws >= chunk) close();
		open = true;
		cur += it.work_ns;
		if (it.draw) cur_draws++;
	}
	end_segment();
	return r;
}

// Pass groups (a group starts at an item that begins a render pass), for the histogram of the log line.
struct GroupStats {
	uint64_t groups       = 0;
	uint64_t work_ns      = 0;
	uint64_t max_group_ns = 0;
	uint64_t p50_ns       = 0;
	uint64_t p90_ns       = 0;
	uint64_t top10_ns     = 0; // the 10 % largest groups together
};
inline GroupStats Groups(const std::vector<Item>& items) {
	std::vector<uint64_t> g;
	uint64_t              cur = 0;
	bool                  any = false;
	for (const auto& it: items) {
		if (it.pass_begin && any) {
			g.push_back(cur);
			cur = 0;
		}
		any = true;
		cur += it.work_ns;
	}
	if (any) g.push_back(cur);
	GroupStats s;
	s.groups = g.size();
	for (const auto v: g) s.work_ns += v;
	if (g.empty()) return s;
	std::sort(g.begin(), g.end());
	s.max_group_ns = g.back();
	s.p50_ns       = g[g.size() / 2];
	s.p90_ns       = g[std::min(g.size() - 1, g.size() * 9 / 10)];
	const size_t top = std::max<size_t>(1, g.size() / 10);
	for (size_t i = g.size() - top; i < g.size(); i++) s.top10_ns += g[i];
	return s;
}

// The raw records: a text header line, then the records as they are in memory (little endian on every host this runs on).
inline bool WriteRecords(const char* path, const std::vector<Rec>& recs, const Window& w) {
	if (path == nullptr || *path == '\0') return false;
	FILE* f = std::fopen(path, "wb");
	if (f == nullptr) return false;
	std::fprintf(f, "KYTYPASSPROBE1 records=%zu rec_bytes=%zu cols=%zu from=%llu to=%llu", recs.size(), sizeof(Rec), kCols,
	             static_cast<unsigned long long>(w.from), static_cast<unsigned long long>(w.to));
	for (const auto* c: kColumns) std::fprintf(f, " %s", c);
	std::fputc('\n', f);
	const bool ok = recs.empty() || std::fwrite(recs.data(), sizeof(Rec), recs.size(), f) == recs.size();
	return std::fclose(f) == 0 && ok;
}

inline uint32_t Sat32(uint64_t v) {
	return v > 0xffffffffull ? 0xffffffffu : static_cast<uint32_t>(v);
}

} // namespace Libs::Graphics::PassProbe

#endif /* GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PASSPROBE_H */
