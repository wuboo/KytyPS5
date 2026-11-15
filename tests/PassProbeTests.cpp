// S611 (KYTY_PASS_PROBE, passProbe.h): the pure part of the recorder probe. Every check ends in exit(1) when it fails, so each mutant
// (CMake pass_probe_mutant_*, WILL_FAIL) is caught by a failed check:
//   SERIAL_IGNORED  - a serial item does not cut the segment (the hand cases; the oracle);
//   PASS_IGNORED    - a pass begin does not start a unit (the hand cases; the oracle);
//   NO_OVERHEAD     - the unit overhead is dropped (the hand cases; the oracle);
//   CHUNK_PLUS_ONE  - a chunk holds K + 1 draws (the hand cases; the oracle);
//   WAIT_KEPT       - the waits stay in an item's work (the item table; the oracle);
//   SYNC_PARALLEL   - a Sync item is not serial (the item table; the oracle);
//   ROUND_ROBIN     - units go to the threads in turn, not to the thread that is free first (the hand cases; the oracle);
//   WINDOW_OPEN_END - the probe records at flip `to` (the window table).
// The oracle recomputes the model from the raw records with its own code: the work and the serial flag of a record from the counter columns, the
// units of a segment as index ranges, and a priority queue of (free time, thread) for the schedule. 4 seeds x 3000 random streams x every model of
// the emulator's report (1..4 threads, pass/64/16 draws, 0/3000 ns) x the three policies must agree with Makespan to the nanosecond.
// -DKYTY_PASSPROBE_ORACLE_ONLY builds the oracle alone (a control of what it catches without the tables: every mutant but WINDOW_OPEN_END).

#include "graphics/guest_gpu/command_processor/passProbe.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics::PassProbe;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "PassProbeTests: failed: %s\n", text);
		std::fflush(stderr);
		std::exit(1);
	}
}

Rec MakeRec(Kind kind, uint32_t dt, std::initializer_list<std::pair<Col, uint32_t>> d = {}) {
	Rec r;
	r.kind  = static_cast<uint8_t>(kind);
	r.dt_ns = dt;
	for (const auto& [c, v]: d) r.d[c] = v;
	return r;
}

Item MakeItem(uint64_t work, bool pass, bool serial, bool draw = true) {
	Item it;
	it.work_ns    = work;
	it.pass_begin = pass;
	it.serial     = serial;
	it.draw       = draw;
	return it;
}

void TestWindow() {
	const auto w = ParseWindow("4800,4950");
	Check(w.on && w.from == 4800 && w.to == 4950, "window 4800,4950");
	for (const char* bad: {"", "x", "10", "10,", ",10", "10,5", "5,5", "1,2x", "1;2", "-"}) Check(!ParseWindow(bad).on, "malformed window is off");
	Check(!ParseWindow(nullptr).on, "unset window is off");
	Check(!InWindow(w, 4799), "before from");
	Check(InWindow(w, 4800), "from is inside");
	Check(InWindow(w, 4949), "to - 1 is inside");
	Check(!InWindow(w, 4950), "to is outside");
	Check(!InWindow(Window {}, 5000), "off window");
}

void TestItems() {
	const Policy loose = kPolicies[0], lenient = kPolicies[1], strict = kPolicies[2];
	Check(std::strcmp(loose.name, "loose") == 0 && std::strcmp(lenient.name, "lenient") == 0 && std::strcmp(strict.name, "strict") == 0, "policy names");
	// waits leave the work
	auto it = ToItem(MakeRec(Kind::Draw, 1000, {{ColStall, 300}, {ColUpasJoin, 100}}), loose);
	Check(it.work_ns == 600 && it.draw && !it.serial && !it.pass_begin, "stall and upload join are waits");
	it = ToItem(MakeRec(Kind::Draw, 1000, {{ColWaitGpu, 1500}}), loose);
	Check(it.work_ns == 0 && it.serial, "a GPU wait longer than the item: no work, serial");
	it = ToItem(MakeRec(Kind::Draw, 800, {{ColPasses, 1}}), loose);
	Check(it.pass_begin && it.work_ns == 800, "a pass begin");
	// Sync is serial under every policy; Ordered only under strict
	for (const auto& p: kPolicies) Check(ToItem(MakeRec(Kind::Sync, 10), p).serial, "sync is serial");
	Check(!ToItem(MakeRec(Kind::Ordered, 10), loose).serial && !ToItem(MakeRec(Kind::Ordered, 10), lenient).serial, "ordered parallel (loose, lenient)");
	Check(ToItem(MakeRec(Kind::Ordered, 10), strict).serial, "ordered serial (strict)");
	Check(!ToItem(MakeRec(Kind::Dispatch, 10), strict).serial, "a dispatch is not serial by itself");
	// submits, GPU waits, readbacks: serial everywhere
	for (const Col c: {ColFlushes, ColEager, ColFlushDraws, ColWaitGpu, ColReadbacks, ColRbDownloads, ColRbEager, ColSyncRfGuest}) {
		for (const auto& p: kPolicies) Check(ToItem(MakeRec(Kind::Draw, 10, {{c, 1}}), p).serial, "submit/wait/readback column is serial");
	}
	// cache mutations: serial in lenient and strict only
	for (const Col c: {ColUploadSync, ColPipelines, ColRtMiss, ColDepthMiss, ColSizeMiss}) {
		Check(!ToItem(MakeRec(Kind::Draw, 10, {{c, 1}}), loose).serial, "mutation column parallel (loose)");
		Check(ToItem(MakeRec(Kind::Draw, 10, {{c, 1}}), lenient).serial, "mutation column serial (lenient)");
		Check(ToItem(MakeRec(Kind::Draw, 10, {{c, 1}}), strict).serial, "mutation column serial (strict)");
	}
	// counters that only describe the item
	for (const Col c: {ColUploadBytes, ColTexSlotMiss, ColFallback, ColDescSets, ColTexUploadBytes, ColDeferFlush, ColDccPublish, ColEmitDrain}) {
		for (const auto& p: kPolicies) Check(!ToItem(MakeRec(Kind::Draw, 10, {{c, 7}}), p).serial, "descriptive column is not serial");
	}
	Check(std::strcmp(kColumns[ColPasses], "render_passes") == 0 && std::strcmp(kColumns[ColWaitGpu], "cp_wait_gpu_ns") == 0 &&
	          std::strcmp(kColumns[ColStall], "pipe_stall_ns") == 0 && std::strcmp(kColumns[ColEmitDrain], "emit_drain_n") == 0,
	      "column names match the enum");
}

void TestHandCases() {
	// one thread, no overhead: the sum
	{
		const std::vector<Item> v = {MakeItem(5, true, false), MakeItem(7, false, false), MakeItem(3, false, true), MakeItem(11, true, false)};
		const auto              r = Makespan(v, Model {1, 0, 0});
		Check(r.total_ns == 26 && r.serial_ns == 3, "T1 = sum");
		Check(r.units == 2 && r.segments == 2, "T1 units and segments");
	}
	// two pass groups on two threads
	{
		const std::vector<Item> v = {MakeItem(4, true, false), MakeItem(6, false, false), MakeItem(3, true, false), MakeItem(2, false, false)};
		Check(Makespan(v, Model {2, 0, 0}).total_ns == 10, "two groups in parallel = the longer group");
		Check(Makespan(v, Model {2, 0, 5}).total_ns == 15, "overhead per unit");
		Check(Makespan(v, Model {3, 0, 0}).total_ns == 10, "more threads than units");
	}
	// a serial item is a barrier
	{
		const std::vector<Item> v = {MakeItem(10, true, false), MakeItem(5, false, true), MakeItem(10, true, false)};
		Check(Makespan(v, Model {2, 0, 0}).total_ns == 25, "serial item cuts the segment");
	}
	// a pass begin starts a unit even inside a group of draws
	{
		const std::vector<Item> v = {MakeItem(10, true, false), MakeItem(10, true, false), MakeItem(10, true, false), MakeItem(10, true, false)};
		Check(Makespan(v, Model {2, 0, 0}).total_ns == 20, "four groups on two threads");
		Check(Makespan(v, Model {4, 0, 0}).total_ns == 10, "four groups on four threads");
	}
	// chunks inside a pass
	{
		std::vector<Item> v = {MakeItem(1, true, false)};
		for (int i = 0; i < 4; i++) v.push_back(MakeItem(1, false, false));
		Check(Makespan(v, Model {3, 2, 0}).total_ns == 2, "5 draws, chunk 2, 3 threads: units 2,2,1");
		Check(Makespan(v, Model {3, 0, 0}).total_ns == 5, "no chunks: one unit");
		v.push_back(MakeItem(4, false, false, false)); // a dispatch does not count as a draw of the chunk
		Check(Makespan(v, Model {3, 2, 0}).total_ns == 5, "a non-draw item joins the open chunk");
	}
	// earliest-free assignment, not round robin
	{
		std::vector<Item> v;
		for (int i = 0; i < 3; i++) {
			v.push_back(MakeItem(10, true, false));
			v.push_back(MakeItem(1, true, false));
		}
		Check(Makespan(v, Model {2, 0, 0}).total_ns == 21, "list scheduling 10,1,10,1,10,1 on 2 threads");
	}
	// empty and all-serial streams
	Check(Makespan({}, Model {2, 16, 3000}).total_ns == 0, "empty");
	{
		const std::vector<Item> v = {MakeItem(3, false, true), MakeItem(4, true, true)};
		const auto              r = Makespan(v, Model {4, 16, 3000});
		Check(r.total_ns == 7 && r.units == 0, "only serial items: no unit, no overhead");
	}
}

void TestGroups() {
	const std::vector<Item> v = {MakeItem(2, false, false), MakeItem(3, true, false), MakeItem(1, false, false), MakeItem(9, true, false),
	                             MakeItem(5, true, true)};
	const auto              g = Groups(v);
	Check(g.groups == 4 && g.work_ns == 20 && g.max_group_ns == 9, "groups: the leading items form a group");
	Check(g.top10_ns == 9, "top 10 % of 4 groups = the largest");
}

// --- the oracle ---
struct OracleRec {
	uint64_t work;
	bool     pass, serial, draw;
};
OracleRec OracleItem(const Rec& r, const Policy& p) {
	OracleRec o {};
	uint64_t  wait = 0;
	for (const Col c: {ColWaitGpu, ColStall, ColUpasJoin}) wait += r.d[c];
	o.work = r.dt_ns >= wait ? r.dt_ns - wait : 0;
	o.pass = r.d[ColPasses] > 0;
	o.draw = r.kind == static_cast<uint8_t>(Kind::Draw);
	bool s = r.kind == static_cast<uint8_t>(Kind::Sync) || (p.ordered_serial && r.kind == static_cast<uint8_t>(Kind::Ordered));
	for (const Col c: {ColFlushes, ColEager, ColFlushDraws, ColWaitGpu, ColReadbacks, ColRbDownloads, ColRbEager, ColSyncRfGuest}) s = s || r.d[c] > 0;
	if (p.mutation_serial) {
		for (const Col c: {ColUploadSync, ColPipelines, ColRtMiss, ColDepthMiss, ColSizeMiss}) s = s || r.d[c] > 0;
	}
	o.serial = s;
	return o;
}
uint64_t OracleTotal(const std::vector<Rec>& recs, const Policy& p, const Model& m) {
	std::vector<OracleRec> v;
	for (const auto& r: recs) v.push_back(OracleItem(r, p));
	uint64_t total = 0;
	size_t   i     = 0;
	while (i < v.size()) {
		if (v[i].serial) {
			total += v[i].work;
			i++;
			continue;
		}
		// the segment [i, j): no serial item inside
		size_t j = i;
		while (j < v.size() && !v[j].serial) j++;
		// units as [begin, end) index ranges
		std::vector<std::pair<size_t, size_t>> ranges;
		size_t                                 begin = i, draws = 0;
		for (size_t k = i; k < j; k++) {
			const bool cut = k > begin && (v[k].pass || (m.chunk != 0 && v[k].draw && draws == m.chunk));
			if (cut) {
				ranges.emplace_back(begin, k);
				begin = k;
				draws = 0;
			}
			if (v[k].draw) draws++;
		}
		ranges.emplace_back(begin, j);
		using Slot = std::pair<uint64_t, uint32_t>; // (free at, thread)
		std::priority_queue<Slot, std::vector<Slot>, std::greater<>> q;
		const uint32_t n = m.threads == 0 ? 1 : m.threads;
		for (uint32_t t = 0; t < n; t++) q.emplace(0, t);
		uint64_t end = 0;
		for (const auto& [b, e]: ranges) {
			uint64_t cost = m.overhead_ns;
			for (size_t k = b; k < e; k++) cost += v[k].work;
			auto [free_at, t] = q.top();
			q.pop();
			free_at += cost;
			end = std::max(end, free_at);
			q.emplace(free_at, t);
		}
		total += end;
		i = j;
	}
	return total;
}

void TestOracle() {
	uint64_t compared = 0, serial_items = 0, passes = 0;
	for (const uint32_t seed: {1u, 7u, 611u, 20261116u}) {
		std::mt19937 rng(seed);
		for (int s = 0; s < 3000; s++) {
			const size_t     n = 1 + rng() % 160;
			std::vector<Rec> recs(n);
			for (auto& r: recs) {
				const uint32_t k = rng() % 100;
				r.kind           = static_cast<uint8_t>(k < 75 ? Kind::Draw : (k < 85 ? Kind::Dispatch : (k < 97 ? Kind::Ordered : Kind::Sync)));
				r.dt_ns          = rng() % 20000;
				if (rng() % 9 == 0) r.d[ColPasses] = 1;
				if (rng() % 6 == 0) r.d[ColStall] = rng() % 6000;
				if (rng() % 15 == 0) r.d[ColUpasJoin] = rng() % 3000;
				if (rng() % 60 == 0) r.d[ColWaitGpu] = rng() % 30000;
				for (const Col c: {ColFlushes, ColEager, ColFlushDraws, ColReadbacks, ColRbDownloads, ColRbEager, ColSyncRfGuest, ColUploadSync, ColPipelines,
				                   ColRtMiss, ColDepthMiss, ColSizeMiss, ColTexSlotMiss, ColFallback, ColDescSets}) {
					if (rng() % 70 == 0) r.d[c] = 1 + rng() % 3;
				}
				passes += r.d[ColPasses];
			}
			for (const auto& p: kPolicies) {
				const auto items = ToItems(recs, p);
				for (const auto& it: items) serial_items += it.serial ? 1 : 0;
				for (const uint32_t t: {1u, 2u, 3u, 4u}) {
					for (const uint32_t chunk: {0u, 64u, 16u, 1u, 3u}) {
						for (const uint64_t over: {uint64_t {0}, uint64_t {3000}}) {
							const Model m {t, chunk, over};
							const auto  got  = Makespan(items, m).total_ns;
							const auto  want = OracleTotal(recs, p, m);
							if (got != want) {
								std::fprintf(stderr, "PassProbeTests: oracle seed %u stream %d policy %s T%u chunk %u over %llu: got %llu want %llu\n", seed, s, p.name, t,
								             chunk, static_cast<unsigned long long>(over), static_cast<unsigned long long>(got), static_cast<unsigned long long>(want));
								Check(false, "Makespan agrees with the oracle");
							}
							compared++;
						}
					}
				}
			}
		}
	}
	Check(serial_items > 10000 && passes > 1000, "the random streams exercise serial items and passes");
	std::printf("PassProbeTests: oracle %llu comparisons, %llu serial items, %llu pass begins\n", static_cast<unsigned long long>(compared),
	            static_cast<unsigned long long>(serial_items), static_cast<unsigned long long>(passes));
}

void TestFile() {
	std::vector<Rec> recs;
	for (uint32_t i = 0; i < 1000; i++) {
		Rec r    = MakeRec(i % 4 == 0 ? Kind::Ordered : Kind::Draw, 100 + i, {{ColPasses, i % 13 == 0 ? 1u : 0u}, {ColDescSets, i}});
		r.t0_ns  = 1'000'000'000ull + i * 5000ull;
		r.flip   = 4800 + i / 200;
		r.opcode = static_cast<uint16_t>(0x40 + i % 7);
		recs.push_back(r);
	}
	const std::string path = "pass_probe_tests.bin";
	Check(WriteRecords(path.c_str(), recs, Window {true, 4800, 4805}), "write the records");
	FILE* f = std::fopen(path.c_str(), "rb");
	Check(f != nullptr, "open the records");
	char line[4096] = {};
	Check(std::fgets(line, sizeof(line), f) != nullptr, "header line");
	const char* want = "KYTYPASSPROBE1 records=1000 rec_bytes=120 cols=24 from=4800 to=4805 render_passes cp_flushes eager_submit_n";
	Check(std::strncmp(line, want, std::strlen(want)) == 0, "header text");
	std::vector<Rec> back(recs.size());
	Check(std::fread(back.data(), sizeof(Rec), back.size(), f) == back.size(), "read the records back");
	Check(std::fgetc(f) == EOF, "nothing after the records");
	std::fclose(f);
	std::remove(path.c_str());
	Check(std::memcmp(back.data(), recs.data(), sizeof(Rec) * recs.size()) == 0, "records round trip");
	Check(!WriteRecords(nullptr, recs, Window {}) && !WriteRecords("", recs, Window {}), "no path: nothing written");
	Check(Sat32(0x1'0000'0005ull) == 0xffffffffu && Sat32(77) == 77, "saturation");
}

} // namespace

int main() {
#ifndef KYTY_PASSPROBE_ORACLE_ONLY
	TestWindow();
	TestItems();
	TestHandCases();
	TestGroups();
#endif
	TestOracle();
#ifndef KYTY_PASSPROBE_ORACLE_ONLY
	TestFile();
#endif
	std::printf("PassProbeTests: ok\n");
	return 0;
}
