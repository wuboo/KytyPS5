// S613 (KYTY_REC_COUNTERS, recCounters.h; KYTY_STREAM_PROBE, streamProbe.h): the recorder's counter table and the stream-copy probe. Every check ends
// in exit(1) when it fails, so each mutant (CMake rec_diet_mutant_*, WILL_FAIL) is caught by a failed check.
//
// Counter table: the mode table; hand cases (a flush moves every non-zero entry once, the last one too, and leaves the table empty; the thread
// sees its pending part, another thread only the global array); an oracle script on the recorder (random adds, Gets, flushes and the seg_record
// computation of commandScheduler.cpp: Get(a) + Get(b) - base at every "submit") against an unbuffered reference in which every add is global:
// every Get on the recorder must equal the reference, the global array never exceeds it and equals it after a flush; threads (4 writers with
// their own tables and a reader of the global array: monotone entries, exact totals at the end; TSan on the Mac).
//   GET_IGNORES_LOCAL, FLUSH_NO_ZERO, FLUSH_SKIPS_LAST, DEFAULT_ON: caught (the oracle and the hand cases).
// Stream probe: hand cases (first, same range and tick with the same / other bytes, an earlier tick, another size at the same address) and an
// oracle over random streams of copies (ranges without slot collisions, ticks, contents) against an exact map: the counts must be equal; with
// collisions the probe may only undercount.
//   TICK_IGNORED, SIZE_IGNORED, HASH_IGNORED: caught.

#include "graphics/host_gpu/renderer/cache/streamProbe.h"
#include "graphics/presentation/recCounters.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace RC = Libs::Graphics::RecCounters;
namespace SP = Libs::Graphics::StreamProbe;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "RecDietTests: failed: %s\n", text);
		std::fflush(stderr);
		std::exit(1);
	}
}

constexpr size_t kN = 16;
using Global = std::array<std::atomic<uint64_t>, kN>;

void ModeTable() {
	Check(RC::ParseMode(nullptr) == RC::Mode::Off, "unset -> off");
	Check(RC::ParseMode("") == RC::Mode::Off, "empty -> off");
	Check(RC::ParseMode("0") == RC::Mode::Off, "0 -> off");
	Check(RC::ParseMode("yes") == RC::Mode::Off, "unknown -> off");
	Check(RC::ParseMode("1") == RC::Mode::Buffered, "1 -> buffered");
	Check(RC::ParseMode("buffered") == RC::Mode::Buffered, "buffered -> buffered");
	Check(SP::ParseMode(nullptr) == SP::Mode::Off && SP::ParseMode("0") == SP::Mode::Off && SP::ParseMode("1") == SP::Mode::On &&
	          SP::ParseMode("on") == SP::Mode::On,
	      "stream probe modes");
}

void CounterHandCases() {
	Global   global {};
	uint64_t local[kN] {};
	local[0] = 3;
	local[kN - 1] = 5;
	global[0].store(10);
	Check(RC::Read(global, local, 0) == 13, "the thread sees global + pending");
	Check(RC::Read(global, nullptr, 0) == 10, "another thread sees the global array");
	Check(RC::Read(global, local, kN - 1) == 5, "pending only");
	Check(RC::FlushInto(local, global, kN) == 2, "two entries move");
	Check(global[0].load() == 13 && global[kN - 1].load() == 5, "the global array has the adds (the last entry too)");
	Check(local[0] == 0 && local[kN - 1] == 0, "the table is empty after a flush");
	Check(RC::FlushInto(local, global, kN) == 0, "nothing moves twice");
	Check(global[0].load() == 13 && global[kN - 1].load() == 5, "a second flush adds nothing");
	Check(RC::Read(global, local, 0) == 13, "after the flush the thread sees the same value");
}

// The recorder: adds go to its table, Gets read through it, flushes at random points; the reference counts every add globally at once.
void CounterOracle() {
	for (uint32_t seed = 1; seed <= 4; seed++) {
		std::mt19937_64 rng(seed * 7919u);
		Global          global {};
		uint64_t        local[kN] {};
		uint64_t        ref[kN] {};
		uint64_t        base = 0, ref_base = 0; // the seg_record wait base (commandScheduler.cpp: cp_wait_guest + cp_wait_gpu at the last submit)
		uint64_t        flushes = 0, gets = 0, segs = 0;
		for (int step = 0; step < 200000; step++) {
			const uint32_t op = static_cast<uint32_t>(rng() % 100);
			const size_t   i  = static_cast<size_t>(rng() % kN);
			if (op < 60) {
				const uint64_t v = 1 + rng() % 1000;
				local[i] += v; // FrameTiming::Add on a buffered thread
				ref[i] += v;
			} else if (op < 80) {
				gets++;
				Check(RC::Read(global, local, i) == ref[i], "a Get on the recorder equals the unbuffered value");
				Check(RC::Read(global, nullptr, i) <= ref[i], "another thread never sees more than was added");
			} else if (op < 90) {
				segs++;
				const uint64_t now     = RC::Read(global, local, 2) + RC::Read(global, local, 3);
				const uint64_t ref_now = ref[2] + ref[3];
				Check(now - base == ref_now - ref_base, "the seg_record wait of a submit equals the unbuffered one");
				base     = now;
				ref_base = ref_now;
			} else {
				flushes++;
				(void)RC::FlushInto(local, global, kN);
				for (size_t k = 0; k < kN; k++) {
					Check(global[k].load() == ref[k], "after a flush the global array equals the reference");
					Check(local[k] == 0, "after a flush the table is empty");
				}
			}
		}
		(void)RC::FlushInto(local, global, kN);
		for (size_t k = 0; k < kN; k++) Check(global[k].load() == ref[k], "final totals");
		Check(flushes > 1000 && gets > 1000 && segs > 1000, "the script exercised every operation");
	}
}

// Four buffered writers and a reader of the global array (what the presentation thread does when it writes a frames.csv row).
void CounterThreads() {
	Global                global {};
	std::atomic<int>      done {0};
	std::atomic<bool>     monotone {true};
	constexpr int         kWriters = 4;
	std::array<uint64_t, kN> expected {};
	std::vector<std::thread> threads;
	std::array<std::array<uint64_t, kN>, kWriters> totals {};
	for (int w = 0; w < kWriters; w++) {
		threads.emplace_back([&, w] {
			std::mt19937_64 rng(1000u + static_cast<uint64_t>(w));
			uint64_t        local[kN] {};
			for (int step = 0; step < 100000; step++) {
				const size_t   i = static_cast<size_t>(rng() % kN);
				const uint64_t v = 1 + rng() % 16;
				local[i] += v;
				totals[static_cast<size_t>(w)][i] += v;
				if (rng() % 64 == 0) (void)RC::FlushInto(local, global, kN);
			}
			(void)RC::FlushInto(local, global, kN);
			done.fetch_add(1, std::memory_order_release);
		});
	}
	std::thread reader([&] {
		std::array<uint64_t, kN> last {};
		while (done.load(std::memory_order_acquire) != kWriters) {
			for (size_t k = 0; k < kN; k++) {
				const uint64_t v = RC::Read(global, nullptr, k);
				if (v < last[k]) monotone.store(false);
				last[k] = v;
			}
		}
	});
	for (auto& t: threads) t.join();
	reader.join();
	for (const auto& t: totals)
		for (size_t k = 0; k < kN; k++) expected[k] += t[k];
	for (size_t k = 0; k < kN; k++) Check(global[k].load() == expected[k], "threads: exact totals");
	Check(monotone.load(), "threads: the global entries never go down");
}

void ProbeHandCases() {
	SP::Table<64> table;
	SP::Stats     st;
	Check(table.Note(0x1000, 256, 7, 0xaa, st) == SP::Kind::First, "first copy");
	Check(table.Note(0x1000, 256, 7, 0xaa, st) == SP::Kind::Same, "same range, tick and bytes");
	Check(table.Note(0x1000, 256, 7, 0xbb, st) == SP::Kind::Repeat, "same range and tick, other bytes");
	Check(table.Note(0x1000, 256, 8, 0xbb, st) == SP::Kind::EarlierSame, "an earlier tick with the same bytes");
	Check(table.Note(0x1000, 256, 9, 0xcc, st) == SP::Kind::EarlierOther, "an earlier tick with other bytes");
	Check(st.seen == 5 && st.rep == 2 && st.same == 1 && st.same_bytes == 256 && st.xtick_same == 1, "hand counts");
	SP::Table<64> t2;
	SP::Stats     s2;
	(void)t2.Note(0x2000, 256, 1, 0x11, s2);
	Check(t2.Note(0x2000, 512, 1, 0x11, s2) != SP::Kind::Same, "another size at the same address is another range");
	Check(s2.same == 0 && s2.rep == 0, "no repeat across sizes");
}

struct Range {
	uint64_t address, size;
	bool operator<(const Range& o) const { return std::tie(address, size) < std::tie(o.address, o.size); }
};

void ProbeOracle() {
	constexpr size_t kSlots = 4096;
	for (uint32_t seed = 1; seed <= 4; seed++) {
		std::mt19937_64 rng(seed * 104729u);
		// Ranges with distinct slots (the property under test is the counting, not the hash of the slot).
		std::vector<Range> ranges;
		std::set<size_t>   used;
		while (ranges.size() < 300) {
			const Range r {0x10000000ull + (rng() % 50000) * 64, 64ull << (rng() % 6)};
			const auto  slot = SP::Table<kSlots>::SlotOf(r.address, r.size);
			bool        dup  = false;
			for (const auto& q: ranges) dup = dup || (q.address == r.address && q.size == r.size);
			if (dup || !used.insert(slot).second) continue;
			ranges.push_back(r);
		}
		// Same addresses with a second size where it gets a slot of its own (catches a key without the size).
		for (size_t k = 0; k < 40; k++) {
			const Range r {ranges[k].address, ranges[k].size * 2};
			if (used.insert(SP::Table<kSlots>::SlotOf(r.address, r.size)).second) ranges.push_back(r);
		}
		auto      table = std::make_unique<SP::Table<kSlots>>();
		SP::Stats st;
		std::map<Range, std::pair<uint64_t, uint64_t>> last; // range -> (tick, hash)
		uint64_t  rep = 0, same = 0, same_bytes = 0, xtick = 0, tick = 1;
		std::map<Range, uint64_t> content; // the guest bytes of a range (hash), changed now and then
		for (int step = 0; step < 300000; step++) {
			if (rng() % 200 == 0) tick++;
			const Range& r = ranges[static_cast<size_t>(rng() % ranges.size())];
			if (rng() % 10 == 0) content[r] = rng();
			const uint64_t hash = content[r];
			const auto     it   = last.find(r);
			if (it != last.end() && it->second.first == tick) {
				rep++;
				if (it->second.second == hash) {
					same++;
					same_bytes += r.size;
				}
			} else if (it != last.end() && it->second.second == hash) {
				xtick++;
			}
			last[r] = {tick, hash};
			(void)table->Note(r.address, r.size, tick, hash, st);
		}
		Check(st.seen == 300000, "oracle: every copy seen");
		Check(st.rep == rep, "oracle: repeats in the same tick");
		Check(st.same == same && st.same_bytes == same_bytes, "oracle: same bytes in the same tick");
		Check(st.xtick_same == xtick, "oracle: same bytes in an earlier tick");
		Check(same > 10000 && rep > same && xtick > 1000, "oracle: the stream had every kind");
	}
	// With collisions (a small table) the probe forgets ranges: it may only undercount.
	{
		std::mt19937_64 rng(77);
		SP::Table<16>   small;
		SP::Stats       st;
		std::map<Range, std::pair<uint64_t, uint64_t>> last;
		uint64_t        same = 0, tick = 1;
		for (int step = 0; step < 100000; step++) {
			if (rng() % 100 == 0) tick++;
			const Range    r {0x4000ull * (rng() % 64), 256};
			const uint64_t hash = rng() % 2;
			const auto     it   = last.find(r);
			if (it != last.end() && it->second.first == tick && it->second.second == hash) same++;
			last[r] = {tick, hash};
			(void)small.Note(r.address, r.size, tick, hash, st);
		}
		Check(st.same <= same, "collisions: the probe only undercounts");
	}
}

} // namespace

int main() {
	ModeTable();
	CounterHandCases();
	CounterOracle();
	CounterThreads();
	ProbeHandCases();
	ProbeOracle();
	std::printf("RecDietTests: ok\n");
	return 0;
}
