#include "common/memEvents.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Common::MemEvents;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "MemEventsTests: failed: %s\n", message);
		std::exit(1);
	}
}

// S125: range intersection (half-open, no overflow, empty never intersects).
void TestIntersects() {
	Check(Intersects(0x1000, 0x1000, 0x1800, 0x10), "inside");
	Check(Intersects(0x1000, 0x1000, 0xff0, 0x20), "straddles the start");
	Check(Intersects(0x1000, 0x1000, 0x1ff0, 0x20), "straddles the end");
	Check(!Intersects(0x1000, 0x1000, 0x2000, 0x10), "starts exactly at the end: no overlap");
	Check(!Intersects(0x1000, 0x1000, 0xff0, 0x10), "ends exactly at the start: no overlap");
	Check(!Intersects(0x1000, 0, 0x1000, 0x10), "empty range");
	Check(Intersects(UINT64_MAX - 0xf, 0x100, UINT64_MAX - 0x1, 0x1), "near 2^64 does not wrap");
}

// S125: the ring keeps the newest events, lists the ones in the range oldest first, merges repeats, counts MOVE events.
void TestRing() {
	Ring ring(8);
	ring.Record(Kind::Map, 10, 0x10000, 0x20000, 0x5000, 0, "terrain");
	ring.Record(Kind::Map, 11, 0x90000, 0x1000);
	ring.Record(Kind::FileRead, 12, 0x18000, 0x4000, 0x100, 0, "/app0/t.bin");
	ring.Record(Kind::PoolMoveEinval, 13, 0x12000, 0x1000, 0x70000, -22, "dst");
	ring.Record(Kind::PoolMoveEinval, 13, 0x70000, 0x1000, 0x12000, -22, "src");
	ring.Record(Kind::Unmap, 14, 0x10000, 0x20000);
	const auto hits = ring.Query(0x19000, 0x1000, 10);
	Check(hits.size() == 3, "map, file-read, unmap touch 0x19000");
	Check(hits.front().kind == Kind::Map && hits.back().kind == Kind::Unmap, "oldest first");
	Check(ring.Query(0x19000, 0x1000, 2).size() == 2 && ring.Query(0x19000, 0x1000, 2).front().kind == Kind::FileRead,
	      "max_events keeps the newest ones");
	Check(ring.CountInRange(Kind::PoolMoveEinval, 0x12000, 0x1000) == 1, "one MOVE event with dst in the range");
	Check(ring.CountInRange(Kind::PoolMoveEinval, 0, UINT64_MAX) == 2, "both ends of the MOVE");
	Check(ring.Total(Kind::PoolMoveEinval) == 2, "lifetime counter");

	// merge of identical consecutive events
	for (int i = 0; i < 5; i++) ring.Record(Kind::CpuWrite, 20, 0x30000, 0x1000);
	ring.Record(Kind::CpuWrite, 20, 0x40000, 0x1000);
	const auto cpu = ring.Query(0x30000, 0x1000, 10);
	Check(cpu.size() == 1 && cpu[0].count == 5, "5 identical CPU writes merge into one event with count 5");
	Check(ring.Query(0x40000, 0x1000, 10).size() == 1 && ring.Query(0x40000, 0x1000, 10)[0].count == 1, "a different address does not merge");
	Check(ring.Total(Kind::CpuWrite) == 6, "merged events still count");
}

// S125: the [watch-tex] history: MOVE->EINVAL counted globally and in range, first / last per kind, newest events in range.
void TestWatchLines() {
	Ring ring(64);
	ring.Record(Kind::Map, 3, 0x100000, 0x200000, 0x7000, 0, "pool");
	ring.Record(Kind::PoolMoveEinval, 5, 0x140000, 0x10000, 0x900000, -22, "dst");
	ring.Record(Kind::PoolMoveEinval, 5, 0x900000, 0x10000, 0x140000, -22, "src");
	ring.Record(Kind::PoolMoveEinval, 6, 0x800000, 0x10000, 0xa00000, -22, "dst"); // outside
	ring.Record(Kind::CpuWrite, 7, 0x150000, 0x1000);
	ring.Record(Kind::CpuWrite, 9, 0x150000, 0x2000);
	Event first, last;
	Check(ring.FirstLast(Kind::CpuWrite, 0x100000, 0x200000, &first, &last) && first.flip == 7 && last.flip == 9, "first/last CPU write flips");
	Check(!ring.FirstLast(Kind::FileRead, 0x100000, 0x200000, &first, &last), "no file read in range");
	const auto lines = WatchLines(ring, 0x100000, 0x200000, 3);
	bool       moves = false, events = false;
	size_t     event_lines = 0;
	for (const auto& line : lines) {
		if (line.find("MOVE->EINVAL global=3 in_range=1") != std::string::npos) moves = true;
		if (line.rfind("event ", 0) == 0) event_lines++;
		if (line.find("cpu-write") != std::string::npos && line.rfind("event ", 0) == 0 && line.find("flip=9") != std::string::npos) events = true;
	}
	Check(moves, "MOVE->EINVAL counters (global 3, in range 1)");
	Check(event_lines == 3 && events, "the newest 3 events in range, including the last CPU write");
}

void TestWrap() {
	Ring ring(4);
	for (uint64_t i = 0; i < 10; i++) ring.Record(Kind::Map, i, 0x1000 * (i + 1), 0x1000);
	Check(ring.Wrapped() && ring.Recorded() == 10, "wrapped");
	Check(ring.Query(0, UINT64_MAX, 100).size() == 4, "only the last 4 survive");
	Check(ring.Query(0x1000, 0x1000, 10).empty(), "the oldest event is gone");
	Check(ring.Query(0xa000, 0x1000, 10).size() == 1, "the newest is there");
}

} // namespace

int main() {
	TestIntersects();
	TestRing();
	TestWatchLines();
	TestWrap();
	std::puts("MemEventsTests: ok");
	return 0;
}
