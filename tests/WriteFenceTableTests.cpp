#include "graphics/host_gpu/memory/writeFenceTable.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Built twice: normally, and with KYTY_UMA_MUTANT_FENCE_TIMELINE (a writer on one timeline is
// reported on timeline 0). The mutant build must FAIL (ctest WILL_FAIL).

namespace {

using Fence = Libs::Graphics::WriteFenceTable;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "WriteFenceTableTests: failed: %s\n", message);
		std::exit(1); // not abort(): ctest WILL_FAIL does not invert signals
	}
}

void TestRanges() {
	Fence fence;
	fence.Record(0x1000, 0x1000, 0, 7);
	Check(fence.MaxTicks(0x1000, 0x10)[0] == 7, "inside");
	Check(fence.MaxTicks(0x2000, 0x10)[0] == 0, "adjacent range is not covered");
	Check(fence.MaxTicks(0x0, 0x1001)[0] == 7, "overlap by one byte");
	fence.Record(0x1000, 0x1000, 0, 9);
	Check(fence.MaxTicks(0x1000, 4)[0] == 9, "newer writer wins");
	fence.Record(0x5000, 0x1000, 0, 8);
	Check(fence.MaxTicks(0x1000, 0x5000)[0] == 9, "max over a span");
	fence.Prune([](uint32_t, uint64_t tick) { return tick <= 8; });
	Check(fence.MaxTicks(0x5000, 4)[0] == 0, "retired entry pruned");
	Check(fence.MaxTicks(0x1000, 4)[0] == 9, "unretired entry kept");

	Fence many;
	for (uint64_t i = 0; i < Fence::MaxEntries + 10; ++i) {
		many.Record(0x10000 * (i + 1), 0x1000, 0, i + 1);
	}
	Check(many.Size() < Fence::MaxEntries, "collapsed");
	Check(many.MaxTicks(0x10000 * 3, 1)[0] == Fence::MaxEntries + 1,
	      "collapsed entry is conservative");
}

void TestTimelines() {
	Fence fence;
	// The same bytes written by a graphics submit (timeline 0) and an async-compute one (1).
	fence.Record(0x1000, 0x1000, 0, 100);
	fence.Record(0x1000, 0x1000, 1, 5);
	auto ticks = fence.MaxTicks(0x1000, 4);
	Check(ticks[0] == 100 && ticks[1] == 5, "each timeline keeps its own tick");
	// A newer writer on timeline 1 does not supersede the timeline-0 entry (ticks are not comparable).
	fence.Record(0x1000, 0x1000, 1, 9);
	ticks = fence.MaxTicks(0x1000, 4);
	Check(ticks[0] == 100 && ticks[1] == 9, "timelines do not supersede each other");
	// Retirement is per timeline.
	fence.Prune([](uint32_t timeline, uint64_t tick) { return timeline == 1 && tick <= 9; });
	ticks = fence.MaxTicks(0x1000, 4);
	Check(ticks[0] == 100 && ticks[1] == 0, "prune per timeline");
	fence.Record(0x9000, 0x10, 99, 1); // unknown timeline is ignored
	Check(fence.MaxTicks(0x9000, 4)[0] == 0, "out-of-range timeline ignored");
}

} // namespace

int main() {
	TestRanges();
	TestTimelines();
	std::puts("WriteFenceTableTests: ok");
	return 0;
}
