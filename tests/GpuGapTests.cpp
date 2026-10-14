#include "graphics/host_gpu/renderer/gpuGap.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

// S327: gap bookkeeping of GpuTimeline. Built twice: normally and with KYTY_GAP_MUTANT_ARM_FIRST (arm the new gap before harvesting).
// The mutant must FAIL (ctest WILL_FAIL).

namespace {

using namespace Libs::Graphics::GpuGap;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "GpuGapTests: failed: %s\n", message);
		std::exit(1); // not abort(): ctest WILL_FAIL does not invert signals
	}
}

struct Cb {
	uint64_t tick, begin, end;
};

// Buffers recorded but not read back yet; Harvest hands them to the ledger like GpuTimeline does.
struct Sim {
	Ledger          ledger;
	Totals          totals;
	std::vector<Cb> pending;
	void Record(Cb cb) { pending.push_back(cb); }
	void Harvest() {
		for (const auto& cb: pending) ledger.Interval(cb.tick, cb.begin, cb.end, totals);
		pending.clear();
	}
	void Drain(uint64_t next_tick) { NoteDrain(ledger, next_tick, [&] { Harvest(); }); }
};

} // namespace

int main() {
	{
		// Two drains in a row with nothing harvested in between (Harvest ran only at Submit in S326): both segments are counted once.
		Sim s;
		s.Record({1, 0, 10});
		s.Drain(2);            // gap A armed for tick 2
		s.Record({2, 30, 40}); // gap A = 30 - 10 = 20
		s.Drain(3);            // must count A before arming B
		s.Record({3, 100, 110});
		s.Harvest();           // gap B = 100 - 40 = 60
		Check(s.totals.gap_n == 2, "two drain gaps counted");
		Check(s.totals.gap_ns == 80, "drain gaps 20 + 60");
		Check(s.totals.mid_ns == 0, "no mid gaps");
	}
	{
		// A segment with several buffers: only the first one after the drain is a drain gap, the others are mid gaps.
		Sim s;
		s.Record({1, 0, 10});
		s.Drain(2);
		s.Record({2, 25, 30});  // drain gap 15
		s.Record({3, 33, 40});  // mid gap 3
		s.Record({4, 40, 50});  // touching: 0
		s.Record({5, 58, 60});  // mid gap 8
		s.Harvest();
		Check(s.totals.gap_n == 1 && s.totals.gap_ns == 15, "drain gap 15 counted once");
		Check(s.totals.mid_ns == 11, "mid gaps 3 + 8");
	}
	{
		// Overlapping buffers (end after the next begin) give no negative gap and keep the latest end.
		Sim s;
		s.Record({1, 0, 50});
		s.Record({2, 10, 20}); // inside: no gap
		s.Record({3, 55, 60}); // gap measured from 50, not from 20
		s.Harvest();
		Check(s.totals.mid_ns == 5, "mid gap from the latest end");
	}
	std::printf("GpuGapTests: ok\n");
	return 0;
}
