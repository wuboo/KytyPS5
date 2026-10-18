// S406: GPU busy-time estimate from timeline completions (gpuDone.h) with a fake clock.
// Mutant KYTY_GPUDONE_MUTANT_NO_MAX (no max with the previous completion) must fail the overlapping case.

#include "graphics/host_gpu/renderer/gpuDone.h"

#include <atomic>
#include <cstdio>

using namespace Libs::Graphics::GpuDone;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	{ // disjoint buffers: busy = sum of (done - submit)
		Estimator e;
		e.NoteDone(0, 100);
		e.NoteDone(200, 250);
		Check(e.ActiveNs() == 150, "disjoint: 100 + 50");
	}
	{ // overlapping: the second one starts when the first finished
		Estimator e;
		e.NoteDone(0, 100);
		e.NoteDone(10, 150);
		Check(e.ActiveNs() == 150, "overlapping: 100 + 50, not 100 + 140");
		Check(e.Count() == 2, "count");
	}
	{ // a buffer completed before the previous one (cannot on a serial queue) adds nothing and does not move time back
		Estimator e;
		e.NoteDone(0, 100);
		e.NoteDone(10, 90);
		e.NoteDone(20, 130);
		Check(e.ActiveNs() == 100 + 0 + 30, "out-of-order completion");
	}
	{ // the tracker thread with a fake clock and a wait that advances it
		std::atomic<int64_t> clock {0};
		std::atomic<uint64_t> busy {0};
		Tracker t([&](uint64_t value) { clock.store(static_cast<int64_t>(value) * 100); }, [&] { return clock.load(); },
		          [&](uint64_t ns) { busy.fetch_add(ns); });
		clock.store(0);
		t.NoteSubmit(1); // submitted at 0 -> done at 100
		for (int i = 0; i < 1000 && t.Count() < 1; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		Check(t.Count() == 1 && t.ActiveNs() == 100 && busy.load() == 100, "tracker: one buffer");
	}
	if (g_failures != 0) return 1;
	std::fprintf(stderr, "GpuDoneTests: ok\n");
	return 0;
}
