// S406: GPU busy-time estimate from timeline completions (gpuDone.h) with a fake clock.
// Mutant KYTY_GPUDONE_MUTANT_NO_MAX (no max with the previous completion) must fail the overlapping case.
// S410: starve (queue empty before a submit): total and the part after the CP recorded something. Mutant KYTY_GPUDONE_MUTANT_STARVE_NO_FIRSTREC
// (work = total) must fail.

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
	{ // S410: disjoint buffers: the starve is the gap; a first record after the previous completion shortens the work part
		Estimator e;
		Starve    s;
		e.NoteDoneStarve(0, 0, 100, &s);
		Check(s.total_ns == 0 && s.work_ns == 0, "starve: the first buffer has no previous completion");
		e.NoteDoneStarve(300, 250, 350, &s); // first record at 250 (after the previous done at 100), submit at 300
		Check(s.total_ns == 200, "starve: gap between done(v-1)=100 and submit(v)=300");
		Check(s.work_ns == 50, "starve_work: from the first record (250) to the submit (300)");
		e.NoteDoneStarve(400, 120, 450, &s); // first record at 120 < previous done at 350: the whole gap 350..400 is work
		Check(s.total_ns == 50 && s.work_ns == 50, "starve: record started before the previous completion");
		e.NoteDoneStarve(420, 410, 500, &s); // submitted while the previous (done at 450) still ran
		Check(s.total_ns == 0 && s.work_ns == 0, "starve: no gap when the queue was not empty");
		Check(ComputeStarve(500, 100, 0).work_ns == 400, "starve_work: unknown first record (0) = the whole gap");
	}
	{ // S410: the tracker passes the first record and the reason to the starve callback
		std::atomic<int64_t> clock {0};
		std::atomic<uint64_t> total {0}, work {0};
		std::atomic<int> last_reason {-1};
		Tracker t([&](uint64_t value) { clock.store(static_cast<int64_t>(value) * 100); }, [&] { return clock.load(); }, [](uint64_t) {},
		          [&](const Starve& st, SubmitReason r) {
			          total.fetch_add(st.total_ns);
			          work.fetch_add(st.work_ns);
			          last_reason.store(static_cast<int>(r));
		          });
		clock.store(0);
		t.NoteSubmit(1); // done at 100
		for (int i = 0; i < 1000 && t.Count() < 1; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		clock.store(160);
		t.NoteSubmit(2, 130, SubmitReason::Eager3Flush); // submitted at 160, first record at 130, done at 200
		for (int i = 0; i < 1000 && t.Count() < 2; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		for (int i = 0; i < 1000 && last_reason.load() != static_cast<int>(SubmitReason::Eager3Flush); i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		Check(total.load() == 60 && work.load() == 30, "tracker: starve 160-100, work 160-130");
		Check(last_reason.load() == static_cast<int>(SubmitReason::Eager3Flush), "tracker: reason passed on");
	}
	if (g_failures != 0) return 1;
	std::fprintf(stderr, "GpuDoneTests: ok\n");
	return 0;
}
