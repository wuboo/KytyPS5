// S441: bookkeeping of cp_frame_ns / guest_lead_ns / thread CPU (cpFrameStats.h). Mutants: KYTY_CPFRAME_MUTANT_{PERCENTILE_LOW,MIN_IS_MAX,
// FIRST_MARK_COUNTS,THREAD_ABSOLUTE,LITE_ABSOLUTE,LITE_NO_RESCAN}. S570: the KYTY_THREAD_CPU=lite pieces (LiteCpu, LitePlan, DeferredFlip).

#include "graphics/presentation/cpFrameStats.h"

#include <cstdio>

using namespace Libs::Graphics::CpFrameStats;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	{
		FrameClock clock;
		Check(!clock.Mark(1000).has_value(), "the first mark gives no interval");
		const auto second = clock.Mark(1000 + 33'000'000);
		Check(second && *second == 33'000'000, "the second mark gives the interval since the first");
		const auto third = clock.Mark(1000 + 33'000'000 + 40'000'000);
		Check(third && *third == 40'000'000, "intervals are between consecutive marks");
	}
	{
		Samples s;
		Check(s.Percentile(50) == 0 && s.Max() == 0 && s.Size() == 0, "no samples: zeros");
		for (uint64_t v = 1; v <= 100; v++) s.Add(v * 10);
		Check(s.Size() == 100 && s.Sum() == 50500, "size and sum");
		Check(s.Percentile(50) == 500, "p50 of 10..1000 is 500");
		Check(s.Percentile(90) == 900, "p90 is 900");
		Check(s.Percentile(99) == 990, "p99 is 990");
		Check(s.Max() == 1000, "max");
		s.Clear();
		Check(s.Size() == 0 && s.Sum() == 0, "clear");
		s.Add(7);
		Check(s.Percentile(99) == 7 && s.Percentile(50) == 7, "one sample is every percentile");
	}
	{
		PerFlipMin m;
		Check(!m.TakeFlip().has_value(), "nothing noted: no value");
		m.Note(500);
		m.Note(120);
		m.Note(900);
		const auto first = m.TakeFlip();
		Check(first && *first == 120, "the minimum of the flip");
		Check(!m.TakeFlip().has_value(), "a new flip starts empty");
		m.Note(30);
		const auto next = m.TakeFlip();
		Check(next && *next == 30, "the next flip is independent of the previous one");
	}
	{
		ThreadCpu cpu;
		const std::vector<ThreadSample> a {{1, "Thread_Gpu", 1000}, {2, "guest", 5000}, {3, "audio", 100}};
		const auto base = cpu.Update(a, "Thread_Gpu");
		Check(base.cp == 0 && base.other == 0, "the first snapshot is a baseline");
		const std::vector<ThreadSample> b {{1, "Thread_Gpu", 1800}, {2, "guest", 5400}, {3, "audio", 130}, {4, "new", 777}};
		const auto d = cpu.Update(b, "Thread_Gpu");
		Check(d.cp == 800, "CP delta is the difference, not the total");
		Check(d.other == 430, "others: 400 + 30, a new thread contributes 0 the first time");
		Check(d.by_name.size() == 4 && d.by_name[0].first == "Thread_Gpu" && d.by_name[1].first == "guest", "names sorted by delta, largest first");
		const std::vector<ThreadSample> c {{1, "Thread_Gpu", 2000}, {2, "guest", 5400}};
		const auto e = cpu.Update(c, "Thread_Gpu");
		Check(e.cp == 200 && e.other == 0, "a thread that left does not break the deltas");
	}
	{
		// S570: per-flip CP time = differences of the cumulative counter; their sum over a window equals the counter's growth.
		LiteCpu lite;
		Check(!lite.Note(5'000'000).has_value(), "lite: the first query is the baseline");
		uint64_t sum = 0, cumulative = 5'000'000;
		for (uint64_t flip = 1; flip <= 50; flip++) {
			cumulative += 40'000'000 + flip * 1000;
			const auto d = lite.Note(cumulative);
			Check(d && *d == 40'000'000 + flip * 1000, "lite: the delta of one flip");
			sum += d ? *d : 0;
		}
		Check(sum == cumulative - 5'000'000, "lite: the deltas add up to the counter's growth");
		const auto back = lite.Note(1000);
		Check(back && *back == 0, "lite: a counter that went back gives 0, not a wrap");
	}
	{
		LitePlan plan;
		uint64_t samples = 0, scans = 0;
		bool     have    = false;
		for (uint64_t flip = 1; flip <= 1800; flip++) {
			if (!plan.Sample(flip)) continue;
			samples++;
			if (plan.FullScan(flip, have)) {
				scans++;
				have = true;
			}
		}
		Check(samples == 180, "lite plan: every 10th flip samples");
		Check(scans == 4, "lite plan: the list is scanned at the first sample (flip 10), then at flips 600, 1200 and 1800");
		Check(plan.FullScan(1800, true), "lite plan: flip 1800 rescans");
		Check(!plan.FullScan(610, true) && plan.FullScan(610, false), "lite plan: no rescan between, except without a list");
	}
	{
		DeferredFlip deferred;
		Check(!deferred.Take().has_value(), "deferred: nothing marked");
		deferred.Mark(9, false);
		Check(!deferred.Take().has_value(), "deferred: a flip that is not due is not taken");
		deferred.Mark(10, true);
		deferred.Mark(11, false);
		const auto t = deferred.Take();
		Check(t && *t == 10, "deferred: the due flip survives a later flip in the same locked section");
		Check(!deferred.Take().has_value(), "deferred: taken once");
	}
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("CpFrameStatsTests: OK");
	return 0;
}
