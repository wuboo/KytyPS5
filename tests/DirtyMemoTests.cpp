// S570 micro (gpuDirtyRanges.h): KYTY_DIRTY_ADD_MEMO and KYTY_DIRTY_QUERY_MEMO. A random walk of Add / Subtract / BeginDownloadCopy /
// Intersects over a few overlapping ranges on two sets, memo off and memo on (and shadow), must give the same answers and the same final
// content at every step; the memos must hit (the walk repeats ranges) and shadow must count no disagreement.
// Mutants (WILL_FAIL): KYTY_DIRTYMEMO_MUTANT_NO_SUB_GEN (a removal keeps the add memo), _QUERY_IGNORES_ADD (an Add keeps the query memo).

#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"

#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

using namespace Libs::Graphics;

namespace {

int g_failures = 0;
void Check(bool ok, const char* what, int step) {
	if (!ok) {
		std::fprintf(stderr, "DirtyMemoTests: failed: %s (step %d)\n", what, step);
		if (++g_failures > 20) std::exit(1);
	}
}

std::vector<std::pair<uint64_t, uint64_t>> Content(const GpuDirtyRanges& set) {
	std::vector<std::pair<uint64_t, uint64_t>> out;
	set.Set().ForEach([&](uint64_t begin, uint64_t end) { out.emplace_back(begin, end); });
	return out;
}

uint64_t g_rng = 0x9e3779b97f4a7c15ull;
uint32_t Next() {
	g_rng ^= g_rng << 13u;
	g_rng ^= g_rng >> 7u;
	g_rng ^= g_rng << 17u;
	return static_cast<uint32_t>(g_rng);
}

void Walk(DirtyMemo::Mode add, DirtyMemo::Mode query) {
	GpuDirtyRanges plain;
	GpuDirtyRanges memo;
	plain.SetMemoModes(DirtyMemo::Mode::Off, DirtyMemo::Mode::Off);
	memo.SetMemoModes(add, query);
	// Ranges that overlap and repeat, like the bindings of consecutive draws.
	const std::pair<uint64_t, uint64_t> ranges[] = {{0x10000, 0x100}, {0x10080, 0x100}, {0x20000, 0x1000}, {0x20800, 0x40},
	                                                {0x30000, 0x10},  {0x10000, 0x40},  {0x21000, 0x800},  {0x30008, 0x8}};
	InflightRanges inflight_a, inflight_b;
	for (int step = 0; step < 20000; step++) {
		const auto [addr, size] = ranges[Next() % 8];
		const auto op           = Next() % 16;
		if (op < 6) {
			plain.Add(addr, size);
			memo.Add(addr, size);
		} else if (op < 7) {
			plain.Subtract(addr, size);
			memo.Subtract(addr, size);
		} else if (op < 8) {
			plain.BeginDownloadCopy(inflight_a, addr, size, [](uint64_t, uint64_t) {});
			memo.BeginDownloadCopy(inflight_b, addr, size, [](uint64_t, uint64_t) {});
		} else {
			Check(plain.Intersects(addr, size) == memo.Intersects(addr, size), "Intersects differs", step);
		}
		if ((step & 255) == 0) Check(Content(plain) == Content(memo), "content differs", step);
	}
	Check(Content(plain) == Content(memo), "final content differs", -1);
	const auto stats = memo.TakeMemoStats();
	if (add != DirtyMemo::Mode::Off) Check(stats.add_hit > 1000, "the add memo hits", -1);
	if (query != DirtyMemo::Mode::Off) Check(stats.query_hit > 1000, "the query memo hits", -1);
	Check(stats.add_ne == 0 && stats.query_ne == 0, "shadow: no disagreement", -1);
	const auto none = memo.TakeMemoStats();
	Check(none.add_hit == 0 && none.query_hit == 0, "stats are taken once", -1);
}

void Direct() {
	// The cases the mutants break, step by step.
	GpuDirtyRanges set;
	set.SetMemoModes(DirtyMemo::Mode::On, DirtyMemo::Mode::On);
	set.Add(0x1000, 0x100);
	Check(set.Intersects(0x1000, 0x10), "added range intersects", 1);
	set.Subtract(0x1000, 0x100);
	Check(!set.Intersects(0x1000, 0x10), "subtracted range does not", 2);
	set.Add(0x1000, 0x100); // the add memo must not skip this one: the range was removed since
	Check(set.Intersects(0x1000, 0x10), "re-added range intersects", 3);
	Check(set.Set().Contains(0x1000, 0x100), "re-added range is in the set", 4);
	Check(!set.Intersects(0x5000, 0x10), "another range does not", 5);
	set.Add(0x5000, 0x10); // the query memo must not answer the old "no"
	Check(set.Intersects(0x5000, 0x10), "an Add invalidates the query memo", 6);
}

} // namespace

int main() {
	Direct();
	Walk(DirtyMemo::Mode::On, DirtyMemo::Mode::Off);
	Walk(DirtyMemo::Mode::Off, DirtyMemo::Mode::On);
	Walk(DirtyMemo::Mode::On, DirtyMemo::Mode::On);
	Walk(DirtyMemo::Mode::Shadow, DirtyMemo::Mode::Shadow);
	if (g_failures != 0) {
		std::fprintf(stderr, "DirtyMemoTests: %d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("DirtyMemoTests: OK");
	return 0;
}
