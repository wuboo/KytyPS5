// S418: ranges of a recorded GPU->host copy that is not in the backing yet (inflightRanges.h), and the backing-read decision built on it.
// Mutants (each must fail): BACKING_IGNORES_INFLIGHT (the backing is read although a copy is in flight),
// REMOVE_ALL_OVERLAPPING (a publish ends every in-flight entry it overlaps, not just its own).

#include "graphics/host_gpu/renderer/cache/inflightRanges.h"

#include <cstdio>

using namespace Libs::Graphics;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	InflightRanges r;
	Check(!r.Intersects(0x1000, 0x100), "empty set intersects nothing");
	r.Add(0x1000, 0x100);
	Check(r.Intersects(0x1000, 0x100), "exact range");
	Check(r.Intersects(0x10ff, 1), "last byte");
	Check(!r.Intersects(0x1100, 0x10), "just past the end");
	Check(!r.Intersects(0x0ff0, 0x10), "just before the begin");
	Check(r.Intersects(0x0ff0, 0x11), "overlap by one byte at the begin");
	Check(r.Intersects(0x0f00, 0x1000), "query covers the entry");
	Check(r.Intersects(0x1040, 0x10), "query inside the entry");
	Check(!r.Intersects(0x1000, 0), "empty query");
	r.Remove(0x1000, 0x100);
	Check(!r.Intersects(0x1000, 0x100), "removed");
	Check(r.Count() == 0, "no entries left");

	// Partial overlap: two different copies, the first publishes.
	r.Add(0x2000, 0x100);
	r.Add(0x2080, 0x100);
	r.Remove(0x2000, 0x100);
	Check(r.Intersects(0x2080, 0x100), "the second copy is still in flight after the first publishes");
	Check(r.Intersects(0x2100, 0x10), "bytes only the second copy covers");
	Check(!r.Intersects(0x2000, 0x80), "bytes only the first copy covered are free");
	Check(r.Count() == 1, "one entry left");
	r.Remove(0x2080, 0x100);

	// The same range twice (dirty, download, dirty again, download): one publish ends one copy.
	r.Add(0x3000, 0x200);
	r.Add(0x3000, 0x200);
	r.Remove(0x3000, 0x200);
	Check(r.Intersects(0x3000, 0x200), "a second copy of the same range is still in flight");
	r.Remove(0x3000, 0x200);
	Check(!r.Intersects(0x3000, 0x200), "both published");

	// Removing something that was never added changes nothing.
	r.Add(0x4000, 0x10);
	r.Remove(0x4000, 0x20);
	Check(r.Intersects(0x4000, 0x10), "remove of an unknown range is a no-op");
	r.Remove(0x4000, 0x10);

	Check(CanReadBackingDecision(false, false), "clean and nothing in flight: the backing is current");
	Check(!CanReadBackingDecision(true, false), "GPU-dirty: the backing is stale");
	Check(!CanReadBackingDecision(false, true), "copy in flight: the backing is stale although the tracker is clear");
	Check(!CanReadBackingDecision(true, true), "both");

	if (g_failures != 0) return 1;
	std::fprintf(stderr, "InflightRangesTests: ok\n");
	return 0;
}
