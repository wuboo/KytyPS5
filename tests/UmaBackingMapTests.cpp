#include "graphics/host_gpu/memory/umaBackingMap.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

using Map = Libs::Graphics::UmaBackingMap<14>;
constexpr uint64_t P = 0x4000;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "UmaBackingMapTests: failed: %s\n", message);
		std::abort();
	}
}

uint64_t Offset(const Map& map, uint64_t page, bool* ok) {
	uint64_t offset = 0;
	*ok             = map.PageOffset(page, &offset);
	return offset;
}

void TestWholePages() {
	Map map;
	map.Map(0x10000000, 4 * P, 0x100000);
	bool ok = false;
	Check(Offset(map, 0x10000000, &ok) == 0x100000 && ok, "first page");
	Check(Offset(map, 0x10000000 + 3 * P, &ok) == 0x100000 + 3 * P && ok, "last page");
	(void)Offset(map, 0x10000000 + 4 * P, &ok);
	Check(!ok, "page after the run is not mapped");
	(void)Offset(map, 0x10000000 - P, &ok);
	Check(!ok, "page before the run is not mapped");
}

void TestPartialPageRejected() {
	// A 4 KB guest mapping inside a 16 KB cache page must not publish the page.
	Map map;
	map.Map(0x20001000, 0x1000, 0x5000);
	bool ok = true;
	(void)Offset(map, 0x20000000, &ok);
	Check(!ok, "4 KB mapping does not cover a 16 KB page");
	// Filling the rest contiguously in guest address AND backing completes it.
	map.Map(0x20000000, 0x1000, 0x4000);
	map.Map(0x20002000, 0x2000, 0x6000);
	(void)Offset(map, 0x20000000, &ok);
	Check(ok, "neighbouring 4 KB runs merge into a full page");
	Check(map.RunCount() == 1, "runs coalesced");
}

void TestNonContiguousBackingRejected() {
	Map map;
	map.Map(0x30000000, P / 2, 0x0);
	map.Map(0x30000000 + P / 2, P / 2, 0x900000); // contiguous guest, scattered backing
	bool ok = true;
	(void)Offset(map, 0x30000000, &ok);
	Check(!ok, "page split across non-contiguous backing is not directly usable");
	Check(map.RunCount() == 2, "not merged");
}

void TestUnmapSplits() {
	Map map;
	map.Map(0x40000000, 8 * P, 0x1000000);
	map.Unmap(0x40000000 + 3 * P, 2 * P);
	bool ok = false;
	Check(Offset(map, 0x40000000 + 2 * P, &ok) == 0x1000000 + 2 * P && ok, "left of the hole");
	(void)Offset(map, 0x40000000 + 3 * P, &ok);
	Check(!ok, "hole page 3");
	(void)Offset(map, 0x40000000 + 4 * P, &ok);
	Check(!ok, "hole page 4");
	Check(Offset(map, 0x40000000 + 5 * P, &ok) == 0x1000000 + 5 * P && ok, "right keeps its offset");
	map.Unmap(0x40000000 + 2 * P + 0x1000, 0x1000); // 4 KB hole breaks page 2
	(void)Offset(map, 0x40000000 + 2 * P, &ok);
	Check(!ok, "unmapping 4 KB breaks its page");
}

void TestRemapReplaces() {
	Map map;
	map.Map(0x50000000, 2 * P, 0x0);
	map.Map(0x50000000, 2 * P, 0x800000);
	bool ok = false;
	Check(Offset(map, 0x50000000 + P, &ok) == 0x800000 + P && ok, "remap replaces the offset");
	Check(map.RunCount() == 1, "single run after remap");
}

void TestForEachPageAndUncovered() {
	Map map;
	map.Map(0x60000000 + 0x1000, 3 * P, 0x10000); // unaligned start: pages 1..2 fully inside
	std::vector<std::pair<uint64_t, uint64_t>> pages;
	map.ForEachPage(0x60000000, 5 * P, [&](uint64_t page, uint64_t offset) {
		pages.emplace_back(page, offset);
	});
	Check(pages.size() == 2, "two full pages");
	Check(pages[0].first == 0x60000000 + P && pages[0].second == 0x10000 + P - 0x1000, "page 1 offset");
	Check(pages[1].first == 0x60000000 + 2 * P, "page 2");

	std::vector<std::pair<uint64_t, uint64_t>> gaps;
	map.ForEachUncovered(0x60000000, 5 * P, [&](uint64_t a, uint64_t b) { gaps.emplace_back(a, b); });
	Check(gaps.size() == 2, "gap before and after");
	Check(gaps[0] == std::make_pair<uint64_t, uint64_t>(0x60000000, 0x60000000 + P), "gap before");
	Check(gaps[1] == std::make_pair<uint64_t, uint64_t>(0x60000000 + 3 * P, 0x60000000 + 5 * P),
	      "gap after");

	// Positive control for the checker: an empty map has one gap equal to the whole range.
	Map empty;
	size_t count = 0;
	empty.ForEachUncovered(0x1000, 0x5000, [&](uint64_t a, uint64_t b) {
		Check(a == 0x1000 && b == 0x6000, "whole range uncovered");
		++count;
	});
	Check(count == 1, "one gap");
}

void TestRangeOffset() {
	Map map;
	map.Map(0x10000000, 4 * P, 0x100000);
	uint64_t offset = 0;
	Check(map.RangeOffset(0x10000000 + 0x123, 0x200, &offset) && offset == 0x100123, "byte range");
	Check(map.RangeOffset(0x10000000, 4 * P, &offset), "whole run");
	Check(!map.RangeOffset(0x10000000, 4 * P + 1, &offset), "one byte past the run");
	Check(!map.RangeOffset(0x10000000 - 1, 2, &offset), "starts before the run");
	Check(!map.RangeOffset(0x10000000, 0, &offset), "empty range");
	map.Unmap(0x10000000 + 2 * P, P);
	Check(!map.RangeOffset(0x10000000 + P, 2 * P, &offset), "range spanning a hole");
	Check(map.RangeOffset(0x10000000 + P, P, &offset), "range before the hole");
}

} // namespace

int main() {
	TestRangeOffset();
	TestWholePages();
	TestPartialPageRejected();
	TestNonContiguousBackingRejected();
	TestUnmapSplits();
	TestRemapReplaces();
	TestForEachPageAndUncovered();
	std::puts("UmaBackingMapTests: ok");
	return 0;
}
