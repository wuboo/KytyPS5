#include "graphics/host_gpu/renderer/syncCensus.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::SyncCensus;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SyncCensusTests: failed: %s\n", message);
		std::exit(1);
	}
}

// Tick of the newest writer over a range that crosses 4 KiB page boundaries.
void TestTickPages() {
	TickPages pages;
	Check(pages.MaxTick(0x10000, 0x3000) == 0, "empty map: no writer known");
	pages.Note(0x10000, 0x1000, 5);       // page 0x10
	pages.Note(0x11000, 0x2000, 7);       // pages 0x11, 0x12
	Check(pages.MaxTick(0x10000, 0x1000) == 5, "single page");
	Check(pages.MaxTick(0x10800, 0x1000) == 7, "range crossing a page boundary sees the newer page");
	Check(pages.MaxTick(0x10000, 0x3000) == 7, "three pages: newest wins");
	Check(pages.MaxTick(0x13000, 0x1000) == 0, "page after the writes");
	Check(pages.MaxTick(0x12fff, 1) == 7, "last byte of the last written page");
	Check(pages.MaxTick(0x10000, 0) == 0, "empty range");
	pages.Note(0x10ff0, 0x20, 9);         // straddles pages 0x10 and 0x11 with a few bytes each
	Check(pages.MaxTick(0x10000, 0x1000) == 9 && pages.MaxTick(0x11000, 0x1000) == 9, "straddling write marks both pages");
	pages.Note(0x10000, 0x1000, 3);       // an older tick never lowers a page
	Check(pages.MaxTick(0x10000, 0x1000) == 9, "older tick ignored");
	pages.Forget(0x10800, 0x2000);        // covers page 0x11 completely, 0x10 and 0x12 only partly
	Check(pages.MaxTick(0x11000, 0x1000) == 0, "fully covered page forgotten");
	Check(pages.MaxTick(0x10000, 0x1000) == 9 && pages.MaxTick(0x12000, 0x1000) == 7, "edge pages keep their tick");
}

void TestLagAndDone() {
	Check(Lag(10, 7) == 3, "lag in ticks");
	Check(Lag(10, 10) == 0, "writer in the buffer being recorded");
	Check(Lag(10, 0) == 0, "unknown writer");
	Check(Lag(5, 9) == 0, "writer newer than the current tick does not underflow");
	Check(WriterDone(7, 7) && WriterDone(7, 9), "writer completed");
	Check(!WriterDone(8, 7), "writer still running");
	Check(WriterDone(0, 0), "no writer known counts as done");
}

// Tags nest, restore on leaving and are per thread.
void TestSiteTag() {
	Check(CurrentSite() == Site::Other, "no scope: other");
	{
		SiteScope outer(Site::RfGuest);
		Check(CurrentSite() == Site::RfGuest, "outer scope");
		{
			SiteScope inner(Site::Dcc);
			Check(CurrentSite() == Site::Dcc, "inner scope wins");
		}
		Check(CurrentSite() == Site::RfGuest, "leaving the inner scope restores the outer tag");
	}
	Check(CurrentSite() == Site::Other, "leaving every scope resets to other");
}

void TestRepeat() {
	RepeatTracker tracker;
	auto first = tracker.Note(0x1000, 0x100, 10, 111);
	Check(!first.repeat && !first.same, "first read of a range");
	auto again = tracker.Note(0x1000, 0x100, 10, 111);
	Check(again.repeat && again.same, "same range, same flip, same bytes");
	auto changed = tracker.Note(0x1000, 0x100, 10, 222);
	Check(changed.repeat && !changed.same, "same range, same flip, different bytes");
	auto next_flip = tracker.Note(0x1000, 0x100, 11, 222);
	Check(!next_flip.repeat, "another flip is not a repeat");
	Check(!tracker.Note(0x2000, 0x100, 11, 222).repeat, "another range is not a repeat");
	const uint8_t a[] = {1, 2, 3}, b[] = {1, 2, 4};
	Check(HashBytes(a, 3) != HashBytes(b, 3) && HashBytes(a, 3) == HashBytes(a, 3), "hash");
}

} // namespace

int main() {
	TestTickPages();
	TestLagAndDone();
	TestSiteTag();
	TestRepeat();
	return 0;
}
