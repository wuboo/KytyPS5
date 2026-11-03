// S497: KYTY_READ_HOT book (PR #2037). The mutant builds must exit 1 (ctest WILL_FAIL).
// NO_NOTE: a guest read is forgotten, so a later GPU write is not queued.
// CONFIRM_SUPERSEDED: a copy superseded by a newer write is treated as current.
// DROP_OVER_BUDGET: a piece that does not fit the batch is discarded instead of staying queued.
// NOOP_INFLIGHT: a read of an in-flight publish is treated as nothing to download.

#include "graphics/host_gpu/renderer/cache/readHot.h"

#include <cstdio>
#include <memory>

using Libs::Graphics::ReadHot::Book;
using Libs::Graphics::ReadHot::Clears;
using Libs::Graphics::ReadHot::Entry;
using Libs::Graphics::ReadHot::EnvOn;
using Libs::Graphics::ReadHot::InflightRead;
using Libs::Graphics::ReadHot::kPage;
using Libs::Graphics::ReadHot::ReadWhilePublished;

static int g_fails = 0;

static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "read_hot: %s\n", what);
		g_fails++;
	}
}

int main() {
	Check(!EnvOn(nullptr), "unset is off");
	Check(!EnvOn("0"), "0 is off");
	Check(!EnvOn(""), "empty is off");
	Check(!EnvOn("yes"), "yes is off");
	Check(!EnvOn("11"), "11 is off");
	Check(EnvOn("1"), "1 is on");

	{
		Book book;
		Check(book.Note(0x10000, 17 * kPage) == 0, "a span wider than 16 pages is not noted");
		Check(book.HotPages() == 0, "wide span leaves no hot page");
		Check(book.Note(0x10000, 16 * kPage) == 16, "16 pages are noted");
		Check(book.HotPages() == 16, "hot page count");
		book.QueueWrite(0x10000, 64);
		Check(book.PublishIntersects(0x10000, kPage), "a GPU write to a hot page is queued");
		Book cold;
		cold.QueueWrite(0x20000, kPage);
		Check(!cold.PublishIntersects(0x20000, kPage), "a GPU write to a cold page is not queued");
	}

	{
		Book book;
		book.Note(0x30000, kPage);
		for (int i = 0; i < 8; i++) {
			book.Decay();
		}
		Check(book.HotPages() == 1, "a page noted this epoch survives 8 flips");
		book.Decay();
		Check(book.HotPages() == 0, "the page drops on the 9th flip");
		book.QueueWrite(0x30000, kPage);
		Check(!book.PublishIntersects(0x30000, kPage), "a decayed page is not queued");
	}

	{
		Book book;
		constexpr uint64_t base = 0x80000000ull;
		constexpr int      pages = 257;
		for (int i = 0; i < pages; i++) {
			Check(book.Note(base + static_cast<uint64_t>(i) * kPage, 4) == 1, "one page noted");
		}
		book.QueueWrite(base, static_cast<uint64_t>(pages) * kPage);
		book.BeginPlan();
		Libs::Graphics::RangeSet retained;
		int                      accepted = 0;
		book.ForEachPublish([&](uint64_t start, uint64_t end) {
			for (auto page = start; page < end; page += kPage) {
				if (book.Accept(page, kPage, &retained)) {
					accepted++;
				}
			}
		});
		Check(accepted > 0, "part of the write fits the budget");
		Check(accepted < pages, "the budget does not take every page");
		Check(!retained.Empty(), "the overflow stays queued");
		Check(retained.Intersects(base + 256ull * kPage, kPage), "the last page is the overflow");
		book.FinishPlan(std::move(retained), true);
		Check(book.PublishIntersects(base + 256ull * kPage, kPage), "finish keeps the overflow");
		Check(!book.PublishIntersects(base, kPage), "finish drops the piece that was taken");
	}

	{
		Book book;
		auto batch = std::make_shared<Libs::Graphics::ReadHot::Batch>();
		batch->push_back(Entry {0x10000, kPage, false});
		book.Remember(batch);
		Check(Clears((*batch)[0]), "a fresh copy may clear the page");
		book.QueueWrite(0x10000, 4);
		Check(!Clears((*batch)[0]), "a newer write supersedes the copy");
		book.SupersedeAll();
		Check(!Clears((*batch)[0]), "an unknown address write supersedes every copy");
	}

	Check(ReadWhilePublished(true, false, true) == InflightRead::Wait,
	      "a read of an in-flight publish waits instead of nooping");
	Check(ReadWhilePublished(true, false, false) == InflightRead::Wait,
	      "a read after the copy landed still waits for confirm");
	Check(ReadWhilePublished(true, true, true) == InflightRead::Absent,
	      "dirty bytes are still a download, not this wait");
	Check(ReadWhilePublished(false, false, true) == InflightRead::Absent,
	      "a clean tracker is not this wait");

	return g_fails == 0 ? 0 : 1;
}
