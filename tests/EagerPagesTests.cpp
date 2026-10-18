#include "graphics/host_gpu/renderer/cache/eagerPages.h"

#include <cstdio>
#include <cstdlib>

// S402: KYTY_DCC_EAGER. A readback of a multi-page range is served by queued eager downloads only when every page is ready.

namespace {

using namespace Libs::Graphics::EagerPages;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "EagerPagesTests: failed: %s\n", message);
		std::exit(1);
	}
}

void TestAllPagesReady() {
	// Four pages: two with a download in flight (still marked), one clean, one with a newer download.
	std::vector<PageState> pages {{true, 5}, {true, 5}, {false, 0}, {true, 6}};
	const auto             plan = MakePlan(pages);
	Check(plan.wait && plan.max_tick == 6, "every page ready: wait for the newest download");
	std::vector<PageState> after {{false, 0}, {false, 0}, {false, 0}, {false, 0}};
	Check(Served(after), "all pages clean after the wait: served");
}

void TestOnePageWithoutEagerMisses() {
	std::vector<PageState> pages {{true, 5}, {true, 0}, {true, 5}};
	Check(!MakePlan(pages).wait, "a GPU-modified page without a download in flight: miss");
	std::vector<PageState> one {{true, 0}};
	Check(!MakePlan(one).wait, "a single page without a download: miss");
}

void TestNewWriteAfterDownloadMisses() {
	std::vector<PageState> pages {{true, 5}, {true, 5}};
	Check(MakePlan(pages).wait, "planned");
	std::vector<PageState> after {{false, 0}, {true, 0}}; // the GPU wrote the second page again after its download
	Check(!Served(after), "a page written again after the download is still GPU-modified: miss");
}

void TestNothingToWaitFor() {
	std::vector<PageState> clean {{false, 0}, {false, 0}};
	Check(!MakePlan(clean).wait, "clean pages and no download: nothing to wait for");
	Check(!MakePlan({}).wait, "empty range");
}

void TestMerge() {
	const auto runs = MergePages({0x4000, 0x1000, 0x2000, 0x3000, 0x8000}, 0x1000);
	Check(runs.size() == 2, "two runs");
	Check(runs[0].begin == 0x1000 && runs[0].end == 0x5000, "adjacent pages merge, whatever the order");
	Check(runs[1].begin == 0x8000 && runs[1].end == 0x9000, "a gap starts a new run");
	Check(MergePages({}, 0x1000).empty(), "no pages, no runs");
}

} // namespace

int main() {
	TestAllPagesReady();
	TestOnePageWithoutEagerMisses();
	TestNewWriteAfterDownloadMisses();
	TestNothingToWaitFor();
	TestMerge();
	return 0;
}
