#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_EAGERPAGES_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_EAGERPAGES_H_

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::EagerPages {

// KYTY_DCC_EAGER=1: a readback of a range of several tracker pages (DCC metadata is 2..72 pages) is served by the eager downloads
// already queued for its pages. One page is state of the tracker + the eager table at the moment of the request.
// KYTY_DCC_EAGER=1, read once.
inline bool EagerEnv() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DCC_EAGER");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return on;
}

struct PageState {
	bool     gpu_modified = false; // the tracker still marks the page GPU-modified
	uint64_t eager_tick   = 0;     // submission carrying the page's queued download; 0 = none in flight
};

struct Plan {
	bool     wait     = false; // every page is ready (clean or has an eager download in flight): waiting for max_tick can serve it
	uint64_t max_tick = 0;
};

// Every page must be clean or have a download in flight; the wait is for the newest of those downloads.
inline Plan MakePlan(const std::vector<PageState>& pages) {
	Plan plan {};
	if (pages.empty()) {
		return plan;
	}
	bool ready = true;
	for (const auto& page: pages) {
		if (page.eager_tick != 0) {
			plan.max_tick = std::max(plan.max_tick, page.eager_tick);
		} else if (page.gpu_modified) {
#ifdef KYTY_EAGERPAGES_MUTANT_ONE_PAGE_ENOUGH
			continue; // mutant: one page with a download is enough, the others are assumed to follow
#else
			ready = false;
#endif
		}
	}
	plan.wait = ready && plan.max_tick != 0;
#ifdef KYTY_EAGERPAGES_MUTANT_ONE_PAGE_ENOUGH
	plan.wait = plan.max_tick != 0;
#endif
	return plan;
}

// After waiting: the readback is served only when no page is GPU-modified any more (a GPU write after the download keeps its page
// marked, so that range takes the synchronous path).
inline bool Served(const std::vector<PageState>& after) {
	return std::none_of(after.begin(), after.end(), [](const PageState& page) { return page.gpu_modified; });
}

struct Run {
	uint64_t begin = 0;
	uint64_t end   = 0;
};

// Adjacent pages of one buffer are downloaded with one copy: sorted page addresses -> runs [begin, end).
inline std::vector<Run> MergePages(std::vector<uint64_t> pages, uint64_t page_size) {
	std::sort(pages.begin(), pages.end());
	std::vector<Run> runs;
	for (const auto page: pages) {
		if (!runs.empty() && runs.back().end == page) {
			runs.back().end = page + page_size;
		} else {
			runs.push_back({page, page + page_size});
		}
	}
	return runs;
}

} // namespace Libs::Graphics::EagerPages

#endif
