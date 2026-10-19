#include "graphics/host_gpu/renderer/cache/imagePageSummary.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace {

using Libs::Graphics::FastHandler::ImagePageSummary;
using Libs::Graphics::FastHandler::Mode;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ImagePageSummaryTests: failed: %s\n", text);
		std::exit(1); // a mutant must fail with an exit code (an abort is not inverted by WILL_FAIL)
	}
}

void TestCoverage() {
	ImagePageSummary summary;
	constexpr uint64_t base = 0x0000000240000000ull;
	Check(!summary.MayCover(base, 1) && !summary.MayCover(base, 1 << 20), "an empty filter covers a page");
	summary.Add(base + 0x1800, 0x2000); // pages base+1 .. base+3 (partial first and last page)
	Check(!summary.MayCover(base, 0x1000), "page before the image is covered");
	Check(summary.MayCover(base + 0x1000, 1) && summary.MayCover(base + 0x2fff, 1) && summary.MayCover(base + 0x3000, 1),
	      "page of the image is not covered");
	Check(summary.MayCover(base, 0x4000), "a range that contains an image page is not covered");
	Check(!summary.MayCover(base + 0x4000, 0x1000), "page after the image is covered");
	// two images on one page: the page stays covered until both are gone
	summary.Add(base + 0x3000, 0x100);
	summary.Remove(base + 0x1800, 0x2000);
	Check(summary.MayCover(base + 0x3000, 1) && !summary.MayCover(base + 0x1000, 1), "removing one of two images uncovered the shared page");
	summary.Remove(base + 0x3000, 0x100);
	Check(!summary.MayCover(base, 0x8000), "page is covered after every image was removed");
}

void TestMode() {
	Check(Libs::Graphics::FastHandler::ParseMode(nullptr) == Mode::Off && Libs::Graphics::FastHandler::ParseMode("0") == Mode::Off &&
	          Libs::Graphics::FastHandler::ParseMode("1") == Mode::On && Libs::Graphics::FastHandler::ParseMode("shadow") == Mode::Shadow,
	      "mode parsing");
}

// ns per page check: the filter against the spinlock-guarded page-owner lookup the handler used to make for every fault (information only).
void Bench() {
	ImagePageSummary summary;
	for (uint64_t i = 0; i < 2000; i++) {
		summary.Add(0x0000000240000000ull + i * 0x100000ull, 0x40000);
	}
	std::atomic_flag lock = ATOMIC_FLAG_INIT;
	volatile uint64_t sink = 0;
	constexpr uint64_t n = 4000000;
	const auto t0 = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < n; i++) {
		sink = sink + (summary.MayCover(0x0000000240080000ull + (i & 0xfff) * 0x1000ull * 7, 1) ? 1 : 0);
	}
	const auto t1 = std::chrono::steady_clock::now();
	for (uint64_t i = 0; i < n; i++) {
		while (lock.test_and_set(std::memory_order_acquire)) {
		}
		sink = sink + (i & 1);
		lock.clear(std::memory_order_release);
	}
	const auto t2 = std::chrono::steady_clock::now();
	std::printf("ImagePageSummaryTests: MayCover %.1f ns/check, bare uncontended spinlock round trip %.1f ns\n",
	            std::chrono::duration<double, std::nano>(t1 - t0).count() / n, std::chrono::duration<double, std::nano>(t2 - t1).count() / n);
}

} // namespace

int main() {
	TestCoverage();
	TestMode();
	Bench();
	std::puts("ImagePageSummaryTests: all cases passed");
	return 0;
}
