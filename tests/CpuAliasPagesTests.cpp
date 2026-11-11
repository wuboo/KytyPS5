#include "graphics/host_gpu/renderer/cache/cpuAliasPages.h"

#include <cstdint>
#include <cstdio>

using Libs::Graphics::CpuAliasPages::Hits;
using Libs::Graphics::CpuAliasPages::Query;

static int g_fails = 0;

static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "cpu_alias_pages: %s\n", what);
		g_fails++;
	}
}

int main() {
	// One 4 KiB image and a write on a different 4 KiB page of the same 1 MiB.
	constexpr uint64_t kBase  = 0x200000ull;
	constexpr uint64_t kImage = kBase + 0x2000ull;
	constexpr uint64_t kWrite = kBase + 0x5000ull;
	constexpr uint64_t kImageSize = 0x1000ull;
	constexpr uint64_t kWriteSize = 64ull;

	const auto fixed = Query(kWrite, kWriteSize, true);
	Check(fixed.page_overlap == false, "fix lookup is not a whole 1 MiB page");
	Check(fixed.address == kWrite && fixed.size == 0x1000ull, "fix lookup is the written 4 KiB page");
	Check(!Hits(kImage, kImageSize, fixed), "other 4 KiB page of the same 1 MiB is not marked");

	const auto old = Query(kWrite, kWriteSize, false);
	Check(old.page_overlap == true && old.address == kWrite && old.size == kWriteSize,
	      "fix off keeps the old lookup");
	Check(Hits(kImage, kImageSize, old), "old lookup marks the other 4 KiB page");

	// A write on the image's own 4 KiB page is still selected, including a write
	// that misses the image bytes but shares that page (the peel stays).
	const auto same = Query(kImage + 0x100ull, 16ull, true);
	Check(Hits(kImage, kImageSize, same), "write on the image 4 KiB page is still selected");
	constexpr uint64_t kPartial = kImage + 0x100ull;
	const auto peel = Query(kImage + 0x800ull, 16ull, true);
	Check(Hits(kPartial, 0x100ull, peel), "same 4 KiB page still peels");

	// A different 1 MiB page is not selected either way.
	const auto other = Query(kBase + 0x100000ull + 0x100ull, 16ull, true);
	Check(!Hits(kImage, kImageSize, other), "another 1 MiB page is not selected");

	if (g_fails != 0) {
		std::fprintf(stderr, "cpu_alias_pages: %d failed\n", g_fails);
		return 1;
	}
	return 0;
}
