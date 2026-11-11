#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_CPUALIASPAGES_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_CPUALIASPAGES_H_

// S532. The image page table is 1 MiB (ImagePageTable::kPageBits == 20).
// FindImagesInRegion(address, size, true) therefore returns every image that
// shares that 1 MiB page, and InvalidateCpuAliases then marks it (byte overlap
// → CPU-dirty + untrack, otherwise a head/tail peel or maybe-dirty).
// Upstream 21f3b9de looks up only the written 4 KiB pages, with page_overlap
// left false, so an image on another 4 KiB page of the same 1 MiB is not
// returned. KYTY_FIX_CPU_ALIAS_PAGES=1 selects that lookup. Unset keeps the
// old call. The #2181 clamp in ReleaseTracked* is unchanged.
// KYTY_ALIAS_PAGES_MUTANT_OLD restores the old lookup so the CPU test fails.

#include "common/alignment.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::CpuAliasPages {

inline constexpr uint64_t kPage = 4096;

struct Lookup {
	uint64_t address     = 0;
	uint64_t size        = 0;
	bool     page_overlap = false;
};

inline bool FixOn() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_FIX_CPU_ALIAS_PAGES");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return on;
}

// fix_on is an argument so the CPU test can force the new lookup without the
// environment. The mutant ignores it and always returns the old lookup.
inline Lookup Query(uint64_t address, uint64_t size, bool fix_on) {
#if defined(KYTY_ALIAS_PAGES_MUTANT_OLD)
	(void)fix_on;
	return {address, size, true};
#else
	if (!fix_on || size == 0) {
		return {address, size, true};
	}
	const uint64_t begin = Common::AlignDown(address, kPage);
	const uint64_t end   = Common::AlignUp(address + size, kPage);
	return {begin, end - begin, false};
#endif
}

inline Lookup Query(uint64_t address, uint64_t size) {
	return Query(address, size, FixOn());
}

// True when FindImagesInRegion(query) would return an image registered on the
// 1 MiB pages of its byte range. page_overlap skips the byte test, which is
// how a write on a different 4 KiB page still selects the image.
inline bool Hits(uint64_t image_address, uint64_t image_size, Lookup query) {
	if (query.size == 0 || image_size == 0) {
		return false;
	}
	const uint64_t query_first = query.address >> 20;
	const uint64_t query_last  = (query.address + query.size - 1) >> 20;
	const uint64_t image_first = image_address >> 20;
	const uint64_t image_last  = (image_address + image_size - 1) >> 20;
	if (image_last < query_first || query_last < image_first) {
		return false;
	}
	if (query.page_overlap) {
		return true;
	}
	return image_address < query.address + query.size && query.address < image_address + image_size;
}

} // namespace Libs::Graphics::CpuAliasPages

#endif
