#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_SRTREADPASS_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_SRTREADPASS_H_

#include <array>
#include <cstdint>

namespace Libs::LibKernel::Memory {

// S425: a page is quiet for the pass when the GPU has no unflushed bytes in it and (KYTY_CP_READ_BACKING != 0) no GPU->host copy of it is
// still in flight; only then does TryReadBufferBacking do nothing but TryReadBacking for a read inside it.
constexpr bool SrtPageQuiet(bool gpu_dirty, bool inflight) {
#ifdef KYTY_SRTMEMO_MUTANT_INFLIGHT_IGNORED
	(void)inflight;
	return !gpu_dirty;
#else
	return !gpu_dirty && !inflight;
#endif
}

struct SrtReadPass {
	static constexpr uint64_t PageSize = 4096;
	std::array<uint64_t, 32>  clean {};
	uint32_t                  count         = 0;
	uint32_t                  pages_checked = 0;
	uint32_t                  page_hits     = 0;
	bool IsClean(uint64_t page) const {
#ifdef KYTY_SRTMEMO_MUTANT_PAGE_ANY
		return count != 0;
#else
		for (uint32_t i = 0; i < count; ++i) {
			if (clean[i] == page) return true;
		}
		return false;
#endif
	}
	void AddClean(uint64_t page) {
		if (count < clean.size()) clean[count++] = page;
	}
};

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_SRTREADPASS_H_ */
