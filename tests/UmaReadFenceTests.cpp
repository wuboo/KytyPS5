#include "graphics/host_gpu/memory/umaReadFence.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Built normally and once per mutant (KYTY_UMA_MUTANT_IGNORE_PENDING_WRITER, KYTY_UMA_MUTANT_CHECK_ALWAYS_BIG,
// KYTY_UMA_MUTANT_DIFF_IGNORES_TAIL). Every mutant build must FAIL (ctest WILL_FAIL).

namespace {

namespace Fence = Libs::Graphics::UmaReadFence;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "UmaReadFenceTests: failed: %s\n", message);
		std::exit(1);
	}
}

void TestGpuDirty() {
	Check(Fence::GpuDirty(true, true, false), "stage 3: a writer that has not retired is dirty");
	Check(!Fence::GpuDirty(true, false, false), "stage 3: no writer, nothing tracked is clean");
	Check(Fence::GpuDirty(true, false, true), "a tracked range is dirty");
	Check(!Fence::GpuDirty(false, true, false), "copy path: a pending writer set means nothing without stage 3");
	Check(Fence::GpuDirty(false, false, true), "copy path: a tracked range is dirty");
}

void TestShouldCheck() {
	Check(Fence::ShouldCheck(4, 4, 64, 65536), "every 4th call is checked");
	Check(!Fence::ShouldCheck(3, 4, 64, 65536), "other calls are not");
	Check(!Fence::ShouldCheck(4, 4, 65537, 65536), "a read over the cap is never checked");
	Check(!Fence::ShouldCheck(4, 4, 0, 65536), "an empty read is never checked");
	Check(!Fence::ShouldCheck(4, 0, 64, 65536), "every = 0 disables the check");
}

void TestDiffers() {
	const uint8_t a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	uint8_t       b[8] = {1, 2, 3, 4, 5, 6, 7, 8};
	size_t        first = 99;
	Check(!Fence::Differs(a, b, 8, &first), "equal bytes do not differ");
	b[7] = 9;
	Check(Fence::Differs(a, b, 8, &first) && first == 7, "a difference in the last byte is found at offset 7");
	b[7] = 8;
	b[2] = 0;
	Check(Fence::Differs(a, b, 8, &first) && first == 2, "the first differing offset is reported");
}

} // namespace

int main() {
	TestGpuDirty();
	TestShouldCheck();
	TestDiffers();
	std::puts("UmaReadFenceTests: ok");
	return 0;
}
