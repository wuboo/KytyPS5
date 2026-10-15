#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCASSERT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCASSERT_H_

// S363: KYTY_DESC_ASSERT=1|0|sample (default 1 = the code path as it was, bit for bit).
// The overlap assertions at the top of CommitBindings (scalar resource reads vs image / attachment / shader buffer writes) only EXIT on a
// violation of the recompiler's own invariants: they insert no barrier, change no layout and take no fallback, so skipping them changes nothing in
// a run that does not violate them. 0 = never run them (and do not compute the digest either); sample = run them on the first draw and then on
// every kSamplePeriod-th draw (a safety compromise: a violation that persists is still caught, one that lasts a single draw may not be).
// No Vulkan or emulator dependency: unit-tested on the CPU (tests/DescAssertTests.cpp).

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::DescAssert {

enum class Mode : uint8_t { On, Off, Sample };

constexpr uint64_t kSamplePeriod = 64;

inline Mode ParseMode(const char* value) {
	if (value == nullptr) {
		return Mode::On;
	}
	if (std::strcmp(value, "0") == 0) {
		return Mode::Off;
	}
	if (std::strcmp(value, "sample") == 0) {
		return Mode::Sample;
	}
	return Mode::On;
}

// Read once; only CommitBindings (descriptors.cpp) uses it.
inline Mode GlobalMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_DESC_ASSERT"));
	return mode;
}

// Decides per draw (that has scalar reads) whether the assertions run. `draw` counts the decisions of this policy, starting at 0.
inline bool ShouldRun(Mode mode, uint64_t draw) {
	switch (mode) {
		case Mode::Off:
#ifdef KYTY_DESCASSERT_MUTANT_OFF_RUNS
			return true;
#else
			return false;
#endif
		case Mode::Sample:
#ifdef KYTY_DESCASSERT_MUTANT_SAMPLE_NEVER
			return false;
#else
			return draw % kSamplePeriod == 0;
#endif
		case Mode::On:
		default:
#ifdef KYTY_DESCASSERT_MUTANT_ON_SAMPLES
			return draw % kSamplePeriod == 0;
#else
			return true;
#endif
	}
}

} // namespace Libs::Graphics::DescAssert

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCASSERT_H_
