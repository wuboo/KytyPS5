// S363 KYTY_DESC_ASSERT: which draws run the overlap assertions. Built with -DKYTY_DESCASSERT_MUTANT_<X>=1 each variant must fail (WILL_FAIL in CMake):
// SAMPLE_NEVER, OFF_RUNS, ON_SAMPLES.
#include "graphics/host_gpu/renderer/pipeline/descAssert.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::DescAssert;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DescAssertTests: failed: %s\n", message);
		std::exit(1);
	}
}

uint64_t Runs(Mode mode, uint64_t draws) {
	uint64_t n = 0;
	for (uint64_t draw = 0; draw < draws; ++draw) {
		n += ShouldRun(mode, draw) ? 1 : 0;
	}
	return n;
}

// The assertion of CommitBindings, as it reads: a scalar read range against a written range (a written range of size 0 is skipped).
// ImageRangeOverlaps (imageInfo.h) is `left < right + right_size && right < left + left_size`; copied here to keep the test free of Vulkan headers.
bool Violates(uint64_t read_address, uint64_t read_size, uint64_t written_address, uint64_t written_size) {
	return written_size != 0 && read_address < written_address + written_size && written_address < read_address + read_size;
}

} // namespace

int main() {
	Check(ParseMode(nullptr) == Mode::On, "unset = On (the old path)");
	Check(ParseMode("1") == Mode::On, "1 = On");
	Check(ParseMode("0") == Mode::Off, "0 = Off");
	Check(ParseMode("sample") == Mode::Sample, "sample = Sample");
	Check(ParseMode("junk") == Mode::On, "unknown = On (safe side)");

	Check(Runs(Mode::On, 1000) == 1000, "On: every draw runs");
	Check(Runs(Mode::Off, 1000) == 0, "Off: no draw runs");
	Check(ShouldRun(Mode::Sample, 0), "Sample: the first draw runs");
	Check(Runs(Mode::Sample, 64) == 1, "Sample: 1 of 64");
	Check(Runs(Mode::Sample, 640) == 10, "Sample: 10 of 640");
	Check(!ShouldRun(Mode::Sample, 1) && !ShouldRun(Mode::Sample, 63) && ShouldRun(Mode::Sample, 64), "Sample: period 64");

	// A persistent violation is caught by Sample within one period; the overlap cases the assertions detect.
	Check(Violates(0x1000, 0x100, 0x1080, 0x100), "read overlaps the write start");
	Check(Violates(0x1080, 0x100, 0x1000, 0x100), "read overlaps the write end");
	Check(Violates(0x1000, 0x1000, 0x1400, 0x10), "read contains the write");
	Check(!Violates(0x1000, 0x100, 0x1100, 0x100), "adjacent ranges are disjoint");
	Check(!Violates(0x1000, 0x100, 0x1000, 0), "an empty write is skipped");
	uint64_t caught_at = 0;
	for (uint64_t draw = 1; draw < 1000 && caught_at == 0; ++draw) {
		if (ShouldRun(Mode::Sample, draw) && Violates(0x1000, 0x100, 0x1080, 0x100)) {
			caught_at = draw;
		}
	}
	Check(caught_at == kSamplePeriod, "Sample catches a violation that persists from draw 1 at draw 64");
	return 0;
}
