#include "graphics/host_gpu/memory/umaBarrierDiag.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Built twice: normally, and with KYTY_UMA_MUTANT_BARRIER_EPOCH_IGNORED (a write with no barrier
// since is reported as covered). The mutant build must FAIL (ctest WILL_FAIL).

namespace {

using namespace Libs::Graphics::UmaBarrierDiag;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "UmaBarrierDiagTests: failed: %s\n", message);
		std::exit(1);
	}
}

void TestVerdicts() {
	SameSubmitTracker tracker;
	// Positive control: a write, then a read of the same bytes with no barrier recorded.
	tracker.NoteWrite(/*tick*/ 10, 0x1000, 0x100, /*epoch*/ 5);
	Check(tracker.Check(10, 0x1000, 0x100, 5) == Verdict::Missing, "read right after the write is missing a barrier");
	Check(tracker.Check(10, 0x10f0, 0x20, 5) == Verdict::Missing, "an overlap of a few bytes counts");
	Check(tracker.Check(10, 0x1100, 0x100, 5) == Verdict::None, "bytes past the write are untouched");
	Check(tracker.Check(10, 0x0f00, 0x100, 5) == Verdict::None, "bytes before the write are untouched");
	// A barrier was recorded in between.
	Check(tracker.Check(10, 0x1000, 0x100, 6) == Verdict::Covered, "a barrier after the write covers the read");
	// Two writers: the old one is covered, the new one is not.
	tracker.NoteWrite(10, 0x2000, 0x100, 6);
	Check(tracker.Check(10, 0x1000, 0x2000, 6) == Verdict::Missing, "one uncovered writer in the range is enough");
	Check(tracker.Check(10, 0x1000, 0x100, 6) == Verdict::Covered, "the older writer stays covered");
	// A new submit starts clean.
	Check(tracker.Check(11, 0x1000, 0x100, 6) == Verdict::None, "a new tick forgets the writers");
	Check(tracker.Size() == 0, "the list is cleared");
}

void TestOverflowKeepsWorking() {
	SameSubmitTracker tracker;
	for (uint64_t i = 0; i < SameSubmitTracker::MaxEntries + 10; ++i) {
		tracker.NoteWrite(1, i * 0x1000, 0x100, 3);
	}
	Check(tracker.Dropped() == 1, "overflow drops the oldest half once");
	Check(tracker.Size() <= SameSubmitTracker::MaxEntries, "bounded");
	const auto last = (SameSubmitTracker::MaxEntries + 9) * 0x1000;
	Check(tracker.Check(1, last, 0x100, 3) == Verdict::Missing, "recent writers are kept");
}

void TestEpochCounter() {
	const auto before = Epoch();
	NoteBarrier();
	NoteBarrier();
	Check(Epoch() == before + 2, "each barrier advances the epoch");
}

} // namespace

int main() {
	TestVerdicts();
	TestOverflowKeepsWorking();
	TestEpochCounter();
	std::puts("UmaBarrierDiagTests: ok");
	return 0;
}
