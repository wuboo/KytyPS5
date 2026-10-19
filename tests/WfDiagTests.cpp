#include "graphics/host_gpu/renderer/wfDiag.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using namespace Libs::Graphics::WriteFault;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "WfDiagTests: failed: %s\n", message);
		std::exit(1);
	}
}

// A faults in every flip, B in every other one, C only in the last 11: at flip 100 all three faulted (B: even flip), only A in >= 54 of 60 flips.
void TestDistinctAndEveryFlip() {
	Aggregator aggregator;
	Finished   finished;
	bool       rolled = false;
	for (uint64_t flip = 1; flip <= 101; flip++) {
		finished = {};
		const bool r = aggregator.Note(0x10000000, flip, Owner::Buffer, 0, &finished);
		if (flip % 2 == 0) {
			(void)aggregator.Note(0x10001000, flip, Owner::Image, 65536, &finished);
		}
		if (flip >= 90 && flip <= 100) {
			(void)aggregator.Note(0x10002000, flip, Owner::None, 0, &finished);
		}
		if (flip == 101) {
			rolled = r;
			// Finished flip is the one before: the first note of 101 rolled 100.
		}
		if (r && flip == 101) {
			Check(finished.flip == 100, "the first fault of a later flip finishes the previous one");
			Check(finished.distinct == 3, "pages with a fault in the finished flip");
			Check(finished.every == 1, "only the page that faulted in >= 54 of the last 60 flips");
		}
	}
	Check(rolled, "the roll was reported");
}

void TestRepeatedFaultsInOneFlipCountOnce() {
	Aggregator aggregator;
	Finished   finished;
	for (int i = 0; i < 50; i++) {
		(void)aggregator.Note(0x20000010 + static_cast<uint64_t>(i), 5, Owner::Buffer, 0, &finished); // 50 bytes of one page
	}
	Check(aggregator.Note(0x20003000, 6, Owner::Buffer, 0, &finished), "roll");
	Check(finished.distinct == 1 && finished.every == 0, "50 faults on one page in a flip are one page");
	const auto top = aggregator.Top(5);
	Check(top.size() == 2 && top[0].page == 0x20000000 && top[0].faults == 50 && top[0].flips == 1, "faults and flips counted apart");
}

void TestUploadSameFlip() {
	Aggregator aggregator;
	Finished   finished;
	(void)aggregator.Note(0x30000000, 7, Owner::Buffer, 0, &finished);
	(void)aggregator.Note(0x30001000, 7, Owner::Buffer, 0, &finished);
	aggregator.NoteUpload(0x30000800, 0x1000, 7); // touches both pages
	aggregator.NoteUpload(0x30000000, 0x1000, 7); // second upload of the same page in the flip: counted once
	aggregator.NoteUpload(0x30001000, 0x1000, 8); // another flip: not the flip of the fault
	const auto top = aggregator.Top(5);
	Check(top.size() == 2, "two pages");
	uint64_t first = 0, second = 0;
	for (const auto& entry: top) {
		(entry.page == 0x30000000 ? first : second) = entry.upload_flips;
	}
	Check(first == 1 && second == 1, "an upload after a fault in the same flip counts once per flip");
}

void TestTopAndFormat() {
	Aggregator aggregator;
	Finished   finished;
	for (uint64_t page = 0; page < 30; page++) {
		for (uint64_t n = 0; n <= page; n++) {
			(void)aggregator.Note(0x40000000 + page * 0x1000, 1, Owner::Image, 4096, &finished);
		}
	}
	const auto top = aggregator.Top(kTopPages);
	Check(top.size() == kTopPages && top[0].page == 0x40000000 + 29 * 0x1000 && top[0].faults == 30, "most faults first");
	const auto line = FormatLine(top[0], 1);
	Check(line.find("[wf] flip=1 page=0x000000004001d000 owner=img image_size=4096 faults=30") == 0, "line format");
	Check(MakeOwner(true, true) == Owner::Both && MakeOwner(false, false) == Owner::None, "owner mapping");
}

} // namespace

int main() {
	TestDistinctAndEveryFlip();
	TestRepeatedFaultsInOneFlipCountOnce();
	TestUploadSameFlip();
	TestTopAndFormat();
	std::puts("WfDiagTests: all cases passed");
	return 0;
}
