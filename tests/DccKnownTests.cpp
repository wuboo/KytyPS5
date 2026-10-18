#include "graphics/host_gpu/renderer/cache/dccFillPublish.h"
#include "graphics/host_gpu/renderer/cache/gpuWriteLedger.h"

#include <cstdio>
#include <cstdlib>
#include <set>
#include <utility>
#include <vector>

// S402: KYTY_DCC_KNOWN. The ledger (which fill is the newest GPU write of a range), the publish decision and its effect on the
// backing, and a model of what MaterializeColorClear does with the published metadata.

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::DccFillPublish;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DccKnownTests: failed: %s\n", message);
		std::exit(1);
	}
}

constexpr uint64_t kBase = 0x560000000;

void TestLedgerKnown() {
	GpuWriteLedger ledger;
	Check(ledger.QueryFill(kBase, 0x1000).kind == FillKind::Partial, "empty ledger knows nothing");
	ledger.RecordFill(kBase, 0x2000, 0x20202020);
	const auto q = ledger.QueryFill(kBase, 0x2000);
	Check(q.kind == FillKind::Known && q.value == 0x20202020, "a fill is known over its own range");
	Check(ledger.QueryFill(kBase + 0x1000, 0x1000).kind == FillKind::Known, "a fill is known over a sub-range");
	uint32_t value = 0;
	Check(ledger.LatestIsFill(kBase, 0x1000, &value) && value == 0x20202020, "LatestIsFill agrees");
	Check(ledger.QueryFill(kBase + 0x10000, 0x1000).kind == FillKind::Partial, "an unrelated range is unknown");
}

void TestLedgerNewerGpuWriteHides() {
	GpuWriteLedger ledger;
	ledger.RecordFill(kBase, 0x2000, 0x01010101);
	ledger.RecordWrite(kBase + 0x800, 0x100); // a shader writes into the middle
	Check(ledger.QueryFill(kBase, 0x2000).kind == FillKind::NewerWrite, "a newer GPU write hides the fill");
	Check(ledger.QueryFill(kBase + 0x1000, 0x1000).kind == FillKind::Known, "... but only where it overlaps");
	uint32_t value = 0;
	Check(!ledger.LatestIsFill(kBase, 0x2000, &value), "LatestIsFill is false behind a newer write");
}

void TestLedgerCpuWriteHides() {
	GpuWriteLedger ledger;
	ledger.RecordFill(kBase, 0x2000, 0x01010101);
	ledger.Forget(kBase + 0x1000, 0x10); // CPU write / download of part of it
	Check(ledger.QueryFill(kBase, 0x2000).kind != FillKind::Known, "a CPU write hides the fill");
	Check(ledger.QueryFill(kBase + 0x1800, 0x100).kind != FillKind::Known, "the entry no longer describes its bytes");
	ledger.RecordFill(kBase, 0x2000, 0x02020202);
	Check(ledger.QueryFill(kBase, 0x2000).kind == FillKind::Known, "a fresh fill after the CPU write is known again");
}

void TestLedgerPartialCover() {
	GpuWriteLedger ledger;
	ledger.RecordFill(kBase, 0x1000, 0x01010101);
	Check(ledger.QueryFill(kBase, 0x2000).kind != FillKind::Known, "a fill of half the range is not known for the range");
	ledger.RecordFill(kBase, 0x2000, 0x03030303);
	ledger.RecordFill(kBase, 0x1000, 0x04040404); // newer fill of the first half
	Check(ledger.QueryFill(kBase, 0x1000).kind == FillKind::Known && ledger.QueryFill(kBase, 0x1000).value == 0x04040404,
	      "the newest fill wins over its range");
	Check(ledger.QueryFill(kBase, 0x2000).kind == FillKind::NewerWrite, "an older big fill is hidden by a newer partial one");
}

void TestLedgerTrimAndEviction() {
	GpuWriteLedger ledger;
	ledger.RecordFill(kBase, 0x4000, 0x05050505);
	ledger.Trim(kBase, 0x1000); // the first part was published
	Check(ledger.QueryFill(kBase + 0x1000, 0x3000).kind == FillKind::Known, "the rest of a trimmed fill stays known");
	Check(ledger.QueryFill(kBase, 0x1000).kind != FillKind::Known, "the trimmed part is not known");
	ledger.Trim(kBase + 0x2000, 0x800); // a hole in the middle cannot be represented
	Check(ledger.QueryFill(kBase + 0x1000, 0x1000).kind != FillKind::Known, "a fill with a hole stops describing bytes");
	GpuWriteLedger lossy;
	lossy.RecordFill(kBase, 0x1000, 0x06060606);
	for (uint32_t i = 0; i < GpuWriteLedger::Size + 4; ++i) {
		lossy.RecordWrite(kBase + 0x100000 + i * 0x1000, 0x1000);
	}
	Check(lossy.QueryFill(kBase, 0x1000).kind != FillKind::Known, "an evicted fill is unknown, never wrong");
}

void TestDecide() {
	GpuWriteLedger ledger;
	ledger.RecordFill(kBase, 0x2000, 0x07070707);
	auto d = Decide(ledger.QueryFill(kBase, 0x2000), false);
	Check(d.publish && d.value == 0x07070707, "a known fill is published");
	d = Decide(ledger.QueryFill(kBase, 0x2000), true);
	Check(!d.publish && d.reason == Reason::FallbackNewerWrite, "a queued download that has not landed blocks the publish");
	ledger.RecordWrite(kBase, 0x10);
	d = Decide(ledger.QueryFill(kBase, 0x2000), false);
	Check(!d.publish && d.reason == Reason::FallbackNewerWrite, "a newer write blocks the publish");
	GpuWriteLedger partial;
	partial.RecordFill(kBase, 0x1000, 0x08080808);
	d = Decide(partial.QueryFill(kBase, 0x2000), false);
	Check(!d.publish && d.reason == Reason::FallbackPartial, "a range that is only partly known takes the old path");
}

// Publish model: pieces = GPU-modified ranges of a read; the backing is a byte array.
struct Model {
	std::vector<uint8_t>                  backing = std::vector<uint8_t>(0x4000, 0xaa);
	std::set<std::pair<uint64_t, uint64_t>> modified;

	void Publish(uint64_t address, uint64_t size, uint32_t value) {
		Apply(
		    address, size, value,
		    [&](uint64_t a, const uint8_t* bytes, uint64_t n) {
			    for (uint64_t i = 0; i < n; ++i) {
				    backing[a - kBase + i] = bytes[i];
			    }
		    },
		    [&](uint64_t a, uint64_t n) { modified.erase({a, n}); });
	}
};

void TestPublishPattern() {
	Model model;
	model.modified.insert({kBase, 0x2000});
	model.Publish(kBase, 0x2000, 0x2a2a2a2a);
	bool pattern = true;
	for (uint64_t i = 0; i < 0x2000; ++i) {
		pattern = pattern && model.backing[i] == 0x2a;
	}
	Check(pattern, "the pattern is in the backing");
	Check(model.backing[0x2000] == 0xaa, "bytes beyond the piece are untouched");
	Check(model.modified.empty(), "the piece is no longer GPU-modified");
	Check(FirstDifference(0x2a2a2a2a, kBase, model.backing.data(), 0x2000) == 0x2000, "the pattern verifies");
	// A 4-byte pattern that is not one repeated byte keeps its dword phase from the address.
	Model phase;
	phase.Publish(kBase + 4, 8, 0x04030201);
	Check(phase.backing[4] == 0x01 && phase.backing[5] == 0x02 && phase.backing[11] == 0x04, "dword phase");
}

void TestPartialPieceTakesOldPath() {
	GpuWriteLedger ledger;
	ledger.RecordFill(kBase, 0x1000, 0x09090909);
	Model model;
	model.modified.insert({kBase, 0x2000}); // the GPU-modified piece is bigger than the known fill
	const auto d = Decide(ledger.QueryFill(kBase, 0x2000), false);
	if (d.publish) {
		model.Publish(kBase, 0x2000, d.value);
	}
	Check(!d.publish && model.backing[0] == 0xaa && !model.modified.empty(), "nothing is published for a partly known piece");
}

// MaterializeColorClear after a publish: the code is decoded from the fill value, then the expanded keys are written to the
// backing. A verifier that compares the backing with the fill pattern afterwards sees a mismatch from offset 0 - that is the
// converter's own key write (kMetadataKeyByte), not a wrong publish (plan-dcc 0.4, the late_mismatch of S366).
void TestMaterializeAfterPublish() {
	Model model;
	model.modified.insert({kBase, 0x2000});
	const uint32_t fill = 0x2a2a2a2a;
	model.Publish(kBase, 0x2000, fill);
	uint8_t code = 0;
	Check(CodeFromFill(fill, &code) && code == 0x2a, "the clear code is decoded from the fill value");
	Check(FirstDifference(fill, kBase, model.backing.data(), 0x2000) == 0x2000, "no mismatch right after the publish");
	for (uint64_t i = 0; i < 0x2000; ++i) {
		model.backing[i] = kMetadataKeyByte;
	}
	Check(model.backing[0] == 0xff && model.backing[0x1fff] == 0xff, "the backing holds the 0xff keys afterwards");
	Check(FirstDifference(fill, kBase, model.backing.data(), 0x2000) == 0, "a late verifier sees a difference from offset 0");
	Check(!CodeFromFill(0x2a2a2a2b, &code) && !CodeFromFill(0x00ff00ff, &code), "a value that is not one repeated code is no fast clear");
	Check(CodeFromFill(0, &code) && code == 0, "zero is a code");
}

void TestMode() {
	Check(ParseMode(nullptr) == Mode::On, "on by default (variable unset)");
	Check(ParseMode("0") == Mode::Off && ParseMode("") == Mode::Off, "0 turns it off");
	Check(ParseMode("1") == Mode::On && ParseMode("shadow") == Mode::Shadow, "1 and shadow");
	Check(ParseMode("yes") == Mode::Off, "anything else is off");
}

} // namespace

int main() {
	TestLedgerKnown();
	TestLedgerNewerGpuWriteHides();
	TestLedgerCpuWriteHides();
	TestLedgerPartialCover();
	TestLedgerTrimAndEviction();
	TestDecide();
	TestPublishPattern();
	TestPartialPieceTakesOldPath();
	TestMaterializeAfterPublish();
	TestMode();
	return 0;
}
