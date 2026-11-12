// S556 KYTY_NARROW_BARRIER (narrowBarrier.h): the range of a narrowed upload/readback barrier covers every byte the copy
// regions touch, and the levels narrow exactly the barriers they document.
// Mutants (each must be caught, WILL_FAIL): FIRST_ONLY (the span is the first region), ASSUME_SORTED (the span ends at the
// last region), IMAGES (level 1 narrows a dispatch that writes storage images), RO_LEVEL1 (level 1 drops the access masks
// after a read-only dispatch).

#include "graphics/host_gpu/renderer/narrowBarrier.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace Libs::Graphics::NarrowBarrier;

namespace {

struct Copy {
	uint64_t srcOffset = 0;
	uint64_t dstOffset = 0;
	uint64_t size      = 0;
};

int g_failures = 0;
void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

// Every byte of every region lies inside the span.
bool Covers(const Span& span, const std::vector<Copy>& copies, bool destination) {
	for (const auto& copy: copies) {
		const uint64_t offset = destination ? copy.dstOffset : copy.srcOffset;
		if (offset < span.offset || offset + copy.size > span.offset + span.size) {
			return false;
		}
	}
	return true;
}

} // namespace

int main() {
	const std::vector<Copy> none;
	const auto empty = CopySpan(std::span<const Copy>(none));
	Check(empty.offset == 0 && empty.size == 0, "no regions: empty span (caller keeps the whole buffer)");

	const std::vector<Copy> one {{0, 0x3000, 0x100}};
	const auto single = CopySpan(std::span<const Copy>(one));
	Check(single.offset == 0x3000 && single.size == 0x100, "one region: exactly its bytes");

	const std::vector<Copy> sorted {{0, 0x1000, 0x40}, {0x40, 0x2000, 0x1000}, {0x1040, 0x8000, 0x10}};
	const auto ascending = CopySpan(std::span<const Copy>(sorted));
	Check(ascending.offset == 0x1000 && ascending.size == 0x7010, "ascending regions: first begin .. last end");
	Check(Covers(ascending, sorted, true), "ascending regions covered");

	// A long region before a short one that starts later: the end is not the last region's end.
	const std::vector<Copy> nested {{0, 0x1000, 0x9000}, {0x9000, 0x2000, 0x10}};
	const auto inner = CopySpan(std::span<const Copy>(nested));
	Check(inner.offset == 0x1000 && inner.size == 0x9000, "a later region inside an earlier one: span of the outer");
	Check(Covers(inner, nested, true), "nested regions covered");

	const std::vector<Copy> unsorted {{0, 0x5000, 0x100}, {0x100, 0x1000, 0x100}, {0x200, 0x3000, 0x100}};
	const auto any_order = CopySpan(std::span<const Copy>(unsorted));
	Check(any_order.offset == 0x1000 && any_order.size == 0x4100, "unsorted regions: min begin .. max end");
	Check(Covers(any_order, unsorted, true), "unsorted regions covered");

	const std::vector<Copy> reads {{0x700, 0, 0x10}, {0x100, 0x10, 0x20}};
	const auto read = ReadSpan(std::span<const Copy>(reads));
	Check(read.offset == 0x100 && read.size == 0x610, "readback span uses the source side");
	Check(Covers(read, reads, false), "readback regions covered");

	Check(ParseLevel(nullptr) == 0, "unset = 0");
	Check(ParseLevel("0") == 0 && ParseLevel("1") == 1 && ParseLevel("2") == 2, "0/1/2 parse");
	Check(ParseLevel("3") == 0 && ParseLevel("on") == 0 && ParseLevel("") == 0 && ParseLevel("12") == 0,
	      "anything else = 0 (old barriers)");

	Check(!NarrowBufferSync(0) && NarrowBufferSync(1) && NarrowBufferSync(2), "upload barriers: levels 1 and 2");
	Check(!NarrowWriteHazard(0, false) && !NarrowWriteHazard(0, true), "level 0 never narrows the write hazard");
	Check(NarrowWriteHazard(1, false), "level 1: buffer-only writer narrowed");
	Check(!NarrowWriteHazard(1, true), "level 1: a storage-image writer keeps the full barrier");
	Check(NarrowWriteHazard(2, true) && NarrowWriteHazard(2, false), "level 2: every writer narrowed");
	Check(!ExecOnlyAfterReadOnlyDispatch(0) && !ExecOnlyAfterReadOnlyDispatch(1), "after read-only dispatch: full at 0 and 1");
	Check(ExecOnlyAfterReadOnlyDispatch(2), "after read-only dispatch: execution only at 2");

	if (g_failures != 0) {
		std::fprintf(stderr, "NarrowBarrierTests: %d failed\n", g_failures);
		return 1;
	}
	std::puts("NarrowBarrierTests: OK");
	return 0;
}
