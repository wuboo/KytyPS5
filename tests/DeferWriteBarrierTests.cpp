// S511 L1: a later draw flushes a deferred buffer-write barrier only when it may read those bytes.
// Mutant KYTY_DEFER_BARRIER_MUTANT_SKIP reports no read, so the barrier is skipped and this fails.

#include "graphics/host_gpu/renderer/deferWriteBarrier.h"

#include <cstdio>
#include <utility>
#include <vector>

using namespace Libs::Graphics::DeferBarrier;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	const Range atomic {0x1000, 0x2000, true};
	const Range store {0x1000, 0x2000, false};
	const Touch atomic_touch {0x1800, 0x1900, true};
	const Touch load_touch {0x1800, 0x1900, false};
	const Touch other {0x3000, 0x3100, false};
	const std::pair<uint64_t, uint64_t> none {0, 0};

	Check(!ReadsPending({}, {}, false, {}), "no pending write: no flush");
	Check(!ReadsPending(std::span<const Range>(&atomic, 1), std::span<const Touch>(&atomic_touch, 1), false, {}),
	      "atomic then atomic of the same range: no barrier");
	Check(ReadsPending(std::span<const Range>(&atomic, 1), std::span<const Touch>(&load_touch, 1), false, {}),
	      "atomic then a plain load: barrier");
	Check(ReadsPending(std::span<const Range>(&store, 1), std::span<const Touch>(&atomic_touch, 1), false, {}),
	      "plain store then an atomic: barrier");
	Check(!ReadsPending(std::span<const Range>(&store, 1), std::span<const Touch>(&other, 1), false, {}),
	      "a different range: no barrier");
	Check(ReadsPending(std::span<const Range>(&atomic, 1), {}, true, {}), "address read (uses_dma): barrier");
	const std::pair<uint64_t, uint64_t> vertex {0x1f00, 0x2100};
	Check(ReadsPending(std::span<const Range>(&atomic, 1), {}, false, std::span(&vertex, 1)),
	      "vertex range overlaps: barrier");
	const std::pair<uint64_t, uint64_t> index {0x8000, 0x8100};
	Check(!ReadsPending(std::span<const Range>(&atomic, 1), {}, false, std::span(&index, 1)),
	      "index range elsewhere: no barrier");
	Check(!ReadsPending(std::span<const Range>(&atomic, 1), {}, false, std::span(&none, 1)),
	      "empty plain range: no barrier");

	Range           slots[2] {};
	uint32_t        count = 0;
	std::span<Range> list {slots};
	Check(NoteWrite(list, count, 0x1000, 0x2000, true), "first range fits");
	Check(count == 1 && slots[0].atomic_only, "stored as atomic-only");
	Check(NoteWrite(list, count, 0x1000, 0x2000, false), "same range reused");
	Check(count == 1 && !slots[0].atomic_only, "a plain store clears atomic-only");
	Check(NoteWrite(list, count, 0x3000, 0x3100, true), "second range fits");
	Check(!NoteWrite(list, count, 0x4000, 0x4100, true), "a third new range forces a flush");
	Check(!NoteWrite(list, count, 0, 0, true) || count == 2, "an empty range is ignored");

	if (g_failures != 0) {
		std::fprintf(stderr, "DeferWriteBarrierTests: %d failed\n", g_failures);
		return 1;
	}
	std::puts("DeferWriteBarrierTests: OK");
	return 0;
}
