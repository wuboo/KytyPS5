#include "graphics/host_gpu/memory/umaSync.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

// Built twice: normally, and with KYTY_UMA_MUTANT_IMMEDIATE_UNMAP (the unmap is published at once,
// before the in-flight submits retire). The mutant build must FAIL (ctest WILL_FAIL).

namespace {

using Sync  = Libs::Graphics::UmaPageSync<14>;
using Event = Sync::Event;
constexpr uint64_t P         = 0x4000;
constexpr uint64_t ArenaBase = 0x700000000000ull;
constexpr uint64_t Sentinel  = 0x6fff00000000ull;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "UmaSyncTests: failed: %s\n", message);
		std::exit(1); // not abort(): ctest WILL_FAIL does not invert signals
	}
}

// A page table as the GPU sees it: published entries only.
struct FakeTable {
	std::map<uint64_t, uint64_t> entries;

	uint64_t Entry(uint64_t page) const {
		const auto it = entries.find(page);
		return it == entries.end() ? 0 : it->second;
	}
};

void Publish(const Sync& sync, FakeTable& table, uint64_t first, uint64_t last) {
	for (auto page = first; page < last; page += P) {
		uint64_t offset = 0;
		table.entries[page] = sync.Map().PageOffset(page, &offset) ? ArenaBase + offset : Sentinel;
	}
}

void TestUnmapWaitsForTick() {
	Sync      sync;
	FakeTable table;
	uint64_t  retired = 0;
	const auto emit   = [&](uint64_t a, uint64_t b) { Publish(sync, table, a, b); };
	const auto done   = [&](uint64_t tick) { return tick <= retired; };

	sync.Apply({Event {0x10000000, 4 * P, 0x100000, true}}, 1, emit);
	Check(table.Entry(0x10000000) == ArenaBase + 0x100000, "mapped entry published at once");

	// A submit recorded at tick 5 may still read the old mapping.
	sync.Apply({Event {0x10000000, 4 * P, 0, false}}, 5, emit);
	sync.Retire(done, emit);
	Check(table.Entry(0x10000000) == ArenaBase + 0x100000, "entry survives while tick 5 is in flight");
	Check(table.Entry(0x10000000 + 3 * P) == ArenaBase + 0x100000 + 3 * P, "last page survives");
	Check(sync.PendingCount() == 1, "unmap is pending");

	retired = 4;
	sync.Retire(done, emit);
	Check(table.Entry(0x10000000) == ArenaBase + 0x100000, "tick 4 is not enough");

	retired = 5;
	sync.Retire(done, emit);
	Check(table.Entry(0x10000000) == Sentinel, "entry replaced after tick 5 retired");
	Check(table.Entry(0x10000000 + 3 * P) == Sentinel, "all pages replaced");
	Check(sync.PendingCount() == 0, "nothing pending");
}

void TestRemapBeforeRetireKeepsNewMapping() {
	Sync      sync;
	FakeTable table;
	uint64_t  retired = 0;
	const auto emit   = [&](uint64_t a, uint64_t b) { Publish(sync, table, a, b); };
	const auto done   = [&](uint64_t tick) { return tick <= retired; };

	sync.Apply({Event {0x20000000, 2 * P, 0x200000, true}}, 1, emit);
	sync.Apply({Event {0x20000000, 2 * P, 0, false}}, 3, emit);
	sync.Apply({Event {0x20000000, 2 * P, 0x300000, true}}, 3, emit);
	Check(table.Entry(0x20000000) == ArenaBase + 0x300000, "new mapping published at once");
	retired = 3;
	sync.Retire(done, emit);
	Check(table.Entry(0x20000000) == ArenaBase + 0x300000, "deferred unmap must not clobber the remap");
	Check(table.Entry(0x20000000 + P) == ArenaBase + 0x300000 + P, "second page too");
}

void TestPartialUnmapBreaksNeighbourLater() {
	Sync      sync;
	FakeTable table;
	uint64_t  retired = 0;
	const auto emit   = [&](uint64_t a, uint64_t b) { Publish(sync, table, a, b); };
	const auto done   = [&](uint64_t tick) { return tick <= retired; };

	sync.Apply({Event {0x30000000, 2 * P, 0x400000, true}}, 1, emit);
	// Unmapping the middle of the second page makes the whole page unusable.
	sync.Apply({Event {0x30000000 + P + 0x1000, 0x1000, 0, false}}, 2, emit);
	Check(table.Entry(0x30000000 + P) == ArenaBase + 0x400000 + P, "partial page still published");
	retired = 2;
	sync.Retire(done, emit);
	Check(table.Entry(0x30000000 + P) == Sentinel, "partial page dropped after the tick");
	Check(table.Entry(0x30000000) == ArenaBase + 0x400000, "untouched page stays");
}

} // namespace

int main() {
	TestUnmapWaitsForTick();
	TestRemapBeforeRetireKeepsNewMapping();
	TestPartialUnmapBreaksNeighbourLater();
	std::puts("UmaSyncTests: ok");
	return 0;
}
