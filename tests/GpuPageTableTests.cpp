#include "graphics/host_gpu/memory/gpuPageTable.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Built twice: normally, and with KYTY_UMA_MUTANT_SPLIT_AS_WHOLE (a 16 KB entry is reported for a
// page whose 4 KB pieces are not consecutive on the device). The mutant build must FAIL.

namespace {

using Table = Libs::Graphics::GpuPageTable<44, 14>;
constexpr uint64_t K4        = 0x1000;
constexpr uint64_t K16       = 0x4000;
constexpr uint64_t Arena     = 0x700000000000ull;
constexpr uint64_t Sentinel  = 0x6fff00000000ull;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "GpuPageTableTests: failed: %s\n", message);
		std::exit(1); // not abort(): ctest WILL_FAIL does not invert signals
	}
}

void TestSetLookupClear() {
	Table table(Sentinel);
	Check(table.Lookup(0x10000000) == Sentinel, "unmapped page reads as the sentinel");
	Check(!table.IsMapped(0x10000000), "unmapped");
	table.Set(0x10000000, Arena + 0x1000);
	Check(table.Lookup(0x10000000) == Arena + 0x1000, "mapped entry");
	Check(table.Lookup(0x10001000) == Sentinel, "neighbour stays unmapped");
	Check(table.MappedPages() == 1, "one page");
	table.Set(0x10000000, Arena + 0x2000); // remap in place
	Check(table.MappedPages() == 1 && table.Lookup(0x10000000) == Arena + 0x2000, "remap");
	table.Clear(0x10000000);
	Check(table.Lookup(0x10000000) == Sentinel && table.MappedPages() == 0, "cleared");
	Check(table.LeafCount() == 0, "empty leaf is released");
	table.Clear(0x10000000); // idempotent
	Check(table.MappedPages() == 0, "double clear");
	table.Set(0x10000000, 0);
	Check(table.MappedPages() == 0, "a zero device address is not an entry");
	table.Set(uint64_t {1} << 44, Arena);
	Check(table.MappedPages() == 0, "address beyond the table is ignored");
}

void TestSparseLeaves() {
	Table table(Sentinel);
	const auto base_memory = table.MemoryBytes();
	// Two regions 8 GB apart: two leaves, not a table sized for the whole span.
	table.SetRange(0x10000000, 64 * K16, Arena);
	table.SetRange(0x10000000 + 0x200000000ull, 64 * K16, Arena + 0x100000);
	Check(table.LeafCount() == 2, "leaves only for the mapped regions");
	Check(table.MappedPages() == 2 * 64 * 4, "page count");
	Check(table.MemoryBytes() < base_memory + 2 * 1024 * 1024, "leaf memory stays small");
	table.ClearRange(0x10000000, 64 * K16);
	Check(table.LeafCount() == 1, "leaf released with its last page");
	uint64_t seen = 0;
	table.ForEachMapped([&](uint64_t page, uint64_t device) {
		Check(device == Arena + 0x100000 + (page - (0x10000000 + 0x200000000ull)), "iteration entry");
		++seen;
	});
	Check(seen == 64 * 4, "iteration covers the mapped pages");
}

void TestWideEntryAndSplit() {
	Table table(Sentinel);
	uint64_t entry = 0;
	// Four consecutive 4 KB pages -> one 16 KB entry.
	table.SetRange(0x20000000, K16, Arena + 0x8000);
	Check(table.LookupWide(0x20000000, 14, &entry) && entry == Arena + 0x8000, "contiguous page");
	Check(table.LookupWide(0x20002abc, 14, &entry) && entry == Arena + 0x8000, "any address inside");

	// A page whose pieces point at different arena offsets is "split": no single entry.
	table.Set(0x20004000, Arena + 0x10000);
	table.Set(0x20005000, Arena + 0x30000); // not consecutive
	table.Set(0x20006000, Arena + 0x12000);
	table.Set(0x20007000, Arena + 0x13000);
	Check(!table.LookupWide(0x20004000, 14, nullptr), "split page has no wide entry");
	Check(table.Lookup(0x20005000) == Arena + 0x30000, "4 KB entries still resolve");

	// A partly mapped page is not wide either.
	table.Set(0x20008000, Arena + 0x20000);
	table.Set(0x20009000, Arena + 0x21000);
	Check(!table.LookupWide(0x20008000, 14, nullptr), "half-mapped page has no wide entry");
	// Unmapped first piece.
	table.Set(0x2000d000, Arena + 0x40000);
	Check(!table.LookupWide(0x2000c000, 14, nullptr), "page without its first piece");
}

} // namespace

int main() {
	TestSetLookupClear();
	TestSparseLeaves();
	TestWideEntryAndSplit();
	std::puts("GpuPageTableTests: ok");
	return 0;
}
