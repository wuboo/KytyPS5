#include "graphics/host_gpu/memory/umaBackingMap.h"
#include "graphics/host_gpu/memory/umaChunks.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// Built several times: normally, and with one mutant each (KYTY_UMA_MUTANT_STRADDLE_OK,
// KYTY_UMA_MUTANT_TRUST_REPORTED_ALIGNMENT, KYTY_UMA_MUTANT_RETRY_FAILED, KYTY_UMA_MUTANT_NO_OVERLAP).
// Every mutant build must
// FAIL (ctest WILL_FAIL).

namespace {

using namespace Libs::Graphics::UmaChunks;

constexpr uint64_t MB = 1024 * 1024;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "UmaChunksTests: failed: %s\n", message);
		std::exit(1); // not abort(): ctest WILL_FAIL does not invert signals
	}
}

void TestChunkBytes() {
	Check(ChunkBytes(0) == 256 * MB, "0 means the default 256 MB");
	Check(ChunkBytes(64) == 64 * MB, "64 MB stays");
	Check(ChunkBytes(100) == 64 * MB, "rounded down to a power of two");
	Check(ChunkBytes(1) == 16 * MB, "clamped up to the minimum");
	Check(ChunkBytes(1u << 20) == 4096 * MB, "clamped down to the maximum");
}

void TestAlignment() {
	// Rosetta reports 4 KB; the 16 KB is ours.
	Check(ImportAlignment(4096) == 16384, "reported 4096 is raised to 16 KB");
	Check(ImportAlignment(0) == 16384, "reported 0 is raised to 16 KB");
	Check(ImportAlignment(65536) == 65536, "a larger reported alignment wins");
	Check(!IsImportable(0x10001000, 64 * MB, 4096), "a 4 KB aligned pointer is refused");
	Check(!IsImportable(0x10004000, 64 * MB + 4096, 4096), "a size that is not a 16 KB multiple is refused");
	Check(IsImportable(0x10004000, 64 * MB, 4096), "16 KB aligned pointer and size pass");
	Check(!IsImportable(0x10004000, 0, 4096), "an empty chunk is refused");
}

void TestLayout() {
	// 13.8 GB-like arena: 13 chunks of 1 GB and a short tail.
	const Layout layout(13 * 1024 * MB + 800 * MB, 1024 * MB);
	Check(layout.Count() == 14, "14 chunks");
	Check(layout.Begin(3) == 3 * 1024 * MB, "chunk begin");
	Check(layout.End(3) == 4 * 1024 * MB, "full chunk end");
	Check(layout.End(13) == 13 * 1024 * MB + 800 * MB, "the last chunk ends at the arena end");

	Check(layout.Find(0, 16384).ok && layout.Find(0, 16384).index == 0, "first page in chunk 0");
	const auto edge = layout.Find(1024 * MB - 16384, 16384);
	Check(edge.ok && edge.index == 0, "a range that ends exactly at the boundary stays in chunk 0");
	Check(!layout.Find(1024 * MB - 16384, 32768).ok, "a range over the boundary is refused");
	Check(!layout.Find(1024 * MB - 1, 2).ok, "a 2-byte range over the boundary is refused");
	Check(layout.Find(1024 * MB, 16384).index == 1, "the first byte after the boundary is chunk 1");
	Check(!layout.Find(layout.ArenaSize() - 8192, 16384).ok, "a range past the arena end is refused");
	Check(layout.Find(layout.ArenaSize() - 16384, 16384).index == 13, "tail chunk");
	Check(!layout.Find(0, 0).ok, "an empty range is refused");
	Check(!layout.Find(UINT64_MAX - 4, 16).ok, "an overflowing range is refused");
}

// On-demand import as the GPU thread sees it: the first request imports, later ones reuse, and a
// failed chunk goes to the copy path without another attempt.
struct FakeGpu {
	Layout                layout {512 * MB, 64 * MB};
	std::vector<State>    state = std::vector<State>(8, State::NotImported);
	std::vector<uint64_t> gpu   = std::vector<uint64_t>(8, 0);
	std::vector<bool>     fail  = std::vector<bool>(8, false);
	uint32_t              attempts = 0;
	uint32_t              legacy   = 0;

	// True when the range is served from the arena.
	bool Request(uint64_t offset, uint64_t size) {
		const auto place = layout.Find(offset, size);
		if (!place.ok) {
			++legacy;
			return false;
		}
		switch (Decide(state[place.index])) {
			case Action::Use: return true;
			case Action::Legacy: ++legacy; return false;
			case Action::Import:
				++attempts;
				if (fail[place.index]) {
					state[place.index] = State::Failed;
					++legacy;
					return false;
				}
				state[place.index] = State::Imported;
				gpu[place.index]   = 0x7000'0000'0000ull + uint64_t {place.index} * 0x1000'0000'0000ull;
				return true;
		}
		return false;
	}
	uint64_t Address(uint64_t offset, uint64_t size) const {
		return DeviceAddress(layout, offset, size, [&](uint32_t i) { return gpu[i]; });
	}
};

void TestImportOnDemand() {
	FakeGpu gpu;
	Check(gpu.Address(5 * MB, 16384) == 0, "nothing is imported at the start");
	Check(gpu.Request(5 * MB, 16384), "first use imports chunk 0");
	Check(gpu.attempts == 1, "one import");
	Check(gpu.Address(5 * MB, 16384) == 0x7000'0000'0000ull + 5 * MB, "address = chunk base + offset in chunk");
	Check(gpu.Request(6 * MB, 16384), "second use of the same chunk reuses it");
	Check(gpu.attempts == 1, "no second import");
	Check(gpu.Address(70 * MB, 16384) == 0, "chunk 1 is still not imported");
	Check(gpu.Request(70 * MB, 16384), "chunk 1 imported on first use");
	Check(gpu.Address(70 * MB, 16384) == 0x7000'0000'0000ull + 0x1000'0000'0000ull + 6 * MB, "chunk 1 address");
	uint32_t imported = 0;
	for (const auto s: gpu.state) {
		imported += s == State::Imported ? 1 : 0;
	}
	Check(imported == 2, "only the used chunks are imported");
}

void TestStraddleTakesCopyPath() {
	FakeGpu gpu;
	Check(!gpu.Request(64 * MB - 8192, 16384), "a window over a chunk boundary is not served");
	Check(gpu.attempts == 0, "and it imports nothing");
	Check(gpu.legacy == 1, "it is counted as a copy-path request");
	Check(gpu.Address(64 * MB - 8192, 16384) == 0, "no address for a straddling range even when both chunks exist");
	gpu.Request(0, 16384);
	gpu.Request(64 * MB, 16384);
	Check(gpu.Address(64 * MB - 8192, 16384) == 0, "still none with both neighbours imported");
}

void TestFailedImportIsNotRetried() {
	FakeGpu gpu;
	gpu.fail[2] = true;
	Check(!gpu.Request(130 * MB, 16384), "failed import: copy path");
	Check(!gpu.Request(131 * MB, 16384), "second request: copy path");
	Check(!gpu.Request(132 * MB, 16384), "third request: copy path");
	Check(gpu.attempts == 1, "the import was attempted once");
	Check(gpu.legacy == 3, "every request counted as copy path");
	Check(gpu.Request(0, 16384), "other chunks are unaffected");
}


// ---- Overlapping chunks (S76): no window up to the overlap size ever crosses a chunk end. ----
void TestOverlapBytes() {
	Check(OverlapBytes(0, false, 256 * MB) == 64 * MB, "default overlap of a 256 MB chunk is 64 MB");
	Check(OverlapBytes(0, false, 64 * MB) == 16 * MB, "default overlap is at most a quarter of the chunk");
	Check(OverlapBytes(0, true, 256 * MB) == 0, "an explicit 0 switches the overlap off");
	Check(OverlapBytes(100, true, 256 * MB) == 64 * MB, "rounded down to a power of two");
	Check(OverlapBytes(1024, true, 256 * MB) == 128 * MB, "at most half of the chunk");
}

void TestOverlapLayout() {
	const uint64_t arena = 13 * 1024 * MB + 800 * MB;
	const Layout   layout(arena, 256 * MB, 64 * MB);
	Check(layout.Stride() == 192 * MB && layout.Overlap() == 64 * MB, "stride = chunk - overlap");
	Check(layout.Begin(1) == 192 * MB && layout.End(1) == 192 * MB + 256 * MB, "chunk 1 geometry");
	Check(layout.End(layout.Count() - 1) == arena, "the last chunk ends at the arena end");
	Check(layout.Begin(layout.Count() - 1) + 256 * MB >= arena, "and nothing past it is needed");
	Check(layout.Count() == 1 + (arena - 256 * MB + 192 * MB - 1) / (192 * MB), "chunk count");
	// Every window of at most the overlap size, at every 16 KB position, fits one chunk.
	uint64_t straddles = 0;
	for (uint64_t offset = 0; offset + 64 * MB <= arena; offset += 3 * 1024 * 1024 + 16384) {
		if (!layout.Find(offset, 64 * MB).ok) {
			++straddles;
		}
	}
	Check(straddles == 0, "no window of the overlap size straddles (chunk_straddles = 0)");
	// Exactly around a cell edge: a window that starts just before the stride boundary and ends in
	// the overlap is cut from the earlier chunk.
	const auto edge = layout.Find(192 * MB - 16384, 64 * MB);
	Check(edge.ok && edge.index == 0, "a window over the cell edge stays in the earlier chunk");
	Check(layout.Find(192 * MB, 64 * MB).index == 1, "a window starting at the edge is chunk 1");
	// Larger than the overlap can still straddle (counted, copy path).
	Check(!layout.Find(192 * MB - 16384, 65 * MB).ok, "a window larger than the overlap may straddle");
	// Cells import: the chunks of a visible range are the cells it touches.
	Check(layout.Cell(0) == 0 && layout.Cell(192 * MB - 1) == 0 && layout.Cell(192 * MB) == 1, "cells");
	Check(layout.LastCell(100 * MB, 200 * MB) == 1, "a range reaching into cell 1 needs chunk 1");
	Check(layout.Cell(arena - 16384) == layout.Count() - 1, "the last cell is the last chunk");
}

void TestOverlapAlias() {
	const Layout          layout(2048 * MB, 256 * MB, 64 * MB);
	std::vector<uint64_t> gpu(layout.Count(), 0);
	const auto            address = [&](uint64_t offset, uint64_t size) {
        return DeviceAddress(layout, offset, size, [&](uint32_t i) { return gpu[i]; });
	};
	// Overlap zone of chunks 0 and 1: [192, 256) MB.
	gpu[0] = 0x1000'0000'0000ull;
	Check(address(200 * MB, 16384) == 0x1000'0000'0000ull + 200 * MB,
	      "in the overlap an imported previous chunk serves the window");
	Check(address(100 * MB, 16384) == 0x1000'0000'0000ull + 100 * MB, "outside the overlap, own chunk");
	gpu[1] = 0x2000'0000'0000ull;
	Check(address(200 * MB, 16384) == 0x2000'0000'0000ull + 8 * MB,
	      "with both imported the cell's own chunk wins");
	Check(address(300 * MB, 16384) == 0x2000'0000'0000ull + 108 * MB, "chunk 1 interior");
	gpu[1] = 0;
	Check(address(300 * MB, 16384) == 0, "past the overlap an unimported cell has no address");
	Check(!layout.FindPrevious(100 * MB, 16384).ok, "no alias outside the overlap zone");
	Check(layout.FindPrevious(200 * MB, 16384).ok && layout.FindPrevious(200 * MB, 16384).index == 0,
	      "alias is the previous chunk");
}

void TestRunsInRange() {
	Libs::Graphics::UmaBackingMap<14> map;
	map.Map(0x10000, 0x40000, 0x1000000);
	map.Map(0x80000, 0x10000, 0x2000000);
	uint64_t runs = 0, bytes = 0, first_offset = 0;
	map.ForEachRunIn(0x20000, 0x70000, [&](uint64_t vaddr, uint64_t size, uint64_t offset) {
		if (runs++ == 0) {
			first_offset = offset;
			Check(vaddr == 0x20000, "the first run is clipped to the range start");
		}
		bytes += size;
	});
	Check(runs == 2 && bytes == 0x30000 + 0x10000, "both runs, clipped");
	Check(first_offset == 0x1010000, "backing offset moves with the clip");
}

} // namespace

int main() {
	TestChunkBytes();
	TestAlignment();
	TestLayout();
	TestImportOnDemand();
	TestStraddleTakesCopyPath();
	TestFailedImportIsNotRetried();
	TestOverlapBytes();
	TestOverlapLayout();
	TestOverlapAlias();
	TestRunsInRange();
	std::puts("UmaChunksTests: ok");
	return 0;
}
