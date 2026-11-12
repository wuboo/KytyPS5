// S546 KYTY_IMAGE_SIZE_MEMO (imageSizeMemo.h): a fake info whose fake UpdateSize reads every key field (the address too, as
// CompactMipSize does) and writes every value field. Memo results must equal the full computation for all inputs, in On and Shadow.
// Mutants: NO_ADDRESS (two infos that differ only in address share a slot), NO_SIZE (a derived size is reused for an explicit one),
// NO_MIPS (a hit leaves the caller's mip layout in place).

#include "graphics/host_gpu/renderer/image/imageSizeMemo.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <thread>

namespace {

int g_failures = 0;
void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

enum class Format : uint32_t { A = 1, B = 7 };
enum class Kind : uint8_t { None, Dcc };

struct Mip {
	uint64_t offset = 0;
	uint64_t size   = 0;
	uint32_t pitch  = 0;
	uint32_t height = 0;
};
struct Tiling {
	uint32_t block = 0;
	uint32_t texel = 1;
};

struct FakeInfo {
	struct {
		uint64_t address = 0;
		uint64_t size    = 0;
	} data;
	struct {
		Kind kind = Kind::None;
	} metadata;
	Format pixel_format = Format::A;
	Format guest_format = Format::A;
	uint32_t type       = 0;
	struct {
		uint32_t width = 1, height = 1, depth = 1;
	} extent;
	struct {
		uint32_t levels = 1, layers = 1;
	} resources;
	uint32_t pitch           = 0;
	uint32_t bytes_per_block = 0;
	uint32_t samples         = 1;
	uint32_t tile_mode       = 0;
	std::array<Mip, 16> mip_layout {};
	Tiling              tiling;
	uint64_t            tiled_slice_stride  = 0;
	uint64_t            linear_slice_stride = 0;
	uint32_t            first_tail_level    = 16;
	uint64_t            untouched           = 0; // not read nor written by the computation
};

int g_computes = 0;

void Compute(FakeInfo& info) {
	++g_computes;
	info.mip_layout = {};
	if (info.bytes_per_block == 0) {
		info.bytes_per_block = static_cast<uint32_t>(info.guest_format) * 2u;
	}
	if (info.pitch == 0) {
		info.pitch = info.extent.width + info.tile_mode;
	}
	const uint64_t slice = uint64_t {info.pitch} * info.extent.height * info.bytes_per_block * info.samples;
	const bool     auto_size = info.data.size == 0;
	if (auto_size) {
		info.data.size = slice * info.resources.layers * info.extent.depth;
	}
	for (uint32_t level = 0; level < info.resources.levels; level++) {
		info.mip_layout[level] = {slice * level, slice >> level, info.pitch >> level, info.extent.height >> level};
	}
	info.tiling             = {static_cast<uint32_t>(info.pixel_format) + info.type, static_cast<uint32_t>(info.metadata.kind) + 1u};
	info.tiled_slice_stride = info.data.size / info.resources.layers;
	// Like CompactMipSize: the alignment of the address changes the result.
	info.linear_slice_stride = (info.data.address & 0xffffu) == 0 ? info.data.size : slice;
	info.first_tail_level    = auto_size ? info.resources.levels : 16u;
}

bool SameOutputs(const FakeInfo& a, const FakeInfo& b) {
	return std::memcmp(&a.mip_layout, &b.mip_layout, sizeof(a.mip_layout)) == 0 &&
	       std::memcmp(&a.tiling, &b.tiling, sizeof(a.tiling)) == 0 && a.data.size == b.data.size &&
	       a.pitch == b.pitch && a.bytes_per_block == b.bytes_per_block &&
	       a.tiled_slice_stride == b.tiled_slice_stride && a.linear_slice_stride == b.linear_slice_stride &&
	       a.first_tail_level == b.first_tail_level && a.untouched == b.untouched;
}

FakeInfo Variant(uint32_t i) {
	FakeInfo info;
	info.data.address      = 0x100000000ull + (uint64_t {i % 3u} * 0x8000u); // aligned / unaligned to 64 KiB
	info.data.size         = (i % 5u == 0) ? 0x40000u + i : 0; // explicit or derived
	info.metadata.kind     = (i % 7u == 0) ? Kind::Dcc : Kind::None;
	info.pixel_format      = (i % 2u) ? Format::A : Format::B;
	info.guest_format      = (i % 4u < 2u) ? Format::A : Format::B;
	info.type              = i % 3u;
	info.extent.width      = 64u << (i % 4u);
	info.extent.height     = 32u << (i % 3u);
	info.extent.depth      = 1u + (i % 2u);
	info.resources.levels  = 1u + (i % 6u);
	info.resources.layers  = 1u + (i % 3u);
	info.pitch             = (i % 6u == 0) ? 512u : 0u;
	info.bytes_per_block   = (i % 8u == 0) ? 16u : 0u;
	info.samples           = (i % 9u == 0) ? 4u : 1u;
	info.tile_mode         = i % 5u;
	return info;
}

void RunMode(Libs::Graphics::SizeMemo::Mode mode, const char* name) {
	using namespace Libs::Graphics;
	uint32_t hits = 0;
	uint32_t ne   = 0;
	for (uint32_t round = 0; round < 3; round++) {
		for (uint32_t i = 0; i < 400; i++) {
			FakeInfo expected = Variant(i);
			Compute(expected);
			FakeInfo memo = Variant(i);
			// Stale outputs in the caller's info must not leak into a hit.
			memo.mip_layout[15]  = {1, 2, 3, 4};
			memo.tiled_slice_stride = 99;
			const auto result = SizeMemo::Run(memo, Compute, mode);
			hits += result == SizeMemo::Result::Hit ? 1u : 0u;
			ne += result == SizeMemo::Result::Disagree ? 1u : 0u;
			if (!SameOutputs(memo, expected)) {
				std::fprintf(stderr, "%s: variant %u round %u differs\n", name, i, round);
				Check(false, "memo result equals the full computation");
				return;
			}
		}
	}
	Check(hits > 0, "repeated descriptions hit");
	Check(ne == 0, "shadow never disagrees");
}

} // namespace

int main() {
	using namespace Libs::Graphics;
	Check(SizeMemo::ParseMode(nullptr) == SizeMemo::Mode::Off, "unset is off");
	Check(SizeMemo::ParseMode("0") == SizeMemo::Mode::Off, "0 is off");
	Check(SizeMemo::ParseMode("1") == SizeMemo::Mode::On, "1 is on");
	Check(SizeMemo::ParseMode("shadow") == SizeMemo::Mode::Shadow, "shadow");
	Check(SizeMemo::ParseMode("yes") == SizeMemo::Mode::Off, "anything else is off");

	// A hit skips the computation in On mode.
	{
		FakeInfo first = Variant(1);
		(void)SizeMemo::Run(first, Compute, SizeMemo::Mode::On);
		const int before = g_computes;
		FakeInfo second  = Variant(1);
		Check(SizeMemo::Run(second, Compute, SizeMemo::Mode::On) == SizeMemo::Result::Hit, "same description hits");
		Check(g_computes == before, "a hit does not compute");
		Check(SameOutputs(first, second), "a hit restores the outputs");
	}

	// Two descriptions that differ only in one key field, computed back to back.
	const auto pair = [](FakeInfo left, FakeInfo right, const char* what) {
		FakeInfo expected = right;
		Compute(expected);
		(void)SizeMemo::Run(left, Compute, SizeMemo::Mode::On);
		(void)SizeMemo::Run(right, Compute, SizeMemo::Mode::On);
		Check(SameOutputs(right, expected), what);
	};
	{
		FakeInfo aligned = Variant(3);
		aligned.data.address = 0x200000000ull;
		aligned.resources.layers = 2;
		FakeInfo unaligned      = aligned;
		unaligned.data.address += 0x8000u;
		pair(aligned, unaligned, "the address is part of the key");
	}
	{
		FakeInfo derived = Variant(4);
		derived.data.size = 0;
		FakeInfo explicit_size = derived;
		explicit_size.data.size = 0x123400u;
		pair(derived, explicit_size, "the given size is part of the key");
	}

	RunMode(SizeMemo::Mode::On, "on");
	RunMode(SizeMemo::Mode::Shadow, "shadow");

	// Tables are per thread: another thread starts empty and gets correct results.
	std::thread other([] { RunMode(SizeMemo::Mode::On, "thread"); });
	other.join();

	if (g_failures != 0) {
		std::fprintf(stderr, "ImageSizeMemoTests: %d failed\n", g_failures);
		return 1;
	}
	std::puts("ImageSizeMemoTests: OK");
	return 0;
}
