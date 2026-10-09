#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/renderer/bcDump.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::DiagDump;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "BcDumpTileTests: failed: %s\n", message);
		std::exit(1);
	}
}

void Element(uint8_t* out, uint32_t level, uint32_t x, uint32_t y) {
	for (int i = 0; i < 16; i++) out[i] = static_cast<uint8_t>((level * 3 + x * 7 + y * 13 + i * 31) & 0xff);
	out[0] = static_cast<uint8_t>(x);
	out[1] = static_cast<uint8_t>(x >> 8);
	out[2] = static_cast<uint8_t>(y);
	out[3] = static_cast<uint8_t>(y >> 8);
}

// The in-block offset of the real Standard64KB layout for 16-byte elements is a permutation of the block's 4096 elements.
void TestBlockPermutation() {
	TileTextureBlockLayout texture {};
	Check(TileGetTextureBlockLayout(Prospero::BufferFormat::kBc7UNorm, Prospero::TileMode::kStandard64KB, false, texture), "BC7 block layout");
	const auto& block = texture.block;
	Check(block.bytes_per_element == 16 && texture.texel_width == 4 && texture.texel_height == 4, "BC7 element layout");
	Check(block.block_size == 65536 && block.block_width * block.block_height * 16 == block.block_size, "BC7 64KB block geometry");
	std::set<uint32_t> seen;
	for (uint32_t y = 0; y < block.block_height; y++) {
		for (uint32_t x = 0; x < block.block_width; x++) {
			uint32_t within = 0;
			Check(TileGetBlockOffset(block, x, y, 0, within) && within % 16 == 0 && within < block.block_size, "in-block offset");
			seen.insert(within);
		}
	}
	Check(seen.size() == block.block_width * block.block_height, "in-block offsets are not a permutation");
}

// Tile a whole mip chain forward with the real layout, then detile each level with BcDetile.
void TestSurface(uint32_t width, uint32_t height, uint32_t levels) {
	TileSurfaceLayout surface {};
	const TileSurfaceDescription description {Prospero::BufferFormat::kBc7UNorm, Prospero::TileMode::kStandard64KB, TileSurfaceDimension::Dim2D,
	                                          width, height, 1, levels, 1};
	Check(TileGetTiledTextureLayout(description, surface), "surface layout");
	const auto& block = surface.texture.block;
	std::vector<uint8_t> memory(surface.total_size, 0xcc);
	const uint32_t columns_of = block.block_width;
	for (uint32_t level = 0; level < levels; level++) {
		const auto& mip  = surface.mips[level];
		const bool  tail = level >= surface.first_tail_level;
		const auto  columns = (mip.padded_width + columns_of - 1) / columns_of;
		for (uint32_t y = 0; y < mip.height; y++) {
			for (uint32_t x = 0; x < mip.width; x++) {
				uint32_t within = 0;
				uint64_t at     = mip.offset;
				if (tail) {
					Check(TileGetBlockOffset(block, x + mip.tail_x, y + mip.tail_y, 0, within), "tail address");
					at += within;
				} else {
					Check(TileGetBlockOffset(block, x % block.block_width, y % block.block_height, 0, within), "address");
					at += (static_cast<uint64_t>(y / block.block_height) * columns + x / block.block_width) * block.block_size + within;
				}
				Check(at + 16 <= memory.size(), "forward tiler stays inside the surface");
				Element(memory.data() + at, level, x, y);
			}
		}
	}
	for (uint32_t level = 0; level < levels; level++) {
		const auto& mip = surface.mips[level];
		BcTileSpec  spec;
		spec.block_width  = block.block_width;
		spec.block_height = block.block_height;
		spec.block_bytes  = block.block_size;
		spec.tiled_offset = mip.offset;
		spec.tiled_size   = mip.size;
		spec.width        = mip.width;
		spec.height       = mip.height;
		spec.tiled_width  = mip.padded_width;
		spec.tiled_height = mip.padded_height;
		spec.tail         = level >= surface.first_tail_level;
		spec.tail_x       = spec.tail ? mip.tail_x : 0;
		spec.tail_y       = spec.tail ? mip.tail_y : 0;
		std::vector<uint8_t> out;
		const auto missing = BcDetile(memory.data(), memory.size(), spec, [&](uint32_t x, uint32_t y, uint32_t& within) {
			return TileGetBlockOffset(block, x, y, 0, within);
		}, out);
		Check(missing == 0, "real layout: elements missing");
		bool ok = true;
		for (uint32_t y = 0; y < mip.height && ok; y++) {
			for (uint32_t x = 0; x < mip.width && ok; x++) {
				uint8_t want[16];
				Element(want, level, x, y);
				ok = std::memcmp(out.data() + (static_cast<size_t>(y) * mip.width + x) * 16, want, 16) == 0;
			}
		}
		if (!ok) std::fprintf(stderr, "BcDumpTileTests: %ux%u level %u (tail %d)\n", width, height, level, spec.tail ? 1 : 0);
		Check(ok, "real layout: detiled element differs");
	}
}

} // namespace

int main() {
	TestBlockPermutation();
	TestSurface(1024, 512, 1);  // whole blocks, row-major block order (4 x 2 tiled blocks)
	TestSurface(2048, 2048, 1); // the terrain mask of the game: 512 x 512 elements = 8 x 8 tiled blocks
	TestSurface(300, 200, 9);   // padded edges and a mip tail
	std::printf("BcDumpTileTests: ok\n");
	return 0;
}
