#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_BCDUMP_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_BCDUMP_H_

// S124 dump_tex: pure helpers (no Vulkan, no tile.cpp) for comparing a BC7 texture in guest memory with the host image
// the shader samples. The guest side is detiled the way gpu_tiler_common.inc addresses a texel (block index within the
// mip * tiled block size + in-block offset); the in-block offset is a parameter, so the emulator passes
// TileGetBlockOffset and the tests pass an independent layout.

#include "graphics/host_gpu/renderer/bc7.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics::DiagDump {

// mip 0 / layer 0 of a tiled surface, in addressable elements (one BC block = one element of 16 bytes).
struct BcTileSpec {
	uint32_t block_width = 0, block_height = 0; // elements per tiled block
	uint32_t block_bytes = 0;                   // bytes of one tiled block
	uint32_t elem_bytes  = 16;
	uint64_t tiled_offset = 0, tiled_size = 0;  // the mip inside the surface
	uint32_t width = 0, height = 0;             // elements of the mip
	uint32_t tiled_width = 0, tiled_height = 0; // padded to whole tiled blocks
	bool     tail = false;
	uint32_t tail_x = 0, tail_y = 0;
};

// Copies the elements of the mip into `out` (width*height*elem_bytes, row-major). `base` is the start of the surface in
// guest memory (`base_size` bytes are readable). `offset(x, y, within)` returns the in-block byte offset of element
// (x, y) of a tiled block, false when it has none. Elements outside the readable range stay zero; the count of those
// (and of elements without an address) is returned.
template <typename OffsetFn>
inline uint64_t BcDetile(const uint8_t* base, uint64_t base_size, const BcTileSpec& spec, OffsetFn&& offset, std::vector<uint8_t>& out) {
	out.assign(static_cast<size_t>(spec.width) * spec.height * spec.elem_bytes, 0);
	if (spec.block_width == 0 || spec.block_height == 0) return static_cast<uint64_t>(spec.width) * spec.height;
	const uint64_t columns = (spec.tiled_width + spec.block_width - 1u) / spec.block_width;
	uint64_t       missing = 0;
	for (uint32_t y = 0; y < spec.height; y++) {
		for (uint32_t x = 0; x < spec.width; x++) {
			uint32_t within = 0;
			uint64_t at     = 0;
			bool     ok     = false;
			if (spec.tail) {
				ok = offset(x + spec.tail_x, y + spec.tail_y, within);
				at = spec.tiled_offset + within;
			} else {
#ifdef KYTY_BCDUMP_MUTANT_BLOCK_ROW
				const uint64_t block = static_cast<uint64_t>(x / spec.block_width) * columns + y / spec.block_height;
#else
				const uint64_t block = static_cast<uint64_t>(y / spec.block_height) * columns + x / spec.block_width;
#endif
				ok = offset(x % spec.block_width, y % spec.block_height, within);
				at = spec.tiled_offset + block * spec.block_bytes + within;
			}
			if (!ok || at + spec.elem_bytes > base_size) {
				missing++;
				continue;
			}
			std::memcpy(out.data() + (static_cast<size_t>(y) * spec.width + x) * spec.elem_bytes, base + at, spec.elem_bytes);
		}
	}
	return missing;
}

// S125: the spec that detiles tile `index` (counted in tiled blocks from the start of the surface) alone: one tiled block of
// block_width x block_height elements, no tail, no padding.
inline BcTileSpec SingleTileSpec(const BcTileSpec& surface, uint64_t index) {
	BcTileSpec each   = surface;
	each.width        = each.tiled_width  = surface.block_width;
	each.height       = each.tiled_height = surface.block_height;
	each.tiled_size   = surface.block_bytes;
	each.tail         = false;
#ifdef KYTY_BCDUMP_MUTANT_TILE0
	each.tiled_offset = 0;
#else
	each.tiled_offset = index * surface.block_bytes;
#endif
	return each;
}

// ---- analysis ---------------------------------------------------------------------------------------------------------

struct Bc7Stats {
	uint32_t blocks = 0, decoded = 0, reserved = 0, zero_blocks = 0, black_blocks = 0; // black: decoded, every texel rgb == 0
	uint32_t mode_count[8] {};
	uint8_t  min[4] {255, 255, 255, 255}, max[4] {};
	double   mean[4] {};
};

inline bool Bc7BlockIsZero(const uint8_t* block) {
	for (int i = 0; i < 16; i++) {
		if (block[i] != 0) return false;
	}
	return true;
}

inline bool Bc7BlockIsBlack(const Bc7Block& block) {
	if (!block.ok) return false;
	for (int i = 0; i < 16; i++) {
		if (block.texels[i][0] != 0 || block.texels[i][1] != 0 || block.texels[i][2] != 0) return false;
	}
	return true;
}

inline Bc7Stats Bc7Analyze(const uint8_t* blocks, uint32_t count) {
	Bc7Stats s;
	double   sum[4] = {};
	for (uint32_t b = 0; b < count; b++) {
		const uint8_t* raw = blocks + static_cast<size_t>(b) * 16;
		const auto     dec = Bc7Decode(raw);
		s.blocks++;
		if (Bc7BlockIsZero(raw)) s.zero_blocks++;
		if (!dec.ok) {
			s.reserved++;
			continue;
		}
		s.decoded++;
		s.mode_count[dec.mode]++;
		if (Bc7BlockIsBlack(dec)) s.black_blocks++;
		for (int i = 0; i < 16; i++) {
			for (int c = 0; c < 4; c++) {
				sum[c] += dec.texels[i][c];
				s.min[c] = std::min(s.min[c], dec.texels[i][c]);
				s.max[c] = std::max(s.max[c], dec.texels[i][c]);
			}
		}
	}
	if (s.decoded != 0) {
		for (int c = 0; c < 4; c++) s.mean[c] = sum[c] / (16.0 * s.decoded);
	} else {
		for (int c = 0; c < 4; c++) s.min[c] = 0;
	}
	return s;
}

inline double Percent(uint64_t part, uint64_t total) {
	return total == 0 ? 0.0 : 100.0 * static_cast<double>(part) / static_cast<double>(total);
}

// "[dump-tex] <side> blocks=N decoded=N reserved=N zero=N (p%) black=N (p%) modes=... min=... max=... mean=..."
inline std::string Bc7StatsLine(const char* side, const Bc7Stats& s) {
	char buf[512];
	std::snprintf(buf, sizeof(buf),
	              "[dump-tex] %s blocks=%u decoded=%u reserved=%u zero_blocks=%u (%.2f%%) black_blocks=%u (%.2f%% of blocks) "
	              "modes=%u,%u,%u,%u,%u,%u,%u,%u min=%u,%u,%u,%u max=%u,%u,%u,%u mean=%.2f,%.2f,%.2f,%.2f",
	              side, s.blocks, s.decoded, s.reserved, s.zero_blocks, Percent(s.zero_blocks, s.blocks), s.black_blocks,
	              Percent(s.black_blocks, s.blocks), s.mode_count[0], s.mode_count[1], s.mode_count[2], s.mode_count[3], s.mode_count[4],
	              s.mode_count[5], s.mode_count[6], s.mode_count[7], s.min[0], s.min[1], s.min[2], s.min[3], s.max[0], s.max[1], s.max[2],
	              s.max[3], s.mean[0], s.mean[1], s.mean[2], s.mean[3]);
	return buf;
}

// Guest blocks vs host blocks (same dimensions, both row-major).
struct Bc7Diff {
	uint32_t blocks = 0, differ = 0;
	uint32_t pixel_differ = 0;         // differing blocks whose decoded texels also differ
	uint32_t guest_black_host_not = 0; // guest block decodes to black, host block does not
	uint32_t host_black_guest_not = 0;
	uint32_t guest_zero_host_not = 0;  // guest block is all-zero bytes, host block is not
	int64_t  first = -1;               // first differing block (row-major index)
	uint32_t grid_diff[16] {}, grid_blocks[16] {}; // 4x4 cells over the surface
};

inline Bc7Diff Bc7Compare(const uint8_t* guest, const uint8_t* host, uint32_t bw, uint32_t bh) {
	Bc7Diff d;
	d.blocks = bw * bh;
	for (uint32_t y = 0; y < bh; y++) {
		for (uint32_t x = 0; x < bw; x++) {
			const size_t   at   = (static_cast<size_t>(y) * bw + x) * 16;
			const uint32_t cell = (y * 4 / std::max(bh, 1u)) * 4 + x * 4 / std::max(bw, 1u);
			d.grid_blocks[cell]++;
#ifdef KYTY_BCDUMP_MUTANT_DIFF_FIRST_BYTE
			const bool same = guest[at] == host[at];
#else
			const bool same = std::memcmp(guest + at, host + at, 16) == 0;
#endif
			if (same) continue;
			d.differ++;
			d.grid_diff[cell]++;
			if (d.first < 0) d.first = static_cast<int64_t>(y) * bw + x;
			const auto g = Bc7Decode(guest + at), h = Bc7Decode(host + at);
			if (Bc7BlockIsBlack(g) && !Bc7BlockIsBlack(h)) d.guest_black_host_not++;
			if (Bc7BlockIsBlack(h) && !Bc7BlockIsBlack(g)) d.host_black_guest_not++;
			if (Bc7BlockIsZero(guest + at) && !Bc7BlockIsZero(host + at)) d.guest_zero_host_not++;
			if (g.ok != h.ok || std::memcmp(g.texels, h.texels, sizeof(g.texels)) != 0) d.pixel_differ++;
		}
	}
	return d;
}

inline std::string Bc7DiffLine(const Bc7Diff& d, uint32_t bw) {
	char       buf[640];
	const auto w = static_cast<int64_t>(std::max(bw, 1u));
	int        n = std::snprintf(buf, sizeof(buf),
	                             "[dump-tex] diff blocks=%u differ=%u (%.2f%%) texels_differ_blocks=%u guest_black_host_not=%u "
	                             "host_black_guest_not=%u guest_zero_host_not=%u first=%" PRId64 "(x=%" PRId64 ",y=%" PRId64
	                             ") grid4x4(%%differ)=",
	                             d.blocks, d.differ, Percent(d.differ, d.blocks), d.pixel_differ, d.guest_black_host_not,
	                             d.host_black_guest_not, d.guest_zero_host_not, d.first, d.first < 0 ? -1 : d.first % w,
	                             d.first < 0 ? -1 : d.first / w);
	for (int i = 0; i < 16 && n > 0 && n < static_cast<int>(sizeof(buf)); i++) {
		n += std::snprintf(buf + n, sizeof(buf) - n, "%s%.0f", i == 0 ? "" : (i % 4 == 0 ? "/" : ","), Percent(d.grid_diff[i], d.grid_blocks[i]));
	}
	return buf;
}

// ---- images -----------------------------------------------------------------------------------------------------------

// RGBA8 of w = 4*bw by h = 4*bh texels.
inline std::vector<uint8_t> Bc7ToRgba(const uint8_t* blocks, uint32_t bw, uint32_t bh) {
	const size_t         width = static_cast<size_t>(bw) * 4;
	std::vector<uint8_t> rgba(width * bh * 4 * 4, 0);
	for (uint32_t by = 0; by < bh; by++) {
		for (uint32_t bx = 0; bx < bw; bx++) {
			const auto b = Bc7Decode(blocks + (static_cast<size_t>(by) * bw + bx) * 16);
			if (!b.ok) continue;
			for (uint32_t i = 0; i < 16; i++) {
				const size_t at = ((static_cast<size_t>(by) * 4 + (i >> 2)) * width + bx * 4 + (i & 3)) * 4;
				std::memcpy(rgba.data() + at, b.texels[i], 4);
			}
		}
	}
	return rgba;
}

// Box filter by an integer factor (the image is cropped to a multiple of it).
inline std::vector<uint8_t> DownscaleRgba(const std::vector<uint8_t>& rgba, uint32_t width, uint32_t height, uint32_t factor,
                                          uint32_t& out_width, uint32_t& out_height) {
	if (factor <= 1) {
		out_width  = width;
		out_height = height;
		return rgba;
	}
	out_width  = width / factor;
	out_height = height / factor;
	std::vector<uint8_t> out(static_cast<size_t>(out_width) * out_height * 4);
	for (uint32_t y = 0; y < out_height; y++) {
		for (uint32_t x = 0; x < out_width; x++) {
			uint32_t sum[4] = {};
			for (uint32_t j = 0; j < factor; j++) {
				for (uint32_t i = 0; i < factor; i++) {
					const size_t at = (static_cast<size_t>(y * factor + j) * width + x * factor + i) * 4;
					for (int c = 0; c < 4; c++) sum[c] += rgba[at + c];
				}
			}
			for (int c = 0; c < 4; c++) out[(static_cast<size_t>(y) * out_width + x) * 4 + c] = static_cast<uint8_t>(sum[c] / (factor * factor));
		}
	}
	return out;
}

// Smallest integer factor that brings the longer side to at most `limit`.
inline uint32_t FactorFor(uint32_t width, uint32_t height, uint32_t limit) {
	uint32_t factor = 1;
	while (std::max(width, height) / factor > limit) factor++;
	return factor;
}

inline uint32_t Crc32(const uint8_t* data, size_t size) {
	uint32_t crc = 0xffffffffu;
	for (size_t i = 0; i < size; i++) {
		crc ^= data[i];
		for (int k = 0; k < 8; k++) crc = (crc & 1u) ? 0xedb88320u ^ (crc >> 1) : crc >> 1;
	}
	return ~crc;
}

inline uint32_t Adler32(const uint8_t* data, size_t size) {
	uint32_t a = 1, b = 0;
	for (size_t i = 0; i < size; i++) {
		a = (a + data[i]) % 65521u;
		b = (b + a) % 65521u;
	}
	return (b << 16) | a;
}

// RGBA8 PNG with stored (uncompressed) deflate blocks: no zlib needed, files are about the size of the raw image.
inline std::vector<uint8_t> PngEncodeRgba(const uint8_t* rgba, uint32_t width, uint32_t height) {
	std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
	const auto put32 = [](std::vector<uint8_t>& v, uint32_t x) {
		for (int s = 24; s >= 0; s -= 8) v.push_back(static_cast<uint8_t>(x >> s));
	};
	const auto chunk = [&](const char* type, const std::vector<uint8_t>& body) {
		put32(png, static_cast<uint32_t>(body.size()));
		const size_t from = png.size();
		png.insert(png.end(), type, type + 4);
		png.insert(png.end(), body.begin(), body.end());
		put32(png, Crc32(png.data() + from, png.size() - from));
	};
	std::vector<uint8_t> ihdr;
	put32(ihdr, width);
	put32(ihdr, height);
	ihdr.insert(ihdr.end(), {8, 6, 0, 0, 0}); // 8 bit, RGBA, deflate, adaptive filter, no interlace
	chunk("IHDR", ihdr);
	std::vector<uint8_t> raw; // one filter byte (0) per scanline
	raw.reserve((static_cast<size_t>(width) * 4 + 1) * height);
	for (uint32_t y = 0; y < height; y++) {
		raw.push_back(0);
		raw.insert(raw.end(), rgba + static_cast<size_t>(y) * width * 4, rgba + static_cast<size_t>(y + 1) * width * 4);
	}
	std::vector<uint8_t> z = {0x78, 0x01};
	size_t               at = 0;
	do {
		const size_t n     = std::min<size_t>(65535, raw.size() - at);
		const bool   final = at + n >= raw.size();
		z.push_back(final ? 1 : 0);
		z.push_back(static_cast<uint8_t>(n & 0xff));
		z.push_back(static_cast<uint8_t>(n >> 8));
		z.push_back(static_cast<uint8_t>(~n & 0xff));
		z.push_back(static_cast<uint8_t>((~n >> 8) & 0xff));
		z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(at), raw.begin() + static_cast<std::ptrdiff_t>(at + n));
		at += n;
	} while (at < raw.size());
	put32(z, Adler32(raw.data(), raw.size()));
	chunk("IDAT", z);
	chunk("IEND", {});
	return png;
}

// One pixel per block (x `scale`): black = identical, white = bytes differ.
inline std::vector<uint8_t> Bc7DiffMap(const uint8_t* guest, const uint8_t* host, uint32_t bw, uint32_t bh, uint32_t scale) {
	std::vector<uint8_t> rgba(static_cast<size_t>(bw) * scale * bh * scale * 4, 255);
	for (uint32_t y = 0; y < bh * scale; y++) {
		for (uint32_t x = 0; x < bw * scale; x++) {
			const size_t at   = (static_cast<size_t>(y / scale) * bw + x / scale) * 16;
			const bool   same = std::memcmp(guest + at, host + at, 16) == 0;
			const size_t px   = (static_cast<size_t>(y) * bw * scale + x) * 4;
			rgba[px] = rgba[px + 1] = rgba[px + 2] = same ? 0 : 255;
		}
	}
	return rgba;
}

// <dir>/<label>-<flip>-dumptex-<ps>-s<slot>-<what>.png
inline std::string DumpTexPath(const std::string& dir, const std::string& label, uint64_t flip, uint64_t ps, uint32_t slot, const char* what) {
	char buf[96];
	std::snprintf(buf, sizeof(buf), "-%" PRIu64 "-dumptex-%016" PRIx64 "-s%u-%s.png", flip, ps, slot, what);
	return dir + "/" + (label.empty() ? std::string("dump_tex") : label) + buf;
}

} // namespace Libs::Graphics::DiagDump

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_BCDUMP_H_ */
