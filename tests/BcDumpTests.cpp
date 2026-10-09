#include "graphics/host_gpu/renderer/bcDump.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics::DiagDump;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "BcDumpTests: failed: %s\n", message);
		std::exit(1);
	}
}

// ---- a tiled surface written by an independent forward tiler ----------------------------------------------------------

// Tiled block of 8x8 elements (1 KiB): the in-block offset interleaves the bits of x and y (Morton order).
uint32_t Morton(uint32_t x, uint32_t y) {
	uint32_t m = 0;
	for (uint32_t b = 0; b < 3; b++) m |= (((x >> b) & 1u) << (2 * b)) | (((y >> b) & 1u) << (2 * b + 1));
	return m;
}

bool MortonOffset(uint32_t x, uint32_t y, uint32_t& within) {
	if (x >= 8 || y >= 8) return false;
	within = Morton(x, y) * 16;
	return true;
}

// Element (x, y) carries its own coordinates, so a wrong address is visible.
void Element(uint8_t* out, uint32_t x, uint32_t y) {
	for (int i = 0; i < 16; i++) out[i] = static_cast<uint8_t>((x * 7 + y * 13 + i * 31) & 0xff);
	out[0] = static_cast<uint8_t>(x);
	out[1] = static_cast<uint8_t>(y);
}

// 24x16 elements = 3 x 2 tiled blocks, starting 256 bytes into the surface.
void TestDetile() {
	const uint32_t columns = 3, rows = 2, width = 24, height = 16;
	std::vector<uint8_t> surface(256 + columns * rows * 1024, 0xee);
	for (uint32_t by = 0; by < rows; by++) {
		for (uint32_t bx = 0; bx < columns; bx++) {
			for (uint32_t iy = 0; iy < 8; iy++) {
				for (uint32_t ix = 0; ix < 8; ix++) {
					Element(surface.data() + 256 + (by * columns + bx) * 1024 + Morton(ix, iy) * 16, bx * 8 + ix, by * 8 + iy);
				}
			}
		}
	}
	BcTileSpec spec;
	spec.block_width = spec.block_height = 8;
	spec.block_bytes                     = 1024;
	spec.tiled_offset                    = 256;
	spec.tiled_size                      = columns * rows * 1024;
	spec.width                           = width;
	spec.height                          = height;
	spec.tiled_width                     = 24;
	spec.tiled_height                    = 16;
	std::vector<uint8_t> linear;
	const auto           missing = BcDetile(surface.data(), surface.size(), spec, MortonOffset, linear);
	Check(missing == 0 && linear.size() == width * height * 16, "detile size / missing");
	bool ok = true;
	for (uint32_t y = 0; y < height; y++) {
		for (uint32_t x = 0; x < width; x++) {
			uint8_t want[16];
			Element(want, x, y);
			ok = ok && std::memcmp(linear.data() + (y * width + x) * 16, want, 16) == 0;
		}
	}
	Check(ok, "detiled element does not match the forward tiler");

	// A surface cut short: the elements past its end stay zero and are counted.
	std::vector<uint8_t> shorter;
	const auto           lost = BcDetile(surface.data(), 256 + 4 * 1024, spec, MortonOffset, shorter);
	Check(lost == 2 * 8 * 8, "elements past the readable range are counted");
	Check(shorter[(15 * width + 23) * 16] == 0 && shorter[(15 * width + 23) * 16 + 1] == 0, "unreadable element not zero");

	// An element without an in-block address is missing, not copied.
	std::vector<uint8_t> none;
	Check(BcDetile(surface.data(), surface.size(), spec, [](uint32_t, uint32_t, uint32_t&) { return false; }, none) == width * height,
	      "elements without an address are counted");

	// Mip tail: a 4x4 mip inside one tiled block at (tail_x, tail_y) = (2, 4); addressed by the in-block offset only.
	BcTileSpec tail = spec;
	tail.tail = true;
	tail.tail_x = 2;
	tail.tail_y = 4;
	tail.tiled_offset = 256 + 1024; // the second tiled block
	tail.width = tail.height = 4;
	std::vector<uint8_t> tail_out;
	Check(BcDetile(surface.data(), surface.size(), tail, MortonOffset, tail_out) == 0, "tail detile missing");
	bool tail_ok = true;
	for (uint32_t y = 0; y < 4; y++) {
		for (uint32_t x = 0; x < 4; x++) {
			uint8_t want[16];
			Element(want, 8 + 2 + x, 4 + y); // block 1 = (bx 1, by 0)
			tail_ok = tail_ok && std::memcmp(tail_out.data() + (y * 4 + x) * 16, want, 16) == 0;
		}
	}
	Check(tail_ok, "mip tail element address");
}

// ---- analysis ---------------------------------------------------------------------------------------------------------

struct Writer {
	uint8_t  bytes[16] {};
	uint32_t pos = 0;
	void     Put(uint32_t value, uint32_t count) {
		for (uint32_t i = 0; i < count; i++, pos++) bytes[pos >> 3] |= static_cast<uint8_t>(((value >> i) & 1u) << (pos & 7));
	}
};

// Mode 6 with both endpoints = `v` (7-bit, p-bit 0) and every index 0: a flat block of colour (2v, 2v|v>>6).
Writer Flat(uint32_t v, uint32_t a) {
	Writer w;
	w.Put(0x40, 7);
	for (int c = 0; c < 4; c++) {
		const uint32_t e = c == 3 ? a : v;
		w.Put(e, 7);
		w.Put(e, 7);
	}
	w.Put(0, 2);
	w.Put(0, 3);
	for (int i = 1; i < 16; i++) w.Put(0, 4);
	return w;
}

void TestAnalyze() {
	std::vector<uint8_t> blocks(6 * 16, 0);
	const auto           black = Flat(0, 0x7f), grey = Flat(0x40, 0x7f);
	std::memcpy(&blocks[0], black.bytes, 16);          // black (alpha 255)
	std::memcpy(&blocks[16], black.bytes, 16);         // black
	std::memcpy(&blocks[32], grey.bytes, 16);          // grey 128
	// blocks[48..64) all zero bytes: reserved, zero
	blocks[64] = 0x00;
	blocks[65] = 0x01; // reserved, not zero
	std::memcpy(&blocks[80], grey.bytes, 16);
	const auto s = Bc7Analyze(blocks.data(), 6);
	Check(s.blocks == 6 && s.decoded == 4 && s.reserved == 2, "analyze: decoded / reserved");
	Check(s.zero_blocks == 1, "analyze: zero blocks (raw bytes)");
	Check(s.black_blocks == 2, "analyze: black blocks");
	Check(s.mode_count[6] == 4, "analyze: mode histogram");
	Check(s.min[0] == 0 && s.max[0] == 128 && s.max[3] == 254, "analyze: min / max");
	Check(std::abs(s.mean[0] - 128.0 * 2 / 4) < 1e-9, "analyze: mean");
	const auto line = Bc7StatsLine("guest", s);
	Check(line.find("zero_blocks=1 (16.67%)") != std::string::npos && line.find("black_blocks=2 (33.33% of blocks)") != std::string::npos,
	      "analyze: stats line");
}

void TestCompare() {
	const uint32_t bw = 4, bh = 4;
	std::vector<uint8_t> guest(bw * bh * 16), host;
	for (size_t i = 0; i < guest.size(); i++) guest[i] = static_cast<uint8_t>(i * 5 + 3);
	host = guest;
	auto same = Bc7Compare(guest.data(), host.data(), bw, bh);
	Check(same.differ == 0 && same.first == -1, "compare: identical surfaces differ");
	host[5 * 16 + 7] ^= 0x10; // a byte past the first one of block 5 = (x 1, y 1)
	host[15 * 16] ^= 0x01;    // block 15 = (3, 3)
	const auto d = Bc7Compare(guest.data(), host.data(), bw, bh);
	Check(d.differ == 2 && d.blocks == 16, "compare: differing blocks");
	Check(d.first == 5, "compare: first differing block");
	Check(d.grid_diff[5] == 1 && d.grid_diff[15] == 1 && d.grid_blocks[0] == 1, "compare: grid cells");
	const auto line = Bc7DiffLine(d, bw);
	Check(line.find("differ=2 (12.50%)") != std::string::npos && line.find("first=5(x=1,y=1)") != std::string::npos, "compare: diff line");
	// A guest block of zeros against a non-zero host block is the "never written" signature.
	std::vector<uint8_t> zero = guest;
	std::memset(&zero[2 * 16], 0, 16);
	const auto z = Bc7Compare(zero.data(), guest.data(), bw, bh);
	Check(z.differ == 1 && z.guest_zero_host_not == 1, "compare: guest zero block");
	const auto map = Bc7DiffMap(guest.data(), host.data(), bw, bh, 2);
	Check(map.size() == 8 * 8 * 4 && map[(2 * 8 + 2) * 4] == 255 && map[0] == 0, "compare: diff map");
}

// ---- image output -----------------------------------------------------------------------------------------------------

uint32_t Be32(const uint8_t* p) {
	return (static_cast<uint32_t>(p[0]) << 24) | (static_cast<uint32_t>(p[1]) << 16) | (static_cast<uint32_t>(p[2]) << 8) | p[3];
}

// Reads the stored-deflate PNG back: returns the RGBA pixels, checks every chunk CRC and the Adler-32.
bool PngRead(const std::vector<uint8_t>& png, uint32_t& width, uint32_t& height, std::vector<uint8_t>& rgba) {
	static const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0d, 0x0a, 0x1a, 0x0a};
	if (png.size() < 8 || std::memcmp(png.data(), sig, 8) != 0) return false;
	std::vector<uint8_t> idat;
	size_t               at = 8;
	bool                 end = false;
	while (at + 12 <= png.size()) {
		const uint32_t n = Be32(&png[at]);
		if (at + 12 + n > png.size()) return false;
		if (Be32(&png[at + 8 + n]) != Crc32(&png[at + 4], n + 4)) return false;
		if (std::memcmp(&png[at + 4], "IHDR", 4) == 0) {
			width  = Be32(&png[at + 8]);
			height = Be32(&png[at + 12]);
			if (png[at + 16] != 8 || png[at + 17] != 6) return false;
		} else if (std::memcmp(&png[at + 4], "IDAT", 4) == 0) {
			idat.insert(idat.end(), png.begin() + at + 8, png.begin() + at + 8 + n);
		} else if (std::memcmp(&png[at + 4], "IEND", 4) == 0) {
			end = true;
		}
		at += 12 + n;
	}
	if (!end || idat.size() < 6 || idat[0] != 0x78) return false;
	std::vector<uint8_t> raw;
	size_t               z = 2;
	for (;;) {
		if (z + 5 > idat.size()) return false;
		const bool     final = (idat[z] & 1) != 0;
		const uint32_t n     = idat[z + 1] | (idat[z + 2] << 8);
		if ((idat[z] & 6) != 0 || (n ^ 0xffffu) != static_cast<uint32_t>(idat[z + 3] | (idat[z + 4] << 8))) return false;
		if (z + 5 + n > idat.size()) return false;
		raw.insert(raw.end(), idat.begin() + z + 5, idat.begin() + z + 5 + n);
		z += 5 + n;
		if (final) break;
	}
	if (z + 4 != idat.size() || Be32(&idat[z]) != Adler32(raw.data(), raw.size())) return false;
	rgba.clear();
	for (uint32_t y = 0; y < height; y++) {
		const size_t row = static_cast<size_t>(y) * (width * 4 + 1);
		if (row + 1 + width * 4 > raw.size() || raw[row] != 0) return false;
		rgba.insert(rgba.end(), raw.begin() + row + 1, raw.begin() + row + 1 + width * 4);
	}
	return true;
}

void TestImages() {
	const uint8_t nine[] = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
	Check(Crc32(nine, 9) == 0xcbf43926u, "crc32 check value");
	const uint8_t wiki[] = {'W', 'i', 'k', 'i', 'p', 'e', 'd', 'i', 'a'};
	Check(Adler32(wiki, 9) == 0x11e60398u, "adler32 check value");

	// Small, and large enough for two stored deflate blocks (200 * 100 * 4 > 65535).
	for (const auto [w, h]: {std::pair<uint32_t, uint32_t> {2, 2}, {200, 100}, {1, 1}}) {
		std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);
		for (size_t i = 0; i < rgba.size(); i++) rgba[i] = static_cast<uint8_t>(i * 37 + i / 251);
		const auto png = PngEncodeRgba(rgba.data(), w, h);
		uint32_t   rw = 0, rh = 0;
		std::vector<uint8_t> back;
		Check(PngRead(png, rw, rh, back) && rw == w && rh == h && back == rgba, "png round trip");
	}

	Check(FactorFor(2048, 2048, 1024) == 2 && FactorFor(1024, 512, 1024) == 1 && FactorFor(4096, 1024, 1024) == 4 && FactorFor(3000, 10, 1024) == 3,
	      "downscale factor");
	const std::vector<uint8_t> four = {10, 20, 30, 40, 30, 40, 50, 60, 50, 60, 70, 80, 70, 80, 90, 100};
	uint32_t                   ow = 0, oh = 0;
	const auto                 half = DownscaleRgba(four, 2, 2, 2, ow, oh);
	Check(ow == 1 && oh == 1 && half[0] == 40 && half[1] == 50 && half[2] == 60 && half[3] == 70, "box filter");

	// Texel placement: block 1 of a 2x1 surface, texel 9 = (x 1, y 2) of that block.
	const auto           grey = Flat(0x40, 0x7f);
	std::vector<uint8_t> blocks(32, 0);
	std::memcpy(&blocks[0], Flat(0, 0x7f).bytes, 16);
	std::memcpy(&blocks[16], grey.bytes, 16);
	const auto image = Bc7ToRgba(blocks.data(), 2, 1);
	Check(image.size() == 8 * 4 * 4, "rgba size");
	Check(image[(2 * 8 + 4 + 1) * 4] == 128 && image[(2 * 8 + 1) * 4] == 0, "texel placement in the decoded image");
}

} // namespace

int main() {
	TestDetile();
	TestAnalyze();
	TestCompare();
	TestImages();
	std::printf("BcDumpTests: ok\n");
	return 0;
}
