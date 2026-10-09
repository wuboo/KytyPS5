#include "graphics/host_gpu/renderer/bc7.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using namespace Libs::Graphics::DiagDump;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "Bc7Tests: failed: %s\n", message);
		std::exit(1);
	}
}

// LSB-first bit packer, the way the format stores its fields.
struct Writer {
	uint8_t  bytes[16] {};
	uint32_t pos = 0;
	void     Put(uint32_t value, uint32_t count) {
		for (uint32_t i = 0; i < count; i++, pos++) bytes[pos >> 3] |= static_cast<uint8_t>(((value >> i) & 1u) << (pos & 7));
	}
};

// Mode 6: endpoint 0 = (0x00,0x20,0x7f,0x7f)+p0=0 -> (0,64,254,254); endpoint 1 = (0x7f,0x10,0x00,0x40)+p1=1 -> (255,33,1,129).
// Pixel 0 (anchor, 3-bit index 7 -> weight 30), the other 15 pixels index 15 (weight 64 = endpoint 1).
void TestMode6() {
	Writer w;
	w.Put(0x40, 7); // mode 6
	const uint32_t e0[4] = {0x00, 0x20, 0x7f, 0x7f}, e1[4] = {0x7f, 0x10, 0x00, 0x40};
	for (int c = 0; c < 4; c++) {
		w.Put(e0[c], 7);
		w.Put(e1[c], 7);
	}
	w.Put(0, 1); // p0
	w.Put(1, 1); // p1
	w.Put(7, 3);
	for (int i = 1; i < 16; i++) w.Put(15, 4);
	Check(w.pos == 128, "mode 6 packs into 128 bits");
	const auto b = Bc7Decode(w.bytes);
	Check(b.ok && b.mode == 6, "mode 6 not decoded");
	// ((64-30)*e0 + 30*e1 + 32) >> 6 per channel: 120, 49, 135, 195.
	Check(b.texels[0][0] == 120 && b.texels[0][1] == 49 && b.texels[0][2] == 135 && b.texels[0][3] == 195, "mode 6 pixel 0");
	Check(b.texels[1][0] == 255 && b.texels[1][1] == 33 && b.texels[1][2] == 1 && b.texels[1][3] == 129, "mode 6 endpoint 1 (p-bit)");
	Check(b.texels[15][0] == 255 && b.texels[15][3] == 129, "mode 6 last pixel");
	const auto s = Bc7Summarize(w.bytes, 16, 256);
	Check(s.first_mode == 6 && s.first_ok && s.first_texel[0] == 120 && s.blocks == 1 && s.decoded == 1, "summary first block");
	Check(std::fabs(s.mean[0] - (120.0 + 15 * 255.0) / 16.0) < 1e-9 && std::fabs(s.mean[3] - (195.0 + 15 * 129.0) / 16.0) < 1e-9,
	      "summary mean");
}

// Mode 5, rotation 0: colour endpoints 7-bit expanded (0x7f -> 255), alpha 8-bit; all indices 0 -> endpoint 0.
void TestMode5() {
	Writer w;
	w.Put(0x20, 6); // mode 5
	w.Put(0, 2);    // rotation
	w.Put(0x7f, 7); w.Put(0x00, 7); // R0 R1
	w.Put(0x40, 7); w.Put(0x00, 7); // G0 G1
	w.Put(0x00, 7); w.Put(0x7f, 7); // B0 B1
	w.Put(0x80, 8); w.Put(0xff, 8); // A0 A1
	w.Put(0, 31);
	w.Put(0, 31);
	Check(w.pos == 128, "mode 5 packs into 128 bits");
	auto b = Bc7Decode(w.bytes);
	Check(b.ok && b.mode == 5 && b.texels[3][0] == 255 && b.texels[3][1] == 129 && b.texels[3][2] == 0 && b.texels[3][3] == 0x80, "mode 5");
	// Rotation 1 swaps alpha and red.
	w.bytes[0] = static_cast<uint8_t>((w.bytes[0] & ~0xC0) | (1 << 6));
	b = Bc7Decode(w.bytes);
	Check(b.texels[0][0] == 0x80 && b.texels[0][3] == 255, "mode 5 rotation");
}

// Mode 4: 5-bit colour, 6-bit alpha; idxMode 0, all indices 0 -> endpoint 0.
void TestMode4() {
	Writer w;
	w.Put(0x10, 5); // mode 4
	w.Put(0, 2);    // rotation
	w.Put(0, 1);    // idx mode
	w.Put(0x1f, 5); w.Put(0, 5); // R
	w.Put(0x10, 5); w.Put(0, 5); // G
	w.Put(0x00, 5); w.Put(0, 5); // B
	w.Put(0x20, 6); w.Put(0, 6); // A
	w.Put(0, 31);
	w.Put(0, 47);
	Check(w.pos == 128, "mode 4 packs into 128 bits");
	const auto b = Bc7Decode(w.bytes);
	Check(b.ok && b.mode == 4 && b.texels[0][0] == 255 && b.texels[0][1] == 132 && b.texels[0][2] == 0 && b.texels[0][3] == 130, "mode 4");
}

// Partitioned modes report their number only; an all-zero block is reserved.
void TestUndecoded() {
	uint8_t block[16] {};
	Check(!Bc7Decode(block).ok && Bc7Decode(block).mode == -1, "reserved block");
	block[0] = 0x02; // mode 1
	const auto b = Bc7Decode(block);
	Check(!b.ok && b.mode == 1, "mode 1 reports its number");
	const auto s = Bc7Summarize(block, 16, 256);
	Check(s.first_mode == 1 && !s.first_ok && s.decoded == 0 && s.blocks == 1, "summary of an undecoded block");
}

} // namespace

int main() {
	TestMode6();
	TestMode5();
	TestMode4();
	TestUndecoded();
	std::printf("Bc7Tests: ok\n");
	return 0;
}
