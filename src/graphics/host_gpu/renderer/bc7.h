#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_BC7_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_BC7_H_

// S123 diag_tex: CPU decode of BC7 blocks (pure, no Vulkan). Modes 4, 5 and 6 (one subset) are decoded; modes 0..3 and 7
// (partitioned) report only their mode number. Used to see what colour a BC7 T# holds in guest memory.

#include <cstdint>

namespace Libs::Graphics::DiagDump {

struct Bc7Block {
	int     mode = -1; // 0..7, -1 = reserved (no bit set in the first byte)
	bool    ok   = false; // texels[] filled
	uint8_t texels[16][4] {}; // RGBA8, row-major 4x4
};

namespace Bc7Detail {

struct Bits {
	const uint8_t* data;
	uint32_t       pos = 0;
	uint32_t       Read(uint32_t count) {
		uint32_t value = 0;
		for (uint32_t i = 0; i < count; i++, pos++) value |= static_cast<uint32_t>((data[pos >> 3] >> (pos & 7)) & 1u) << i;
		return value;
	}
};

inline uint32_t Weight(uint32_t bits, uint32_t index) {
	static const uint8_t w2[4]  = {0, 21, 43, 64};
	static const uint8_t w3[8]  = {0, 9, 18, 27, 37, 46, 55, 64};
	static const uint8_t w4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};
	return bits == 2 ? w2[index & 3] : bits == 3 ? w3[index & 7] : w4[index & 15];
}

inline uint8_t Lerp(uint32_t a, uint32_t b, uint32_t w) {
	return static_cast<uint8_t>(((64u - w) * a + w * b + 32u) >> 6);
}

// n-bit endpoint -> 8 bits by bit replication.
inline uint32_t Expand(uint32_t value, uint32_t bits) {
	return bits == 8 ? value : (value << (8 - bits)) | (value >> (2 * bits - 8));
}

inline void Rotate(uint8_t texel[4], uint32_t rotation) {
	if (rotation == 0) return;
	const auto t       = texel[3];
	texel[3]           = texel[rotation - 1];
	texel[rotation - 1] = t;
}

} // namespace Bc7Detail

// `block` points at 16 bytes.
inline Bc7Block Bc7Decode(const uint8_t* block) {
	using namespace Bc7Detail;
	Bc7Block out;
	int      mode = 0;
	while (mode < 8 && ((block[0] >> mode) & 1) == 0) mode++;
	if (mode == 8) return out;
	out.mode = mode;
	Bits bits {block};
	bits.pos = static_cast<uint32_t>(mode + 1);
	if (mode == 6) {
		uint32_t e[2][4];
		for (uint32_t c = 0; c < 4; c++) {
			e[0][c] = bits.Read(7);
			e[1][c] = bits.Read(7);
		}
		const uint32_t p0 = bits.Read(1), p1 = bits.Read(1);
		for (uint32_t c = 0; c < 4; c++) {
#ifdef KYTY_BC7_MUTANT_NO_PBIT
			(void)p0; (void)p1;
			e[0][c] <<= 1;
			e[1][c] <<= 1;
#else
			e[0][c] = (e[0][c] << 1) | p0;
			e[1][c] = (e[1][c] << 1) | p1;
#endif
		}
		for (uint32_t i = 0; i < 16; i++) { // pixel 0 is the anchor: its index has no top bit
			const uint32_t w = Weight(4, bits.Read(i == 0 ? 3 : 4));
			for (uint32_t c = 0; c < 4; c++) out.texels[i][c] = Lerp(e[0][c], e[1][c], w);
		}
		out.ok = true;
	} else if (mode == 5) {
		const uint32_t rotation = bits.Read(2);
		uint32_t       e[2][4];
		for (uint32_t c = 0; c < 3; c++) {
			e[0][c] = Expand(bits.Read(7), 7);
			e[1][c] = Expand(bits.Read(7), 7);
		}
		e[0][3] = bits.Read(8);
		e[1][3] = bits.Read(8);
		uint32_t color[16], alpha[16];
		for (uint32_t i = 0; i < 16; i++) color[i] = bits.Read(i == 0 ? 1 : 2);
		for (uint32_t i = 0; i < 16; i++) alpha[i] = bits.Read(i == 0 ? 1 : 2);
		for (uint32_t i = 0; i < 16; i++) {
			for (uint32_t c = 0; c < 3; c++) out.texels[i][c] = Lerp(e[0][c], e[1][c], Weight(2, color[i]));
			out.texels[i][3] = Lerp(e[0][3], e[1][3], Weight(2, alpha[i]));
			Rotate(out.texels[i], rotation);
		}
		out.ok = true;
	} else if (mode == 4) {
		const uint32_t rotation = bits.Read(2);
		const uint32_t idx_mode = bits.Read(1);
		uint32_t       e[2][4];
		for (uint32_t c = 0; c < 3; c++) {
			e[0][c] = Expand(bits.Read(5), 5);
			e[1][c] = Expand(bits.Read(5), 5);
		}
		e[0][3] = Expand(bits.Read(6), 6);
		e[1][3] = Expand(bits.Read(6), 6);
		uint32_t two[16], three[16];
		for (uint32_t i = 0; i < 16; i++) two[i] = bits.Read(i == 0 ? 1 : 2);
		for (uint32_t i = 0; i < 16; i++) three[i] = bits.Read(i == 0 ? 2 : 3);
		for (uint32_t i = 0; i < 16; i++) {
			const uint32_t wc = idx_mode == 0 ? Weight(2, two[i]) : Weight(3, three[i]);
			const uint32_t wa = idx_mode == 0 ? Weight(3, three[i]) : Weight(2, two[i]);
			for (uint32_t c = 0; c < 3; c++) out.texels[i][c] = Lerp(e[0][c], e[1][c], wc);
			out.texels[i][3] = Lerp(e[0][3], e[1][3], wa);
			Rotate(out.texels[i], rotation);
		}
		out.ok = true;
	}
	return out;
}

// First block plus the mean RGBA over the first `max_blocks` blocks of `bytes` (decodable blocks only).
struct Bc7Summary {
	int      first_mode = -1;
	bool     first_ok   = false;
	uint8_t  first_texel[4] {}; // texel 0 of the first block
	uint32_t blocks = 0, decoded = 0;
	double   mean[4] {};
};

inline Bc7Summary Bc7Summarize(const uint8_t* data, uint64_t bytes, uint32_t max_blocks) {
	Bc7Summary     s;
	const uint64_t count  = bytes / 16 < max_blocks ? bytes / 16 : max_blocks;
	double         sum[4] = {};
	for (uint64_t b = 0; b < count; b++) {
		const auto block = Bc7Decode(data + b * 16);
		if (b == 0) {
			s.first_mode = block.mode;
			s.first_ok   = block.ok;
			if (block.ok) {
				for (int c = 0; c < 4; c++) s.first_texel[c] = block.texels[0][c];
			}
		}
		s.blocks++;
		if (!block.ok) continue;
		s.decoded++;
		for (uint32_t i = 0; i < 16; i++) {
			for (int c = 0; c < 4; c++) sum[c] += block.texels[i][c];
		}
	}
	if (s.decoded != 0) {
		for (int c = 0; c < 4; c++) s.mean[c] = sum[c] / (16.0 * s.decoded);
	}
	return s;
}

} // namespace Libs::Graphics::DiagDump

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_BC7_H_ */
