#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_BC7_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_BC7_H_

// S123 diag_tex / S124 dump_tex: CPU decode of BC7 blocks (pure, no Vulkan), all modes 0..7 (S124 added the partitioned
// modes 0..3 and 7; the partition and anchor tables were read back from an independent decoder, see Bc7Tests.cpp).
// Used to see what colour a BC7 T# holds in guest memory and in the host image.

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

// 2-bit subset index per pixel, pixel i in bits 2i..2i+1 (from the DXGI_FORMAT_BC7 partition tables)
inline constexpr uint32_t kPartition2[64] = {
	0x50505050, 0x40404040, 0x54545454, 0x54505040, 0x50404000, 0x55545450, 0x55545040, 0x54504000,
	0x50400000, 0x55555450, 0x55544000, 0x54400000, 0x55555440, 0x55550000, 0x55555500, 0x55000000,
	0x55150100, 0x00004054, 0x15010000, 0x00405054, 0x00004050, 0x15050100, 0x05010000, 0x40505054,
	0x00404050, 0x05010100, 0x14141414, 0x05141450, 0x01155440, 0x00555500, 0x15014054, 0x05414150,
	0x44444444, 0x55005500, 0x11441144, 0x05055050, 0x05500550, 0x11114444, 0x41144114, 0x44111144,
	0x15055054, 0x01055040, 0x05041050, 0x05455150, 0x14414114, 0x50050550, 0x41411414, 0x00141400,
	0x00041504, 0x00105410, 0x10541000, 0x04150400, 0x50410514, 0x41051450, 0x05415014, 0x14054150,
	0x41050514, 0x41505014, 0x40011554, 0x54150140, 0x50505500, 0x00555050, 0x15151010, 0x54540404,
};
inline constexpr uint32_t kPartition3[64] = {
	0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050, 0x5555a0a0, 0x5a5a5050,
	0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090, 0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250,
	0xa5945040, 0x0a425054, 0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
	0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414, 0x50a4a450, 0x6a5a0200,
	0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424, 0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50,
	0x500aa550, 0xaaaa4444, 0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
	0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580, 0xaa141414, 0x96960000,
	0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000, 0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
};
inline constexpr uint8_t kAnchor2[64] = {
	15, 15, 15, 15, 15, 15, 15, 15,
	15, 15, 15, 15, 15, 15, 15, 15,
	15, 2, 8, 2, 2, 8, 8, 15,
	2, 8, 2, 2, 8, 8, 2, 2,
	15, 15, 6, 8, 2, 8, 15, 15,
	2, 8, 2, 2, 2, 15, 15, 6,
	6, 2, 6, 8, 15, 15, 2, 2,
	15, 15, 15, 15, 15, 2, 2, 15,
};
inline constexpr uint8_t kAnchor3a[64] = {
	3, 3, 15, 15, 8, 3, 15, 15,
	8, 8, 6, 6, 6, 5, 3, 3,
	3, 3, 8, 15, 3, 3, 6, 10,
	5, 8, 8, 6, 8, 5, 15, 15,
	8, 15, 3, 5, 6, 10, 8, 15,
	15, 3, 15, 5, 15, 15, 15, 15,
	3, 15, 5, 5, 5, 8, 5, 10,
	5, 10, 8, 13, 15, 12, 3, 3,
};
inline constexpr uint8_t kAnchor3b[64] = {
	15, 8, 8, 3, 15, 15, 3, 8,
	15, 15, 15, 15, 15, 15, 15, 8,
	15, 8, 15, 3, 15, 8, 15, 8,
	3, 15, 6, 10, 15, 15, 10, 8,
	15, 3, 15, 10, 10, 8, 9, 10,
	6, 15, 8, 15, 3, 6, 6, 8,
	15, 3, 15, 15, 15, 15, 15, 15,
	15, 15, 15, 15, 3, 15, 15, 8,
};

// Modes 0..3 and 7: 2 or 3 subsets chosen by a partition number.
struct PartMode {
	uint8_t subsets, partition_bits, color_bits, alpha_bits, pbits_kind /*0 none, 1 per endpoint, 2 shared per subset*/, index_bits;
};
inline constexpr PartMode kPartModes[8] = {
    {3, 4, 4, 0, 1, 3}, // mode 0
    {2, 6, 6, 0, 2, 3}, // mode 1
    {3, 6, 5, 0, 0, 2}, // mode 2
    {2, 6, 7, 0, 1, 2}, // mode 3
    {},                 // 4..6 are decoded separately
    {},
    {},
    {2, 6, 5, 5, 1, 2}, // mode 7
};

inline uint32_t PartitionSubset(uint32_t subsets, uint32_t partition, uint32_t pixel) {
#ifdef KYTY_BC7_MUTANT_SUBSET
	(void)subsets; (void)partition; (void)pixel;
	return 0;
#else
	return ((subsets == 2 ? kPartition2[partition] : kPartition3[partition]) >> (2 * pixel)) & 3u;
#endif
}

inline bool IsAnchor(uint32_t subsets, uint32_t partition, uint32_t subset, uint32_t pixel) {
	if (subset == 0) return pixel == 0;
	if (subsets == 2) return pixel == kAnchor2[partition];
	return pixel == (subset == 1 ? kAnchor3a[partition] : kAnchor3b[partition]);
}

inline void DecodePartitioned(Bits& bits, uint32_t mode, Bc7Block& out) {
	const PartMode& m         = kPartModes[mode];
	const uint32_t  partition = bits.Read(m.partition_bits);
	const uint32_t  endpoints = m.subsets * 2u;
	const uint32_t  channels  = m.alpha_bits != 0 ? 4u : 3u;
	uint32_t        e[6][4] {};
	for (uint32_t c = 0; c < channels; c++) {
		for (uint32_t i = 0; i < endpoints; i++) e[i][c] = bits.Read(c == 3 ? m.alpha_bits : m.color_bits);
	}
	uint32_t pbit[6] {};
	if (m.pbits_kind == 1) {
		for (uint32_t i = 0; i < endpoints; i++) pbit[i] = bits.Read(1);
	} else if (m.pbits_kind == 2) {
		for (uint32_t s = 0; s < m.subsets; s++) pbit[2 * s] = pbit[2 * s + 1] = bits.Read(1);
#ifdef KYTY_BC7_MUTANT_SHARED_PBIT
		pbit[1] = 0; // mode 1: the second endpoint of a subset loses the shared p-bit
		pbit[3] = 0;
#endif
	}
	for (uint32_t i = 0; i < endpoints; i++) {
		for (uint32_t c = 0; c < channels; c++) {
			uint32_t value = e[i][c];
			uint32_t width = c == 3 ? m.alpha_bits : m.color_bits;
			if (m.pbits_kind != 0) {
				value = (value << 1) | pbit[i];
				width++;
			}
			e[i][c] = Expand(value, width);
		}
		if (channels == 3) e[i][3] = 255;
	}
	uint32_t index[16];
	for (uint32_t p = 0; p < 16; p++) {
		const uint32_t subset = PartitionSubset(m.subsets, partition, p);
#ifdef KYTY_BC7_MUTANT_ANCHOR
		index[p] = bits.Read(m.index_bits);
		(void)subset;
#else
		index[p] = bits.Read(IsAnchor(m.subsets, partition, subset, p) ? m.index_bits - 1u : m.index_bits);
#endif
	}
	for (uint32_t p = 0; p < 16; p++) {
		const uint32_t subset = PartitionSubset(m.subsets, partition, p);
		const uint32_t w      = Weight(m.index_bits, index[p]);
		for (uint32_t c = 0; c < 4; c++) out.texels[p][c] = Lerp(e[2 * subset][c], e[2 * subset + 1][c], w);
	}
	out.ok = true;
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
	} else {
		DecodePartitioned(bits, static_cast<uint32_t>(mode), out);
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
