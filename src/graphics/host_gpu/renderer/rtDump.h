#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RTDUMP_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RTDUMP_H_

// S94 dump_rt: pure helpers (no Vulkan) for dumping the render targets a pixel shader wrote.
// Formats are named by their VkFormat number so the CPU test needs no Vulkan headers.

#include "common/controlFile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics::DiagDump {

enum class RtKind : uint8_t {
	Unknown,
	Unorm8,
	Snorm8,
	Unorm16,
	Snorm16,
	Float16,
	Float32,
	A2B10G10R10, // VK_FORMAT_A2B10G10R10_UNORM_PACK32: r in the low bits
	A2R10G10B10,
	B10G11R11, // unsigned float, r in the low bits
	Depth16,
	Depth24, // X8_D24: 24 bit depth in the low bits
	Depth32,
};

struct RtLayout {
	RtKind   kind     = RtKind::Unknown;
	uint32_t channels = 0; // decoded channels per texel (R, RG, RGB, RGBA or 1 for depth)
	bool     bgr      = false; // stored B,G,R(,A): swap to R,G,B(,A) when decoding
	uint32_t bytes    = 0;     // bytes per texel in the readback buffer
	// UNORM is shown as stored (value * 255); everything else is min-max normalised.
	[[nodiscard]] bool Direct() const { return kind == RtKind::Unorm8 || kind == RtKind::Unorm16 ||
	                                           kind == RtKind::A2B10G10R10 || kind == RtKind::A2R10G10B10; }
	[[nodiscard]] bool Depth() const { return kind == RtKind::Depth16 || kind == RtKind::Depth24 || kind == RtKind::Depth32; }
};

// `vk_format` is the numeric VkFormat of the copy (the depth aspect copy format for depth targets).
inline RtLayout RtLayoutFor(int vk_format) {
	switch (vk_format) {
		case 9: return {RtKind::Unorm8, 1, false, 1};     // R8_UNORM
		case 10: return {RtKind::Snorm8, 1, false, 1};    // R8_SNORM
		case 16: return {RtKind::Unorm8, 2, false, 2};    // R8G8_UNORM
		case 17: return {RtKind::Snorm8, 2, false, 2};    // R8G8_SNORM
		case 37:
		case 43: return {RtKind::Unorm8, 4, false, 4};    // R8G8B8A8_UNORM / _SRGB
		case 38: return {RtKind::Snorm8, 4, false, 4};    // R8G8B8A8_SNORM
		case 44:
		case 50: return {RtKind::Unorm8, 4, true, 4};     // B8G8R8A8_UNORM / _SRGB
		case 58: return {RtKind::A2R10G10B10, 4, false, 4};
		case 64: return {RtKind::A2B10G10R10, 4, false, 4};
		case 70: return {RtKind::Unorm16, 1, false, 2};   // R16_UNORM
		case 71: return {RtKind::Snorm16, 1, false, 2};   // R16_SNORM
		case 76: return {RtKind::Float16, 1, false, 2};   // R16_SFLOAT
		case 77: return {RtKind::Unorm16, 2, false, 4};   // R16G16_UNORM
		case 78: return {RtKind::Snorm16, 2, false, 4};   // R16G16_SNORM
		case 83: return {RtKind::Float16, 2, false, 4};   // R16G16_SFLOAT
		case 91: return {RtKind::Unorm16, 4, false, 8};   // R16G16B16A16_UNORM
		case 92: return {RtKind::Snorm16, 4, false, 8};   // R16G16B16A16_SNORM
		case 97: return {RtKind::Float16, 4, false, 8};   // R16G16B16A16_SFLOAT
		case 100: return {RtKind::Float32, 1, false, 4};  // R32_SFLOAT
		case 103: return {RtKind::Float32, 2, false, 8};  // R32G32_SFLOAT
		case 109: return {RtKind::Float32, 4, false, 16}; // R32G32B32A32_SFLOAT
		case 122: return {RtKind::B10G11R11, 3, false, 4};
		case 124: return {RtKind::Depth16, 1, false, 2};  // D16_UNORM
		case 125: return {RtKind::Depth24, 1, false, 4};  // X8_D24_UNORM_PACK32
		case 126: return {RtKind::Depth32, 1, false, 4};  // D32_SFLOAT
		default: return {};
	}
}

inline const char* RtKindName(RtKind kind) {
	switch (kind) {
		case RtKind::Unorm8: return "UNORM8";
		case RtKind::Snorm8: return "SNORM8";
		case RtKind::Unorm16: return "UNORM16";
		case RtKind::Snorm16: return "SNORM16";
		case RtKind::Float16: return "FLOAT16";
		case RtKind::Float32: return "FLOAT32";
		case RtKind::A2B10G10R10: return "A2B10G10R10";
		case RtKind::A2R10G10B10: return "A2R10G10B10";
		case RtKind::B10G11R11: return "B10G11R11";
		case RtKind::Depth16: return "D16";
		case RtKind::Depth24: return "D24";
		case RtKind::Depth32: return "D32F";
		default: return "UNKNOWN";
	}
}

inline float RtHalfToFloat(uint16_t h) {
	const uint32_t sign = (h >> 15u) & 1u, exp = (h >> 10u) & 0x1fu, man = h & 0x3ffu;
	float          value;
	if (exp == 0) {
		value = std::ldexp(static_cast<float>(man), -24);
	} else if (exp == 31) {
		value = man != 0 ? std::nanf("") : INFINITY;
	} else {
		value = std::ldexp(static_cast<float>(man | 0x400u), static_cast<int>(exp) - 25);
	}
	return sign != 0 ? -value : value;
}

// Unsigned small float with `mbits` mantissa bits (B10G11R11: 6 for r/g, 5 for b), 5 exponent bits.
inline float RtSmallUfloat(uint32_t bits, uint32_t mbits) {
	const uint32_t man = bits & ((1u << mbits) - 1u), exp = bits >> mbits;
	if (exp == 0) return std::ldexp(static_cast<float>(man), -14 - static_cast<int>(mbits));
	if (exp == 31) return man != 0 ? std::nanf("") : INFINITY;
	return std::ldexp(static_cast<float>(man | (1u << mbits)), static_cast<int>(exp) - 15 - static_cast<int>(mbits));
}

// Decodes `count` texels to `layout.channels` floats each, in R,G,B,A order. UNORM values are in
// [0,1], SNORM in [-1,1], depth in [0,1] (D16/D24) or as stored (D32F).
inline std::vector<float> RtDecode(const RtLayout& layout, const uint8_t* data, size_t count) {
	std::vector<float> out(count * layout.channels);
	for (size_t i = 0; i < count; i++) {
		const uint8_t* p = data + i * layout.bytes;
		float*         o = &out[i * layout.channels];
		switch (layout.kind) {
			case RtKind::Unorm8:
				for (uint32_t c = 0; c < layout.channels; c++) o[c] = static_cast<float>(p[c]) / 255.0f;
				break;
			case RtKind::Snorm8:
				for (uint32_t c = 0; c < layout.channels; c++)
					o[c] = std::max(static_cast<float>(static_cast<int8_t>(p[c])) / 127.0f, -1.0f);
				break;
			case RtKind::Unorm16:
			case RtKind::Snorm16:
			case RtKind::Float16:
			case RtKind::Depth16:
				for (uint32_t c = 0; c < layout.channels; c++) {
					uint16_t v;
					std::memcpy(&v, p + c * 2, 2);
					o[c] = layout.kind == RtKind::Unorm16 || layout.kind == RtKind::Depth16 ? static_cast<float>(v) / 65535.0f
					       : layout.kind == RtKind::Snorm16 ? std::max(static_cast<float>(static_cast<int16_t>(v)) / 32767.0f, -1.0f)
					                                        : RtHalfToFloat(v);
				}
				break;
			case RtKind::Float32:
			case RtKind::Depth32:
				for (uint32_t c = 0; c < layout.channels; c++) std::memcpy(&o[c], p + c * 4, 4);
				break;
			case RtKind::Depth24: {
				uint32_t v;
				std::memcpy(&v, p, 4);
				o[0] = static_cast<float>(v & 0xffffffu) / 16777215.0f;
				break;
			}
			case RtKind::A2B10G10R10:
			case RtKind::A2R10G10B10: {
				uint32_t v;
				std::memcpy(&v, p, 4);
				const float a = static_cast<float>(v & 0x3ffu) / 1023.0f, g = static_cast<float>((v >> 10u) & 0x3ffu) / 1023.0f,
				            b = static_cast<float>((v >> 20u) & 0x3ffu) / 1023.0f, al = static_cast<float>(v >> 30u) / 3.0f;
				o[0] = layout.kind == RtKind::A2B10G10R10 ? a : b;
				o[1] = g;
				o[2] = layout.kind == RtKind::A2B10G10R10 ? b : a;
				o[3] = al;
				break;
			}
			case RtKind::B10G11R11: {
				uint32_t v;
				std::memcpy(&v, p, 4);
				o[0] = RtSmallUfloat(v & 0x7ffu, 6);
				o[1] = RtSmallUfloat((v >> 11u) & 0x7ffu, 6);
				o[2] = RtSmallUfloat((v >> 22u) & 0x3ffu, 5);
				break;
			}
			default: break;
		}
		if (layout.bgr && layout.channels >= 3) std::swap(o[0], o[2]);
	}
	return out;
}

struct RtStats {
	uint32_t channels = 0;
	float    min[4]   = {0, 0, 0, 0};
	float    max[4]   = {0, 0, 0, 0};
	double   mean[4]  = {0, 0, 0, 0};
	uint64_t nonfinite = 0; // texels-channels that were NaN/Inf (excluded from the stats)
};

inline RtStats RtComputeStats(const std::vector<float>& values, uint32_t channels) {
	RtStats stats;
	stats.channels = std::min(channels, 4u);
	if (channels == 0 || values.empty()) return stats;
	uint64_t count[4] = {0, 0, 0, 0};
	for (uint32_t c = 0; c < stats.channels; c++) {
		stats.min[c] = INFINITY;
		stats.max[c] = -INFINITY;
	}
	for (size_t i = 0; i + channels <= values.size(); i += channels) {
		for (uint32_t c = 0; c < stats.channels; c++) {
			const float v = values[i + c];
			if (!std::isfinite(v)) {
				stats.nonfinite++;
				continue;
			}
			stats.min[c] = std::min(stats.min[c], v);
			stats.max[c] = std::max(stats.max[c], v);
			stats.mean[c] += static_cast<double>(v);
			count[c]++;
		}
	}
	for (uint32_t c = 0; c < stats.channels; c++) {
		if (count[c] == 0) {
			stats.min[c] = stats.max[c] = 0;
		} else {
			stats.mean[c] /= static_cast<double>(count[c]);
		}
	}
	return stats;
}

// Range used to map a non-UNORM target to 8 bit: min and max over the (up to three) colour
// channels shown, or the one depth channel. Returns false when the range is degenerate (max<=min).
inline bool RtDisplayRange(const RtLayout& layout, const RtStats& stats, float& lo, float& hi) {
	const uint32_t shown = std::min(stats.channels, 3u);
	lo = INFINITY;
	hi = -INFINITY;
	for (uint32_t c = 0; c < shown; c++) {
		lo = std::min(lo, stats.min[c]);
		hi = std::max(hi, stats.max[c]);
	}
	(void)layout;
	return shown != 0 && hi > lo;
}

// RGB8 image from decoded values: UNORM as stored, others min-max over [lo,hi]; one- and
// two-channel targets show channel 0 as grey and channel 1 as green resp. grey. Non-finite = 0.
inline std::vector<uint8_t> RtToRgb8(const RtLayout& layout, const std::vector<float>& values, const RtStats& stats) {
	const size_t count = layout.channels != 0 ? values.size() / layout.channels : 0;
	std::vector<uint8_t> rgb(count * 3, 0);
	float lo = 0, hi = 1;
	const bool direct = layout.Direct();
	const bool ranged = !direct && RtDisplayRange(layout, stats, lo, hi);
	const auto map    = [&](float v) -> uint8_t {
        if (!std::isfinite(v)) return 0;
        float n = direct ? v : ranged ? (v - lo) / (hi - lo) : 0.0f;
        n       = std::min(std::max(n, 0.0f), 1.0f);
        return static_cast<uint8_t>(n * 255.0f + 0.5f);
    };
	for (size_t i = 0; i < count; i++) {
		const float* v = &values[i * layout.channels];
		const float r = v[0], g = layout.channels >= 2 ? v[1] : v[0],
		            b = layout.channels >= 3 ? v[2] : (layout.channels == 2 ? 0.0f : v[0]);
		rgb[i * 3]     = map(r);
		rgb[i * 3 + 1] = map(g);
		rgb[i * 3 + 2] = map(b);
	}
	return rgb;
}

inline std::string RtPpm(const std::vector<uint8_t>& rgb, uint32_t width, uint32_t height) {
	std::string out = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
	out.append(reinterpret_cast<const char*>(rgb.data()), rgb.size());
	return out;
}

// "[dump-rt] mrt=N fmt=... w×h min/max/mean per channel", plus the display range for mapped targets.
inline std::string RtDescribe(const char* what, const RtLayout& layout, uint32_t width, uint32_t height, const RtStats& stats) {
	char        buf[160];
	std::string line = std::string("[dump-rt] ") + what + " fmt=" + RtKindName(layout.kind) + " " + std::to_string(width) +
	                   "x" + std::to_string(height);
	static const char* names[4] = {"r", "g", "b", "a"};
	for (uint32_t c = 0; c < stats.channels; c++) {
		std::snprintf(buf, sizeof(buf), " %s=%.6g/%.6g/%.6g", layout.Depth() ? "d" : names[c], static_cast<double>(stats.min[c]),
		              static_cast<double>(stats.max[c]), stats.mean[c]);
		line += buf;
	}
	if (stats.nonfinite != 0) line += " nonfinite=" + std::to_string(stats.nonfinite);
	if (layout.Direct()) {
		line += " map=unorm";
	} else {
		float lo, hi;
		const bool ok = RtDisplayRange(layout, stats, lo, hi);
		std::snprintf(buf, sizeof(buf), " map=minmax[%.6g,%.6g]%s", static_cast<double>(ok ? lo : 0), static_cast<double>(ok ? hi : 0),
		              ok ? "" : " (flat)");
		line += buf;
	}
	return line;
}

// State of the "no flip given" entries: they fire in the first flip that draws the PS after the
// snapshot generation changed, and only in that flip.
struct RtOnceState {
	uint64_t gen  = ~0ull;
	uint64_t flip = 0;
};

inline bool DumpRtMatch(const std::vector<Common::ControlFile::DumpRt>& list, uint64_t ps, uint64_t flip, uint64_t gen,
                        RtOnceState& once) {
	if (ps == 0) return false;
	for (const auto& entry: list) {
#ifndef KYTY_RTDUMP_MUTANT_IGNORE_PS
		if (entry.ps != ps) continue;
#endif
		if (entry.flip != 0) {
#ifdef KYTY_RTDUMP_MUTANT_IGNORE_FLIP
			return true;
#endif
			if (entry.flip == flip) return true;
			continue;
		}
		if (once.gen != gen) {
			once.gen  = gen;
			once.flip = flip;
		}
		if (once.flip == flip) return true;
	}
	return false;
}

// <dir>/<label>-<flip>-mrt<N>.ppm (label empty: dump_rt-<flip>-...), N = depth for the depth target.
inline std::string RtDumpPath(const std::string& dir, const std::string& label, uint64_t flip, int mrt) {
	return dir + "/" + (label.empty() ? std::string("dump_rt") : label) + "-" + std::to_string(flip) +
	       (mrt < 0 ? std::string("-depth") : "-mrt" + std::to_string(mrt)) + ".ppm";
}

} // namespace Libs::Graphics::DiagDump

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_RTDUMP_H_ */
