#include "graphics/host_gpu/renderer/rtDump.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using namespace Libs::Graphics::DiagDump;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "RtDumpTests: failed: %s\n", message);
		std::exit(1);
	}
}

bool Near(float a, float b, float eps = 1e-5f) {
	return std::fabs(a - b) <= eps;
}

// S94: VkFormat number -> layout.
void TestLayouts() {
	Check(RtLayoutFor(37).kind == RtKind::Unorm8 && RtLayoutFor(37).channels == 4 && RtLayoutFor(37).bytes == 4 && !RtLayoutFor(37).bgr,
	      "R8G8B8A8_UNORM");
	Check(RtLayoutFor(44).bgr && RtLayoutFor(44).kind == RtKind::Unorm8, "B8G8R8A8_UNORM is BGR");
	Check(RtLayoutFor(38).kind == RtKind::Snorm8, "R8G8B8A8_SNORM is SNORM");
	Check(RtLayoutFor(97).kind == RtKind::Float16 && RtLayoutFor(97).bytes == 8, "R16G16B16A16_SFLOAT: 8 bytes");
	Check(RtLayoutFor(109).kind == RtKind::Float32 && RtLayoutFor(109).bytes == 16, "R32G32B32A32_SFLOAT: 16 bytes");
	Check(RtLayoutFor(122).kind == RtKind::B10G11R11 && RtLayoutFor(122).channels == 3, "B10G11R11");
	Check(RtLayoutFor(126).Depth() && RtLayoutFor(126).bytes == 4, "D32_SFLOAT");
	Check(RtLayoutFor(124).Depth() && RtLayoutFor(124).bytes == 2, "D16_UNORM");
	Check(RtLayoutFor(125).Depth() && RtLayoutFor(125).bytes == 4, "X8_D24");
	Check(RtLayoutFor(0).kind == RtKind::Unknown && RtLayoutFor(0).bytes == 0, "undefined format is unknown");
	Check(RtLayoutFor(37).Direct() && !RtLayoutFor(97).Direct() && !RtLayoutFor(38).Direct(), "UNORM is shown as stored, FLOAT/SNORM are mapped");
}

void TestDecode() {
	// BGRA8: stored B,G,R,A = 10,20,30,255 -> r=30 g=20 b=10
	const uint8_t bgra[4] = {10, 20, 30, 255};
	auto          v       = RtDecode(RtLayoutFor(44), bgra, 1);
	Check(Near(v[0], 30 / 255.0f) && Near(v[1], 20 / 255.0f) && Near(v[2], 10 / 255.0f) && Near(v[3], 1.0f), "BGRA swapped to RGBA");
	// SNORM8: -128 clamps to -1
	const uint8_t sn[4] = {0x80, 0x7f, 0x00, 0xc1};
	v                   = RtDecode(RtLayoutFor(38), sn, 1);
	Check(Near(v[0], -1.0f) && Near(v[1], 1.0f) && Near(v[2], 0.0f) && Near(v[3], -63 / 127.0f), "SNORM8 decode and clamp");
	// half: 1.0=0x3c00, -2.0=0xc000, 0.5=0x3800, inf=0x7c00
	const uint16_t half[4] = {0x3c00, 0xc000, 0x3800, 0x7c00};
	v                      = RtDecode(RtLayoutFor(97), reinterpret_cast<const uint8_t*>(half), 1);
	Check(Near(v[0], 1.0f) && Near(v[1], -2.0f) && Near(v[2], 0.5f) && std::isinf(v[3]), "FLOAT16 decode");
	Check(Near(RtHalfToFloat(0x0001), std::ldexp(1.0f, -24), 1e-12f), "smallest half denormal");
	// float32
	const float f[4] = {1.5f, -0.25f, 100.0f, 0.0f};
	v                = RtDecode(RtLayoutFor(109), reinterpret_cast<const uint8_t*>(f), 1);
	Check(Near(v[0], 1.5f) && Near(v[1], -0.25f) && Near(v[2], 100.0f), "FLOAT32 decode");
	// A2B10G10R10: r=1023, g=0, b=511, a=3
	const uint32_t packed = 1023u | (0u << 10u) | (511u << 20u) | (3u << 30u);
	v                     = RtDecode(RtLayoutFor(64), reinterpret_cast<const uint8_t*>(&packed), 1);
	Check(Near(v[0], 1.0f) && Near(v[1], 0.0f) && Near(v[2], 511 / 1023.0f) && Near(v[3], 1.0f), "A2B10G10R10 decode");
	// A2R10G10B10 swaps r and b
	v = RtDecode(RtLayoutFor(58), reinterpret_cast<const uint8_t*>(&packed), 1);
	Check(Near(v[2], 1.0f) && Near(v[0], 511 / 1023.0f), "A2R10G10B10 has r in the high field");
	// B10G11R11: r=1.0 (exp 15, man 0 = 0x3c0), g=2.0 (exp 16 = 0x400), b=0.5 (exp 14 = 0x1c0 in 10 bits... 5-bit man)
	const uint32_t b10 = 0x3c0u | (0x400u << 11u) | (0x1c0u << 22u);
	v                  = RtDecode(RtLayoutFor(122), reinterpret_cast<const uint8_t*>(&b10), 1);
	Check(Near(v[0], 1.0f) && Near(v[1], 2.0f) && Near(v[2], 0.5f), "B10G11R11 decode");
	// depth
	const uint16_t d16 = 0xffff;
	Check(Near(RtDecode(RtLayoutFor(124), reinterpret_cast<const uint8_t*>(&d16), 1)[0], 1.0f), "D16 max is 1");
	const uint32_t d24 = 0xff800000u | 0x800000u; // high byte ignored: 0x800000 of 0xffffff
	Check(Near(RtDecode(RtLayoutFor(125), reinterpret_cast<const uint8_t*>(&d24), 1)[0], 8388608.0f / 16777215.0f), "D24 ignores the high byte");
}

void TestStatsAndRange() {
	const std::vector<float> v = {0.0f, 1.0f, 2.0f, 0.5f, -1.0f, 4.0f, INFINITY, 3.0f, 0.0f};
	const auto               s = RtComputeStats(v, 3);
	Check(s.channels == 3, "channel count");
	Check(Near(s.min[0], 0.0f) && Near(s.max[0], 0.5f), "channel 0 min/max skip the inf of the third texel");
	Check(s.nonfinite == 1, "one non-finite counted");
	Check(Near(s.min[1], -1.0f) && Near(s.max[1], 3.0f) && Near(static_cast<float>(s.mean[1]), 1.0f), "channel 1 min/max/mean");
	Check(Near(s.min[2], 0.0f) && Near(s.max[2], 4.0f), "channel 2 min/max");
	float lo, hi;
	Check(RtDisplayRange(RtLayoutFor(97), s, lo, hi) && Near(lo, -1.0f) && Near(hi, 4.0f), "display range over the three colour channels");
	RtStats flat = RtComputeStats({2.0f, 2.0f}, 1);
	Check(!RtDisplayRange(RtLayoutFor(126), flat, lo, hi), "flat image has no range");
}

void TestRgb8() {
	// FLOAT16 texel (-1, 0, 1) over range [-1,1] -> 0, 128, 255 ; min-max is applied
	const std::vector<float> v = {-1.0f, 0.0f, 1.0f, 1.0f};
	const auto               layout = RtLayoutFor(97);
	const auto               rgb    = RtToRgb8(layout, v, RtComputeStats(v, 4));
	Check(rgb.size() == 3 && rgb[0] == 0 && rgb[1] == 128 && rgb[2] == 255, "float min-max mapping to 0/128/255");
	// UNORM is direct: 0.5 -> 128 whatever the range
	const std::vector<float> u    = {0.5f, 0.25f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
	const auto               urgb = RtToRgb8(RtLayoutFor(37), u, RtComputeStats(u, 4));
	Check(urgb.size() == 6 && urgb[0] == 128 && urgb[1] == 64 && urgb[2] == 255 && urgb[3] == 0, "UNORM8 shown as stored, not stretched");
	// non-finite is black, not a crash
	const std::vector<float> bad = {INFINITY, 0.0f, 1.0f, 1.0f};
	const auto               brgb = RtToRgb8(layout, bad, RtComputeStats(bad, 4));
	Check(brgb[0] == 0, "non-finite maps to 0");
	// depth: one channel -> grey
	const std::vector<float> depth = {0.25f, 0.75f};
	const auto               drgb  = RtToRgb8(RtLayoutFor(126), depth, RtComputeStats(depth, 1));
	Check(drgb.size() == 6 && drgb[0] == 0 && drgb[1] == 0 && drgb[2] == 0 && drgb[3] == 255 && drgb[4] == 255 && drgb[5] == 255,
	      "depth min-max grey");
	// flat float target: all zero (no division by zero)
	const std::vector<float> flat = {3.0f, 3.0f, 3.0f, 1.0f};
	const auto               frgb = RtToRgb8(layout, flat, RtComputeStats(flat, 4));
	Check(frgb[0] == 0 && frgb[1] == 0 && frgb[2] == 0, "flat float image maps to black");
	// two channels: b is 0
	const std::vector<float> rg   = {1.0f, 0.5f};
	const auto               rgrgb = RtToRgb8(RtLayoutFor(16), rg, RtComputeStats(rg, 2));
	Check(rgrgb[0] == 255 && rgrgb[1] == 128 && rgrgb[2] == 0, "RG: blue stays 0");
}

void TestPpmAndDescribe() {
	const auto ppm = RtPpm({1, 2, 3, 4, 5, 6}, 2, 1);
	Check(ppm.rfind("P6\n2 1\n255\n", 0) == 0 && ppm.size() == 11 + 6 && static_cast<uint8_t>(ppm[11]) == 1 && static_cast<uint8_t>(ppm[16]) == 6,
	      "PPM header and payload");
	const std::vector<float> v = {-1.0f, 0.0f, 1.0f, 1.0f};
	const auto               line = RtDescribe("mrt=2", RtLayoutFor(97), 1, 1, RtComputeStats(v, 4));
	Check(line.rfind("[dump-rt] mrt=2 fmt=FLOAT16 1x1 r=-1/-1/-1 g=0/0/0 b=1/1/1 a=1/1/1", 0) == 0, "describe: per-channel min/max/mean");
	Check(line.find("map=minmax[-1,1]") != std::string::npos, "describe: range of the mapping is printed");
	const std::vector<float> u = {0.5f, 0.5f, 0.5f, 1.0f};
	Check(RtDescribe("mrt=0", RtLayoutFor(37), 1, 1, RtComputeStats(u, 4)).find("map=unorm") != std::string::npos, "describe: unorm");
	const std::vector<float> d = {0.5f};
	Check(RtDescribe("depth", RtLayoutFor(126), 1, 1, RtComputeStats(d, 1)).find(" d=0.5/0.5/0.5") != std::string::npos, "describe: depth channel is d");
}

void TestMatch() {
	using Common::ControlFile::DumpRt;
	RtOnceState        once;
	std::vector<DumpRt> list = {{0xb96, 5697}};
	Check(DumpRtMatch(list, 0xb96, 5697, 1, once), "ps and flip match");
	Check(!DumpRtMatch(list, 0xb96, 5698, 1, once), "other flip does not match");
	Check(!DumpRtMatch(list, 0xaaa, 5697, 1, once), "other ps does not match");
	Check(!DumpRtMatch(list, 0, 5697, 1, once), "ps 0 never matches");
	std::vector<DumpRt> zero = {{0, 5697}};
	Check(!DumpRtMatch(zero, 0, 5697, 1, once), "ps 0 never matches even when listed");
	// once-per-generation entry
	RtOnceState        once2;
	std::vector<DumpRt> any = {{0xb96, 0}};
	Check(DumpRtMatch(any, 0xb96, 100, 7, once2), "first flip that draws the PS is chosen");
	Check(DumpRtMatch(any, 0xb96, 100, 7, once2), "later draws of the same flip match too");
	Check(!DumpRtMatch(any, 0xb96, 101, 7, once2), "later flips of the same generation do not");
	Check(DumpRtMatch(any, 0xb96, 150, 8, once2), "a new generation arms it again");
	Check(!DumpRtMatch(any, 0xb96, 100, 8, once2), "and fixes the new flip");
	// the once state is only consumed by matching entries
	RtOnceState        once3;
	Check(!DumpRtMatch(any, 0xccc, 50, 1, once3) && once3.gen == ~0ull, "other ps leaves the once state alone");
}

void TestPath() {
	Check(RtDumpPath("/d", "armA", 5697, 2) == "/d/armA-5697-mrt2.ppm", "path with label");
	Check(RtDumpPath("/d", "", 5697, 0) == "/d/dump_rt-5697-mrt0.ppm", "path without label");
	Check(RtDumpPath("/d", "armA", 5697, -1) == "/d/armA-5697-depth.ppm", "depth path");
}

} // namespace

int main() {
	TestLayouts();
	TestDecode();
	TestStatsAndRange();
	TestRgb8();
	TestPpmAndDescribe();
	TestMatch();
	TestPath();
	Check(RtAddrText(0x70d1770000ull) == " rt_addr=0x70d1770000" && RtAddrText(0).empty(), "rt address text");
	std::printf("RtDumpTests: ok\n");
	return 0;
}
