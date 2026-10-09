#include "graphics/host_gpu/renderer/diagTex.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using namespace Libs::Graphics::DiagDump;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DiagTexTests: failed: %s\n", message);
		std::exit(1);
	}
}

bool Has(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

// S122: the log line carries the whole T#.
void TestDescribe() {
	TexInfo t;
	t.flip = 6620; t.ps = 0xb96c2898f05637dfull; t.slot = 30; t.addr = 0x7f0000100000ull;
	t.guest_format = 10; t.numeric_class = 1; t.vk_format = 37; t.width = 512; t.height = 256; t.depth = 1;
	t.base_level = 0; t.last_level = 9; t.max_mip = 9; t.tile_mode = 13; t.pitch = 512; t.type = 9; t.dimension = 1;
	t.dst_sel = 0x688; t.byte_size = 699050; t.host_image = 0xdead0000ull; t.from_cache = true; t.gpu_modified = true;
	t.uma = "3";
	t.min_lod = 1100;
	for (uint32_t i = 0; i < 8; i++) t.dwords[i] = 0x11111111u * (i + 1);
	const auto line = TexDescribe(t);
	Check(line.rfind("[diag-tex] flip=6620 ps=b96c2898f05637df slot=30 addr=0x7f0000100000 ", 0) == 0, "head wrong");
	Check(Has(line, "fmt=10 numeric=1 vkfmt=37 512x256x1 mips=0..9/9 tile=13 pitch=512 type=9 dim=1 dst_sel=0x688"), "format part wrong");
	Check(Has(line, "size=699050 texcache=1 host_image=0xdead0000 writer=gpu uma=3"), "cache part wrong");
	Check(Has(line, "dst_sel=0x688 minlod=1100 size=699050"), "minlod wrong");
	Check(Has(line, "dwords=11111111,22222222,33333333,44444444,55555555,66666666,77777777,88888888"), "dwords wrong");
	t.gpu_modified = false;
	t.cpu_dirty    = true;
	Check(Has(TexDescribe(t), "writer=cpu-dirty"), "cpu writer wrong");
}

// All-zero texels are counted over whole texels, not channels; a 2x2 RGBA8 image with one zero texel -> 25%.
void TestZeroFraction() {
	const uint8_t     bytes[16] = {0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 255, 10, 20, 30, 40};
	const auto        layout    = RtLayoutFor(37);
	const auto        values    = RtDecode(layout, bytes, 4);
	Check(std::fabs(TexZeroFraction(values, 4) - 0.25) < 1e-9, "zero fraction of 1 of 4 texels");
	const auto stats = TexStats(layout, 2, 2, values);
	Check(stats.rfind("[diag-tex] stats", 0) == 0, "stats prefix");
	Check(Has(stats, "2x2") && Has(stats, " r=0/1/") && Has(stats, "zero=25.0000%"), "stats content");
	const uint8_t zero[8] = {};
	Check(std::fabs(TexZeroFraction(RtDecode(layout, zero, 2), 4) - 1.0) < 1e-9, "all zero texels");
}

// once per (generation, ps, slot).
void TestOnce() {
	TexOnceState once;
	Check(TexFirst(once, 3, 0xaa, 30), "first request must fire");
	Check(!TexFirst(once, 3, 0xaa, 30), "second request in the same block must not fire");
	Check(TexFirst(once, 3, 0xaa, 31), "another slot fires");
	Check(TexFirst(once, 3, 0xbb, 30), "another shader fires");
	Check(TexFirst(once, 4, 0xaa, 30), "a new block (generation) fires again");
}

void TestPath() {
	Check(TexDumpPath("/tmp/d", "armA", 6620, 0xabc, 30, "ppm") == "/tmp/d/armA-6620-tex-0000000000000abc-s30.ppm", "path");
	Check(TexDumpPath("/tmp/d", "", 1, 0xabc, 2, "raw") == "/tmp/d/diag_tex-1-tex-0000000000000abc-s2.raw", "path without label");
}

// S123: the per-draw line carries seq, slot, address, format, size, mips and the draw's index count and target.
void TestDrawLine() {
	TexInfo t;
	t.flip = 6760; t.ps = 0xb96c2898f05637dfull; t.slot = 30; t.addr = 0x70d1770000ull; t.guest_format = 181; t.width = 4; t.height = 4;
	t.base_level = 0; t.last_level = 0; t.max_mip = 0; t.byte_size = 16;
	const auto line = TexDrawDescribe(7, t, 3600, 0x40000000ull, nullptr);
	Check(line.rfind("[diag-tex-d] flip=6760 seq=7 ps=b96c2898f05637df slot=30 addr=0x70d1770000 fmt=181 4x4 mips=0..0/0 minlod=0 size=16", 0) == 0,
	      "draw line head");
	Check(Has(line, "indices=3600 rt=0x40000000") && !Has(line, "bc7"), "draw line tail");
	Bc7Summary bc7;
	bc7.first_mode = 6; bc7.first_ok = true; bc7.first_texel[0] = 0x78; bc7.first_texel[3] = 0xc3; bc7.blocks = 1; bc7.decoded = 1;
	bc7.mean[0] = 246.5; bc7.mean[3] = 130.0;
	Check(Has(TexDrawDescribe(8, t, 1, 0, &bc7), " bc7 mode=6 first=780000c3 mean=246.5,0.0,0.0,130.0 blocks=1/1"), "draw line bc7 part");
}

// S123: histogram per (ps, slot, addr, fmt, size): draw count, index sum and share of the slot's draws.
void TestHistogram() {
	TexHist hist;
	TexInfo a;
	a.flip = 9; a.ps = 0xaa; a.slot = 30; a.addr = 0x70d1770000ull; a.guest_format = 181; a.width = 4; a.height = 4;
	TexInfo b = a;
	b.addr = 0x5000; b.guest_format = 10; b.width = 1; b.height = 1;
	TexInfo c = a; // same address and size as `a`, other format: its own row
	c.guest_format = 182;
	for (int i = 0; i < 3; i++) TexHistAdd(hist, a, 100, nullptr);
	TexHistAdd(hist, b, 7, nullptr);
	TexHistAdd(hist, c, 1, nullptr);
	TexInfo other = a; // another slot: separate total
	other.slot = 4;
	TexHistAdd(hist, other, 5, nullptr);
	const auto lines = TexHistLines(hist, 9);
	Check(lines.size() == 4, "histogram rows");
	Check(lines[0].rfind("[diag-tex-h] flip=9 ps=00000000000000aa slot=30 addr=0x70d1770000 fmt=181 4x4 draws=3 (60.0% of 5) indices=300", 0) == 0,
	      "histogram top row");
	Check(Has(lines[1], "draws=1 (") && Has(lines[2], "draws=1 (") && Has(lines[3], "draws=1 ("), "histogram other rows");
	bool seen_b = false, seen_c = false, seen_other = false;
	for (const auto& l: lines) {
		seen_b |= Has(l, "addr=0x5000 fmt=10 1x1 draws=1 (20.0% of 5) indices=7");
		seen_c |= Has(l, "addr=0x70d1770000 fmt=182 4x4 draws=1 (20.0% of 5) indices=1");
		seen_other |= Has(l, "slot=4 ") && Has(l, "(100.0% of 1) indices=5");
	}
	Check(seen_b && seen_c && seen_other, "histogram keys (format / slot / indices)");
	Bc7Summary bc7;
	bc7.first_mode = 6; bc7.blocks = bc7.decoded = 1; bc7.mean[0] = 10;
	TexHist h2;
	TexHistAdd(h2, a, 1, &bc7);
	Check(Has(TexHistLines(h2, 1)[0], " bc7 mode=6 first=00000000 mean=10.0,0.0,0.0,0.0"), "histogram bc7 part");
}

} // namespace

int main() {
	TestDescribe();
	TestZeroFraction();
	TestOnce();
	TestPath();
	TestDrawLine();
	TestHistogram();
	return 0;
}
