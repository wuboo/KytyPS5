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
	for (uint32_t i = 0; i < 8; i++) t.dwords[i] = 0x11111111u * (i + 1);
	const auto line = TexDescribe(t);
	Check(line.rfind("[diag-tex] flip=6620 ps=b96c2898f05637df slot=30 addr=0x7f0000100000 ", 0) == 0, "head wrong");
	Check(Has(line, "fmt=10 numeric=1 vkfmt=37 512x256x1 mips=0..9/9 tile=13 pitch=512 type=9 dim=1 dst_sel=0x688"), "format part wrong");
	Check(Has(line, "size=699050 texcache=1 host_image=0xdead0000 writer=gpu uma=3"), "cache part wrong");
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

} // namespace

int main() {
	TestDescribe();
	TestZeroFraction();
	TestOnce();
	TestPath();
	return 0;
}
