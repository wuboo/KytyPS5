#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_DIAGTEX_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_DIAGTEX_H_

// S122 diag_tex: pure helpers (no Vulkan) for describing a sampled T# and dumping the host image it reads.

#include "graphics/host_gpu/renderer/rtDump.h"

#include <cinttypes>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace Libs::Graphics::DiagDump {

// Everything the log line shows about one sampled texture. `dwords` is the raw T# (8 dwords).
struct TexInfo {
	uint64_t flip = 0, ps = 0;
	uint32_t slot = 0;
	uint64_t addr = 0;
	uint32_t guest_format = 0, numeric_class = 0, vk_format = 0;
	uint32_t width = 0, height = 0, depth = 0, base_level = 0, last_level = 0, max_mip = 0;
	uint32_t tile_mode = 0, pitch = 0, type = 0, dimension = 0, dst_sel = 0; // dst_sel: x | y<<3 | z<<6 | w<<9
	uint32_t image_levels = 0, image_layers = 0;
	uint64_t byte_size = 0;
	uint64_t host_image = 0; // address of the host VkImage, 0 = not backed
	bool     from_cache = false, gpu_modified = false, cpu_dirty = false;
	const char* uma = "0";
	uint32_t dwords[8] {};
};

inline std::string TexDescribe(const TexInfo& t) {
	char buf[640];
	std::snprintf(buf, sizeof(buf),
	              "[diag-tex] flip=%" PRIu64 " ps=%016" PRIx64 " slot=%u addr=0x%" PRIx64 " fmt=%u numeric=%u vkfmt=%u %ux%ux%u "
	              "mips=%u..%u/%u tile=%u pitch=%u type=%u dim=%u dst_sel=0x%03x size=%" PRIu64 " texcache=%d host_image=0x%" PRIx64
	              " writer=%s uma=%s dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
	              t.flip, t.ps, t.slot, t.addr, t.guest_format, t.numeric_class, t.vk_format, t.width, t.height, t.depth,
	              t.base_level, t.last_level, t.max_mip, t.tile_mode, t.pitch, t.type, t.dimension, t.dst_sel, t.byte_size,
	              t.from_cache ? 1 : 0, t.host_image,
	              // The tracker knows only "a GPU pass wrote it since the last upload" / "the CPU wrote it"; the writing
	              // pass (seq/hash) is not recorded.
	              t.gpu_modified ? "gpu" : (t.cpu_dirty ? "cpu-dirty" : "upload-or-none"), t.uma, t.dwords[0], t.dwords[1],
	              t.dwords[2], t.dwords[3], t.dwords[4], t.dwords[5], t.dwords[6], t.dwords[7]);
	return buf;
}

// Fraction of texels whose every decoded channel is exactly zero.
inline double TexZeroFraction(const std::vector<float>& values, uint32_t channels) {
	if (channels == 0 || values.size() < channels) return 0.0;
	uint64_t zeros = 0, total = 0;
	for (size_t i = 0; i + channels <= values.size(); i += channels) {
		bool zero = true;
		for (uint32_t c = 0; c < channels; c++) zero = zero && values[i + c] == 0.0f;
#ifdef KYTY_DIAGTEX_MUTANT_NONZERO
		zeros += zero ? 0 : 1;
#else
		zeros += zero ? 1 : 0;
#endif
		total++;
	}
	return static_cast<double>(zeros) / static_cast<double>(total);
}

// "[diag-tex] stats ... r=min/max/mean ... zero=<percent of all-zero texels>".
inline std::string TexStats(const RtLayout& layout, uint32_t width, uint32_t height, const std::vector<float>& values) {
	const auto stats = RtComputeStats(values, layout.channels);
	char       buf[64];
	std::string line = RtDescribe("stats", layout, width, height, stats);
	line.replace(0, 9, "[diag-tex]"); // "[dump-rt]" -> "[diag-tex]"
	std::snprintf(buf, sizeof(buf), " zero=%.4f%%", TexZeroFraction(values, layout.channels) * 100.0);
	return line + buf;
}

// diag_tex fires once per (snapshot generation, ps, slot): once per control-file block.
struct TexOnceState {
	std::set<std::tuple<uint64_t, uint64_t, uint32_t>> fired;
};

inline bool TexFirst(TexOnceState& state, uint64_t gen, uint64_t ps, uint32_t slot) {
#ifdef KYTY_DIAGTEX_MUTANT_NO_ONCE
	(void)state; (void)gen; (void)ps; (void)slot;
	return true;
#else
	return state.fired.emplace(gen, ps, slot).second;
#endif
}

// <dir>/<label>-<flip>-tex-<ps>-s<slot>.<ext>
inline std::string TexDumpPath(const std::string& dir, const std::string& label, uint64_t flip, uint64_t ps, uint32_t slot,
                               const char* ext) {
	char buf[64];
	std::snprintf(buf, sizeof(buf), "-%" PRIu64 "-tex-%016" PRIx64 "-s%u.", flip, ps, slot);
	return dir + "/" + (label.empty() ? std::string("diag_tex") : label) + buf + ext;
}

} // namespace Libs::Graphics::DiagDump

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_DIAGTEX_H_ */
