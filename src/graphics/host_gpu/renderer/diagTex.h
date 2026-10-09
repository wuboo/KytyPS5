#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_DIAGTEX_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_DIAGTEX_H_

// S122 diag_tex: pure helpers (no Vulkan) for describing a sampled T# and dumping the host image it reads.

#include "graphics/host_gpu/renderer/bc7.h"
#include "graphics/host_gpu/renderer/rtDump.h"

#include <algorithm>
#include <cinttypes>
#include <map>
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
	uint32_t min_lod = 0; // T# MinLod, U4.8 absolute
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
	              "mips=%u..%u/%u tile=%u pitch=%u type=%u dim=%u dst_sel=0x%03x minlod=%u size=%" PRIu64 " texcache=%d host_image=0x%" PRIx64
	              " writer=%s uma=%s dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x",
	              t.flip, t.ps, t.slot, t.addr, t.guest_format, t.numeric_class, t.vk_format, t.width, t.height, t.depth,
	              t.base_level, t.last_level, t.max_mip, t.tile_mode, t.pitch, t.type, t.dimension, t.dst_sel, t.min_lod, t.byte_size,
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

// ---- S123: diag_tex=<ps>:<slot>:all ----------------------------------------------------------------------------------

// One line per draw: the T# of the slot, how many indices the draw has and the address of its first render target.
inline std::string TexDrawDescribe(uint64_t seq, const TexInfo& t, uint64_t index_count, uint64_t rt_addr, const Bc7Summary* bc7) {
	char buf[512];
	std::snprintf(buf, sizeof(buf),
	              "[diag-tex-d] flip=%" PRIu64 " seq=%" PRIu64 " ps=%016" PRIx64 " slot=%u addr=0x%" PRIx64
	              " fmt=%u %ux%u mips=%u..%u/%u minlod=%u size=%" PRIu64 " indices=%" PRIu64 " rt=0x%" PRIx64,
	              t.flip, seq, t.ps, t.slot, t.addr, t.guest_format, t.width, t.height, t.base_level, t.last_level, t.max_mip, t.min_lod,
	              t.byte_size, index_count, rt_addr);
	std::string line = buf;
	if (bc7 != nullptr) {
		std::snprintf(buf, sizeof(buf), " bc7 mode=%d first=%02x%02x%02x%02x mean=%.1f,%.1f,%.1f,%.1f blocks=%u/%u", bc7->first_mode,
		              bc7->first_texel[0], bc7->first_texel[1], bc7->first_texel[2], bc7->first_texel[3], bc7->mean[0], bc7->mean[1],
		              bc7->mean[2], bc7->mean[3], bc7->decoded, bc7->blocks);
		line += buf;
	}
	return line;
}

// Draws and index sums per (ps, slot, address, format, size) over one flip.
struct TexHist {
	struct Entry {
		uint64_t   draws = 0, indices = 0;
		Bc7Summary bc7;
		bool       has_bc7 = false;
	};
	using Key = std::tuple<uint64_t, uint32_t, uint64_t, uint32_t, uint32_t, uint32_t>;
	std::map<Key, Entry> entries;
};

inline void TexHistAdd(TexHist& hist, const TexInfo& t, uint64_t index_count, const Bc7Summary* bc7) {
#ifdef KYTY_DIAGTEX_MUTANT_HIST_NO_FMT
	const uint32_t fmt = 0;
#else
	const uint32_t fmt = t.guest_format;
#endif
	auto& e = hist.entries[{t.ps, t.slot, t.addr, fmt, t.width, t.height}];
	e.draws++;
#ifndef KYTY_DIAGTEX_MUTANT_HIST_NO_INDICES
	e.indices += index_count;
#endif
	if (bc7 != nullptr && !e.has_bc7) {
		e.bc7     = *bc7;
		e.has_bc7 = true;
	}
}

// "[diag-tex-h] ..." lines, most draws first; the percentage is of all draws of the same (ps, slot).
inline std::vector<std::string> TexHistLines(const TexHist& hist, uint64_t flip) {
	std::map<std::pair<uint64_t, uint32_t>, uint64_t> totals;
	for (const auto& [key, e]: hist.entries) totals[{std::get<0>(key), std::get<1>(key)}] += e.draws;
	std::vector<std::pair<TexHist::Key, TexHist::Entry>> rows(hist.entries.begin(), hist.entries.end());
	std::stable_sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.draws > b.second.draws; });
	std::vector<std::string> lines;
	for (const auto& [key, e]: rows) {
		const auto total = totals[{std::get<0>(key), std::get<1>(key)}];
		char       buf[400];
		std::snprintf(buf, sizeof(buf),
		              "[diag-tex-h] flip=%" PRIu64 " ps=%016" PRIx64 " slot=%u addr=0x%" PRIx64 " fmt=%u %ux%u draws=%" PRIu64
		              " (%.1f%% of %" PRIu64 ") indices=%" PRIu64,
		              flip, std::get<0>(key), std::get<1>(key), std::get<2>(key), std::get<3>(key), std::get<4>(key),
		              std::get<5>(key), e.draws, 100.0 * static_cast<double>(e.draws) / static_cast<double>(total), total, e.indices);
		std::string line = buf;
		if (e.has_bc7) {
			std::snprintf(buf, sizeof(buf), " bc7 mode=%d first=%02x%02x%02x%02x mean=%.1f,%.1f,%.1f,%.1f", e.bc7.first_mode,
			              e.bc7.first_texel[0], e.bc7.first_texel[1], e.bc7.first_texel[2], e.bc7.first_texel[3], e.bc7.mean[0],
			              e.bc7.mean[1], e.bc7.mean[2], e.bc7.mean[3]);
			line += buf;
		}
		lines.push_back(line);
	}
	return lines;
}

} // namespace Libs::Graphics::DiagDump

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_DIAGTEX_H_ */
