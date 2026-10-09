#include "graphics/host_gpu/renderer/diagDump.h"

#include "common/controlFile.h"
#include "common/logging/log.h"
#include "common/memEvents.h"
#include "kernel/memory.h"
#include "graphics/host_gpu/renderer/diagTex.h"
#include "graphics/host_gpu/renderer/rtDump.h"
#include "graphics/presentation/frameTiming.h"

#include <cinttypes>
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>

namespace Libs::Graphics::DiagDump {

namespace {

std::mutex                                   g_mutex;
RtOnceState                                  g_once;
std::map<std::pair<uint64_t, int>, Target>   g_pending; // (ps, mrt) -> last draw's copy
std::map<std::pair<uint64_t, int>, uint32_t> g_draws;
TexOnceState                                 g_tex_once;
std::vector<TexRequest>                      g_tex_requests;
std::vector<std::pair<TexInfo, Target>>      g_tex_done;
std::vector<TexDraw>                         g_tex_draws; // queued by ResolveTexture, finished at the end of the draw
TexHist                                      g_tex_hist;
TexOnceState                                 g_dump_tex_once;
std::vector<DumpTexRequest>                  g_dump_tex_requests;
uint64_t                                     g_tex_seq = 0;

// S125 watch_tex
TexOnceState g_watch_tex_once;
struct SampledTex {
	uint64_t ps, addr;
	uint32_t slot, fmt, width, height;
};
struct WatchRange {
	uint64_t flip, ps, addr, size;
	uint32_t slot;
};
std::vector<SampledTex> g_sampled_this_flip; // deduplicated by (ps, slot, addr), capped
std::vector<WatchRange> g_watch_ranges;      // armed by WatchTexTrigger, reported (and dropped) at the end of their flip

} // namespace

bool WantsTex(uint64_t ps_hash, uint32_t slot) {
	const auto& snapshot = Common::ControlFile::Current();
	if (snapshot.diag_tex.empty() || ps_hash == 0 || !snapshot.DiagTexListed(ps_hash, slot)) return false;
	std::lock_guard lock(g_mutex);
	return TexFirst(g_tex_once, snapshot.gen, ps_hash, slot);
}

bool WantsDumpTex(uint64_t ps_hash, uint32_t slot, const Common::ControlFile::TexKey& key) {
	const auto& snapshot = Common::ControlFile::Current();
	if (snapshot.dump_tex.empty() || ps_hash == 0 || !snapshot.DumpTexListed(ps_hash, slot, key)) return false;
	std::lock_guard lock(g_mutex);
	return TexFirst(g_dump_tex_once, snapshot.gen, ps_hash, slot);
}

bool WantsWatchTex(uint64_t ps_hash, uint32_t slot, const Common::ControlFile::TexKey& key) {
	const auto& snapshot = Common::ControlFile::Current();
	if (snapshot.watch_tex.empty() || ps_hash == 0 || !snapshot.WatchTexListed(ps_hash, slot, key)) return false;
	std::lock_guard lock(g_mutex);
	return TexFirst(g_watch_tex_once, snapshot.gen, ps_hash, slot);
}

void NoteSampledTex(uint64_t ps_hash, uint32_t slot, uint64_t addr, uint32_t fmt, uint32_t width, uint32_t height) {
	if (ps_hash == 0 || !Common::MemEvents::Enabled()) return;
	std::lock_guard lock(g_mutex);
	if (g_sampled_this_flip.size() >= 8192) return;
	for (const auto& t: g_sampled_this_flip) {
		if (t.ps == ps_hash && t.slot == slot && t.addr == addr) return;
	}
	g_sampled_this_flip.push_back({ps_hash, addr, slot, fmt, width, height});
}

void WatchTexTrigger(const TexInfo& info, uint64_t range_size) {
	LOGF("[watch-tex] flip=%" PRIu64 " ps=%016" PRIx64 " slot=%u T# addr=0x%" PRIx64 " fmt=%u %ux%u tile=%u image_size=0x%" PRIx64 " (%" PRIu64 " tiles of 64 KB)\n",
	     info.flip, info.ps, info.slot, info.addr, info.guest_format, info.width, info.height, info.tile_mode, range_size, (range_size + 0xffffu) / 0x10000u);
	if (!Common::MemEvents::Enabled()) {
		LOGF("[watch-tex] the memory event ring is off (KYTY_CONTROL_FILE was not set at start): no history\n");
		return;
	}
	for (const auto& line: Libs::LibKernel::Memory::DiagDescribeRange(info.addr, range_size)) LOGF("[watch-tex] %s\n", line.c_str());
	for (const auto& line: Common::MemEvents::WatchLines(Common::MemEvents::Global(), info.addr, range_size, 64)) LOGF("[watch-tex] %s\n", line.c_str());
	std::lock_guard lock(g_mutex);
	g_watch_ranges.push_back({info.flip, info.ps, info.addr, range_size, info.slot});
}

void AddDumpTexRequest(DumpTexRequest&& request) {
	std::lock_guard lock(g_mutex);
	g_dump_tex_requests.push_back(std::move(request));
}

std::vector<DumpTexRequest> TakeDumpTexRequests() {
	std::lock_guard lock(g_mutex);
	return std::exchange(g_dump_tex_requests, {});
}

// Writes `rgba` (w x h) as a PNG no larger than `limit` pixels on a side (box filter by an integer factor).
static void WritePng(const std::string& path, const std::vector<uint8_t>& rgba, uint32_t w, uint32_t h, uint32_t limit) {
	uint32_t   ow = 0, oh = 0;
	const auto small = DownscaleRgba(rgba, w, h, FactorFor(w, h, limit), ow, oh);
	const auto png   = PngEncodeRgba(small.data(), ow, oh);
	if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
		std::fwrite(png.data(), 1, png.size(), file);
		std::fclose(file);
		LOGF("[dump-tex] wrote %s (%ux%u from %ux%u)\n", path.c_str(), ow, oh, w, h);
	} else {
		LOGF("[dump-tex] cannot write %s\n", path.c_str());
	}
}

void WriteDumpTex(const DumpTexResult& r) {
	const auto&    t = r.info;
	char           head[256];
	std::snprintf(head, sizeof(head), "[dump-tex] flip=%" PRIu64 " ps=%016" PRIx64 " slot=%u addr=0x%" PRIx64 " fmt=%u %ux%u tile=%u blocks=%ux%u",
	              t.flip, t.ps, t.slot, t.addr, t.guest_format, t.width, t.height, t.tile_mode, r.blocks_w, r.blocks_h);
	LOGF("%s\n", head);
	for (const auto& note: r.notes) LOGF("%s\n", note.c_str());
	const uint32_t count = r.blocks_w * r.blocks_h;
	const size_t   bytes = static_cast<size_t>(count) * 16;
	const bool     have_guest = count != 0 && r.guest.size() >= bytes, have_host = count != 0 && r.host.size() >= bytes;
	const char*    dir = std::getenv("KYTY_DIAG_DIR");
	const bool     files = dir != nullptr && dir[0] != '\0';
	const auto&    label = Common::ControlFile::Current().label;
	constexpr uint32_t kLimit = 1024;
	if (have_guest) {
		LOGF("%s\n", Bc7StatsLine("guest", Bc7Analyze(r.guest.data(), count)).c_str());
		if (files) WritePng(DumpTexPath(dir, label, t.flip, t.ps, t.slot, "guest"), Bc7ToRgba(r.guest.data(), r.blocks_w, r.blocks_h), r.blocks_w * 4, r.blocks_h * 4, kLimit);
	}
	if (have_host) {
		LOGF("%s\n", Bc7StatsLine("host", Bc7Analyze(r.host.data(), count)).c_str());
		if (files) WritePng(DumpTexPath(dir, label, t.flip, t.ps, t.slot, "host"), Bc7ToRgba(r.host.data(), r.blocks_w, r.blocks_h), r.blocks_w * 4, r.blocks_h * 4, kLimit);
	}
	if (have_guest && have_host) {
		LOGF("%s\n", Bc7DiffLine(Bc7Compare(r.guest.data(), r.host.data(), r.blocks_w, r.blocks_h), r.blocks_w).c_str());
		if (files) {
			const uint32_t scale = std::max(1u, 512u / std::max(std::max(r.blocks_w, r.blocks_h), 1u));
			WritePng(DumpTexPath(dir, label, t.flip, t.ps, t.slot, "diff"), Bc7DiffMap(r.guest.data(), r.host.data(), r.blocks_w, r.blocks_h, scale), r.blocks_w * scale, r.blocks_h * scale, kLimit);
		}
	} else {
		LOGF("[dump-tex] diff unavailable (guest %s, host %s)\n", have_guest ? "ok" : "missing", have_host ? "ok" : "missing");
	}
}

void WriteTileMosaic(const TexInfo& t, const std::vector<std::vector<uint8_t>>& tiles, uint32_t bw, uint32_t bh) {
	if (tiles.empty() || bw == 0 || bh == 0) return;
	const uint32_t tw = bw * 4, th = bh * 4, cols = 10, rows = static_cast<uint32_t>((tiles.size() + cols - 1) / cols);
	std::vector<uint8_t> mosaic(static_cast<size_t>(cols) * tw * rows * th * 4, 0);
	for (size_t k = 0; k < tiles.size(); k++) {
		const auto stats = Bc7Analyze(tiles[k].data(), bw * bh);
		LOGF("[dump-tex-tile] flip=%" PRIu64 " k=%zu offset=0x%zx blocks=%u zero=%u black=%u decoded=%u reserved=%u mean_rgba=%.0f,%.0f,%.0f,%.0f\n", t.flip, k,
		     k * static_cast<size_t>(bw) * bh * 16, stats.blocks, stats.zero_blocks, stats.black_blocks, stats.decoded, stats.reserved, stats.mean[0],
		     stats.mean[1], stats.mean[2], stats.mean[3]);
		const auto rgba = Bc7ToRgba(tiles[k].data(), bw, bh);
		const uint32_t cx = static_cast<uint32_t>(k % cols), cy = static_cast<uint32_t>(k / cols);
		for (uint32_t y = 0; y < th; y++) {
			std::memcpy(mosaic.data() + ((static_cast<size_t>(cy) * th + y) * cols * tw + static_cast<size_t>(cx) * tw) * 4,
			            rgba.data() + static_cast<size_t>(y) * tw * 4, static_cast<size_t>(tw) * 4);
		}
	}
	const char* dir = std::getenv("KYTY_DIAG_DIR");
	if (dir != nullptr && dir[0] != '\0') {
		WritePng(DumpTexPath(dir, Common::ControlFile::Current().label, t.flip, t.ps, t.slot, "tiles"), mosaic, cols * tw, rows * th, 1280);
	}
}

bool WantsTexAll(uint64_t ps_hash, uint32_t slot) {
	const auto& snapshot = Common::ControlFile::Current();
	return !snapshot.diag_tex.empty() && ps_hash != 0 && snapshot.DiagTexAll(ps_hash, slot);
}

void AddTexDraw(TexDraw&& draw) {
	std::lock_guard lock(g_mutex);
	for (const auto& queued: g_tex_draws) { // a draw can resolve a slot twice (prepare + re-resolve)
		if (queued.info.ps == draw.info.ps && queued.info.slot == draw.info.slot) return;
	}
	g_tex_draws.push_back(std::move(draw));
}

void TexDrawsDone(uint64_t index_count, uint64_t rt_addr) {
	std::lock_guard lock(g_mutex);
	for (const auto& draw: g_tex_draws) {
		const auto* bc7 = draw.has_bc7 ? &draw.bc7 : nullptr;
		LOGF("%s\n", TexDrawDescribe(++g_tex_seq, draw.info, index_count, rt_addr, bc7).c_str());
		TexHistAdd(g_tex_hist, draw.info, index_count, bc7);
	}
	g_tex_draws.clear();
}

void AddTexRequest(TexRequest&& request) {
	std::lock_guard lock(g_mutex);
	g_tex_requests.push_back(std::move(request));
}

std::vector<TexRequest> TakeTexRequests() {
	std::lock_guard lock(g_mutex);
	return std::exchange(g_tex_requests, {});
}

void StoreTex(const TexInfo& info, Target&& target) {
	std::lock_guard lock(g_mutex);
	g_tex_done.emplace_back(info, std::move(target));
}

static void ReportTextures(uint64_t flip, const char* dir, const std::string& label) {
	for (auto& [info, t]: g_tex_done) {
		const auto layout = RtLayoutFor(t.vk_format);
		const size_t texels = static_cast<size_t>(t.width) * t.height;
		if (layout.kind == RtKind::Unknown || layout.bytes == 0 || texels == 0 || t.data.size() < texels * layout.bytes) {
			LOGF("[diag-tex] flip=%" PRIu64 " ps=%016" PRIx64 " slot=%u no readback (host format %d %ux%u, %zu bytes)\n", flip,
			     info.ps, info.slot, t.vk_format, t.width, t.height, t.data.size());
			continue;
		}
		const auto values = RtDecode(layout, t.data.data(), texels);
		const auto stats  = RtComputeStats(values, layout.channels);
		LOGF("%s ps=%016" PRIx64 " slot=%u\n", TexStats(layout, t.width, t.height, values).c_str(), info.ps, info.slot);
		if (dir == nullptr) continue;
		const auto ppm_path = TexDumpPath(dir, label, flip, info.ps, info.slot, "ppm");
		const auto raw_path = TexDumpPath(dir, label, flip, info.ps, info.slot, "raw");
		const auto ppm      = RtPpm(RtToRgb8(layout, values, stats), t.width, t.height);
		if (FILE* file = std::fopen(ppm_path.c_str(), "wb"); file != nullptr) {
			std::fwrite(ppm.data(), 1, ppm.size(), file);
			std::fclose(file);
			LOGF("[diag-tex] wrote %s\n", ppm_path.c_str());
		}
		if (FILE* file = std::fopen(raw_path.c_str(), "wb"); file != nullptr) {
			std::fwrite(t.data.data(), 1, texels * layout.bytes, file);
			std::fclose(file);
			LOGF("[diag-tex] wrote %s (%ux%u host vkfmt %d)\n", raw_path.c_str(), t.width, t.height, t.vk_format);
		}
	}
	g_tex_done.clear();
}

bool Wants(uint64_t ps_hash) {
	const auto& snapshot = Common::ControlFile::Current();
	if (snapshot.dump_rt.empty()) return false;
	std::lock_guard lock(g_mutex);
	return DumpRtMatch(snapshot.dump_rt, ps_hash, FrameTiming::GuestFlips() + 1, snapshot.gen, g_once);
}

void Store(uint64_t ps_hash, Target&& target) {
	std::lock_guard lock(g_mutex);
	const auto      key = std::make_pair(ps_hash, target.mrt);
	g_draws[key]++;
	g_pending[key] = std::move(target);
}

void FlipReport(uint64_t flip) {
	std::lock_guard lock(g_mutex);
	for (const auto& watch: g_watch_ranges) {
		size_t n = 0;
		for (const auto& t: g_sampled_this_flip) {
			if (t.addr < watch.addr || t.addr >= watch.addr + watch.size) continue;
			LOGF("[watch-tex] flip=%" PRIu64 " T# inside the watched range: ps=%016" PRIx64 " slot=%u addr=0x%" PRIx64 " (+0x%" PRIx64 ") fmt=%u %ux%u%s\n", flip, t.ps,
			     t.slot, t.addr, t.addr - watch.addr, t.fmt, t.width, t.height, t.addr == watch.addr && t.ps == watch.ps && t.slot == watch.slot ? " (the watched T#)" : "");
			n++;
		}
		LOGF("[watch-tex] flip=%" PRIu64 " %zu T# sampled in this flip start inside [0x%" PRIx64 ", 0x%" PRIx64 ") (of %zu distinct T# sampled)\n", flip, n, watch.addr,
		     watch.addr + watch.size, g_sampled_this_flip.size());
	}
	g_watch_ranges.clear();
	g_sampled_this_flip.clear();
	const char* diag_dir = std::getenv("KYTY_DIAG_DIR");
	for (const auto& line: TexHistLines(g_tex_hist, flip)) LOGF("%s\n", line.c_str());
	g_tex_hist.entries.clear();
	ReportTextures(flip, diag_dir != nullptr && diag_dir[0] != '\0' ? diag_dir : nullptr, Common::ControlFile::Current().label);
	if (g_pending.empty()) return;
	const char* dir   = std::getenv("KYTY_FRAME_DUMP_DIR");
	const auto& label = Common::ControlFile::Current().label;
	for (auto& [key, t]: g_pending) {
		const auto layout = RtLayoutFor(t.vk_format);
		char       what[96];
		std::snprintf(what, sizeof(what), "flip=%" PRIu64 " ps=%016" PRIx64 " %s%d draws=%u%s", flip, key.first,
		              t.mrt < 0 ? "depth" : "mrt", t.mrt < 0 ? 0 : t.mrt, g_draws[key], RtAddrText(t.addr).c_str());
		const size_t texels = static_cast<size_t>(t.width) * t.height;
		if (layout.kind == RtKind::Unknown || layout.bytes == 0 || t.data.size() < texels * layout.bytes) {
			LOGF("[dump-rt] %s fmt=%d %ux%u unsupported or short readback (%zu bytes)\n", what, t.vk_format, t.width,
			     t.height, t.data.size());
			continue;
		}
		const auto values = RtDecode(layout, t.data.data(), texels);
		const auto stats  = RtComputeStats(values, layout.channels);
		LOGF("%s\n", RtDescribe(what, layout, t.width, t.height, stats).c_str());
		if (dir == nullptr) continue;
		const auto path = RtDumpPath(dir, label, flip, t.mrt);
		const auto ppm  = RtPpm(RtToRgb8(layout, values, stats), t.width, t.height);
		if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
			std::fwrite(ppm.data(), 1, ppm.size(), file);
			std::fclose(file);
			LOGF("[dump-rt] wrote %s\n", path.c_str());
		}
	}
	g_pending.clear();
	g_draws.clear();
}

} // namespace Libs::Graphics::DiagDump
