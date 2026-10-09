#include "graphics/host_gpu/renderer/diagDump.h"

#include "common/controlFile.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/diagTex.h"
#include "graphics/host_gpu/renderer/rtDump.h"
#include "graphics/presentation/frameTiming.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
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
uint64_t                                     g_tex_seq = 0;

} // namespace

bool WantsTex(uint64_t ps_hash, uint32_t slot) {
	const auto& snapshot = Common::ControlFile::Current();
	if (snapshot.diag_tex.empty() || ps_hash == 0 || !snapshot.DiagTexListed(ps_hash, slot)) return false;
	std::lock_guard lock(g_mutex);
	return TexFirst(g_tex_once, snapshot.gen, ps_hash, slot);
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
