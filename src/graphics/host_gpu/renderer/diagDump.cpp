#include "graphics/host_gpu/renderer/diagDump.h"

#include "common/controlFile.h"
#include "common/logging/log.h"
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

} // namespace

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
	if (g_pending.empty()) return;
	const char* dir   = std::getenv("KYTY_FRAME_DUMP_DIR");
	const auto& label = Common::ControlFile::Current().label;
	for (auto& [key, t]: g_pending) {
		const auto layout = RtLayoutFor(t.vk_format);
		char       what[96];
		std::snprintf(what, sizeof(what), "flip=%" PRIu64 " ps=%016" PRIx64 " %s%d draws=%u", flip, key.first,
		              t.mrt < 0 ? "depth" : "mrt", t.mrt < 0 ? 0 : t.mrt, g_draws[key]);
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
