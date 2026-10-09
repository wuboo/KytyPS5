#include "graphics/presentation/frameTiming.h"

#include "common/memEvents.h"
#include "common/controlFile.h"
#include "common/logging/log.h"
#include "common/profiler.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <string>

namespace Libs::Graphics::FrameTiming {

namespace {

std::atomic<uint64_t> g_presented_frames {0};
std::atomic<uint64_t> g_guest_flips {0};
std::atomic<uint64_t> g_gpu_busy_ns {0};
std::atomic<uint64_t> g_submits {0};

constexpr const char* kCounterNames[] = {
    "empty_submits",
    "readbacks",
    "readback_downloads",
    "readback_wait_ns",
    "rb_read_fault",
    "rb_read_fault_gpu",
    "rb_write_fault",
    "rb_dcc",
    "write_faults",
    "write_faults_gpu",
    "finishes",
    "draws",
    "dispatches",
    "descriptor_sets",
    "render_passes",
    "pipelines_created",
    "pipeline_compile_ns",
    "dma_memcpy_bytes",
    "buffer_delete_drains",
    "buffer_delete_wait_ns",
    "submits_flush",
    "submits_flush_and_wait",
    "submits_finish",
    "submits_wait_current",
    "cp_flushes",
    "slice_suspends",
    "skipped_submits",
    "descriptor_sets_identical",
    "descriptor_sets_reused",
    "resource_memo_hits",
    "resource_memo_misses",
    "compute_fills_replaced",
    "dcc_known_from_fill",
    "flushes_deferred",
    "prototype_draws",
    "prototype_vertices",
    "prototype_render_passes",
    "prototype_capture_cache_hits",
    "rb_eager_hits",
    "rb_eager_downloads",
    "prototype_range_memo_hits",
    "prototype_range_memo_misses",
    "proto_unsupported_draws",
    "minlod_dropped",
    "minlod_remapped",
    "minlod_frac",
    "minlod_resinfo",
    "drop_depth_clamp",
    "drop_depth_bounds",
    "drop_color_write_dyn",
    "drop_sampler_border_table",
    "drop_sampler_lod_bias",
    "cp_wait_guest_ns",
    "cp_busy_ns",
    "cp_proto_ns",
    "cp_wait_gpu_ns",
    "submit_encode_ns",
    "submit_queue_wait_ns",
    "disp_guest",
    "disp_proto_unpack",
    "disp_proto_capture",
    "disp_proto_capture_nonnative",
    "disp_tiler",
    "disp_fault",
    "disp_other",
    "proto_skipped_draws",
    "prototype_unique_draws",
    "prototype_unique_vertices",
    "prototype_unique_fallback",
};
static_assert(std::size(kCounterNames) == static_cast<size_t>(Counter::Count));
static_assert(static_cast<uint32_t>(Counter::DispOther) - static_cast<uint32_t>(Counter::DispGuest) ==
              static_cast<uint32_t>(DispatchClass::Other));

std::array<std::atomic<uint64_t>, static_cast<size_t>(Counter::Count)> g_counters {};

FILE* OpenFrameLog() {
	const char* path = std::getenv("KYTY_FRAME_LOG");
	if (path == nullptr || *path == '\0') {
		return nullptr;
	}
	FILE* file = std::fopen(path, "w");
	if (file != nullptr) {
		std::fputs("frame,host_ns,flips,gpu_busy_ns,submits", file);
		for (const auto* name: kCounterNames) {
			std::fprintf(file, ",%s", name);
		}
		std::fputs("\n", file);
	}
	return file;
}

} // namespace

void OnFramePresented() {
	// Tracy runs with a manual lifetime and is only started by --profile.
	if (tracy::ProfilerAvailable()) {
		FrameMark;
	}

	// Presentation happens on a single thread, so the log needs no locking.
	static FILE* log = OpenFrameLog();

	const auto frame = g_presented_frames.fetch_add(1, std::memory_order_relaxed) + 1;
	if (log != nullptr) {
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		                    std::chrono::steady_clock::now().time_since_epoch())
		                    .count();
		std::fprintf(log, "%llu,%lld,%llu,%llu,%llu", static_cast<unsigned long long>(frame),
		             static_cast<long long>(ns),
		             static_cast<unsigned long long>(g_guest_flips.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(g_gpu_busy_ns.load(std::memory_order_relaxed)),
		             static_cast<unsigned long long>(g_submits.load(std::memory_order_relaxed)));
		for (const auto& counter: g_counters) {
			std::fprintf(log, ",%llu",
			             static_cast<unsigned long long>(counter.load(std::memory_order_relaxed)));
		}
		std::fputs("\n", log);
		// The emulator usually exits through a hard stop, so keep the file current.
		std::fflush(log);
	}
}

namespace {

// KYTY_CONTROL_FILE: re-read when it changed, resolve the lines for the next flip (`flip` flips are
// done, the draws that follow belong to flip + 1), publish on change. The base applies from flip 2.
void PollControlFile(uint64_t flip) {
	static const char* path = std::getenv("KYTY_CONTROL_FILE");
	if (path == nullptr || path[0] == '\0') return;
	static Common::ControlFile::Poller poller {.path = path};
	for (const auto& error: poller.Refresh()) {
		LOGF("[control] %s: %s\n", path, error.c_str());
	}
	if (!poller.loaded) return;
	if (!Common::ControlFile::Publish(Common::ControlFile::Resolve(poller.file, flip + 1))) return;
	const auto& now = Common::ControlFile::Current();
	LOGF("[control] gen=%llu from_flip=%llu block=%s label=%s skip_cs=%zu dump_rt=%zu force_tex=%zu diag_tex=%zu dump_tex=%zu watch_tex=%zu\n",
	     static_cast<unsigned long long>(now.gen), static_cast<unsigned long long>(flip + 1),
	     now.block.empty() ? "base" : now.block.c_str(), now.label.c_str(), now.skip_cs.size(),
	     now.dump_rt.size(), now.force_tex.size(), now.diag_tex.size(), now.dump_tex.size(), now.watch_tex.size());
}

} // namespace

void OnGuestFlip() {
	const auto flip = g_guest_flips.fetch_add(1, std::memory_order_relaxed) + 1;
	Common::MemEvents::SetCompletedFlips(flip);
	PollControlFile(flip);
}

void AddGpuBusy(uint64_t ns) {
	g_gpu_busy_ns.fetch_add(ns, std::memory_order_relaxed);
}

void OnQueueSubmit() {
	g_submits.fetch_add(1, std::memory_order_relaxed);
}

void Add(Counter counter, uint64_t value) {
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

bool OptEnabled(const char* name) {
	static const std::string disabled = [] {
		const char* value = std::getenv("KYTY_OPT_OFF");
		return value != nullptr ? "," + std::string(value) + "," : std::string();
	}();
	return disabled.find("," + std::string(name) + ",") == std::string::npos;
}

bool OptForcedOn(const char* name) {
	static const std::string enabled = [] {
		const char* value = std::getenv("KYTY_OPT_ON");
		return value != nullptr ? "," + std::string(value) + "," : std::string();
	}();
	return enabled.find("," + std::string(name) + ",") != std::string::npos;
}

uint64_t PresentedFrames() {
	return g_presented_frames.load(std::memory_order_relaxed);
}

uint64_t GuestFlips() {
	return g_guest_flips.load(std::memory_order_relaxed);
}

} // namespace Libs::Graphics::FrameTiming
