#include "graphics/presentation/frameTiming.h"
#include "graphics/presentation/cpPhases.h"
#include "graphics/shader/nullDescDiag.h"

#include "common/memEvents.h"
#include "common/controlFile.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/signalCensus.h"

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
    "pv_vsstore_draws",
    "pv_fallback_instanced",
    "pv_fallback_range",
    "pv_fallback_topology",
    "pv_fallback_param31",
    "pv_fallback_ring",
    "pv_fallback_interface",
    "pv_fallback_index_tail",
    "pv_vsstore_instanced",
    "pv_push_nonzero_programs",
    "pv_vsstore_oob",
    "pv_vsstore_idx_gpu",
    "depth_ro_elided",
    "depth_ro_reject_dirty",
    "depth_ro_reject_not_subset",
    "depth_ro_reject_layout",
    "depth_ro_bad_elide",
    "cp_parse_ns",
    "cp_handler_ns",
    "cp_draw_shaders_ns",
    "cp_draw_bind_ns",
    "cp_draw_pipeline_ns",
    "cp_draw_targets_ns",
    "cp_draw_desc_ns",
    "cp_draw_vkcmd_ns",
    "cp_pvstore_ns",
    "cp_dispatch_ns",
    "cp_eop_ns",
    "cp_barrier_ns",
    "cp_readback_cpu_ns",
    "cp_gc_ns",
    "cp_submit_cpu_ns",
    "cp_cmds_ns",
    "cp_phase_wait_ns",
    "pm4_packets",
    "mprotect_n",
    "mprotect_ns",
    "rb_download_bytes",
    "pv_vsstore_index_bytes",
    "seg_n",
    "seg_drain_to_first_submit_ns",
    "seg_record_ns",
    "gpu_gap_after_drain_ns",
    "eager_submit_n",
    "gpu_gap_n",
    "gpu_gap_mid_ns",
    "eager_submit_pass_n",
    "seg_first_pass_end_ns",
    "eager_mid_n",
    "first_submit_n",
    "first_submit_enq_to_vk_ns",
    "first_submit_encode_ns",
    "drain_wake_lag_n",
    "drain_wake_lag_ns",
    "first_submit_vk_to_gpu_begin_n",
    "first_submit_vk_to_gpu_begin_ns",
    "desc_assert_cached_n",
    "desc_assert_run_n",
    "desc_assert_skip_n",
    "desc_assert_reads_n",
    "desc_assert_ns",
};
static_assert(static_cast<uint32_t>(Counter::CpPhaseWaitNs) - static_cast<uint32_t>(Counter::CpParseNs) ==
              static_cast<uint32_t>(CpPhases::Phase::Wait));
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
		Common::SignalCensus::WriteCsvHeader(file); // S276
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
		Common::SignalCensus::WriteCsvRow(log); // S276
		std::fputs("\n", log);
		// The emulator usually exits through a hard stop, so keep the file current.
		std::fflush(log);
	}
	NullDesc::OnPresented(static_cast<uint32_t>(frame),
	                      static_cast<uint32_t>(g_guest_flips.load(std::memory_order_relaxed)),
	                      Get(Counter::MinLodRemapped), Get(Counter::MinLodDropped));
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
	// S276: KYTY_DIAG_SIGCENSUS=1 dumps the RIP histogram every 600 guest flips (and at exit, see ProgramExitHandler).
	if (flip % 600 == 0 && Common::SignalCensus::HistogramEnabled()) {
		char why[32];
		std::snprintf(why, sizeof(why), "flip %llu", static_cast<unsigned long long>(flip));
		Common::SignalCensus::DumpHistogram(why, [](const char* line) { LOGF("%s", line); });
	}
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

void Set(Counter counter, uint64_t value) {
	g_counters[static_cast<size_t>(counter)].store(value, std::memory_order_relaxed);
}

uint64_t Get(Counter counter) {
	return g_counters[static_cast<size_t>(counter)].load(std::memory_order_relaxed);
}

void Max(Counter counter, uint64_t value) {
	auto& slot  = g_counters[static_cast<size_t>(counter)];
	auto  known = slot.load(std::memory_order_relaxed);
	while (known < value &&
	       !slot.compare_exchange_weak(known, value, std::memory_order_relaxed)) {
	}
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
