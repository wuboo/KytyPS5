#include "graphics/presentation/frameTiming.h"
#include "graphics/presentation/cpPhases.h"
#include "graphics/presentation/cpSampler.h"
#include "graphics/host_gpu/renderer/rfOpDiag.h"
#include "graphics/host_gpu/renderer/cpWaitDiag.h"
#include "graphics/shader/nullDescDiag.h"

#include "common/memEvents.h"
#include "common/controlFile.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/signalCensus.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

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
    "sync_dcc_n",
    "sync_rf_gpu_n",
    "sync_rf_guest_n",
    "sync_write_fault_n",
    "sync_spec_n",
    "sync_other_n",
    "sync_dcc_ns",
    "sync_rf_gpu_ns",
    "sync_rf_guest_ns",
    "sync_write_fault_ns",
    "sync_spec_ns",
    "sync_other_ns",
    "sync_eager_hit_dcc_n",
    "sync_eager_hit_rf_gpu_n",
    "sync_eager_hit_rf_guest_n",
    "sync_eager_hit_write_fault_n",
    "sync_eager_hit_spec_n",
    "sync_eager_hit_other_n",
    "sync_lag0_n",
    "sync_lag_sum",
    "sync_writer_done_n",
    "sync_writer_done_ns",
    "sync_dcc_repeat_n",
    "sync_dcc_repeat_same_n",
    "indirect_n",
    "indirect_dirty_n",
    "indirect_dirty_wait_ns",
    "indirect_regs_patched_n",
    "indirect_pv_n",
    "indirect_args_changed_n",
    "rt_max_w",
    "rt_max_h",
    "dcc_publish_n",
    "dcc_publish_fallback_n",
    "dcc_known_would_hit_n",
    "dcc_known_mismatch_n",
    "dcc_eager_hit_n",
    "dcc_eager_miss_n",
    "eager_pages_n",
    "sync_stale_eq_spec_n",
    "sync_stale_ne_spec_n",
    "sync_stale_eq_rf_gpu_n",
    "sync_stale_ne_rf_gpu_n",
    "sync_eager_already_done_n",
    "sync_eager_lag_sum",
    "sync_eager_wait_ns",
    "sync_queue_ns",
    "sync_lambda_ns",
    "dcc_fallback_no_fill_n",
    "dcc_fallback_partial_n",
    "dcc_fallback_newer_write_n",
    "dcc_fallback_pending_download_n",
    "dcc_fallback_other_n",
    "indirect_gpu_candidate_n",
    "indirect_gpu_eligible_n",
    "indirect_gpu_host_eligible_n",
    "indirect_gpu_n",
    "indirect_gpu_fallback_n",
    "indirect_gpu_clamp_risk_n",
    "cp_sample_n",
    "cp_sample_wait_n",
    "sync_spec_purpose_cond_n",
    "sync_spec_purpose_slot_n",
    "sync_spec_purpose_desc_n",
    "sync_spec_purpose_fill_n",
    "sync_spec_purpose_wg_n",
    "sync_spec_purpose_call_n",
    "sync_spec_purpose_other_n",
    "spec_dec_stale_eq_n",
    "spec_dec_stale_ne_n",
    "spec_dec_super_ok_n",
    "spec_dec_super_fail_n",
    "spec_super_extra_sources_n",
    "spec_perm_same_n",
    "spec_super_new_images_n",
    "indirect_pv_args_eq_n",
    "indirect_pv_args_ne_n",
    "indirect_pv_inst_max",
    "mat_memo_vs_n",
    "mat_memo_ps_n",
    "mat_memo_cs_n",
    "mat_memo_in_hit_vs_n",
    "mat_memo_in_hit_ps_n",
    "mat_memo_in_hit_cs_n",
    "mat_memo_unsafe_vs_n",
    "mat_memo_unsafe_ps_n",
    "mat_memo_unsafe_cs_n",
    "mat_memo_out_hit_vs_n",
    "mat_memo_out_hit_ps_n",
    "mat_memo_out_hit_cs_n",
    "mat_memo_rebase_vs_n",
    "mat_memo_rebase_ps_n",
    "mat_memo_rebase_cs_n",
    "gpu_active_est_ns",
    "pv_gen_hit_n",
    "pv_gen_miss_n",
    "pv_gen_mismatch_n",
    "pv_hash_bytes_skipped",
    "pv_gen_ineligible_host_n",
    "pv_gen_ineligible_unreg_n",
    "pv_gen_ineligible_dirty_n",
    "spec_cond_superset_n",
    "spec_cond_superset_skip_miss_n",
    "spec_super_perm_same_n",
    "spec_super_spec_eq_n",
    "spec_super_extra_buffer_n",
    "spec_super_extra_image_n",
    "spec_super_extra_sampler_n",
    "spec_super_extra_written_n",
    "spec_super_extra_bad_addr_n",
    "spec_super_dirty_noncond_n",
    "pv_ind_n",
    "pv_ind_learn_n",
    "pv_ind_cooldown_n",
    "pv_ind_decline_n",
    "pv_ind_decline_range_n",
    "pv_ind_decline_ring_n",
    "pv_ind_decline_index_tail_n",
    "pv_ind_decline_too_large_n",
    "pv_ind_decline_off_n",
    "pv_ind_decline_residual_n",
    "pv_ind_decline_other_n",
    "pv_ind_pass_break_n",
    "pv_ind_mismatch_n",
    "pv_ind_overflow_n",
    "pv_ind_cap",
    "pv_ind_max_instance",
    "pv_ind_shadow_n",
    "pv_ind_shadow_args_eq_n",
    "pv_ind_shadow_args_ne_n",
    "pv_ind_shadow_fields_ok_n",
    "pv_ind_shadow_fields_mismatch_n",
    "pv_ind_shadow_overflow_n",
    "pv_ind_shadow_plan_eq_n",
    "pv_ind_shadow_plan_ne_n",
    "pv_ind_shadow_no_store_n",
    "pv_ind_shadow_multi_n",
    "gpu_starve_ns",
    "gpu_starve_work_ns",
    "gpu_starve_n",
    "gpu_starve_ge1ms_n",
    "gpu_starve_work_ns_drain",
    "gpu_starve_work_ns_lazy_max",
    "gpu_starve_work_ns_lazy_1ms",
    "gpu_starve_work_ns_lazy_idle",
    "gpu_starve_work_ns_eager",
    "gpu_starve_work_ns_eager_pass",
    "gpu_starve_work_ns_eager_mid",
    "gpu_starve_work_ns_eager3_mid",
    "gpu_starve_work_ns_eager3_flush",
    "gpu_starve_work_ns_lowwater",
    "gpu_starve_work_ns_flip",
    "gpu_starve_work_ns_other",
    "eager3_mid_n",
    "eager3_flush_n",
    "eager_lowwater_n",
    "eager_decline_pass_open_n",
    "eager_decline_age_n",
    "eager_decline_no_work_n",
    "eager_decline_mid_limit_n",
    "eager_decline_first_cb_only_n",
    "pv_vsstore_record_bytes",
    "pv_compact_draws_n",
    "rf_op_wait_reg_mem_n",
    "rf_op_cond_exec_n",
    "rf_op_draw_indirect_multi_n",
    "rf_op_draw_index_indirect_multi_n",
    "rf_op_dispatch_indirect_n",
    "rf_op_set_predication_n",
    "rf_op_event_write_eop_n",
    "rf_op_indirect_buffer_n",
    "rf_op_other_n",
    "rf_op_none_n",
    "rf_op_wait_ns",
    "rf_op_draw_index_indirect_n",
    "rf_op_draw_index_indirect_wait_ns",
    "cp_wait_cause_rf_gpu_ns",
    "cp_wait_cause_spec_ns",
    "cp_wait_cause_dcc_ns",
    "cp_wait_cause_readmem_ns",
    "cp_wait_cause_eager_ns",
    "cp_wait_cause_delete_drain_ns",
    "cp_wait_cause_drain_ns",
    "cp_wait_cause_stream_ns",
    "cp_wait_cause_desc_heap_ns",
    "cp_wait_cause_other_ns",
    "pv_ind_range_reuse_n",
    "pv_ind_range_miss_n",
    "pv_ind_range_violation_n",
    "pv_ind_range_eq_n",
    "pv_ind_range_ne_n",
    "pv_ind_range_ne_within_n",
    "pv_idx_backing_n",
    "pv_idx_backing_bytes",
    "pv_idx_backing_fallback_n",
    "pv_idx_backing_eq_n",
    "pv_idx_backing_ne_n",
};
static_assert(static_cast<uint32_t>(Counter::CpPhaseWaitNs) - static_cast<uint32_t>(Counter::CpParseNs) ==
              static_cast<uint32_t>(CpPhases::Phase::Wait));
static_assert(std::size(kCounterNames) == static_cast<size_t>(Counter::Count));
static_assert(static_cast<uint32_t>(Counter::RfOpNoneN) - static_cast<uint32_t>(Counter::RfOpWaitRegMemN) + 1 == RfOp::kSlotCount);
constexpr bool RfOpColumnsMatch() {
	for (uint32_t slot = 0; slot < RfOp::kSlotCount; slot++) {
		if (std::string_view(kCounterNames[static_cast<uint32_t>(Counter::RfOpWaitRegMemN) + slot]) != RfOp::ColumnName(slot)) {
			return false;
		}
	}
	return true;
}
static_assert(RfOpColumnsMatch(), "frames.csv rf_op columns out of step with RfOp::SlotOf");
static_assert(static_cast<uint32_t>(Counter::CpWaitCauseOtherNs) - static_cast<uint32_t>(Counter::CpWaitCauseRfGpuNs) + 1 == CpWait::kCauseCount);
constexpr bool CpWaitColumnsMatch() {
	for (uint32_t cause = 0; cause < CpWait::kCauseCount; cause++) {
		if (std::string_view(kCounterNames[static_cast<uint32_t>(Counter::CpWaitCauseRfGpuNs) + cause]) != CpWait::ColumnName(cause)) {
			return false;
		}
	}
	return true;
}
static_assert(CpWaitColumnsMatch(), "frames.csv cp_wait_cause columns out of step with CpWait::Cause");
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

// S406: production side of the CP breadcrumb sampler (cpSampler.h).
namespace Libs::Graphics::CpSampler {

bool Enabled() {
	static const bool on = FrameTiming::DiagFlagOn(std::getenv("KYTY_DIAG_CP_SAMPLE"));
	return on;
}

namespace {

void DumpHistogram(const Histogram& hist, uint64_t from, uint64_t to) {
	struct Cell {
		uint32_t site;
		size_t   op;
		uint64_t n;
		uint32_t parent; // S413: for the Wait site, the site that asked for the wait
	};
	std::vector<Cell> cells;
	for (uint32_t site = 0; site < kSiteCount; site++) {
		for (size_t op = 0; op < Histogram::kOps; op++) {
			if (site == kWaitSite) {
				for (uint32_t parent = 0; parent < kSiteCount; parent++) {
					if (const auto n = hist.WaitCell(parent, op); n != 0) cells.push_back({site, op, n, parent});
				}
			} else if (const auto n = hist.Cell(site, op); n != 0) {
				cells.push_back({site, op, n, 0});
			}
		}
	}
	std::sort(cells.begin(), cells.end(), [](const Cell& a, const Cell& b) { return a.n > b.n; });
	for (size_t i = 0; i < cells.size() && i < 80; i++) {
		char op[8];
		if (cells[i].op == Histogram::kOps - 1) {
			std::snprintf(op, sizeof(op), "-");
		} else {
			std::snprintf(op, sizeof(op), "0x%02zx", cells[i].op);
		}
		char site[64];
		if (cells[i].site == kWaitSite) {
			std::snprintf(site, sizeof(site), "wait@%s", SiteName(cells[i].parent));
		} else {
			std::snprintf(site, sizeof(site), "%s", SiteName(cells[i].site));
		}
		LOGF("[cp-sample] flips=%llu-%llu total=%llu site=%s op=%s n=%llu\n", static_cast<unsigned long long>(from),
		     static_cast<unsigned long long>(to), static_cast<unsigned long long>(hist.Total()), site, op,
		     static_cast<unsigned long long>(cells[i].n));
	}
}

// Sampler-thread state (only that thread touches it).
bool OnSample(Histogram& hist) {
	const auto window = Window();
	static uint64_t     base_total = 0, base_wait = 0, seg_from = 0;
	static bool         in_segment = false;

	const uint64_t flips  = FrameTiming::GuestFlips();
	const bool     inside = flips >= window.from && flips <= window.to;
	if (in_segment && (!inside || flips / 100 != seg_from / 100)) {
		DumpHistogram(hist, seg_from, flips);
		base_total += hist.Total();
		base_wait += hist.Wait();
		hist.Clear();
		in_segment = false;
	}
	if (!inside) return false;
	if (!in_segment) {
		seg_from   = flips;
		in_segment = true;
	}
	FrameTiming::Set(FrameTiming::Counter::CpSampleN, base_total + hist.Total());
	FrameTiming::Set(FrameTiming::Counter::CpSampleWaitN, base_wait + hist.Wait());
	return true;
}

} // namespace

void StartProduction() {
	static Sampler sampler;
	static std::once_flag once;
	std::call_once(once, [] {
		Active().store(true);
		LOGF("[cp-sample] sampler on: 10 kHz, window from KYTY_DIAG_CP_SAMPLE_WINDOW (default 6300:6800)\n");
		sampler.Start(100, OnSample);
	});
}

} // namespace Libs::Graphics::CpSampler
