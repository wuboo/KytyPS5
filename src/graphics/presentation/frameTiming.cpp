#include "graphics/presentation/frameTiming.h"
#include "kernel/eventQueue.h"
#include "graphics/presentation/cpPhases.h"
#include "graphics/presentation/cpSampler.h"
#include "graphics/presentation/cpPcSampler.h"
#include "graphics/host_gpu/hotPages.h"
#include "graphics/host_gpu/renderer/cache/imagePageSummary.h"
#include "graphics/host_gpu/renderer/rfOpDiag.h"
#include "graphics/host_gpu/renderer/cpWaitDiag.h"
#include "graphics/shader/nullDescDiag.h"

#include "common/memEvents.h"
#include "common/controlFile.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/signalCensus.h"
#include "graphics/presentation/cpFrameStats.h"
#include "graphics/shader/recompiler/ir/passes/srtTreeReject.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#include <pthread.h>
#endif
#if defined(_WIN32)
#include "graphics/presentation/threadCpuWin.h"
#endif
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
#include <unordered_map>
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
    "pv_idx_backing_inflight_n",
    "disp_indirect_backing_inflight_n",
    "cp_read_backing_n",
    "cp_read_backing_bytes",
    "cp_read_backing_fallback_n",
    "cp_read_backing_inflight_n",
    "cp_read_backing_eq_n",
    "cp_read_backing_ne_n",
    "cp_read_backing_attrib_n",
    "cp_read_backing_attrib_fallback_n",
    "cp_read_backing_buffer_n",
    "cp_read_backing_buffer_fallback_n",
    "spec_backing_inflight_n",
    "equeue_coalesced_n",
    "equeue_pending_max",
    "equeue_received_n",
    "equeue_recv_graphics_n",
    "equeue_recv_ampr_n",
    "equeue_recv_videoout_n",
    "gf_n",
    "gf_wait_ns",
    "gf_after_backing_n",
    "srt_memo_hit",
    "srt_memo_miss",
    "srt_memo_mismatch",
    "srt_pages_checked",
    "srt_page_hit",
    "gf_emu_n",
    "gf_emu_fallback_n",
    "gf_emu_eq_n",
    "gf_emu_ne_n",
    "gf_emu_vex_noavx_n",
    "gf_emu_fb_dirty_n",
    "gf_emu_fb_inflight_n",
    "gf_emu_fb_vex_n",
    "gf_emu_fb_form_n",
    "gf_emu_fb_lockbusy_n",
    "gf_emu_page_max",
    "eop_irq_deferred_n",
    "eop_irq_coalesced_n",
    "eop_coalesce_would_n",
    "spec_fast_hit_n",
    "spec_fast_miss_n",
    "spec_fast_stale_n",
    "spec_fast_mismatch_n",
    "tex_memo_hit_n",
    "tex_memo_miss_n",
    "tex_memo_invalidate_n",
    "tex_memo_uncacheable_n",
    "tex_memo_ne_n",
    "srt_walk_tree_n",
    "srt_walk_tree_fail_n",
    "srt_walk_capture_n",
    "srt_walk_slot_jit_n",
    "srt_walk_interp_n",
    "srt_interp_slots_n",
    "spec_read_n",
    "spec_read_xpage_n",
    "pv_layout_fast_n",
    "pv_layout_ne_n",
    "pv_validate_skipped_n",
    "pv_validate_ne_n",
    "pv_layout_fallback_n",
    "pipeline_save_n",
    "pipeline_save_ns",
    "pipeline_save_bytes",
    "pipeline_save_fail_n",
    "pipeline_save_cp_ns",
    "pipeline_save_skipped_n",
    "shader_program_miss_n",
    "shader_perm_miss_n",
    "shader_call_program_n",
    "get_graphics_programs_ns",
    "get_compute_program_ns",
    "get_graphics_pipeline_ns",
    "shader_prepare_ns",
    "shader_translate_ns",
    "shader_resource_plan_ns",
    "shader_materialize_first_ns",
    "shader_compile_perm_ns",
    "shader_emit_spirv_ns",
    "shader_validate_ns",
    "shader_module_ns",
    "shader_pv_variants_ns",
    "shader_lead_lt1ms",
    "shader_lead_lt5ms",
    "shader_lead_lt20ms",
    "shader_lead_lt50ms",
    "shader_lead_lt100ms",
    "shader_lead_lt250ms",
    "shader_lead_lt1000ms",
    "shader_lead_ge1000ms",
    "prog_disk_hit_n",
    "prog_disk_miss_n",
    "prog_disk_write_n",
    "prog_disk_bytes",
    "prog_disk_write_fail_n",
    "prog_disk_eq_n",
    "prog_disk_ne_n",
    "prog_disk_reject_n",
    "prog_disk_skip_call_n",
    "prog_disk_load_ns",
    "prewarm_queued_n",
    "prewarm_done_n",
    "prewarm_fail_n",
    "prewarm_hit_n",
    "prewarm_wait_n",
    "prewarm_wait_ns",
    "prewarm_cancel_n",
    "prewarm_timeout_n",
    "prewarm_mismatch_n",
    "prewarm_skip_n",
    "prewarm_manifest_n",
    "prewarm_ne_n",
    "prewarm_build_ns",
    "prewarm_ready_peak",
    "prewarm_evict_n",
    "cp_thread_cpu_ns",
    "cp_thread_wall_ns",
    "cp_walk_cpu_ns",
    "cp_walk_wall_ns",
    "walk_n",
    "walk_read_n",
    "walk_read_bytes",
    "walk_pages_n",
    "walk_interp_call_n",
    "walk_jit_slot_call_n",
    "walk_jit_tree_call_n",
    "cp_pc_sample_n",
    "cp_pc_fail_n",
    "wf_n",
    "wf_handler_ns",
    "wf_pages_distinct",
    "wf_pages_every_flip",
    "wf_hot_pages_n",
    "wf_hot_promote_n",
    "wf_hot_demote_n",
    "wf_hot_hash_n",
    "wf_hot_hash_ns",
    "wf_hot_skip_n",
    "wf_hot_changed_n",
    "wf_hot_ne_n",
    "wf_hot_race_n",
    "wf_flip_refault_n",
    "wf_flip_promote_n",
    "wf_flip_demote_n",
    "wf_flip_keep_n",
    "wf_flip_pages_n",
    "wf_flip_cand_n",
    "wf_flip_hash_n",
    "wf_flip_ne_n",
    "wf_flip_ne_unstable_n",
    "wf_fast_n",
    "wf_slow_n",
    "wf_fast_ne_n",
    "wf_flip_sticky_refault_n",
    "prewarm_deferred_n",
    "prewarm_ready_bytes",
    "prewarm_ready_bytes_peak",
    "cp_frame_n",
    "cp_frame_ns",
    "cp_frame_max_ns",
    "guest_lead_n",
    "guest_lead_min_ns",
    "readback_wait_cp_ns",
    "readback_wait_other_ns",
    "thread_cpu_cp_ns",
    "thread_cpu_other_ns",
    "thread_cpu_n",
    "srt_cover_noreads_n",
    "srt_cover_cb_n",
    "srt_cover_capture_n",
    "srt_jit_cover_ne_n",
    "srt_batch_runs_n",
    "srt_batch_slots_n",
    "srt_batch_single_n",
    "srt_batch_fail_n",
    "srt_batch_mismatch_n",
    "gpu_label_n",
    "gpu_label_ns",
    "gfp_vs_prep_ns",
    "gfp_ps_prep_ns",
    "gfp_key_ns",
    "gfp_runtime_ns",
    "gfp_find_ns",
    "gfp_get_n",
    "gfp_get_hit_n",
    "gfp_vs_get_ns",
    "gfp_ps_get_ns",
    "gfp_memo_hit_n",
    "gfp_memo_miss_n",
    "gfp_memo_entries_n",
    "gfp_memo_ne_n",
    "gfp_mat_walk_ns",
    "gfp_mat_total_ns",
    "gfp_perm_search_ns",
    "gfp_walk_tree_ns",
    "gfp_walk_cf_ns",
    "gfp_walk_slots_ns",
    "gfp_cf_walk_n",
    "gfp_cf_iter_n",
    "gfp_cf_blocks_n",
    "gfp_cf_cond_n",
    "gfp_cf_eval_n",
    "gfp_cf_rfl_n",
    "gfp_memo_skip_n",
    "gfp_memo_miss_cold_n",
    "gfp_memo_miss_inputs_n",
    "gfp_memo_miss_reads_n",
    "gfp_cf_read_n",
    "gfp_cf_read_ns",
    "bind_prep_ns",
    "bind_find_ns",
    "bind_bda_ns",
    "bind_img_ns",
    "bind_rt_ns",
    "bind_buf_ns",
    "disp_prog_ns",
    "disp_prep_ns",
    "disp_commit_ns",
    "hdl_draw_ns",
    "hdl_reg_ns",
    "hdl_nop_ns",
    "hdl_wait_ns",
    "hdl_disp_ns",
    "hdl_other_ns",
    "hdl_draw_n",
    "hdl_reg_n",
    "hdl_nop_n",
    "hdl_wait_n",
    "hdl_disp_n",
    "hdl_other_n",
    "draw_head_ns",
    "draw_state_ns",
    "draw_prolog_ns",
    "draw_vb_ns",
    "draw_preemit_ns",
    "draw_dyn_ns",
    "draw_emit_ns",
    "gfp_cf_loop_ns",
    "indirect_host_pv_store_n",
    "indirect_host_pv_proto_n",
    "indirect_host_mesh_n",
    "indirect_host_copy_depth_to_color_n",
    "indirect_host_primitive_type_n",
    "indirect_host_primitive_restart_n",
    "indirect_host_native_fetch_n",
    "indirect_host_cmd_n",
    "mesh_ind_gpu_n",
    "mesh_ind_ring_full_n",
    "mesh_ind_shadow_n",
    "mesh_ind_shadow_eq_n",
    "mesh_ind_shadow_ne_n",
    "mesh_ind_violation_n",
    "mesh_ind_cpu_n",
    "mesh_ind_cpu_empty_n",
    "mesh_ind_diag_a_n",
    "mesh_ind_diag_b_n",
    "mesh_ind_diag_c_n",
    "mesh_ind_diag_d_n",
    "mesh_ind_diag_e_n",
    "mesh_ind_diag_f_n",
    "prog_disk_adopt_ns",
    "prog_disk_read_ns",
    "prog_disk_pv_import_ns",
    "prog_disk_prefetch_hit_n",
    "prog_disk_prefetch_miss_n",
    "prog_disk_prefetch_files_n",
    "prog_disk_prefetch_bytes",
    "prog_disk_prefetch_read_ns",
    "gpu_ts_busy_ns",
    "gpu_ts_idle_ns",
    "gpu_ts_n",
    "gpu_ts_sum_ns",
    "gpu_ts_overlap_n",
    "gpu_ts_skip_n",
    "pipe_enq_n",
    "pipe_draw_n",
    "pipe_dispatch_n",
    "pipe_ordered_n",
    "pipe_drain_cls_n",
    "pipe_drain_wait_n",
    "pipe_drain_end_n",
    "pipe_full_n",
    "pipe_wait_virt_n",
    "pipe_len_mismatch_n",
    "pipe_snapshot_ns",
    "pipe_stall_ns",
    "pipe_job_n",
    "pipe_job_bail_n",
    "pipe_fallback_n",
    "pipe_worker_ns",
    "pipe_shadow_n",
    "pipe_shadow_ne_n",
    "pipe_join_dirty_n",
    "pipe_used_n",
    "pipe_exec_ns",
    "pipe_queue_ns",
    "pipe_walk_wait_ns",
    "pipe_walk_ns",
    "pipe_job_ps_ns",
    "pipe_job_vs_ns",
    "pipe_memo_hit_n",
    "pipe_memo_miss_n",
    "pipe_memo_hit_ns",
    "pipe_memo_miss_ns",
    "pipe_join_pub_n",
    "pipe_fail1_n",
    "pipe_fail2_n",
    "pipe_fail3_n",
    "pipe_fail4_n",
    "pipe_fail5_n",
    "pipe_fail6_n",
    "pipe_fail7_n",
    "pipe_fail8_n",
    "pipe_fail9_n",
    "pipe_fail10_n",
    "pipe_fail11_n",
    "srt_pa_walk_n",
    "srt_pa_read_n",
    "srt_pa_fallback_n",
    "srt_pa_shadow_ne_n",
    "srt_pa_fault_note_n",
    "land_dcc_gpu_n",
    "land_dcc_gpu_miss_n",
    "land_guest_emu_n",
    "upload_pass_break_n",
    "upload_bytes",
    "draw_state_same_n",
    "rt_same_n",
    "rb_clean_walk_n",
    "rb_clean_walk_shadow_n",
    "rb_clean_walk_shadow_ne_n",
    "defer_write_draw_n",
    "defer_write_flush_n",
    "defer_write_read_n",
    "defer_write_now_n",
    "flush_draws_n",
    "backing_nolock_read_n",
    "pipeline_skip_n",
    "size_memo_hit_n",
    "size_memo_miss_n",
    "size_memo_ne_n",
    "front_items_n",
    "front_run_ns",
    "front_full_wait_n",
    "front_full_wait_ns",
    "front_sync_n",
    "front_sync_ns",
    "front_drain_n",
    "front_drain_ns",
    "front_rec_wait_ns",
    "front_idle_cmd_n",
    "front_shadow_n",
    "front_shadow_ne_n",
    "rt_memo_hit_n",
    "rt_memo_miss_n",
    "rt_memo_ne_n",
    "exact_front_hit_n",
    "exact_front_ne_n",
    "lru_skip_n",
    "lru_skip_ne_n",
    "upload_sync_n",
    "upload_buffer_bytes",
    "upload_barrier_bytes",
    "narrow_barrier_n",
    "disp_hazard_n",
    "disp_hazard_img_n",
    "disp_access_ro_n",
    "predecode_img_n",
    "predecode_decline_n",
    "predecode_nojob_n",
    "predecode_ne_n",
    "predecode_worker_ns",
    "w2_handle_n",
    "w2_handle_other_n",
    "w2_wait_call_n",
    "w2_wait_notfree_n",
    "w2_wait_block_n",
    "w2_wait_submit_n",
    "w2_wait_rf_gpu_n",
    "w2_wait_spec_n",
    "w2_wait_dcc_n",
    "w2_wait_readmem_n",
    "w2_wait_eager_n",
    "w2_wait_delete_drain_n",
    "w2_wait_drain_n",
    "w2_wait_stream_n",
    "w2_wait_desc_heap_n",
    "w2_wait_other_n",
    "w2_destroy_unsub_n",
    "w2_destroy_deferred_n",
    "w2_destroy_immediate_n",
    "w2_destroy_other_thread_n",
    "tex_slot_memo_hit_n",
    "tex_slot_memo_miss_n",
    "tex_slot_memo_stale_n",
    "tex_slot_memo_ne_n",
    "depth_memo_hit_n",
    "depth_memo_miss_n",
    "depth_memo_ne_n",
    "ledger_memo_hit_n",
    "ledger_memo_miss_n",
    "ledger_memo_ne_n",
    "buf_touch_skip_n",
    "buf_touch_skip_ne_n",
    "emit_ops_n",
    "emit_bytes",
    "emit_drain_n",
    "emit_drain_handle_n",
    "emit_drain_conv_n",
    "emit_drain_submit_n",
    "emit_drain_wait_n",
    "emit_drain_other_n",
    "emit_record_ns",
    "emit_replay_ns",
    "emit_ne_n",
    "emit_fallback_n",
    "emit_thread_wait_ns",
    "emit_thread_wait_handle_ns",
    "emit_thread_wait_conv_ns",
    "emit_thread_wait_submit_ns",
    "emit_thread_wait_wait_ns",
    "emit_thread_wait_other_ns",
    "emit_thread_wait_full_ns",
    "emit_thread_busy_ns",
    "emit_queue_depth_max",
    "emit_queue_depth_sum",
    "emit_thread_pass_n",
    "emit_thread_seg_n",
    "emit_thread_publish_n",
    "emit_thread_steal_n",
    "emit_thread_steal_seg_n",
    "emit_thread_wake_n",
    "emit_thread_park_n",
    "thread_cpu_lite_cp_n",
    "thread_cpu_full_scan_n",
    "thread_cpu_sample_ns",
    "rf_gpu_walker_n",
    "rf_gpu_walker_ns",
    "rf_gpu_other_n",
    "rf_gpu_other_ns",
    "rf_patch_defer_n",
    "rf_patch_reject_not_patchable_n",
    "rf_patch_reject_address_base_n",
    "rf_patch_reject_not_dirty_n",
    "rf_patch_reject_inflight_n",
    "rf_patch_reject_full_n",
    "rf_patch_copy_n",
    "rf_patch_dispatch_n",
    "rf_patch_shadow_n",
    "rf_patch_ne_n",
    "rf_patch_stale_ne_n",
    "dirty_add_memo_hit_n",
    "dirty_add_memo_miss_n",
    "dirty_add_memo_ne_n",
    "dirty_query_memo_hit_n",
    "dirty_query_memo_miss_n",
    "dirty_query_memo_ne_n",
    "numinst_eager_n",
    "numinst_lazy_skip_n",
    "numinst_read_n",
    "numinst_read_ns",
    "numinst_hit_n",
    "numinst_shadow_n",
    "numinst_ne_n",
    "numinst_taint_use_n",
    "krf2_defer_n",
    "krf2_fallback_loop_n",
    "krf2_fallback_address_n",
    "krf2_fallback_descriptor_n",
    "krf2_fallback_output_n",
    "krf2_fallback_derived_flat_n",
    "krf2_fallback_slots_n",
    "krf2_fallback_phi_address_n",
    "krf2_fallback_other_n",
    "krf2_fallback_walkfail_n",
    "krf2_fallback_retry_n",
    "krf2_extra_res_n",
    "krf2_extra_wr_n",
    "krf2_extra_flat_n",
    "krf2_variant_n",
    "krf2_compile_n",
    "krf2_gpu_check_n",
    "krf3_inflight_defer_n",
    "krf3_page_ban_n",
    "krf3_page_would_ban_n",
    "krf3_page_drain_n",
    "krf3_page_trial_n",
    "krf3_guest_page_n",
    "krf3_guest_page_wait_ns",
    "gf_emu_vex_init_n",
    "gf_emu_vex_hw_n",
    "gw_spin_n",
    "gw_spin_busy_n",
    "gw_byte_wait_n",
    "gw_byte_wait_ns",
    "gw_byte_done_n",
    "gw_byte_older_n",
    "gw_byte_served_n",
    "gw_byte_refused_n",
    "gw_byte_unsubmitted_n",
    "gw_byte_shadow_ne_n",
    "diag_gs_n",
    "diag_gs_ne_n",
    "diag_gs_inflight_n",
    "diag_gs_unsettled_n",
    "diag_gs_busy_n",
    "present_n",
    "present_ns",
    "present_acquire_ns",
    "present_slot_wait_ns",
    "present_slot_wait_n",
    "present_queue_ns",
    "present_inflight_sum",
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
// S560: w2_wait_<cause>_n are in CpWait::Cause order too.
static_assert(static_cast<uint32_t>(Counter::W2WaitOtherN) - static_cast<uint32_t>(Counter::W2WaitRfGpuN) + 1 == CpWait::kCauseCount);
constexpr bool W2WaitColumnsMatch() {
	for (uint32_t cause = 0; cause < CpWait::kCauseCount; cause++) {
		const std::string_view name   = kCounterNames[static_cast<uint32_t>(Counter::W2WaitRfGpuN) + cause];
		const std::string_view column = CpWait::ColumnName(cause); // cp_wait_cause_<cause>_ns
		if (name.substr(0, 8) != "w2_wait_" || name.substr(8, name.size() - 10) != column.substr(14, column.size() - 17)) return false;
	}
	return true;
}
static_assert(W2WaitColumnsMatch(), "frames.csv w2_wait_<cause>_n columns out of step with CpWait::Cause");
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
		namespace Eq = Libs::LibKernel::EventQueue; // S421
		Set(Counter::EqueueReceivedN, Eq::ReceivedCount(Eq::ReceivedKind::All));
		Set(Counter::EqueueRecvGraphicsN, Eq::ReceivedCount(Eq::ReceivedKind::Graphics));
		Set(Counter::EqueueRecvAmprN, Eq::ReceivedCount(Eq::ReceivedKind::Ampr));
		Set(Counter::EqueueRecvVideoOutN, Eq::ReceivedCount(Eq::ReceivedKind::VideoOut));
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

// S440: the tracker's HotPages statistics (header-only, the tracker cannot reach FrameTiming) as counters.
static void PublishHotPages() {
	const auto& stats = HotPages::GetStats();
	Set(Counter::WfHotPagesN, static_cast<uint64_t>(std::max<int64_t>(0, stats.hot_pages.load(std::memory_order_relaxed))));
	Set(Counter::WfHotPromoteN, stats.promote.load(std::memory_order_relaxed));
	Set(Counter::WfHotDemoteN, stats.demote.load(std::memory_order_relaxed));
	Set(Counter::WfHotHashN, stats.hash.load(std::memory_order_relaxed));
	Set(Counter::WfHotHashNs, stats.hash_ns.load(std::memory_order_relaxed));
	Set(Counter::WfHotSkipN, stats.skip.load(std::memory_order_relaxed));
	Set(Counter::WfHotChangedN, stats.changed.load(std::memory_order_relaxed));
	Set(Counter::WfHotNeN, stats.ne.load(std::memory_order_relaxed));
	Set(Counter::WfHotRaceN, stats.race.load(std::memory_order_relaxed));
	const auto& flip = FlipUnprotect::GetStats();
	Set(Counter::WfFlipRefaultN, flip.refault.load(std::memory_order_relaxed));
	Set(Counter::WfFlipPromoteN, flip.promote.load(std::memory_order_relaxed));
	Set(Counter::WfFlipDemoteN, flip.demote.load(std::memory_order_relaxed));
	Set(Counter::WfFlipKeepN, flip.keep.load(std::memory_order_relaxed));
	Set(Counter::WfFlipPagesN, static_cast<uint64_t>(std::max<int64_t>(0, flip.pages.load(std::memory_order_relaxed))));
	Set(Counter::WfFlipCandN, flip.cand.load(std::memory_order_relaxed));
	Set(Counter::WfFlipHashN, flip.hash.load(std::memory_order_relaxed));
	Set(Counter::WfFlipNeN, flip.ne.load(std::memory_order_relaxed));
	Set(Counter::WfFlipNeUnstableN, flip.ne_unstable.load(std::memory_order_relaxed));
	const auto& fast = FastHandler::GetStats();
	Set(Counter::WfFastN, fast.fast.load(std::memory_order_relaxed));
	Set(Counter::WfSlowN, fast.slow.load(std::memory_order_relaxed));
	Set(Counter::WfFastNeN, fast.ne.load(std::memory_order_relaxed));
	Set(Counter::WfFlipStickyRefaultN, flip.sticky_refault.load(std::memory_order_relaxed));
}

// ---- S441: CP frame interval, guest lead, thread CPU ---------------------------------------------------------------------------------------------

namespace {

// Only the CP thread touches this.
struct CpFrameState {
	CpFrameStats::FrameClock clock;
	CpFrameStats::Samples    intervals;
	CpFrameStats::Samples    leads; // per-flip minimum, one per flip that had a submission
	CpFrameStats::PerFlipMin lead_min;
	uint64_t                 first_flip = 0;
} g_cp_frame;

constexpr size_t kCpFrameSegment = 100;

void ReportCpFrame(uint64_t from, uint64_t to) {
	auto& st = g_cp_frame;
	LOGF("[cp-frame] flips=%llu-%llu n=%zu mean_us=%llu p50_us=%llu p90_us=%llu p99_us=%llu max_us=%llu | lead_n=%zu lead_min_p50_us=%llu lead_min_p90_us=%llu "
	     "lead_min_p1_us=%llu\n",
	     static_cast<unsigned long long>(from), static_cast<unsigned long long>(to), st.intervals.Size(),
	     static_cast<unsigned long long>(st.intervals.Size() ? st.intervals.Sum() / st.intervals.Size() / 1000 : 0),
	     static_cast<unsigned long long>(st.intervals.Percentile(50) / 1000), static_cast<unsigned long long>(st.intervals.Percentile(90) / 1000),
	     static_cast<unsigned long long>(st.intervals.Percentile(99) / 1000), static_cast<unsigned long long>(st.intervals.Max() / 1000), st.leads.Size(),
	     static_cast<unsigned long long>(st.leads.Percentile(50) / 1000), static_cast<unsigned long long>(st.leads.Percentile(90) / 1000),
	     static_cast<unsigned long long>(st.leads.Percentile(1) / 1000));
}

// S570 T1b: KYTY_THREAD_CPU unset / anything but 0 and lite = Full (the S467 snapshot every 10th flip, inside the flip locks);
// 0 = Off; lite = the CP's own counter once per flip (NoteCpFlipPrepared) plus the other threads every 10th flip through cached handles,
// after the flip locks are released (AfterGuestFlipUnlocked), the thread list rescanned every 600th flip.
enum class ThreadCpuMode : uint8_t { Off, Full, Lite };

ThreadCpuMode ThreadCpuEnvMode() {
	static const ThreadCpuMode mode = [] {
		const char* v = std::getenv("KYTY_THREAD_CPU");
		if (v == nullptr) return ThreadCpuMode::Full;
		if (std::string_view(v) == "lite") return ThreadCpuMode::Lite;
		return v[0] != '0' ? ThreadCpuMode::Full : ThreadCpuMode::Off;
	}();
	return mode;
}

struct ThreadCpuLiteState {
	CpFrameStats::LiteCpu      cp;       // CP thread only (NoteCpFlipPrepared)
	CpFrameStats::LitePlan     plan;
	CpFrameStats::DeferredFlip deferred; // marked under the flip locks, taken after them
	std::atomic<uint64_t>      cp_handle {0};
#if defined(__APPLE__)
	std::atomic<uint32_t>      cp_port {0};
#endif
} g_thread_lite;

uint64_t LiteCpNs() {
#if defined(_WIN32)
	return ThreadCpuWin::ThreadCpuNs(reinterpret_cast<void*>(g_thread_lite.cp_handle.load(std::memory_order_acquire)));
#elif defined(__APPLE__)
	const auto port = static_cast<mach_port_t>(g_thread_lite.cp_port.load(std::memory_order_acquire));
	if (port == 0) return 0;
	thread_basic_info_data_t basic {};
	mach_msg_type_number_t   count = THREAD_BASIC_INFO_COUNT;
	if (thread_info(port, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&basic), &count) != KERN_SUCCESS) return 0;
	const auto us = [](const time_value_t& t) { return static_cast<uint64_t>(t.seconds) * 1000000ull + static_cast<uint64_t>(t.microseconds); };
	return (us(basic.user_time) + us(basic.system_time)) * 1000ull;
#else
	return 0;
#endif
}

} // namespace

void RegisterCpThread() {
	if (ThreadCpuEnvMode() != ThreadCpuMode::Lite) return;
#if defined(_WIN32)
	(void)ThreadCpuWin::NsPerCycle(); // the 50 ms calibration here, at the CP's start, not later under a flip lock
	g_thread_lite.cp_handle.store(reinterpret_cast<uint64_t>(ThreadCpuWin::OpenCurrentThread()), std::memory_order_release);
#elif defined(__APPLE__)
	g_thread_lite.cp_port.store(static_cast<uint32_t>(pthread_mach_thread_np(pthread_self())), std::memory_order_release);
#endif
}

void NoteGuestLead(uint64_t ns) {
	g_cp_frame.lead_min.Note(ns);
}

void NoteCpFlipPrepared() {
	auto& st = g_cp_frame;
	if (ThreadCpuEnvMode() == ThreadCpuMode::Lite) {
		// S570 T1b: one query of the CP thread's own counter per flip (thread_cpu_cp_ns per flip, no snapshot).
		if (const auto delta = g_thread_lite.cp.Note(LiteCpNs())) {
			Add(Counter::ThreadCpuCpNs, *delta);
			Add(Counter::ThreadCpuLiteCpN);
		}
	}
	if (const auto lead = st.lead_min.TakeFlip()) {
		Add(Counter::GuestLeadN);
		Add(Counter::GuestLeadMinNs, *lead);
		st.leads.Add(*lead);
	}
	const auto interval = st.clock.Mark(static_cast<uint64_t>(SteadyNowNs()));
	if (!interval) return;
	Add(Counter::CpFrameN);
	Add(Counter::CpFrameNs, *interval);
	Max(Counter::CpFrameMaxNs, *interval);
	st.intervals.Add(*interval);
	if (st.intervals.Size() >= kCpFrameSegment) {
		const uint64_t flips = GuestFlips();
		ReportCpFrame(st.first_flip, flips);
		st.first_flip = flips;
		st.intervals.Clear();
		st.leads.Clear();
	}
}

namespace {

#if defined(__APPLE__)
// CPU time of every thread of the process, with the thread name (pthread name: Thread_Gpu for the CP thread).
std::vector<CpFrameStats::ThreadSample> SnapshotThreads() {
	std::vector<CpFrameStats::ThreadSample> out;
	thread_act_array_t                      threads = nullptr;
	mach_msg_type_number_t                  count   = 0;
	if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) return out;
	for (mach_msg_type_number_t i = 0; i < count; i++) {
		CpFrameStats::ThreadSample sample;
		thread_identifier_info_data_t ident {};
		mach_msg_type_number_t        ident_count = THREAD_IDENTIFIER_INFO_COUNT;
		thread_basic_info_data_t      basic {};
		mach_msg_type_number_t        basic_count = THREAD_BASIC_INFO_COUNT;
		if (thread_info(threads[i], THREAD_IDENTIFIER_INFO, reinterpret_cast<thread_info_t>(&ident), &ident_count) == KERN_SUCCESS &&
		    thread_info(threads[i], THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&basic), &basic_count) == KERN_SUCCESS) {
			const auto us = [](const time_value_t& t) { return static_cast<uint64_t>(t.seconds) * 1000000ull + static_cast<uint64_t>(t.microseconds); };
			sample.id     = ident.thread_id;
			sample.cpu_ns = (us(basic.user_time) + us(basic.system_time)) * 1000ull;
			char name[64] = {};
			if (const pthread_t pt = pthread_from_mach_thread_np(threads[i]); pt != nullptr) pthread_getname_np(pt, name, sizeof(name));
			sample.name = name[0] != 0 ? name : "unnamed";
			out.push_back(std::move(sample));
		}
		mach_port_deallocate(mach_task_self(), threads[i]);
	}
	vm_deallocate(mach_task_self(), reinterpret_cast<vm_address_t>(threads), count * sizeof(thread_act_t));
	return out;
}
#endif

// The [thread-cpu] line: ms per flip of the largest threads over `samples` samples.
struct ThreadWindow {
	std::unordered_map<std::string, uint64_t> by_name;
	uint64_t                                  from = 0, n = 0;

	void Note(const std::vector<std::pair<std::string, uint64_t>>& delta, uint64_t flip, uint64_t samples) {
		for (const auto& [name, ns]: delta) by_name[name] += ns;
		if (++n < samples) return;
		// ms per flip: every sample spans (flip - from) / n flips
		const double flips = static_cast<double>(std::max<uint64_t>(1, flip - from));
		std::vector<std::pair<std::string, uint64_t>> top(by_name.begin(), by_name.end());
		std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
		std::string line;
		for (size_t i = 0; i < top.size() && i < 8; i++) {
			char item[96];
			std::snprintf(item, sizeof(item), " %s=%.2f", top[i].first.c_str(), static_cast<double>(top[i].second) / 1e6 / flips);
			line += item;
		}
#if defined(_WIN32)
		// To a file next to KYTY_FRAME_LOG (stdout is lost when the bench kills the process); flushed per line.
		if (const char* fl = std::getenv("KYTY_FRAME_LOG"); fl != nullptr && fl[0] != 0) {
			const std::string path = std::string(fl) + ".thread-cpu.txt";
			if (FILE* f = std::fopen(path.c_str(), "ab")) {
				std::fprintf(f, "[thread-cpu] flips=%llu-%llu ms_per_flip:%s\n", static_cast<unsigned long long>(from), static_cast<unsigned long long>(flip), line.c_str());
				std::fclose(f);
			}
		}
#endif
		LOGF("[thread-cpu] flips=%llu-%llu ms_per_flip:%s\n", static_cast<unsigned long long>(from), static_cast<unsigned long long>(flip), line.c_str());
		by_name.clear();
		n    = 0;
		from = flip;
	}
};

// KYTY_THREAD_CPU=0 switches the snapshots off (default on: one task_threads + 2 thread_info per thread per guest flip; on Windows one
// Toolhelp snapshot + OpenThread/QueryThreadCycleTime per thread every kThreadCpuEvery-th flip, the deltas are between snapshots so the sums stay exact).
void SampleThreadCpu(uint64_t flip) {
#if defined(__APPLE__) || defined(_WIN32)
	if (ThreadCpuEnvMode() != ThreadCpuMode::Full) return; // S570: lite samples in NoteCpFlipPrepared / AfterGuestFlipUnlocked
#if defined(_WIN32)
	constexpr uint64_t kThreadCpuEvery = 10;
	if (flip % kThreadCpuEvery != 0) return;
	constexpr uint64_t kWindowSamples = 10;
#else
	constexpr uint64_t kWindowSamples = 100;
#endif
	static std::mutex                                           mutex;
	static CpFrameStats::ThreadCpu                              tracker;
	static ThreadWindow                                         window;
	std::unique_lock                                            lock(mutex, std::try_to_lock);
	if (!lock.owns_lock()) return;
#if defined(_WIN32)
	const auto snapshot = ThreadCpuWin::SnapshotThreads();
#else
	const auto snapshot = SnapshotThreads();
#endif
	if (snapshot.empty()) return;
	const auto delta = tracker.Update(snapshot, "Thread_Gpu");
	Add(Counter::ThreadCpuCpNs, delta.cp);
	Add(Counter::ThreadCpuOtherNs, delta.other);
	Add(Counter::ThreadCpuN);
	window.Note(delta.by_name, flip, kWindowSamples);
#else
	(void)flip;
#endif
}

// S570 T1b (KYTY_THREAD_CPU=lite): the other threads, outside the flip locks (present thread). The CP's own time comes from
// NoteCpFlipPrepared, so delta.cp is dropped here (it would count the CP twice).
void SampleThreadCpuLite(uint64_t flip) {
#if defined(__APPLE__) || defined(_WIN32)
	static std::mutex              mutex;
	static CpFrameStats::ThreadCpu tracker;
	static ThreadWindow            window;
#if defined(_WIN32)
	static ThreadCpuWin::ThreadList list;
#endif
	std::unique_lock lock(mutex, std::try_to_lock);
	if (!lock.owns_lock()) return;
	const auto begin = SteadyNowNs();
#if defined(_WIN32)
	if (g_thread_lite.plan.FullScan(flip, !list.Empty())) {
		list.Rescan();
		Add(Counter::ThreadCpuFullScanN);
	}
	const auto snapshot = list.Sample();
#else
	const auto snapshot = SnapshotThreads();
#endif
	if (!snapshot.empty()) {
		const auto delta = tracker.Update(snapshot, "Thread_Gpu");
		Add(Counter::ThreadCpuOtherNs, delta.other);
		Add(Counter::ThreadCpuN);
		window.Note(delta.by_name, flip, 10);
	}
	Add(Counter::ThreadCpuSampleNs, ElapsedNs(begin, SteadyNowNs()));
#else
	(void)flip;
#endif
}

} // namespace

void AfterGuestFlipUnlocked() {
	if (ThreadCpuEnvMode() != ThreadCpuMode::Lite) return;
	if (const auto flip = g_thread_lite.deferred.Take()) SampleThreadCpuLite(*flip);
}


namespace {

// S441 DIAG (KYTY_DIAG_JIT_TREE_REJECT=1): the census of walks without the tree JIT, per 100 flips: totals per reason and the top rows.
void DumpTreeReject(uint64_t from, uint64_t to) {
	namespace TR = ShaderRecompiler::IR::SrtTreeReject;
	if (!TR::Active().load(std::memory_order_relaxed)) return;
	const auto   rows  = TR::Global().Take();
	const double flips = static_cast<double>(std::max<uint64_t>(1, to - from));
	for (const auto& [reason, n]: TR::Census::ByReason(rows)) {
		LOGF("[jit-tree-reject-sum] flips=%llu-%llu reason=%s per_flip=%.2f\n", static_cast<unsigned long long>(from), static_cast<unsigned long long>(to),
		     TR::Name(reason), static_cast<double>(n) / flips);
	}
	for (size_t i = 0; i < rows.size() && i < 10; i++) {
		LOGF("[jit-tree-reject] flips=%llu-%llu plan=%016llx reason=%s detail=%s per_flip=%.2f\n", static_cast<unsigned long long>(from),
		     static_cast<unsigned long long>(to), static_cast<unsigned long long>(rows[i].plan), TR::Name(rows[i].reason),
		     rows[i].detail.empty() ? "-" : rows[i].detail.c_str(), static_cast<double>(rows[i].n) / flips);
	}
}

} // namespace

void OnGuestFlip() {
	const auto flip = g_guest_flips.fetch_add(1, std::memory_order_relaxed) + 1;
	SampleThreadCpu(flip);
	if (ThreadCpuEnvMode() == ThreadCpuMode::Lite) g_thread_lite.deferred.Mark(flip, g_thread_lite.plan.Sample(flip)); // S570: sampled after the locks
	if (flip % 100 == 0) DumpTreeReject(flip - 100, flip);
	Common::MemEvents::SetCompletedFlips(flip);
	// S276: KYTY_DIAG_SIGCENSUS=1 dumps the RIP histogram every 600 guest flips (and at exit, see ProgramExitHandler).
	if (flip % 600 == 0 && Common::SignalCensus::HistogramEnabled()) {
		char why[32];
		std::snprintf(why, sizeof(why), "flip %llu", static_cast<unsigned long long>(flip));
		Common::SignalCensus::DumpHistogram(why, [](const char* line) { LOGF("%s", line); });
	}
	PublishHotPages();
	PollControlFile(flip);
}

void AddGpuBusy(uint64_t ns) {
	g_gpu_busy_ns.fetch_add(ns, std::memory_order_relaxed);
}

void OnQueueSubmit() {
	g_submits.fetch_add(1, std::memory_order_relaxed);
}

namespace {
thread_local bool      t_thread_mute = false;  // S479: a pipeline worker's timers would add its time to the CP's columns
thread_local uint64_t* t_thread_buf  = nullptr; // S549: the CP front thread adds here (its counters share cache lines with the recorder's)
} // namespace

void SetThreadMute(bool mute) {
	t_thread_mute = mute;
}

void SetThreadBuffered(bool buffered) {
	if (buffered && t_thread_buf == nullptr) {
		t_thread_buf = new uint64_t[static_cast<size_t>(Counter::Count)](); // one per front thread, never freed (the thread never ends)
	} else if (!buffered && t_thread_buf != nullptr) {
		FlushThreadCounters();
		delete[] t_thread_buf;
		t_thread_buf = nullptr;
	}
}

void FlushThreadCounters() {
	if (t_thread_buf == nullptr) return;
	for (size_t i = 0; i < static_cast<size_t>(Counter::Count); i++) {
		if (t_thread_buf[i] != 0) {
			g_counters[i].fetch_add(t_thread_buf[i], std::memory_order_relaxed);
			t_thread_buf[i] = 0;
		}
	}
}

void Add(Counter counter, uint64_t value) {
	if (t_thread_mute) return;
	if (t_thread_buf != nullptr) {
		t_thread_buf[static_cast<size_t>(counter)] += value;
		return;
	}
	g_counters[static_cast<size_t>(counter)].fetch_add(value, std::memory_order_relaxed);
}

void AddAlways(Counter counter, uint64_t value) {
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

// S439 DIAG: PCs of the CP thread in sh.walk, dumped with the segment of the site histogram.
Pc::PcHistogram& PcHist() {
	static Pc::PcHistogram hist;
	return hist;
}

void DumpPc(uint64_t from, uint64_t to) {
	auto& hist = PcHist();
	if (hist.Total() + hist.Fail() == 0) return;
	LOGF("[cp-pc] flips=%llu-%llu samples=%llu fail=%llu distinct=%llu jit_registered=%u jit_dropped=%u\n", static_cast<unsigned long long>(from),
	     static_cast<unsigned long long>(to), static_cast<unsigned long long>(hist.Total()), static_cast<unsigned long long>(hist.Fail()),
	     static_cast<unsigned long long>(hist.Distinct()), Pc::JitCount().load(), Pc::JitDropped().load());
	for (const auto& [pc, n]: hist.Top(40)) {
		LOGF("[cp-pc] pc=0x%llx n=%llu sym=%s\n", static_cast<unsigned long long>(pc), static_cast<unsigned long long>(n), Pc::Symbolize(pc).c_str());
	}
	for (const auto& [sym, n]: hist.TopSymbols(25)) {
		LOGF("[cp-pc-sym] n=%llu sym=%s\n", static_cast<unsigned long long>(n), sym.c_str());
	}
	hist.Clear();
}

// S439 DIAG: CPU time vs wall time of the CP thread, added per tick (100 us) to the per-flip counters; ticks that fall in sh.walk also to the walk pair.
void TickCpuWall() {
#if defined(__APPLE__)
	static Pc::CpuWall tracker;
	uint64_t           cpu = 0;
	if (!Pc::ThreadCpuNs(Pc::CpThreadPort().load(std::memory_order_relaxed), cpu)) return;
	const auto [dcpu, dwall] = tracker.Tick(cpu, static_cast<uint64_t>(FrameTiming::SteadyNowNs()));
	FrameTiming::Add(FrameTiming::Counter::CpThreadCpuNs, dcpu);
	FrameTiming::Add(FrameTiming::Counter::CpThreadWallNs, dwall);
	if (CpSite().load(std::memory_order_relaxed) == SiteId(Site::ShWalk)) {
		FrameTiming::Add(FrameTiming::Counter::CpWalkCpuNs, dcpu);
		FrameTiming::Add(FrameTiming::Counter::CpWalkWallNs, dwall);
	}
#endif
}

// Sampler-thread state (only that thread touches it).
bool OnSample(Histogram& hist) {
	const auto window = Window();
	static uint64_t     base_total = 0, base_wait = 0, seg_from = 0;
	static bool         in_segment = false;
	if (Pc::Enabled()) TickCpuWall();

	const uint64_t flips  = FrameTiming::GuestFlips();
	const bool     inside = flips >= window.from && flips <= window.to;
	if (in_segment && (!inside || flips / 100 != seg_from / 100)) {
		DumpHistogram(hist, seg_from, flips);
		DumpPc(seg_from, flips);
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
#if defined(__APPLE__)
	if (Pc::Enabled() && CpSite().load(std::memory_order_relaxed) == SiteId(Site::ShWalk)) {
		uint64_t pc = 0;
		if (Pc::ReadPc(Pc::CpThreadPort().load(std::memory_order_relaxed), pc)) {
			PcHist().Add(pc);
			FrameTiming::Add(FrameTiming::Counter::CpPcSampleN);
		} else {
			PcHist().AddFail();
			FrameTiming::Add(FrameTiming::Counter::CpPcFailN);
		}
	}
#endif
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
