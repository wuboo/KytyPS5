#ifndef KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_
#define KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_

#include "graphics/presentation/cpCounters.h"

#include <chrono>
#include <cstdint>

namespace Libs::Graphics::FrameTiming {

// Called once per successfully presented frame. Emits a Tracy frame mark (with --profile) and,
// when the KYTY_FRAME_LOG environment variable names a file, appends
// "<frame>,<host_ns>,<flips>,<gpu_busy_ns>,<submits>" to it: cumulative guest flips, cumulative
// time the GPU command thread spent processing and cumulative queue submits, so a reader can
// tell re-presented frames from real ones and see GPU-thread load and submit rate even when the
// frame rate is capped by the vblank.
void OnFramePresented();

// Called once per guest flip that reached the screen (as opposed to a re-presented frame).
void OnGuestFlip();
// S441: the CP prepares a flip (Sync::PrepareVideoOutFlip, CP thread): closes the per-flip guest-lead minimum and the CP interval; every 100 intervals logs
// `[cp-frame] ... p50 p90 p99 max`. NoteGuestLead: the CP picked up a submission that the guest enqueued `ns` earlier (CP thread).
void NoteCpFlipPrepared();
void NoteGuestLead(uint64_t ns);

// Adds time the GPU command thread spent processing a submission or command.
void AddGpuBusy(uint64_t ns);

// Called once per vkQueueSubmit of the renderer's command scheduler.
void OnQueueSubmit();

// Cumulative bench counters, logged as extra KYTY_FRAME_LOG columns (see kCounterNames).
enum class Counter : uint32_t {
	EmptySubmits,         // submits of a command buffer nothing was recorded into
	Readbacks,            // BufferCache::ReadMemory calls (each drains the GPU queue)
	ReadbackDownloads,    // ... of which downloaded GPU-modified bytes
	ReadbackWaitNs,       // time the requesting thread spent inside ReadMemory
	ReadbackReadFault,    // CPU read of a GPU-modified page (guest threads)
	ReadbackReadFaultGpu, // ... raised on the GPU thread itself (indirect args and the like)
	ReadbackWriteFault,   // CPU write to a GPU-modified page
	ReadbackDcc,          // DCC metadata read to materialize a fast clear
	WriteFaults,          // write faults on tracked pages (guest threads)
	WriteFaultsGpu,       // ... on the GPU thread (CPU-side DMA copies)
	Finishes,             // full scheduler Finish() calls
	Draws,
	Dispatches,
	DescriptorSets,
	RenderPasses,
	PipelinesCreated,
	PipelineCompileNs,
	DmaMemcpyBytes,
	BufferDeleteDrains, // buffer deletions that waited for all submitted GPU work (MoltenVK)
	BufferDeleteWaitNs,
	SubmitsFlush, // submits by path: CommandScheduler::Flush(SubmitInfo&)
	SubmitsFlushAndWait,
	SubmitsFinish,
	SubmitsWaitCurrent,      // Wait() on the tick being recorded (submits, then waits for it)
	CpFlushes,               // CommandProcessor::BufferFlush (PM4 slices, ReleaseMem, ...)
	SliceSuspends,           // PM4 submissions suspended on a wait and resumed later
	SkippedSubmits,          // empty submits skipped by the empty_submit optimization
	DescriptorSetsIdentical, // descriptor sets identical to the previous one of the same layout
	DescriptorSetsReused,    // ... reused instead of allocated and written
	ResourceMemoHits,        // shader resource materializations skipped (inputs unchanged)
	ResourceMemoMisses,
	ComputeFillsReplaced, // uniform-fill dispatches with no image target replaced by a fill
	DccKnownFromFill,     // DCC materializations that used a known fill instead of a readback
	FlushesDeferred,      // command-processor flushes batched by lazy_flush
	PrototypeDraws,       // draws through the per-vertex-prototype (unpack+capture+replay) path
	PrototypeVertices,    // sum of vertex*instance counts of those draws
	PrototypeRenderPasses, // BeginRendering calls forced by the prototype path specifically
	PrototypeCaptureCacheHits, // of PrototypeDraws, how many reused a cached capture (skipped
	                           // the compute unpack+capture stages entirely)
	ReadbackEagerHits,      // read faults served by waiting for an eager page download
	ReadbackEagerDownloads, // page downloads queued right after a GPU write (eager_readback)
	PrototypeRangeMemoHits, // record ranges of indexed prototype draws taken from the memo
	PrototypeRangeMemoMisses, // ... and the ones that had to scan the index buffer
	ProtoUnsupportedDraws, // per-vertex-prototype draws skipped: topology/restart/index count it cannot run
	MinLodDropped,         // sampled T# with MinLod above its base level bound with no clamp at all (no view feature, no remap)
	MinLodRemapped,        // ... and the ones whose MinLod became a view base level (no VK_EXT_image_view_min_lod)
	MinLodFrac,            // sampled T# binds with a fractional MinLod (MinLod % 256 != 0), which the remap rounds up
	MinLodResinfo,         // binds of an image with MinLod != 0 into a program that queries its size (RESINFO)
	// S129: guest state the host cannot honor (counting only, nothing recorded differently), see appleDrop.h.
	DropDepthClamp,        // draws with Z clip disabled by the guest (DB clamps) while the host has no depth clamp/clip control
	DropDepthBounds,       // draws with the depth bounds test enabled while the host skips it
	DropColorWriteDyn,     // draws with an attachment slot without a view while the host has no dynamic color write enable
	DropSamplerBorderTable, // samplers created with a border color from the table (approximated as transparent black)
	DropSamplerLodBias,    // samplers whose LOD bias was clamped to the device limit
	// S132: CP thread and submit time (ns, cumulative; divide by the flip delta) and dispatch classes.
	CpWaitGuestNs,         // GPU thread waiting on its condition variable for guest work (idle or blocked submission)
	CpBusyNs,              // GPU thread processing a submission or command (same span as gpu_busy_ns)
	CpProtoNs,             // ... of which inside ExecutePerVertexPrototype
	CpWaitGpuNs,           // GPU thread blocked in MasterSemaphore::Wait (nested waits inside readbacks count again there)
	SubmitEncodeNs,        // time inside vkQueueSubmit (either thread)
	SubmitQueueWaitNs,     // async_submit: from push of a job to its pop by the SubmitThread
	DispGuest, DispProtoUnpack, DispProtoCapture, DispProtoCaptureNonNative, DispTiler, DispFault, DispOther,
	ProtoSkippedDraws,     // KYTY_DIAG_PROTO_SKIP=1: supported prototype draws not drawn
	// S134 (KYTY_PV_UNIQUE=1): capture per unique vertex.
	PrototypeUniqueDraws,    // prototype draws that took the unique path
	PrototypeUniqueVertices, // ... capture invocations of those draws (unique vertices * instances)
	PrototypeUniqueFallback, // prototype draws that stayed on the per-corner path with the switch on
	PvVsStoreDraws,   // S149 (KYTY_PV_VSSTORE=1): draws that took the VS-store path instead of the prototype
	// S150: per-vertex draws that stayed on the prototype with the switch on, by reason.
	PvFallbackInstanced, // instance_count > 1 and the region / range / ring did not fit
	PvFallbackRange,     // vertex range unknown, dropped corner, empty, record math too large
	PvFallbackTopology,  // not a triangle list, not indexed, primitive restart, 8-bit indices
	PvFallbackParam31,   // VS or PS uses host location 31 (carries the instance index)
	PvFallbackRing,      // no cap slab free (all in flight on the GPU)
	PvFallbackInterface, // the PS reads a per-vertex location the VS does not write
	PvFallbackIndexTail, // S151: the index buffer ends inside the last word the PS reads (descriptor would pass the buffer end)
	PvVsStoreInstanced,  // S151: VS-store draws with instance_count > 1
	PvPushNonzeroPrograms, // S384: capture / VS-store variants emitted with push data starting at a nonzero dword
	PvVsStoreOob,          // S388 (KYTY_PV_VSSTORE_GUARD=1): VS invocations whose record lay outside the draw's range (read 2 flips late)
	PvVsStoreIdxGpu,       // S388: VS-store candidates whose index bytes are GPU-modified guest memory (both KYTY_PV_VSSTORE_GPUIDX modes)
	DepthRoElided,              // S316: KYTY_DEPTH_RO_UNION=1|shadow: barriers skipped (shadow: that would have been skipped) on a read-only depth attachment
	DepthRoRejectDirty,         // S316: ... rejected: the image was written since the last barrier
	DepthRoRejectNotSubset,     // S316: ... rejected: the new access is not within the visible union
	DepthRoRejectLayout,        // S316: ... rejected: layout changes or is not read-only
	DepthRoBadElide,            // S316: ... elision of a dirty image or in a writable layout (MUST stay 0)
	// S321 (KYTY_DIAG_GPU_PAD_AT=idle): fixed-length pads sent right after a blocking CP wait, at most one per flip.
	// S323 (KYTY_DIAG_CP_PHASES=1): self ns of each CP phase (cpPhases.h); phases + cp_phase_wait_ns = outer span <= cp_busy.
	CpParseNs,
	CpHandlerNs,
	CpDrawShadersNs,
	CpDrawBindNs,
	CpDrawPipelineNs,
	CpDrawTargetsNs,
	CpDrawDescNs,
	CpDrawVkcmdNs,
	CpPvstoreNs,
	CpDispatchNs,
	CpEopNs,
	CpBarrierNs,
	CpReadbackCpuNs,
	CpGcNs,
	CpSubmitCpuNs,
	CpCmdsNs,
	CpPhaseWaitNs,        // blocked in MasterSemaphore::Wait while a phase was open (carved out of the phases)
	Pm4Packets,           // PM4 packets walked by the CP
	MprotectN,            // ProtectGuestHostMemory calls on the CP thread
	MprotectNs,           // ... time inside them
	RbDownloadBytes,      // bytes DownloadBufferMemory copied back to guest memory
	PvVsstoreIndexBytes,  // index bytes hashed by TryPvVsStoreDraw
	// S326: segments = from the end of a drain (Wait on the open tick / FlushAndWait / Finish) to the next drain.
	SegN,
	SegDrainToFirstSubmitNs, // sum: end of the drain -> first submit with work
	SegRecordNs,             // sum: CP time in the segment outside waiting (segment length minus cp_wait_guest/gpu deltas)
	GpuGapAfterDrainNs,      // KYTY_DIAG_GPUTRUTH=1: end of the last CB before a drain -> start of the first CB after it (GPU timestamps)
	EagerSubmitN,            // KYTY_EAGER_SUBMIT=1: command buffers split by the eager policy at a draw / dispatch
	// S327
	GpuGapN,                 // KYTY_DIAG_GPUTRUTH=1: drain gaps counted (gpu_gap_after_drain_ns is their sum)
	GpuGapMidNs,             // KYTY_DIAG_GPUTRUTH=1: GPU idle between two buffers with no drain between them
	EagerSubmitPassN,        // KYTY_EAGER_SUBMIT=1: command buffers split at a pass boundary (the old pass was closing anyway)
	SegFirstPassEndNs,       // sum: end of the drain -> end of the first render pass of the segment
	// S328
	EagerMidN,               // KYTY_EAGER_SUBMIT=2: buffers split in the middle of an open pass (first buffer after a drain)
	FirstSubmitN,            // first submits with work after a drain (the denominator of first_submit_enq_to_vk_ns / first_submit_encode_ns)
	FirstSubmitEnqToVkNs,    // sum: CP hands the first buffer over -> the submit thread calls vkQueueSubmit
	FirstSubmitEncodeNs,     // sum: vkQueueSubmit (MoltenVK encode) of the first buffer
	DrainWakeLagN,           // KYTY_DIAG_GPUTRUTH=1: drain waits whose wake lag was measured
	DrainWakeLagNs,          // sum: end of the buffer on the GPU (timestamp) -> the CP returns from Wait; minus the smallest lag seen (see GpuGap::ClockMap)
	FirstSubmitVkToGpuBeginN,  // KYTY_DIAG_GPUTRUTH=1: first buffers whose GPU start was measured
	FirstSubmitVkToGpuBeginNs, // sum: vkQueueSubmit returned -> start of the buffer on the GPU (timestamp, same clock estimate)
	DescAssertCachedN,
	DescAssertRunN, DescAssertSkipN, DescAssertReadsN,
	DescAssertNs,
	// S398 (KYTY_DIAG_SYNC=1): readbacks by call site (SyncCensus::Site order: dcc, rf_gpu, rf_guest, write_fault, spec, other).
	SyncDccN, SyncRfGpuN, SyncRfGuestN, SyncWriteFaultN, SyncSpecN, SyncOtherN,
	SyncDccNs, SyncRfGpuNs, SyncRfGuestNs, SyncWriteFaultNs, SyncSpecNs, SyncOtherNs,
	SyncEagerHitDccN, SyncEagerHitRfGpuN, SyncEagerHitRfGuestN, SyncEagerHitWriteFaultN, SyncEagerHitSpecN, SyncEagerHitOtherN,
	SyncLag0N,          // downloads whose newest writer sat in the buffer being recorded (nothing to wait for but the submit)
	SyncLagSum,         // sum of the writer lag in ticks over downloads
	SyncWriterDoneN,    // downloads whose newest writer had already completed on the GPU
	SyncWriterDoneNs,   // ... and the time those readbacks spent in ReadMemory (what a read-after-writer would save)
	SyncDccRepeatN,     // DCC metadata ranges read again in the same flip
	SyncDccRepeatSameN, // ... of which the bytes the CPU saw were identical to the previous read
	// S398 (KYTY_DIAG_INDIRECT=1): DrawIndirectMulti calls.
	IndirectN, IndirectDirtyN, IndirectDirtyWaitNs, IndirectRegsPatchedN, IndirectPvN, IndirectArgsChangedN,
	// S398: largest color target of the current flip.
	RtMaxW, RtMaxH,
	// S402 (KYTY_DCC_KNOWN=1|shadow): known-fill ranges handed to the guest without a drain; shadow compare of the prediction.
	DccPublishN,             // pieces published (KNOWN=1)
	DccPublishFallbackN,     // ranges whose GPU-modified content was not one known fill (old path)
	DccKnownWouldHitN,       // shadow: readbacks of DCC ranges the ledger predicted as one known fill
	DccKnownMismatchN,       // shadow: ... of which the downloaded bytes differ from the prediction
	// S402 (KYTY_DCC_EAGER=1): multi-page eager readbacks of DCC metadata.
	DccEagerHitN,            // DCC readbacks served by queued eager downloads
	DccEagerMissN,           // DCC readbacks that wanted eager and took the synchronous path
	EagerPagesN,             // gauge: hot tracker pages tracked for eager readback
	// S404 (KYTY_DIAG_SYNC=1): spec / rf_gpu readbacks of <= 64 bytes that drained the queue: the bytes the CPU would have seen
	// before the drain equal (Eq) or differ from (Ne) the bytes after it.
	SyncStaleEqSpecN, SyncStaleNeSpecN, SyncStaleEqRfGpuN, SyncStaleNeRfGpuN,
	SyncEagerAlreadyDoneN,   // rf_guest eager waits whose download tick had already completed
	SyncEagerLagSum,         // ... sum of CurrentTick - eager tick at the fault
	SyncEagerWaitNs,         // ... time of the eager Wait itself
	SyncQueueNs,             // sum over drains: SendCommandSync sent -> lambda starts on the GPU thread
	SyncLambdaNs,            // sum over drains: the lambda
	// S404: why a piece took the old path in PlanKnownFill (DccFillPublish::Why order), counted always.
	DccFallbackNoFillN, DccFallbackPartialN, DccFallbackNewerWriteN, DccFallbackPendingN, DccFallbackOtherN,
	// S404 (KYTY_INDIRECT_GPU=shadow|1): indexed indirect draws seen / qualified by the command processor / qualified by the executor too
	// (shadow) / recorded as vkCmdDrawIndexedIndirect (1) / taken on the CPU although the mode was 1 / whose index count exceeded
	// INDEX_BUFFER_SIZE (the CPU path clamps it, the GPU path cannot).
	IndirectGpuCandidateN, IndirectGpuEligibleN, IndirectGpuHostEligibleN, IndirectGpuN, IndirectGpuFallbackN, IndirectGpuClampRiskN,
	// S406 (KYTY_DIAG_CP_SAMPLE=1): samples taken in the flip window, and the ones that found the CP thread in a wait.
	CpSampleN, CpSampleWaitN,
	// S406 (KYTY_DIAG_SYNC=1): target of a spec readback (SyncCensus::SpecPurpose order).
	SyncSpecPurposeCondN, SyncSpecPurposeSlotN, SyncSpecPurposeDescN, SyncSpecPurposeFillN, SyncSpecPurposeWgN, SyncSpecPurposeCallN,
	SyncSpecPurposeOtherN,
	// S406: shadow of the decision on the spec hit path (stale reader vs fresh; superset reader vs fresh).
	SpecDecStaleEqN, SpecDecStaleNeN, SpecDecSuperOkN, SpecDecSuperFailN, SpecSuperExtraSourcesN, SpecPermSameN, SpecSuperNewImagesN,
	// S406 (KYTY_DIAG_INDIRECT=1): pv_store indirect draws: args stale vs fresh (20 B) and the largest instance count.
	IndirectPvArgsEqN, IndirectPvArgsNeN, IndirectPvInstMax,
	// S406 (KYTY_DIAG_MATMEMO=1): shadow memo after the walk, by stage (vs, ps, cs): lookups, input hits, unsafe hits
	// (same input, different output: MUST stay 0), output hits (same output as the entry), rebase hits (same after masking buffer bases).
	MatMemoVsN, MatMemoPsN, MatMemoCsN, MatMemoInHitVsN, MatMemoInHitPsN, MatMemoInHitCsN, MatMemoUnsafeVsN, MatMemoUnsafePsN,
	MatMemoUnsafeCsN, MatMemoOutHitVsN, MatMemoOutHitPsN, MatMemoOutHitCsN, MatMemoRebaseVsN, MatMemoRebasePsN, MatMemoRebaseCsN,
	// S406 (KYTY_DIAG_GPU_DONE=1): estimated GPU busy time from timeline completions (sum over submits of t_done - max(t_submit, t_done_prev)).
	GpuActiveEstNs,
	// S406 (KYTY_PV_RANGE_GEN=1|shadow): VS-store range memo by page generation.
	PvGenHitN, PvGenMissN, PvGenMismatchN, PvHashBytesSkipped,
	// ... draws that could not use the generation at all: host copy of the indices / range not registered in the tracker / range CPU- or
	// GPU-modified (dirty) at the time of the draw. Hit + miss + these three = VS-store scans seen with the switch on.
	PvGenIneligibleHostN, PvGenIneligibleUnregN, PvGenIneligibleDirtyN,
	// S408 (KYTY_SPEC_COND=superset): condition reads of GPU-dirty bytes failed without a drain (the walker follows both sides) /
	// ... that would have been failed but belong to a program being compiled (old read).
	SpecCondSupersetN, SpecCondSupersetSkipMissN,
	// S408 (KYTY_DIAG_SYNC=1, SPEC_COND=0): what the superset shadow adds over the fresh run. Permutation of the superset
	// specialization is the one of the fresh run / the specialization is equal; sources active only in the superset by kind;
	// ... of which the buffer is written by the shader / has an address outside guest memory; dirty reads of purposes other than cond
	// that the superset run makes and the stale run does not (new drains in the added blocks).
	SpecSuperPermSameN, SpecSuperSpecEqN, SpecSuperExtraBufferN, SpecSuperExtraImageN, SpecSuperExtraSamplerN,
	SpecSuperExtraWrittenN, SpecSuperExtraBadAddrN, SpecSuperDirtyNoncondN,
	// S408 (KYTY_PV_INDIRECT=1|shadow): draws recorded from prepared args / first calls of a candidate that went to the CPU read because the
	// args address was not learned yet or was cooling down after a mismatch / draws that were declined after the decision (per reason) /
	// passes closed for the prepare compute (a pass was open) / mismatches and overflows the GPU reported (status, two flips late) /
	// gauges: cap of the last prepared entry, largest instance count seen.
	PvIndN, PvIndLearnN, PvIndCooldownN, PvIndDeclineN, PvIndDeclineRangeN, PvIndDeclineRingN, PvIndDeclineIndexTailN,
	PvIndDeclineTooLargeN, PvIndDeclineOffN, PvIndDeclineResidualN, PvIndDeclineOtherN, PvIndPassBreakN, PvIndMismatchN, PvIndOverflowN,
	PvIndCap, PvIndMaxInstance,
	// ... shadow: prepared draws recorded next to the old draw / the prepared args equal (ne) the args the CPU read in that flip /
	// fields as expected (mismatch) / draws over cap / the plan made with cap equals (ne) the real plan / shadow draws that did not use
	// the VS-store / flips with more than one draw of the entry (no args comparison).
	PvIndShadowN, PvIndShadowArgsEqN, PvIndShadowArgsNeN, PvIndShadowFieldsOkN, PvIndShadowFieldsMismatchN, PvIndShadowOverflowN,
	PvIndShadowPlanEqN, PvIndShadowPlanNeN, PvIndShadowNoStoreN, PvIndShadowMultiN,
	// S410 (KYTY_DIAG_STARVE=1, needs KYTY_DIAG_GPU_DONE=1): GPU queue empty before a buffer was submitted: sum of max(0, t_submit(v) - t_done(v-1)),
	// the same without the part before the CP recorded anything for v (work), buffers with starve >= 20 us / >= 1 ms, and `work` split by why
	// the buffer was submitted (order of GpuDone::SubmitReason).
	GpuStarveNs, GpuStarveWorkNs, GpuStarveN, GpuStarveGe1msN,
	GpuStarveWorkDrainNs, GpuStarveWorkLazyMaxNs, GpuStarveWorkLazy1msNs, GpuStarveWorkLazyIdleNs, GpuStarveWorkEagerNs, GpuStarveWorkEagerPassNs, GpuStarveWorkEagerMidNs, GpuStarveWorkEager3MidNs, GpuStarveWorkEager3FlushNs, GpuStarveWorkLowwaterNs, GpuStarveWorkFlipNs, GpuStarveWorkOtherNs,
	// S410 (KYTY_EAGER_SUBMIT=3): mid-pass splits in any buffer / RequestFlush submits with an idle GPU / splits while the GPU ran the last
	// submitted buffer (KYTY_EAGER_LOWWATER=1).
	Eager3MidN, Eager3FlushN, EagerLowwaterN,
	// S410 (KYTY_DIAG_STARVE=1): why the eager policy did not split although KnownGpuTick said idle (no extra Refresh).
	EagerDeclinePassOpenN, EagerDeclineAgeN, EagerDeclineNoWorkN, EagerDeclineMidLimitN, EagerDeclineFirstCbOnlyN,
	// S409 (both KYTY_PV_VSSTORE_COMPACT modes): bytes of the VS-store records bound by the draws of the flip (records * stride * 16) /
	// draws that used the compact record (stride = locations the PS reads).
	PvVsstoreRecordBytes, PvCompactDrawsN,
	// S412 (KYTY_DIAG_RFOP=1): page read faults on the GPU thread by the PM4 opcode being handled (slots of RfOp::SlotOf, in order:
	// wait_reg_mem, cond_exec, draw_indirect_multi, draw_index_indirect_multi, dispatch_indirect, set_predication, event_write_eop,
	// indirect_buffer, other opcode, none = outside a handler).
	RfOpWaitRegMemN, RfOpCondExecN, RfOpDrawIndirectMultiN, RfOpDrawIndexIndirectMultiN, RfOpDispatchIndirectN, RfOpSetPredicationN,
	RfOpEventWriteEopN, RfOpIndirectBufferN, RfOpOtherN, RfOpNoneN,
	// S413 (KYTY_DIAG_RFOP=1): time of the drains inside the faults (all opcodes), and the faults / drain time of IT_DRAW_INDEX_INDIRECT (0x25).
	RfOpWaitNs, RfOpDrawIndexIndirectN, RfOpDrawIndexIndirectWaitNs,
	// S413 (KYTY_DIAG_CP_WAIT=1): ns the CP thread waited in MasterSemaphore::Wait by cause (order of CpWait::Cause).
	CpWaitCauseRfGpuNs, CpWaitCauseSpecNs, CpWaitCauseDccNs, CpWaitCauseReadMemNs, CpWaitCauseEagerNs, CpWaitCauseDeleteDrainNs,
	CpWaitCauseDrainNs, CpWaitCauseStreamNs, CpWaitCauseDescHeapNs, CpWaitCauseOtherNs,
	// S413 (KYTY_PV_IND_RANGE): On draws that took the remembered vertex range of GPU-dirty indices (no hash, no scan) / found none for their key /
	// draws the prepare compute zeroed for an index outside the range; shadow: the CPU scan equal / different from the remembered range, and the
	// different ones that lie inside it (the GPU check would not have fired).
	PvIndRangeReuseN, PvIndRangeMissN, PvIndRangeViolationN, PvIndRangeEqN, PvIndRangeNeN, PvIndRangeNeWithinN,
	// S417 (KYTY_PV_IDX_BACKING): draws whose indices were copied from the backing (and the bytes), clean-looking draws that stayed on the guest read
	// (GPU-dirty bytes or an unreadable backing); shadow: the copy equal / different from the old read.
	PvIdxBackingN, PvIdxBackingBytes, PvIdxBackingFallbackN, PvIdxBackingEqN, PvIdxBackingNeN,
	// S418: backing reads refused because a recorded GPU->host copy of the bytes had not been published yet (tracker already clear):
	// F3 index bytes, the zero-args dispatch skip of DispatchIndirect.
	PvIdxBackingInflightN, DispIndirectBackingInflightN,
	// S419 (KYTY_CP_READ_BACKING): CP table reads copied from the backing (and the bytes), reads that stayed on the guest read (GPU-dirty / in-flight /
	// unreadable) of which in flight; shadow: the copy equal / different from the old read. Per site (CpReadBacking::Site order): copied / fallback.
	// SpecBackingInflightN: shader-spec backing reads (TryReadBufferBacking) that found a GPU->host copy of the range not published yet.
	CpReadBackingN, CpReadBackingBytes, CpReadBackingFallbackN, CpReadBackingInflightN, CpReadBackingEqN, CpReadBackingNeN,
	CpReadBackingAttribN, CpReadBackingAttribFallbackN, CpReadBackingBufferN, CpReadBackingBufferFallbackN, SpecBackingInflightN,
	EqueueCoalescedN, // S420: GRAPHICS equeue triggers merged into the last pending entry (queue at its cap)
	EqueuePendingMax, // S420: largest pending_events size seen on an interrupt event
	// S421: events handed to the guest by KernelWaitEqueue (all / GRAPHICS / AMPR / VIDEO_OUT); set from the kernel's own atomics once per presented frame.
	EqueueReceivedN, EqueueRecvGraphicsN, EqueueRecvAmprN, EqueueRecvVideoOutN,
	// S421 (KYTY_DIAG_GUEST_FAULT=1): read faults of non-CP (guest) threads, ns they waited in ReadMemory, of those the page the CP had read from the backing in the same flip.
	GfN, GfWaitNs, GfAfterBackingN,
	// S425 (KYTY_SRT_FAST=1|shadow): MaterializeResources memo; pages whose sync state was queried / reads served from a quiet page.
	SrtMemoHit,
	SrtMemoMiss,
	SrtMemoMismatch,      // shadow: the memo would have reused a snapshot the walk did not reproduce
	SrtPagesChecked,
	SrtPageHit,
	// S426 (KYTY_GF_EMULATE=1|shadow): guest read faults served by executing the load on the backing / handed to the ordinary drain (unsupported form,
	// dirty or in-flight bytes); shadow: the emulated bytes equal / different from the raw read after the drain.
	GfEmuN, GfEmuFallbackN, GfEmuEqN, GfEmuNeN, GfEmuVexNoAvxN,
	// S428: why a fault was handed to the ordinary drain (they add up to gf_emu_fallback_n): GPU-dirty bytes, a GPU->host copy in flight, VEX.L = 1 / EVEX,
	// any other form / address the decoder refuses, the buffer cache lock held by the command processor; and the most emulated loads on one 4 KiB
	// page in one flip so far (a running maximum).
	GfEmuFbDirtyN, GfEmuFbInflightN, GfEmuFbVexN, GfEmuFbFormN, GfEmuFbLockbusyN, GfEmuPageMax,
	// S429 (KYTY_EOP_COALESCE=0|1|shadow): EOP interrupts deferred as priority operations / dropped because the same (event, context) was already deferred in
	// the tick; shadow: how many would have been dropped.
	EopIrqDeferredN, EopIrqCoalescedN, EopCoalesceWouldN,
	// S430 DIAG (only with KYTY_DIAG_CP_SAMPLE / KYTY_DIAG_CP_PHASES): SrtWalker::RefreshFlatBuffer paths (srtWalkPaths.h) and strict spec reads
	// (ReadShaderGuestMemory calls / of those spanning a 4 KiB page).
	// S430 (KYTY_SPEC_READ_FAST=1|shadow): strict reads served from a remembered quiet page / that had to ask (and maybe drain); shadow: hits whose range
	// is GPU-dirty or in flight now (stale, must be 0) / reads whose ok or bytes differ from the full read (mismatch, must be 0).
	SpecFastHitN, SpecFastMissN, SpecFastStaleN, SpecFastMismatchN,
	// S438 (KYTY_TEX_RESOLVE_MEMO=1|shadow), per ResolveTexture of a non-null texture: remembered image used (shadow: could be used) / nothing remembered /
	// remembered but a registration or rebind made it stale / the full resolve gave a result that cannot be remembered / shadow only: memo and full resolve
	// differ (must be 0).
	TexMemoHitN, TexMemoMissN, TexMemoInvalidateN, TexMemoUncacheableN, TexMemoNeN,
	SrtWalkTreeN, SrtWalkTreeFailN, SrtWalkCaptureN, SrtWalkSlotJitN, SrtWalkInterpN, SrtInterpSlotsN, SpecReadN, SpecReadXpageN,
	// S432 (KYTY_PV_LAYOUT_FAST=1|shadow): vertex layouts taken from the SPIR-V words / Validate calls skipped on the capture and VS-store variants;
	// shadow: layouts that differ from the disassembly route (must be 0) / variants Validate rejected (must be 0).
	PvLayoutFastN, PvLayoutNeN, PvValidateSkippedN, PvValidateNeN, PvLayoutFallbackN,
	// S433 (S440): PipelineCache::Save - jobs, time of the jobs (wherever they ran), bytes written, failures, and the time the command processor
	// thread itself spent in Save (all of it with KYTY_PIPELINE_SAVE_ASYNC=0, only the hand-over with 1).
	PipelineSaveN, PipelineSaveNs, PipelineSaveBytes, PipelineSaveFailN, PipelineSaveCpNs, PipelineSaveSkippedN,
	// S433 (S441): cost of the first use of a shader on the CP thread, per flip. Misses: program (new ProgramKey) / permutation (new specialization or
	// push start) / of those, programs with a retained call source. Ns: GetGraphicsPrograms and GetComputeProgram as a whole (hits included),
	// GetGraphicsPipeline as a whole (includes an inline Save), then the pieces of a miss: PrepareShaderSource, TranslateProgram, ExtractResourcePlan,
	// the first MaterializeResources of a program, CompilePermutation as a whole and inside it CompileProgram (the SPIR-V emission), Validate,
	// vkCreateShaderModule and RememberPerVertexVariants. Lead: shaders by time from ShaderMapUserData to the first lookup (see shaderCost.h).
	ShaderProgramMissN, ShaderPermMissN, ShaderCallProgramN,
	GetGraphicsProgramsNs, GetComputeProgramNs, GetGraphicsPipelineNs,
	ShaderPrepareNs, ShaderTranslateNs, ShaderResourcePlanNs, ShaderMaterializeFirstNs,
	ShaderCompilePermNs, ShaderEmitSpirvNs, ShaderValidateNs, ShaderModuleNs, ShaderPvVariantsNs,
	ShaderLeadB0, ShaderLeadB1, ShaderLeadB2, ShaderLeadB3, ShaderLeadB4, ShaderLeadB5, ShaderLeadB6, ShaderLeadB7,
	// S434 (KYTY_PROGRAM_DISK_CACHE): permutations answered from a file (hit; with 1 the compile is skipped) / looked up and not found (miss) /
	// files written (the writer thread), bytes of them, writes that failed / verify: stored file equal to the fresh compile, different (must be 0),
	// files refused (damaged, other key, does not answer the request) / lookups skipped because the program has a call source / time of the lookups.
	ProgDiskHitN, ProgDiskMissN, ProgDiskWriteN, ProgDiskBytes, ProgDiskWriteFailN, ProgDiskEqN, ProgDiskNeN, ProgDiskRejectN, ProgDiskSkipCallN,
	ProgDiskLoadNs,
	// S435 (KYTY_SHADER_PREWARM): jobs queued from the manifest / finished by the worker (with a result or a failure) / failures / results CP took
	// finished (hit) or after waiting (wait) and the time it waited / queued jobs CP cancelled / waits that ran out / results refused (key or options
	// differ) / programs not pre-warmable (call source, back code) / manifest entries / shadow: SPIR-V of the pre-warmed translation differs (must be 0) /
	// worker time in the builds / most finished results waiting to be taken at once (the worker stops at 64: each holds a whole program).
	PrewarmQueuedN, PrewarmDoneN, PrewarmFailN, PrewarmHitN, PrewarmWaitN, PrewarmWaitNs, PrewarmCancelN, PrewarmTimeoutN, PrewarmMismatchN,
	PrewarmSkipN, PrewarmManifestN, PrewarmNeN, PrewarmBuildNs, PrewarmReadyPeak, PrewarmEvictN,
	// S439 DIAG (KYTY_DIAG_CP_SAMPLE + KYTY_DIAG_CP_PC): CPU time vs wall time of the CP thread per flip (all ticks / ticks in the sh.walk site), and the
	// guest reads of the walks: walks, JIT slot reads counted from the address list (reads, bytes, distinct 4 KiB pages per walk), walks that
	// entered the interpreter / per-slot JIT / tree JIT (calls), PC samples taken in sh.walk / failed.
	CpThreadCpuNs, CpThreadWallNs, CpWalkCpuNs, CpWalkWallNs,
	WalkN, WalkReadN, WalkReadBytes, WalkPagesN, WalkInterpCallN, WalkJitSlotCallN, WalkJitTreeCallN, CpPcSampleN, CpPcFailN,
	// S440 DIAG (KYTY_DIAG_WRITE_FAULT): write faults of the guest / time of the handler's write branch / pages that faulted in the last finished flip /
	// pages that faulted in >= 90 % of the last 60 flips (the last two are gauges).
	WfN, WfHandlerNs, WfPagesDistinct, WfPagesEveryFlip,
	// S440 (KYTY_WF_HOT_UNPROTECT=1|shadow), tracker pages: hot now (gauge) / became hot / left the hot state; page hashes taken and their time; dirty hot
	// pages whose hash was unchanged (no upload; shadow: would be skipped) / changed (uploaded); shadow: clean hot pages whose hash differs from their last
	// upload (a write the tracker never saw, must be 0); pages that changed between the last hash and the return of the protection (uploaded after all).
	WfHotPagesN, WfHotPromoteN, WfHotDemoteN, WfHotHashN, WfHotHashNs, WfHotSkipN, WfHotChangedN, WfHotNeN, WfHotRaceN,
	// S443 (KYTY_WF_FLIP_UNPROTECT=1|shadow): second and later write faults of a page within one flip / pages kept unprotected for their flip / kept pages protected
	// again / uploads that left a kept page unprotected / kept now (gauge) / shadow: candidates, page hashes, unseen writes (stable = must be 0, unstable).
	// (KYTY_WF_FAST_HANDLER=1|shadow): write faults that skipped the texture-cache lock / that took it / shadow: image found after the filter said none (must be 0).
	WfFlipRefaultN, WfFlipPromoteN, WfFlipDemoteN, WfFlipKeepN, WfFlipPagesN, WfFlipCandN, WfFlipHashN, WfFlipNeN, WfFlipNeUnstableN, WfFastN, WfSlowN, WfFastNeN,
	// S444 (KYTY_WF_FLIP_UNPROTECT=1): write faults on a page that was kept unprotected at that moment.
	WfFlipStickyRefaultN,
	// S441 (KYTY_SHADER_PREWARM_MAX_MB): times the prewarm worker stopped at the byte limit instead of evicting (jobs wait in the queue) / estimated bytes of
	// the finished results waiting for CP (gauge) / their peak.
	PrewarmDeferredN, PrewarmReadyBytes, PrewarmReadyBytesPeak,
	// S441 (verify-442 step 0), cumulative: CP intervals between consecutive PrepareVideoOutFlip (count, sum, longest) / flips with a guest lead and the sum of
	// the per-flip minimum of (CP picks up a submission - the guest enqueued it), the guest's slack / ReadbackWaitNs split by the waiting thread (CP, the rest) /
	// CPU time of the CP thread and of all other threads of the process, per guest flip, and the number of snapshots.
	CpFrameN, CpFrameNs, CpFrameMaxNs, GuestLeadN, GuestLeadMinNs, ReadbackWaitCpNs, ReadbackWaitOtherNs, ThreadCpuCpNs, ThreadCpuOtherNs, ThreadCpuN,
	// S442 (KYTY_SRT_JIT_COVER=1|shadow): walks of plans with no reads answered from the plan / slots served by the per-slot JIT that read a constant buffer
	// (shadow: compared with the interpreter) / slots the per-slot JIT served in plans that capture specialization reads; shadow: flat buffers that differ
	// from the walk (must be 0).
	SrtCoverNoreadsN, SrtCoverCbN, SrtCoverCaptureN, SrtJitCoverNeN,
	// S445 (KYTY_SRT_SLOT_BATCH=1|shadow): batch entries / slots they filled / compiled slots left to the per-slot entry (a run of one) / runs that stopped at a failing
	// slot / shadow: results that differ from the per-slot path (must be 0).
	SrtBatchRunsN, SrtBatchSlotsN, SrtBatchSingleN, SrtBatchFailN, SrtBatchMismatchN,
	// S446 (KYTY_GPU_LABELS=1|2): labels put on encoders and the CPU time the label calls took (the last columns of frames.csv).
	GpuLabelN, GpuLabelNs,
	// S451: where GetGraphicsPrograms spends its time (graphics VS/PS/tess only, compute excluded). PrepareProgram of the vertex stages / of the pixel
	// stage; ProgramCache::Get split into the stage static key + lookup, the SrtRuntime set-up (SrtReader::Apply) and FindPermutation (the check of the
	// remembered specialization against guest memory); Get calls and Gets answered by a cached permutation; Get time of vertex / pixel stage.
	// The memo (KYTY_PROGS_MEMO): draws answered from it / draws that went the full way / entries held / shadow: memo result differs from the full one
	// (must be 0).
	GfpVsPrepNs, GfpPsPrepNs, GfpKeyNs, GfpRuntimeNs, GfpFindNs, GfpGetN, GfpGetHitN, GfpVsGetNs, GfpPsGetNs,
	GfpMemoHitN, GfpMemoMissN, GfpMemoEntriesN, GfpMemoNeN,
	// S451: inside FindPermutation: the SRT walk (RefreshFlatBuffer) / all of MaterializeResources (walk included) / the permutation search.
	GfpMatWalkNs, GfpMatTotalNs, GfpPermSearchNs,
	// S451: inside RefreshFlatBufferWalk: the tree JIT call / the control-flow reachability loop (block conditions) / the slot loops.
	GfpWalkTreeNs, GfpWalkCfNs, GfpWalkSlotsNs,
	// S451: the control-flow loop: walks that ran it / iterations of the reachability loop / blocks of the plan / conditions evaluated / EvaluateInst calls /
	// of those ReadFirstLane (each builds two SrtWalker objects).
	GfpCfWalkN, GfpCfIterN, GfpCfBlocksN, GfpCfCondN, GfpCfEvalN, GfpCfRflN,
	// S451 (KYTY_PROGS_MEMO): control-flow walks the memo could not serve by design (reads captured for the specialization, NullDesc diag, a nested mask).
	// shadow only: why the memo missed - nothing stored yet / the inputs (user data words read, shader base, sizes, readers) differ / a re-read answered differently.
	GfpMemoSkipN, GfpMemoMissColdN, GfpMemoMissInputsN, GfpMemoMissReadsN,
	// S451 DIAG: guest reads of the control-flow loop (calls, time in the reader).
	GfpCfReadN, GfpCfReadNs,
	// S463 DIAG (KYTY_DIAG_CP_PHASES=1): sub-phases of the CP thread. Draw bind (self ns, inside cp_draw_bind): PrepareBindings loop / FindBuffers /
	// PrepareBda / RebindImages / render-target rebind / RebindBuffers. Dispatch: GetComputeProgram+Pipeline / bind preparation / CommitBindings.
	// Handler self ns and packet counts by opcode class: draw, SET_*_REG, NOP, WAIT_REG_MEM, dispatch, other.
	BindPrepNs, BindFindNs, BindBdaNs, BindImgNs, BindRtNs, BindBufNs,
	DispProgNs, DispPrepNs, DispCommitNs,
	HdlDrawNs, HdlRegNs, HdlNopNs, HdlWaitNs, HdlDispNs, HdlOtherNs,
	HdlDrawN, HdlRegN, HdlNopN, HdlWaitN, HdlDispN, HdlOtherN,
	// S463 DIAG round 2 (wall ns, children included): DrawIndex head / PrepareDrawRenderState / ExecutePreparedDraw segments.
	DrawHeadNs, DrawStateNs, DrawPrologNs, DrawVbNs, DrawPreEmitNs, DrawDynNs, DrawEmitNs,
	// S456 DIAG: time of the reachability loop itself (the walks the memo did not serve).
	GfpCfLoopNs,
	// S465 DIAG: KYTY_INDIRECT_GPU=1 draws that went to the CPU read although the command qualified, by the host reason (QualifyHost) - pv_store, pv_proto, mesh,
	// copy_depth_to_color, primitive_type, primitive_restart, native_fetch - and those the command itself declined (cmd reason, any).
	IndirectHostPvStoreN, IndirectHostPvProtoN, IndirectHostMeshN, IndirectHostCopyDepthN, IndirectHostPrimTypeN, IndirectHostPrimRestartN,
	IndirectHostNativeFetchN, IndirectHostCmdN,
	// S472 (KYTY_INDIRECT_GPU_MESH): mesh draws recorded with the prepare compute + vkCmdDrawMeshTasksIndirectEXT (gpu), those the ring refused
	// (ring_full), shadow compares (n, equal, not equal) and plans that exceeded a host limit (violation).
	MeshIndGpuN, MeshIndRingFullN, MeshIndShadowN, MeshIndShadowEqN, MeshIndShadowNeN, MeshIndViolationN,
	Count,
};

void Add(Counter counter, uint64_t value = 1);

// Stage index of the MatMemo columns: 0 = vs, 1 = ps, 2 = cs.
enum class MatMemoMetric : uint32_t { N, InHit, Unsafe, OutHit, Rebase };
constexpr Counter MatMemoCounter(MatMemoMetric metric, uint32_t stage) {
	return static_cast<Counter>(static_cast<uint32_t>(Counter::MatMemoVsN) + static_cast<uint32_t>(metric) * 3u + stage);
}

// Counter of a dispatch class (the DispGuest..DispOther columns are in DispatchClass order).
constexpr Counter DispatchCounter(DispatchClass cls) {
	return static_cast<Counter>(static_cast<uint32_t>(Counter::DispGuest) + static_cast<uint32_t>(cls));
}
inline void NoteDispatch(DispatchClass cls) { Add(DispatchCounter(cls)); }

inline int64_t SteadyNowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
	           std::chrono::steady_clock::now().time_since_epoch())
	    .count();
}

// Adds the lifetime of the scope, in nanoseconds, to a *Ns counter.
class ScopedCounterNs {
public:
	explicit ScopedCounterNs(Counter counter): m_counter(counter), m_begin(SteadyNowNs()) {}
	~ScopedCounterNs() { Add(m_counter, ElapsedNs(m_begin, SteadyNowNs())); }
	ScopedCounterNs(const ScopedCounterNs&)            = delete;
	ScopedCounterNs& operator=(const ScopedCounterNs&) = delete;

private:
	Counter m_counter;
	int64_t m_begin;
};
// Gauge / running maximum for counters that are not sums (EopPending, EopLabelLatencyMaxUs).
void Set(Counter counter, uint64_t value);
void Max(Counter counter, uint64_t value);
uint64_t Get(Counter counter);
// vkQueueSubmits so far (the `submits` column of KYTY_FRAME_LOG).
uint64_t QueueSubmits();

// Bench switches: KYTY_OPT_OFF is a comma-separated list of optimization names to disable, so a
// single binary can be A/B tested. Returns true unless the name is listed.
bool OptEnabled(const char* name);
// KYTY_OPT_ON=<name>[,<name>...] switches an optimization that is off by default on.
bool OptForcedOn(const char* name);

// Number of frames presented since startup.
uint64_t PresentedFrames();

// Guest flips so far: the `flips` column of KYTY_FRAME_LOG.
uint64_t GuestFlips();

} // namespace Libs::Graphics::FrameTiming

#endif /* KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_ */
