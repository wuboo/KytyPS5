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
