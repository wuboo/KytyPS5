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
	Count,
};

void Add(Counter counter, uint64_t value = 1);

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
