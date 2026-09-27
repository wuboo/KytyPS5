#ifndef KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_
#define KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_

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
	Count,
};

void Add(Counter counter, uint64_t value = 1);

// Bench switches: KYTY_OPT_OFF is a comma-separated list of optimization names to disable, so a
// single binary can be A/B tested. Returns true unless the name is listed.
bool OptEnabled(const char* name);

// Number of frames presented since startup.
uint64_t PresentedFrames();

} // namespace Libs::Graphics::FrameTiming

#endif /* KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_ */
