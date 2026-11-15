#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPEJOB_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPEJOB_H

// S479 (KYTY_CP_PIPE): a GetGraphicsPrograms call for a deferred draw, run by a pipeline worker on the registers snapshotted with the draw (S490: PipelineCache::GetGraphicsProgramsOffThread). The recorder joins it
// when the draw is executed (RefreshShaders) and uses the result or, when the worker could not answer or a page it read has become GPU-dirty, computes the
// programs itself as before.

#include "graphics/guest_gpu/command_processor/cpPipeSteal.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/scoutTableLog.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace Libs::Graphics::CpPipe {

struct GfpJob {
	// Held: parsed, not handed to a worker yet (KYTY_CP_PIPE_SIGNALS). S606 (KYTY_CP_PIPE_STEAL, cpPipeSteal.h): Running = a worker started it,
	// Cancelled = the recorder took it before any worker did (the worker only marks it Done).
	enum : uint8_t { Idle = Steal::kIdle, Queued = Steal::kQueued, Done = Steal::kDone, Held = Steal::kHeld, Running = Steal::kRunning, Cancelled = Steal::kCancelled };

	std::atomic<uint8_t> state {Idle};
	// S606: the recorder computed the draw's programs itself and no longer waits for this job (it set abandoned and steal_ns before the item's
	// MarkDone; the front reads them after WaitSlot and waits for Done before it rewrites the slot). run_ns / done_ns: the worker's start and end.
	bool                 abandoned = false;
	int64_t              steal_ns  = 0;
	std::atomic<int64_t> run_ns {0};
	std::atomic<int64_t> done_ns {0};

	// inputs: the worker's own copy of the registers (the recorder executes the draw on the FIFO item's copy and the renderer may write to it)
	PipelineCache*        cache = nullptr;
	HW::Context           ctx_copy;
	HW::Shader            sh_copy;
	HW::UserConfig        ucfg_copy;
	const HW::Context*    ctx   = nullptr;
	const HW::Shader*     sh    = nullptr;
	const HW::UserConfig* ucfg  = nullptr;

	// outputs
	bool                                   answered = false; // the worker produced programs for every stage the draw needs (out.fail says why not)
	PipelineCache::OffThreadPrograms       out;              // the stage inputs point at out.vs_res / out.ps_res
	ScoutTableLog::Log                     tables;
	std::vector<uint64_t>                  pages; // 4 KiB pages of guest memory the worker read, sorted and unique
	int64_t                                worker_ns = 0;
	int64_t                                submit_ns = 0;
	uint64_t                               write_seq = 0; // the log of backing writes when the worker started
	uint64_t                               shard_key = 0;
	// S558 (KYTY_CP_PREDECODE, texturePredecode.h): the worker's decode of the T#s of out.vs_res (pre[0]) and out.ps_res (pre[1]) and its ns. Written
	// by the worker only (RunGfpJob, before state = Done), read by the recorder only after JoinJob.
	PreStages pre;
	int64_t   pre_ns = 0;
};

// The worker's body (renderDraw.cpp: it needs the draw's own register decoding).
void RunGfpJob(GfpJob& job);

// The job of the draw the recorder is executing (nullptr: none). CP thread only.
GfpJob*& CurrentJob();

// cpPipe.cpp: wait for the job; true when its result may be used (answered, and none of the pages it read is GPU-dirty now). S606: false with
// job.abandoned set = the recorder did not wait (KYTY_CP_PIPE_STEAL=1); it computes the programs itself and must not read the job any more.
bool JoinJob(GfpJob& job);

// S606 (KYTY_CP_PIPE_STEAL): on (1 or count, with the front thread); the CP's own GetGraphicsPrograms time for a draw that had a job (cost model).
bool StealActive();
void NoteCpPrograms(const GfpJob* job, int64_t ns);

} // namespace Libs::Graphics::CpPipe

#endif
