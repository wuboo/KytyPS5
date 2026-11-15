#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPEJOB_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPEJOB_H

// S479 (KYTY_CP_PIPE): a GetGraphicsPrograms call for a deferred draw, run by a pipeline worker on the registers snapshotted with the draw (S490: PipelineCache::GetGraphicsProgramsOffThread). The recorder joins it
// when the draw is executed (RefreshShaders) and uses the result or, when the worker could not answer or a page it read has become GPU-dirty, computes the
// programs itself as before.

#include "graphics/guest_gpu/command_processor/cpPipe.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/scoutTableLog.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::CpPipe {

struct GfpJob {
	enum : uint8_t { Idle, Queued, Done, Held }; // Held: parsed, not handed to a worker yet (KYTY_CP_PIPE_SIGNALS)

	std::atomic<uint8_t> state {Idle};

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
	// S607 (KYTY_WORKER_CLEAN_WALK, cpPipeCleanWalk.h): walker loads of the job the worker served from the backing (0: none, then the two lists
	// are empty and JoinJob is S490's), the read ranges on their pages that JoinJob checks byte by byte, and those pages (asked for backing writes).
	uint32_t                                   cw_served = 0;
	std::vector<std::pair<uint64_t, uint64_t>> cw_bytes;
	std::vector<uint64_t>                      cw_pages;
	// S609 (KYTY_CP_PIPE_SIGNALS, cpPipeChecked.h): set by the front before the job is handed out. gated: held at parse (every policy; the join counts its
	// waits apart). early: checked let it start before the item fifo would have waited for. cpu_writes: the CPU-write ranges of the ordered items still in
	// the FIFO at its parse (checked only; the join refuses the job when a page it read lies in one).
	bool                    gated  = false;
	bool                    early  = false;
	bool                    shadow = false; // checked_shadow and early: the recorder computes the programs as well and compares (the result is not used)
	std::vector<WriteRange> cpu_writes;
};

// S607: installs the workers' reader of hot-page walker loads (KYTY_WORKER_CLEAN_WALK; renderDraw.cpp). Called once by the CP before the
// workers start, only when the switch is active.
void InstallWorkerCleanWalk();

// The worker's body (renderDraw.cpp: it needs the draw's own register decoding).
void RunGfpJob(GfpJob& job);

// The job of the draw the recorder is executing (nullptr: none). CP thread only.
GfpJob*& CurrentJob();

// cpPipe.cpp: wait for the job; true when its result may be used (answered, and none of the pages it read is GPU-dirty now).
bool JoinJob(GfpJob& job);

} // namespace Libs::Graphics::CpPipe

#endif
