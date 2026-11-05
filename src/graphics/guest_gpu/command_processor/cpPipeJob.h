#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPEJOB_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPEJOB_H

// A GetGraphicsPrograms call for one deferred draw. A worker runs it on the registers snapshotted
// with the draw. The recorder joins it when the draw executes and uses the result, or computes
// the programs itself when the worker could not answer.

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <atomic>
#include <cstdint>
#include <vector>

namespace Libs::Graphics::CpPipe {

struct GfpJob {
	enum : uint8_t { Idle, Queued, Done, Held };

	std::atomic<uint8_t> state {Idle};

	// The worker's own copy. The recorder executes the draw on the FIFO item's copy.
	PipelineCache*        cache = nullptr;
	HW::Context           ctx_copy;
	HW::Shader            sh_copy;
	HW::UserConfig        ucfg_copy;
	const HW::Context*    ctx  = nullptr;
	const HW::Shader*     sh   = nullptr;
	const HW::UserConfig* ucfg = nullptr;

	bool                             answered = false;
	PipelineCache::OffThreadPrograms out;
	std::vector<uint64_t>            pages;
	uint64_t                         write_seq = 0;
};

void RunGfpJob(GfpJob& job);

// The job of the draw the recorder is executing. Null when there is none. CP thread only.
GfpJob*& CurrentJob();

// Wait for the job. True when its result may be used: it answered, and no backing write since
// it started touched a page it read. The caller still rejects pages the GPU has written.
bool JoinJob(GfpJob& job);

} // namespace Libs::Graphics::CpPipe

#endif
