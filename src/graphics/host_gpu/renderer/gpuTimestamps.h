#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_GPUTIMESTAMPS_H_

// S477 D3 (KYTY_GPU_TIMESTAMPS=1, DIAG, default off): the real GPU time of every submit. A command buffer starts with vkCmdResetQueryPool + a TOP_OF_PIPE
// timestamp and ends with a BOTTOM_OF_PIPE one (2 queries per slot, 256 slots). The submit side notes (slot, timeline tick); Poll() reads the slots whose tick
// the GPU has passed (no waiting). The queue is serial, so the busy time of a submit is its end minus max(its start, the previous end): TOP_OF_PIPE of the
// next buffer may fire while the previous one is still in the pipe, the max() keeps the overlap out of the sum (gpu_ts_overlap_n counts those). The gap
// between the previous end and this start is the idle time the GPU spent without work (gpu_ts_idle_ns).
// Counters (cumulative, frames.csv): gpu_ts_busy_ns, gpu_ts_idle_ns, gpu_ts_n, gpu_ts_sum_ns (raw end-start), gpu_ts_overlap_n, gpu_ts_skip_n.
// Log: "[gpu-ts]" line with a duration histogram of the submits at the flips of KYTY_GPU_TIMESTAMPS_AT (default 6300,6800,6990), counters reset after each.
// Pure part (Accumulate, Bucket) is header-inline; the Vulkan part is in gpuTimestamps.cpp.

#include <cstdint>

#include "graphics/host_gpu/vulkanCommon.h"

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

namespace GpuTs {

constexpr uint32_t kSlots = 256;

struct Acc {
	uint64_t prev_end = 0;
	bool     have     = false;
	uint64_t busy_ns = 0, idle_ns = 0, sum_ns = 0, overlap_n = 0;
};

// start/end in device ticks (already masked to valid bits), period in ns per tick
inline uint64_t Accumulate(Acc& a, uint64_t start, uint64_t end, double period) {
	const uint64_t dur = end >= start ? end - start : 0;
	a.sum_ns += static_cast<uint64_t>(static_cast<double>(dur) * period);
	uint64_t eff_start = start;
	if (a.have) {
		if (start < a.prev_end) {
			a.overlap_n++;
			eff_start = a.prev_end;
		} else {
			a.idle_ns += static_cast<uint64_t>(static_cast<double>(start - a.prev_end) * period);
		}
	}
	const uint64_t busy = end > eff_start ? end - eff_start : 0;
	a.busy_ns += static_cast<uint64_t>(static_cast<double>(busy) * period);
	a.prev_end = end > a.prev_end || !a.have ? end : a.prev_end;
	a.have     = true;
	return static_cast<uint64_t>(static_cast<double>(busy) * period);
}

// microsecond buckets: <100, <300, <1000, <3000, <10000, >=10000
inline uint32_t Bucket(uint64_t ns) {
	const uint64_t us = ns / 1000;
	return us < 100 ? 0 : us < 300 ? 1 : us < 1000 ? 2 : us < 3000 ? 3 : us < 10000 ? 4 : 5;
}

[[nodiscard]] bool Enabled();
// Called while the buffer is being recorded; Begin returns the slot (or -1: disabled / ring full).
[[nodiscard]] int32_t OnBegin(GraphicContext& graphics, vk::CommandBuffer buffer);
void                  OnEnd(GraphicContext& graphics, vk::CommandBuffer buffer, int32_t slot);
void                  OnSubmit(int32_t slot, uint64_t tick);
void                  Poll(GraphicContext& graphics, MasterSemaphore& master);

} // namespace GpuTs
} // namespace Libs::Graphics

#endif
