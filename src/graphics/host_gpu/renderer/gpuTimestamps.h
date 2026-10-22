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

// S478 DIAG (same switch KYTY_GPU_TIMESTAMPS=1): what a command buffer holds (draws, dispatches, render passes with their size) and the GPU time of every pass
// (a BOTTOM_OF_PIPE timestamp before BeginRendering and after EndRendering, 2 queries per pass, at most kMaxPass passes) and of the clock probe.
// Queries of a slot: 0/1 = start/end of the buffer, 2/3 = clock probe, 4+2i / 5+2i = pass i.
constexpr uint32_t kMaxPass        = 24;
constexpr uint32_t kMemQuery       = 4 + 2 * kMaxPass; // memory probe (a fill of a big buffer), 2 queries
constexpr uint32_t kMaxDraw        = 256;               // KYTY_GPU_DRAW_TS=1: a TOP_OF_PIPE timestamp before each of the first kMaxDraw draws of a buffer
constexpr uint32_t kDrawQuery      = 6 + 2 * kMaxPass;
constexpr uint32_t kQueriesPerSlot = kDrawQuery + kMaxDraw;

struct PassRec {
	uint16_t w = 0, h = 0;
	uint8_t  ncol = 0, flags = 0; // flags: 1 = depth, 2 = a clear
	uint32_t draw0 = 0;           // draws of the buffer before this pass
};

struct DrawRec {
	uint64_t vs = 0, ps = 0;
	uint32_t count = 0;
	uint8_t  pass = 0;
};

struct BufStats {
	uint32_t draws = 0, disp = 0, npass = 0, ndraw_ts = 0;
	bool     probed = false, mem_probed = false, pass_open = false;
	PassRec  pass[kMaxPass] {};
	DrawRec  draw_ts[kMaxDraw] {};
};

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
[[nodiscard]] int32_t OnBegin(GraphicContext& graphics, vk::CommandBuffer buffer, BufStats& stats);
void                  OnEnd(GraphicContext& graphics, vk::CommandBuffer buffer, int32_t slot);
void                  OnPassBegin(vk::CommandBuffer buffer, int32_t slot, BufStats& stats, uint32_t w, uint32_t h, uint32_t ncol, uint32_t flags);
void                  OnDraw(vk::CommandBuffer buffer, int32_t slot, BufStats& stats, uint64_t vs, uint64_t ps, uint32_t count);
void                  OnPassEnd(vk::CommandBuffer buffer, int32_t slot, BufStats& stats);
void                  OnSubmit(int32_t slot, uint64_t tick, uint8_t reason, const BufStats& stats, uint32_t debug_op);
void                  Poll(GraphicContext& graphics, MasterSemaphore& master);
// S478 DIAG, H-ZEGAR (b): KYTY_DIAG_CP_DELAY_NS=N busy-waits N ns on the calling thread per call (one per guest draw): the CP rhythm of the scout without the scout.
void                  CpDelay();

} // namespace GpuTs
} // namespace Libs::Graphics

#endif
