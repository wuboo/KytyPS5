#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_DRAWREPEATDIAG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_DRAWREPEATDIAG_H_

// S507 DIAG: whether the next draw in the same command buffer repeats the previous draw's
// pipeline and bindings (draw_state_same_n) or its render targets and texture-cache generation (rt_same_n).
// Pure comparison. Nothing here changes a draw.

#include <cstdint>
#include <vector>

namespace Libs::Graphics::DrawRepeat {

struct Finger {
	uint64_t              cb         = 0;
	uint64_t              pipeline   = 0;
	uint64_t              generation = 0;
	uint32_t              color_n    = 0;
	uint32_t              depth      = 0xFFFFFFFFu;
	uint32_t              colors[8] {};
	std::vector<uint32_t> images;
	std::vector<uint64_t> samplers;
	std::vector<uint32_t> buffers;
	std::vector<uint32_t> shader_data;
	bool                  valid = false;
};

inline bool SameState(const Finger& prev, const Finger& now) {
	if (!prev.valid || !now.valid || prev.cb == 0 || prev.cb != now.cb) {
		return false;
	}
	if (prev.pipeline != now.pipeline || prev.images != now.images || prev.samplers != now.samplers || prev.buffers != now.buffers) {
		return false;
	}
#if defined(KYTY_DRAWREPEAT_MUTANT_IGNORE_SHADER)
	return true;
#else
	return prev.shader_data == now.shader_data;
#endif
}

inline bool SameRt(const Finger& prev, const Finger& now) {
	if (!prev.valid || !now.valid || prev.cb == 0 || prev.cb != now.cb) {
		return false;
	}
	if (prev.color_n != now.color_n || prev.depth != now.depth) {
		return false;
	}
	for (uint32_t i = 0; i < prev.color_n; i++) {
		if (prev.colors[i] != now.colors[i]) {
			return false;
		}
	}
#if defined(KYTY_DRAWREPEAT_MUTANT_IGNORE_GENERATION)
	return true;
#else
	return prev.generation == now.generation;
#endif
}

inline Finger& Previous() {
	static thread_local Finger finger;
	return finger;
}

inline void Observe(Finger now, bool* state_same, bool* rt_same) {
	*state_same = SameState(Previous(), now);
	*rt_same    = SameRt(Previous(), now);
	now.valid   = true;
	Previous()  = std::move(now);
}

} // namespace Libs::Graphics::DrawRepeat

#endif
