#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_DRSNOZERO_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_DRSNOZERO_H_

// S511 L2 (KYTY_DRS_NO_ZERO=0|1, unset = 0). The value-init of DrawRenderState zeroes eight
// RenderColorInfo and three ShaderVertexInputInfo on every draw. The fast path leaves the unused
// slots unconstructed. Fields a draw reads before GetGraphicsPrograms fills the used stage, and
// who writes them:
//   depth_info            default-init (ResolveRenderDepthTarget returns without writing when inactive)
//   color_info[0]         InitDrawRenderStateFast (logs and sample count read slot 0 with color_count == 0)
//   color_info[1..]       constructed in PrepareDrawRenderState just before ResolveRenderColorTarget,
//                         which assigns the whole entry (`r = {}`)
//   color_count, mrt_mask, ps_active, topology, primitive_restart  in-class defaults, then the caller
//   ps_input_info         default-init (object_primitive_id is read even when the pixel shader is off)
//   indirect_offsets, indirect_draw  InitDrawRenderStateFast from the draw packet
//   vertex_info[0]        construct_at here, then `info = {}` in ShaderGetStaticVertexInputInfo
//   vertex_info[1], [2]   not read while vertex_alive == 1 (VertexStages). Tessellation constructs
//                         them before PrepareTessellationPrograms writes them and sets vertex_alive = 3
// Not codegen. Mutant KYTY_DRS_MUTANT_ZERO_ALL zeroes the unused slots anyway (the poison test fails).

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <cstdlib>
#include <cstring>

namespace Libs::Graphics {

inline bool DrsNoZero() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_DRS_NO_ZERO");
		return text != nullptr && std::strcmp(text, "1") == 0;
	}();
	return on;
}

} // namespace Libs::Graphics

#endif
