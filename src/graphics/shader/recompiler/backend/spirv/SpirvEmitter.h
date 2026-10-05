#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// PerVertex capture ABI: set 1 contains attributes (uvec4), outputs (vec4), and IDs (uvec2).
// Fused ABI (`fused`): the capture decodes the guest vertices itself. Set 1 binding 0 is the raw
// vertex buffer (uint[]), binding 2 the raw index buffer (uint[]; the vertex buffer for
// non-indexed draws) and the push block grows by kFusedCapturePushDwords dwords at byte offset
// 128 (layout of UnpackPushConstants in perVertexPrototype.h).
constexpr uint32_t kFusedCapturePushDwords = 40;

struct VertexCaptureInfo {
	uint32_t                     host_subgroup_size = 32;
	uint32_t                     num_attributes     = 0;
	uint32_t                     record_stride_vec4 = 0;
	uint32_t                     clip_slot          = UINT32_MAX;
	std::map<uint32_t, uint32_t> parameter_slots;
	bool                         fused = false;
};

std::vector<uint32_t> EmitProgram(const IR::Program& program, ShaderStageInputInfo input_info,
                                  const VertexCaptureInfo* vertex_capture = nullptr);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
