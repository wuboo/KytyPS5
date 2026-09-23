#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

// PerVertex capture ABI: set 1 contains attributes (uvec4), outputs (vec4), and IDs (uvec2).
struct VertexCaptureInfo {
	uint32_t                     host_subgroup_size = 32;
	uint32_t                     num_attributes     = 0;
	uint32_t                     record_stride_vec4 = 0;
	uint32_t                     clip_slot          = UINT32_MAX;
	std::map<uint32_t, uint32_t> parameter_slots;
};

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword = 0,
                                  bool maintenance8_enabled = false,
                                  bool emit_debug_info = false,
                                  const VertexCaptureInfo* vertex_capture = nullptr);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
