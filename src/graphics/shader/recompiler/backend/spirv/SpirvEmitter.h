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

// S149 (KYTY_PV_VSSTORE): the VS writes its PerVertex parameters to cap[record * stride + slot]
// (uvec4, bits unchanged) and the PS reads the three corners of its primitive back from there
// instead of PerVertexKHR. Set 1 binding 0 = cap (uvec4[]), binding 1 (PS) = the index buffer
// (uint[]); a push block of kPvVsStorePushDwords dwords follows the native push data (layout and
// record math in host_gpu/renderer/perVertexVsStore.h).
struct PvStoreInfo {
	enum class Role { Vertex, Pixel };
	Role                         role               = Role::Vertex;
	uint32_t                     record_stride_vec4 = 0;
	// record_stride_vec4 == 0 (S150): slot == host parameter location, the stride is read from the push
	// block, and `location_mask` (when set) collects the host locations the stage stores / reads.
	std::map<uint32_t, uint32_t> parameter_slots; // host parameter location -> slot in a record
	uint32_t*                    location_mask      = nullptr;
	bool                         coherent           = false; // Coherent cap + OpMemoryBarrier after the VS stores
	// S388 (KYTY_PV_VSSTORE_GUARD=1): push dword kPvPushRecords (the draw's record count) bounds the records; the VS skips a
	// store outside it and counts it in set 1 binding 2, the PS paints a read outside it magenta. Off: the module is unchanged.
	bool                         guard              = false;
};

// Resets and writes backend IDs and descriptor indices; do not emit the same program concurrently.
std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword = 0,
                                  bool maintenance8_enabled = false,
                                  bool emit_debug_info = false,
                                  const VertexCaptureInfo* vertex_capture = nullptr,
                                  const PvStoreInfo* pv_store = nullptr);

} // namespace Libs::Graphics::ShaderRecompiler::Spirv

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SPIRVEMITTER_H_ */
