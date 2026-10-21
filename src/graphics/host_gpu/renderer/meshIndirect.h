#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_

// S472: KYTY_INDIRECT_GPU_MESH=0|1|shadow (unset = 0). An indexed indirect draw (KYTY_INDIRECT_GPU candidate) whose host shader is a mesh shader
// does not wait for the CPU read of the args (the read drains the queue: RESULT-S463, ~1.7-2.0 ms/flip; RESULT-S465: mesh was 100 % of the
// fallbacks). What the CPU computed from index_count for a mesh draw:
//   groups   = (InputPrimitiveCount(index_count) - 1) / primitives_per_group + 1      -> vkCmdDrawMeshTasksEXT(groups, instance_count, 1)
//   push     = index_count, vertex_offset, first_instance, index bytes, index address (lo, hi)    -> read by the mesh shader
// On the GPU:
//   - a one-invocation compute pass (mesh_indirect_prepare.comp, same arithmetic as Plan) reads the args and writes the 3 dwords of
//     vkCmdDrawMeshTasksIndirectEXT (all zero when a host limit is exceeded: the CPU path stops with EXIT there);
//   - the mesh shader itself reads the guest args (push dword 3 bit 31) and derives index_count / offsets / index address (Translate.cpp,
//     same arithmetic as Plan); the push constants carry only what the CPU knows without the args (INDEX_BUFFER_SIZE, args address, index base).
//   0       the old path, bit for bit
//   shadow  the old path (the CPU reads the args); the compute pass is recorded next to the draw and what it wrote is compared with what the
//           CPU computed (no indirect draw)
//   1       qualified draws (indexed, one draw, no register patches, a known index size, a mesh shader that is not fast launch) take the GPU path
// Pure part; tests/MeshIndirectTests.cpp pins it.

#include "graphics/host_gpu/renderer/indirectGpu.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::MeshIndirect {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) {
#ifdef KYTY_MESHINDIRECT_MUTANT_DEFAULT_ON
		return Mode::On;
#else
		return Mode::Off;
#endif
	}
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_INDIRECT_GPU_MESH"));
	return mode;
}

// ShaderMeshInputInfo, the part the group count needs.
struct MeshInfo {
	uint32_t primitive_size       = 3; // vertices of one input primitive (InputPrimitiveSize)
	uint32_t primitive_step       = 3; // vertices between two input primitives (InputPrimitiveStep)
	uint32_t primitives_per_group = 0;
	bool     fast_launch          = false;
};

struct Limits {
	uint32_t max_x     = 0; // maxMeshWorkGroupCount[0]
	uint32_t max_y     = 0; // maxMeshWorkGroupCount[1]
	uint32_t max_total = 0; // maxMeshWorkGroupTotalCount
};

// VkDrawIndexedIndirectCommand as the guest wrote it.
struct Args {
	uint32_t index_count_per_instance = 0;
	uint32_t instance_count           = 0;
	uint32_t start_index_location     = 0;
	uint32_t base_vertex_location     = 0;
	uint32_t start_instance_location  = 0;
	bool     operator==(const Args& o) const { return std::memcmp(this, &o, sizeof(Args)) == 0; }
};

// What the draw is, once the args are known: the 3 dwords of the indirect command, and the values the shader takes from the args.
struct Plan {
	uint32_t groups_x       = 0;
	uint32_t groups_y       = 0;
	uint32_t groups_z       = 0;
	uint32_t index_count    = 0; // clamped to INDEX_BUFFER_SIZE
	uint32_t vertex_offset  = 0;
	uint32_t first_instance = 0;
	uint32_t start_index    = 0;
	uint32_t violation      = 0; // a host limit would be exceeded: nothing is drawn (the CPU path EXITs)
	bool     operator==(const Plan& o) const { return std::memcmp(this, &o, sizeof(Plan)) == 0; }
};

constexpr uint32_t InputPrimitiveCount(const MeshInfo& mesh, uint32_t vertices) {
	return vertices < mesh.primitive_size ? 0u : (vertices - mesh.primitive_size) / mesh.primitive_step + 1u;
}

constexpr Plan MakePlan(const Args& args, const MeshInfo& mesh, uint32_t index_buffer_size, bool native_fetch, const Limits& limits) {
	Plan plan;
	plan.start_index = args.start_index_location;
#ifdef KYTY_MESHINDIRECT_MUTANT_NO_CLAMP
	plan.index_count = args.index_count_per_instance;
#else
	plan.index_count = index_buffer_size != 0 && args.index_count_per_instance > index_buffer_size ? index_buffer_size
	                                                                                                : args.index_count_per_instance;
#endif
#ifdef KYTY_MESHINDIRECT_MUTANT_NATIVE_OFFSETS
	plan.vertex_offset  = args.base_vertex_location;
	plan.first_instance = args.start_instance_location;
#else
	plan.vertex_offset  = native_fetch ? 0u : args.base_vertex_location;
	plan.first_instance = native_fetch ? 0u : args.start_instance_location;
#endif
	const uint32_t primitives = InputPrimitiveCount(mesh, plan.index_count);
	if (primitives == 0 || args.instance_count == 0 || mesh.primitives_per_group == 0) {
		return plan; // CPU: "return" before anything is recorded; here a draw of zero groups
	}
#ifdef KYTY_MESHINDIRECT_MUTANT_GROUPS_FLOOR
	const uint32_t groups = primitives / mesh.primitives_per_group;
#else
	const uint32_t groups = (primitives - 1u) / mesh.primitives_per_group + 1u;
#endif
	if (groups > limits.max_x || args.instance_count > limits.max_y ||
	    static_cast<uint64_t>(groups) * args.instance_count > limits.max_total) {
		plan.violation = 1;
		return plan;
	}
	plan.groups_x = groups;
	plan.groups_y = args.instance_count;
	plan.groups_z = 1;
	return plan;
}

// The index address the shader forms: the 64-bit base plus start_index * element bytes (the CPU: ResolveOnCpu).
constexpr uint64_t IndexAddress(uint64_t base, uint32_t start_index, uint32_t element_bytes) {
	return base + static_cast<uint64_t>(start_index) * element_bytes;
}

// The push dword 3 the mesh shader sees. Bit 31: the parameters are in the guest args; bit 30: native vertex fetch; low 16 bits: index bytes.
constexpr uint32_t kPushArgsFlag   = 0x80000000u;
constexpr uint32_t kPushNativeFlag = 0x40000000u;
constexpr uint32_t PushElementWord(uint32_t element_bytes, bool native_fetch) {
	return kPushArgsFlag | (native_fetch ? kPushNativeFlag : 0u) | (element_bytes & 0xffffu);
}

// A mesh host path that the GPU can take: not fast launch (that path is indexed-incompatible), a group size, and the host reason is only "mesh".
inline bool MeshCanGoGpu(const MeshInfo& mesh) {
	return !mesh.fast_launch && mesh.primitives_per_group != 0;
}

} // namespace Libs::Graphics::MeshIndirect

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_MESHINDIRECT_H_ */
