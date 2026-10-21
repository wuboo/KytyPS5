#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_INDIRECTGPU_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_INDIRECTGPU_H_

// S404: KYTY_INDIRECT_GPU=0|1|shadow (unset = 1, unknown = 0). An indexed indirect draw of one draw, without a count buffer, whose
// registers need no patching (all SH_NOP) and whose index buffer size is known is recorded as vkCmdDrawIndexedIndirect(args) instead
// of reading the args on the CPU first. The CPU read of GPU-written args drains the queue (the S401 census: draw B, args pv=0, regs=0).
//   0       the old path, bit for bit (KYTY_INDIRECT_GPU=0).
//   shadow  the old path; every candidate is counted and logged ([indirect-gpu]) with the reason it would not qualify.
//   1       qualified draws go to the GPU; the others take the old path (and say why).
// The decision is made in two steps, because the host path of a draw is only known after its shader state is built:
//   Qualify()      what the command processor knows (no read of the args);
//   QualifyHost()  what the executor knows (no read of the args either: nothing here may depend on index_count / instance_count).
// This header is the pure part (tests/IndirectGpuTests.cpp pins it).

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::IndirectGpu {

enum class Mode : uint32_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) {
#ifdef KYTY_INDIRECTGPU_MUTANT_DEFAULT_OFF
		return Mode::Off;
#else
		return Mode::On; // S405: on by default; KYTY_INDIRECT_GPU=0 restores the CPU read of the args
#endif
	}
	if (std::strcmp(text, "1") == 0) {
		return Mode::On;
	}
	if (std::strcmp(text, "shadow") == 0) {
		return Mode::Shadow;
	}
	return Mode::Off;
}

// KYTY_INDIRECT_GPU, read once.
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_INDIRECT_GPU"));
	return mode;
}

enum class Reason : uint32_t {
	Eligible,
	NotIndexed,       // DrawIndirect (non-indexed) stays on the CPU
	CountAddr,        // DrawIndirectCount / DrawIndexedIndirectCount
	DrawCount,        // draw_count != 1
	RegisterPatch,    // a vertex / instance / index offset register has to be patched from the args
	NoIndexSize,      // m_index_buffer_size == 0: the bound range is unknown
	Index8,           // 8-bit indices are widened on the CPU
	NoFirstInstance,  // the device lacks drawIndirectFirstInstance (a non-zero first instance in the args would be invalid)
	PerVertexStore,   // KYTY_PV_VSSTORE takes (or may take) the draw: it reads index_count
	PerVertexProto,   // the per-vertex prototype takes (or may take) the draw
	Mesh,             // mesh shader groups are computed from the counts
	CopyDepthToColor, // render_control.copy_depth_to_color
	PrimitiveType,    // quad list legacy loops over index_count
	PrimitiveRestart, // restart handling scans the indices
	NativeFetch,      // native vertex fetch: the CPU path draws with offsets 0 and the patched SGPRs
	Count,
};

constexpr const char* ReasonName(Reason reason) {
	switch (reason) {
		case Reason::Eligible: return "eligible";
		case Reason::NotIndexed: return "not_indexed";
		case Reason::CountAddr: return "count_addr";
		case Reason::DrawCount: return "draw_count";
		case Reason::RegisterPatch: return "register_patch";
		case Reason::NoIndexSize: return "no_index_size";
		case Reason::Index8: return "index8";
		case Reason::NoFirstInstance: return "no_first_instance";
		case Reason::PerVertexStore: return "pv_store";
		case Reason::PerVertexProto: return "pv_proto";
		case Reason::Mesh: return "mesh";
		case Reason::CopyDepthToColor: return "copy_depth_to_color";
		case Reason::PrimitiveType: return "primitive_type";
		case Reason::PrimitiveRestart: return "primitive_restart";
		case Reason::NativeFetch: return "native_fetch";
		default: return "?";
	}
}

// What the command processor knows when DrawIndirectMulti starts (nothing here reads the args).
struct Candidate {
	bool     indexed           = false;
	bool     has_count_addr    = false;
	uint32_t draw_count        = 0;
	bool     registers_patched = false; // any of vertex / instance / index offset registers != SH_NOP
	uint32_t index_buffer_size = 0;     // m_index_buffer_size, in indices
	uint32_t index_element     = 0;     // bytes per index (1, 2, 4)
	bool     first_instance_ok = false; // drawIndirectFirstInstance is enabled
};

inline Reason Qualify(const Candidate& c) {
	if (!c.indexed) {
		return Reason::NotIndexed;
	}
	if (c.has_count_addr) {
		return Reason::CountAddr;
	}
	if (c.draw_count != 1) {
		return Reason::DrawCount;
	}
	if (c.registers_patched) {
		return Reason::RegisterPatch;
	}
	if (c.index_buffer_size == 0) {
		return Reason::NoIndexSize;
	}
	if (c.index_element == 1) {
		return Reason::Index8;
	}
	if (!c.first_instance_ok) {
		return Reason::NoFirstInstance;
	}
	return Reason::Eligible;
}

// What the executor knows once the shader state of the draw is built (nothing here reads the args either).
struct HostPath {
	bool pv_store_possible = false; // KYTY_PV_VSSTORE is on and the program pair is one the store knows
	bool proto_possible    = false; // the vertex program has a per-vertex prototype
	bool mesh              = false;
	bool mesh_gpu          = false; // S472: KYTY_INDIRECT_GPU_MESH is on and the mesh shader can read the args itself (MeshIndirect::MeshCanGoGpu)
	bool copy_depth        = false;
	bool quad_list_legacy  = false;
	bool restart           = false; // primitive restart control is set for a strip / fan topology
	bool native_fetch      = false; // the vertex fetch is native (not rewritten): offsets come from patched SGPRs
};

inline Reason QualifyHost(const HostPath& h) {
#if defined(KYTY_INDIRECTGPU_MUTANT_IGNORE_PV)
	// Mutant: the VS-store and prototype paths (which read index_count) are not looked at.
	const bool pv_store = false, proto = false;
#else
	const bool pv_store = h.pv_store_possible, proto = h.proto_possible;
#endif
	if (pv_store) {
		return Reason::PerVertexStore;
	}
	if (proto) {
		return Reason::PerVertexProto;
	}
	if (h.mesh && !h.mesh_gpu) {
		return Reason::Mesh;
	}
	if (h.copy_depth) {
		return Reason::CopyDepthToColor;
	}
	if (h.quad_list_legacy) {
		return Reason::PrimitiveType;
	}
	if (h.restart) {
		return Reason::PrimitiveRestart;
	}
	if (h.native_fetch) {
		return Reason::NativeFetch;
	}
	return Reason::Eligible;
}

// The draw the CPU path makes from the args (DrawIndirectMulti, indexed): the fallback of a candidate that did not qualify.
struct IndexedArgs {
	uint32_t index_count_per_instance = 0;
	uint32_t instance_count           = 0;
	uint32_t start_index_location     = 0;
	uint32_t base_vertex_location     = 0;
	uint32_t start_instance_location  = 0;
};
struct CpuDraw {
	uint32_t index_count    = 0;
	uint64_t index_addr     = 0;
	uint32_t instance_count = 0;
	int32_t  base_vertex    = 0;
	uint32_t first_instance = 0;
};
inline CpuDraw ResolveOnCpu(const IndexedArgs& args, uint64_t index_base, uint64_t index_size, uint32_t index_buffer_size) {
	CpuDraw draw;
	draw.index_addr = index_base + static_cast<uint64_t>(args.start_index_location) * index_size;
	draw.index_count =
	    index_buffer_size != 0 && args.index_count_per_instance > index_buffer_size ? index_buffer_size : args.index_count_per_instance;
	draw.instance_count = args.instance_count;
	draw.base_vertex    = static_cast<int32_t>(args.base_vertex_location);
	draw.first_instance = args.start_instance_location;
	return draw;
}

// The index count of the draw exceeds the bound index buffer: the CPU path clamps it, the GPU path cannot.
constexpr bool ClampRisk(uint32_t index_count, uint32_t index_buffer_size) {
	return index_buffer_size != 0 && index_count > index_buffer_size;
}

// Bytes the index binding of a qualified draw covers.
constexpr uint64_t IndexRangeBytes(uint32_t index_buffer_size, uint64_t index_element) {
	return static_cast<uint64_t>(index_buffer_size) * index_element;
}

} // namespace Libs::Graphics::IndirectGpu

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_INDIRECTGPU_H_ */
