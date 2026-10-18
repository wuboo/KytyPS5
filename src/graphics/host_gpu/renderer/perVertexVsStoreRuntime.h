#pragma once

// S150: renderer side of KYTY_PV_VSSTORE=1 (default OFF; the math and the shader side are in
// perVertexVsStore.h and the SPIR-V emitter). Shader variants of the VS and of the per-vertex PS are
// emitted when the guest shaders are compiled, paired into pipeline programs on first use, and the
// cap regions of the draws come from slabs that are recycled only after the GPU has finished them.

#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/perVertexVsStore.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {
class CommandScheduler;
struct GraphicContext;
namespace ShaderRecompiler {
struct CompileResult;
struct CompileOptions;
namespace IR {
struct Program;
} // namespace IR
namespace Spirv {
struct PvStoreInfo;
} // namespace Spirv
} // namespace ShaderRecompiler

// Compile-time hook (next to RememberPerVertexPrototypeShader): emits the VS-store variant of a vertex
// shader and of a pixel shader with PerVertexKHR inputs. No-op unless KYTY_PV_VSSTORE=1.
// S384: push_data_start_dword / maintenance8_enabled are the values of the regular program's compilation
// (after PerVertexReemitPushStart / PerVertexReemitMaintenance8); the variant starts its push data there.
void RememberPvVsStoreShader(const ShaderProgram& program, ShaderRecompiler::CompileResult& result,
                             const ShaderRecompiler::CompileOptions& options,
                             uint32_t push_data_start_dword, bool maintenance8_enabled);
// S409: the PS side of a compact record: stride = locations of ps_mask, slot = rank of the location in it. vs_mask only feeds the
// harness mutant (COMPACT_PS_USES_VS_SLOT).
void FillPvCompactPixelInfo(ShaderRecompiler::Spirv::PvStoreInfo& info, uint32_t ps_mask, uint32_t vs_mask);
// Re-emits `program` as the VS-store variant; leaves program.bindings as emitted.
std::vector<uint32_t> EmitPvVsStoreVariant(ShaderRecompiler::IR::Program&          program,
                                           const ShaderRecompiler::CompileOptions& options,
                                           const ShaderRecompiler::Spirv::PvStoreInfo& info,
                                           uint32_t push_data_start_dword, bool maintenance8_enabled);
// Both re-emission hooks for a freshly compiled program: the variants take the regular program's push
// start, each variant's start and shader-data size are checked against it, and program.bindings is
// restored after each re-emission. push_data_start_dword is what CompileProgram received.
void RememberPerVertexVariants(const ShaderProgram& program, ShaderRecompiler::CompileResult& result,
                               const ShaderRecompiler::CompileOptions& options, uint32_t host_subgroup_size,
                               uint32_t push_data_start_dword);

struct PvVsStorePrograms {
	PvVsStoreInterfaceVerdict       verdict     = PvVsStoreInterfaceVerdict::NoPerVertex;
	uint32_t                        stride_vec4 = 0;
	PipelineCache::GraphicsPrograms graphics; // valid when verdict == Use
	// S389 (diag): what the first-use log line says about the pair.
	uint32_t                        index   = 0; // among the candidate pairs, in the order they are first drawn
	uint64_t                        vs_hash = 0;
	uint64_t                        ps_hash = 0;
	uint32_t                        vs_mask = 0; // host locations the VS stores
	uint32_t                        ps_mask = 0; // host locations the PS reads per vertex
	uint32_t                        corners = 0; // bit k: the PS reads corner k (P0 / P10 / P20), i.e. uses gl_PrimitiveID
	bool                            compact = false; // S409: KYTY_PV_VSSTORE_COMPACT=1 record (stride_vec4 = locations of ps_mask)
	mutable bool                    first_use_logged = false;
};

// The VS-store programs of a VS/PS pair; nullptr when the pair is not a candidate at all (switch off,
// PS without per-vertex input, a variant failed to emit or validate, geometry/tessellation stages).
const PvVsStorePrograms* GetPvVsStorePrograms(GraphicContext& graphics, const PipelineCache::GraphicsPrograms& programs);

// S389: one `[pv-pairs] first use` line per pair (at most 200 pair lines in all): hashes, stride, slots the VS stores and
// the PS reads, the corners the PS reads, gl_PrimitiveID, topology, instancing. Call for every draw that reaches the pair.
void LogPvPairFirstUse(const PvVsStorePrograms& programs, uint32_t topology, bool indexed, uint32_t instance_count);

// Set 1 of the VS-store pipelines: binding 0 = cap (VS write, PS read), binding 1 = guest indices (PS), and with
// KYTY_PV_VSSTORE_GUARD=1 binding 2 = the guard's counter (VS).
vk::DescriptorSetLayout PvVsStoreSetLayout(GraphicContext& graphics);

// Push constant offset of the VS-store block (it follows the native push data).
uint32_t PvVsStorePushOffset();

// Cap slabs of one render executor. A region is valid until the command buffer it was placed in has
// finished on the GPU; a slab is reused only then (PvVsStoreSlabs).
class PvVsStoreRuntime {
public:
	PvVsStoreRuntime(GraphicContext& graphics, CommandScheduler& scheduler);
	~PvVsStoreRuntime();
	PvVsStoreRuntime(const PvVsStoreRuntime&)            = delete;
	PvVsStoreRuntime& operator=(const PvVsStoreRuntime&) = delete;

	struct Region {
		bool       ok          = false;
		uint32_t   slab        = 0;
		uint32_t   base_record = 0; // first record (stride_vec4 vec4s each) of the region
		vk::Buffer buffer      = nullptr;
		uint64_t   buffer_size = 0;
	};
	// `records` records of `stride_vec4` vec4s, for the command buffer that is open now.
	Region Place(uint64_t records, uint32_t stride_vec4);
	// Call when the draw is recorded (after resource preparation, which may flush the scheduler).
	void Touch(const Region& region);
	// S388 (KYTY_PV_VSSTORE_GUARD=1): the counter slot of guest flip `flip`. The first call of a new flip reports the
	// slot of flip-2 into pv_vsstore_oob and clears the new one; nothing is waited for.
	struct GuardSlot {
		vk::Buffer buffer = nullptr;
		uint64_t   offset = 0;
		uint64_t   range  = 0;
	};
	GuardSlot AcquireGuard(uint64_t flip);
	// Largest region (in vec4s) a draw may take.
	[[nodiscard]] static uint64_t SlabVec4s();
	// S406 (DIAG): records left in the slab the last placement used.
	[[nodiscard]] uint64_t FreeRecords() const { return m_slabs.CurrentFreeRecords(); }

private:
	GraphicContext&                      m_graphics;
	CommandScheduler&                    m_scheduler;
	PvVsStoreSlabs                       m_slabs;
	std::vector<std::unique_ptr<Buffer>> m_buffers;
	std::unique_ptr<Buffer>              m_guard;
	uint64_t                             m_guard_flip = 0;
};

// A draw that was accepted for the VS-store path (decided before any resource of the draw is prepared).
struct PvVsStoreDrawSetup {
	const PvVsStorePrograms*  programs = nullptr;
	PvVsStoreDraw             draw;
	uint32_t                  index_element_size = 0;
	PvVsStoreRuntime::Region  region;
	PvVsStoreRuntime::GuardSlot guard; // KYTY_PV_VSSTORE_GUARD=1 only
};

} // namespace Libs::Graphics
