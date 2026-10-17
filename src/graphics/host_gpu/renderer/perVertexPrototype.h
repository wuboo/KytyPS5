#pragma once

#include "graphics/host_gpu/renderer/perVertexTransform.h"
#include "graphics/host_gpu/renderer/perVertexUnique.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <cstddef>
#include <map>
#include <span>
#include <vector>

namespace Libs::Graphics {
namespace ShaderRecompiler {
struct CompileResult;
struct CompileOptions;
namespace IR {
struct Program;
struct Bindings;
} // namespace IR
namespace Spirv {
struct VertexCaptureInfo;
} // namespace Spirv
} // namespace ShaderRecompiler
bool PerVertexPrototypeEnabled();
// S134: KYTY_PV_UNIQUE=1 (value "1" only) captures once per unique vertex; default OFF.
bool PerVertexUniqueEnabled();
// S149: KYTY_PV_VSSTORE=1 (value "1" only) lets the VS write PerVertex values to an SSBO that the PS
// reads back (perVertexVsStore.h); default OFF. The emitter side exists, the draw path is not wired yet.
bool PerVertexVsStoreEnabled();
bool HasPerVertexPrototypeInput(std::span<const uint32_t> words);
bool DecodePerVertexPrototypeAttribute(Prospero::BufferFormat   format,
                                       std::span<const uint8_t> bytes,
                                       std::span<uint32_t, 4>   components);
// S384: KYTY_PV_PUSH_START=fix|zero (read once, default fix). fix: the capture and VS-store re-emissions
// start their push data where the regular program does (the CPU writes the push dwords under that
// start); zero: the pre-S384 behaviour (start 0, maintenance8 off). The first query logs "[pv-push] mode=".
bool     PerVertexPushStartFix();
uint32_t PerVertexReemitPushStart(uint32_t push_data_start_dword);
bool     PerVertexReemitMaintenance8(bool maintenance8_enabled);
// A variant must read its shader data from where the regular program does.
bool PvVariantPushMatches(const ShaderRecompiler::IR::Bindings& regular,
                          const ShaderRecompiler::IR::Bindings& variant);
// Logs "[pv-push] role=... hash=... start=... dwords=..." once per emitted variant and counts the
// variants whose push data starts at a nonzero dword (pv_push_nonzero_programs).
void NotePvPushVariant(const char* role, uint64_t shader_hash,
                       const ShaderRecompiler::IR::Bindings& variant);
// Re-emits `program` as the per-vertex capture compute shader; leaves program.bindings as emitted.
std::vector<uint32_t> EmitPerVertexCaptureVariant(ShaderRecompiler::IR::Program&                program,
                                                  const ShaderRecompiler::CompileOptions&       options,
                                                  const ShaderRecompiler::Spirv::VertexCaptureInfo& capture,
                                                  uint32_t push_data_start_dword,
                                                  bool     maintenance8_enabled);
// push_data_start_dword / maintenance8_enabled are the values the regular program was compiled with
// (already passed through PerVertexReemitPushStart / PerVertexReemitMaintenance8).
void RememberPerVertexPrototypeShader(const ShaderProgram&                    program,
                                      ShaderRecompiler::CompileResult&        result,
                                      const ShaderRecompiler::CompileOptions& options,
                                      uint32_t                                host_subgroup_size,
                                      uint32_t                                push_data_start_dword,
                                      bool                                    maintenance8_enabled);

// What the prototype path does with a draw whose pixel shader needs PerVertexKHR (the host cannot
// run those natively, so the draw either goes through the prototype or is skipped, never exits).
// Only a triangle list without primitive restart is supported; a trailing partial triangle is
// dropped like the hardware does.
struct PerVertexDrawVerdict {
	bool        run         = false;
	uint32_t    index_count = 0; // count to draw when run
	const char* reason      = nullptr; // why it is skipped (or truncated), for the log
};
PerVertexDrawVerdict ClassifyPerVertexPrototypeDraw(bool ps_active, vk::PrimitiveTopology topology,
                                                    bool primitive_restart, uint32_t index_count);

struct PerVertexPrototypePrograms {
	bool                            native_capture = false;
	vk::ShaderModule                capture;
	PipelineCache::GraphicsPrograms graphics;
	// S134 (KYTY_PV_UNIQUE=1 only): replay shaders reading captured[map[corner]]; empty otherwise.
	bool                            unique_available = false;
	PipelineCache::GraphicsPrograms graphics_unique;
	vk::DescriptorSetLayout         extra_layout;
	PipelineCache::Pipeline         capture_pipeline;
	PerVertexLayout                 layout;
};

struct PerVertexUnpackPipeline {
	vk::ShaderModule        module;
	vk::DescriptorSetLayout set_layout;
	vk::PipelineLayout      pipeline_layout;
	vk::Pipeline            pipeline;
};

// Vulkan's minimum maxPushConstantsSize is 128 bytes, which held 12 Attrs. MoltenVK reports
// 4096 (MVKDevice.mm), so 16 Attrs (160 bytes) fit on this host. RES_MAX stays 32; a draw past
// this cap is skipped instead of exiting.
constexpr uint32_t kPerVertexPrototypeMaxAttributes = 16;

struct UnpackPushConstants {
	uint32_t total_invocations; // word 0
	uint32_t index_count;       // word 1
	uint32_t first_vertex;      // word 2
	int32_t  vertex_offset;     // word 3 (signed)
	uint32_t first_instance;    // word 4
	uint32_t vertex_stride;     // word 5
	uint32_t num_records;       // word 6: strict logical guest buffer bound
	uint32_t packed_flags; // word 7: [15:0] index_base_offset, [16] is_indexed, [18:17] index_type,
	                       // [23:19] num_attributes (5 bits; 16 must not truncate to 0)
	struct Attr {
		uint32_t meta;       // [15:0] byte offset, [19:17] component count, [22:20] format kind
		uint32_t bit_counts; // [cnt0, cnt1, cnt2, cnt3]
	} attrs[kPerVertexPrototypeMaxAttributes];
	uint32_t unique_first; // S134 map mode ([24] of packed_flags): first unique vertex
	uint32_t unique_count; // ... and unique vertices per instance
};
constexpr uint32_t kUnpackMapModeFlag = 1u << 24u;
static_assert(sizeof(UnpackPushConstants::Attr) == 8);
static_assert(offsetof(UnpackPushConstants, attrs) == 32);
static_assert(offsetof(UnpackPushConstants::Attr, bit_counts) == 4);
static_assert(offsetof(UnpackPushConstants, unique_first) == 160);
static_assert(sizeof(UnpackPushConstants) == 168);

const PerVertexPrototypePrograms*
GetPerVertexPrototypePrograms(GraphicContext&                        graphics,
                              const PipelineCache::GraphicsPrograms& programs,
                              const ShaderVertexInputInfo& vs_info, vk::PipelineCache driver_cache);
const PerVertexUnpackPipeline* GetPerVertexUnpackPipeline(GraphicContext&   graphics,
                                                          vk::PipelineCache driver_cache);

} // namespace Libs::Graphics
