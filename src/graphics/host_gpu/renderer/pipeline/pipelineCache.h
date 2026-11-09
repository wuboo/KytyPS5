#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <queue>
#include <shared_mutex>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct ShaderParams;
class CommandBuffer;

namespace ShaderRecompiler {
struct CompileOptions;
struct ShaderSource;
struct TranslateResult;
} // namespace ShaderRecompiler

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

namespace ShaderRecompiler::Spirv {
struct Profile;
} // namespace ShaderRecompiler::Spirv

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	// 0 without tessellation, 3 for a rectangle list, else the input control points.
	uint32_t                   patch_control_points     = 0;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
};

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride   = 0;
		uint8_t  instance = 0;
	};
	struct Attribute {
		uint32_t offset  = 0;
		uint8_t  binding = 0;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;
};

// Everything a graphics pipeline is made of besides the stage modules its ids name. Aligned so
// that rendering.color_formats can be passed to Vulkan.
struct alignas(4) GraphicsPipelineKey {
	// VS, LS or mesh; HS; TES; PS. 0 marks an absent stage.
	std::array<uint64_t, 4>  stage_ids {};
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;

	bool operator==(const GraphicsPipelineKey& other) const;
};

#pragma pack(pop)

// Keys are compared and hashed as bytes.
static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>);

// A compiled stage with what pipeline creation needs from it besides the key.
struct ShaderModule {
	uint64_t                                     id     = 0;
	vk::ShaderModule                             module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	// VS or LS: the host format of each vertex input attribute.
	std::vector<vk::Format>                      vertex_formats;
	// Rectangle-list PS: the parameters it reads.
	std::vector<RectListParameter>               rect_parameters;
	// Indirect mesh draw: a task shader reads the records and launches the groups.
	bool                                         mesh_task = false;
};

// A draw's topology, shader stages (from PipelineCache::GetGraphicsPrograms) and targets.
// S511 (KYTY_DRS_NO_ZERO): color_info and vertex_info sit in unions so a default-initialized
// state does not construct the unused slots. Value-init (`DrawRenderState{}` and the designated
// call sites) still zeroes every slot. color_constructed / vertex_alive say how many are alive
// (8 and 3 after value-init; the fast path sets 1 and 1).
struct DrawRenderState {
	RenderDepthInfo depth_info;
	union {
		RenderColorInfo color_info[RENDER_COLOR_ATTACHMENTS_MAX];
	};
	uint32_t color_count = 0;
	// How many leading color_info entries have been constructed. Resolve constructs the next one
	// when color_count catches up. Value-init leaves all eight alive.
	uint32_t color_constructed = RENDER_COLOR_ATTACHMENTS_MAX;
	// Color targets the pixel program writes.
	uint32_t              mrt_mask = 0;
	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	bool                  primitive_restart_enable = false;
	bool                  ps_active                = false;
	union {
		ShaderVertexInputInfo vertex_info[3];
	};
	// 1: only slot 0 is constructed (VertexStages must not read slot 1). 3: all three are alive.
	uint32_t             vertex_alive = 3;
	ShaderPixelInputInfo ps_input_info;
	// Input: the user-data slots a GPU-read indirect draw patches.
	ShaderIndirectOffsets indirect_offsets;
	bool                  indirect_draw = false;

	// Constructs every color and vertex slot. Tests and KYTY_DRS_NO_ZERO=0 use this.
	DrawRenderState() {
		for (uint32_t i = 0; i < RENDER_COLOR_ATTACHMENTS_MAX; i++) {
			std::construct_at(&color_info[i]);
		}
		for (uint32_t i = 0; i < 3; i++) {
			std::construct_at(&vertex_info[i]);
		}
	}
	// S511: constructs color slot 0 and vertex slot 0 only. The unused slots stay unconstructed.
	struct Fast {};
	DrawRenderState(Fast, ShaderIndirectOffsets offsets, bool indirect) {
#if defined(KYTY_DRS_MUTANT_ZERO_ALL)
		for (uint32_t i = 0; i < RENDER_COLOR_ATTACHMENTS_MAX; i++) {
			std::construct_at(&color_info[i]);
		}
		for (uint32_t i = 0; i < 3; i++) {
			std::construct_at(&vertex_info[i]);
		}
		color_constructed = RENDER_COLOR_ATTACHMENTS_MAX;
		vertex_alive      = 3;
#else
		std::construct_at(&color_info[0]);
		std::construct_at(&vertex_info[0]);
		color_constructed = 1;
		vertex_alive      = 1;
#endif
		indirect_offsets = offsets;
		indirect_draw    = indirect;
	}

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		const uint32_t n = vertex_alive == 1 ? 1u : (vertex_info[1].stage ? 3u : 1u);
		return {vertex_info, n};
	}
};

// Writes the shader store on its own thread, running jobs in the order they were posted.
class ShaderStoreWriter {
public:
	ShaderStoreWriter();
	// Keeps the first `kept` bytes of the store at `path`, or starts it over with `header`.
	void Open(const std::filesystem::path& path, std::span<const uint8_t> header, uint64_t kept);
	// Whether the store is open.
	explicit operator bool() const { return !m_file.IsInvalid(); }
	// Compresses `record` and appends it on the store thread, in call order.
	void Append(std::vector<uint8_t>&& record);
	void Post(Common::UniqueFunction<void>&& job);
	// Runs the queued jobs and stops the thread; jobs posted later never run. Safe to call again.
	void Finish() { m_thread = {}; }

private:
	friend struct PipelineCacheTestAccess;

	Common::File                             m_file;
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	// Declared last, so it drains the queue and joins before the rest is destroyed.
	std::jthread m_thread;
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	// Without a shader store path nothing is kept across runs.
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap,
	              const std::filesystem::path& shader_store = {});
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// Finishes the queued writes and stops the store thread, then writes the driver cache if it
	// changed.
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		// The push constant range's stages, which every update names.
		vk::ShaderStageFlags    push_stages;
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
	};

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	// S490 (KYTY_CP_PIPE): what a pipeline worker computes for a draw from a snapshot of its registers, with GetGraphicsPrograms' results.
	struct OffThreadPrograms {
		std::array<ShaderVertexInputInfo, 3>   vertex_info {};
		ShaderPixelInputInfo                   pixel_info {};
		bool                                   ps_active = false;
		uint32_t                               mrt_mask  = 0;
		ShaderRecompiler::IR::ResourceSnapshot vs_res; // vertex_info[0].stage.resources / pixel_info.stage.resources point here when the call answered
		ShaderRecompiler::IR::ResourceSnapshot ps_res;
		uint32_t                               fail = 0; // 0 answered, else the code of FrameTiming::Counter::PipeFail1N..11N
	};
	// GetGraphicsPrograms for a thread other than the CP: it never compiles, adds an entry or writes a cached program; a draw whose programs are not all in the cache
	// (or whose resources would need the CP) is left to the CP (false, out.fail). The thread has to be marked (ResourcePlan SetScratchPerThread, SetSrtWalkerWorker),
	// read guest memory only through Memory::TryReadBackingWorker, and have a ScoutTableLog active.
	bool GetGraphicsProgramsOffThread(const HW::Context& context, const HW::UserConfig& user_config, const HW::Shader& shaders, OffThreadPrograms& out);
	// Null skips the dispatch.
	const ShaderModule* GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                      ShaderComputeInputInfo&      input_info);

	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info);

private:
	friend struct PipelineCacheTestAccess;

	struct Program {
		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// A deque: a module's address stays valid while the CP adds variants (a pipeline worker's result names modules the CP uses later).
		std::deque<ShaderModule>                     modules;
		bool                                         skip_dispatch = false;

		[[nodiscard]] std::span<const uint32_t> FunctionCode() const;
		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime);
		template <bool HasCalls>
		ShaderModule* FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                         uint32_t                                push_data_cursor);
	};

	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	// Variants of one source share a bucket; ProgramKey equality compares the state.
	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const { return key.hash; }
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	using ModuleMap = std::unordered_map<uint64_t, const ShaderModule*>;

	GraphicContext&                        m_graphics;
	ShaderRecompiler::IR::ImageHeapSource* m_image_heap;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash> m_program_cache;
	// Modules compiled per stage.
	std::array<uint32_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> m_compiled {};
	ProgramKey                             m_lookup_key;
	// S490: the CP adds programs and modules under the unique lock; a pipeline worker looks them up under the shared one.
	mutable std::shared_mutex              m_program_mutex;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	bool                          m_driver_cache_dirty = false;
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_graphics_pipelines;
	std::unordered_map<uint64_t, Pipeline>                                     m_compute_pipelines;
	std::unordered_map<uint64_t, ShaderTessellationInputInfo>                  m_tess_layouts;
	// Records everything first created if a shader store is open, and writes the driver cache.
	ShaderStoreWriter m_store;

	template <typename InputInfo>
	const ShaderModule* GetProgram(const ShaderParams& params, InputInfo& input_info,
	                               uint32_t& push_data_cursor);
	template <typename InputInfo>
	const ShaderModule* GetProgramOffThread(const ShaderParams& params, InputInfo& input_info, uint32_t& push_data_cursor,
	                                        ShaderRecompiler::IR::ResourceSnapshot& resources, uint32_t& fail);
	template <bool OffThread>
	bool GraphicsProgramsImpl(const HW::Context& context, const HW::UserConfig& user_config, const HW::Shader& shaders,
	                          std::span<ShaderVertexInputInfo, 3> vertex_info, ShaderPixelInputInfo& pixel_info, bool& ps_active_out,
	                          uint32_t& mrt_mask, OffThreadPrograms* off_thread, DrawRenderState* draw_state);
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions  options,
	                            ShaderRecompiler::TranslateResult translated,
	                            uint32_t                          push_data_cursor);
	void InitializeDriverCache();
	// Marks the driver cache changed and, once per interval, has the store thread write it.
	void CheckpointDriverCache();
	void WriteDriverCache() const;

	// pipelineSerialization.cpp
	void WarmUp(const std::filesystem::path& path);
	bool LoadRecord(std::span<const uint8_t>& records);
	void StoreModule(const ProgramKey& key, const Program& program, const ShaderModule& module,
	                 const std::vector<uint32_t>& spirv);
	void StorePipeline(const GraphicsPipelineKey& key);
	void StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout);
	// Every module by id, with id 0 naming an absent stage.
	[[nodiscard]] ModuleMap MapModules() const;
	// False when a stage id of the key is not in `modules`.
	bool CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                            const ModuleMap& modules);
};

// The shader store of the running title; empty unless --shader-cache is on in a Release build
// without debug dumps, which change the SPIR-V.
std::filesystem::path ShaderStorePath();
// Host capabilities that change the emitted SPIR-V.
ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics);

// Stages are indexed like GraphicsPipelineKey::stage_ids.
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
