#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pvIndirect.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/indirectGpu.h"
#include "graphics/host_gpu/renderer/meshIndirect.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <memory>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

namespace NullDesc {
struct Hint;
}

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
class MeshIndirectRuntime;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	DispatchIndirect = 9,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// S404 (KYTY_INDIRECT_GPU): one indexed indirect draw handed to the executor without its args being read.
//   On      index_count / instance_count of the DrawIndexArgs are placeholders. The executor records vkCmdDrawIndexedIndirect(args_addr)
//           or sets `declined` (before recording anything) and the caller takes the CPU path.
//   Shadow  the DrawIndexArgs are the real ones; the executor only reports what it would have decided (host_reason).
// S472 (KYTY_INDIRECT_GPU_MESH): what the command processor tells the executor about an indexed indirect draw that may be a mesh draw.
//   On      the first call of an eligible candidate (together with IndirectGpuRequest, args not read): a mesh shader takes the draw as
//           prepare compute + vkCmdDrawMeshTasksIndirectEXT, any other shader declines (the caller takes the CPU path).
//   Shadow  the call that carries the args the CPU read (cpu_args): the executor records the prepare compute next to the draw and compares.
struct MeshIndirectRequest {
	MeshIndirect::Mode mode      = MeshIndirect::Mode::Off;
	uint64_t           args_addr = 0;
	MeshIndirect::Args cpu_args {};
	uint32_t           index_buffer_size = 0; // m_index_buffer_size, the clamp of the CPU path
	bool               declined = false; // set by the executor (before anything of the draw is recorded): the caller takes the CPU path
};

struct IndirectGpuRequest {
	IndirectGpu::Mode   mode         = IndirectGpu::Mode::Off;
	uint64_t            args_addr    = 0;
	bool                declined     = false;
	IndirectGpu::Reason host_reason  = IndirectGpu::Reason::Eligible;
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t address = 0;
	uint32_t count   = 0;
	uint32_t stride  = 0;
};

struct DrawIndexArgs {
	uint32_t         index_count                = 0;
	const void*      index_addr                 = nullptr;
	uint32_t         instance_count             = 0;
	uint32_t         index_type_and_size        = 0;
	int32_t          base_vertex                = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	IndirectGpuRequest* indirect_gpu            = nullptr; // S404
	MeshIndirectRequest* mesh_indirect          = nullptr; // S472
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
	DrawIndirectArgs indirect;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const;

	void SetDebugInfo(uint32_t op, uint64_t submit_id, uint32_t arg0 = 0, uint32_t arg1 = 0,
	                  uint32_t arg2 = 0, uint32_t arg3 = 0, uint64_t arg4 = 0);
	void BeginRendering(const RenderState& state) const;
	// S328: true when `state` needs no new pass because the open one is it (after a mid-pass eager split the open pass has its clears
	// stripped, so compare through the same normalisation BeginRendering uses).
	[[nodiscard]] bool SamePass(const RenderState& state) const;
	// S328: the pass that is (re)opened next has no clears (reopen of a split pass); lasts until that pass is closed.
	void ArmReopenLoad() const { m_strip_clears = true; }
	// S410: a flush closed the pass `before` (not at a draw, so the next draw is not known). If the next pass opened is exactly `before`, it is
	// the continuation of that pass and loads (no clear repeated); any other state, in particular one with a clear `before` did not have, is opened as asked.
	void ArmReopenLoadIfSame(const RenderState& before) const {
		m_reopen_if_same = true;
		m_reopen_state   = before;
	}
	void EndRendering() const;

	[[nodiscard]] vk::CommandBuffer Handle() const;
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer          = nullptr;
	uint32_t            m_debug_op        = 0;
	uint64_t            m_debug_submit_id = 0;
	uint32_t            m_debug_arg0      = 0;
	uint32_t            m_debug_arg1      = 0;
	uint32_t            m_debug_arg2      = 0;
	uint32_t            m_debug_arg3      = 0;
	uint64_t            m_debug_arg4      = 0;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	// S327: the scheduler arms this at a drain end; the next real EndRendering stamps the time (seg_first_pass_end_ns).
	mutable bool        m_strip_clears = false; // S328
	mutable bool        m_note_pass_end = false;
	mutable bool        m_reopen_if_same = false; // S410
	mutable RenderState m_reopen_state;           // S410
	mutable int64_t     m_t_first_rec   = 0;      // S410 (KYTY_DIAG_STARVE=1): steady clock at the first command recorded into this buffer
	mutable int64_t     m_pass_end_ns   = 0;
	mutable int32_t     m_ts_slot       = -1;     // S477 D3 (KYTY_GPU_TIMESTAMPS): query slot of this buffer's start/end timestamps
	// Set by Handle(): something (possibly) recorded since Begin(). Over-reporting is harmless.
	mutable bool        m_recorded    = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
};

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(uint64_t submit_id, CommandBuffer& buffer, uint32_t thread_group_x,
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode, bool async_compute);


	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	// Sorted, disjoint ranges written by the bound images and shader buffers of a draw.
	void GatherWriteRanges(std::span<PreparedBindings* const> prepared_bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);

private:
	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer,
	                                          const DrawCallInfo& draw,
	                                          uint32_t            render_target_slice_offset,
	                                          DrawRenderState& state);
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, const DrawEmitInfo& emit,
	                         const DrawIndexBufferSource& index_source, MeshIndirectRequest* mesh_gpu = nullptr);
	[[nodiscard]] MeshIndirectRuntime& MeshIndirectRt(); // S472
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t render_target_slice_offset);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer,
	                                                       uint32_t render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<GuestRange>               m_write_ranges;
	struct CachedDescriptorSet {
		uint64_t          hash                = 0;
		uint64_t          pool_generation     = 0;
		uint64_t          resource_generation = 0;
		vk::DescriptorSet set                 = nullptr;
	};
	std::unordered_map<vk::DescriptorSetLayout, CachedDescriptorSet> m_last_descriptor_sets;
	std::vector<uint8_t>                                             m_descriptor_key;

	std::shared_ptr<MeshIndirectRuntime>                 m_mesh_ind_rt; // S472, created on the first mesh indirect draw
	struct TextureDescriptionEntry {
		ShaderRecompiler::IR::ImageResource   inputs;
		ShaderRecompiler::IR::DescriptorValue value;
		TextureDescription                    description;
		// S438 (KYTY_TEX_RESOLVE_MEMO): the image this description resolved to; valid until a registration touches its pages.
		TexMemo::Entry<ImageId>               memo;
	};
	// Descriptions by input hash; an entry is used only when its inputs compare equal.
	std::unordered_map<uint64_t, TextureDescriptionEntry> m_texture_descriptions;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

// Sorts nonempty ranges and merges those that overlap or touch; exits on an empty or wrapping
// range.
void MergeGuestRanges(std::vector<GuestRange>& ranges);
// True when a scalar read is invalid or overlaps one of the sorted, disjoint written ranges.
[[nodiscard]] bool ScalarReadsMayOverlap(std::span<const std::pair<uint64_t, uint64_t>> reads,
                                         std::span<const GuestRange>                    writes);

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
