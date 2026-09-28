#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
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

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
struct PerVertexPrototypePrograms;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class CommandBufferDebugOp : uint32_t {
	DispatchDirect,
	DrawIndex,
	DrawIndexAuto,
	EopWrite,
	EopInterrupt,
	EopWriteBack,
	EopFlip,
	EopWriteBackFlip,
	EopOnlyFlip,
	DispatchIndirect,
	Unknown,
};

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
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
};

// Args for a GNM indexed indirect draw whose count/instance/offset fields are read by the GPU
// itself (native vkCmdDrawIndexedIndirect) instead of by the CPU -- see DrawIndexIndirect()'s
// comment for why and its eligibility conditions; anything not eligible reads `args_addr` on the
// CPU internally, same as before this existed.
struct DrawIndexIndirectArgs {
	uint64_t args_addr                 = 0; // guest DrawIndexedIndirectCommand-layout args
	uint32_t index_type_and_size       = 0;
	uint64_t index_addr                = 0; // base of the guest index buffer (unoffset)
	// Index COUNT (VGT_INDEX_BUFFER_SIZE is a count of indices, not bytes -- see
	// CommandProcessor::SetIndexBufferSize's caller). 0 = unknown; forces the CPU-read fallback.
	uint32_t index_buffer_size = 0;
	uint32_t render_target_slice_offset = 0;
};

struct DrawAutoArgs {
	uint32_t         vertex_count               = 0;
	uint32_t         instance_count             = 0;
	uint32_t         first_vertex               = 0;
	uint32_t         first_instance             = 0;
	DrawOffsetSource offset_source              = DrawOffsetSource::DrawState;
	uint32_t         render_target_slice_offset = 0;
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
	void EndRendering() const;

	[[nodiscard]] vk::CommandBuffer Handle() const;

	// Graphics dynamic state last recorded by the draw path, so identical values are not
	// recorded again. Anything else that records dynamic state or binds a graphics pipeline with
	// static state must call InvalidateDynamicState().
	struct DynamicState {
		bool                              valid          = false;
		uint32_t                          viewport_count = 0;
		std::array<vk::Viewport, 16>      viewports {};
		std::array<vk::Rect2D, 16>        scissors {};
		float                             line_width = 0.0f;
		std::array<float, 4>              blend_constants {};
		vk::Bool32                        depth_test_enable  = VK_FALSE;
		vk::Bool32                        depth_write_enable = VK_FALSE;
		vk::CompareOp                     depth_compare_op   = vk::CompareOp::eNever;
		vk::Bool32                        depth_bias_enable  = VK_FALSE;
		std::array<float, 3>              depth_bias {};
		bool                              depth_bias_known    = false;
		vk::Bool32                        stencil_test_enable = VK_FALSE;
		std::array<vk::StencilOpState, 2> stencil {};
		std::array<bool, 2>               stencil_known {};
	};
	[[nodiscard]] DynamicState& Dynamic() const noexcept { return m_dynamic; }
	void                        InvalidateDynamicState() const noexcept { m_dynamic.valid = false; }
	[[nodiscard]] bool          IsRendering() const noexcept { return m_rendering; }
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
	// Set by Handle(): something (possibly) recorded since Begin(). Over-reporting is harmless.
	mutable bool        m_recorded    = false;
	mutable DynamicState m_dynamic;
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
	                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode);
	void DispatchIndirect(uint64_t submit_id, CommandBuffer& buffer, uint64_t args_addr,
	                      uint32_t mode);

	// Indexed indirect draw. When eligible (see the .cpp), reads `args.args_addr` on the GPU
	// itself via vkCmdDrawIndexedIndirect -- no CPU synchronization with whatever GPU work wrote
	// it, unlike the CPU std::memcpy this replaces (bench-notes.md 2026-09-28, "DrawIndirect
	// readback"). Not eligible falls back to reading it on the CPU internally, same behavior as
	// before this existed.
	void DrawIndexIndirect(uint64_t submit_id, CommandBuffer& buffer,
	                       const DrawIndexIndirectArgs& args);

	// Replays and clears any per-vertex-prototype draws batched by
	// ExecutePerVertexPrototype() (bench-notes.md 2026-09-27 "Kierunek A"). The PM4 interpreter
	// calls this before every op it does not itself know to be a compatible batchable draw, so a
	// pending batch is always flushed before anything else in the command stream can observe it;
	// a no-op when nothing is pending.
	void FlushPendingPrototypeBatch(CommandBuffer& buffer);
	// TEMP diagnostic: why the next FlushPendingPrototypeBatch() call happens.
	const char* m_diag_flush_reason = "unset";
	uint32_t    m_diag_flush_opcode = 0;

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void FindBuffers(PreparedBindings& bindings);
	void RebindBuffers(PreparedBindings& bindings);
	void RebindImages(PreparedBindings& bindings);
	void TransitionBoundResources(CommandBuffer&                     buffer,
	                              std::span<PreparedBindings* const> bindings,
	                              bool                               prototype_vertex_capture);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);

private:
	void DrawIndex(uint64_t submit_id, CommandBuffer& buffer, const DrawIndexArgs& args);
	void DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		std::optional<PreparedBindings> pixel;
	};

	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);
	void                         PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                                                     std::span<RenderColorInfo>         colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slice_offset, uint32_t render_target_slot,
	                              bool ignore_target_mask = false, bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                          uint32_t         render_target_slice_offset,
	                                          DrawRenderState& state);
	// `indirect_args_buffer` non-null draws with vkCmdDrawIndexedIndirect from
	// [indirect_args_buffer, indirect_args_offset) instead of the concrete counts in `draw`/
	// `emit` (only read otherwise); see DrawIndexIndirect(), the only caller that sets it.
	void ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer, const DrawCallInfo& draw,
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable, vk::Buffer indirect_args_buffer = nullptr,
	                         vk::DeviceSize indirect_args_offset = 0);
	void ExecutePerVertexPrototype(uint64_t submit_id, CommandBuffer& buffer,
	                               const DrawCallInfo& draw, DrawRenderState& state,
	                               vk::PrimitiveTopology topology, const DrawEmitInfo& emit,
	                               const DrawIndexBufferSource& index_source,
	                               bool                         primitive_restart_enable);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer,
	                                              uint32_t       render_target_slice_offset);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool        TryConsumeComputeImageClear(const ShaderComputeInputInfo& input,
	                                                      CommandBuffer& command, uint32_t group_x,
	                                                      uint32_t group_y, uint32_t group_z,
	                                                      uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                      m_graphics_bindings;
	PreparedBindings                      m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	std::vector<vk::DescriptorBufferInfo> m_descriptor_buffers;
	std::vector<vk::DescriptorImageInfo>  m_descriptor_images;
	std::vector<vk::WriteDescriptorSet>   m_descriptor_writes;
	std::vector<uint32_t>                 m_image_occurrences;
	struct CachedDescriptorSet {
		uint64_t          hash                = 0;
		uint64_t          pool_generation     = 0;
		uint64_t          resource_generation = 0;
		vk::DescriptorSet set                 = nullptr;
	};
	std::unordered_map<vk::DescriptorSetLayout, CachedDescriptorSet> m_last_descriptor_sets;
	std::vector<uint8_t>                                             m_descriptor_key;

	// Content-addressed cache of per-vertex-prototype capture output (see perVertexPrototypeDraw.inc,
	// bench-notes.md 2026-09-27 "Kierunek B"). Keyed by a hash of {guest vertex-buffer address,
	// VS hash, PS hash}; validated on lookup against a hash of the actual bytes/params that feed
	// the capture compute stage, so a key collision or a reused address can only cause an
	// unnecessary miss, never a false hit.
	struct CachedPrototypeCapture {
		uint64_t                 addr               = 0;
		uint64_t                 vs_hash            = 0;
		uint64_t                 ps_hash            = 0;
		uint64_t                 content_hash       = 0;
		uint64_t                 count64            = 0;
		uint32_t                 record_stride_vec4 = 0;
		uint64_t                 last_used_tick     = 0;
		std::unique_ptr<Buffer>  captured;
		vk::DescriptorPool       pool               = nullptr;
		vk::DescriptorSet        extra_set          = nullptr;
	};
	std::unordered_map<uint64_t, CachedPrototypeCapture> m_prototype_capture_cache;

	// Batching of consecutive per-vertex-prototype replay draws into one render pass (see
	// perVertexPrototypeDraw.inc, bench-notes.md 2026-09-27 "Kierunek A"). A draw is appended here
	// instead of replayed immediately when it matches `key` (below) exactly; FlushPendingBatch()
	// -- called by the PM4 interpreter before anything that is not itself a matching draw --
	// replays every pending item inside a single BeginRendering/EndRendering pair.
	struct PendingPrototypeReplayItem {
		vk::DescriptorSet      extra_set = nullptr;
		uint32_t                count    = 0;
		ShaderVertexInputInfo    vs;
		PreparedBindings         vertex_bindings;
		PreparedBindings         pixel_bindings;
		// Only set when this item's capture output is not owned by m_prototype_capture_cache
		// (cache disabled, or this VS has side-effecting writes): owns the buffer/pool until the
		// batch flush records the replay draw, then is freed the same deferred way a non-batched
		// draw would free them.
		std::unique_ptr<Buffer> owned_captured;
		vk::DescriptorPool      owned_pool = nullptr;
		// Resolved when the draw is issued, from the registers current then: the guest may set
		// registers for later draws before this batch is flushed (see ProcessPm4's flush hook).
		PipelineCache::Pipeline*          pipeline = nullptr;
		std::shared_ptr<const HW::Context> registers;
	};
	struct PendingPrototypeBatch {
		bool                                       active      = false;
		uint64_t                                   vs_hash     = 0;
		uint64_t                                   ps_hash     = 0;
		std::array<RenderColorInfo, RENDER_COLOR_ATTACHMENTS_MAX> color_info {};
		uint32_t                                   color_count = 0;
		RenderDepthInfo                            depth_info;
		bool                                        ps_active   = false;
		ShaderPixelInputInfo                        ps_input_info;
		vk::PrimitiveTopology                       topology    = vk::PrimitiveTopology::eTriangleList;
		const PerVertexPrototypePrograms*           programs    = nullptr;
		std::vector<PendingPrototypeReplayItem>     items;
		// Diagnostic (bug #4 hunt, 2026-09-28): the vk::CommandBuffer in use when this batch
		// started. If a submit happens between an item's capture and the batch's eventual flush --
		// e.g. triggered internally by some unrelated operation's own buffer-capacity management,
		// not tied to any PM4 opcode our flush hooks intercept -- the capture's barrier is stuck in
		// an already-submitted, separate command buffer and does not protect a replay draw recorded
		// into a new one. See FlushPendingPrototypeBatch().
		vk::CommandBuffer command_buffer_at_start = nullptr;
	};
	PendingPrototypeBatch m_prototype_batch;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
