#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/perVertexTransform.h"
#include "graphics/host_gpu/renderer/pvIndirect.h"
#include "graphics/host_gpu/renderer/pvRangeGen.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/indirectGpu.h"
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
struct ShaderBufferResource;
struct ShaderComputeInputInfo;
struct RenderDepthInfo;
struct RenderColorInfo;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawRenderState;
struct PerVertexPrototypePrograms;
class PvVsStoreRuntime;
class PvIndirectRuntime;
struct PvVsStoreDrawSetup;
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
struct IndirectGpuRequest {
	IndirectGpu::Mode   mode         = IndirectGpu::Mode::Off;
	uint64_t            args_addr    = 0;
	bool                declined     = false;
	IndirectGpu::Reason host_reason  = IndirectGpu::Reason::Eligible;
};

// S408 (KYTY_PV_INDIRECT): what the command processor tells the executor about an indirect draw that may use prepared args.
//   !have_cpu_args  the first call of an eligible candidate (KYTY_INDIRECT_GPU=1): the args have NOT been read. Mode On takes the draw
//                   from the prepared args when its entry is learned (else `declined`); the CPU then reads the args and calls again.
//   have_cpu_args   the old path: the CPU has read the args (cpu_args); the executor learns them, and in Shadow records the prepare
//                   compute next to the draw and compares its output with cpu_args two flips later.
struct PvIndirectRequest {
	PvIndirect::Mode mode              = PvIndirect::Mode::Off;
	uint64_t         args_addr         = 0;
	uint32_t         index_buffer_size = 0; // INDEX_BUFFER_SIZE, in indices
	bool             have_cpu_args     = false;
	PvIndirect::Args cpu_args;
};

// The draw the executor takes from the prepared args (On) or the old draw that gets a prepare next to it (Shadow).
struct PvIndirectDraw {
	PvIndirect::Mode    mode = PvIndirect::Mode::Off;
	uint64_t            args_addr = 0;
	PvIndirect::Fields  expected;
	uint32_t            cap      = 0;
	uint32_t            entry_id = 0;
	PvIndirect::Decline why      = PvIndirect::Decline::Other; // out, with `declined`
	bool                declined = false;                      // out: nothing has been drawn, take the old path
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
	PvIndirectRequest*  pv_indirect             = nullptr; // S408
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
	// S328: true when `state` needs no new pass because the open one is it (after a mid-pass eager split the open pass has its clears
	// stripped, so compare through the same normalisation BeginRendering uses).
	[[nodiscard]] bool SamePass(const RenderState& state) const;
	// S328: the pass that is (re)opened next has no clears (reopen of a split pass); lasts until that pass is closed.
	void ArmReopenLoad() const { m_strip_clears = true; }
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
	mutable int64_t     m_pass_end_ns   = 0;
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

	// Replays and clears any per-vertex-prototype draws batched by
	// ExecutePerVertexPrototype() (bench-notes.md 2026-09-27 "Kierunek A"). The PM4 interpreter
	// calls this before every op it does not itself know to be a compatible batchable draw, so a
	// pending batch is always flushed before anything else in the command stream can observe it;
	// a no-op when nothing is pending.
	void FlushPendingPrototypeBatch(CommandBuffer& buffer);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	// Sorted, disjoint ranges written by the bound images and shader buffers of a draw.
	void GatherWriteRanges(std::span<PreparedBindings* const> prepared_bindings);
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

	// ps_hash/slot (pixel stage only, 0 = none) let the control file's force_tex replace the T#.
	// hint is the S379 stand-in reason for this slot (null when the diag is off).
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource& resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value,
	                                            uint64_t ps_hash = 0, uint32_t slot = 0,
	                                            const NullDesc::Hint* null_hint = nullptr);
	void                         PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                                                     std::span<RenderColorInfo>         colors);

	// The description of a texture binding, kept for inputs seen before.
	[[nodiscard]] const TextureDescription&
	FindTextureDescription(const ShaderRecompiler::IR::ImageResource&   resource,
	                       const ShaderRecompiler::IR::DescriptorValue& value);

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
	                         DrawRenderState& state, vk::PrimitiveTopology topology,
	                         const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                         bool primitive_restart_enable, PvIndirectDraw* pv_indirect = nullptr);
	// S150 (KYTY_PV_VSSTORE=1): true when the draw takes the VS-store path (fills `setup`); counts the
	// fallback reason otherwise.
	[[nodiscard]] bool TryPvVsStoreDraw(const DrawCallInfo& draw, const DrawRenderState& state,
	                                    vk::PrimitiveTopology topology, bool primitive_restart_enable,
	                                    const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	                                    PvVsStoreDrawSetup& setup, const PvIndirectDraw* pv_indirect = nullptr,
	                                    PvIndirect::Decline* why = nullptr);

	// ---- S408 (KYTY_PV_INDIRECT), renderPvIndirect.cpp ----
	// The state of the args addresses seen, and the decision for one draw (learns when the request carries the args the CPU read).
	// `pv_candidate`: the draw's host path is a VS-store pair with nothing else in the way.
	[[nodiscard]] PvIndirectDraw PvIndirectDecide(const PvIndirectRequest& request, bool pv_candidate);
	// Moves the status rows to the current flip and applies what the GPU wrote two flips ago.
	void PvIndirectAdvance();
	// Notes the plan of a prepared draw (cap limit of the entry, the dry plan of a shadow draw).
	void PvIndirectNotePlan(const PvIndirectDraw& draw, const PvVsStorePlan& cap_plan, const PvVsStorePlan& real_plan,
	                        uint32_t stride_vec4);
	[[nodiscard]] PvIndirectRuntime& PvIndirectRt();
	struct PvIndirectEntryState {
		PvIndirect::Entry entry;
		uint64_t          addr      = 0;
		uint32_t          cap_limit = UINT32_MAX;
		// Shadow: the args the CPU read in a flip (compared with the prepared ones when the status is reported).
		struct Seen {
			uint64_t         flip = 0;
			bool             have = false;
			PvIndirect::Args cpu_args;
			bool             plan_checked = false;
		};
		std::array<Seen, 8> seen {};
	};
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
		bool                     unique             = false; // S134: captured per unique vertex (+ map)
		std::unique_ptr<Buffer>  captured;
		std::unique_ptr<Buffer>  map;               // S134: corner -> record map of the unique mode
		vk::DescriptorPool       pool               = nullptr;
		vk::DescriptorSet        extra_set          = nullptr;
	};
	std::unordered_map<uint64_t, CachedPrototypeCapture> m_prototype_capture_cache;
	PerVertexRangeMemo                                   m_prototype_range_memo;
	PvRangeGen::Memo                                     m_pv_range_gen; // S406 (KYTY_PV_RANGE_GEN)
	std::shared_ptr<PvVsStoreRuntime>                    m_pv_vsstore; // S150, created on the first VS-store draw
	std::shared_ptr<PvIndirectRuntime>                   m_pvi_rt;     // S408, created on the first prepared draw
	std::array<PvIndirectEntryState, PvIndirect::kMaxEntries> m_pvi_entries;
	uint32_t                                             m_pvi_count = 0;

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
		std::unique_ptr<Buffer> owned_map; // S134
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
	};
	PendingPrototypeBatch m_prototype_batch;
	struct TextureDescriptionEntry {
		ShaderRecompiler::IR::ImageResource   inputs;
		ShaderRecompiler::IR::DescriptorValue value;
		TextureDescription                    description;
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
