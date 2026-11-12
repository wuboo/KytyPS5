#include "common/assert.h"
#include "common/common.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/gpuLabelsVk.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/eagerSubmit.h"
#include "graphics/host_gpu/renderer/gpuDone.h"
#include "graphics/host_gpu/renderer/gpuTimestamps.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameTiming.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <bit>
#include <cstring>
#include <utility>
namespace Libs::Graphics {

CommandBuffer::CommandBuffer(CommandScheduler& scheduler)
    : m_context(scheduler.Context()), m_graphics(scheduler.Graphics()) {
	m_emit.SetOwner(this, &CommandBuffer::EmitDrainThunk);
	static std::atomic_bool logged = false;
	if (!logged.exchange(true, std::memory_order_relaxed)) {
		std::fprintf(stderr, "[cp-emit] mode=%s\n", CpEmit::ModeName(CpEmit::EnvMode()));
	}
}

// S561: MasterSemaphore::Wait (or another stream of the thread) drains this buffer's IR through the thread's pending slot.
void CommandBuffer::EmitDrainThunk(void* self, CpEmit::Reason why) {
	const auto* buffer = static_cast<const CommandBuffer*>(self);
	buffer->m_emit.Drain(buffer->m_buffer, why);
}

void CommandBuffer::FlushEmitStats() const {
	if (m_emit.Current() == CpEmit::Mode::Off) return;
	// The record and replay times are TSC ticks until the clock is calibrated (the first ~20 ms); kept until then.
	static thread_local uint64_t carry_rec = 0, carry_replay = 0;
	const auto stats = m_emit.TakeStats();
	using FrameTiming::Counter;
	FrameTiming::Add(Counter::EmitOpsN, stats.ops);
	FrameTiming::Add(Counter::EmitBytes, stats.bytes);
	FrameTiming::Add(Counter::EmitDrainN, stats.drains);
	FrameTiming::Add(Counter::EmitDrainHandleN, stats.by[static_cast<size_t>(CpEmit::Reason::Handle)]);
	FrameTiming::Add(Counter::EmitDrainConvN, stats.by[static_cast<size_t>(CpEmit::Reason::Conv)]);
	FrameTiming::Add(Counter::EmitDrainSubmitN, stats.by[static_cast<size_t>(CpEmit::Reason::Submit)]);
	FrameTiming::Add(Counter::EmitDrainWaitN, stats.by[static_cast<size_t>(CpEmit::Reason::Wait)]);
	FrameTiming::Add(Counter::EmitDrainOtherN, stats.by[static_cast<size_t>(CpEmit::Reason::Other)]);
	FrameTiming::Add(Counter::EmitNeN, stats.ne);
	FrameTiming::Add(Counter::EmitFallbackN, stats.fallback);
	carry_rec += stats.rec_ticks;
	carry_replay += stats.replay_ticks;
	if (const double ns_per_tick = CpEmit::NsPerTick(); ns_per_tick != 0.0) {
		FrameTiming::Add(Counter::EmitRecordNs, static_cast<uint64_t>(static_cast<double>(carry_rec) * ns_per_tick));
		FrameTiming::Add(Counter::EmitReplayNs, static_cast<uint64_t>(static_cast<double>(carry_replay) * ns_per_tick));
		carry_rec    = 0;
		carry_replay = 0;
	}
}

void CommandBuffer::Begin() {
	EXIT_IF(m_rendering || m_pending_shader_writes || IsInvalid() || m_emit.Pending());
	auto buffer = Handle();

	vk::CommandBufferBeginInfo begin_info {};
	begin_info.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;

	auto result = buffer.begin(&begin_info);

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	m_recorded       = false;
	m_t_first_rec    = 0;
	m_reopen_if_same = false;
	m_bound_graphics_pipeline = nullptr;
	m_ts_slot        = GpuTs::OnBegin(m_graphics, buffer); // S477 D3 DIAG
}

void CommandBuffer::End() const {
	EndRendering();
	m_emit.Drain(m_buffer, CpEmit::Reason::Submit); // S561: everything the IR sites wrote, before the buffer ends
	FlushEmitStats();
	auto buffer = Handle();
	GpuTs::OnEnd(m_graphics, buffer, m_ts_slot); // S477 D3 DIAG

	auto result = buffer.end();

	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

bool CommandBuffer::SamePass(const RenderState& state) const {
	if (!m_rendering) return false;
	if (m_render_state == state) return true;
	if (!m_strip_clears) return false;
	RenderState stripped = state;
	Sync::StripClearsOnReopen(stripped);
	return m_render_state == stripped;
}

void CommandBuffer::BeginRendering(const RenderState& wanted) const {
	if (SamePass(wanted)) {
		return;
	}
	RenderState state = wanted;
	if (m_reopen_if_same && !m_rendering) { // S410
		m_reopen_if_same = false;
		Sync::ReopenAfterFlush(m_reopen_state, state);
	}
	// S328: the pass armed by a mid-pass eager split is opened here: load, never clear again (an open stripped pass that is replaced is not).
	if (m_strip_clears && !m_rendering) Sync::StripClearsOnReopen(state);
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.num_color_attachments > RENDER_COLOR_ATTACHMENTS_MAX);
	EndRendering();
	FrameTiming::Add(FrameTiming::Counter::RenderPasses);

	std::array<vk::RenderingAttachmentInfo, RENDER_COLOR_ATTACHMENTS_MAX> colors {};
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		const auto& attachment = state.color_attachments[i];
		colors[i].imageView    = attachment.image_view;
		colors[i].imageLayout  = attachment.image_layout;
		colors[i].loadOp       = attachment.is_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
		colors[i].clearValue.color.uint32[0] = attachment.clear_value[0];
		colors[i].clearValue.color.uint32[1] = attachment.clear_value[1];
		colors[i].storeOp      = vk::AttachmentStoreOp::eStore;
	}

	const auto&                 depth_stencil = state.depth_stencil_attachment;
	vk::RenderingAttachmentInfo depth {};
	depth.imageView   = depth_stencil.image_view;
	depth.imageLayout = depth_stencil.image_layout;
	depth.loadOp =
	    depth_stencil.depth_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	depth.storeOp                       = vk::AttachmentStoreOp::eStore;
	depth.clearValue.depthStencil.depth = std::bit_cast<float>(depth_stencil.clear_value[0]);

	vk::RenderingAttachmentInfo stencil {};
	stencil.imageView   = depth_stencil.image_view;
	stencil.imageLayout = depth_stencil.image_layout;
	stencil.loadOp =
	    depth_stencil.stencil_clear ? vk::AttachmentLoadOp::eClear : vk::AttachmentLoadOp::eLoad;
	stencil.storeOp                         = vk::AttachmentStoreOp::eStore;
	stencil.clearValue.depthStencil.stencil = depth_stencil.clear_value[1];

	vk::RenderingInfo rendering {};
	rendering.renderArea.extent    = {state.width, state.height};
	rendering.layerCount           = state.num_layers;
	rendering.colorAttachmentCount = state.num_color_attachments;
	rendering.pColorAttachments    = colors.data();
	rendering.pDepthAttachment     = depth_stencil.has_depth ? &depth : nullptr;
	rendering.pStencilAttachment   = depth_stencil.has_stencil ? &stencil : nullptr;
	static std::atomic<uint32_t> pass_id {0};
	if (CpEmit::PassToIr(m_emit) && !GpuLabel::Active(GpuLabel::Class::GuestRp)) {
		// S561: the IR already holds commands, so the pass opens after them (no label: a labelled pass records directly below, which drains).
		pass_id.fetch_add(1, std::memory_order_relaxed);
		Emit().beginRendering(rendering);
	} else {
		GpuLabel::With(
		    Handle(), GpuLabel::Class::GuestRp, [&] { Handle().beginRendering(rendering); }, "G.rp.%ux%u.%u.%u", state.width, state.height,
		    state.num_color_attachments, pass_id.fetch_add(1, std::memory_order_relaxed));
	}
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state = state;
	// The open pass holds the cleared contents, so a later clear begins a new pass.
	m_render_state.depth_stencil_attachment.depth_clear   = false;
	m_render_state.depth_stencil_attachment.stencil_clear = false;
	m_rendering                                           = true;
}

void CommandBuffer::EndRendering() const {
	if (m_rendering) {
		m_strip_clears = false; // S328: a closed pass ends the reopen
		// S561: to the IR only while it holds commands; callers holding a raw handle (Image::Transit) see an empty IR, so it records directly.
		if (CpEmit::PassToIr(m_emit)) {
			Emit().endRendering();
		} else {
			Handle().endRendering();
		}
		if (m_note_pass_end) {
			m_note_pass_end = false;
			m_pass_end_ns   = FrameTiming::SteadyNowNs();
		}
		m_rendering    = false;
		m_render_state = {};
	}
	// S511: a deferred shader-write barrier is recorded here, including when the pass was already closed.
	if (m_pending_shader_writes) {
		ShaderWriteBarrier(Handle(), std::exchange(m_pending_shader_writes, {}));
		FrameTiming::Add(FrameTiming::Counter::DeferWriteFlushN);
	}
}

} // namespace Libs::Graphics
