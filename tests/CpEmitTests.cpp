// S561 KYTY_CP_EMIT (cpEmit.h): the IR of the recording sites replays in the order of the direct calls.
// One script of a draw-like sequence (IR sites through CmdT, direct sites through TakeRaw / a raw handle, Begin/EndRendering through PassToIr,
// a conversion of the IR handle, a GPU wait through the thread's pending slot, enough draws to fill several arena chunks) runs three times:
// mode 0 (everything direct: the reference), sync (IR replayed at the drains) and shadow (direct + IR digest). The fake command buffer logs a
// digest per command, so sync must log the reference sequence and shadow must log it with emit_ne = 0. Caller memory (descriptor infos, push
// constants, attachments) is overwritten after each IR command and before its drain: the IR must hold values.
// Mutants (must fail, WILL_FAIL): NO_DRAIN (a direct site does not drain first), SHALLOW (push-descriptor infos keep the caller's pointers),
// UNCOND_PASS (Begin/EndRendering go to the IR even when it is empty, so a raw handle taken before them records out of order).
//
// S565 (thread mode): the same script with an emitter thread replaying the segments (several configurations: tiny segments, batch 1 and 8,
// steal on/off, a slow command buffer). The log must be the reference again, which checks the order across the two threads; the fake
// command buffer also checks the ownership rule: a call on the recording thread while the emitter has a segment left (the recorder touches
// the command buffer during the emission) or two calls at once fail the test. At each End the emitter must have replayed everything.
// Mutants (WILL_FAIL): NO_SUBMIT_DRAIN (the drain before the submit does not wait for the emitter), TOUCH_DURING_EMIT (a direct site
// records while the emitter still replays). Built with -fsanitize=thread the thread runs are also a data-race check of the hand-over.

#include "graphics/host_gpu/renderer/cpEmit.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::CpEmit;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "CpEmitTests: failed: %s\n", what);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

// What a fake command buffer logs into, and the thread-mode ownership probe.
struct Ctx {
	std::vector<uint64_t>* log = nullptr;
	std::mutex             mutex; // the log of a mutant may be written by two threads at once: no crash, the probe reports it
	std::atomic<int>       inside {0};
	const Emitter*         emitter = nullptr; // thread mode
	std::thread::id        recorder;
	uint32_t               slow_spins = 0; // per command, to widen the emitter's window

	void Note(uint64_t value) {
		if (inside.fetch_add(1, std::memory_order_acq_rel) != 0) Check(false, "two threads record into the command buffer at once");
		if (emitter != nullptr && std::this_thread::get_id() == recorder && !emitter->IdleNow() && !emitter->HeldByRecorder()) {
			Check(false, "the recorder touches the command buffer while the emitter has segments left");
		}
		for (uint32_t i = 0; i < slow_spins; i++) CpuRelax();
		{
			std::lock_guard lock(mutex);
			log->push_back(value);
		}
		inside.fetch_sub(1, std::memory_order_acq_rel);
	}
};

// Logs one digest per command (the same HashSink the shadow uses, one fresh per command).
struct FakeRaw {
	Ctx* ctx = nullptr;

#define KYTY_FAKE(name, params, args)                                                                                                            \
	void name params const {                                                                                                                       \
		HashSink h;                                                                                                                                \
		h.name args;                                                                                                                               \
		ctx->Note(h.Value());                                                                                                                      \
	}
	KYTY_FAKE(pipelineBarrier2, (const vk::DependencyInfo& d), (d))
	KYTY_FAKE(pipelineBarrier,
	          (vk::PipelineStageFlags a, vk::PipelineStageFlags b, vk::DependencyFlags c, uint32_t nm, const vk::MemoryBarrier* m, uint32_t nb,
	           const vk::BufferMemoryBarrier* bb, uint32_t ni, const vk::ImageMemoryBarrier* ii),
	          (a, b, c, nm, m, nb, bb, ni, ii))
	KYTY_FAKE(pushConstants, (vk::PipelineLayout l, vk::ShaderStageFlags s, uint32_t o, uint32_t n, const void* d), (l, s, o, n, d))
	KYTY_FAKE(pushDescriptorSetKHR, (vk::PipelineBindPoint bp, vk::PipelineLayout l, uint32_t set, uint32_t n, const vk::WriteDescriptorSet* w),
	          (bp, l, set, n, w))
	KYTY_FAKE(bindDescriptorSets,
	          (vk::PipelineBindPoint bp, vk::PipelineLayout l, uint32_t f, uint32_t n, const vk::DescriptorSet* s, uint32_t nd, const uint32_t* d),
	          (bp, l, f, n, s, nd, d))
	KYTY_FAKE(bindPipeline, (vk::PipelineBindPoint bp, vk::Pipeline p), (bp, p))
	KYTY_FAKE(bindVertexBuffers2,
	          (uint32_t f, uint32_t n, const vk::Buffer* b, const vk::DeviceSize* o, const vk::DeviceSize* s, const vk::DeviceSize* st),
	          (f, n, b, o, s, st))
	KYTY_FAKE(bindIndexBuffer, (vk::Buffer b, vk::DeviceSize o, vk::IndexType t), (b, o, t))
	KYTY_FAKE(setViewportWithCount, (uint32_t n, const vk::Viewport* v), (n, v))
	KYTY_FAKE(setScissorWithCount, (uint32_t n, const vk::Rect2D* r), (n, r))
	KYTY_FAKE(setDepthTestEnable, (vk::Bool32 v), (v))
	KYTY_FAKE(setDepthWriteEnable, (vk::Bool32 v), (v))
	KYTY_FAKE(setDepthCompareOp, (vk::CompareOp v), (v))
	KYTY_FAKE(setDepthBoundsTestEnable, (vk::Bool32 v), (v))
	KYTY_FAKE(setDepthBounds, (float a, float b), (a, b))
	KYTY_FAKE(setDepthBiasEnable, (vk::Bool32 v), (v))
	KYTY_FAKE(setDepthBias, (float a, float b, float c), (a, b, c))
	KYTY_FAKE(setStencilTestEnable, (vk::Bool32 v), (v))
	KYTY_FAKE(setStencilOp, (vk::StencilFaceFlags f, vk::StencilOp a, vk::StencilOp b, vk::StencilOp c, vk::CompareOp d), (f, a, b, c, d))
	KYTY_FAKE(setStencilReference, (vk::StencilFaceFlags f, uint32_t v), (f, v))
	KYTY_FAKE(setStencilWriteMask, (vk::StencilFaceFlags f, uint32_t v), (f, v))
	KYTY_FAKE(setStencilCompareMask, (vk::StencilFaceFlags f, uint32_t v), (f, v))
	KYTY_FAKE(setCullMode, (vk::CullModeFlags v), (v))
	KYTY_FAKE(setFrontFace, (vk::FrontFace v), (v))
	KYTY_FAKE(setBlendConstants, (const float c[4]), (c))
	KYTY_FAKE(setLineWidth, (float v), (v))
	KYTY_FAKE(setAttachmentFeedbackLoopEnableEXT, (vk::ImageAspectFlags v), (v))
	KYTY_FAKE(draw, (uint32_t a, uint32_t b, uint32_t c, uint32_t d), (a, b, c, d))
	KYTY_FAKE(drawIndexed, (uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e), (a, b, c, d, e))
	KYTY_FAKE(drawIndirect, (vk::Buffer b, vk::DeviceSize o, uint32_t n, uint32_t s), (b, o, n, s))
	KYTY_FAKE(drawIndexedIndirect, (vk::Buffer b, vk::DeviceSize o, uint32_t n, uint32_t s), (b, o, n, s))
	KYTY_FAKE(drawMeshTasksEXT, (uint32_t x, uint32_t y, uint32_t z), (x, y, z))
	KYTY_FAKE(drawMeshTasksIndirectEXT, (vk::Buffer b, vk::DeviceSize o, uint32_t n, uint32_t s), (b, o, n, s))
	KYTY_FAKE(beginRendering, (const vk::RenderingInfo& r), (r))
	KYTY_FAKE(endRendering, (), ())
	KYTY_FAKE(dispatch, (uint32_t x, uint32_t y, uint32_t z), (x, y, z))
	KYTY_FAKE(dispatchBase, (uint32_t a, uint32_t b, uint32_t c, uint32_t x, uint32_t y, uint32_t z), (a, b, c, x, y, z))
	KYTY_FAKE(dispatchIndirect, (vk::Buffer b, vk::DeviceSize o), (b, o))
	KYTY_FAKE(copyBuffer, (vk::Buffer s, vk::Buffer d, uint32_t n, const vk::BufferCopy* r), (s, d, n, r))
	KYTY_FAKE(copyBufferToImage, (vk::Buffer s, vk::Image d, vk::ImageLayout l, uint32_t n, const vk::BufferImageCopy* r), (s, d, l, n, r))
	KYTY_FAKE(copyImageToBuffer, (vk::Image s, vk::ImageLayout l, vk::Buffer d, uint32_t n, const vk::BufferImageCopy* r), (s, l, d, n, r))
	KYTY_FAKE(copyImage, (vk::Image s, vk::ImageLayout sl, vk::Image d, vk::ImageLayout dl, uint32_t n, const vk::ImageCopy* r), (s, sl, d, dl, n, r))
	KYTY_FAKE(fillBuffer, (vk::Buffer b, vk::DeviceSize o, vk::DeviceSize n, uint32_t v), (b, o, n, v))
	KYTY_FAKE(clearColorImage, (vk::Image i, vk::ImageLayout l, const vk::ClearColorValue* c, uint32_t n, const vk::ImageSubresourceRange* r),
	          (i, l, c, n, r))
	KYTY_FAKE(clearDepthStencilImage,
	          (vk::Image i, vk::ImageLayout l, const vk::ClearDepthStencilValue* c, uint32_t n, const vk::ImageSubresourceRange* r), (i, l, c, n, r))
#undef KYTY_FAKE
	// A command no IR site records (an upload copy, a dispatch ...).
	void foreign(uint32_t id) const { ctx->Note(0xF0F0F0F000000000ull | id); }
};

} // namespace

// The segment carries the context pointer.
template <>
struct Libs::Graphics::CpEmit::RawTraits<FakeRaw> {
	static uint64_t Bits(const FakeRaw& raw) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(raw.ctx)); }
	static FakeRaw  From(uint64_t bits) { return FakeRaw {reinterpret_cast<Ctx*>(static_cast<uintptr_t>(bits))}; }
};

namespace {

// The command buffer: its stream, its raw handle, the pending-slot thunk (as CommandBuffer in render.h); in thread mode its emitter.
struct FakeBuffer {
	explicit FakeBuffer(Mode mode, const EmitterConfig* config = nullptr, uint32_t slow_spins = 0): stream(mode) {
		stream.SetOwner(this, &Thunk);
		ctx.recorder   = std::this_thread::get_id();
		ctx.slow_spins = slow_spins;
		if (mode == Mode::Thread && config != nullptr) {
			emitter = std::make_unique<Emitter>(&Stream::ReplaySegment<FakeRaw>, *config);
			stream.AttachEmitter(emitter.get());
			ctx.emitter = emitter.get();
		}
		stream.BindRaw(RawTraits<FakeRaw>::Bits(raw));
	}
	~FakeBuffer() {
		if (emitter) emitter->Stop();
	}
	static void Thunk(void* self, Reason why) {
		auto* b = static_cast<FakeBuffer*>(self);
		b->stream.Drain(b->raw, why);
	}
	[[nodiscard]] FakeRaw          Handle() { return TakeRaw(stream, raw); }
	[[nodiscard]] CmdT<FakeRaw>    Emit() { return {&stream, raw}; }
	[[nodiscard]] CmdT<FakeRaw>    EmitIfPending() { return {&stream, raw, stream.Pending()}; } // as CommandBuffer::EmitIfPending
	void                           End() {
        stream.Drain(raw, Reason::Submit);
        if (emitter) {
            Check(emitter->IdleNow(), "the submit drain left segments to the emitter");
            std::lock_guard lock(ctx.mutex);
            Check(ctx.log->size() == expected_at_end, "the command buffer holds every command at the submit");
        }
	}
	size_t expected_at_end = 0; // thread mode: log size the submit must see (set by the runner from the reference)
	void                           EndRenderingOp() {
        if (PassToIr(stream)) Emit().endRendering();
        else Handle().endRendering();
	}
	void BeginRenderingOp(const vk::RenderingInfo& info) {
		if (PassToIr(stream)) Emit().beginRendering(info);
		else Handle().beginRendering(info);
	}

	Ctx                      ctx;
	Stream                   stream;
	FakeRaw                  raw {&ctx};
	std::unique_ptr<Emitter> emitter; // declared last: stops before the stream's arena goes away
};

template <class H>
H Fake(uint64_t v) {
	return H(reinterpret_cast<typename H::CType>(static_cast<uintptr_t>(v)));
}

constexpr uint32_t kDraws = 5000; // 5000 x (draw + push constants) > 64 KB: several chunks

void Script(FakeBuffer& b) {
	const auto layout   = Fake<vk::PipelineLayout>(0x1000);
	const auto pipeline = Fake<vk::Pipeline>(0x2000);
	const auto image    = Fake<vk::Image>(0x3000);
	const auto view     = Fake<vk::ImageView>(0x3100);
	const auto buffer   = Fake<vk::Buffer>(0x4000);

	// 1. Transit of a render target (IR), then the pass it ends (IR: the stream is not empty).
	{
		vk::ImageMemoryBarrier2 barrier {};
		barrier.image     = image;
		barrier.oldLayout = vk::ImageLayout::eUndefined;
		barrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
		vk::DependencyInfo dep {};
		dep.imageMemoryBarrierCount = 1;
		dep.pImageMemoryBarriers    = &barrier;
		b.Emit().pipelineBarrier2(dep);
		barrier.newLayout = vk::ImageLayout::eGeneral; // the caller's memory changes before the drain
		b.EndRenderingOp();
	}
	// 2. A direct site (upload copy): drains first.
	b.Handle().foreign(1);
	// 3. A raw handle taken while the stream is empty, then a pass op, then the raw handle again (Image::Transit with a raw buffer).
	{
		const FakeRaw raw = b.Handle();
		vk::RenderingAttachmentInfo color {};
		color.imageView   = view;
		color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		vk::RenderingInfo info {};
		info.renderArea.extent    = vk::Extent2D {640, 360};
		info.layerCount           = 1;
		info.colorAttachmentCount = 1;
		info.pColorAttachments    = &color;
		b.BeginRenderingOp(info);
		raw.foreign(2);
		b.EndRenderingOp();
		raw.foreign(3);
	}
	// 4. CommitBindings-like: push descriptors from reused arrays, bind sets, push constants, a GDS-style v1 barrier.
	{
		std::vector<vk::DescriptorImageInfo>  images(3);
		std::vector<vk::DescriptorBufferInfo> buffers(2);
		for (uint32_t i = 0; i < 3; i++) images[i] = {Fake<vk::Sampler>(0x5000 + i), Fake<vk::ImageView>(0x5100 + i), vk::ImageLayout::eGeneral};
		for (uint32_t i = 0; i < 2; i++) buffers[i] = {buffer, 256u * i, 256u};
		std::array<vk::WriteDescriptorSet, 2> writes {};
		writes[0].dstBinding      = 0;
		writes[0].descriptorCount = 3;
		writes[0].descriptorType  = vk::DescriptorType::eCombinedImageSampler;
		writes[0].pImageInfo      = images.data();
		writes[1].dstBinding      = 1;
		writes[1].descriptorCount = 2;
		writes[1].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[1].pBufferInfo     = buffers.data();
		const auto cmd            = b.Emit();
		vk::BufferMemoryBarrier gds {};
		gds.buffer = buffer;
		gds.size   = 64;
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eVertexShader, {}, 0, nullptr, 1, &gds, 0, nullptr);
		gds.size = 128;
		cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, layout, 0, 2, writes.data());
		for (auto& info: images) info.imageView = Fake<vk::ImageView>(0xDEAD); // m_descriptor_images is reused by the next draw
		for (auto& info: buffers) info.offset = 0xBAD;
		writes[0].descriptorCount = 1;
		std::array<uint32_t, 32> push {};
		for (uint32_t i = 0; i < 32; i++) push[i] = i * 7u;
		cmd.pushConstants(layout, vk::ShaderStageFlagBits::eAll, 0, sizeof(push), push.data());
		push[5] = 0xFFFF;
		const std::array<vk::DescriptorSet, 1> sets {Fake<vk::DescriptorSet>(0x6000)};
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, layout, 0, 1, sets.data(), 0, nullptr);
	}
	// 5. The seam: vertex / index buffers, dynamic state, pass, pipeline, draws; a helper that takes the raw handle in the middle (conversion).
	{
		const auto cmd = b.Emit();
		std::array<vk::Buffer, 2>     vbs {buffer, Fake<vk::Buffer>(0x4100)};
		std::array<vk::DeviceSize, 2> offsets {16, 32};
		std::array<vk::DeviceSize, 2> sizes {1024, 2048};
		cmd.bindVertexBuffers2(0, 2, vbs.data(), offsets.data(), sizes.data(), nullptr);
		offsets[0] = 999;
		cmd.bindIndexBuffer(buffer, 4096, vk::IndexType::eUint16);
		std::array<vk::Viewport, 2> viewports {vk::Viewport {0, 0, 640, 360, 0, 1}, vk::Viewport {1, 2, 3, 4, 0, 1}};
		cmd.setViewportWithCount(2, viewports.data());
		viewports[0].width = 1;
		const vk::Rect2D scissor {{0, 0}, {640, 360}};
		cmd.setScissorWithCount(1, &scissor);
		cmd.setDepthTestEnable(true);
		cmd.setDepthWriteEnable(false);
		cmd.setDepthCompareOp(vk::CompareOp::eGreaterOrEqual);
		cmd.setDepthBoundsTestEnable(false);
		cmd.setDepthBounds(0.25f, 0.75f);
		cmd.setDepthBiasEnable(true);
		cmd.setDepthBias(1.0f, 0.5f, 2.0f);
		cmd.setStencilTestEnable(true);
		cmd.setStencilOp(vk::StencilFaceFlagBits::eFront, vk::StencilOp::eKeep, vk::StencilOp::eReplace, vk::StencilOp::eZero, vk::CompareOp::eAlways);
		cmd.setStencilReference(vk::StencilFaceFlagBits::eFrontAndBack, 0x80);
		cmd.setStencilWriteMask(vk::StencilFaceFlagBits::eBack, 0xF0);
		cmd.setStencilCompareMask(vk::StencilFaceFlagBits::eFront, 0x0F);
		cmd.setCullMode(vk::CullModeFlagBits::eBack);
		cmd.setFrontFace(vk::FrontFace::eClockwise);
		std::array<float, 4> blend {0.1f, 0.2f, 0.3f, 0.4f};
		cmd.setBlendConstants(blend.data());
		blend[0] = 9.0f;
		cmd.setLineWidth(1.0f);
		cmd.setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlagBits::eColor);
		vk::RenderingAttachmentInfo color {};
		color.imageView   = view;
		color.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
		color.loadOp      = vk::AttachmentLoadOp::eClear;
		vk::RenderingAttachmentInfo depth {};
		depth.imageView = Fake<vk::ImageView>(0x3200);
		vk::RenderingInfo info {};
		info.renderArea.extent    = vk::Extent2D {640, 360};
		info.layerCount           = 1;
		info.colorAttachmentCount = 1;
		info.pColorAttachments    = &color;
		info.pDepthAttachment     = &depth;
		b.BeginRenderingOp(info); // the stream holds the seam's commands: IR
		color.loadOp = vk::AttachmentLoadOp::eLoad;
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
		cmd.drawIndexed(36, 1, 0, -4, 0);
		const FakeRaw converted = cmd; // occlusion query / write-hazard barrier helper
		converted.foreign(4);
		cmd.draw(3, 1, 0, 0);
		cmd.drawIndirect(buffer, 64, 2, 16);
		cmd.drawIndexedIndirect(buffer, 128, 1, 20);
		cmd.drawMeshTasksEXT(8, 1, 1);
		cmd.drawMeshTasksIndirectEXT(buffer, 256, 1, 12);
	}
	// 6. A GPU wait (MasterSemaphore::Wait drains the thread's pending stream), then a command on the raw handle the buffer had.
	DrainThreadPending(Reason::Wait);
	b.raw.foreign(5);
	// 7. A command the IR cannot hold (pNext chain): drained before, recorded directly.
	{
		b.Emit().draw(1, 1, 0, 0);
		vk::MemoryBarrier2 chained {};
		vk::MemoryBarrier2 next {};
		chained.pNext = &next;
		vk::DependencyInfo dep {};
		dep.memoryBarrierCount = 1;
		dep.pMemoryBarriers    = &chained;
		b.Emit().pipelineBarrier2(dep);
		b.Emit().draw(2, 1, 0, 0);
	}
	// 9. S565: sites outside the seam through EmitIfPending. With commands in the stream they follow the IR, with an empty one they go
	//    straight on (and the stream stays empty).
	{
		b.Emit().draw(7, 1, 0, 0); // the stream holds commands
		std::array<vk::BufferCopy, 2> regions {vk::BufferCopy {0, 64, 128}, vk::BufferCopy {256, 512, 64}};
		b.EmitIfPending().copyBuffer(buffer, Fake<vk::Buffer>(0x4200), 2, regions.data());
		regions[0].size = 1; // the caller's memory changes before the drain
		b.EmitIfPending().fillBuffer(buffer, 0, 256, 0xABCDu);
		vk::BufferImageCopy bic {};
		bic.bufferOffset                = 32;
		bic.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
		bic.imageSubresource.layerCount = 1;
		bic.imageExtent                 = vk::Extent3D {16, 16, 1};
		b.EmitIfPending().copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &bic);
		bic.bufferOffset = 4096;
		b.EmitIfPending().copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, buffer, 1, &bic);
		bic.bufferOffset = 7;
		std::array<vk::ImageCopy, 2> copies {};
		copies[0].srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
		copies[0].dstSubresource = {vk::ImageAspectFlagBits::eColor, 1, 0, 1};
		copies[0].extent         = vk::Extent3D {8, 8, 1};
		copies[1]                = copies[0];
		copies[1].dstOffset      = vk::Offset3D {8, 0, 0};
		b.EmitIfPending().copyImage(image, vk::ImageLayout::eTransferSrcOptimal, Fake<vk::Image>(0x3300), vk::ImageLayout::eTransferDstOptimal, 2,
		                            copies.data());
		copies[1].extent.width = 1;
		vk::ClearColorValue color {};
		color.uint32[0] = 5;
		vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
		b.EmitIfPending().clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
		color.uint32[0] = 9;
		range.levelCount = 2;
		const vk::ClearDepthStencilValue depth_stencil {0.5f, 3};
		vk::ImageSubresourceRange depth_range {vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1};
		b.EmitIfPending().clearDepthStencilImage(Fake<vk::Image>(0x3400), vk::ImageLayout::eTransferDstOptimal, &depth_stencil, 1, &depth_range);
		const auto cmd = b.EmitIfPending();
		cmd.bindPipeline(vk::PipelineBindPoint::eCompute, Fake<vk::Pipeline>(0x2100));
		cmd.dispatch(4, 2, 1);
		cmd.dispatchBase(1, 0, 0, 2, 1, 1);
		cmd.dispatchIndirect(buffer, 64);
		Check(b.stream.Current() == Mode::Off || cmd.ToIr(), "with commands in the stream EmitIfPending follows the IR");
		b.Handle().foreign(6); // a direct site: drains
		Check(!b.EmitIfPending().ToIr(), "with an empty stream EmitIfPending records directly");
		b.EmitIfPending().fillBuffer(buffer, 64, 64, 1);
		b.EmitIfPending().dispatch(1, 1, 1);
		Check(!b.stream.Pending(), "EmitIfPending leaves an empty stream empty");
	}
	// 8. Many draws: the IR crosses several chunks, then the submit drains it.
	{
		const auto cmd = b.Emit();
		for (uint32_t i = 0; i < kDraws; i++) {
			uint32_t object_id = i;
			cmd.pushConstants(layout, vk::ShaderStageFlagBits::eVertex, 0, 4, &object_id);
			cmd.draw(3 + i, 1, i, 0);
		}
		b.EndRenderingOp();
	}
	b.End();
}

std::vector<uint64_t> Run(Mode mode, Stats* stats, const EmitterConfig* config = nullptr, uint32_t slow_spins = 0, size_t expected = 0,
                          EmitterStats* emitter_stats = nullptr) {
	std::vector<uint64_t> log;
	FakeBuffer            b(mode, config, slow_spins);
	b.ctx.log         = &log;
	b.expected_at_end = expected;
	Script(b);
	if (stats != nullptr) *stats = b.stream.TakeStats();
	Check(!b.stream.Pending(), "the stream is empty after End");
	Check(ThreadPending().owner == nullptr, "the thread's pending slot is clear after End");
	if (b.emitter) {
		b.emitter->Stop();
		if (emitter_stats != nullptr) *emitter_stats = b.emitter->ReadStats();
	}
	return log;
}

} // namespace

int main() {
	Check(ParseMode(nullptr) == Mode::Off && ParseMode("0") == Mode::Off && ParseMode("sync") == Mode::Sync && ParseMode("shadow") == Mode::Shadow &&
	          ParseMode("thread") == Mode::Thread && ParseMode("1") == Mode::Off && ParseMode("SYNC") == Mode::Off,
	      "ParseMode");

	Stats off_stats {};
	const auto reference = Run(Mode::Off, &off_stats);
	Check(off_stats.ops == 0 && off_stats.drains == 0, "mode 0 writes no IR");
	Check(reference.size() > 2 * kDraws, "the reference has every command");

	// sync: the IR replay logs exactly the direct sequence.
	Stats sync_stats {};
	const auto sync = Run(Mode::Sync, &sync_stats);
	Check(sync.size() == reference.size(), "sync: as many commands as the direct path");
	for (size_t i = 0; i < reference.size(); i++) {
		if (sync[i] != reference[i]) {
			std::fprintf(stderr, "CpEmitTests: sync differs at command %zu of %zu\n", i, reference.size());
			Check(false, "sync replays the direct sequence");
		}
	}
	Check(sync_stats.by[static_cast<size_t>(Reason::Handle)] >= 1, "a direct site drained");
	Check(sync_stats.by[static_cast<size_t>(Reason::Conv)] == 1, "the conversion drained once");
	Check(sync_stats.by[static_cast<size_t>(Reason::Wait)] == 1, "the wait drained once");
	Check(sync_stats.by[static_cast<size_t>(Reason::Submit)] == 1, "the submit drained once");
	Check(sync_stats.by[static_cast<size_t>(Reason::Other)] == 1 && sync_stats.fallback == 1, "the pNext barrier fell back once");
	Check(sync_stats.ops > 2 * kDraws && sync_stats.ops < reference.size(), "IR commands counted (the direct-only ones are not)");
	Check(sync_stats.bytes > 2 * Arena::kChunk, "the IR crossed chunks");
	Check(sync_stats.ne == 0, "sync counts no mismatch");

	// The arena is reused: a second and third pass over the script allocate no new chunk.
	{
		std::vector<uint64_t> log;
		FakeBuffer            b(Mode::Sync);
		b.ctx.log = &log;
		Script(b);
		const size_t chunks = b.stream.ArenaChunks();
		Check(chunks >= 3, "the script's IR needs several chunks");
		Script(b);
		Check(b.stream.ArenaChunks() == chunks, "the second pass allocates no chunk");
		// After End the arena is rewound, its chunks kept.
		b.Emit().draw(1, 1, 1, 1);
		b.End();
		Stats s = b.stream.TakeStats();
		Check(s.drains > 0, "reuse run drained");
		Check(log.size() == 2 * reference.size() + 1, "two scripts and one draw");
	}

	// shadow: direct calls give the reference, the IR digest agrees at every drain.
	Stats shadow_stats {};
	const auto shadow = Run(Mode::Shadow, &shadow_stats);
	Check(shadow == reference, "shadow records the direct sequence");
	Check(shadow_stats.drains == sync_stats.drains, "shadow drains where sync does");
	Check(shadow_stats.ne == 0, "shadow: IR digest equals the direct one");

	// The shadow comparison does detect a difference (a direct command the IR missed).
	{
		std::vector<uint64_t> log;
		FakeBuffer            b(Mode::Shadow);
		b.ctx.log = &log;
		b.Emit().draw(1, 2, 3, 4);
		b.stream.Direct().draw(5, 6, 7, 8);
		b.End();
		Check(b.stream.TakeStats().ne == 1, "shadow counts a missed command");
	}

	// Another stream of the thread becoming pending drains the first one (one pending slot per thread).
	{
		std::vector<uint64_t> log;
		FakeBuffer            a(Mode::Sync), c(Mode::Sync);
		a.ctx.log = &log;
		c.ctx.log = &log;
		a.Emit().draw(1, 0, 0, 0);
		c.Emit().draw(2, 0, 0, 0);
		Check(!a.stream.Pending() && c.stream.Pending(), "the first stream drained when the second became pending");
		DrainThreadPending(Reason::Wait);
		Check(!c.stream.Pending() && log.size() == 2, "the wait drained the second");
		a.End();
		c.End();
	}

	// S565 thread mode: the emitter replays the segments; the log is the reference (order across the threads), the ownership probe in
	// Ctx::Note holds, every End finds the emitter done, and the drains are those of sync. The slow command buffer runs first (the mutants
	// fail there for sure).
	{
		struct Case {
			uint32_t publish, batch, spin, slow, repeat;
			bool     steal;
		};
		const Case cases[] = {
		    {256, 2, 0, 400, 2, true},   // slow command buffer: the emitter lags, drains wait or replay the rest
		    {64, 1, 0, 0, 20, true},     // a segment per draw, the emitter woken for each
		    {2048, 4, 0, 0, 20, true},   // the production defaults
		    {64, 8, 0, 0, 10, false},    // no stealing: every segment by the emitter, the drain wakes it
		    {512, 4, 2000, 0, 10, true}, // the emitter spins before it parks
		};
		uint64_t by_emitter = 0, stolen = 0;
		for (const auto& c: cases) {
			EmitterConfig config;
			config.publish_bytes = c.publish;
			config.batch         = c.batch;
			config.spin          = c.spin;
			config.steal         = c.steal;
			for (uint32_t r = 0; r < c.repeat; r++) {
				Stats        st {};
				EmitterStats es {};
				const auto   log = Run(Mode::Thread, &st, &config, c.slow, reference.size(), &es);
				Check(log.size() == reference.size(), "thread: as many commands as the direct path");
				for (size_t i = 0; i < reference.size(); i++) {
					if (log[i] != reference[i]) {
						std::fprintf(stderr, "CpEmitTests: thread (publish %u batch %u steal %d) differs at command %zu of %zu\n", c.publish, c.batch,
						             c.steal ? 1 : 0, i, reference.size());
						Check(false, "thread replays the direct sequence");
					}
				}
				for (size_t k = 0; k < static_cast<size_t>(Reason::Full); k++) Check(st.by[k] == sync_stats.by[k], "thread drains where sync does");
				Check(st.drains == sync_stats.drains && st.ops == sync_stats.ops, "thread: the same IR as sync");
				Check(st.publishes >= 1 && es.segments + st.steal_segs == st.publishes + 1, "every segment (and the stop marker) replayed once");
				if (!c.steal) Check(st.steal_segs == 0 && es.segments >= st.publishes, "no stealing: the emitter replayed every segment");
				by_emitter += es.segments;
				stolen += st.steal_segs;
			}
		}
		Check(by_emitter > 0, "the emitter thread replayed segments");
		std::printf("CpEmitTests: thread OK (segments by the emitter %llu, replayed by the recorder at drains %llu)\n",
		            static_cast<unsigned long long>(by_emitter), static_cast<unsigned long long>(stolen));
	}

	std::printf("CpEmitTests: OK (%zu commands, %llu IR ops, %llu bytes, %llu drains)\n", reference.size(),
	            static_cast<unsigned long long>(sync_stats.ops), static_cast<unsigned long long>(sync_stats.bytes),
	            static_cast<unsigned long long>(sync_stats.drains));
	return 0;
}
