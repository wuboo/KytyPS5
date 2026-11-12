// S561 KYTY_CP_EMIT (cpEmit.h): the IR of the recording sites replays in the order of the direct calls.
// One script of a draw-like sequence (IR sites through CmdT, direct sites through TakeRaw / a raw handle, Begin/EndRendering through PassToIr,
// a conversion of the IR handle, a GPU wait through the thread's pending slot, enough draws to fill several arena chunks) runs three times:
// mode 0 (everything direct: the reference), sync (IR replayed at the drains) and shadow (direct + IR digest). The fake command buffer logs a
// digest per command, so sync must log the reference sequence and shadow must log it with emit_ne = 0. Caller memory (descriptor infos, push
// constants, attachments) is overwritten after each IR command and before its drain: the IR must hold values.
// Mutants (must fail, WILL_FAIL): NO_DRAIN (a direct site does not drain first), SHALLOW (push-descriptor infos keep the caller's pointers),
// UNCOND_PASS (Begin/EndRendering go to the IR even when it is empty, so a raw handle taken before them records out of order).

#include "graphics/host_gpu/renderer/cpEmit.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using namespace Libs::Graphics::CpEmit;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "CpEmitTests: failed: %s\n", what);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

// Logs one digest per command (the same HashSink the shadow uses, one fresh per command).
struct FakeRaw {
	std::vector<uint64_t>* log = nullptr;

#define KYTY_FAKE(name, params, args)                                                                                                            \
	void name params const {                                                                                                                       \
		HashSink h;                                                                                                                                \
		h.name args;                                                                                                                               \
		log->push_back(h.Value());                                                                                                                 \
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
#undef KYTY_FAKE
	// A command no IR site records (an upload copy, a dispatch ...).
	void foreign(uint32_t id) const { log->push_back(0xF0F0F0F000000000ull | id); }
};

// The command buffer: its stream, its raw handle, the pending-slot thunk (as CommandBuffer in render.h).
struct FakeBuffer {
	explicit FakeBuffer(Mode mode): stream(mode) { stream.SetOwner(this, &Thunk); }
	static void Thunk(void* self, Reason why) {
		auto* b = static_cast<FakeBuffer*>(self);
		b->stream.Drain(b->raw, why);
	}
	[[nodiscard]] FakeRaw          Handle() { return TakeRaw(stream, raw); }
	[[nodiscard]] CmdT<FakeRaw>    Emit() { return {&stream, raw}; }
	void                           End() { stream.Drain(raw, Reason::Submit); }
	void                           EndRenderingOp() {
        if (PassToIr(stream)) Emit().endRendering();
        else Handle().endRendering();
	}
	void BeginRenderingOp(const vk::RenderingInfo& info) {
		if (PassToIr(stream)) Emit().beginRendering(info);
		else Handle().beginRendering(info);
	}

	Stream  stream;
	FakeRaw raw;
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

std::vector<uint64_t> Run(Mode mode, Stats* stats) {
	std::vector<uint64_t> log;
	FakeBuffer            b(mode);
	b.raw.log = &log;
	Script(b);
	if (stats != nullptr) *stats = b.stream.TakeStats();
	Check(!b.stream.Pending(), "the stream is empty after End");
	Check(ThreadPending().owner == nullptr, "the thread's pending slot is clear after End");
	return log;
}

} // namespace

int main() {
	Check(ParseMode(nullptr) == Mode::Off && ParseMode("0") == Mode::Off && ParseMode("sync") == Mode::Sync && ParseMode("shadow") == Mode::Shadow &&
	          ParseMode("1") == Mode::Off && ParseMode("SYNC") == Mode::Off,
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
		b.raw.log = &log;
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
		b.raw.log = &log;
		b.Emit().draw(1, 2, 3, 4);
		b.stream.Direct().draw(5, 6, 7, 8);
		b.End();
		Check(b.stream.TakeStats().ne == 1, "shadow counts a missed command");
	}

	// Another stream of the thread becoming pending drains the first one (one pending slot per thread).
	{
		std::vector<uint64_t> log;
		FakeBuffer            a(Mode::Sync), c(Mode::Sync);
		a.raw.log = &log;
		c.raw.log = &log;
		a.Emit().draw(1, 0, 0, 0);
		c.Emit().draw(2, 0, 0, 0);
		Check(!a.stream.Pending() && c.stream.Pending(), "the first stream drained when the second became pending");
		DrainThreadPending(Reason::Wait);
		Check(!c.stream.Pending() && log.size() == 2, "the wait drained the second");
		a.End();
		c.End();
	}

	std::printf("CpEmitTests: OK (%zu commands, %llu IR ops, %llu bytes, %llu drains)\n", reference.size(),
	            static_cast<unsigned long long>(sync_stats.ops), static_cast<unsigned long long>(sync_stats.bytes),
	            static_cast<unsigned long long>(sync_stats.drains));
	return 0;
}
