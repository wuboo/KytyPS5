#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPEMIT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPEMIT_H_

// S561 (KYTY_CP_EMIT=0|sync|shadow, default 0): the first step of K7 W2 (deferred Vulkan recording). The hot recording sites of the draw path
// (RESULT-S560 sites 1-6: the colour / depth Transit of AcquireRenderTargets, CommitBindings, the ExecutePreparedDraw seam, Begin/EndRendering)
// write their commands as values into an IR stream of the command buffer instead of calling Vulkan; the same thread replays the stream into
// the VkCommandBuffer at a drain point. No thread yet: this step measures what writing the IR costs the recorder.
//
// - 0: every command goes straight to Vulkan, as before (one predictable branch per command more).
// - sync: the commands are written to the IR and replayed at the next drain.
// - shadow: the commands go straight to Vulkan AND are written to the IR; at each drain the IR is replayed into a digest that is compared with
//   the digest of the direct calls since the previous drain (emit_ne_n counts the drains where they differ).
//
// Drain points (the stream is replayed, in order, before anything else reaches the VkCommandBuffer):
// - any CommandBuffer::Handle() (a site that records directly: TakeRaw, Reason::Handle);
// - a Cmd converted to vk::CommandBuffer (a helper of the seam that still takes the raw handle: Reason::Conv);
// - CommandBuffer::End() before the submit (Reason::Submit);
// - MasterSemaphore::Wait on a tick that is not known free (Reason::Wait, through the thread's pending slot);
// - a command the IR cannot hold (a pNext chain), another stream of the thread becoming pending (Reason::Other).
// Begin/EndRendering are called from many sites that hold a raw handle (Image::Transit with a raw buffer ends the pass, then records its barrier
// on the raw handle). They go to the IR only while the stream already holds commands (PassToIr); with an empty stream recording directly is the
// same as appending and draining at once, so a raw handle taken before them stays in order.
//
// The IR is plain values: Vulkan structures are copied (arrays deep, pNext must be null), never a pointer into live state (lesson of S478).
// Storage is an arena of 64 KB chunks per command buffer, reused after each drain: no allocation in steady state.
//
// S565 (KYTY_CP_EMIT=thread, K7 W2 step 2): the stream of the renderer's command buffer is replayed by an emitter thread (Emitter below,
// pattern of yuzu's Scheduler::Record + worker). The recorder hands contiguous ranges of the arena (segments) over an SPSC ring; the emitter
// replays them into the VkCommandBuffer. A drain is the same as in sync, except that it publishes what is left and waits until every segment
// is replayed (the recorder replays the rest itself when the emitter is not working: Emitter::WaitIdle). Ownership of the VkCommandBuffer and
// of its VkCommandPool: the recorder from a drain (done == last pushed, acquire) until its next push; the holder of Emitter::m_owner while it
// replays segments otherwise. The arena is rewound only by a drain, so a segment's bytes stay put until the emitter is done with them.
// A command buffer without an emitter (the presenter's) behaves as in sync.
// Sites outside the S561 IR sites record through CommandBuffer::EmitIfPending(): to the IR while the stream holds commands, else directly.
// Like Begin/EndRendering they never drain and never put a command into an empty stream, so a raw handle taken before them stays in order.
//
// Header-only (tests/CpEmitTests.cpp). Mutants (must fail the test, WILL_FAIL): NO_DRAIN (TakeRaw does not drain), SHALLOW (push-descriptor
// infos keep the caller's pointers), UNCOND_PASS (Begin/EndRendering always go to the IR), NO_SUBMIT_DRAIN (thread: the drain before the
// submit does not wait for the emitter), TOUCH_DURING_EMIT (thread: a direct site records while the emitter still replays).

#include "graphics/host_gpu/vulkanCommon.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#define KYTY_CPEMIT_TSC 1
#endif

namespace Libs::Graphics::CpEmit {

enum class Mode : uint8_t { Off = 0, Sync = 1, Shadow = 2, Thread = 3 };

inline Mode ParseMode(const char* value) {
	if (value == nullptr) return Mode::Off;
	if (std::strcmp(value, "sync") == 0) return Mode::Sync;
	if (std::strcmp(value, "shadow") == 0) return Mode::Shadow;
	if (std::strcmp(value, "thread") == 0) return Mode::Thread;
	return Mode::Off;
}

inline const char* ModeName(Mode mode) {
	return mode == Mode::Sync ? "sync" : mode == Mode::Shadow ? "shadow" : mode == Mode::Thread ? "thread" : "0";
}

inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_CP_EMIT"));
	return mode;
}

// Full: not a drain; the thread mode's recorder waited for a slot of the emitter's ring (counted in the wait times only).
enum class Reason : uint8_t { Handle, Conv, Submit, Wait, Other, Full, Count };

enum class Op : uint16_t {
	Barrier2,
	Barrier1,
	PushConstants,
	PushDescriptorSet,
	BindDescriptorSets,
	BindPipeline,
	BindVertexBuffers2,
	BindIndexBuffer,
	SetViewportWithCount,
	SetScissorWithCount,
	SetDepthTestEnable,
	SetDepthWriteEnable,
	SetDepthCompareOp,
	SetDepthBoundsTestEnable,
	SetDepthBounds,
	SetDepthBiasEnable,
	SetDepthBias,
	SetStencilTestEnable,
	SetStencilOp,
	SetStencilReference,
	SetStencilWriteMask,
	SetStencilCompareMask,
	SetCullMode,
	SetFrontFace,
	SetBlendConstants,
	SetLineWidth,
	SetFeedbackLoop,
	Draw,
	DrawIndexed,
	DrawIndirect,
	DrawIndexedIndirect,
	DrawMeshTasks,
	DrawMeshTasksIndirect,
	BeginRendering,
	EndRendering,
	// S565: the sites outside the draw seam (EmitIfPending).
	Dispatch,
	DispatchBase,
	DispatchIndirect,
	CopyBuffer,
	CopyBufferToImage,
	CopyImageToBuffer,
	CopyImage,
	FillBuffer,
	ClearColorImage,
	ClearDepthStencilImage,
	Count,
};

struct Stats {
	uint64_t ops       = 0; // commands written to the IR
	uint64_t bytes     = 0; // IR bytes (headers included)
	uint64_t drains    = 0; // drains of a non-empty stream
	uint64_t by[static_cast<size_t>(Reason::Count)] {};
	uint64_t rec_ticks = 0; // sampled record time (TSC ticks, already scaled by the sampling period, rdtsc overhead removed)
	uint64_t replay_ticks = 0; // time in the drains (replay into Vulkan, or into the digest in shadow), TSC ticks
	uint64_t ne        = 0; // shadow: drains whose IR digest differs from the direct one
	uint64_t fallback  = 0; // commands the IR could not hold (recorded directly after a drain)
	// S565 thread mode (recorder side): time the recorder waited for the emitter (a drain, by reason; Full: a ring slot), segments pushed,
	// segments the recorder replayed itself because the emitter was not working at a drain, drains that did so.
	uint64_t wait_ticks = 0;
	uint64_t wait_by[static_cast<size_t>(Reason::Count)] {};
	uint64_t publishes   = 0;
	uint64_t steal_segs  = 0;
	uint64_t steals      = 0;
	uint64_t wakes       = 0; // the recorder woke the parked emitter
};

// ---------------------------------------------------------------------------------------------------------------------------------------
// Clocks. Record time is sampled: every kSamplePeriod-th command is timed with rdtsc and counted kSamplePeriod times.

inline constexpr uint32_t kSamplePeriod = 16;

inline uint64_t Ticks() {
#if defined(KYTY_CPEMIT_TSC)
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

inline uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Minimum of back-to-back reads: removed from each sample.
inline uint64_t TickOverhead() {
	static const uint64_t overhead = [] {
		uint64_t best = UINT64_MAX;
		for (int i = 0; i < 64; i++) {
			const uint64_t a = Ticks();
			const uint64_t b = Ticks();
			if (b - a < best) best = b - a;
		}
		return best == UINT64_MAX ? 0 : best;
	}();
	return overhead;
}

// ns per tick, calibrated lazily against steady_clock without sleeping: 0 until 20 ms have passed since the first call.
inline double NsPerTick() {
#if defined(KYTY_CPEMIT_TSC)
	static std::atomic<uint64_t> t0 {0}, n0 {0};
	static std::atomic<double>   ratio {0.0};
	if (const double r = ratio.load(std::memory_order_relaxed); r != 0.0) return r;
	const uint64_t t = Ticks();
	const uint64_t n = NowNs();
	uint64_t       expected = 0;
	if (n0.compare_exchange_strong(expected, n)) {
		t0.store(t);
		return 0.0;
	}
	const uint64_t base_t = t0.load();
	if (n - expected < 20'000'000 || base_t == 0 || t <= base_t) return 0.0;
	const double r = static_cast<double>(n - expected) / static_cast<double>(t - base_t);
	ratio.store(r, std::memory_order_relaxed);
	return r;
#else
	return 1.0;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Digest of a command sequence (shadow). Every structure is hashed by its fields (never its padding or its pointers, only whether a pointer
// is set), arrays of padding-free values by their bytes, so the direct call and the IR replay (whose structures point into the arena) hash
// the same when the IR copied every value.

class HashSink {
public:
	[[nodiscard]] uint64_t Value() const noexcept { return m_h; }
	[[nodiscard]] uint64_t Count() const noexcept { return m_n; }
	void                   Reset() noexcept { *this = HashSink {}; }

	void pipelineBarrier2(const vk::DependencyInfo& d) {
		Begin(Op::Barrier2);
		Word(static_cast<uint32_t>(d.dependencyFlags));
		Array(d.pMemoryBarriers, d.memoryBarrierCount);
		Array(d.pBufferMemoryBarriers, d.bufferMemoryBarrierCount);
		Array(d.pImageMemoryBarriers, d.imageMemoryBarrierCount);
	}
	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst, vk::DependencyFlags flags, uint32_t nm,
	                     const vk::MemoryBarrier* m, uint32_t nb, const vk::BufferMemoryBarrier* b, uint32_t ni, const vk::ImageMemoryBarrier* i) {
		Begin(Op::Barrier1);
		Word(static_cast<uint32_t>(src));
		Word(static_cast<uint32_t>(dst));
		Word(static_cast<uint32_t>(flags));
		Array(m, nm);
		Array(b, nb);
		Array(i, ni);
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data) {
		Begin(Op::PushConstants);
		Handle(layout);
		Word(static_cast<uint32_t>(stages));
		Word(offset);
		Word(size);
		Bytes(data, size);
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bp, vk::PipelineLayout layout, uint32_t set, uint32_t count, const vk::WriteDescriptorSet* w) {
		Begin(Op::PushDescriptorSet);
		Word(static_cast<uint32_t>(bp));
		Handle(layout);
		Word(set);
		Word(count);
		for (uint32_t k = 0; k < count; k++) {
			const auto& x = w[k];
			Handle(x.dstSet);
			Word(x.dstBinding);
			Word(x.dstArrayElement);
			Word(x.descriptorCount);
			Word(static_cast<uint32_t>(x.descriptorType));
			Word(x.pImageInfo != nullptr ? 1u : 0u);
			Word(x.pBufferInfo != nullptr ? 1u : 0u);
			Word(x.pTexelBufferView != nullptr ? 1u : 0u);
			Array(x.pImageInfo, x.pImageInfo != nullptr ? x.descriptorCount : 0u);
			Array(x.pBufferInfo, x.pBufferInfo != nullptr ? x.descriptorCount : 0u);
			Array(x.pTexelBufferView, x.pTexelBufferView != nullptr ? x.descriptorCount : 0u);
		}
	}
	void bindDescriptorSets(vk::PipelineBindPoint bp, vk::PipelineLayout layout, uint32_t first, uint32_t count, const vk::DescriptorSet* sets,
	                        uint32_t ndyn, const uint32_t* dyn) {
		Begin(Op::BindDescriptorSets);
		Word(static_cast<uint32_t>(bp));
		Handle(layout);
		Word(first);
		Array(sets, count);
		Array(dyn, ndyn);
	}
	void bindPipeline(vk::PipelineBindPoint bp, vk::Pipeline pipeline) {
		Begin(Op::BindPipeline);
		Word(static_cast<uint32_t>(bp));
		Handle(pipeline);
	}
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers, const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) {
		Begin(Op::BindVertexBuffers2);
		Word(first);
		Array(buffers, count);
		Array(offsets, count);
		Word(sizes != nullptr ? 1u : 0u);
		Array(sizes, sizes != nullptr ? count : 0u);
		Word(strides != nullptr ? 1u : 0u);
		Array(strides, strides != nullptr ? count : 0u);
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
		Begin(Op::BindIndexBuffer);
		Handle(buffer);
		Quad(offset);
		Word(static_cast<uint32_t>(type));
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* v) {
		Begin(Op::SetViewportWithCount);
		Array(v, count);
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* r) {
		Begin(Op::SetScissorWithCount);
		Array(r, count);
	}
	void setDepthTestEnable(vk::Bool32 v) { Small(Op::SetDepthTestEnable, v); }
	void setDepthWriteEnable(vk::Bool32 v) { Small(Op::SetDepthWriteEnable, v); }
	void setDepthCompareOp(vk::CompareOp v) { Small(Op::SetDepthCompareOp, static_cast<uint32_t>(v)); }
	void setDepthBoundsTestEnable(vk::Bool32 v) { Small(Op::SetDepthBoundsTestEnable, v); }
	void setDepthBounds(float a, float b) { Small(Op::SetDepthBounds, Bits(a), Bits(b)); }
	void setDepthBiasEnable(vk::Bool32 v) { Small(Op::SetDepthBiasEnable, v); }
	void setDepthBias(float a, float b, float c) { Small(Op::SetDepthBias, Bits(a), Bits(b), Bits(c)); }
	void setStencilTestEnable(vk::Bool32 v) { Small(Op::SetStencilTestEnable, v); }
	void setStencilOp(vk::StencilFaceFlags f, vk::StencilOp a, vk::StencilOp b, vk::StencilOp c, vk::CompareOp d) {
		Small(Op::SetStencilOp, static_cast<uint32_t>(f), static_cast<uint32_t>(a), static_cast<uint32_t>(b), static_cast<uint32_t>(c),
		      static_cast<uint32_t>(d));
	}
	void setStencilReference(vk::StencilFaceFlags f, uint32_t v) { Small(Op::SetStencilReference, static_cast<uint32_t>(f), v); }
	void setStencilWriteMask(vk::StencilFaceFlags f, uint32_t v) { Small(Op::SetStencilWriteMask, static_cast<uint32_t>(f), v); }
	void setStencilCompareMask(vk::StencilFaceFlags f, uint32_t v) { Small(Op::SetStencilCompareMask, static_cast<uint32_t>(f), v); }
	void setCullMode(vk::CullModeFlags v) { Small(Op::SetCullMode, static_cast<uint32_t>(v)); }
	void setFrontFace(vk::FrontFace v) { Small(Op::SetFrontFace, static_cast<uint32_t>(v)); }
	void setBlendConstants(const float c[4]) { Small(Op::SetBlendConstants, Bits(c[0]), Bits(c[1]), Bits(c[2]), Bits(c[3])); }
	void setLineWidth(float v) { Small(Op::SetLineWidth, Bits(v)); }
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags v) { Small(Op::SetFeedbackLoop, static_cast<uint32_t>(v)); }
	void draw(uint32_t a, uint32_t b, uint32_t c, uint32_t d) { Small(Op::Draw, a, b, c, d); }
	void drawIndexed(uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e) { Small(Op::DrawIndexed, a, b, c, static_cast<uint32_t>(d), e); }
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) { Indirect(Op::DrawIndirect, buffer, offset, count, stride); }
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) {
		Indirect(Op::DrawIndexedIndirect, buffer, offset, count, stride);
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) { Small(Op::DrawMeshTasks, x, y, z); }
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) {
		Indirect(Op::DrawMeshTasksIndirect, buffer, offset, count, stride);
	}
	void beginRendering(const vk::RenderingInfo& r) {
		Begin(Op::BeginRendering);
		Word(static_cast<uint32_t>(r.flags));
		Bytes(&r.renderArea, sizeof(r.renderArea));
		Word(r.layerCount);
		Word(r.viewMask);
		Array(r.pColorAttachments, r.colorAttachmentCount);
		Word(r.pDepthAttachment != nullptr ? 1u : 0u);
		Array(r.pDepthAttachment, r.pDepthAttachment != nullptr ? 1u : 0u);
		Word(r.pStencilAttachment != nullptr ? 1u : 0u);
		Array(r.pStencilAttachment, r.pStencilAttachment != nullptr ? 1u : 0u);
	}
	void endRendering() { Begin(Op::EndRendering); }
	// S565
	void dispatch(uint32_t x, uint32_t y, uint32_t z) { Small(Op::Dispatch, x, y, z); }
	void dispatchBase(uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z) {
		Small(Op::DispatchBase, bx, by, bz, x, y);
		Mix(z);
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) {
		Begin(Op::DispatchIndirect);
		Handle(buffer);
		Quad(offset);
	}
	void copyBuffer(vk::Buffer src, vk::Buffer dst, uint32_t n, const vk::BufferCopy* regions) {
		Begin(Op::CopyBuffer);
		Handle(src);
		Handle(dst);
		Array(regions, n);
	}
	void copyBufferToImage(vk::Buffer src, vk::Image dst, vk::ImageLayout layout, uint32_t n, const vk::BufferImageCopy* regions) {
		Begin(Op::CopyBufferToImage);
		Handle(src);
		Handle(dst);
		Word(U(layout));
		Array(regions, n);
	}
	void copyImageToBuffer(vk::Image src, vk::ImageLayout layout, vk::Buffer dst, uint32_t n, const vk::BufferImageCopy* regions) {
		Begin(Op::CopyImageToBuffer);
		Handle(src);
		Word(U(layout));
		Handle(dst);
		Array(regions, n);
	}
	void copyImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst, vk::ImageLayout dst_layout, uint32_t n, const vk::ImageCopy* regions) {
		Begin(Op::CopyImage);
		Handle(src);
		Word(U(src_layout));
		Handle(dst);
		Word(U(dst_layout));
		Array(regions, n);
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, uint32_t data) {
		Begin(Op::FillBuffer);
		Handle(buffer);
		Quad(offset);
		Quad(size);
		Word(data);
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color, uint32_t n, const vk::ImageSubresourceRange* ranges) {
		Begin(Op::ClearColorImage);
		Handle(image);
		Word(U(layout));
		Word(color != nullptr ? 1u : 0u);
		if (color != nullptr) Bytes(color, sizeof(*color)); // a union of 16 bytes, no padding
		Array(ranges, n);
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout, const vk::ClearDepthStencilValue* value, uint32_t n,
	                            const vk::ImageSubresourceRange* ranges) {
		Begin(Op::ClearDepthStencilImage);
		Handle(image);
		Word(U(layout));
		Word(value != nullptr ? 1u : 0u);
		if (value != nullptr) Bytes(value, sizeof(*value)); // float + uint32, no padding
		Array(ranges, n);
	}

private:
	static uint32_t Bits(float f) {
		uint32_t u = 0;
		std::memcpy(&u, &f, 4);
		return u;
	}
	void Mix(uint64_t v) {
		m_h ^= v + 0x9E3779B97F4A7C15ull + (m_h << 6u) + (m_h >> 2u);
		m_h *= 0xFF51AFD7ED558CCDull;
	}
	void Begin(Op op) {
		m_n++;
		Mix(0xC0DE0000ull | static_cast<uint64_t>(op));
	}
	void Word(uint32_t v) { Mix(v); }
	void Quad(uint64_t v) { Mix(v); }
	template <class H>
	void Handle(H h) {
		Mix(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<typename H::CType>(h))));
	}
	void Bytes(const void* p, size_t n) {
		Mix(n);
		const auto* b = static_cast<const uint8_t*>(p);
		size_t      i = 0;
		for (; i + 8 <= n; i += 8) {
			uint64_t v = 0;
			std::memcpy(&v, b + i, 8);
			Mix(v);
		}
		uint64_t tail = 0;
		if (i < n) {
			std::memcpy(&tail, b + i, n - i);
			Mix(tail);
		}
	}
	template <class T>
	void Array(const T* p, uint32_t n) {
		Word(n);
		if (n == 0 || p == nullptr) return;
		if constexpr (std::is_same_v<T, vk::Viewport> || std::is_same_v<T, vk::Rect2D> || std::is_same_v<T, vk::Buffer> ||
		              std::is_same_v<T, vk::DeviceSize> || std::is_same_v<T, vk::DescriptorSet> || std::is_same_v<T, vk::BufferView> ||
		              std::is_same_v<T, uint32_t> || std::is_same_v<T, vk::BufferCopy> || std::is_same_v<T, vk::BufferImageCopy> ||
		              std::is_same_v<T, vk::ImageCopy> || std::is_same_v<T, vk::ImageSubresourceRange>) {
			Bytes(p, sizeof(T) * n); // no padding
		} else {
			for (uint32_t i = 0; i < n; i++) Leaf(p[i]);
		}
	}
	// Structures with padding (the 4 bytes after sType, tail padding) are hashed field by field: their padding is not a value.
	template <class F>
	static uint64_t M(F f) {
		return static_cast<uint64_t>(static_cast<typename F::MaskType>(f));
	}
	template <class E>
	static uint32_t U(E e) {
		return static_cast<uint32_t>(e);
	}
	void Range(const vk::ImageSubresourceRange& r) {
		Mix(M(r.aspectMask));
		Mix(r.baseMipLevel);
		Mix(r.levelCount);
		Mix(r.baseArrayLayer);
		Mix(r.layerCount);
	}
	void Leaf(const vk::MemoryBarrier2& b) {
		Mix(b.pNext != nullptr);
		Mix(M(b.srcStageMask));
		Mix(M(b.srcAccessMask));
		Mix(M(b.dstStageMask));
		Mix(M(b.dstAccessMask));
	}
	void Leaf(const vk::BufferMemoryBarrier2& b) {
		Mix(b.pNext != nullptr);
		Mix(M(b.srcStageMask));
		Mix(M(b.srcAccessMask));
		Mix(M(b.dstStageMask));
		Mix(M(b.dstAccessMask));
		Mix(b.srcQueueFamilyIndex);
		Mix(b.dstQueueFamilyIndex);
		Handle(b.buffer);
		Mix(b.offset);
		Mix(b.size);
	}
	void Leaf(const vk::ImageMemoryBarrier2& b) {
		Mix(b.pNext != nullptr);
		Mix(M(b.srcStageMask));
		Mix(M(b.srcAccessMask));
		Mix(M(b.dstStageMask));
		Mix(M(b.dstAccessMask));
		Mix(U(b.oldLayout));
		Mix(U(b.newLayout));
		Mix(b.srcQueueFamilyIndex);
		Mix(b.dstQueueFamilyIndex);
		Handle(b.image);
		Range(b.subresourceRange);
	}
	void Leaf(const vk::MemoryBarrier& b) {
		Mix(b.pNext != nullptr);
		Mix(M(b.srcAccessMask));
		Mix(M(b.dstAccessMask));
	}
	void Leaf(const vk::BufferMemoryBarrier& b) {
		Mix(b.pNext != nullptr);
		Mix(M(b.srcAccessMask));
		Mix(M(b.dstAccessMask));
		Mix(b.srcQueueFamilyIndex);
		Mix(b.dstQueueFamilyIndex);
		Handle(b.buffer);
		Mix(b.offset);
		Mix(b.size);
	}
	void Leaf(const vk::ImageMemoryBarrier& b) {
		Mix(b.pNext != nullptr);
		Mix(M(b.srcAccessMask));
		Mix(M(b.dstAccessMask));
		Mix(U(b.oldLayout));
		Mix(U(b.newLayout));
		Mix(b.srcQueueFamilyIndex);
		Mix(b.dstQueueFamilyIndex);
		Handle(b.image);
		Range(b.subresourceRange);
	}
	void Leaf(const vk::DescriptorImageInfo& i) {
		Handle(i.sampler);
		Handle(i.imageView);
		Mix(U(i.imageLayout));
	}
	void Leaf(const vk::DescriptorBufferInfo& i) {
		Handle(i.buffer);
		Mix(i.offset);
		Mix(i.range);
	}
	void Leaf(const vk::RenderingAttachmentInfo& a) {
		Mix(a.pNext != nullptr);
		Handle(a.imageView);
		Mix(U(a.imageLayout));
		Mix(U(a.resolveMode));
		Handle(a.resolveImageView);
		Mix(U(a.resolveImageLayout));
		Mix(U(a.loadOp));
		Mix(U(a.storeOp));
		Bytes(&a.clearValue, sizeof(a.clearValue)); // a union of 16 bytes, no padding
	}
	void Small(Op op, uint32_t a, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0, uint32_t e = 0) {
		Begin(op);
		Mix(a);
		Mix(b);
		Mix(c);
		Mix(d);
		Mix(e);
	}
	void Indirect(Op op, vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) {
		Begin(op);
		Handle(buffer);
		Quad(offset);
		Word(count);
		Word(stride);
	}

	uint64_t m_h = 0x84222325CBF29CE4ull;
	uint64_t m_n = 0;
};

// The region structures hashed by their bytes have no padding (S565).
static_assert(sizeof(vk::BufferCopy) == 24 && sizeof(vk::BufferImageCopy) == 56 && sizeof(vk::ImageCopy) == 68 &&
              sizeof(vk::ImageSubresourceRange) == 20 && sizeof(vk::ClearColorValue) == 16 && sizeof(vk::ClearDepthStencilValue) == 8);

// ---------------------------------------------------------------------------------------------------------------------------------------
// Arena: 64 KB chunks, bump allocation, kept for reuse after Reset.

class Arena {
public:
	static constexpr size_t kChunk = 64u * 1024u;

	Arena() = default;
	Arena(const Arena&)            = delete;
	Arena& operator=(const Arena&) = delete;

	// `n` is a multiple of 8; the result is 8-aligned.
	uint8_t* Alloc(size_t n) {
		if (static_cast<size_t>(m_end - m_ptr) < n) [[unlikely]] {
			Grow(n);
		}
		uint8_t* p = m_ptr;
		m_ptr += n;
		return p;
	}

	void Reset() noexcept {
		m_cur = 0;
		if (m_chunks.empty()) {
			m_ptr = m_end = nullptr;
			return;
		}
		m_ptr = m_chunks[0].data.get();
		m_end = m_ptr + m_chunks[0].cap;
	}

	// S565: Alloc(n) stays in the current chunk; the next byte Alloc hands out.
	[[nodiscard]] bool           Fits(size_t n) const noexcept { return static_cast<size_t>(m_end - m_ptr) >= n && m_ptr != nullptr; }
	[[nodiscard]] const uint8_t* Cursor() const noexcept { return m_ptr; }

	[[nodiscard]] size_t Used() const noexcept {
		size_t n = 0;
		ForEach([&](const uint8_t*, size_t used) { n += used; });
		return n;
	}
	[[nodiscard]] size_t Chunks() const noexcept { return m_chunks.size(); }

	template <class F>
	void ForEach(F&& f) const {
		if (m_ptr == nullptr) return;
		for (size_t i = 0; i < m_cur; i++) {
			f(m_chunks[i].data.get(), m_chunks[i].used);
		}
		f(m_chunks[m_cur].data.get(), static_cast<size_t>(m_ptr - m_chunks[m_cur].data.get()));
	}

private:
	struct Chunk {
		std::unique_ptr<uint8_t[]> data;
		size_t                     cap  = 0;
		size_t                     used = 0;
	};

	void Grow(size_t n) {
		if (m_ptr != nullptr) {
			m_chunks[m_cur].used = static_cast<size_t>(m_ptr - m_chunks[m_cur].data.get());
			m_cur++;
		}
		if (m_cur == m_chunks.size() || m_chunks[m_cur].cap < n) {
			Chunk c;
			c.cap  = n > kChunk ? n : kChunk;
			c.data = std::make_unique<uint8_t[]>(c.cap);
			m_chunks.insert(m_chunks.begin() + static_cast<std::ptrdiff_t>(m_cur), std::move(c));
		}
		m_ptr = m_chunks[m_cur].data.get();
		m_end = m_ptr + m_chunks[m_cur].cap;
	}

	std::vector<Chunk> m_chunks;
	size_t             m_cur = 0;
	uint8_t*           m_ptr = nullptr;
	uint8_t*           m_end = nullptr;
};

// ---------------------------------------------------------------------------------------------------------------------------------------
// S565 thread mode: the emitter.
//
// Segments s = 1, 2, ... (a range of arena bytes and the command buffer it is replayed into) live in ring slot s % capacity. The recorder
// (producer) writes slot s once segment s - capacity is replayed (done >= s - capacity), then publishes tail = s. Segments are replayed in
// order, each exactly once, by the holder of m_owner: the emitter thread (1) or, at a drain while the emitter does not hold it, the recorder
// itself (2). The holder replays done + 1 .. tail and stores done (release) after each segment, and releases m_owner after its last one, so
// the next holder (acquire) continues where it stopped. The recorder touches the VkCommandBuffer (and its pool) directly only after it has
// seen done == the last segment it pushed (acquire) and before it pushes again; in that window there is nothing for the emitter to replay.
// The emitter parks on tail (atomic wait) with a target `batch` segments ahead; Push wakes it only when it reaches the target, so a busy
// recorder pays one wake-up per `batch` segments. A drain does not wake it: it takes m_owner and replays the rest itself (no wake-up latency
// on the recorder's path); with steal off (KYTY_CP_EMIT_STEAL=0) it wakes the emitter and spins until done.

inline void CpuRelax() {
#if defined(KYTY_CPEMIT_TSC)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

// vk::CommandBuffer <-> the 64-bit value a segment carries (a test's fake command buffer specialises this).
template <class Raw>
struct RawTraits {
	static uint64_t Bits(const Raw& raw) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<typename Raw::CType>(raw))); }
	static Raw      From(uint64_t bits) { return Raw(reinterpret_cast<typename Raw::CType>(static_cast<uintptr_t>(bits))); }
};

struct EmitterConfig {
	uint32_t capacity      = 1024; // ring slots (segments)
	uint32_t batch         = 4;    // a parked emitter is woken when this many segments wait
	uint32_t spin          = 0;    // pause loops the emitter polls for work before it parks
	uint32_t publish_bytes = 2048; // the recorder publishes at the end of a draw / dispatch once this many bytes wait (16x that anywhere)
	bool     steal         = true; // a drain replays the rest itself when the emitter does not hold m_owner
};

inline uint32_t EnvU32(const char* name, uint32_t fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || v[0] == '\0') return fallback;
	return static_cast<uint32_t>(std::strtoul(v, nullptr, 10));
}

// KYTY_CP_EMIT_BATCH, KYTY_CP_EMIT_SPIN, KYTY_CP_EMIT_PUBLISH_BYTES, KYTY_CP_EMIT_STEAL (0|1).
inline EmitterConfig EnvEmitterConfig() {
	EmitterConfig c;
	c.batch         = std::max(1u, EnvU32("KYTY_CP_EMIT_BATCH", c.batch));
	c.spin          = EnvU32("KYTY_CP_EMIT_SPIN", c.spin);
	c.publish_bytes = std::max(64u, EnvU32("KYTY_CP_EMIT_PUBLISH_BYTES", c.publish_bytes));
	c.steal         = EnvU32("KYTY_CP_EMIT_STEAL", 1) != 0;
	return c;
}

// Counters the emitter thread writes (read by the recorder with ReadStats: one line, written by the emitter only).
struct EmitterStats {
	uint64_t busy_ticks = 0; // replaying (TSC ticks)
	uint64_t segments   = 0; // segments it replayed
	uint64_t passes     = 0; // times it took m_owner and found work
	uint64_t depth_sum  = 0; // segments waiting when a pass started (sum over passes)
	uint64_t depth_max  = 0;
	uint64_t parks      = 0; // times it blocked in the atomic wait
};

class Emitter {
public:
	using ReplayFn = void (*)(uint64_t raw, const uint8_t* p, size_t n);

	// `on_start` runs first on the emitter thread (name, affinity).
	Emitter(ReplayFn replay, EmitterConfig config, std::function<void()> on_start = {})
	    : m_replay(replay), m_cfg(config), m_ring(std::max<uint32_t>(config.capacity, 2u)) {
		m_thread = std::thread([this, start = std::move(on_start)] {
			if (start) start();
			Run();
		});
	}
	~Emitter() { Stop(); }
	Emitter(const Emitter&)            = delete;
	Emitter& operator=(const Emitter&) = delete;

	[[nodiscard]] const EmitterConfig& Config() const noexcept { return m_cfg; }

	// ---- recorder ----
	struct Waited {
		uint64_t ticks      = 0;
		uint64_t steal_segs = 0; // segments the recorder replayed itself
		bool     stole      = false;
		bool     woke       = false;
	};

	// Publishes [p, p + n) to be replayed into `raw`. Blocks (or replays) while the ring is full.
	Waited Push(uint64_t raw, const uint8_t* p, size_t n) {
		Waited         w;
		const uint64_t seq = m_next;
		const uint64_t cap = m_ring.size();
		if (seq > cap) {
			const uint64_t need = seq - cap;
			if (m_done_seen < need && (m_done_seen = m_done.load(std::memory_order_acquire)) < need) w = WaitDone(need);
		}
		m_ring[seq % cap] = {raw, p, n};
		m_next            = seq + 1;
		m_tail.store(seq, std::memory_order_seq_cst);
		uint64_t target = m_target.load(std::memory_order_seq_cst);
		if (target != 0 && seq >= target && m_target.compare_exchange_strong(target, 0, std::memory_order_seq_cst)) {
			Wake();
			w.woke = true;
		}
		return w;
	}

	// Returns once every pushed segment is replayed (acquire: the recorder may then touch the command buffer).
	Waited WaitIdle() {
		const uint64_t want = m_next - 1;
		if (m_done_seen >= want || (m_done_seen = m_done.load(std::memory_order_acquire)) >= want) return {};
		return WaitDone(want);
	}

	// Test / diag: every pushed segment is replayed (recorder thread).
	[[nodiscard]] bool IdleNow() const { return m_done.load(std::memory_order_acquire) >= m_next - 1; }
	[[nodiscard]] uint64_t Pushed() const noexcept { return m_next - 1; }
	// Test: the recorder holds m_owner (it replays the rest itself at a drain).
	[[nodiscard]] bool HeldByRecorder() const { return m_owner.load(std::memory_order_acquire) == 2; }

	[[nodiscard]] EmitterStats ReadStats() const {
		EmitterStats s;
		s.busy_ticks = m_busy.load(std::memory_order_relaxed);
		s.segments   = m_segments.load(std::memory_order_relaxed);
		s.passes     = m_passes.load(std::memory_order_relaxed);
		s.depth_sum  = m_depth_sum.load(std::memory_order_relaxed);
		s.depth_max  = m_depth_max.load(std::memory_order_relaxed);
		s.parks      = m_parks.load(std::memory_order_relaxed);
		return s;
	}

	// Recorder thread, with nothing left to record: replays the rest, stops and joins the emitter.
	void Stop() {
		if (!m_thread.joinable()) return;
		(void)WaitIdle();
		m_stop.store(true, std::memory_order_seq_cst);
		(void)Push(0, nullptr, 0); // a stop marker
		m_target.store(0, std::memory_order_seq_cst);
		Wake();
		m_thread.join();
	}

private:
	struct Segment {
		uint64_t       raw = 0;
		const uint8_t* p   = nullptr; // nullptr: the stop marker
		size_t         n   = 0;
	};

	// The caller holds m_owner: replays done + 1 .. upto in order.
	uint64_t ReplayOwned(uint64_t upto) {
		uint64_t done = m_done.load(std::memory_order_relaxed);
		uint64_t n    = 0;
		while (done < upto) {
			const Segment& s = m_ring[(done + 1) % m_ring.size()];
			if (s.p != nullptr) m_replay(s.raw, s.p, s.n);
			done++;
			n++;
			m_done.store(done, std::memory_order_release);
		}
		return n;
	}

	// Recorder: waits for done >= want; replays the rest itself whenever m_owner is free (steal on).
	Waited WaitDone(uint64_t want) {
		Waited         w;
		const uint64_t t0    = Ticks();
		uint64_t       spins = 0;
		for (;;) {
			uint32_t free_owner = 0;
			if (m_cfg.steal && m_owner.load(std::memory_order_relaxed) == 0 &&
			    m_owner.compare_exchange_strong(free_owner, 2, std::memory_order_acq_rel)) {
				const uint64_t done_before = m_done.load(std::memory_order_relaxed);
				const uint64_t tail        = m_next - 1;
				const uint64_t n           = ReplayOwned(tail);
				m_owner.store(0, std::memory_order_release);
				w.steal_segs += n;
				w.stole = w.stole || n != 0;
				(void)done_before;
				m_done_seen = tail;
				break;
			}
			if (!m_cfg.steal && !w.woke) {
				uint64_t target = m_target.load(std::memory_order_seq_cst);
				if (target != 0 && m_target.compare_exchange_strong(target, 0, std::memory_order_seq_cst)) {
					Wake();
					w.woke = true;
				}
			}
			if ((m_done_seen = m_done.load(std::memory_order_acquire)) >= want) break;
			CpuRelax();
			if ((++spins & 0xFFFFu) == 0) std::this_thread::yield(); // the emitter holds m_owner but is descheduled
		}
		w.ticks = Ticks() - t0;
		return w;
	}

	void Run() {
		for (;;) {
			uint32_t free_owner = 0;
			if (m_owner.load(std::memory_order_relaxed) == 0 && m_owner.compare_exchange_strong(free_owner, 1, std::memory_order_acq_rel)) {
				uint64_t       tail  = m_tail.load(std::memory_order_acquire);
				uint64_t       done  = m_done.load(std::memory_order_relaxed);
				const uint64_t t0    = done < tail ? Ticks() : 0;
				uint64_t       total = 0;
				if (done < tail) {
					const uint64_t depth = tail - done;
					m_passes.store(m_passes.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
					m_depth_sum.store(m_depth_sum.load(std::memory_order_relaxed) + depth, std::memory_order_relaxed);
					if (depth > m_depth_max.load(std::memory_order_relaxed)) m_depth_max.store(depth, std::memory_order_relaxed);
				}
				while (done < tail) {
					total += ReplayOwned(tail);
					done = tail;
					tail = m_tail.load(std::memory_order_acquire);
				}
				m_owner.store(0, std::memory_order_release);
				if (total != 0) {
					m_busy.store(m_busy.load(std::memory_order_relaxed) + (Ticks() - t0), std::memory_order_relaxed);
					m_segments.store(m_segments.load(std::memory_order_relaxed) + total, std::memory_order_relaxed);
				}
			}
			const uint64_t done = m_done.load(std::memory_order_acquire);
			if (m_tail.load(std::memory_order_acquire) > done) {
				CpuRelax(); // more work, or the recorder holds m_owner for a moment
				continue;
			}
			if (m_stop.load(std::memory_order_acquire)) return;
			bool more = false;
			for (uint32_t i = 0; i < m_cfg.spin && !more; i++) {
				CpuRelax();
				more = m_tail.load(std::memory_order_acquire) > done;
			}
			if (!more) Park(done);
		}
	}

	// Blocks until the recorder pushed `batch` segments past `done` (it then clears the target and wakes it) or a stop. The target is set
	// and tail checked under the mutex the waker takes before notifying, so a wake-up between the check and the block is not lost (the
	// recorder: store tail, load target; the emitter: store target, load tail, both seq_cst: one of them sees the other).
	// A mutex + condition variable, not std::atomic::wait: with atomic wait/notify the macOS (libc++) test run hung on a lost wake-up.
	void Park(uint64_t done) {
		const uint64_t   want = done + m_cfg.batch;
		std::unique_lock lock(m_park_mutex);
		m_target.store(want, std::memory_order_seq_cst);
		if (m_tail.load(std::memory_order_seq_cst) == done && !m_stop.load(std::memory_order_seq_cst)) {
			m_parks.store(m_parks.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
			m_park_cv.wait(lock, [&] { return m_target.load(std::memory_order_seq_cst) != want || m_stop.load(std::memory_order_seq_cst); });
		}
		uint64_t mine = want;
		m_target.compare_exchange_strong(mine, 0, std::memory_order_seq_cst);
	}
	// Recorder, after it cleared the target.
	void Wake() {
		{ std::lock_guard lock(m_park_mutex); }
		m_park_cv.notify_one();
	}

	ReplayFn             m_replay;
	EmitterConfig        m_cfg;
	std::vector<Segment> m_ring;
	// Recorder-private.
	uint64_t m_next      = 1;
	uint64_t m_done_seen = 0;
	// One line each: the recorder writes tail, the replaying holder done, the emitter target (and the recorder clears it), owner both.
	alignas(64) std::atomic<uint64_t> m_tail {0};
	alignas(64) std::atomic<uint64_t> m_done {0};
	alignas(64) std::atomic<uint64_t> m_target {0};
	alignas(64) std::atomic<uint32_t> m_owner {0}; // 0 free, 1 emitter, 2 recorder
	std::atomic<bool>                 m_stop {false};
	// Emitter-written counters.
	alignas(64) std::atomic<uint64_t> m_busy {0};
	std::atomic<uint64_t>             m_segments {0};
	std::atomic<uint64_t>             m_passes {0};
	std::atomic<uint64_t>             m_depth_sum {0};
	std::atomic<uint64_t>             m_depth_max {0};
	std::atomic<uint64_t>             m_parks {0};
	std::mutex                        m_park_mutex;
	std::condition_variable           m_park_cv;
	std::thread                       m_thread;
};

// ---------------------------------------------------------------------------------------------------------------------------------------
// The thread's pending stream (MasterSemaphore::Wait drains it). Set when a stream gets its first command, cleared by its drain.

struct PendingSlot {
	void* owner                       = nullptr;
	void (*drain)(void* owner, Reason) = nullptr;
};

inline PendingSlot& ThreadPending() {
	static thread_local PendingSlot slot;
	return slot;
}

inline void DrainThreadPending(Reason why) {
	auto& slot = ThreadPending();
	if (slot.owner != nullptr) slot.drain(slot.owner, why);
}

// ---------------------------------------------------------------------------------------------------------------------------------------

struct Hdr {
	Op       op;
	uint16_t pad;
	uint32_t size; // whole record, header included, multiple of 8
};
static_assert(sizeof(Hdr) == 8);

constexpr size_t Align8(size_t n) {
	return (n + 7u) & ~size_t {7u};
}

class Stream {
public:
	explicit Stream(Mode mode = EnvMode()): m_mode(mode) {}
	~Stream() {
		auto& slot = ThreadPending();
		if (m_owner != nullptr && slot.owner == m_owner) slot = {};
	}
	Stream(const Stream&)            = delete;
	Stream& operator=(const Stream&) = delete;

	[[nodiscard]] Mode Current() const noexcept { return m_mode; }
	[[nodiscard]] bool Pending() const noexcept { return m_ops != 0; }
	[[nodiscard]] uint32_t PendingOps() const noexcept { return m_ops; }
	[[nodiscard]] size_t   ArenaChunks() const noexcept { return m_arena.Chunks(); }

	// `drain(owner, reason)` drains this stream into the owner's current buffer (for the thread's pending slot).
	void SetOwner(void* owner, void (*drain)(void*, Reason)) noexcept {
		m_owner = owner;
		m_drain = drain;
	}

	// S565: thread mode. The stream's segments are replayed by `emitter` (nullptr: as sync). Only with an empty stream.
	void AttachEmitter(Emitter* emitter) noexcept {
		m_emitter       = m_mode == Mode::Thread ? emitter : nullptr;
		m_publish_bytes = emitter != nullptr ? emitter->Config().publish_bytes : 2048u;
		m_pub           = nullptr;
	}
	[[nodiscard]] Emitter* GetEmitter() const noexcept { return m_emitter; }
	// The command buffer the next segments are replayed into (RawTraits bits); set by Begin, the stream is empty then.
	void BindRaw(uint64_t bits) noexcept { m_raw_bits = bits; }

	// Thread mode: hands the bytes written since the last publish to the emitter.
	void Publish() {
		if (m_pub == nullptr) return;
		const uint8_t* end = m_arena.Cursor();
		if (end <= m_pub) return;
		const auto w = m_emitter->Push(m_raw_bits, m_pub, static_cast<size_t>(end - m_pub));
		m_pub        = end;
		m_stats.publishes++;
		NoteWaited(w, Reason::Full);
	}

	// Emitter::ReplayFn of a vk::CommandBuffer-like Raw: decodes the records of one segment.
	template <class Raw>
	static void ReplaySegment(uint64_t raw_bits, const uint8_t* p, size_t n) {
		Raw raw = RawTraits<Raw>::From(raw_bits);
		ReplayRange(raw, p, n);
	}
	template <class Sink>
	static void ReplayRange(Sink& sink, const uint8_t* base, size_t used) {
		size_t at = 0;
		while (at < used) {
			const auto* h = reinterpret_cast<const Hdr*>(base + at);
			Decode(sink, h->op, base + at + sizeof(Hdr));
			at += h->size;
		}
	}

	[[nodiscard]] Stats TakeStats() noexcept {
		Stats s = m_stats;
		m_stats = {};
		return s;
	}

	// Replays the stream into `raw` (sync) or into the digest compared with the direct one (shadow), then empties it.
	template <class Raw>
	void Drain(Raw raw, Reason why) {
		if (m_ops == 0) return;
		auto& slot = ThreadPending();
		if (slot.owner != nullptr && slot.owner == m_owner) slot = {};
		const uint64_t t0 = Ticks();
		if (m_mode == Mode::Shadow) {
			HashSink replayed;
			Replay(replayed);
			if (replayed.Value() != m_direct.Value() || replayed.Count() != m_direct.Count()) m_stats.ne++;
			m_direct.Reset();
		} else if (m_emitter != nullptr) {
			// S565: what is left goes to the emitter; then wait until it (or this thread, when it does not hold the command buffer) replayed all.
			Publish();
#if defined(KYTY_EMIT_MUTANT_TOUCH_DURING_EMIT)
			if (why == Reason::Handle) return; // the direct site records while the emitter may still replay
#endif
#if defined(KYTY_EMIT_MUTANT_NO_SUBMIT_DRAIN)
			if (why != Reason::Submit) NoteWaited(m_emitter->WaitIdle(), why);
#else
			NoteWaited(m_emitter->WaitIdle(), why);
#endif
			m_pub = nullptr;
			(void)raw;
		} else {
			Replay(raw);
		}
		if (m_emitter == nullptr) m_stats.replay_ticks += Ticks() - t0;
		m_stats.drains++;
		m_stats.by[static_cast<size_t>(why)]++;
		m_stats.ops += m_ops;
		m_stats.bytes += m_arena.Used();
		m_ops = 0;
		m_arena.Reset();
	}

	// Decodes every record in order into `sink` (vk::CommandBuffer, HashSink, a test sink).
	template <class Sink>
	void Replay(Sink& sink) const {
		m_arena.ForEach([&](const uint8_t* base, size_t used) {
			size_t at = 0;
			while (at < used) {
				const auto* h = reinterpret_cast<const Hdr*>(base + at);
				Decode(sink, h->op, base + at + sizeof(Hdr));
				at += h->size;
			}
		});
	}

	// ---- recording (only called with mode != Off). The struct ones return false when the IR cannot hold the command. ----

	bool Barrier2(const vk::DependencyInfo& d) {
		if (d.pNext != nullptr || !NoNext(d.pMemoryBarriers, d.memoryBarrierCount) || !NoNext(d.pBufferMemoryBarriers, d.bufferMemoryBarrierCount) ||
		    !NoNext(d.pImageMemoryBarriers, d.imageMemoryBarrierCount)) {
			m_stats.fallback++;
			return false;
		}
		Rec     r(*this);
		Writer  w(*this, Op::Barrier2,
		          sizeof(B2) + sizeof(vk::MemoryBarrier2) * d.memoryBarrierCount + sizeof(vk::BufferMemoryBarrier2) * d.bufferMemoryBarrierCount +
		              sizeof(vk::ImageMemoryBarrier2) * d.imageMemoryBarrierCount);
		auto* p = w.Put<B2>();
		p->flags = static_cast<uint32_t>(d.dependencyFlags);
		p->nm    = d.memoryBarrierCount;
		p->nb    = d.bufferMemoryBarrierCount;
		p->ni    = d.imageMemoryBarrierCount;
		w.Copy(d.pMemoryBarriers, d.memoryBarrierCount);
		w.Copy(d.pBufferMemoryBarriers, d.bufferMemoryBarrierCount);
		w.Copy(d.pImageMemoryBarriers, d.imageMemoryBarrierCount);
		if (m_mode == Mode::Shadow) m_direct.pipelineBarrier2(d);
		return true;
	}

	bool Barrier1(vk::PipelineStageFlags src, vk::PipelineStageFlags dst, vk::DependencyFlags flags, uint32_t nm, const vk::MemoryBarrier* m,
	              uint32_t nb, const vk::BufferMemoryBarrier* b, uint32_t ni, const vk::ImageMemoryBarrier* i) {
		if (!NoNext(m, nm) || !NoNext(b, nb) || !NoNext(i, ni)) {
			m_stats.fallback++;
			return false;
		}
		Rec    r(*this);
		Writer w(*this, Op::Barrier1,
		         sizeof(B1) + sizeof(vk::MemoryBarrier) * nm + sizeof(vk::BufferMemoryBarrier) * nb + sizeof(vk::ImageMemoryBarrier) * ni);
		auto* p  = w.Put<B1>();
		p->src   = static_cast<uint32_t>(src);
		p->dst   = static_cast<uint32_t>(dst);
		p->flags = static_cast<uint32_t>(flags);
		p->nm    = nm;
		p->nb    = nb;
		p->ni    = ni;
		w.Copy(m, nm);
		w.Copy(b, nb);
		w.Copy(i, ni);
		if (m_mode == Mode::Shadow) m_direct.pipelineBarrier(src, dst, flags, nm, m, nb, b, ni, i);
		return true;
	}

	void PushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data) {
		Rec    r(*this);
		Writer w(*this, Op::PushConstants, sizeof(PC) + size);
		auto*  p = w.Put<PC>();
		p->layout = static_cast<VkPipelineLayout>(layout);
		p->stages = static_cast<uint32_t>(stages);
		p->offset = offset;
		p->size   = size;
		w.CopyBytes(data, size);
		if (m_mode == Mode::Shadow) m_direct.pushConstants(layout, stages, offset, size, data);
	}

	bool PushDescriptorSet(vk::PipelineBindPoint bp, vk::PipelineLayout layout, uint32_t set, uint32_t count, const vk::WriteDescriptorSet* writes) {
		size_t infos = 0;
		for (uint32_t k = 0; k < count; k++) {
			const auto& x = writes[k];
			if (x.pNext != nullptr) {
				m_stats.fallback++;
				return false;
			}
			if (x.pImageInfo != nullptr) infos += Align8(sizeof(vk::DescriptorImageInfo) * x.descriptorCount);
			if (x.pBufferInfo != nullptr) infos += Align8(sizeof(vk::DescriptorBufferInfo) * x.descriptorCount);
			if (x.pTexelBufferView != nullptr) infos += Align8(sizeof(vk::BufferView) * x.descriptorCount);
		}
		Rec    r(*this);
		Writer w(*this, Op::PushDescriptorSet, sizeof(PD) + sizeof(vk::WriteDescriptorSet) * count + infos);
		auto*  p  = w.Put<PD>();
		p->bp     = static_cast<uint32_t>(bp);
		p->set    = set;
		p->layout = static_cast<VkPipelineLayout>(layout);
		p->count  = count;
		w.Copy(writes, count);
		for (uint32_t k = 0; k < count; k++) {
			const auto& x = writes[k];
#if defined(KYTY_EMIT_MUTANT_SHALLOW)
			(void)x; // the copied writes keep pointing at the caller's info arrays
#else
			if (x.pImageInfo != nullptr) w.Copy(x.pImageInfo, x.descriptorCount);
			if (x.pBufferInfo != nullptr) w.Copy(x.pBufferInfo, x.descriptorCount);
			if (x.pTexelBufferView != nullptr) w.Copy(x.pTexelBufferView, x.descriptorCount);
#endif
		}
		if (m_mode == Mode::Shadow) m_direct.pushDescriptorSetKHR(bp, layout, set, count, writes);
		return true;
	}

	void BindDescriptorSets(vk::PipelineBindPoint bp, vk::PipelineLayout layout, uint32_t first, uint32_t count, const vk::DescriptorSet* sets,
	                        uint32_t ndyn, const uint32_t* dyn) {
		Rec    r(*this);
		Writer w(*this, Op::BindDescriptorSets, sizeof(BD) + sizeof(vk::DescriptorSet) * count + Align8(sizeof(uint32_t) * ndyn));
		auto*  p  = w.Put<BD>();
		p->bp     = static_cast<uint32_t>(bp);
		p->first  = first;
		p->layout = static_cast<VkPipelineLayout>(layout);
		p->count  = count;
		p->ndyn   = ndyn;
		w.Copy(sets, count);
		w.Copy(dyn, ndyn);
		if (m_mode == Mode::Shadow) m_direct.bindDescriptorSets(bp, layout, first, count, sets, ndyn, dyn);
	}

	void BindPipeline(vk::PipelineBindPoint bp, vk::Pipeline pipeline) {
		Rec    r(*this);
		Writer w(*this, Op::BindPipeline, sizeof(BP));
		auto*  p    = w.Put<BP>();
		p->bp       = static_cast<uint32_t>(bp);
		p->pipeline = static_cast<VkPipeline>(pipeline);
		if (m_mode == Mode::Shadow) m_direct.bindPipeline(bp, pipeline);
	}

	void BindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers, const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) {
		Rec          r(*this);
		const size_t per = sizeof(vk::Buffer) + sizeof(vk::DeviceSize) * (1u + (sizes != nullptr ? 1u : 0u) + (strides != nullptr ? 1u : 0u));
		Writer       w(*this, Op::BindVertexBuffers2, sizeof(VB) + per * count);
		auto*        p = w.Put<VB>();
		p->first       = first;
		p->count       = count;
		p->has_sizes   = sizes != nullptr ? 1u : 0u;
		p->has_strides = strides != nullptr ? 1u : 0u;
		w.Copy(buffers, count);
		w.Copy(offsets, count);
		if (sizes != nullptr) w.Copy(sizes, count);
		if (strides != nullptr) w.Copy(strides, count);
		if (m_mode == Mode::Shadow) m_direct.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
	}

	void BindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) {
		Rec    r(*this);
		Writer w(*this, Op::BindIndexBuffer, sizeof(IB));
		auto*  p  = w.Put<IB>();
		p->buffer = static_cast<VkBuffer>(buffer);
		p->offset = offset;
		p->type   = static_cast<uint32_t>(type);
		if (m_mode == Mode::Shadow) m_direct.bindIndexBuffer(buffer, offset, type);
	}

	void SetViewports(uint32_t count, const vk::Viewport* v) {
		Rec    r(*this);
		Writer w(*this, Op::SetViewportWithCount, sizeof(CountHdr) + Align8(sizeof(vk::Viewport) * count));
		w.Put<CountHdr>()->count = count;
		w.Copy(v, count);
		if (m_mode == Mode::Shadow) m_direct.setViewportWithCount(count, v);
	}

	void SetScissors(uint32_t count, const vk::Rect2D* s) {
		Rec    r(*this);
		Writer w(*this, Op::SetScissorWithCount, sizeof(CountHdr) + sizeof(vk::Rect2D) * count);
		w.Put<CountHdr>()->count = count;
		w.Copy(s, count);
		if (m_mode == Mode::Shadow) m_direct.setScissorWithCount(count, s);
	}

	// Commands of up to six 32-bit words (dynamic state, direct draws).
	template <size_t N>
	void Words(Op op, const std::array<uint32_t, N>& words) {
		static_assert(N >= 1 && N <= 6);
		Rec    r(*this);
		Writer w(*this, op, Align8(sizeof(uint32_t) * N));
		w.CopyWords(words.data(), N);
	}

	void Indirect(Op op, vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) {
		Rec    r(*this);
		Writer w(*this, op, sizeof(ID));
		auto*  p  = w.Put<ID>();
		p->buffer = static_cast<VkBuffer>(buffer);
		p->offset = offset;
		p->count  = count;
		p->stride = stride;
	}

	bool BeginRendering(const vk::RenderingInfo& info) {
		if (info.pNext != nullptr || !NoNext(info.pColorAttachments, info.colorAttachmentCount) || !NoNext(info.pDepthAttachment, info.pDepthAttachment ? 1u : 0u) ||
		    !NoNext(info.pStencilAttachment, info.pStencilAttachment ? 1u : 0u)) {
			m_stats.fallback++;
			return false;
		}
		const uint32_t n = info.colorAttachmentCount + (info.pDepthAttachment != nullptr ? 1u : 0u) + (info.pStencilAttachment != nullptr ? 1u : 0u);
		Rec            r(*this);
		Writer         w(*this, Op::BeginRendering, sizeof(BR) + sizeof(vk::RenderingAttachmentInfo) * n);
		auto*          p = w.Put<BR>();
		p->flags         = static_cast<uint32_t>(info.flags);
		p->layers        = info.layerCount;
		p->view_mask     = info.viewMask;
		p->colors        = info.colorAttachmentCount;
		p->has_depth     = info.pDepthAttachment != nullptr ? 1u : 0u;
		p->has_stencil   = info.pStencilAttachment != nullptr ? 1u : 0u;
		p->area          = info.renderArea;
		w.Copy(info.pColorAttachments, info.colorAttachmentCount);
		if (info.pDepthAttachment != nullptr) w.Copy(info.pDepthAttachment, 1);
		if (info.pStencilAttachment != nullptr) w.Copy(info.pStencilAttachment, 1);
		if (m_mode == Mode::Shadow) m_direct.beginRendering(info);
		return true;
	}

	void EndRendering() {
		Rec    r(*this);
		Writer w(*this, Op::EndRendering, 0);
		if (m_mode == Mode::Shadow) m_direct.endRendering();
	}

	// S565: the commands of the sites outside the draw seam.
	void DispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) {
		Rec    r(*this);
		Writer w(*this, Op::DispatchIndirect, sizeof(DI));
		auto*  p  = w.Put<DI>();
		p->buffer = static_cast<VkBuffer>(buffer);
		p->offset = offset;
		if (m_mode == Mode::Shadow) m_direct.dispatchIndirect(buffer, offset);
	}
	void CopyBuffer(vk::Buffer src, vk::Buffer dst, uint32_t n, const vk::BufferCopy* regions) {
		Rec    r(*this);
		Writer w(*this, Op::CopyBuffer, sizeof(CB) + sizeof(vk::BufferCopy) * n);
		auto*  p = w.Put<CB>();
		p->src   = static_cast<VkBuffer>(src);
		p->dst   = static_cast<VkBuffer>(dst);
		p->n     = n;
		p->pad   = 0;
		w.Copy(regions, n);
		if (m_mode == Mode::Shadow) m_direct.copyBuffer(src, dst, n, regions);
	}
	void CopyBufferToImage(vk::Buffer src, vk::Image dst, vk::ImageLayout layout, uint32_t n, const vk::BufferImageCopy* regions) {
		Rec    r(*this);
		Writer w(*this, Op::CopyBufferToImage, sizeof(BI) + sizeof(vk::BufferImageCopy) * n);
		auto*  p  = w.Put<BI>();
		p->buffer = static_cast<VkBuffer>(src);
		p->image  = static_cast<VkImage>(dst);
		p->layout = static_cast<uint32_t>(layout);
		p->n      = n;
		w.Copy(regions, n);
		if (m_mode == Mode::Shadow) m_direct.copyBufferToImage(src, dst, layout, n, regions);
	}
	void CopyImageToBuffer(vk::Image src, vk::ImageLayout layout, vk::Buffer dst, uint32_t n, const vk::BufferImageCopy* regions) {
		Rec    r(*this);
		Writer w(*this, Op::CopyImageToBuffer, sizeof(BI) + sizeof(vk::BufferImageCopy) * n);
		auto*  p  = w.Put<BI>();
		p->buffer = static_cast<VkBuffer>(dst);
		p->image  = static_cast<VkImage>(src);
		p->layout = static_cast<uint32_t>(layout);
		p->n      = n;
		w.Copy(regions, n);
		if (m_mode == Mode::Shadow) m_direct.copyImageToBuffer(src, layout, dst, n, regions);
	}
	void CopyImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst, vk::ImageLayout dst_layout, uint32_t n, const vk::ImageCopy* regions) {
		Rec    r(*this);
		Writer w(*this, Op::CopyImage, sizeof(CI) + Align8(sizeof(vk::ImageCopy) * n));
		auto*  p      = w.Put<CI>();
		p->src        = static_cast<VkImage>(src);
		p->dst        = static_cast<VkImage>(dst);
		p->src_layout = static_cast<uint32_t>(src_layout);
		p->dst_layout = static_cast<uint32_t>(dst_layout);
		p->n          = n;
		p->pad        = 0;
		w.Copy(regions, n);
		if (m_mode == Mode::Shadow) m_direct.copyImage(src, src_layout, dst, dst_layout, n, regions);
	}
	void FillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, uint32_t data) {
		Rec    r(*this);
		Writer w(*this, Op::FillBuffer, sizeof(FB));
		auto*  p  = w.Put<FB>();
		p->buffer = static_cast<VkBuffer>(buffer);
		p->offset = offset;
		p->size   = size;
		p->data   = data;
		p->pad    = 0;
		if (m_mode == Mode::Shadow) m_direct.fillBuffer(buffer, offset, size, data);
	}
	void ClearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color, uint32_t n, const vk::ImageSubresourceRange* ranges) {
		Rec    r(*this);
		Writer w(*this, Op::ClearColorImage, sizeof(CL) + Align8(sizeof(vk::ImageSubresourceRange) * n));
		auto*  p   = w.Put<CL>();
		p->image   = static_cast<VkImage>(image);
		p->layout  = static_cast<uint32_t>(layout);
		p->n       = n;
		p->has     = color != nullptr ? 1u : 0u;
		p->pad     = 0;
		std::memset(p->value, 0, sizeof(p->value));
		if (color != nullptr) std::memcpy(p->value, color, sizeof(*color));
		w.Copy(ranges, n);
		if (m_mode == Mode::Shadow) m_direct.clearColorImage(image, layout, color, n, ranges);
	}
	void ClearDepthStencilImage(vk::Image image, vk::ImageLayout layout, const vk::ClearDepthStencilValue* value, uint32_t n,
	                            const vk::ImageSubresourceRange* ranges) {
		Rec    r(*this);
		Writer w(*this, Op::ClearDepthStencilImage, sizeof(CL) + Align8(sizeof(vk::ImageSubresourceRange) * n));
		auto*  p   = w.Put<CL>();
		p->image   = static_cast<VkImage>(image);
		p->layout  = static_cast<uint32_t>(layout);
		p->n       = n;
		p->has     = value != nullptr ? 1u : 0u;
		p->pad     = 0;
		std::memset(p->value, 0, sizeof(p->value));
		if (value != nullptr) std::memcpy(p->value, value, sizeof(*value));
		w.Copy(ranges, n);
		if (m_mode == Mode::Shadow) m_direct.clearDepthStencilImage(image, layout, value, n, ranges);
	}

	// The shadow digest of the direct calls (Words / Indirect commands are folded in by the caller, CmdT).
	HashSink& Direct() noexcept { return m_direct; }

private:
	// Payloads (POD, sizes multiple of 8).
	struct B2 {
		uint32_t flags, nm, nb, ni;
	};
	struct B1 {
		uint32_t src, dst, flags, nm, nb, ni;
	};
	struct PC {
		VkPipelineLayout layout;
		uint32_t         stages, offset, size, pad;
	};
	struct PD {
		uint32_t         bp, set;
		VkPipelineLayout layout;
		uint32_t         count, pad;
	};
	struct BD {
		uint32_t         bp, first;
		VkPipelineLayout layout;
		uint32_t         count, ndyn;
	};
	struct BP {
		uint32_t   bp, pad;
		VkPipeline pipeline;
	};
	struct VB {
		uint32_t first, count, has_sizes, has_strides;
	};
	struct IB {
		VkBuffer     buffer;
		VkDeviceSize offset;
		uint32_t     type, pad;
	};
	struct CountHdr {
		uint32_t count, pad;
	};
	struct ID {
		VkBuffer     buffer;
		VkDeviceSize offset;
		uint32_t     count, stride;
	};
	struct BR {
		uint32_t   flags, layers, view_mask, colors, has_depth, has_stencil;
		vk::Rect2D area;
	};
	struct DI {
		VkBuffer     buffer;
		VkDeviceSize offset;
	};
	struct CB {
		VkBuffer src, dst;
		uint32_t n, pad;
	};
	struct BI {
		VkBuffer buffer;
		VkImage  image;
		uint32_t layout, n;
	};
	struct CI {
		VkImage  src, dst;
		uint32_t src_layout, dst_layout, n, pad;
	};
	struct FB {
		VkBuffer     buffer;
		VkDeviceSize offset, size;
		uint32_t     data, pad;
	};
	struct CL {
		VkImage  image;
		uint32_t layout, n, has, pad;
		uint8_t  value[16]; // ClearColorValue or ClearDepthStencilValue
	};
	static_assert(sizeof(DI) % 8 == 0 && sizeof(CB) % 8 == 0 && sizeof(BI) % 8 == 0 && sizeof(CI) % 8 == 0 && sizeof(FB) % 8 == 0 &&
	              sizeof(CL) % 8 == 0 && sizeof(vk::BufferCopy) % 8 == 0 && sizeof(vk::BufferImageCopy) % 8 == 0);
	// Arrays of these follow one another in a record: each must keep the next one 8-aligned.
	static_assert(sizeof(vk::MemoryBarrier2) % 8 == 0 && sizeof(vk::BufferMemoryBarrier2) % 8 == 0 && sizeof(vk::ImageMemoryBarrier2) % 8 == 0 &&
	              sizeof(vk::MemoryBarrier) % 8 == 0 && sizeof(vk::BufferMemoryBarrier) % 8 == 0 && sizeof(vk::ImageMemoryBarrier) % 8 == 0 &&
	              sizeof(vk::WriteDescriptorSet) % 8 == 0 && sizeof(vk::DescriptorImageInfo) % 8 == 0 && sizeof(vk::DescriptorBufferInfo) % 8 == 0 &&
	              sizeof(vk::RenderingAttachmentInfo) % 8 == 0 && sizeof(vk::Rect2D) % 8 == 0 && sizeof(vk::Buffer) == 8 && sizeof(vk::DeviceSize) == 8 &&
	              sizeof(vk::DescriptorSet) == 8);
	static_assert(sizeof(B2) % 8 == 0 && sizeof(B1) % 8 == 0 && sizeof(PC) % 8 == 0 && sizeof(PD) % 8 == 0 && sizeof(BD) % 8 == 0 && sizeof(BP) % 8 == 0 &&
	              sizeof(VB) % 8 == 0 && sizeof(IB) % 8 == 0 && sizeof(CountHdr) % 8 == 0 && sizeof(ID) % 8 == 0 && sizeof(BR) % 8 == 0);

	template <class T>
	static bool NoNext(const T* p, uint32_t n) {
		for (uint32_t i = 0; i < n; i++) {
			if (p[i].pNext != nullptr) return false;
		}
		return true;
	}

	// Times every kSamplePeriod-th command.
	struct Rec {
		explicit Rec(Stream& s): s(s) {
			if (--s.m_sample_left == 0) [[unlikely]] {
				s.m_sample_left = kSamplePeriod;
				t0              = Ticks();
			}
		}
		~Rec() {
			if (t0 != 0) [[unlikely]] {
				const uint64_t dt = Ticks() - t0;
				const uint64_t oh = TickOverhead();
				s.m_stats.rec_ticks += (dt > oh ? dt - oh : 0) * kSamplePeriod;
			}
		}
		Stream&  s;
		uint64_t t0 = 0;
	};

	// One record: header + payload; the payload size is known up front and rounded to 8.
	struct Writer {
		Writer(Stream& s, Op op, size_t payload): s(s), op(op) {
			const size_t size = sizeof(Hdr) + Align8(payload);
			if (s.m_emitter != nullptr && !s.m_arena.Fits(size)) {
				s.Publish(); // S565: the record opens a new chunk; the tail of this one goes first (a segment never spans chunks)
				s.m_pub = nullptr;
			}
			p       = s.m_arena.Alloc(size);
			if (s.m_pub == nullptr) s.m_pub = p;
			auto* h = reinterpret_cast<Hdr*>(p);
			h->op   = op;
			h->pad  = 0;
			h->size = static_cast<uint32_t>(size);
			p += sizeof(Hdr);
			if (s.m_ops++ == 0) s.NotePending();
		}
		// The record is complete here (the payload is written in the body of the recording method).
		~Writer() {
			if (s.m_emitter != nullptr) s.AfterRecord(op);
		}
		Writer(const Writer&)            = delete;
		Writer& operator=(const Writer&) = delete;
		template <class T>
		T* Put() {
			auto* t = reinterpret_cast<T*>(p);
			p += sizeof(T);
			return t;
		}
		template <class T>
		void Copy(const T* src, uint32_t n) {
			if (n == 0) return;
			const size_t bytes = sizeof(T) * n;
			std::memcpy(p, src, bytes);
			p += Align8(bytes);
		}
		void CopyBytes(const void* src, uint32_t n) {
			if (n == 0) return;
			std::memcpy(p, src, n);
			p += Align8(n);
		}
		void CopyWords(const uint32_t* src, size_t n) {
			std::memcpy(p, src, sizeof(uint32_t) * n);
			p += Align8(sizeof(uint32_t) * n);
		}
		Stream&  s;
		Op       op;
		uint8_t* p;
	};

	// S565: publish at the end of a draw / dispatch once publish_bytes wait, anywhere once 16x that wait.
	static bool EndsWork(Op op) {
		return (op >= Op::Draw && op <= Op::DrawMeshTasksIndirect) || op == Op::Dispatch || op == Op::DispatchBase || op == Op::DispatchIndirect;
	}
	void AfterRecord(Op op) {
		const size_t waiting = static_cast<size_t>(m_arena.Cursor() - m_pub);
		if (waiting >= m_publish_bytes * 16u || (waiting >= m_publish_bytes && EndsWork(op))) Publish();
	}
	void NoteWaited(const Emitter::Waited& w, Reason why) {
		if (w.ticks != 0) {
			m_stats.wait_ticks += w.ticks;
			m_stats.wait_by[static_cast<size_t>(why)] += w.ticks;
		}
		m_stats.steal_segs += w.steal_segs;
		if (w.stole) m_stats.steals++;
		if (w.woke) m_stats.wakes++;
	}

	void NotePending() {
		if (m_owner == nullptr) return;
		auto& slot = ThreadPending();
		if (slot.owner != nullptr && slot.owner != m_owner) slot.drain(slot.owner, Reason::Other);
		slot = {m_owner, m_drain};
	}

	template <class T>
	static const T* Take(const uint8_t*& p, uint32_t n) {
		const auto* t = reinterpret_cast<const T*>(p);
		p += Align8(sizeof(T) * n);
		return n != 0 ? t : nullptr;
	}

	static float F(uint32_t u) {
		float f = 0;
		std::memcpy(&f, &u, 4);
		return f;
	}

	template <class Sink>
	static void Decode(Sink& sink, Op op, const uint8_t* p) {
		const auto* w = reinterpret_cast<const uint32_t*>(p);
		switch (op) {
			case Op::Barrier2: {
				const auto* b = Take<B2>(p, 1);
				vk::DependencyInfo d {};
				d.dependencyFlags          = vk::DependencyFlags(b->flags);
				d.memoryBarrierCount       = b->nm;
				d.pMemoryBarriers          = Take<vk::MemoryBarrier2>(p, b->nm);
				d.bufferMemoryBarrierCount = b->nb;
				d.pBufferMemoryBarriers    = Take<vk::BufferMemoryBarrier2>(p, b->nb);
				d.imageMemoryBarrierCount  = b->ni;
				d.pImageMemoryBarriers     = Take<vk::ImageMemoryBarrier2>(p, b->ni);
				sink.pipelineBarrier2(d);
				break;
			}
			case Op::Barrier1: {
				const auto* b  = Take<B1>(p, 1);
				const auto* m  = Take<vk::MemoryBarrier>(p, b->nm);
				const auto* bb = Take<vk::BufferMemoryBarrier>(p, b->nb);
				const auto* ii = Take<vk::ImageMemoryBarrier>(p, b->ni);
				sink.pipelineBarrier(vk::PipelineStageFlags(b->src), vk::PipelineStageFlags(b->dst), vk::DependencyFlags(b->flags), b->nm, m, b->nb, bb,
				                     b->ni, ii);
				break;
			}
			case Op::PushConstants: {
				const auto* c = Take<PC>(p, 1);
				sink.pushConstants(vk::PipelineLayout(c->layout), vk::ShaderStageFlags(c->stages), c->offset, c->size, p);
				break;
			}
			case Op::PushDescriptorSet: {
				const auto* d = Take<PD>(p, 1);
				// The writes are rewritten in place (the arena is ours): each info pointer that was set points at its copy.
				auto* writes = const_cast<vk::WriteDescriptorSet*>(Take<vk::WriteDescriptorSet>(p, d->count));
#if !defined(KYTY_EMIT_MUTANT_SHALLOW)
				for (uint32_t k = 0; k < d->count; k++) {
					auto& x = writes[k];
					if (x.pImageInfo != nullptr) x.pImageInfo = Take<vk::DescriptorImageInfo>(p, x.descriptorCount);
					if (x.pBufferInfo != nullptr) x.pBufferInfo = Take<vk::DescriptorBufferInfo>(p, x.descriptorCount);
					if (x.pTexelBufferView != nullptr) x.pTexelBufferView = Take<vk::BufferView>(p, x.descriptorCount);
				}
#endif
				sink.pushDescriptorSetKHR(vk::PipelineBindPoint(d->bp), vk::PipelineLayout(d->layout), d->set, d->count, writes);
				break;
			}
			case Op::BindDescriptorSets: {
				const auto* d    = Take<BD>(p, 1);
				const auto* sets = Take<vk::DescriptorSet>(p, d->count);
				const auto* dyn  = Take<uint32_t>(p, d->ndyn);
				sink.bindDescriptorSets(vk::PipelineBindPoint(d->bp), vk::PipelineLayout(d->layout), d->first, d->count, sets, d->ndyn, dyn);
				break;
			}
			case Op::BindPipeline: {
				const auto* b = Take<BP>(p, 1);
				sink.bindPipeline(vk::PipelineBindPoint(b->bp), vk::Pipeline(b->pipeline));
				break;
			}
			case Op::BindVertexBuffers2: {
				const auto* v       = Take<VB>(p, 1);
				const auto* buffers = Take<vk::Buffer>(p, v->count);
				const auto* offsets = Take<vk::DeviceSize>(p, v->count);
				const auto* sizes   = v->has_sizes != 0 ? Take<vk::DeviceSize>(p, v->count) : nullptr;
				const auto* strides = v->has_strides != 0 ? Take<vk::DeviceSize>(p, v->count) : nullptr;
				sink.bindVertexBuffers2(v->first, v->count, buffers, offsets, sizes, strides);
				break;
			}
			case Op::BindIndexBuffer: {
				const auto* b = Take<IB>(p, 1);
				sink.bindIndexBuffer(vk::Buffer(b->buffer), b->offset, vk::IndexType(b->type));
				break;
			}
			case Op::SetViewportWithCount: {
				const auto* c = Take<CountHdr>(p, 1);
				sink.setViewportWithCount(c->count, Take<vk::Viewport>(p, c->count));
				break;
			}
			case Op::SetScissorWithCount: {
				const auto* c = Take<CountHdr>(p, 1);
				sink.setScissorWithCount(c->count, Take<vk::Rect2D>(p, c->count));
				break;
			}
			case Op::SetDepthTestEnable: sink.setDepthTestEnable(w[0]); break;
			case Op::SetDepthWriteEnable: sink.setDepthWriteEnable(w[0]); break;
			case Op::SetDepthCompareOp: sink.setDepthCompareOp(vk::CompareOp(w[0])); break;
			case Op::SetDepthBoundsTestEnable: sink.setDepthBoundsTestEnable(w[0]); break;
			case Op::SetDepthBounds: sink.setDepthBounds(F(w[0]), F(w[1])); break;
			case Op::SetDepthBiasEnable: sink.setDepthBiasEnable(w[0]); break;
			case Op::SetDepthBias: sink.setDepthBias(F(w[0]), F(w[1]), F(w[2])); break;
			case Op::SetStencilTestEnable: sink.setStencilTestEnable(w[0]); break;
			case Op::SetStencilOp:
				sink.setStencilOp(vk::StencilFaceFlags(w[0]), vk::StencilOp(w[1]), vk::StencilOp(w[2]), vk::StencilOp(w[3]), vk::CompareOp(w[4]));
				break;
			case Op::SetStencilReference: sink.setStencilReference(vk::StencilFaceFlags(w[0]), w[1]); break;
			case Op::SetStencilWriteMask: sink.setStencilWriteMask(vk::StencilFaceFlags(w[0]), w[1]); break;
			case Op::SetStencilCompareMask: sink.setStencilCompareMask(vk::StencilFaceFlags(w[0]), w[1]); break;
			case Op::SetCullMode: sink.setCullMode(vk::CullModeFlags(w[0])); break;
			case Op::SetFrontFace: sink.setFrontFace(vk::FrontFace(w[0])); break;
			case Op::SetBlendConstants: {
				const float c[4] = {F(w[0]), F(w[1]), F(w[2]), F(w[3])};
				sink.setBlendConstants(c);
				break;
			}
			case Op::SetLineWidth: sink.setLineWidth(F(w[0])); break;
			case Op::SetFeedbackLoop: sink.setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags(w[0])); break;
			case Op::Draw: sink.draw(w[0], w[1], w[2], w[3]); break;
			case Op::DrawIndexed: sink.drawIndexed(w[0], w[1], w[2], static_cast<int32_t>(w[3]), w[4]); break;
			case Op::DrawIndirect:
			case Op::DrawIndexedIndirect:
			case Op::DrawMeshTasksIndirect: {
				const auto* d = Take<ID>(p, 1);
				if (op == Op::DrawIndirect) sink.drawIndirect(vk::Buffer(d->buffer), d->offset, d->count, d->stride);
				else if (op == Op::DrawIndexedIndirect) sink.drawIndexedIndirect(vk::Buffer(d->buffer), d->offset, d->count, d->stride);
				else sink.drawMeshTasksIndirectEXT(vk::Buffer(d->buffer), d->offset, d->count, d->stride);
				break;
			}
			case Op::DrawMeshTasks: sink.drawMeshTasksEXT(w[0], w[1], w[2]); break;
			case Op::BeginRendering: {
				const auto*       b = Take<BR>(p, 1);
				vk::RenderingInfo info {};
				info.flags                = vk::RenderingFlags(b->flags);
				info.renderArea           = b->area;
				info.layerCount           = b->layers;
				info.viewMask             = b->view_mask;
				info.colorAttachmentCount = b->colors;
				info.pColorAttachments    = Take<vk::RenderingAttachmentInfo>(p, b->colors);
				info.pDepthAttachment     = b->has_depth != 0 ? Take<vk::RenderingAttachmentInfo>(p, 1) : nullptr;
				info.pStencilAttachment   = b->has_stencil != 0 ? Take<vk::RenderingAttachmentInfo>(p, 1) : nullptr;
				sink.beginRendering(info);
				break;
			}
			case Op::EndRendering: sink.endRendering(); break;
			case Op::Dispatch: sink.dispatch(w[0], w[1], w[2]); break;
			case Op::DispatchBase: sink.dispatchBase(w[0], w[1], w[2], w[3], w[4], w[5]); break;
			case Op::DispatchIndirect: {
				const auto* d = Take<DI>(p, 1);
				sink.dispatchIndirect(vk::Buffer(d->buffer), d->offset);
				break;
			}
			case Op::CopyBuffer: {
				const auto* c = Take<CB>(p, 1);
				sink.copyBuffer(vk::Buffer(c->src), vk::Buffer(c->dst), c->n, Take<vk::BufferCopy>(p, c->n));
				break;
			}
			case Op::CopyBufferToImage: {
				const auto* c = Take<BI>(p, 1);
				sink.copyBufferToImage(vk::Buffer(c->buffer), vk::Image(c->image), vk::ImageLayout(c->layout), c->n, Take<vk::BufferImageCopy>(p, c->n));
				break;
			}
			case Op::CopyImageToBuffer: {
				const auto* c = Take<BI>(p, 1);
				sink.copyImageToBuffer(vk::Image(c->image), vk::ImageLayout(c->layout), vk::Buffer(c->buffer), c->n, Take<vk::BufferImageCopy>(p, c->n));
				break;
			}
			case Op::CopyImage: {
				const auto* c = Take<CI>(p, 1);
				sink.copyImage(vk::Image(c->src), vk::ImageLayout(c->src_layout), vk::Image(c->dst), vk::ImageLayout(c->dst_layout), c->n,
				               Take<vk::ImageCopy>(p, c->n));
				break;
			}
			case Op::FillBuffer: {
				const auto* f = Take<FB>(p, 1);
				sink.fillBuffer(vk::Buffer(f->buffer), f->offset, f->size, f->data);
				break;
			}
			case Op::ClearColorImage: {
				const auto* c = Take<CL>(p, 1);
				vk::ClearColorValue value {};
				std::memcpy(&value, c->value, sizeof(value));
				sink.clearColorImage(vk::Image(c->image), vk::ImageLayout(c->layout), c->has != 0 ? &value : nullptr, c->n,
				                     Take<vk::ImageSubresourceRange>(p, c->n));
				break;
			}
			case Op::ClearDepthStencilImage: {
				const auto* c = Take<CL>(p, 1);
				vk::ClearDepthStencilValue value {};
				std::memcpy(&value, c->value, sizeof(value));
				sink.clearDepthStencilImage(vk::Image(c->image), vk::ImageLayout(c->layout), c->has != 0 ? &value : nullptr, c->n,
				                            Take<vk::ImageSubresourceRange>(p, c->n));
				break;
			}
			case Op::Count: break;
		}
	}

	Mode     m_mode;
	uint32_t m_ops         = 0;
	uint32_t m_sample_left = kSamplePeriod;
	Arena    m_arena;
	HashSink m_direct;
	Stats    m_stats;
	void*    m_owner = nullptr;
	void (*m_drain)(void*, Reason) = nullptr;
	// S565 thread mode
	Emitter*       m_emitter       = nullptr;
	uint64_t       m_raw_bits      = 0;
	const uint8_t* m_pub           = nullptr; // first byte not yet published (current chunk)
	uint32_t       m_publish_bytes = 2048;
};

// ---------------------------------------------------------------------------------------------------------------------------------------
// The handle the IR sites record through: the vk::CommandBuffer method names, routed by the stream's mode. Converting it to the raw handle
// drains the stream first (Reason::Conv), so a helper that still takes vk::CommandBuffer records in order.

// S565: `ir` = false routes every command straight to `raw` whatever the mode (EmitIfPending with an empty stream): the handle is then a raw handle.
template <class Raw>
class CmdT {
public:
	CmdT(Stream* stream, Raw raw, bool ir = true) noexcept: m_s(stream), m_raw(raw), m_ir(ir && stream->Current() != Mode::Off) {}
	[[nodiscard]] bool ToIr() const noexcept { return m_ir; }

	// NOLINTNEXTLINE(google-explicit-constructor)
	operator Raw() const {
		m_s->Drain(m_raw, Reason::Conv);
		return m_raw;
	}
	[[nodiscard]] Stream& GetStream() const noexcept { return *m_s; }

	void pipelineBarrier2(const vk::DependencyInfo& d) const {
		if (Off()) return m_raw.pipelineBarrier2(d);
		Struct(m_s->Barrier2(d), [&] { m_raw.pipelineBarrier2(d); });
	}
	void pipelineBarrier(vk::PipelineStageFlags src, vk::PipelineStageFlags dst, vk::DependencyFlags flags, uint32_t nm, const vk::MemoryBarrier* m,
	                     uint32_t nb, const vk::BufferMemoryBarrier* b, uint32_t ni, const vk::ImageMemoryBarrier* i) const {
		if (Off()) return m_raw.pipelineBarrier(src, dst, flags, nm, m, nb, b, ni, i);
		Struct(m_s->Barrier1(src, dst, flags, nm, m, nb, b, ni, i), [&] { m_raw.pipelineBarrier(src, dst, flags, nm, m, nb, b, ni, i); });
	}
	void pushConstants(vk::PipelineLayout layout, vk::ShaderStageFlags stages, uint32_t offset, uint32_t size, const void* data) const {
		if (Off()) return m_raw.pushConstants(layout, stages, offset, size, data);
		m_s->PushConstants(layout, stages, offset, size, data);
		if (Shadow()) m_raw.pushConstants(layout, stages, offset, size, data);
	}
	void pushDescriptorSetKHR(vk::PipelineBindPoint bp, vk::PipelineLayout layout, uint32_t set, uint32_t count, const vk::WriteDescriptorSet* w) const {
		if (Off()) return m_raw.pushDescriptorSetKHR(bp, layout, set, count, w);
		Struct(m_s->PushDescriptorSet(bp, layout, set, count, w), [&] { m_raw.pushDescriptorSetKHR(bp, layout, set, count, w); });
	}
	void bindDescriptorSets(vk::PipelineBindPoint bp, vk::PipelineLayout layout, uint32_t first, uint32_t count, const vk::DescriptorSet* sets,
	                        uint32_t ndyn, const uint32_t* dyn) const {
		if (Off()) return m_raw.bindDescriptorSets(bp, layout, first, count, sets, ndyn, dyn);
		m_s->BindDescriptorSets(bp, layout, first, count, sets, ndyn, dyn);
		if (Shadow()) m_raw.bindDescriptorSets(bp, layout, first, count, sets, ndyn, dyn);
	}
	void bindPipeline(vk::PipelineBindPoint bp, vk::Pipeline pipeline) const {
		if (Off()) return m_raw.bindPipeline(bp, pipeline);
		m_s->BindPipeline(bp, pipeline);
		if (Shadow()) m_raw.bindPipeline(bp, pipeline);
	}
	void bindVertexBuffers2(uint32_t first, uint32_t count, const vk::Buffer* buffers, const vk::DeviceSize* offsets, const vk::DeviceSize* sizes,
	                        const vk::DeviceSize* strides) const {
		if (Off()) return m_raw.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
		m_s->BindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
		if (Shadow()) m_raw.bindVertexBuffers2(first, count, buffers, offsets, sizes, strides);
	}
	void bindIndexBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::IndexType type) const {
		if (Off()) return m_raw.bindIndexBuffer(buffer, offset, type);
		m_s->BindIndexBuffer(buffer, offset, type);
		if (Shadow()) m_raw.bindIndexBuffer(buffer, offset, type);
	}
	void setViewportWithCount(uint32_t count, const vk::Viewport* v) const {
		if (Off()) return m_raw.setViewportWithCount(count, v);
		m_s->SetViewports(count, v);
		if (Shadow()) m_raw.setViewportWithCount(count, v);
	}
	void setScissorWithCount(uint32_t count, const vk::Rect2D* r) const {
		if (Off()) return m_raw.setScissorWithCount(count, r);
		m_s->SetScissors(count, r);
		if (Shadow()) m_raw.setScissorWithCount(count, r);
	}
	void setDepthTestEnable(vk::Bool32 v) const {
		if (Off()) return m_raw.setDepthTestEnable(v);
		W(Op::SetDepthTestEnable, std::array<uint32_t, 1> {v});
		if (Shadow()) Both([&](auto& s) { s.setDepthTestEnable(v); });
	}
	void setDepthWriteEnable(vk::Bool32 v) const {
		if (Off()) return m_raw.setDepthWriteEnable(v);
		W(Op::SetDepthWriteEnable, std::array<uint32_t, 1> {v});
		if (Shadow()) Both([&](auto& s) { s.setDepthWriteEnable(v); });
	}
	void setDepthCompareOp(vk::CompareOp v) const {
		if (Off()) return m_raw.setDepthCompareOp(v);
		W(Op::SetDepthCompareOp, std::array<uint32_t, 1> {static_cast<uint32_t>(v)});
		if (Shadow()) Both([&](auto& s) { s.setDepthCompareOp(v); });
	}
	void setDepthBoundsTestEnable(vk::Bool32 v) const {
		if (Off()) return m_raw.setDepthBoundsTestEnable(v);
		W(Op::SetDepthBoundsTestEnable, std::array<uint32_t, 1> {v});
		if (Shadow()) Both([&](auto& s) { s.setDepthBoundsTestEnable(v); });
	}
	void setDepthBounds(float a, float b) const {
		if (Off()) return m_raw.setDepthBounds(a, b);
		W(Op::SetDepthBounds, std::array<uint32_t, 2> {Bits(a), Bits(b)});
		if (Shadow()) Both([&](auto& s) { s.setDepthBounds(a, b); });
	}
	void setDepthBiasEnable(vk::Bool32 v) const {
		if (Off()) return m_raw.setDepthBiasEnable(v);
		W(Op::SetDepthBiasEnable, std::array<uint32_t, 1> {v});
		if (Shadow()) Both([&](auto& s) { s.setDepthBiasEnable(v); });
	}
	void setDepthBias(float a, float b, float c) const {
		if (Off()) return m_raw.setDepthBias(a, b, c);
		W(Op::SetDepthBias, std::array<uint32_t, 3> {Bits(a), Bits(b), Bits(c)});
		if (Shadow()) Both([&](auto& s) { s.setDepthBias(a, b, c); });
	}
	void setStencilTestEnable(vk::Bool32 v) const {
		if (Off()) return m_raw.setStencilTestEnable(v);
		W(Op::SetStencilTestEnable, std::array<uint32_t, 1> {v});
		if (Shadow()) Both([&](auto& s) { s.setStencilTestEnable(v); });
	}
	void setStencilOp(vk::StencilFaceFlags f, vk::StencilOp a, vk::StencilOp b, vk::StencilOp c, vk::CompareOp d) const {
		if (Off()) return m_raw.setStencilOp(f, a, b, c, d);
		W(Op::SetStencilOp, std::array<uint32_t, 5> {static_cast<uint32_t>(f), static_cast<uint32_t>(a), static_cast<uint32_t>(b),
		                                             static_cast<uint32_t>(c), static_cast<uint32_t>(d)});
		if (Shadow()) Both([&](auto& s) { s.setStencilOp(f, a, b, c, d); });
	}
	void setStencilReference(vk::StencilFaceFlags f, uint32_t v) const {
		if (Off()) return m_raw.setStencilReference(f, v);
		W(Op::SetStencilReference, std::array<uint32_t, 2> {static_cast<uint32_t>(f), v});
		if (Shadow()) Both([&](auto& s) { s.setStencilReference(f, v); });
	}
	void setStencilWriteMask(vk::StencilFaceFlags f, uint32_t v) const {
		if (Off()) return m_raw.setStencilWriteMask(f, v);
		W(Op::SetStencilWriteMask, std::array<uint32_t, 2> {static_cast<uint32_t>(f), v});
		if (Shadow()) Both([&](auto& s) { s.setStencilWriteMask(f, v); });
	}
	void setStencilCompareMask(vk::StencilFaceFlags f, uint32_t v) const {
		if (Off()) return m_raw.setStencilCompareMask(f, v);
		W(Op::SetStencilCompareMask, std::array<uint32_t, 2> {static_cast<uint32_t>(f), v});
		if (Shadow()) Both([&](auto& s) { s.setStencilCompareMask(f, v); });
	}
	void setCullMode(vk::CullModeFlags v) const {
		if (Off()) return m_raw.setCullMode(v);
		W(Op::SetCullMode, std::array<uint32_t, 1> {static_cast<uint32_t>(v)});
		if (Shadow()) Both([&](auto& s) { s.setCullMode(v); });
	}
	void setFrontFace(vk::FrontFace v) const {
		if (Off()) return m_raw.setFrontFace(v);
		W(Op::SetFrontFace, std::array<uint32_t, 1> {static_cast<uint32_t>(v)});
		if (Shadow()) Both([&](auto& s) { s.setFrontFace(v); });
	}
	void setBlendConstants(const float c[4]) const {
		if (Off()) return m_raw.setBlendConstants(c);
		W(Op::SetBlendConstants, std::array<uint32_t, 4> {Bits(c[0]), Bits(c[1]), Bits(c[2]), Bits(c[3])});
		if (Shadow()) Both([&](auto& s) { s.setBlendConstants(c); });
	}
	void setLineWidth(float v) const {
		if (Off()) return m_raw.setLineWidth(v);
		W(Op::SetLineWidth, std::array<uint32_t, 1> {Bits(v)});
		if (Shadow()) Both([&](auto& s) { s.setLineWidth(v); });
	}
	void setAttachmentFeedbackLoopEnableEXT(vk::ImageAspectFlags v) const {
		if (Off()) return m_raw.setAttachmentFeedbackLoopEnableEXT(v);
		W(Op::SetFeedbackLoop, std::array<uint32_t, 1> {static_cast<uint32_t>(v)});
		if (Shadow()) Both([&](auto& s) { s.setAttachmentFeedbackLoopEnableEXT(v); });
	}
	void draw(uint32_t a, uint32_t b, uint32_t c, uint32_t d) const {
		if (Off()) return m_raw.draw(a, b, c, d);
		W(Op::Draw, std::array<uint32_t, 4> {a, b, c, d});
		if (Shadow()) Both([&](auto& s) { s.draw(a, b, c, d); });
	}
	void drawIndexed(uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e) const {
		if (Off()) return m_raw.drawIndexed(a, b, c, d, e);
		W(Op::DrawIndexed, std::array<uint32_t, 5> {a, b, c, static_cast<uint32_t>(d), e});
		if (Shadow()) Both([&](auto& s) { s.drawIndexed(a, b, c, d, e); });
	}
	void drawIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const {
		if (Off()) return m_raw.drawIndirect(buffer, offset, count, stride);
		m_s->Indirect(Op::DrawIndirect, buffer, offset, count, stride);
		if (Shadow()) Both([&](auto& s) { s.drawIndirect(buffer, offset, count, stride); });
	}
	void drawIndexedIndirect(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const {
		if (Off()) return m_raw.drawIndexedIndirect(buffer, offset, count, stride);
		m_s->Indirect(Op::DrawIndexedIndirect, buffer, offset, count, stride);
		if (Shadow()) Both([&](auto& s) { s.drawIndexedIndirect(buffer, offset, count, stride); });
	}
	void drawMeshTasksEXT(uint32_t x, uint32_t y, uint32_t z) const {
		if (Off()) return m_raw.drawMeshTasksEXT(x, y, z);
		W(Op::DrawMeshTasks, std::array<uint32_t, 3> {x, y, z});
		if (Shadow()) Both([&](auto& s) { s.drawMeshTasksEXT(x, y, z); });
	}
	void drawMeshTasksIndirectEXT(vk::Buffer buffer, vk::DeviceSize offset, uint32_t count, uint32_t stride) const {
		if (Off()) return m_raw.drawMeshTasksIndirectEXT(buffer, offset, count, stride);
		m_s->Indirect(Op::DrawMeshTasksIndirect, buffer, offset, count, stride);
		if (Shadow()) Both([&](auto& s) { s.drawMeshTasksIndirectEXT(buffer, offset, count, stride); });
	}
	void beginRendering(const vk::RenderingInfo& info) const {
		if (Off()) return m_raw.beginRendering(info);
		Struct(m_s->BeginRendering(info), [&] { m_raw.beginRendering(info); });
	}
	void endRendering() const {
		if (Off()) return m_raw.endRendering();
		m_s->EndRendering();
		if (Shadow()) m_raw.endRendering();
	}
	// S565
	void dispatch(uint32_t x, uint32_t y, uint32_t z) const {
		if (Off()) return m_raw.dispatch(x, y, z);
		W(Op::Dispatch, std::array<uint32_t, 3> {x, y, z});
		if (Shadow()) Both([&](auto& s) { s.dispatch(x, y, z); });
	}
	void dispatchBase(uint32_t bx, uint32_t by, uint32_t bz, uint32_t x, uint32_t y, uint32_t z) const {
		if (Off()) return m_raw.dispatchBase(bx, by, bz, x, y, z);
		W(Op::DispatchBase, std::array<uint32_t, 6> {bx, by, bz, x, y, z});
		if (Shadow()) Both([&](auto& s) { s.dispatchBase(bx, by, bz, x, y, z); });
	}
	void dispatchIndirect(vk::Buffer buffer, vk::DeviceSize offset) const {
		if (Off()) return m_raw.dispatchIndirect(buffer, offset);
		m_s->DispatchIndirect(buffer, offset);
		if (Shadow()) m_raw.dispatchIndirect(buffer, offset);
	}
	void copyBuffer(vk::Buffer src, vk::Buffer dst, uint32_t n, const vk::BufferCopy* regions) const {
		if (Off()) return m_raw.copyBuffer(src, dst, n, regions);
		m_s->CopyBuffer(src, dst, n, regions);
		if (Shadow()) m_raw.copyBuffer(src, dst, n, regions);
	}
	void copyBufferToImage(vk::Buffer src, vk::Image dst, vk::ImageLayout layout, uint32_t n, const vk::BufferImageCopy* regions) const {
		if (Off()) return m_raw.copyBufferToImage(src, dst, layout, n, regions);
		m_s->CopyBufferToImage(src, dst, layout, n, regions);
		if (Shadow()) m_raw.copyBufferToImage(src, dst, layout, n, regions);
	}
	void copyImageToBuffer(vk::Image src, vk::ImageLayout layout, vk::Buffer dst, uint32_t n, const vk::BufferImageCopy* regions) const {
		if (Off()) return m_raw.copyImageToBuffer(src, layout, dst, n, regions);
		m_s->CopyImageToBuffer(src, layout, dst, n, regions);
		if (Shadow()) m_raw.copyImageToBuffer(src, layout, dst, n, regions);
	}
	void copyImage(vk::Image src, vk::ImageLayout src_layout, vk::Image dst, vk::ImageLayout dst_layout, uint32_t n, const vk::ImageCopy* regions) const {
		if (Off()) return m_raw.copyImage(src, src_layout, dst, dst_layout, n, regions);
		m_s->CopyImage(src, src_layout, dst, dst_layout, n, regions);
		if (Shadow()) m_raw.copyImage(src, src_layout, dst, dst_layout, n, regions);
	}
	void fillBuffer(vk::Buffer buffer, vk::DeviceSize offset, vk::DeviceSize size, uint32_t data) const {
		if (Off()) return m_raw.fillBuffer(buffer, offset, size, data);
		m_s->FillBuffer(buffer, offset, size, data);
		if (Shadow()) m_raw.fillBuffer(buffer, offset, size, data);
	}
	void clearColorImage(vk::Image image, vk::ImageLayout layout, const vk::ClearColorValue* color, uint32_t n, const vk::ImageSubresourceRange* ranges) const {
		if (Off()) return m_raw.clearColorImage(image, layout, color, n, ranges);
		m_s->ClearColorImage(image, layout, color, n, ranges);
		if (Shadow()) m_raw.clearColorImage(image, layout, color, n, ranges);
	}
	void clearDepthStencilImage(vk::Image image, vk::ImageLayout layout, const vk::ClearDepthStencilValue* value, uint32_t n,
	                            const vk::ImageSubresourceRange* ranges) const {
		if (Off()) return m_raw.clearDepthStencilImage(image, layout, value, n, ranges);
		m_s->ClearDepthStencilImage(image, layout, value, n, ranges);
		if (Shadow()) m_raw.clearDepthStencilImage(image, layout, value, n, ranges);
	}

private:
	[[nodiscard]] bool Off() const noexcept { return !m_ir; }
	[[nodiscard]] bool Shadow() const noexcept { return m_ir && m_s->Current() == Mode::Shadow; }
	static uint32_t    Bits(float f) {
        uint32_t u = 0;
        std::memcpy(&u, &f, 4);
        return u;
	}
	template <size_t N>
	void W(Op op, const std::array<uint32_t, N>& words) const {
		m_s->Words(op, words);
	}
	// Shadow: the direct digest and the real call.
	template <class F>
	void Both(F&& f) const {
		f(m_s->Direct());
		f(m_raw);
	}
	// A struct command: recorded (and in shadow also called directly), or, when the IR cannot hold it, drained before and called directly.
	template <class F>
	void Struct(bool recorded, F&& direct) const {
		if (!recorded) m_s->Drain(m_raw, Reason::Other);
		if (!recorded || Shadow()) direct();
	}

	Stream* m_s;
	Raw     m_raw;
	bool    m_ir;
};

using Cmd = CmdT<vk::CommandBuffer>;

// A site that records directly (CommandBuffer::Handle): the stream is replayed first. Mutant NO_DRAIN skips that.
template <class Raw>
inline Raw TakeRaw(Stream& stream, Raw raw, Reason why = Reason::Handle) {
#if !defined(KYTY_EMIT_MUTANT_NO_DRAIN)
	if (stream.Pending()) [[unlikely]] {
		stream.Drain(raw, why);
	}
#endif
	return raw;
}

// Begin/EndRendering: to the IR only while the stream already holds commands (see the top of the file). Mutant UNCOND_PASS: always.
inline bool PassToIr(const Stream& stream) {
#if defined(KYTY_EMIT_MUTANT_UNCOND_PASS)
	return stream.Current() != Mode::Off;
#else
	return stream.Pending();
#endif
}

} // namespace Libs::Graphics::CpEmit

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CPEMIT_H_
