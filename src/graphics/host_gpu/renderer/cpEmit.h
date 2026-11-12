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
// Header-only (tests/CpEmitTests.cpp). Mutants (must fail the test, WILL_FAIL): NO_DRAIN (TakeRaw does not drain), SHALLOW (push-descriptor
// infos keep the caller's pointers), UNCOND_PASS (Begin/EndRendering always go to the IR).

#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
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

enum class Mode : uint8_t { Off = 0, Sync = 1, Shadow = 2 };

inline Mode ParseMode(const char* value) {
	if (value == nullptr) return Mode::Off;
	if (std::strcmp(value, "sync") == 0) return Mode::Sync;
	if (std::strcmp(value, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline const char* ModeName(Mode mode) {
	return mode == Mode::Sync ? "sync" : mode == Mode::Shadow ? "shadow" : "0";
}

inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_CP_EMIT"));
	return mode;
}

enum class Reason : uint8_t { Handle, Conv, Submit, Wait, Other, Count };

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
		              std::is_same_v<T, uint32_t>) {
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
		} else {
			Replay(raw);
		}
		m_stats.replay_ticks += Ticks() - t0;
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
		Writer(Stream& s, Op op, size_t payload) {
			const size_t size = sizeof(Hdr) + Align8(payload);
			p                 = s.m_arena.Alloc(size);
			auto* h           = reinterpret_cast<Hdr*>(p);
			h->op             = op;
			h->pad            = 0;
			h->size           = static_cast<uint32_t>(size);
			p += sizeof(Hdr);
			if (s.m_ops++ == 0) s.NotePending();
		}
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
		uint8_t* p;
	};

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
};

// ---------------------------------------------------------------------------------------------------------------------------------------
// The handle the IR sites record through: the vk::CommandBuffer method names, routed by the stream's mode. Converting it to the raw handle
// drains the stream first (Reason::Conv), so a helper that still takes vk::CommandBuffer records in order.

template <class Raw>
class CmdT {
public:
	CmdT(Stream* stream, Raw raw) noexcept: m_s(stream), m_raw(raw) {}

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

private:
	[[nodiscard]] bool Off() const noexcept { return m_s->Current() == Mode::Off; }
	[[nodiscard]] bool Shadow() const noexcept { return m_s->Current() == Mode::Shadow; }
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
