#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PERVERTEXVSSTORE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PERVERTEXVSSTORE_H_

// S149: KYTY_PV_VSSTORE=1 (default OFF). Instead of the capture/replay prototype, the guest vertex
// shader writes its PerVertex parameters (uint4, bits unchanged) to an SSBO record
// cap[record(instance, vertex)] and the pixel shader reads the three corners of its primitive back
// from there: cap[record(instance, index[first_index + primitive * 3 + k] + vertex_offset)].
// Pure helpers (no Vulkan, no globals) shared by the renderer, the SPIR-V emitter, the CPU tests and
// the GPU harness: the plan for a draw, the record numbering both shader sides must agree on, and
// the per-pass region ring. The emitter mirrors the arithmetic below in SPIR-V.

#include <cstdint>
#include <map>
#include <span>

namespace Libs::Graphics {

// Guest parameters 0..30; location 31 carries the instance index from the VS to the PS (flat).
constexpr uint32_t kPvVsStoreInstanceLocation = 31;
constexpr uint32_t kPvVsStoreCorners          = 3;

// Dword layout of the push block that follows the native push data in both stages.
enum PvVsStorePush : uint32_t {
	kPvPushBase          = 0, // record of (instance 0, vertex `first`) in the pass ring
	kPvPushSpan          = 1, // records per instance
	kPvPushFirst         = 2, // first vertex index (index + vertex_offset) of the draw's record range
	kPvPushFirstInstance = 3,
	kPvPushFirstIndex    = 4, // element offset into the index buffer
	kPvPushVertexOffset  = 5, // signed
	kPvPushIndex32       = 6, // 1: 32-bit indices, 0: 16-bit
	kPvPushReserved      = 7,
};
constexpr uint32_t kPvVsStorePushDwords = 8;

// Which index element corner k of a triangle reads (the PerVertexKHR order of the guest PS).
constexpr uint32_t PvVsStoreCornerElement(uint32_t k) {
#if defined(KYTY_PVVS_MUTANT_FIRST_THIRD)
	return k == 0 ? 2u : (k == 2 ? 0u : k);
#else
	return k;
#endif
}

// Record written by the VS invocation (instance, vertex_index); vertex_index is what the host
// shader sees as gl_VertexIndex (index + vertex_offset for indexed draws).
constexpr uint32_t PvVsStoreVertexRecord(uint32_t base, uint32_t span, uint32_t first,
                                         uint32_t first_instance, uint32_t instance,
                                         uint32_t vertex_index) {
#if defined(KYTY_PVVS_MUTANT_NO_INSTANCE)
	(void)first_instance;
	(void)instance;
	(void)span;
	return base + (vertex_index - first);
#elif defined(KYTY_PVVS_MUTANT_NO_FIRST)
	return base + (instance - first_instance) * span + vertex_index;
#else
	return base + (instance - first_instance) * span + (vertex_index - first);
#endif
}

// Index element `element` of the bound index buffer (little endian dwords).
constexpr uint32_t PvVsStoreFetchIndex(std::span<const uint32_t> words, bool index32,
                                       uint32_t element) {
	if (index32) return element < words.size() ? words[element] : 0u;
	const uint32_t word = (element >> 1u) < words.size() ? words[element >> 1u] : 0u;
#if defined(KYTY_PVVS_MUTANT_HALF_SWAP)
	return (word >> (((element & 1u) ^ 1u) * 16u)) & 0xffffu;
#else
	return (word >> ((element & 1u) * 16u)) & 0xffffu;
#endif
}

// Draw parameters both sides use (the push block as a struct).
struct PvVsStoreDraw {
	uint32_t base           = 0;
	uint32_t span           = 0;
	uint32_t first          = 0;
	uint32_t first_instance = 0;
	uint32_t first_index    = 0;
	int32_t  vertex_offset  = 0;
	bool     index32        = true;
};

// Record the PS reads for corner k of `primitive` in `instance`.
constexpr uint32_t PvVsStoreCornerRecord(const PvVsStoreDraw& draw,
                                         std::span<const uint32_t> index_words, uint32_t instance,
                                         uint32_t primitive, uint32_t k) {
	const uint32_t element = draw.first_index + primitive * kPvVsStoreCorners +
	                         PvVsStoreCornerElement(k);
#if defined(KYTY_PVVS_MUTANT_NO_FIRST_INDEX)
	const uint32_t index = PvVsStoreFetchIndex(index_words, draw.index32, element - draw.first_index);
#else
	const uint32_t index = PvVsStoreFetchIndex(index_words, draw.index32, element);
#endif
#if defined(KYTY_PVVS_MUTANT_NO_VERTEX_OFFSET)
	const uint32_t vertex_index = index;
#else
	const uint32_t vertex_index = index + static_cast<uint32_t>(draw.vertex_offset);
#endif
	return PvVsStoreVertexRecord(draw.base, draw.span, draw.first, draw.first_instance,
	                             draw.first_instance + instance, vertex_index);
}

enum class PvVsStoreVerdict : uint32_t {
	Disabled,         // switch off: nothing counted, the prototype as before
	Use,              // the draw goes through the VS store
	NotTriangleList,  // prototype unchanged (RectList and the rest)
	NotIndexed,       // prototype unchanged
	PrimitiveRestart, // prototype unchanged
	BadIndexSize,     // index buffer is neither 16 nor 32 bit
	NoRange,          // the vertex range of the draw is not known
	Dropped,          // a corner points outside the vertex buffer: the prototype renders vertex 0
	Empty,            // nothing to draw
	TooLarge,         // record count overflows the 32-bit math of the shaders
	RingFull,         // the pass ring has no room for a new region
};

struct PvVsStoreRequest {
	bool     enabled           = false; // KYTY_PV_VSSTORE=1
	bool     triangle_list     = false;
	bool     indexed           = false;
	bool     primitive_restart = false;
	uint32_t index_size        = 0; // bytes
	bool     range_known       = false;
	bool     range_dropped     = false;
	uint64_t range_first       = 0;
	uint64_t range_end         = 0; // [first, end) vertex indices the draw reads
	uint32_t index_count       = 0;
	uint32_t instance_count    = 0;
	uint32_t record_stride_vec4 = 0;
};

struct PvVsStorePlan {
	PvVsStoreVerdict verdict = PvVsStoreVerdict::Disabled;
	uint32_t         first   = 0; // first vertex of the range
	uint32_t         span    = 0; // records per instance
	uint64_t         records = 0; // span * instances
	[[nodiscard]] bool Use() const { return verdict == PvVsStoreVerdict::Use; }
};

constexpr PvVsStorePlan PlanPvVsStore(const PvVsStoreRequest& request) {
	using V = PvVsStoreVerdict;
	PvVsStorePlan plan;
	if (!request.enabled) return plan;
	const auto reject = [&](V verdict) {
		plan.verdict = verdict;
		return plan;
	};
	if (!request.triangle_list) return reject(V::NotTriangleList);
	if (!request.indexed) return reject(V::NotIndexed);
	if (request.primitive_restart) return reject(V::PrimitiveRestart);
	if (request.index_size != 2 && request.index_size != 4) return reject(V::BadIndexSize);
	if (!request.range_known) return reject(V::NoRange);
#if !defined(KYTY_PVVS_MUTANT_DROPPED_OK)
	if (request.range_dropped) return reject(V::Dropped);
#endif
	if (request.index_count < kPvVsStoreCorners || request.instance_count == 0 ||
	    request.range_end <= request.range_first)
		return reject(V::Empty);
	const uint64_t span    = request.range_end - request.range_first;
	const uint64_t records = span * request.instance_count;
	// Record * stride * 4 dwords must stay below 2^32 in the shaders.
	if (request.range_end > 0xffffffffull || records * request.record_stride_vec4 * 4ull > 0xffffffffull)
		return reject(V::TooLarge);
	plan.verdict = V::Use;
	plan.first   = static_cast<uint32_t>(request.range_first);
	plan.span    = static_cast<uint32_t>(span);
	plan.records = records;
	return plan;
}

// Regions of one render pass: every draw gets its own, none is reused until the pass ends (the
// probe of RESULT-262 showed a shared region is overwritten by the next draw).
class PvVsStoreRing {
public:
	explicit PvVsStoreRing(uint64_t capacity_records = 0) : m_capacity(capacity_records) {}
	void SetCapacity(uint64_t capacity_records) { m_capacity = capacity_records; }
	// Returns false when the pass ring is full (the caller falls back to the prototype).
	bool Allocate(uint64_t records, uint32_t& base) {
		if (records == 0 || records > m_capacity - m_used) return false;
		base = static_cast<uint32_t>(m_used);
#if !defined(KYTY_PVVS_MUTANT_REUSE_REGION)
		m_used += records;
#endif
		m_draws++;
		return true;
	}
	void                 EndPass() { m_used = 0; }
	[[nodiscard]] uint64_t Used() const { return m_used; }
	[[nodiscard]] uint64_t Draws() const { return m_draws; }

private:
	uint64_t m_capacity = 0;
	uint64_t m_used     = 0;
	uint64_t m_draws    = 0;
};

// Slot of every VS Parameter output in a record (sorted by location, dense from 0).
inline std::map<uint32_t, uint32_t> PvVsStoreSlots(std::span<const uint32_t> parameter_locations) {
	std::map<uint32_t, uint32_t> sorted;
	for (const auto location: parameter_locations) sorted.emplace(location, 0);
	uint32_t slot = 0;
	for (auto& [location, value]: sorted) value = slot++;
	return sorted;
}

} // namespace Libs::Graphics

#endif
