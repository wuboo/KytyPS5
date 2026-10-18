#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PERVERTEXVSSTORE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PERVERTEXVSSTORE_H_

// S149: KYTY_PV_VSSTORE=1 (default OFF). Instead of the capture/replay prototype, the guest vertex
// shader writes its PerVertex parameters (uint4, bits unchanged) to an SSBO record
// cap[record(instance, vertex)] and the pixel shader reads the three corners of its primitive back
// from there: cap[record(instance, index[first_index + primitive * 3 + k] + vertex_offset)].
// Pure helpers (no Vulkan, no globals) shared by the renderer, the SPIR-V emitter, the CPU tests and
// the GPU harness: the plan for a draw, the record numbering both shader sides must agree on, and
// the per-pass region ring. The emitter mirrors the arithmetic below in SPIR-V.

#include <array>
#include <bit>
#include <cstdint>
#include <map>
#include <span>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

// Guest parameters 0..30; location 31 carries the instance index from the VS to the PS (flat).
constexpr uint32_t kPvVsStoreInstanceLocation = 31;
constexpr uint32_t kPvVsStoreCorners          = 3;

// S389: the guard's counter is read back by the CPU (Buffer::Invalidate needs MemoryUsage::Download); a Stream buffer
// made the first read abort in streamBuffer.cpp (RESULT-389).
enum class PvGuardMemory : uint32_t { Stream, Download };
#if defined(KYTY_PVVS_MUTANT_GUARD_STREAM)
constexpr PvGuardMemory kPvGuardMemory = PvGuardMemory::Stream;
#else
constexpr PvGuardMemory kPvGuardMemory = PvGuardMemory::Download;
#endif

// Dword layout of the push block that follows the native push data in both stages.
enum PvVsStorePush : uint32_t {
	kPvPushBase          = 0, // record of (instance 0, vertex `first`) in the pass ring
	kPvPushSpan          = 1, // records per instance
	kPvPushFirst         = 2, // first vertex index (index + vertex_offset) of the draw's record range
	kPvPushFirstInstance = 3,
	kPvPushFirstIndex    = 4, // element offset into the index buffer
	kPvPushVertexOffset  = 5, // signed
	kPvPushIndex32       = 6, // 1: 32-bit indices, 0: 16-bit
	kPvPushStride        = 7, // vec4 slots per record (S150: slot == host parameter location, stride from the VS)
};
constexpr uint32_t kPvVsStorePushDwords = 8;
// S388 (KYTY_PV_VSSTORE_GUARD=1): one more dword after the block, the record count of the draw (span * instances);
// the VS skips records at or past it and the PS paints a read past it magenta. Default 0: the block has 8 dwords.
constexpr uint32_t kPvPushRecords = 8;
constexpr uint32_t PvVsStorePushDwords(bool guard) {
	return kPvVsStorePushDwords + (guard ? 1u : 0u);
}
// Parameter locations of the guard counter: set 1 binding 2, 16 bytes of which the first dword counts.
constexpr uint32_t kPvGuardBinding = 2;
// `rel` = record - base. Out of the draw's record range (wraps for a vertex below `first`).
constexpr bool PvVsStoreRecordOutOfRange(uint32_t record, uint32_t base, uint32_t records) {
#if defined(KYTY_PVVS_MUTANT_OOB_OFF_BY_ONE)
	return record - base > records;
#elif defined(KYTY_PVVS_MUTANT_OOB_SIGNED)
	return static_cast<int32_t>(record - base) >= static_cast<int32_t>(records);
#else
	return record - base >= records;
#endif
}
// Parser of KYTY_PV_VSSTORE_GUARD (0|1, unset = 0).
inline bool ParsePvVsStoreGuard(const char* text, bool& guard) {
	guard = false;
	if (text == nullptr) return true;
	const std::string_view value {text};
	if (value == "0") return true;
	if (value == "1") {
		guard = true;
		return true;
	}
	return false;
}

// Parser of KYTY_PV_VSSTORE_COMPACT (0|1, unset = 0; the grammar of the guard).
inline bool ParsePvVsStoreCompact(const char* text, bool& compact) {
	return ParsePvVsStoreGuard(text, compact);
}

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
	uint32_t records        = 0; // span * instances (the guard's dword, pushed apart from the block)
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
	InstancedOff,     // S388: instance_count > 1 while KYTY_PV_VSSTORE_INSTANCED=0
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
	bool     instanced_allowed  = true; // S388: KYTY_PV_VSSTORE_INSTANCED (default 1)
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
#if defined(KYTY_PVVS_MUTANT_INSTANCED_OFF_IGNORED)
	// the switch is not consulted
#elif defined(KYTY_PVVS_MUTANT_INSTANCED_OFF_ALL)
	if (!request.instanced_allowed) return reject(V::InstancedOff);
#else
	if (request.instance_count > 1 && !request.instanced_allowed) return reject(V::InstancedOff);
#endif
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
	[[nodiscard]] uint64_t Free() const { return m_capacity - m_used; }
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

// S150: the two shader variants are emitted at compile time, independently (the PS does not know its
// VS), so a Parameter output of host location L lives in slot L of a record and the record stride
// (push dword kPvPushStride) is the highest location the VS writes plus one.
constexpr uint32_t PvVsStoreStrideFor(uint32_t vs_location_mask) {
	return vs_location_mask == 0 ? 0u : 32u - static_cast<uint32_t>(std::countl_zero(vs_location_mask));
}

// S409 (KYTY_PV_VSSTORE_COMPACT=1): a record holds only the locations the PS reads. The slot of a location is its rank in the PS
// mask (the PS knows its mask when it is compiled, the VS gets it with the pair) and the stride is the number of such locations.
constexpr uint32_t PvCompactSlot(uint32_t ps_location_mask, uint32_t location) {
#if defined(KYTY_PVVS_MUTANT_RANK_OFF_BY_ONE)
	return static_cast<uint32_t>(std::popcount(ps_location_mask & ((2u << location) - 1u)));
#else
	return static_cast<uint32_t>(std::popcount(ps_location_mask & ((1u << location) - 1u)));
#endif
}

constexpr uint32_t PvCompactStride(uint32_t ps_location_mask) {
#if defined(KYTY_PVVS_MUTANT_STRIDE_FROM_VS)
	return ps_location_mask == 0 ? 0u : 32u - static_cast<uint32_t>(std::countl_zero(ps_location_mask));
#else
	return static_cast<uint32_t>(std::popcount(ps_location_mask));
#endif
}

enum class PvVsStoreInterfaceVerdict : uint32_t {
	Use,
	Param31,       // the VS or the PS uses host location 31, which carries the instance index
	MissingOutput, // the PS reads a per-vertex location the VS does not write
	NoPerVertex,   // the PS has no per-vertex input at all
};

constexpr PvVsStoreInterfaceVerdict ClassifyPvVsStoreInterface(uint32_t vs_location_mask,
                                                              uint32_t ps_location_mask) {
	using V = PvVsStoreInterfaceVerdict;
	if (ps_location_mask == 0) return V::NoPerVertex;
#if !defined(KYTY_PVVS_MUTANT_PARAM31_OK)
	if (((vs_location_mask | ps_location_mask) >> kPvVsStoreInstanceLocation) & 1u) return V::Param31;
#endif
	if ((ps_location_mask & ~vs_location_mask) != 0) return V::MissingOutput;
	return V::Use;
}

// The push block both stages read after the native push data (dword order = PvVsStorePush).
constexpr std::array<uint32_t, kPvVsStorePushDwords> PvVsStorePushBlock(const PvVsStoreDraw& draw,
                                                                       uint32_t stride_vec4) {
	std::array<uint32_t, kPvVsStorePushDwords> block {};
	block[kPvPushBase]          = draw.base;
	block[kPvPushSpan]          = draw.span;
	block[kPvPushFirst]         = draw.first;
#if defined(KYTY_PVVS_MUTANT_PUSH_NO_FIRST_INSTANCE)
	block[kPvPushFirstInstance] = 0;
#else
	block[kPvPushFirstInstance] = draw.first_instance;
#endif
	block[kPvPushFirstIndex]    = draw.first_index;
	block[kPvPushVertexOffset]  = static_cast<uint32_t>(draw.vertex_offset);
	block[kPvPushIndex32]       = draw.index32 ? 1u : 0u;
	block[kPvPushStride]        = stride_vec4;
	return block;
}

// Push constant range of a graphics pipeline layout: the native data, plus the VS-store block only for
// the VS-store variants (every other pipeline keeps the layout it always had).
constexpr uint32_t PvVsStorePushRangeBytes(uint32_t native_bytes, bool vs_store_variant, bool guard = false) {
#if defined(KYTY_PVVS_MUTANT_PUSH_ALWAYS)
	(void)vs_store_variant;
	return native_bytes + PvVsStorePushDwords(guard) * 4u;
#elif defined(KYTY_PVVS_MUTANT_GUARD_PUSH_ALWAYS)
	return native_bytes + (vs_store_variant ? PvVsStorePushDwords(true) * 4u : 0u);
#else
	return native_bytes + (vs_store_variant ? PvVsStorePushDwords(guard) * 4u : 0u);
#endif
}

// Storage-buffer descriptors must start at minStorageBufferOffsetAlignment: the draw binds the
// index buffer from the aligned-down offset and moves the difference into first_index.
struct PvVsStoreIndexBinding {
	uint64_t offset      = 0; // descriptor offset (aligned)
	uint32_t first_index = 0; // element offset inside the descriptor range
	bool     ok          = false;
};
constexpr PvVsStoreIndexBinding PvVsStoreBindIndex(uint64_t byte_offset, uint32_t element_size,
                                                   uint64_t alignment) {
	if (alignment == 0 || (alignment & (alignment - 1)) != 0 || (element_size != 2 && element_size != 4))
		return {};
	const uint64_t aligned = byte_offset & ~(alignment - 1);
	const uint64_t delta   = byte_offset - aligned;
	if (delta % element_size != 0) return {};
#if defined(KYTY_PVVS_MUTANT_INDEX_NO_ADJUST)
	return {aligned, 0, true};
#else
	return {aligned, static_cast<uint32_t>(delta / element_size), true};
#endif
}

// Descriptor range of the index buffer: the PS reads it as uint32 words, so a range that ends inside a
// word (16-bit indices, odd end) makes the last word out of range and the last index reads wrong.
constexpr uint64_t PvVsStoreIndexRange(uint32_t first_index, uint32_t element_size, uint64_t source_size) {
	const uint64_t bytes = uint64_t {first_index} * element_size + source_size;
#if defined(KYTY_PVVS_MUTANT_INDEX_RANGE_NO_ROUND)
	return bytes;
#else
	return (bytes + 3u) & ~uint64_t {3};
#endif
}

// What the draw binds for the index buffer: the aligned descriptor offset, first_index, and the range
// clamped so that offset + range never exceeds the buffer (a descriptor past the end is invalid). When the
// buffer ends inside the last word the PS reads, that word does not fit: `tail_fits` is false and the
// draw stays on the prototype (pv_fallback_index_tail). `clamp` is a seam for the tests' mutant.
struct PvVsStoreIndexDescriptor {
	uint64_t offset      = 0;
	uint32_t first_index = 0;
	uint64_t range       = 0;
	bool     ok          = false; // the offset can be bound at all
	bool     tail_fits   = false; // the whole last word is inside the buffer
};
constexpr PvVsStoreIndexDescriptor PvVsStoreIndexDescriptorFor(uint64_t byte_offset, uint32_t element_size,
                                                               uint64_t alignment, uint64_t source_size,
                                                               uint64_t buffer_size, bool clamp = true) {
	PvVsStoreIndexDescriptor out;
	const auto               bind = PvVsStoreBindIndex(byte_offset, element_size, alignment);
	if (!bind.ok) return out;
	out.ok          = true;
	out.offset      = bind.offset;
	out.first_index = bind.first_index;
	const uint64_t wanted    = PvVsStoreIndexRange(bind.first_index, element_size, source_size);
	const uint64_t available = buffer_size > bind.offset ? buffer_size - bind.offset : 0;
	out.range                = clamp && wanted > available ? available : wanted;
	out.tail_fits            = !clamp || wanted <= available;
	return out;
}

// Whether the index binding the draw really got (offset and size of the buffer PrepareIndexBuffer
// returned) can be bound as the descriptor: the offset is bindable and the last word the PS reads is
// inside the buffer. The decision is taken on this binding, not on an earlier ObtainBuffer: the buffer
// cache may merge or move the range while the other resources of the draw are prepared.
constexpr bool PvVsStoreIndexBindingFits(uint64_t byte_offset, uint32_t element_size, uint64_t alignment,
                                         uint64_t source_size, uint64_t buffer_size) {
	const auto desc = PvVsStoreIndexDescriptorFor(byte_offset, element_size, alignment, source_size, buffer_size);
#if defined(KYTY_PVVS_MUTANT_BINDING_FITS_NO_TAIL)
	return desc.ok;
#else
	return desc.ok && desc.tail_fits;
#endif
}

// S388 (KYTY_PV_VSSTORE_PAIRS=all|<VS:PS,...>|first=N, default all): which VS/PS pairs may use the store; the
// others stay on the prototype. A list token is `VS:PS`, each side a decimal program id or a 0x guest shader
// hash (hashes are stable between runs, ids are not); `first=N` admits the first N candidate pairs in the order
// they are first drawn. S389: `only=VSHASH:PSHASH,...` is the list form that matches guest shader hashes only (ids
// change between runs); `range=a-b` admits the candidate pairs whose index (the `#N` of the `[pv-pairs]` log line,
// counted in the order the pairs are first drawn) lies in a..b inclusive.
struct PvVsStorePairFilter {
	enum class Kind : uint32_t { All, First, List, Only, Range };
	struct Pair {
		uint64_t vs = 0;
		uint64_t ps = 0;
	};
	Kind              kind    = Kind::All;
	uint32_t          first_n = 0;
	uint32_t          range_a = 0; // Range: candidate pair indices a..b, inclusive
	uint32_t          range_b = 0;
	std::vector<Pair> pairs;
};

inline bool ParsePvVsStorePairNumber(std::string_view text, uint64_t& value) {
	if (text.empty()) return false;
	int base = 10;
	if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
		base = 16;
		text.remove_prefix(2);
	}
	value = 0;
	for (const char c: text) {
		uint64_t digit = 0;
		if (c >= '0' && c <= '9') digit = static_cast<uint64_t>(c - '0');
		else if (base == 16 && c >= 'a' && c <= 'f') digit = static_cast<uint64_t>(c - 'a' + 10);
		else if (base == 16 && c >= 'A' && c <= 'F') digit = static_cast<uint64_t>(c - 'A' + 10);
		else return false;
		if (value > (UINT64_MAX - digit) / static_cast<uint64_t>(base)) return false;
		value = value * static_cast<uint64_t>(base) + digit;
	}
	return true;
}

// nullptr = unset = all. False: not a valid value (the caller aborts at startup).
inline bool ParsePvVsStorePairs(const char* text, PvVsStorePairFilter& filter) {
	filter = {};
	if (text == nullptr) return true;
	std::string_view value {text};
	if (value == "all") return true;
	if (value.starts_with("first=")) {
		uint64_t n = 0;
		if (!ParsePvVsStorePairNumber(value.substr(6), n) || n > 0xffffffffull) return false;
		filter.kind    = PvVsStorePairFilter::Kind::First;
		filter.first_n = static_cast<uint32_t>(n);
		return true;
	}
	if (value.starts_with("range=")) {
		value.remove_prefix(6);
		const auto dash = value.find('-');
		uint64_t   a = 0;
		uint64_t   b = 0;
		if (dash == std::string_view::npos || !ParsePvVsStorePairNumber(value.substr(0, dash), a) ||
		    !ParsePvVsStorePairNumber(value.substr(dash + 1), b) || a > b || b > 0xffffffffull) {
			return false;
		}
		filter.kind    = PvVsStorePairFilter::Kind::Range;
		filter.range_a = static_cast<uint32_t>(a);
		filter.range_b = static_cast<uint32_t>(b);
		return true;
	}
	filter.kind = PvVsStorePairFilter::Kind::List;
	if (value.starts_with("only=")) {
		filter.kind = PvVsStorePairFilter::Kind::Only;
		value.remove_prefix(5);
	}
	while (!value.empty()) {
		const auto comma = value.find(',');
		const auto token = value.substr(0, comma);
		const auto colon = token.find(':');
		PvVsStorePairFilter::Pair pair;
		if (colon == std::string_view::npos || !ParsePvVsStorePairNumber(token.substr(0, colon), pair.vs) ||
		    !ParsePvVsStorePairNumber(token.substr(colon + 1), pair.ps)) {
			filter = {};
			return false;
		}
		filter.pairs.push_back(pair);
		if (comma == std::string_view::npos) break;
		value.remove_prefix(comma + 1);
		if (value.empty()) {
			filter = {};
			return false; // trailing comma
		}
	}
	if (filter.pairs.empty()) {
		filter = {};
		return false;
	}
	return true;
}

// `ordinal` = the index of this pair among the candidate pairs, in the order they are first drawn (S389: it was the
// admitted count, which is the same for first=N).
inline bool PvVsStorePairAllowed(const PvVsStorePairFilter& filter, uint64_t vs_id, uint64_t vs_hash, uint64_t ps_id,
                                 uint64_t ps_hash, uint32_t ordinal) {
	switch (filter.kind) {
		case PvVsStorePairFilter::Kind::All: return true;
		case PvVsStorePairFilter::Kind::First:
#if defined(KYTY_PVVS_MUTANT_PAIRS_FIRST_OFF_BY_ONE)
			return ordinal <= filter.first_n;
#else
			return ordinal < filter.first_n;
#endif
		case PvVsStorePairFilter::Kind::Range:
#if defined(KYTY_PVVS_MUTANT_PAIRS_RANGE_EXCLUSIVE)
			return ordinal >= filter.range_a && ordinal < filter.range_b;
#else
			return ordinal >= filter.range_a && ordinal <= filter.range_b;
#endif
		case PvVsStorePairFilter::Kind::Only:
			for (const auto& pair: filter.pairs) {
#if defined(KYTY_PVVS_MUTANT_PAIRS_ONLY_USES_IDS)
				if (pair.vs == vs_id && pair.ps == ps_id) return true;
#else
				if (pair.vs == vs_hash && pair.ps == ps_hash) return true;
#endif
			}
			return false;
		case PvVsStorePairFilter::Kind::List:
			for (const auto& pair: filter.pairs) {
				const bool vs_ok = pair.vs == vs_id || pair.vs == vs_hash;
#if defined(KYTY_PVVS_MUTANT_PAIRS_LIST_IGNORES_PS)
				const bool ps_ok = true;
#else
				const bool ps_ok = pair.ps == ps_id || pair.ps == ps_hash;
#endif
				if (vs_ok && ps_ok) return true;
			}
			return false;
	}
	return false;
}

// S388 (KYTY_PV_VSSTORE_INSTANCED=1|0, default 1): 0 sends every draw with instance_count > 1 to the prototype.
inline bool ParsePvVsStoreInstanced(const char* text, bool& allowed) {
	allowed = true;
	if (text == nullptr) return true;
	const std::string_view value {text};
	if (value == "1") return true;
	if (value == "0") {
		allowed = false;
		return true;
	}
	return false;
}

// S388 (KYTY_PV_VSSTORE_GPUIDX=allow|fallback, default allow): the vertex range of a draw is scanned from the
// index bytes the CPU sees; when the GPU wrote those indices (the guest memory is GPU-modified) the scan can be
// stale, so `fallback` sends such a draw to the prototype (counted as a range fallback).
enum class PvVsStoreGpuIdxMode : uint32_t { Allow, Fallback };

// nullptr = unset = allow. False: not a mode name.
inline bool ParsePvVsStoreGpuIdxMode(const char* text, PvVsStoreGpuIdxMode& mode) {
	mode = PvVsStoreGpuIdxMode::Allow;
	if (text == nullptr) return true;
	const std::string_view value {text};
	if (value == "allow") return true;
	if (value == "fallback") {
		mode = PvVsStoreGpuIdxMode::Fallback;
		return true;
	}
	return false;
}

constexpr bool PvVsStoreGpuIndexFallback(PvVsStoreGpuIdxMode mode, bool gpu_modified) {
#if defined(KYTY_PVVS_MUTANT_GPUIDX_ALLOW_ALWAYS)
	(void)mode;
	(void)gpu_modified;
	return false;
#elif defined(KYTY_PVVS_MUTANT_GPUIDX_IGNORES_GPU)
	(void)gpu_modified;
	return mode == PvVsStoreGpuIdxMode::Fallback;
#else
	return mode == PvVsStoreGpuIdxMode::Fallback && gpu_modified;
#endif
}

// Cap slabs of the renderer. A region lives until the command buffer of its draw has finished on the
// GPU, so a slab is recycled only after the tick it was last used in is free; regions inside a slab
// are never reused (PvVsStoreRing). The buffers themselves are the renderer's; this is the bookkeeping.
class PvVsStoreSlabs {
public:
	struct Placement {
		bool     ok      = false;
		bool     created = false; // the caller must create the buffer of slab `slab`
		uint32_t slab    = 0;
		uint32_t base    = 0;
	};
	PvVsStoreSlabs(uint64_t slab_records, uint32_t max_slabs)
	    : m_slab_records(slab_records), m_max_slabs(max_slabs) {}

	template <class IsFree>
	Placement Place(uint64_t records, uint64_t tick, IsFree&& is_free) {
		Placement out;
		if (records == 0 || records > m_slab_records) return out;
		const auto try_slab = [&](uint32_t index) {
			if (!m_slabs[index].ring.Allocate(records, out.base)) return false;
			m_slabs[index].tick = tick;
			m_current           = index;
			out.ok              = true;
			out.slab            = index;
			return true;
		};
		if (m_current < m_slabs.size() && m_slabs[m_current].tick == tick && try_slab(m_current))
			return out;
		for (uint32_t i = 0; i < m_slabs.size(); ++i) {
			if (i == m_current) continue;
			auto& slab = m_slabs[i];
#if defined(KYTY_PVVS_MUTANT_POOL_REUSE_BUSY)
			const bool reusable = true;
#else
			const bool reusable = slab.tick == tick || !slab.used || is_free(slab.tick);
#endif
			if (!reusable) continue;
			if (slab.tick != tick) slab.ring.EndPass();
			slab.used = true;
			if (try_slab(i)) return out;
		}
		if (m_current < m_slabs.size() && m_slabs[m_current].tick != tick &&
		    (!m_slabs[m_current].used || is_free(m_slabs[m_current].tick))) {
			m_slabs[m_current].ring.EndPass();
			if (try_slab(m_current)) return out;
		}
		if (m_slabs.size() >= m_max_slabs) return out;
		m_slabs.push_back({PvVsStoreRing(m_slab_records), tick, true});
		out.created = true;
		if (!try_slab(static_cast<uint32_t>(m_slabs.size() - 1))) out.created = false;
		return out;
	}
	// The draw of a placed region is recorded in `tick`, which can be newer than the tick of Place() when
	// resource preparation flushed the scheduler in between: the slab is busy until that tick finishes.
	void Touch(uint32_t slab, uint64_t tick) {
#if !defined(KYTY_PVVS_MUTANT_NO_TOUCH)
		if (slab < m_slabs.size() && m_slabs[slab].tick < tick) m_slabs[slab].tick = tick;
#else
		(void)slab;
		(void)tick;
#endif
	}
	[[nodiscard]] size_t   Size() const { return m_slabs.size(); }
	[[nodiscard]] uint64_t SlabRecords() const { return m_slab_records; }
	// S406 (DIAG): records left in the slab the last placement used (0 when none was used yet).
	[[nodiscard]] uint64_t CurrentFreeRecords() const { return m_current < m_slabs.size() ? m_slabs[m_current].ring.Free() : 0; }

private:
	struct Slab {
		PvVsStoreRing ring;
		uint64_t      tick = 0;
		bool          used = false;
	};
	std::vector<Slab> m_slabs;
	uint64_t          m_slab_records;
	uint32_t          m_max_slabs;
	uint32_t          m_current = UINT32_MAX;
};

// Which pv_fallback_* counter a draw that stays on the prototype is counted in. Instanced draws that
// fail for a size reason (range / region too large / no ring room) are counted as instanced.
enum class PvVsStoreFallback : uint32_t { None, Topology, Range, Instanced, Ring, Param31, Interface, IndexTail };

constexpr PvVsStoreFallback ClassifyPvVsStoreFallback(PvVsStoreVerdict verdict, uint32_t instance_count) {
	using V = PvVsStoreVerdict;
	using F = PvVsStoreFallback;
	switch (verdict) {
		case V::Disabled:
		case V::Use: return F::None;
		case V::NotTriangleList:
		case V::NotIndexed:
		case V::PrimitiveRestart:
		case V::BadIndexSize: return F::Topology;
		case V::RingFull: return instance_count > 1 ? F::Instanced : F::Ring;
		case V::InstancedOff: return F::Instanced;
		case V::NoRange:
		case V::Dropped:
		case V::Empty:
		case V::TooLarge:
#if defined(KYTY_PVVS_MUTANT_NO_INSTANCED_CLASS)
			return F::Range;
#else
			return instance_count > 1 ? F::Instanced : F::Range;
#endif
	}
	return F::Range;
}

constexpr PvVsStoreFallback ClassifyPvVsStoreFallback(PvVsStoreInterfaceVerdict verdict) {
	using V = PvVsStoreInterfaceVerdict;
	switch (verdict) {
		case V::Use:
		case V::NoPerVertex: return PvVsStoreFallback::None;
		case V::Param31: return PvVsStoreFallback::Param31;
		case V::MissingOutput: return PvVsStoreFallback::Interface;
	}
	return PvVsStoreFallback::Interface;
}


// The whole decision of a draw, in the order the renderer applies it: the shader pair must be a
// candidate, the plan must accept the draw, the index buffer must be bindable as an SSBO, and the cap
// ring must have room (`place(plan)` reserves the region). Anything else stays on the prototype and is
// counted under `fallback`.
struct PvVsStoreDecision {
	bool              use      = false;
	PvVsStoreFallback fallback = PvVsStoreFallback::None;
	PvVsStorePlan     plan;
};

template <class Place>
PvVsStoreDecision DecidePvVsStore(PvVsStoreInterfaceVerdict interface_verdict, const PvVsStoreRequest& request,
                                  bool index_aligned, Place&& place,
                                  bool index_tail_fits = true, bool check_tail = true) {
	PvVsStoreDecision decision;
	if (interface_verdict != PvVsStoreInterfaceVerdict::Use) {
		decision.fallback = ClassifyPvVsStoreFallback(interface_verdict);
		return decision;
	}
	decision.plan = PlanPvVsStore(request);
	if (!decision.plan.Use()) {
		decision.fallback = ClassifyPvVsStoreFallback(decision.plan.verdict, request.instance_count);
		return decision;
	}
#if !defined(KYTY_PVVS_MUTANT_ALIGN_IGNORED)
	if (!index_aligned) {
		decision.fallback = PvVsStoreFallback::Range;
		return decision;
	}
#endif
	if (check_tail && !index_tail_fits) { // check before the ring is touched; check_tail=false is the tests' mutant seam
		decision.fallback = PvVsStoreFallback::IndexTail;
		return decision;
	}
#if !defined(KYTY_PVVS_MUTANT_RING_IGNORED)
	if (!place(decision.plan)) {
		decision.fallback = ClassifyPvVsStoreFallback(PvVsStoreVerdict::RingFull, request.instance_count);
		return decision;
	}
#else
	(void)place(decision.plan);
#endif
	decision.use = true;
	return decision;
}

} // namespace Libs::Graphics

#endif
