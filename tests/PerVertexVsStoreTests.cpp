// S149: KYTY_PV_VSSTORE planner, record model and pass ring (CPU). The model plays the VS store and
// the PS corner read of perVertexVsStore.h bit for bit against a direct read of the vertex array.
// Mutants (each must be caught, WILL_FAIL): corner 0 and 2 swapped, instance term dropped, "- first"
// dropped, 16-bit halves swapped, first_index ignored, vertex_offset ignored, dropped corners
// accepted, ring region reused inside a pass.
#include "graphics/host_gpu/renderer/perVertexVsStore.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace Libs::Graphics;

namespace {

int g_failures = 0;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		g_failures++;
	}
}

struct Vec4 {
	uint32_t w[4];
	bool     operator==(const Vec4&) const = default;
};

// Bits chosen to catch float canonicalisation: NaN with payload, denormal, -0, inf, plain values.
Vec4 SourceVertexBase(uint32_t vertex) {
	switch (vertex % 6) {
		case 0: return {{0x7FFF0001u, 0x00000001u, 0x80000000u, vertex}};
		case 1: return {{0x7F800000u, 0xFF800000u, 0x007FFFFFu, vertex * 3u}};
		default: return {{0x3F800000u + vertex, vertex * 0x9E3779B9u, 0xA5A5A5A5u ^ vertex, vertex}};
	}
}

// What the guest VS outputs for (vertex, instance).
Vec4 SourceVertex(uint32_t vertex, uint32_t instance) {
	Vec4 v = SourceVertexBase(vertex);
	v.w[3] ^= instance * 0x01000193u;
	return v;
}

struct Case {
	const char* name;
	bool        index32;
	uint32_t    first_index;
	int32_t     vertex_offset;
	uint32_t    instances;
	uint32_t    first_instance;
	uint32_t    prims;
};

// Plays one draw. Returns the number of corner reads that differ from the source vertex.
uint32_t RunCase(const Case& c, uint32_t ring_base) {
	const uint32_t stride_dummy = 1;
	const uint32_t vertex_count = 40;
	// Index values stay in [0, 24); the shaders see index + vertex_offset (vertex_offset may be < 0).
	std::vector<uint32_t> indices(c.first_index + c.prims * 3);
	for (size_t i = 0; i < indices.size(); i++) indices[i] = static_cast<uint32_t>((i * 7 + 3) % 24);
	std::vector<uint32_t> words;
	if (c.index32) {
		words = indices;
	} else {
		words.assign((indices.size() + 1) / 2, 0);
		for (size_t i = 0; i < indices.size(); i++) words[i >> 1] |= indices[i] << ((i & 1) * 16);
	}
	const auto vertex_of = [&](uint32_t index) { return index + static_cast<uint32_t>(c.vertex_offset); };

	uint32_t lo = UINT32_MAX, hi = 0;
	for (size_t i = c.first_index; i < indices.size(); i++) {
		lo = std::min(lo, vertex_of(indices[i]));
		hi = std::max(hi, vertex_of(indices[i]) + 1);
	}
	PvVsStoreRequest request;
	request.enabled = true;
	request.triangle_list = true;
	request.indexed = true;
	request.index_size = c.index32 ? 4 : 2;
	request.range_known = true;
	request.range_first = lo;
	request.range_end = hi;
	request.index_count = c.prims * 3;
	request.instance_count = c.instances;
	request.record_stride_vec4 = stride_dummy;
	const auto plan = PlanPvVsStore(request);
	if (!plan.Use()) return 0xffffffffu;

	PvVsStoreDraw draw;
	draw.base = ring_base;
	draw.span = plan.span;
	draw.first = plan.first;
	draw.first_instance = c.first_instance;
	draw.first_index = c.first_index;
	draw.vertex_offset = c.vertex_offset;
	draw.index32 = c.index32;

	std::vector<Vec4> cap(ring_base + plan.records, Vec4 {{0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu, 0xDEADBEEFu}});
	// VS side: every (instance, vertex) the pipeline invokes writes its record.
	for (uint32_t inst = 0; inst < c.instances; inst++) {
		for (size_t i = c.first_index; i < indices.size(); i++) {
			const uint32_t v = vertex_of(indices[i]);
			const uint32_t record = PvVsStoreVertexRecord(draw.base, draw.span, draw.first, draw.first_instance,
			                                              draw.first_instance + inst, v);
			if (record >= cap.size()) return 0xffffffffu; // a store outside the region
			cap[record] = SourceVertex(v, draw.first_instance + inst);
		}
	}
	// PS side.
	uint32_t bad = 0;
	for (uint32_t inst = 0; inst < c.instances; inst++) {
		for (uint32_t prim = 0; prim < c.prims; prim++) {
			for (uint32_t k = 0; k < kPvVsStoreCorners; k++) {
				const uint32_t expected_vertex = vertex_of(indices[c.first_index + prim * 3 + k]);
				const uint32_t record = PvVsStoreCornerRecord(draw, words, inst, prim, k);
				if (record >= cap.size() || !(cap[record] == SourceVertex(expected_vertex, draw.first_instance + inst))) bad++;
			}
		}
	}
	(void)vertex_count;
	return bad;
}

void TestModel() {
	const Case cases[] = {
	    {"idx32 plain", true, 0, 0, 1, 0, 9},
	    {"idx16 plain", false, 0, 0, 1, 0, 9},
	    {"idx32 first_index", true, 6, 0, 1, 0, 8},
	    {"idx16 first_index odd", false, 3, 0, 1, 0, 8},
	    {"idx16 first_index even", false, 6, 0, 1, 0, 8},
	    {"idx32 vertex_offset", true, 0, 5, 1, 0, 9},
	    {"idx16 vertex_offset", false, 0, 12, 1, 0, 9},
	    {"idx32 2 instances", true, 0, 0, 2, 0, 9},
	    {"idx16 2 instances first_instance", false, 0, 0, 2, 3, 9},
	    {"idx32 everything", true, 9, 7, 2, 5, 11},
	    {"idx16 everything", false, 5, 2, 2, 1, 11},
	};
	for (const auto& c: cases) {
		for (const uint32_t base: {0u, 1000u}) {
			const auto bad = RunCase(c, base);
			if (bad != 0) {
				std::fprintf(stderr, "FAIL: model case '%s' base %u: %u wrong corner reads\n", c.name, base, bad);
				g_failures++;
			}
		}
	}
}

PvVsStoreRequest GoodRequest() {
	PvVsStoreRequest r;
	r.enabled = true;
	r.triangle_list = true;
	r.indexed = true;
	r.index_size = 2;
	r.range_known = true;
	r.range_first = 10;
	r.range_end = 50;
	r.index_count = 30;
	r.instance_count = 2;
	r.record_stride_vec4 = 3;
	return r;
}

void TestPlan() {
	using V = PvVsStoreVerdict;
	auto r = GoodRequest();
	auto plan = PlanPvVsStore(r);
	Check(plan.Use() && plan.first == 10 && plan.span == 40 && plan.records == 80, "plan of a plain draw");
	r.enabled = false;
	Check(PlanPvVsStore(r).verdict == V::Disabled, "disabled switch");
	r = GoodRequest();
	r.triangle_list = false;
	Check(PlanPvVsStore(r).verdict == V::NotTriangleList, "rect list stays on the prototype");
	r = GoodRequest();
	r.indexed = false;
	Check(PlanPvVsStore(r).verdict == V::NotIndexed, "non-indexed stays on the prototype");
	r = GoodRequest();
	r.primitive_restart = true;
	Check(PlanPvVsStore(r).verdict == V::PrimitiveRestart, "restart stays on the prototype");
	r = GoodRequest();
	r.index_size = 1;
	Check(PlanPvVsStore(r).verdict == V::BadIndexSize, "8-bit index buffer rejected");
	r = GoodRequest();
	r.range_known = false;
	Check(PlanPvVsStore(r).verdict == V::NoRange, "unknown range rejected");
	r = GoodRequest();
	r.range_dropped = true;
	Check(PlanPvVsStore(r).verdict == V::Dropped, "dropped corner rejected");
	r = GoodRequest();
	r.index_count = 2;
	Check(PlanPvVsStore(r).verdict == V::Empty, "less than one triangle");
	r = GoodRequest();
	r.range_end = 0x100000000ull;
	Check(PlanPvVsStore(r).verdict == V::TooLarge, "32-bit overflow rejected");
}

void TestRing() {
	PvVsStoreRing ring(100);
	uint32_t a = 0, b = 0, c = 0;
	Check(ring.Allocate(40, a) && a == 0, "first region at 0");
	Check(ring.Allocate(40, b) && b == 40, "second region does not overlap the first");
	Check(!ring.Allocate(40, c), "full ring refuses");
	Check(ring.Allocate(20, c) && c == 80, "exact fit");
	Check(ring.Used() == 100 && ring.Draws() == 3, "ring counters");
	ring.EndPass();
	Check(ring.Allocate(40, a) && a == 0, "ring restarts after the pass");
	uint32_t zero = 7;
	Check(!ring.Allocate(0, zero), "empty region refused");
}

void TestSlots() {
	const uint32_t locations[] = {7, 2, 7, 30};
	const auto     slots       = PvVsStoreSlots(locations);
	Check(slots.size() == 3 && slots.at(2) == 0 && slots.at(7) == 1 && slots.at(30) == 2, "dense slots by location");
}

// S150: interface of a VS/PS pair, push block, index binding, slab pool.
void TestInterface() {
	using V = PvVsStoreInterfaceVerdict;
	Check(PvVsStoreStrideFor(0) == 0 && PvVsStoreStrideFor(1) == 1 && PvVsStoreStrideFor(0b10100) == 5 &&
	          PvVsStoreStrideFor(1u << 30) == 31,
	      "stride is the highest written location + 1");
	Check(ClassifyPvVsStoreInterface(0b111, 0b101) == V::Use, "PS subset of the VS outputs");
	Check(ClassifyPvVsStoreInterface(0b111, 0b1000) == V::MissingOutput, "PS reads a location the VS lacks");
	Check(ClassifyPvVsStoreInterface(0b111, 0) == V::NoPerVertex, "no per-vertex input");
	Check(ClassifyPvVsStoreInterface(1u << 31 | 1u, 1u) == V::Param31, "VS writes location 31");
	Check(ClassifyPvVsStoreInterface(0b11, 1u << 31) == V::Param31, "PS reads location 31");
}

// S409: record of the locations the PS reads: slot = rank in the PS mask, stride = popcount.
void TestCompact() {
	struct Row { uint32_t mask, location, slot; };
	const Row rows[] = {{0x1, 0, 0},    {0x8, 3, 0},    {0x205, 0, 0},  {0x205, 2, 1},  {0x205, 9, 2},   {0x205, 1, 1},
	                    {0x7fffffffu, 30, 30}, {0x80000001u, 0, 0}, {0b10100, 4, 1}, {0b10100, 2, 0}, {0b10100, 3, 1}};
	for (const auto& r : rows) {
		const bool in_mask = (r.mask >> r.location) & 1u;
		// a location outside the mask has the rank of the next one in it (never stored); only a location in the mask has a slot
		if (in_mask) Check(PvCompactSlot(r.mask, r.location) == r.slot, "compact slot is the rank in the PS mask");
	}
	Check(PvCompactSlot(0x205, 0) == 0 && PvCompactSlot(0x205, 2) == 1 && PvCompactSlot(0x205, 9) == 2, "gapped mask 0x205");
	Check(PvCompactSlot(0x8, 3) == 0, "single location 3 has slot 0");
	Check(PvCompactSlot(0b10100, 4) == 1 && PvCompactSlot(0b10100, 2) == 0, "mask 0b10100");
	Check(PvCompactSlot(0x7fffffffu, 30) == 30, "dense mask, last location");
	Check(PvCompactStride(0) == 0 && PvCompactStride(0x8) == 1 && PvCompactStride(0x205) == 3 && PvCompactStride(0b10100) == 2 &&
	          PvCompactStride(0x7fffffffu) == 31,
	      "compact stride is the number of locations");
	// the slots of one mask are dense 0..stride-1 in location order
	for (const uint32_t mask : {0x205u, 0x8u, 0b10100u, 0x40000001u, 0x7fffffffu}) {
		uint32_t expect = 0;
		for (uint32_t l = 0; l < 31; ++l) {
			if (((mask >> l) & 1u) == 0) continue;
			Check(PvCompactSlot(mask, l) == expect, "slots are dense and ordered");
			expect++;
		}
		Check(expect == PvCompactStride(mask), "stride equals the number of slots");
	}
	// the stride never exceeds the old one: the whole point of the layout
	Check(PvCompactStride(0x205) < PvVsStoreStrideFor(0x205), "gapped mask is smaller than the highest-location stride");
	bool compact = false;
	Check(ParsePvVsStoreCompact(nullptr, compact) && compact, "COMPACT unset = 1 (default since S411)");
	Check(ParsePvVsStoreCompact("1", compact) && compact, "COMPACT=1");
	Check(ParsePvVsStoreCompact("0", compact) && !compact, "COMPACT=0");
	Check(!ParsePvVsStoreCompact("2", compact), "COMPACT=2 is rejected");
}

// S409: the per-pair literal of the specialization constant goes into the right instruction.
void TestCompactPatch() {
	// header (5 words), OpSpecConstant %2 = 7 (id 3 first, an unrelated one), OpConstant, OpSpecConstant id 9
	std::vector<uint32_t> words = {0x07230203, 0x10300, 0, 20, 0,
	                               (4u << 16) | 50u, 1, 3, 111,  // other spec constant: untouched
	                               (4u << 16) | 43u, 1, 4, 222,  // OpConstant
	                               (4u << 16) | 50u, 1, 9, 333}; // the mask
	Check(PvCompactPatchMask(words, 9, 0x205) && words[8] == 111 && words[12] == 222 && words[16] == 0x205, "mask literal patched");
	Check(!PvCompactPatchMask(words, 77, 1), "unknown spec id is not patched");
	std::vector<uint32_t> broken = {0x07230203, 0x10300, 0, 20, 0, (4u << 16) | 50u, 1};
	Check(!PvCompactPatchMask(broken, 9, 1), "truncated module is refused");
}

void TestPushBlock() {
	PvVsStoreDraw draw;
	draw.base = 11; draw.span = 22; draw.first = 33; draw.first_instance = 44;
	draw.first_index = 55; draw.vertex_offset = -66; draw.index32 = false;
	const auto block = PvVsStorePushBlock(draw, 7);
	Check(block[kPvPushBase] == 11 && block[kPvPushSpan] == 22 && block[kPvPushFirst] == 33 &&
	          block[kPvPushFirstInstance] == 44 && block[kPvPushFirstIndex] == 55 &&
	          block[kPvPushVertexOffset] == static_cast<uint32_t>(-66) && block[kPvPushIndex32] == 0 &&
	          block[kPvPushStride] == 7,
	      "push block dwords");
	draw.index32 = true;
	Check(PvVsStorePushBlock(draw, 7)[kPvPushIndex32] == 1, "32-bit flag");
}

void TestIndexBinding() {
	auto b = PvVsStoreBindIndex(256, 2, 16);
	Check(b.ok && b.offset == 256 && b.first_index == 0, "aligned offset needs no adjustment");
	b = PvVsStoreBindIndex(270, 2, 16);
	Check(b.ok && b.offset == 256 && b.first_index == 7, "16-bit: 14 bytes = 7 elements");
	b = PvVsStoreBindIndex(268, 4, 16);
	Check(b.ok && b.offset == 256 && b.first_index == 3, "32-bit: 12 bytes = 3 elements");
	Check(!PvVsStoreBindIndex(257, 2, 16).ok, "misaligned to the element size");
	Check(!PvVsStoreBindIndex(256, 1, 16).ok && !PvVsStoreBindIndex(256, 2, 12).ok, "bad size / alignment");
}

void TestIndexRange() {
	Check(PvVsStoreIndexRange(0, 2, 24) == 24, "aligned 16-bit range");
	Check(PvVsStoreIndexRange(1, 2, 24) == 28, "16-bit odd start: 26 bytes rounded to a whole word");
	Check(PvVsStoreIndexRange(3, 2, 18) == 24, "16-bit: 24 bytes");
	Check(PvVsStoreIndexRange(0, 2, 6) == 8, "16-bit, 3 indices");
	Check(PvVsStoreIndexRange(3, 4, 12) == 24, "32-bit unchanged");
}

// S151: the index descriptor stays inside the buffer. The test mutants (KYTY_PVVS_TEST_MUTANT_*) are seams
// of this file only: they call the production functions with the check switched off.
#if defined(KYTY_PVVS_TEST_MUTANT_NO_CLAMP)
constexpr bool kClamp = false;
#else
constexpr bool kClamp = true;
#endif
#if defined(KYTY_PVVS_TEST_MUTANT_NO_TAIL_CHECK)
constexpr bool kCheckTail = false;
#else
constexpr bool kCheckTail = true;
#endif

void TestIndexDescriptor() {
	// 16-bit, 5 indices (10 bytes) at the very end of a 10-byte buffer: the last word (bytes 8..11) does not fit
	auto d = PvVsStoreIndexDescriptorFor(0, 2, 16, 10, 10, kClamp);
	Check(d.ok && !d.tail_fits, "16-bit odd end at the buffer end: the last word does not fit");
	Check(d.offset + d.range <= 10, "descriptor range never passes the buffer end");
	// the same draw with a 12-byte buffer fits and is rounded to a whole word
	d = PvVsStoreIndexDescriptorFor(0, 2, 16, 10, 12, kClamp);
	Check(d.ok && d.tail_fits && d.range == 12 && d.offset + d.range <= 12, "word-padded buffer: range 12");
	// aligned-down offset: byte 36 of a 64-byte buffer, 24 bytes, alignment 16 -> offset 32, first_index 1 (32-bit)
	d = PvVsStoreIndexDescriptorFor(36, 4, 16, 24, 64, kClamp);
	Check(d.ok && d.tail_fits && d.offset == 32 && d.first_index == 1 && d.range == 28, "aligned-down offset keeps the delta in the range");
	// the delta moves the end: 36..60 inside 60 bytes would need offset 32 + 28 = 60 (fits), a 59-byte buffer does not
	d = PvVsStoreIndexDescriptorFor(36, 4, 16, 24, 59, kClamp);
	Check(d.ok && !d.tail_fits && d.offset + d.range <= 59, "range reaching past the buffer is clamped and reported");
	// a buffer smaller than the offset
	d = PvVsStoreIndexDescriptorFor(32, 4, 16, 8, 16, kClamp);
	Check(d.ok && !d.tail_fits && d.range == 0, "buffer ends before the aligned offset");
	// 32-bit, exact fit
	d = PvVsStoreIndexDescriptorFor(0, 4, 16, 12, 12, kClamp);
	Check(d.ok && d.tail_fits && d.range == 12, "32-bit exact fit");
	Check(!PvVsStoreIndexDescriptorFor(0, 1, 16, 12, 12, kClamp).ok, "bad element size");
}

// The decision is taken on the binding the draw really got: a range that fits where an early ObtainBuffer put it can sit
// elsewhere (another offset, a smaller buffer) once the cache has merged or moved it.
void TestIndexBindingFits() {
	Check(PvVsStoreIndexBindingFits(0, 4, 16, 24, 64), "early binding: offset 0 in a 64-byte buffer fits");
	Check(!PvVsStoreIndexBindingFits(36, 4, 16, 24, 59), "actual binding: offset 36 in a 59-byte buffer does not fit (tail)");
	Check(!PvVsStoreIndexBindingFits(34, 4, 16, 24, 64), "actual binding: offset 34 is not a multiple of the element size");
	Check(PvVsStoreIndexBindingFits(36, 4, 16, 24, 64), "actual binding: offset 36 in a 64-byte buffer fits");
	Check(!PvVsStoreIndexBindingFits(0, 2, 16, 10, 10), "16-bit odd end at the buffer end does not fit");
	Check(PvVsStoreIndexBindingFits(0, 2, 16, 10, 12), "a word-padded buffer fits");
}

// S388 (H1): KYTY_PV_VSSTORE_GPUIDX decides a draw whose index bytes the GPU wrote.
void TestGpuIdxMode() {
	PvVsStoreGpuIdxMode mode = PvVsStoreGpuIdxMode::Fallback;
	Check(ParsePvVsStoreGpuIdxMode(nullptr, mode) && mode == PvVsStoreGpuIdxMode::Allow, "unset = allow");
	Check(ParsePvVsStoreGpuIdxMode("allow", mode) && mode == PvVsStoreGpuIdxMode::Allow, "allow");
	Check(ParsePvVsStoreGpuIdxMode("fallback", mode) && mode == PvVsStoreGpuIdxMode::Fallback, "fallback");
	Check(!ParsePvVsStoreGpuIdxMode("Fallback", mode), "case matters");
	Check(!ParsePvVsStoreGpuIdxMode("", mode), "empty is invalid");
	Check(PvVsStoreGpuIndexFallback(PvVsStoreGpuIdxMode::Fallback, true), "fallback + GPU-written indices falls back");
	Check(!PvVsStoreGpuIndexFallback(PvVsStoreGpuIdxMode::Fallback, false), "fallback + CPU indices stays on the store");
	Check(!PvVsStoreGpuIndexFallback(PvVsStoreGpuIdxMode::Allow, true), "allow keeps GPU-written indices on the store");
	Check(!PvVsStoreGpuIndexFallback(PvVsStoreGpuIdxMode::Allow, false), "allow + CPU indices");
}

// S388 (H2): KYTY_PV_VSSTORE_INSTANCED=0 sends instanced draws to the prototype and only those.
void TestInstancedSwitch() {
	bool allowed = false;
	Check(ParsePvVsStoreInstanced(nullptr, allowed) && allowed, "unset = 1");
	Check(ParsePvVsStoreInstanced("1", allowed) && allowed, "1");
	Check(ParsePvVsStoreInstanced("0", allowed) && !allowed, "0");
	Check(!ParsePvVsStoreInstanced("2", allowed), "2 is invalid");
	Check(!ParsePvVsStoreInstanced("", allowed), "empty is invalid");
	PvVsStoreRequest r;
	r.enabled = true;
	r.triangle_list = true;
	r.indexed = true;
	r.index_size = 2;
	r.range_known = true;
	r.range_first = 0;
	r.range_end = 8;
	r.index_count = 12;
	r.record_stride_vec4 = 2;
	r.instance_count = 3;
	r.instanced_allowed = false;
	const auto place_ok = [](const PvVsStorePlan&) { return true; };
	auto decision = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, place_ok);
	Check(!decision.use && decision.fallback == PvVsStoreFallback::Instanced, "instanced draw with the switch at 0 falls back as Instanced");
	r.instance_count = 1;
	decision = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, place_ok);
	Check(decision.use, "a single-instance draw is unaffected by the switch at 0");
	r.instance_count = 3;
	r.instanced_allowed = true;
	decision = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, place_ok);
	Check(decision.use, "instanced draw with the switch at 1 takes the store");
	r.instanced_allowed = false;
	r.triangle_list = false;
	decision = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, place_ok);
	Check(decision.fallback == PvVsStoreFallback::Topology, "a topology reject keeps its own class under the switch");
}

// S388 (H3): KYTY_PV_VSSTORE_PAIRS parser and filter.
void TestPairFilter() {
	PvVsStorePairFilter f;
	Check(ParsePvVsStorePairs(nullptr, f) && f.kind == PvVsStorePairFilter::Kind::All, "unset = all");
	Check(ParsePvVsStorePairs("all", f) && f.kind == PvVsStorePairFilter::Kind::All, "all");
	Check(PvVsStorePairAllowed(f, 1, 2, 3, 4, 99), "all admits everything");
	Check(ParsePvVsStorePairs("first=3", f) && f.kind == PvVsStorePairFilter::Kind::First && f.first_n == 3, "first=3");
	Check(PvVsStorePairAllowed(f, 1, 2, 3, 4, 0) && PvVsStorePairAllowed(f, 1, 2, 3, 4, 2), "first=3 admits ordinals 0..2");
	Check(!PvVsStorePairAllowed(f, 1, 2, 3, 4, 3), "first=3 refuses ordinal 3");
	Check(ParsePvVsStorePairs("first=0", f) && !PvVsStorePairAllowed(f, 1, 2, 3, 4, 0), "first=0 admits none");
	Check(ParsePvVsStorePairs("12:34,0xAbC:0xdef", f) && f.kind == PvVsStorePairFilter::Kind::List && f.pairs.size() == 2 &&
	          f.pairs[1].vs == 0xabc && f.pairs[1].ps == 0xdef,
	      "list of two pairs, decimal and hex");
	Check(PvVsStorePairAllowed(f, 12, 777, 34, 888, 0), "pair by ids");
	Check(PvVsStorePairAllowed(f, 5, 0xabc, 6, 0xdef, 0), "pair by hashes");
	Check(PvVsStorePairAllowed(f, 12, 777, 6, 0xdef, 0) == false, "the VS of one pair and the PS of another do not match");
	Check(!PvVsStorePairAllowed(f, 12, 777, 35, 888, 0), "right VS, wrong PS");
	Check(!ParsePvVsStorePairs("", f), "empty is invalid");
	Check(!ParsePvVsStorePairs("first=", f), "first= without a number");
	Check(!ParsePvVsStorePairs("first=x", f), "first=x");
	Check(!ParsePvVsStorePairs("12", f), "a token without a colon");
	Check(!ParsePvVsStorePairs("1:2,", f), "trailing comma");
	Check(!ParsePvVsStorePairs("1:z", f), "non-numeric side");
	Check(!ParsePvVsStorePairs("0x:1", f), "0x without digits");
	Check(!ParsePvVsStorePairs("99999999999999999999:1", f), "overflow");

	// S389: only=<hash>:<hash>,... matches guest hashes only.
	Check(ParsePvVsStorePairs("only=0xabc:0xdef,0x1:0x2", f) && f.kind == PvVsStorePairFilter::Kind::Only && f.pairs.size() == 2,
	      "only= with two hash pairs");
	Check(PvVsStorePairAllowed(f, 5, 0xabc, 6, 0xdef, 0) && PvVsStorePairAllowed(f, 5, 0x1, 6, 0x2, 7), "only= by hashes, any ordinal");
	Check(!PvVsStorePairAllowed(f, 0xabc, 77, 0xdef, 88, 0), "only= does not match program ids");
	Check(!PvVsStorePairAllowed(f, 5, 0xabc, 6, 0x2, 0), "only=: the VS of one pair and the PS of another do not match");
	Check(!ParsePvVsStorePairs("only=", f) && !ParsePvVsStorePairs("only=1", f) && !ParsePvVsStorePairs("only=1:2,", f), "only= malformed");
	// S389: range=a-b is inclusive over the candidate index.
	Check(ParsePvVsStorePairs("range=2-4", f) && f.kind == PvVsStorePairFilter::Kind::Range && f.range_a == 2 && f.range_b == 4, "range=2-4");
	Check(!PvVsStorePairAllowed(f, 1, 2, 3, 4, 1) && PvVsStorePairAllowed(f, 1, 2, 3, 4, 2) && PvVsStorePairAllowed(f, 1, 2, 3, 4, 3) &&
	          PvVsStorePairAllowed(f, 1, 2, 3, 4, 4) && !PvVsStorePairAllowed(f, 1, 2, 3, 4, 5),
	      "range=2-4 admits indices 2, 3, 4");
	Check(ParsePvVsStorePairs("range=0x10-0x10", f) && PvVsStorePairAllowed(f, 0, 0, 0, 0, 16) && !PvVsStorePairAllowed(f, 0, 0, 0, 0, 17),
	      "range=a-a admits one index (hex)");
	Check(!ParsePvVsStorePairs("range=4-2", f) && !ParsePvVsStorePairs("range=3", f) && !ParsePvVsStorePairs("range=-3", f) &&
	          !ParsePvVsStorePairs("range=3-", f) && !ParsePvVsStorePairs("range=a-b", f),
	      "range= malformed");
}

// S389: the guard's counter is read on the CPU, so its buffer must be host memory (MemoryUsage::Download).
void TestGuardMemory() {
	Check(kPvGuardMemory == PvGuardMemory::Download, "the guard's counter buffer is a Download buffer");
}

// S388 (H1): the guard's out-of-range rule (VS: skip the store and count; PS: magenta) and the switch parser.
void TestGuardRule() {
	bool guard = true;
	Check(ParsePvVsStoreGuard(nullptr, guard) && !guard, "unset = 0");
	Check(ParsePvVsStoreGuard("0", guard) && !guard, "0");
	Check(ParsePvVsStoreGuard("1", guard) && guard, "1");
	Check(!ParsePvVsStoreGuard("2", guard) && !ParsePvVsStoreGuard("", guard) && !ParsePvVsStoreGuard("on", guard), "other values are invalid");
	Check(!PvVsStoreRecordOutOfRange(100, 100, 12), "first record of the draw is in range");
	Check(!PvVsStoreRecordOutOfRange(111, 100, 12), "last record (base + records - 1) is in range");
	Check(PvVsStoreRecordOutOfRange(112, 100, 12), "base + records is out of range");
	Check(PvVsStoreRecordOutOfRange(99, 100, 12), "a record below base wraps and is out of range");
	Check(PvVsStoreRecordOutOfRange(100u + 0x80000000u, 100, 12), "a far record is out of range (unsigned compare)");
	// The same rule on the model's own record numbering: a vertex past the scanned range is out, one inside is not.
	PvVsStoreDraw d;
	d.base = 40; d.span = 6; d.first = 10; d.first_instance = 2; d.records = 18; // 3 instances
	Check(!PvVsStoreRecordOutOfRange(PvVsStoreVertexRecord(d.base, d.span, d.first, d.first_instance, 4, 15), d.base, d.records),
	      "last vertex of the last instance is in range");
	Check(PvVsStoreRecordOutOfRange(PvVsStoreVertexRecord(d.base, d.span, d.first, d.first_instance, 4, 16), d.base, d.records),
	      "one vertex past the scanned range of the last instance is out");
	Check(PvVsStoreRecordOutOfRange(PvVsStoreVertexRecord(d.base, d.span, d.first, d.first_instance, 2, 9), d.base, d.records),
	      "a vertex below the scanned range is out");
}

void TestIndexTailDecision() {
	const auto place_ok = [](const PvVsStorePlan&) { return true; };
	PvVsStoreRequest r;
	r.enabled = r.triangle_list = r.indexed = r.range_known = true;
	r.index_size = 2; r.range_first = 0; r.range_end = 12; r.index_count = 12; r.instance_count = 1; r.record_stride_vec4 = 2;
	auto d = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, place_ok, false, kCheckTail);
	Check(!d.use && d.fallback == PvVsStoreFallback::IndexTail, "index tail that does not fit stays on the prototype (pv_fallback_index_tail)");
	int asked = 0;
	(void)DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, [&](const PvVsStorePlan&) { asked++; return true; }, false, kCheckTail);
	Check(asked == 0, "no ring region is reserved for a draw rejected by the index tail");
	d = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, place_ok, true, kCheckTail);
	Check(d.use, "a fitting tail is taken");
}

void TestSlabs() {
	const auto never = [](uint64_t) { return false; };
	const auto done  = [](uint64_t) { return true; };
	PvVsStoreSlabs slabs(100, 2);
	auto a = slabs.Place(60, 5, never);
	Check(a.ok && a.created && a.slab == 0 && a.base == 0, "first region creates slab 0");
	auto b = slabs.Place(30, 5, never);
	Check(b.ok && !b.created && b.slab == 0 && b.base == 60, "same tick shares the slab, regions do not overlap");
	auto c = slabs.Place(60, 5, never);
	Check(c.ok && c.created && c.slab == 1 && c.base == 0, "a full slab moves to a new one");
	auto d = slabs.Place(60, 5, never);
	Check(!d.ok, "no free slab: the caller falls back");
	auto e = slabs.Place(10, 6, never);
	Check(!e.ok, "slabs of an unfinished tick are not recycled");
	auto f = slabs.Place(10, 7, done);
	Check(f.ok && !f.created && f.base == 0, "a finished tick recycles a slab from the start");
	Check(!slabs.Place(101, 8, done).ok, "region larger than a slab");
	Check(slabs.Size() == 2, "slab count capped");
	// A flush between Place() and the recording moves the draw to a newer tick.
	PvVsStoreSlabs touched(100, 1);
	auto t = touched.Place(10, 10, never);
	touched.Touch(t.slab, 11);
	const auto finished_10 = [](uint64_t tick) { return tick <= 10; };
	Check(!touched.Place(10, 12, finished_10).ok, "a slab touched in tick 11 is busy while only tick 10 finished");
	Check(touched.Place(10, 12, done).ok, "... and free once tick 11 has finished");
}

void TestFallbackClass() {
	using V = PvVsStoreVerdict;
	using F = PvVsStoreFallback;
	Check(ClassifyPvVsStoreFallback(V::Use, 1) == F::None && ClassifyPvVsStoreFallback(V::Disabled, 1) == F::None, "no fallback");
	Check(ClassifyPvVsStoreFallback(V::NotTriangleList, 1) == F::Topology &&
	          ClassifyPvVsStoreFallback(V::PrimitiveRestart, 3) == F::Topology,
	      "topology class");
	Check(ClassifyPvVsStoreFallback(V::NoRange, 1) == F::Range && ClassifyPvVsStoreFallback(V::TooLarge, 1) == F::Range,
	      "range class");
	Check(ClassifyPvVsStoreFallback(V::TooLarge, 4) == F::Instanced && ClassifyPvVsStoreFallback(V::RingFull, 2) == F::Instanced,
	      "instanced size failures");
	Check(ClassifyPvVsStoreFallback(V::RingFull, 1) == F::Ring, "ring class");
	Check(ClassifyPvVsStoreFallback(PvVsStoreInterfaceVerdict::Param31) == F::Param31 &&
	          ClassifyPvVsStoreFallback(PvVsStoreInterfaceVerdict::MissingOutput) == F::Interface,
	      "interface classes");
}

void TestPushRange() {
	Check(PvVsStorePushRangeBytes(128, false) == 128, "default pipelines keep the native push range");
	Check(PvVsStorePushRangeBytes(128, true) == 128 + kPvVsStorePushDwords * 4, "VS-store variants add the block");
	// S388: the guard adds one dword to the VS-store variants only.
	Check(PvVsStorePushRangeBytes(128, true, true) == 128 + 9 * 4, "guarded VS-store variants add the block and the record count");
	Check(PvVsStorePushRangeBytes(128, true, false) == 128 + 8 * 4, "unguarded VS-store variants keep 8 dwords");
	Check(PvVsStorePushRangeBytes(128, false, true) == 128, "the guard does not touch the other pipelines");
	Check(PvVsStorePushDwords(false) == 8 && PvVsStorePushDwords(true) == 9 && kPvPushRecords == 8, "dword counts");
}

void TestDecision() {
	using F = PvVsStoreFallback;
	const auto place_ok   = [](const PvVsStorePlan&) { return true; };
	const auto place_full = [](const PvVsStorePlan&) { return false; };
	const auto good       = [] {
        PvVsStoreRequest r;
        r.enabled = r.triangle_list = r.indexed = r.range_known = true;
        r.index_size = 2; r.range_first = 0; r.range_end = 12; r.index_count = 12; r.instance_count = 1; r.record_stride_vec4 = 2;
        return r;
	};
	auto d = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, good(), true, place_ok);
	Check(d.use && d.fallback == F::None && d.plan.records == 12, "a plain draw is taken");
	d = DecidePvVsStore(PvVsStoreInterfaceVerdict::Param31, good(), true, place_ok);
	Check(!d.use && d.fallback == F::Param31, "location 31 stays on the prototype");
	d = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, good(), false, place_ok);
	Check(!d.use && d.fallback == F::Range, "an index buffer that cannot be bound stays on the prototype");
	d = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, good(), true, place_full);
	Check(!d.use && d.fallback == F::Ring, "no ring room stays on the prototype");
	auto r = good();
	r.instance_count = 3;
	d = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, place_full);
	Check(!d.use && d.fallback == F::Instanced, "instanced draw without ring room is counted as instanced");
	r = good();
	r.triangle_list = false;
	d = DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, place_ok);
	Check(!d.use && d.fallback == F::Topology, "rect list stays on the prototype");
	// the ring is only asked for a draw the plan accepts
	int asked = 0;
	r = good();
	r.primitive_restart = true;
	(void)DecidePvVsStore(PvVsStoreInterfaceVerdict::Use, r, true, [&](const PvVsStorePlan&) { asked++; return true; });
	Check(asked == 0, "no region is reserved for a rejected draw");
}

} // namespace

int main() {
	TestModel();
	TestPlan();
	TestRing();
	TestSlots();
	TestInterface();
	TestCompact();
	TestCompactPatch();
	TestPushBlock();
	TestIndexBinding();
	TestIndexRange();
	TestIndexDescriptor();
	TestIndexBindingFits();
	TestGpuIdxMode();
	TestInstancedSwitch();
	TestPairFilter();
	TestGuardRule();
	TestGuardMemory();
	TestIndexTailDecision();
	TestSlabs();
	TestFallbackClass();
	TestDecision();
	TestPushRange();
	if (g_failures != 0) {
		std::fprintf(stderr, "per_vertex_vsstore_tests: %d failure(s)\n", g_failures);
		return EXIT_FAILURE;
	}
	std::printf("per_vertex_vsstore_tests: PASS\n");
	return EXIT_SUCCESS;
}
