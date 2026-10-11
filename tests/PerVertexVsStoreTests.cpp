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

} // namespace

int main() {
	TestModel();
	TestPlan();
	TestRing();
	TestSlots();
	if (g_failures != 0) {
		std::fprintf(stderr, "per_vertex_vsstore_tests: %d failure(s)\n", g_failures);
		return EXIT_FAILURE;
	}
	std::printf("per_vertex_vsstore_tests: PASS\n");
	return EXIT_SUCCESS;
}
