// S134: KYTY_PV_UNIQUE=1 (capture per unique vertex), CPU side: the draw planner, the record model
// that per_vertex_unpack.comp (map mode) and the replay shaders must agree on, the "dropped" flag of
// the record range and the separate transformation signature. The GPU proof (images ON == OFF, GPU
// map == model) is `per_vertex_gpu_execution_tests --unique-only`.
// Mutants (KYTY_PVU_MUTANT_*, each must fail): NO_INSTANCE, NO_FIRST, NO_RATIO, WRITES_OK,
// DROPPED_OK (perVertexUnique.h) and NO_DROPPED (perVertexTransform.cpp).

#include "graphics/host_gpu/renderer/perVertexTransform.h"
#include "graphics/host_gpu/renderer/perVertexUnique.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace Libs::Graphics;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

namespace {
PerVertexUniqueRequest Request(bool known, uint64_t first, uint64_t end, uint32_t corners, uint32_t instances) {
	PerVertexUniqueRequest r;
	r.enabled        = true;
	r.range_known    = known;
	r.range_first    = first;
	r.range_end      = end;
	r.index_count    = corners;
	r.instance_count = instances;
	return r;
}
} // namespace

int main() {
	// ---- record model: decode (record -> vertex, instance) must invert the corner mapping ----
	{
		const uint32_t first = 7, count = 5, first_instance = 3, instances = 3;
		bool           all   = true;
		for (uint32_t inst = 0; inst < instances; ++inst) {
			for (uint32_t v = first; v < first + count; ++v) {
				const uint32_t rec = PerVertexUniqueRecord(first_instance + inst, first_instance, v, first, count);
				all &= rec < count * instances;
				all &= first + rec % count == v;
				all &= first_instance + rec / count == first_instance + inst;
			}
		}
		Check(all, "record model: decode does not invert (vertex - first, instance - first_instance)");
		Check(PerVertexUniqueRecord(4, 3, 9, 7, 5) == 1 * 5 + 2, "record model: instance 1, vertex first+2 -> 7");
	}

	// ---- planner ----
	{
		auto r = Request(true, 3, 12, 24, 2); // 9 unique of 24 corners, 2 instances
		auto p = PlanPerVertexUnique(r);
		Check(p.Use() && p.first == 3 && p.count == 9 && p.records == 18, "planner: shared grid x2 instances takes the unique path (first 3, N 9, 18 records)");
		auto off = r;
		off.enabled = false;
		Check(PlanPerVertexUnique(off).verdict == PerVertexUniqueVerdict::Disabled, "planner: switch off -> Disabled");
		auto writes = r;
		writes.capture_has_writes = true;
		Check(PlanPerVertexUnique(writes).verdict == PerVertexUniqueVerdict::HasWrites, "planner: capture with buffer writes falls back");
		auto unknown = r;
		unknown.range_known = false;
		Check(PlanPerVertexUnique(unknown).verdict == PerVertexUniqueVerdict::NoRange, "planner: unknown range falls back");
		auto dropped = r;
		dropped.range_dropped = true;
		Check(PlanPerVertexUnique(dropped).verdict == PerVertexUniqueVerdict::Dropped, "planner: dropped corners fall back");
		Check(PlanPerVertexUnique(Request(true, 4, 4, 24, 2)).verdict == PerVertexUniqueVerdict::Empty, "planner: empty range falls back");
		Check(PlanPerVertexUnique(Request(true, 0, 10, 0, 2)).verdict == PerVertexUniqueVerdict::Empty, "planner: no corners falls back");
		// 0.7 boundary: records > 0.7 * corners falls back, == stays.
		Check(PlanPerVertexUnique(Request(true, 0, 8, 10, 1)).verdict == PerVertexUniqueVerdict::TooManyRecords, "planner: 8 of 10 corners (80%) falls back");
		Check(PlanPerVertexUnique(Request(true, 0, 7, 10, 1)).Use(), "planner: 7 of 10 corners (70%) stays unique");
		Check(PlanPerVertexUnique(Request(true, 0, 6, 6, 2), 1.0).Use(), "planner: threshold 1.0 keeps a draw without sharing");
		Check(PlanPerVertexUnique(Request(true, 0, 6, 6, 2)).verdict == PerVertexUniqueVerdict::TooManyRecords, "planner: a draw without sharing falls back by default");
	}

	// ---- the record range reports dropped corners ----
	{
		const uint16_t idx_ok[3]  = {0, 1, 2};
		const uint16_t idx_bad[3] = {0, 1, 99};
		PerVertexRecordRange range;
		Check(ComputePerVertexRecordRange(true, idx_ok, sizeof(idx_ok), 2, 3, 0, 0, 10, range) && !range.dropped && range.first == 0 && range.end == 3, "range: indices inside the buffer are not dropped");
		Check(ComputePerVertexRecordRange(true, idx_bad, sizeof(idx_bad), 2, 3, 0, 0, 10, range) && range.dropped && range.end == 2, "range: an index past num_records is dropped");
		Check(ComputePerVertexRecordRange(true, idx_ok, sizeof(idx_ok), 2, 3, 0, -1, 10, range) && range.dropped, "range: a negative adjusted index is dropped");
		Check(ComputePerVertexRecordRange(false, nullptr, 0, 0, 6, 2, 0, 5, range) && range.dropped && range.first == 2 && range.end == 5, "range: first_vertex + count past num_records is dropped");
		Check(ComputePerVertexRecordRange(false, nullptr, 0, 0, 3, 2, 0, 5, range) && !range.dropped, "range: non-indexed inside the buffer is not dropped");
	}

	// ---- separate transformation signature / cache files ----
	{
		const auto classic = GetPerVertexTransformSignature();
		const auto unique  = GetPerVertexTransformSignature(true);
		Check(classic != unique, "signature: unique replay shaders have their own signature");
		Check(unique.size() > classic.size() && unique.compare(0, classic.size(), classic) == 0, "signature: unique = classic + suffix (classic unchanged)");
		Check(classic.find("pvu") == std::string::npos, "signature: the classic signature carries no unique marker");
	}

	if (g_failures != 0) return EXIT_FAILURE;
	std::puts("per_vertex_unique_tests: OK");
	return 0;
}
