#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PERVERTEXUNIQUE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PERVERTEXUNIQUE_H_

// S134: KYTY_PV_UNIQUE=1 runs the prototype capture once per UNIQUE vertex instead of once per
// index. Pure helpers (no Vulkan, no globals) shared by the renderer, the CPU tests and the GPU
// harness: the decision whether a draw may use the unique path, and the record numbering that
// per_vertex_unpack.comp (map mode) and the capture/replay shaders must agree on.

#include <cstdint>

namespace Libs::Graphics {

// Unique records are the vertices [first, first + count) of the vertex buffer, once per instance:
// record = (instance - first_instance) * count + (vertex - first). Instance-major, so the unique
// unpack (non-indexed, first_vertex = first, count * instances invocations) produces them in order.
constexpr uint32_t PerVertexUniqueRecord(uint32_t instance, uint32_t first_instance, uint32_t vertex,
                                         uint32_t first, uint32_t count) {
#if defined(KYTY_PVU_MUTANT_NO_INSTANCE)
	(void)instance;
	(void)first_instance;
	return vertex - first;
#elif defined(KYTY_PVU_MUTANT_NO_FIRST)
	(void)first;
	return (instance - first_instance) * count + vertex;
#else
	return (instance - first_instance) * count + (vertex - first);
#endif
}

struct PerVertexUniqueRequest {
	bool     enabled           = false; // KYTY_PV_UNIQUE=1
	bool     capture_has_writes = false; // the VS writes buffers: its invocations are observable
	bool     range_known       = false; // ComputePerVertexRecordRange could derive the range
	bool     range_dropped     = false; // some index/vertex fell outside the vertex buffer
	uint64_t range_first       = 0;
	uint64_t range_end         = 0;     // [first, end) records the draw reads
	uint32_t index_count       = 0;     // corners per instance (what the replay draws)
	uint32_t instance_count    = 0;
};

enum class PerVertexUniqueVerdict : uint32_t {
	Disabled,     // switch off: nothing counted, the classic path
	Unique,       // run the unique path
	NoRange,      // fallback: the record range is not known
	Dropped,      // fallback: a corner reads outside the vertex buffer (classic path renders vertex 0)
	HasWrites,    // fallback: the VS has side-effecting buffer writes
	TooManyRecords, // fallback: unique records > 0.7 of the corners (little to gain)
	Empty,        // fallback: no record at all
};

struct PerVertexUniquePlan {
	PerVertexUniqueVerdict verdict = PerVertexUniqueVerdict::Disabled;
	uint32_t               first   = 0; // first unique vertex
	uint32_t               count   = 0; // unique vertices per instance (N)
	uint64_t               records = 0; // N * instances: capture invocations
	[[nodiscard]] bool     Use() const { return verdict == PerVertexUniqueVerdict::Unique; }
};

// `max_ratio` is the largest records / corners share that still takes the unique path.
constexpr PerVertexUniquePlan PlanPerVertexUnique(const PerVertexUniqueRequest& request,
                                                  double max_ratio = 0.7) {
	using V = PerVertexUniqueVerdict;
	PerVertexUniquePlan plan;
	if (!request.enabled) return plan;
#if !defined(KYTY_PVU_MUTANT_WRITES_OK)
	if (request.capture_has_writes) {
		plan.verdict = V::HasWrites;
		return plan;
	}
#endif
	if (!request.range_known) {
		plan.verdict = V::NoRange;
		return plan;
	}
#if !defined(KYTY_PVU_MUTANT_DROPPED_OK)
	if (request.range_dropped) {
		plan.verdict = V::Dropped;
		return plan;
	}
#endif
	if (request.range_end <= request.range_first || request.index_count == 0 ||
	    request.instance_count == 0 || request.range_end > (uint64_t {1} << 32u)) {
		plan.verdict = V::Empty;
		return plan;
	}
	const uint64_t unique  = request.range_end - request.range_first;
	const uint64_t records = unique * request.instance_count;
	const uint64_t corners = uint64_t {request.index_count} * request.instance_count;
#if defined(KYTY_PVU_MUTANT_NO_RATIO)
	(void)max_ratio;
	(void)corners;
#else
	if (static_cast<double>(records) > max_ratio * static_cast<double>(corners)) {
		plan.verdict = V::TooManyRecords;
		return plan;
	}
#endif
	plan.verdict = V::Unique;
	plan.first   = static_cast<uint32_t>(request.range_first);
	plan.count   = static_cast<uint32_t>(unique);
	plan.records = records;
	return plan;
}

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_PERVERTEXUNIQUE_H_ */
