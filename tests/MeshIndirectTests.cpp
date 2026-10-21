// S472: KYTY_INDIRECT_GPU_MESH (meshIndirect.h): the plan of a mesh indirect draw computed from the guest args (the prepare compute and the mesh
// shader do the same arithmetic) against the CPU path it replaces (renderDraw.cpp: InputPrimitiveCount, groups, ResolveOnCpu, emit offsets).
// Mutants (each must fail): NO_CLAMP (index count not clamped to INDEX_BUFFER_SIZE), NATIVE_OFFSETS (native fetch keeps the args' vertex offset
// and first instance), GROUPS_FLOOR (the group count is rounded down), DEFAULT_ON (the mode is on when KYTY_INDIRECT_GPU_MESH is unset).

#include "graphics/host_gpu/renderer/indirectGpu.h"
#include "graphics/host_gpu/renderer/meshIndirect.h"

#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::MeshIndirect;

int g_failures = 0;
void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "MeshIndirectTests: failed: %s\n", what);
		++g_failures;
	}
}

// The CPU path, as renderDraw.cpp / graphicsRun.cpp do it today (written out independently of MakePlan).
struct CpuDraw {
	bool     skipped = false; // primitives == 0 || instance_count == 0
	uint32_t groups  = 0;
	uint32_t index_count = 0;
};
CpuDraw CpuPath(const Args& args, const MeshInfo& mesh, uint32_t index_buffer_size) {
	CpuDraw d;
	d.index_count = index_buffer_size != 0 ? (args.index_count_per_instance < index_buffer_size ? args.index_count_per_instance : index_buffer_size)
	                                       : args.index_count_per_instance;
	const uint32_t primitives = d.index_count < mesh.primitive_size ? 0u : (d.index_count - mesh.primitive_size) / mesh.primitive_step + 1u;
	if (primitives == 0 || args.instance_count == 0) {
		d.skipped = true;
		return d;
	}
	d.groups = (primitives - 1u) / mesh.primitives_per_group + 1u;
	return d;
}

constexpr Limits kLimits {65535, 65535, 4194304};

void TestParseMode() {
	Check(ParseMode(nullptr) == Mode::Off, "unset = off");
	Check(ParseMode("0") == Mode::Off, "0 = off");
	Check(ParseMode("1") == Mode::On, "1 = on");
	Check(ParseMode("shadow") == Mode::Shadow, "shadow");
	Check(ParseMode("2") == Mode::Off && ParseMode("") == Mode::Off && ParseMode("on") == Mode::Off, "an unknown value is off");
}

void TestPlanMatchesCpu() {
	const MeshInfo triangles {3, 3, 64, false};
	const MeshInfo strip {3, 1, 126, false};
	for (const MeshInfo& mesh : {triangles, strip}) {
		for (uint32_t index_count : {0u, 1u, 2u, 3u, 4u, 191u, 192u, 193u, 600u, 601u, 100000u}) {
			for (uint32_t instances : {0u, 1u, 7u}) {
				const Args args {index_count, instances, 5, 0xfffffffbu, 3};
				const auto plan = MakePlan(args, mesh, 600, false, kLimits);
				const auto cpu  = CpuPath(args, mesh, 600);
				Check(plan.index_count == cpu.index_count, "index count is clamped to INDEX_BUFFER_SIZE like the CPU path");
				if (cpu.skipped) {
					Check(plan.groups_x == 0 && plan.groups_y == 0 && plan.groups_z == 0 && plan.violation == 0, "a skipped draw has no groups");
				} else {
					Check(plan.groups_x == cpu.groups && plan.groups_y == instances && plan.groups_z == 1, "groups as the CPU computes them");
				}
				Check(plan.vertex_offset == 0xfffffffbu && plan.first_instance == 3 && plan.start_index == 5, "offsets come from the args");
			}
		}
	}
}

void TestGroupsBoundary() {
	const MeshInfo mesh {3, 3, 64, false};
	// 64 primitives = 192 indices fit one group, 195 need two.
	Check(MakePlan({192, 1, 0, 0, 0}, mesh, 1000, false, kLimits).groups_x == 1, "a full group is one group");
	Check(MakePlan({195, 1, 0, 0, 0}, mesh, 1000, false, kLimits).groups_x == 2, "one primitive more is a second group");
	Check(MakePlan({3, 1, 0, 0, 0}, mesh, 1000, false, kLimits).groups_x == 1, "one primitive is one group");
	Check(MakePlan({2, 1, 0, 0, 0}, mesh, 1000, false, kLimits).groups_x == 0, "less than one primitive is no group");
}

void TestNativeFetch() {
	const MeshInfo mesh {3, 3, 64, false};
	const auto     plan = MakePlan({300, 2, 9, 123, 4}, mesh, 1000, true, kLimits);
	Check(plan.vertex_offset == 0 && plan.first_instance == 0, "native vertex fetch: offsets 0 (the SGPRs carry them)");
	Check(plan.start_index == 9, "the start index is kept");
	const auto non_native = MakePlan({300, 2, 9, 123, 4}, mesh, 1000, false, kLimits);
	Check(non_native.vertex_offset == 123 && non_native.first_instance == 4, "rewritten fetch: offsets from the args");
}

void TestLimits() {
	const MeshInfo mesh {3, 3, 64, false};
	const Limits   small {10, 4, 20};
	Check(MakePlan({3 * 64 * 10, 1, 0, 0, 0}, mesh, 100000, false, small).violation == 0, "10 groups fit max_x = 10");
	auto plan = MakePlan({3 * 64 * 11, 1, 0, 0, 0}, mesh, 100000, false, small);
	Check(plan.violation == 1 && plan.groups_x == 0 && plan.groups_y == 0, "11 groups exceed max_x: nothing is drawn");
	plan = MakePlan({3 * 64 * 2, 5, 0, 0, 0}, mesh, 100000, false, small);
	Check(plan.violation == 1, "5 instances exceed max_y");
	plan = MakePlan({3 * 64 * 6, 4, 0, 0, 0}, mesh, 100000, false, small);
	Check(plan.violation == 1, "6 x 4 groups exceed the total 20");
	plan = MakePlan({3 * 64 * 5, 4, 0, 0, 0}, mesh, 100000, false, small);
	Check(plan.violation == 0 && plan.groups_x == 5 && plan.groups_y == 4, "5 x 4 groups = the total 20 fit");
	const Limits big {0xffffffffu, 0xffffffffu, 0xffffffffu};
	plan = MakePlan({3 * 64 * 70000, 70000, 0, 0, 0}, mesh, 0, false, big);
	Check(plan.violation == 1, "the product is formed in 64 bits");
}

void TestIndexAddress() {
	Check(IndexAddress(0x100000000ull, 10, 2) == 0x100000014ull, "16-bit indices");
	Check(IndexAddress(0xfffffff0ull, 8, 4) == 0x100000010ull, "the sum carries into the high dword");
	Check(PushElementWord(2, false) == 0x80000002u && PushElementWord(4, true) == 0xC0000004u, "push dword 3: args flag, native flag, element bytes");
}

void TestQualifyHost() {
	IndirectGpu::HostPath host;
	host.mesh = true;
	Check(IndirectGpu::QualifyHost(host) == IndirectGpu::Reason::Mesh, "a mesh host path stays on the CPU unless the GPU path is on");
	host.mesh_gpu = true;
	Check(IndirectGpu::QualifyHost(host) == IndirectGpu::Reason::Eligible, "a mesh host path the GPU can take is eligible");
	host.restart = true;
	Check(IndirectGpu::QualifyHost(host) == IndirectGpu::Reason::PrimitiveRestart, "the other host reasons still apply to a mesh draw");
	host = {};
	host.mesh = true;
	host.native_fetch = true;
	Check(IndirectGpu::QualifyHost(host) == IndirectGpu::Reason::Mesh, "native fetch, mesh not on the GPU: mesh");
	host.mesh_gpu = true;
	Check(IndirectGpu::QualifyHost(host) == IndirectGpu::Reason::Eligible, "a mesh draw on the GPU path zeroes the offsets itself: native fetch is no obstacle");
	host = {};
	host.native_fetch = true;
	host.mesh_gpu = true;
	Check(IndirectGpu::QualifyHost(host) == IndirectGpu::Reason::NativeFetch, "a non-mesh draw with native fetch stays on the CPU");
	host = {};
	host.mesh_gpu = true;
	Check(IndirectGpu::QualifyHost(host) == IndirectGpu::Reason::Eligible, "mesh_gpu alone changes nothing for a non-mesh draw");
	Check(MeshCanGoGpu({3, 3, 64, false}) && !MeshCanGoGpu({3, 3, 64, true}) && !MeshCanGoGpu({3, 3, 0, false}), "fast launch / no group size: CPU");
}

} // namespace

int main() {
	TestParseMode();
	TestPlanMatchesCpu();
	TestGroupsBoundary();
	TestNativeFetch();
	TestLimits();
	TestIndexAddress();
	TestQualifyHost();
	if (g_failures != 0) {
		std::fprintf(stderr, "MeshIndirectTests: %d failure(s)\n", g_failures);
		return 1;
	}
	return 0;
}
