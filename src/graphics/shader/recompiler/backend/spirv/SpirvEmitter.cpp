#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::ShaderRecompiler::Spirv {

namespace {
std::atomic<uint32_t> g_host_fragment_subgroup_size {32};
}

void SetHostFragmentSubgroupSize(uint32_t size) {
	g_host_fragment_subgroup_size.store(size, std::memory_order_relaxed);
}

bool PixelWave32OnWideHost(uint32_t wave_size) {
	const char* v   = std::getenv("KYTY_FIX_PS_WAVE32_MASK");
	const bool  fix = v != nullptr && v[0] == '1';
	return fix && wave_size == 32u && g_host_fragment_subgroup_size.load(std::memory_order_relaxed) >= 64u;
}

namespace {

[[noreturn]] void Fail(const IR::Program& program, const char* reason) {
	EXIT("SPIR-V validation failed: hash=0x%016" PRIx64 " stage=%u reason=%s\n",
	     program.shader_hash, static_cast<unsigned>(program.stage), reason);
	std::abort();
}

} // namespace

std::vector<uint32_t> EmitProgram(IR::Program& program, ShaderStageInputInfo input_info,
                                  uint32_t push_data_start_dword, const Profile& profile,
                                  bool emit_debug_info, const VertexCaptureInfo* vertex_capture,
                                  const PvStoreInfo* pv_store, uint32_t push_data_ceiling) {
	using namespace Emitter;

	if (!program.srt_plan_complete || !program.resource_tracking_complete ||
	    !program.shader_info_complete) {
		Fail(program, "SPIR-V emitter requires a fully planned native shader program");
	}
	IR::ValidateProgram(program);
	program.bindings = {.push_data_start_dword = push_data_start_dword};
	EmitterState state(program, input_info, profile);
	state.vertex_capture = vertex_capture;
	state.pv_store = pv_store;
	if (pv_store != nullptr) {
		const bool vertex = pv_store->role == PvStoreInfo::Role::Vertex;
		if (vertex_capture != nullptr ||
		    program.stage != (vertex ? ShaderType::Vertex : ShaderType::Pixel) ||
		    (vertex ? input_info.vertex == nullptr : input_info.pixel == nullptr)) {
			Fail(program, "invalid PerVertex VS-store stage or record layout");
		}
	}
	if (emit_debug_info) {
		char name[64];
		std::snprintf(name, sizeof(name), "rdna2-%016" PRIx64 "-block-byte-pc-plus-one",
		              program.shader_hash);
		state.debug_source = state.builder.AddString(name);
		state.builder.EnableNames();
	}
	const auto* workgroup = ShaderWorkgroupInput(program.stage, input_info);
	state.lane_count =
	    workgroup != nullptr && program.wave_size == 64u && workgroup->host_subgroup_size == 32u
	        ? 2u
	        : 1u;
	state.pixel_wave32_wide_host = program.stage == ShaderType::Pixel && PixelWave32OnWideHost(program.wave_size);
	if (vertex_capture != nullptr) {
		if (program.stage != ShaderType::Vertex || input_info.vertex == nullptr ||
		    (vertex_capture->host_subgroup_size != 32 &&
		     vertex_capture->host_subgroup_size != 64)) {
			Fail(program, "invalid vertex capture stage or host subgroup size");
		}
		state.lane_count =
		    program.wave_size == 64 && vertex_capture->host_subgroup_size == 32 ? 2u : 1u;
	}
	DefineModule(state, push_data_ceiling);
	EmitProgram(state);
	state.builder.AddEntryPoint(vertex_capture != nullptr
	                                ? spv::ExecutionModelGLCompute
	                                : ExecutionModelForStage(state.program.stage),
	                            state.main_func, "main", state.interface_variables);

	return state.builder.Build();
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv
