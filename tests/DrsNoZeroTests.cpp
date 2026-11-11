// S511 L2: the fast DrawRenderState init leaves unused vertex slots unconstructed. The color and
// depth targets moved to RenderExecutor::m_targets (upstream 7ae328b3).
// Mutant KYTY_DRS_MUTANT_ZERO_ALL zeroes them; the poison bytes are gone and vertex_alive stays 3.

#include "graphics/host_gpu/renderer/drsNoZero.h"

#include <cstdio>
#include <cstring>
#include <new>

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	using namespace Libs::Graphics;
	alignas(DrawRenderState) unsigned char raw[sizeof(DrawRenderState)];
	std::memset(raw, 0xA5, sizeof(raw));
	ShaderIndirectOffsets offsets {};
	offsets.vertex = 4;
	auto* state = new (raw) DrawRenderState(DrawRenderState::Fast {}, offsets, true);

	Check(state->vertex_alive == 1, "only vertex slot 0 is alive");
	Check(state->VertexStages().size() == 1, "VertexStages does not read slot 1");
	Check(state->vertex_info[0].buffers_num == 0, "vertex slot 0 was constructed");
	Check(state->vertex_info[0].stage.module == nullptr, "vertex slot 0 has no module yet");
	Check(!state->ps_active && !state->ps_input_info.object_primitive_id, "pixel defaults are off");
	Check(state->indirect_draw && state->indirect_offsets.vertex == 4, "indirect fields are copied");

	const auto* vert1  = reinterpret_cast<const unsigned char*>(&state->vertex_info[1]);
	Check(vert1[0] == 0xA5, "vertex slot 1 was not left unconstructed");

	state->~DrawRenderState();
	if (g_failures != 0) {
		std::fprintf(stderr, "DrsNoZeroTests: %d failed\n", g_failures);
		return 1;
	}
	std::puts("DrsNoZeroTests: OK");
	return 0;
}
