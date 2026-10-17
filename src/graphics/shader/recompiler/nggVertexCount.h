#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_NGGVERTEXCOUNT_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_NGGVERTEXCOUNT_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler {

// Apple bring-up: a vertex function has no SubgroupLocalInvocationId, so the lane ordinal is made up from
// gl_VertexIndex & 31 (KYTY_VS_LANE_FROM_INDEX=0 restores the builtin). Read once; one answer for the
// translator (NGG s3 prologue) and the emitter (lane id).
inline bool VertexLanesFromIndex() {
#if defined(__APPLE__)
	static const bool value = [] {
		const char* env = std::getenv("KYTY_VS_LANE_FROM_INDEX");
		return env == nullptr || env[0] != '0';
	}();
	return value;
#else
	return false;
#endif
}

// What the NGG vertex count in s3 of a vertex shader counts. With real lanes the number of active
// invocations (upstream cf77fb43); with lanes made up from the vertex index a lane ordinal can exceed
// that count and the guest-rebuilt EXEC would switch the vertex off, so the full wave is used there.
// `mode` is KYTY_NGG_VCOUNT (nullptr, "wave" or "active").
inline bool NggVertexCountFromActive(bool lanes_from_index, const char* mode) {
#if defined(KYTY_NGG_MUTANT_ACTIVE)
	(void)lanes_from_index;
	(void)mode;
	return true;
#else
	if (mode != nullptr && std::strcmp(mode, "active") == 0) {
		return true;
	}
	if (mode != nullptr && std::strcmp(mode, "wave") == 0) {
		return false;
	}
	return !lanes_from_index;
#endif
}

// Reads KYTY_NGG_VCOUNT on every call (cheap, tests switch it).
inline bool NggVertexCountFromActive(bool lanes_from_index) {
	return NggVertexCountFromActive(lanes_from_index, std::getenv("KYTY_NGG_VCOUNT"));
}

// s3 the translator hands a vertex shader: bit 28 plus the vertex count, `active_count` invocations of a
// `wave_size` wide wave being live.
inline uint32_t NggPackedS3(bool count_active, uint32_t wave_size, uint32_t active_count) {
	return (1u << 28u) | (count_active ? active_count : wave_size);
}

// The guest rebuilds EXEC as `s_sub_u32 vcc_lo, 0, s3; s_lshr_b64 exec, -1, vcc_lo`; the host then
// tests one bit of it with the lane ordinal (Translator::ThreadBit).
inline bool NggExecLaneActive(uint32_t s3, uint32_t lane) {
	const uint64_t exec = ~uint64_t{0} >> ((0u - s3) & 63u);
	return ((exec >> lane) & 1u) != 0;
}

} // namespace Libs::Graphics::ShaderRecompiler

#endif
