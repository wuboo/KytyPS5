#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_SRTFAST_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_SRTFAST_H_

// S425: KYTY_SRT_FAST=0|1|shadow (default 0 = the pre-S324 path bit for bit).
#include "graphics/shader/recompiler/ir/SrtMemo.h"

#include <cstdlib>

namespace Libs::Graphics::SrtFast {

inline ShaderRecompiler::IR::SrtFastMode Mode() {
	static const auto mode = [] {
		// S477 D2 (KYTY_DIAG_RDHAZARD): the oracle copies the memo of every walk, so the walk must record it (shadow = walk anyway, same result)
		const char* hazard = std::getenv("KYTY_DIAG_RDHAZARD");
		if (std::getenv("KYTY_SRT_FAST") == nullptr && hazard != nullptr && hazard[0] != '\0' && !(hazard[0] == '0' && hazard[1] == '\0')) {
			return ShaderRecompiler::IR::SrtFastMode::Shadow;
		}
		return ShaderRecompiler::IR::ParseSrtFastMode(std::getenv("KYTY_SRT_FAST"));
	}();
	return mode;
}

} // namespace Libs::Graphics::SrtFast

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_SRTFAST_H_ */
