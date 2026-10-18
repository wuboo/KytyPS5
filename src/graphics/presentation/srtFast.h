#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_SRTFAST_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_SRTFAST_H_

// S425: KYTY_SRT_FAST=0|1|shadow (default 0 = the pre-S324 path bit for bit).
#include "graphics/shader/recompiler/ir/SrtMemo.h"

#include <cstdlib>

namespace Libs::Graphics::SrtFast {

inline ShaderRecompiler::IR::SrtFastMode Mode() {
	static const auto mode = ShaderRecompiler::IR::ParseSrtFastMode(std::getenv("KYTY_SRT_FAST"));
	return mode;
}

} // namespace Libs::Graphics::SrtFast

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_PRESENTATION_SRTFAST_H_ */
