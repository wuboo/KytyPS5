#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_S538DIAG_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_S538DIAG_H_

// S538 DIAG: where the card shader 5e7524f281e7d196 takes the T# of image slot 2 from.
//
// KYTY_DIAG_S538=1: the CP walk of the card pixel shader records the range of every guest load (capture_reads); every
// draw of the shader between flips 3300 and 4480 logs, when anything changed (and on the first draw of every 10th
// flip), the user data, the T# the renderer binds, the guest bytes at the T#'s place now, the GPU state of that place
// and the memory events in it ([s538-ts], [s538-loc], [s538-ev]). A host thread polls the T#'s place and the small
// (pointer) loads of the walk every 0.5 ms and logs every change of their bytes ([s538-w]).
// KYTY_DIAG_S538_FORCE=1: a causal control. The card draw's slot-2 T# is replaced by the last T# of that slot whose
// guest mip 0 hashed to the particle asset 3b467065b61988b2, when the current one hashes to something else.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler::S538 {

inline constexpr uint64_t kGoodMip0 = 0x3b467065b61988b2ull;

inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_S538");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return on;
}

inline bool Force() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_S538_FORCE");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return on;
}

} // namespace Libs::Graphics::ShaderRecompiler::S538

#endif /* KYTY_GRAPHICS_SHADER_RECOMPILER_S538DIAG_H_ */
