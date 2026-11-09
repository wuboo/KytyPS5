#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CLEANWALK_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CLEANWALK_H_

// S507 K2 (KYTY_RB_CLEAN_WALK=0|1|shadow, unset = 0).
// A compiled SRT walker load is `mov eax, [r10]`. When the 4 KiB page is protected because the GPU wrote
// other bytes, the load faults and the handler drains the queue. Garden flips 5000-5030: the draw_shaders
// faults are dirty=0, page_dirty=1 and stale==fresh, so the dword is already in the backing.
//   0       the fault drains, as before
//   1       a load whose bytes are not GPU-dirty and not in flight completes from the backing
//   shadow  those loads still drain; the backing copy is compared with the bytes after the drain
// The switch is not codegen: the served bytes are the drained bytes (shadow ne stays 0), so SPIR-V does not change.
// Mutant KYTY_CLEANWALK_MUTANT_SERVE_DIRTY serves a range the byte tracker marks GPU-dirty.

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::CleanWalk {

enum class Mode : uint8_t { Off, On, Shadow };
enum class Action : uint8_t { Drain, Serve, Compare };

inline Mode Parse(const char* text) {
	if (text == nullptr) {
		return Mode::Off;
	}
	if (std::strcmp(text, "1") == 0) {
		return Mode::On;
	}
	if (std::strcmp(text, "shadow") == 0) {
		return Mode::Shadow;
	}
	return Mode::Off;
}

inline Mode EnvMode() {
	static const Mode mode = Parse(std::getenv("KYTY_RB_CLEAN_WALK"));
	return mode;
}

// busy: the byte tracker could not be queried. dirty / inflight: the backing may not match a drain.
constexpr Action Decide(Mode mode, bool dirty, bool inflight, bool busy) {
	if (mode == Mode::Off || busy) {
		return Action::Drain;
	}
#if defined(KYTY_CLEANWALK_MUTANT_SERVE_DIRTY)
	dirty    = false;
	inflight = false;
#endif
	if (dirty || inflight) {
		return Action::Drain;
	}
	return mode == Mode::Shadow ? Action::Compare : Action::Serve;
}

inline bool Equal(const void* a, const void* b, uint64_t size) {
	return std::memcmp(a, b, static_cast<size_t>(size)) == 0;
}

} // namespace Libs::Graphics::CleanWalk

#endif
