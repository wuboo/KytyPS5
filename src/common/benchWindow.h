#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_BENCHWINDOW_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_BENCHWINDOW_H_

#include <string>
#include <string_view>

namespace Common {

// KYTY_BENCH_FULLSCREEN: "1" asks for a native full-screen window (macOS Game Mode needs one); unset, empty or
// anything else leaves the window as configured.
inline bool BenchFullscreenRequested(const char* value) {
	return value != nullptr && std::string_view(value) == "1";
}

// One log line for the benchmark window: the window flags as SDL reports them and, when known, the NSWindow
// occlusion bit (-1 = not available).
inline std::string FormatBenchWindowState(bool fullscreen, bool hidden, bool occluded, bool focus, int occlusion_visible,
                                          unsigned long long frames) {
	const bool onscreen = !hidden && !occluded && occlusion_visible != 0;
	return std::string("[bench-window] fullscreen=") + (fullscreen ? "1" : "0") + " visible=" + (hidden ? "0" : "1") +
	       " onscreen=" + (onscreen ? "1" : "0") + " focus=" + (focus ? "1" : "0") +
	       " ns_occlusion_visible=" + std::to_string(occlusion_visible) + " frames=" + std::to_string(frames);
}

} // namespace Common

#endif
