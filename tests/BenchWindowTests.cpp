#include "common/benchWindow.h"
#include "common/knownEnv.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "BenchWindowTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

} // namespace

int main() {
	// KYTY_BENCH_FULLSCREEN: only "1" turns it on; default OFF.
	Check(Common::BenchFullscreenRequested("1"), "1 not parsed as on");
	Check(!Common::BenchFullscreenRequested(nullptr), "unset is on");
	Check(!Common::BenchFullscreenRequested(""), "empty is on");
	Check(!Common::BenchFullscreenRequested("0"), "0 is on");
	Check(!Common::BenchFullscreenRequested("true"), "true accepted (only 1 is a value)");

	// The variable is known, so KYTY_ENV_STRICT=1 does not abort on it.
	const auto* const end = std::end(Common::kKnownKytyEnv);
	Check(std::find(std::begin(Common::kKnownKytyEnv), end, "KYTY_BENCH_FULLSCREEN") != end, "KYTY_BENCH_FULLSCREEN unknown to env audit");

	// Log line fields.
	const auto on = Common::FormatBenchWindowState(true, false, false, true, 1, 600);
	Check(on == "[bench-window] fullscreen=1 visible=1 onscreen=1 focus=1 ns_occlusion_visible=1 frames=600", "state line wrong");
	Check(Common::FormatBenchWindowState(true, false, true, true, -1, 0).find("onscreen=0") != std::string::npos, "occluded reported onscreen");
	Check(Common::FormatBenchWindowState(true, true, false, true, -1, 0).find("visible=0 onscreen=0") != std::string::npos, "hidden reported visible");
	Check(Common::FormatBenchWindowState(true, false, false, true, 0, 0).find("onscreen=0") != std::string::npos, "NS occlusion ignored");
	Check(Common::FormatBenchWindowState(false, false, false, true, -1, 0).find("onscreen=1") != std::string::npos, "unknown NS occlusion blocks onscreen");
	std::puts("BenchWindowTests: ok");
	return 0;
}
