#include "graphics/shader/recompiler/rtMiss.h"

#include <cstdio>
#include <cstdlib>

namespace SR = Libs::Graphics::ShaderRecompiler;

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "RtMissSwitchTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

} // namespace

int main() {
	unsetenv("KYTY_RT_MISS");
	Check(!SR::RtMissEnabled(), "off without the variable");
	setenv("KYTY_RT_MISS", "", 1);
	Check(!SR::RtMissEnabled(), "off for an empty value");
	setenv("KYTY_RT_MISS", "0", 1);
	Check(!SR::RtMissEnabled(), "off for 0");
	setenv("KYTY_RT_MISS", "1", 1);
	Check(SR::RtMissEnabled(), "KYTY_RT_MISS=1 not seen (read per call, so a test can switch it)");
	unsetenv("KYTY_RT_MISS");
	Check(!SR::RtMissEnabled(), "still on after unset");
	return 0;
}
