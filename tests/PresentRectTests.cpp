#include "graphics/presentation/window/presentRect.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "PresentRectTests: failed: %s\n", message);
		std::exit(1);
	}
}

// S482: pillarbox on ultrawide, letterbox on tall windows, stretch when the flag is off.
void TestFit() {
	const auto wide = FitPresentRect(1920, 1080, 3440, 1440, true);
	Check(wide.h == 1440 && wide.y == 0 && wide.w == 2560 && wide.x == 440, "1920x1080 in 3440x1440: pillarbox 2560x1440 at x=440");
	const auto tall = FitPresentRect(1920, 1080, 1280, 1024, true);
	Check(tall.w == 1280 && tall.x == 0 && tall.h == 720 && tall.y == 152, "1920x1080 in 1280x1024: letterbox 1280x720 at y=152");
	const auto same = FitPresentRect(1920, 1080, 1280, 720, true);
	Check(same.x == 0 && same.y == 0 && same.w == 1280 && same.h == 720, "equal aspect fills the window");
	const auto off = FitPresentRect(1920, 1080, 3440, 1440, false);
	Check(off.x == 0 && off.y == 0 && off.w == 3440 && off.h == 1440, "flag off: whole destination");
	const auto zero = FitPresentRect(0, 1080, 3440, 1440, true);
	Check(zero.w == 3440 && zero.h == 1440, "degenerate source: whole destination");
}

void TestEnv() {
	Check(!KeepAspectEnabled(nullptr) && !KeepAspectEnabled("0") && !KeepAspectEnabled(""), "unset/0/empty = off");
	Check(KeepAspectEnabled("1"), "1 = on");
}

} // namespace

int main() {
	TestFit();
	TestEnv();
	std::puts("PresentRectTests: ok");
	return 0;
}
