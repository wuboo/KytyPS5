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

// S484: bars stay valid only for the very same rectangle and swapchain extent.
void TestKey() {
	const auto a = FitPresentRect(1920, 1080, 3440, 1440, true);
	const auto b = FitPresentRect(1920, 1080, 1280, 1024, true);
	Check(PresentRectKey(a, 3440, 1440) != 0, "key is never 0");
	Check(PresentRectKey(a, 3440, 1440) == PresentRectKey(a, 3440, 1440), "same rectangle: same key");
	Check(PresentRectKey(a, 3440, 1440) != PresentRectKey(b, 1280, 1024), "different rectangle/extent: different key");
	PresentRect moved = a;
	moved.x += 1;
	Check(PresentRectKey(a, 3440, 1440) != PresentRectKey(moved, 3440, 1440), "shifted rectangle: different key");
	Check(PresentRectKey(a, 3440, 1440) != PresentRectKey(a, 3440, 1441), "same rectangle in a different extent: different key");
}

} // namespace

int main() {
	TestFit();
	TestEnv();
	TestKey();
	std::puts("PresentRectTests: ok");
	return 0;
}
