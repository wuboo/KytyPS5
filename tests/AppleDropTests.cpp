#include "graphics/host_gpu/renderer/appleDrop.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "AppleDropTests: failed: %s\n", message);
		std::exit(1);
	}
}

constexpr AppleDropHost kNone {.depth_clamp = false, .depth_bounds = false, .color_write_enable = false};
constexpr AppleDropHost kFull {};

// S129: drop_depth_clamp counts a draw only when the guest disabled Z clip and the host has no clamp.
void TestDepthClamp() {
	AppleDropDraw clip_off {.z_clip_enabled = false};
	AppleDropDraw clip_on {.z_clip_enabled = true};
	Check((AppleDropMask(clip_off, kNone) & AppleDropDepthClamp) != 0u, "Z clip disabled on a host without clamp is a drop");
	Check((AppleDropMask(clip_on, kNone) & AppleDropDepthClamp) == 0u, "Z clip enabled is what the host does anyway");
	Check((AppleDropMask(clip_off, kFull) & AppleDropDepthClamp) == 0u, "a host with depth clamp drops nothing");
}

// S129: drop_depth_bounds counts only draws that enable the test.
void TestDepthBounds() {
	AppleDropDraw on {.depth_bounds_test = true};
	AppleDropDraw off {.depth_bounds_test = false};
	Check((AppleDropMask(on, kNone) & AppleDropDepthBounds) != 0u, "depth bounds enabled and skipped is a drop");
	Check((AppleDropMask(off, kNone) & AppleDropDepthBounds) == 0u, "depth bounds disabled: nothing to drop");
	Check((AppleDropMask(on, kFull) & AppleDropDepthBounds) == 0u, "host with depth bounds drops nothing");
}

// S129: drop_color_write_dyn needs a declared slot without a view.
void TestColorWrite() {
	AppleDropDraw masked {.color_attachments = 3, .color_attachments_bound = 2};
	AppleDropDraw all {.color_attachments = 3, .color_attachments_bound = 3};
	AppleDropDraw none {};
	Check((AppleDropMask(masked, kNone) & AppleDropColorWrite) != 0u, "slot without a view is a drop");
	Check((AppleDropMask(all, kNone) & AppleDropColorWrite) == 0u, "all slots bound: static mask is right");
	Check((AppleDropMask(none, kNone) & AppleDropColorWrite) == 0u, "no attachments: nothing to drop");
	Check((AppleDropMask(masked, kFull) & AppleDropColorWrite) == 0u, "host with color write enable drops nothing");
}

void TestIndependent() {
	AppleDropDraw all {.z_clip_enabled = false, .depth_bounds_test = true, .color_attachments = 1, .color_attachments_bound = 0};
	Check(AppleDropMask(all, kNone) == (AppleDropDepthClamp | AppleDropDepthBounds | AppleDropColorWrite), "classes add up");
	Check(AppleDropMask({}, kNone) == 0u, "a default draw drops nothing");
}

void TestLodBias() {
	Check(SamplerLodBiasClamped(20.0f, 15.99f), "positive bias beyond the limit");
	Check(SamplerLodBiasClamped(-20.0f, 15.99f), "negative bias beyond the limit");
	Check(!SamplerLodBiasClamped(2.0f, 15.99f) && !SamplerLodBiasClamped(-2.0f, 15.99f), "inside the limit");
}

} // namespace

int main() {
	TestDepthClamp();
	TestDepthBounds();
	TestColorWrite();
	TestIndependent();
	TestLodBias();
	std::puts("AppleDropTests: ok");
	return 0;
}
