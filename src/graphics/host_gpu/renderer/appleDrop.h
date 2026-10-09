#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_APPLEDROP_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_APPLEDROP_H_

// S129: draws whose guest state asks for something the host cannot do (MoltenVK lacks the Vulkan feature).
// Pure helpers, no Vulkan: the renderer only counts, it never changes what is recorded.

#include <cstdint>

namespace Libs::Graphics {

enum AppleDropBit : uint32_t {
	AppleDropDepthClamp   = 1u << 0u, // guest disabled Z clip (DB clamps to the viewport range), host clips
	AppleDropDepthBounds  = 1u << 1u, // guest enabled the depth bounds test, host skips it
	AppleDropColorWrite   = 1u << 2u, // an attachment slot has no view (write disabled), host uses the static mask
};

struct AppleDropDraw {
	bool     z_clip_enabled           = true;
	bool     depth_bounds_test        = false;
	uint32_t color_attachments        = 0; // slots the render state declares
	uint32_t color_attachments_bound  = 0; // ... of which have an image view
};

struct AppleDropHost {
	bool depth_clamp       = true; // depthClampEnable + VK_EXT_depth_clip_enable
	bool depth_bounds      = true;
	bool color_write_enable = true; // VK_EXT_color_write_enable
};

constexpr uint32_t AppleDropMask(const AppleDropDraw& draw, const AppleDropHost& host) {
	uint32_t mask = 0;
#if defined(KYTY_APPLEDROP_MUTANT_INVERT_CLIP)
	if (!host.depth_clamp && draw.z_clip_enabled) mask |= AppleDropDepthClamp;
#else
	if (!host.depth_clamp && !draw.z_clip_enabled) mask |= AppleDropDepthClamp;
#endif
#if defined(KYTY_APPLEDROP_MUTANT_ALWAYS_BOUNDS)
	if (!host.depth_bounds) mask |= AppleDropDepthBounds;
#else
	if (!host.depth_bounds && draw.depth_bounds_test) mask |= AppleDropDepthBounds;
#endif
#if defined(KYTY_APPLEDROP_MUTANT_NATIVE_COUNTS)
	if (draw.color_attachments_bound < draw.color_attachments) mask |= AppleDropColorWrite;
#else
	if (!host.color_write_enable && draw.color_attachments_bound < draw.color_attachments) mask |= AppleDropColorWrite;
#endif
	return mask;
}

// S# LOD bias is clamped to the device limit when recorded; true when that changes the value.
constexpr bool SamplerLodBiasClamped(float bias, float max_bias) {
#if defined(KYTY_APPLEDROP_MUTANT_BIAS_ABS)
	return bias > max_bias;
#else
	return bias > max_bias || bias < -max_bias;
#endif
}

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_APPLEDROP_H_ */
