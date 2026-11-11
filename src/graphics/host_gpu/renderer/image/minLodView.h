#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_MINLODVIEW_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_MINLODVIEW_H_

// S126: T# MinLod on hosts without VK_EXT_image_view_min_lod (MoltenVK). Pure helpers, no Vulkan.

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#include "graphics/shader/minLodShift.h"

namespace Libs::Graphics {

struct MinLodViewRange {
	uint32_t base_level  = 0;
	uint32_t level_count = 0;
	uint32_t min_lod     = 0; // U4.8 clamp still to be applied relative to base_level (0 = fully mapped)
};

// `min_lod` is the T# clamp relative to `base_level` in 1/256 level units. Without a native view clamp the
// view starts at MinLodShift() levels higher (shared with the recompiler's RESINFO correction). The sampler
// (and the shader, for Dynamic mip mode) keeps working relative to the new base.
constexpr MinLodViewRange MapMinLodToBaseLevel(uint32_t base_level, uint32_t level_count, uint32_t min_lod) {
	if (min_lod == 0 || level_count == 0) {
		return {base_level, level_count, min_lod};
	}
	const uint32_t shift = MinLodShift(level_count, min_lod);
	return {base_level + shift, level_count - shift, 0};
}

// True when a T# MinLod above its base level is bound without any clamp: no native view clamp, remap not
// applied. Dynamic mip mode applies the clamp in the shader and storage views have none.
constexpr bool MinLodDropped(bool has_view_min_lod, bool remapped, bool dynamic_mips, bool storage, uint32_t min_lod) {
	return !has_view_min_lod && !remapped && !dynamic_mips && !storage && min_lod != 0;
}

// S538 (KYTY_FIX_VIEW_MIN_LOD=1). Since s500 (de1954df, the upstream device setup) nothing sets has_view_min_lod, although
// the device setup requires VK_EXT_image_view_min_lod and enables its minLod feature: every sampled T# MinLod is dropped and
// the shader samples levels the guest clamped away. Astro Bot's streamed textures bind with MinLod 2.0 while their two finest
// levels still hold another asset's bytes (the galaxy card's white particle splashes). On: the view gets the native clamp.
constexpr bool ViewMinLodEnabled(bool fix_env, bool feature_enabled) {
#ifdef KYTY_MINLOD_MUTANT_VIEW_OFF
	(void)fix_env;
	(void)feature_enabled;
	return false;
#else
	return fix_env && feature_enabled;
#endif
}

// VkImageViewMinLodCreateInfoEXT::minLod is absolute (from level 0 of the image), the T# clamp is U4.8 above the view base.
constexpr float ViewMinLodValue(uint32_t base_level, uint32_t min_lod) {
#ifdef KYTY_MINLOD_MUTANT_VIEW_NO_BASE
	(void)base_level;
	return static_cast<float>(min_lod) / 256.0f;
#else
	return static_cast<float>(base_level) + static_cast<float>(min_lod) / 256.0f;
#endif
}

inline bool FixViewMinLodEnv() {
	const char* value = std::getenv("KYTY_FIX_VIEW_MIN_LOD");
	return value != nullptr && value[0] == '1' && value[1] == '\0';
}

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_MINLODVIEW_H_ */
