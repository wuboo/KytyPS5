#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_MINLODVIEW_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_MINLODVIEW_H_

// S126: T# MinLod on hosts without VK_EXT_image_view_min_lod (MoltenVK). Pure helpers, no Vulkan.

#include <algorithm>
#include <cstdint>

namespace Libs::Graphics {

struct MinLodViewRange {
	uint32_t base_level  = 0;
	uint32_t level_count = 0;
	uint32_t min_lod     = 0; // U4.8 clamp still to be applied relative to base_level (0 = fully mapped)
};

// `min_lod` is the T# clamp relative to `base_level` in 1/256 level units. Without a native view clamp the
// view starts at ceil(min_lod / 256) levels higher, never past its last level. The sampler (and the shader,
// for Dynamic mip mode) keeps working relative to the new base.
constexpr MinLodViewRange MapMinLodToBaseLevel(uint32_t base_level, uint32_t level_count, uint32_t min_lod) {
	if (min_lod == 0 || level_count == 0) {
		return {base_level, level_count, min_lod};
	}
#ifdef KYTY_MINLOD_MUTANT_FLOOR
	const uint32_t shift = std::min(min_lod / 256u, level_count - 1u);
#else
#ifdef KYTY_MINLOD_MUTANT_NO_CAP
	const uint32_t shift = (min_lod + 255u) / 256u;
#else
	const uint32_t shift = std::min((min_lod + 255u) / 256u, level_count - 1u);
#endif
#endif
	return {base_level + shift, level_count - shift, 0};
}

// True when a T# MinLod above its base level is bound without any clamp: Apple host, no native extension,
// remap not applied. Dynamic mip mode applies the clamp in the shader and storage views have none.
constexpr bool MinLodDropped(bool apple, bool remapped, bool dynamic_mips, bool storage, uint32_t min_lod) {
	return apple && !remapped && !dynamic_mips && !storage && min_lod != 0;
}

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_MINLODVIEW_H_ */
