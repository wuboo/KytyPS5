#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_COMPUTEMETAFILL_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_COMPUTEMETAFILL_H_

namespace Libs::Graphics {

// S491. One dispatch can be both a write-only metadata clear and a uniform buffer fill.
// TryConsumeComputeMetaClear updates the image and, since 05d25ef2, replaces the shader.
// On 1db45856 the clear is only a side effect and TryConsumeComputeFill still runs, which
// records the fill. Short-circuiting on the clear (meta || fill) drops that record: the
// later read is no_fill and takes a synchronous readback. The fill always runs. Either
// match still replaces the shader.
inline bool RunUniformFillBesideMetaClear(bool meta_matched) {
#ifdef KYTY_S491_MUTANT_META_SKIPS_FILL
	return !meta_matched;
#else
	(void)meta_matched;
	return true;
#endif
}

} // namespace Libs::Graphics

#endif
