#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_PRESENTRECT_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_PRESENTRECT_H_

// S482 KYTY_KEEP_ASPECT=0|1 (default 0): the present blit keeps the aspect ratio of the game image (pillarbox / letterbox, centred) instead of
// stretching it over the whole swapchain. Pure helper.

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics {

struct PresentRect {
	int32_t x = 0;
	int32_t y = 0;
	int32_t w = 0;
	int32_t h = 0;
};

inline bool KeepAspectEnabled(const char* text) {
	return text != nullptr && text[0] == '1';
}

// Largest centred rectangle of the source aspect that fits in dst_w x dst_h (all sizes > 0, else the whole destination).
inline PresentRect FitPresentRect(uint32_t src_w, uint32_t src_h, uint32_t dst_w, uint32_t dst_h, bool keep_aspect) {
	PresentRect r {0, 0, static_cast<int32_t>(dst_w), static_cast<int32_t>(dst_h)};
#if !defined(KYTY_PRESENTRECT_MUTANT_IGNORE_FLAG)
	if (!keep_aspect) {
		return r;
	}
#endif
	if (src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0) {
		return r;
	}
	// Compare src_w/src_h with dst_w/dst_h by cross-multiplication: wider source -> full width, else full height.
	const uint64_t lhs = static_cast<uint64_t>(src_w) * dst_h;
	const uint64_t rhs = static_cast<uint64_t>(dst_w) * src_h;
#if defined(KYTY_PRESENTRECT_MUTANT_SWAP_AXES)
	if (lhs < rhs) {
#else
	if (lhs > rhs) {
#endif
		r.h = static_cast<int32_t>((static_cast<uint64_t>(dst_w) * src_h + src_w / 2) / src_w);
		if (r.h < 1) {
			r.h = 1;
		}
		r.y = (static_cast<int32_t>(dst_h) - r.h) / 2;
	} else if (lhs < rhs) {
		r.w = static_cast<int32_t>((static_cast<uint64_t>(dst_h) * src_w + src_h / 2) / src_h);
		if (r.w < 1) {
			r.w = 1;
		}
		r.x = (static_cast<int32_t>(dst_w) - r.w) / 2;
	}
	return r;
}

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_PRESENTATION_WINDOW_PRESENTRECT_H_ */
