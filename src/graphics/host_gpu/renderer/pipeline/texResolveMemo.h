#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXRESOLVEMEMO_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXRESOLVEMEMO_H_

// S438 KYTY_TEX_RESOLVE_MEMO=0|1|shadow: the image a texture description (T# + shader inputs) resolved to is remembered with the
// registration generation of the texture cache it was found at, and used again until a registration touches the pages of the image.
// This header holds only the decision (no Vulkan, no cache), so tests/TexResolveMemoTests.cpp can pin it with fake hooks.
//
// What a hit skips: the ExactImageKey build, its hash and lookup, ValidateImageDesc, the second lookup of m_exact_images.
// What a hit still does (same as the `known` branch of TextureCache::FindImage): lock, the slot and registration checks,
// RegistrationsSince, tick + LRU touch, and MaterializeColorClear (DCC/CMASK depend on guest memory, not on registrations).
// The depth/stencil association is read live from the image by the caller, not remembered.
//
// Hooks (H): Lock() -> guard held during the checks and the touch; State(id) -> ImageState; RegistrationsSince(generation) -> bool;
// Touch(id); ColorClear(id) (called after the guard is released, the cache takes its own lock there).
//
// Mutants (tests, WILL_FAIL): KYTY_TEXMEMO_MUTANT_{IGNORES_EPOCH, IGNORES_REBIND, SKIPS_CLEAR, STORE_OVERLAP, STORE_INEXACT}.

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::TexMemo {

enum class Mode : uint8_t { Off, On, Shadow };

// null/""/"0" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
[[nodiscard]] inline Mode ParseMode(const char* text) {
	if (text == nullptr || text[0] == '\0') return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

[[nodiscard]] inline const char* ModeName(Mode mode) {
	return mode == Mode::On ? "1" : mode == Mode::Shadow ? "shadow" : "0";
}

[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_TEX_RESOLVE_MEMO"));
	return mode;
}

template <class Id>
struct Entry {
	Id       id {};
	uint64_t generation = 0;
	bool     valid      = false;
};

struct ImageState {
	bool exists       = false; // the slot holds an image
	bool registered   = false;
	bool needs_rebind = false;
};

enum class Verdict : uint8_t {
	NoEntry, // nothing remembered
	Stale,   // remembered, but the image or its pages changed: resolve in full
	Hit,
};

// What a full resolve (FindImage) says about the result, to decide whether it may be remembered.
struct Found {
	bool exact         = false; // an exact match, no BC alias left behind (the cache would put it in m_exact_images)
	bool view_adjusted = false; // the request was moved to a mip/layer of a larger image
	bool null_image    = false; // the null image or no image
	bool video_out     = false; // video-out requests have an extra check in FindImage
};

[[nodiscard]] inline bool CanStore(const Found& found) {
#if defined(KYTY_TEXMEMO_MUTANT_STORE_OVERLAP)
	return !found.null_image && !found.video_out && found.exact;
#elif defined(KYTY_TEXMEMO_MUTANT_STORE_INEXACT)
	return !found.null_image && !found.video_out && !found.view_adjusted;
#else
	return found.exact && !found.view_adjusted && !found.null_image && !found.video_out;
#endif
}

// Under the hooks' Lock(). commit=false (shadow) only answers; commit=true also touches the image.
template <class Id, class H>
[[nodiscard]] Verdict Check(const Entry<Id>& entry, H& hooks) {
	if (!entry.valid) return Verdict::NoEntry;
	const ImageState state = hooks.State(entry.id);
	if (!state.exists || !state.registered) return Verdict::Stale;
#if !defined(KYTY_TEXMEMO_MUTANT_IGNORES_REBIND)
	if (state.needs_rebind) return Verdict::Stale;
#endif
#if !defined(KYTY_TEXMEMO_MUTANT_IGNORES_EPOCH)
	if (hooks.RegistrationsSince(entry.generation)) return Verdict::Stale;
#endif
	return Verdict::Hit;
}

template <class Id, class H>
[[nodiscard]] Verdict Reuse(const Entry<Id>& entry, H& hooks, bool commit) {
	{
		auto guard = hooks.Lock();
		(void)guard;
		const Verdict verdict = Check(entry, hooks);
		if (verdict != Verdict::Hit || !commit) return verdict;
		hooks.Touch(entry.id);
	}
#if !defined(KYTY_TEXMEMO_MUTANT_SKIPS_CLEAR)
	hooks.ColorClear(entry.id);
#endif
	return Verdict::Hit;
}

// The shadow verdict: what the memo would have returned against what the full resolve returned.
template <class Id>
[[nodiscard]] bool ShadowAgrees(const Entry<Id>& entry, const Id& full_id, bool desc_same) {
	return entry.id == full_id && desc_same;
}

} // namespace Libs::Graphics::TexMemo

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXRESOLVEMEMO_H_
