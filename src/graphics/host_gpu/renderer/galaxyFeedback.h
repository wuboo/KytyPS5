#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_GALAXYFEEDBACK_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_GALAXYFEEDBACK_H_

// S533. A draw that samples a VkImage while that image is a color or depth
// attachment of the same draw is a feedback loop. KYTY_DIAG_FB_LOOP=1 only
// logs matches in the guest window. KYTY_DIAG_SPLIT_PASS=1 ends the render
// pass and barriers that image to a shader read before the draw.
// KYTY_DIAG_SPLIT_PASS=blend does that before every blend-on draw to the
// galaxy card instead. KYTY_FIX_GALAXY_IMG=1 is the split path.
// Unset: one getenv, then no log and no barrier. Not codegen.
// KYTY_FB_MUTANT_NEVER reports no match and never splits, so the CPU test fails.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::GalaxyFeedback {

inline constexpr long kLo = 4000;
inline constexpr long kHi = 4500;

enum class Mode : uint8_t { Off, Log, Split, Blend, Fix };

struct Bound {
	uint64_t handle  = 0;
	uint64_t address = 0;
	uint32_t slot    = 0;
	uint32_t mip     = 0;
	uint8_t  storage = 0;
	uint8_t  stage   = 0; // 1 = pixel shader
};

struct Attach {
	uint64_t handle  = 0;
	uint64_t address = 0;
	uint8_t  depth   = 0;
};

struct Hit {
	uint64_t address      = 0;
	uint64_t attach       = 0;
	uint32_t slot         = 0;
	uint32_t mip          = 0;
	uint8_t  storage      = 0;
	uint8_t  stage        = 0;
	uint8_t  depth        = 0;
	uint8_t  attach_index = 0;
};

inline const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::Off: return "off";
		case Mode::Log: return "log";
		case Mode::Split: return "split";
		case Mode::Blend: return "blend";
		case Mode::Fix: return "fix";
	}
	return "off";
}

// fix wins over the diagnostic switches. split=blend is the card fallback.
inline Mode Parse(const char* log, const char* split, const char* fix) {
	if (fix != nullptr && std::strcmp(fix, "1") == 0) return Mode::Fix;
	if (split != nullptr && std::strcmp(split, "blend") == 0) return Mode::Blend;
	if (split != nullptr && std::strcmp(split, "1") == 0) return Mode::Split;
	if (log != nullptr && std::strcmp(log, "1") == 0) return Mode::Log;
	return Mode::Off;
}

inline Mode Current() {
	static const Mode mode = [] {
		const Mode parsed = Parse(std::getenv("KYTY_DIAG_FB_LOOP"), std::getenv("KYTY_DIAG_SPLIT_PASS"),
		                          std::getenv("KYTY_FIX_GALAXY_IMG"));
		std::fprintf(stderr, "[s533-fb] cfg mode=%s lo=%ld hi=%ld\n", ModeName(parsed), kLo, kHi);
		std::fflush(stderr);
		return parsed;
	}();
	return mode;
}

// Same VkImage, including another view, format or mip of that image.
// A matching guest address with a different image is not a hit.
inline uint32_t Find(const Bound* bounds, uint32_t nbound, const Attach* attachments, uint32_t nattach, Hit* out,
                     uint32_t cap) {
#if defined(KYTY_FB_MUTANT_NEVER)
	(void)bounds;
	(void)nbound;
	(void)attachments;
	(void)nattach;
	(void)out;
	(void)cap;
	return 0;
#else
	uint32_t n = 0;
	if (bounds == nullptr || attachments == nullptr) return 0;
	for (uint32_t i = 0; i < nbound && n < cap; i++) {
		if (bounds[i].handle == 0) continue;
		for (uint32_t j = 0; j < nattach; j++) {
			if (attachments[j].handle == 0 || attachments[j].handle != bounds[i].handle) continue;
			if (out != nullptr) {
				out[n] = Hit {bounds[i].address, attachments[j].address, bounds[i].slot, bounds[i].mip,
				              bounds[i].storage, bounds[i].stage, attachments[j].depth,
				              static_cast<uint8_t>(j)};
			}
			n++;
			break;
		}
	}
	return n;
#endif
}

// Same guest address, different VkImage. Logged, not split.
inline uint32_t FindAddress(const Bound* bounds, uint32_t nbound, const Attach* attachments, uint32_t nattach,
                            Hit* out, uint32_t cap) {
#if defined(KYTY_FB_MUTANT_NEVER)
	(void)bounds;
	(void)nbound;
	(void)attachments;
	(void)nattach;
	(void)out;
	(void)cap;
	return 0;
#else
	uint32_t n = 0;
	if (bounds == nullptr || attachments == nullptr) return 0;
	for (uint32_t i = 0; i < nbound && n < cap; i++) {
		if (bounds[i].handle == 0 || bounds[i].address == 0) continue;
		for (uint32_t j = 0; j < nattach; j++) {
			if (attachments[j].handle == 0 || attachments[j].address != bounds[i].address) continue;
			if (attachments[j].handle == bounds[i].handle) continue;
			if (out != nullptr) {
				out[n] = Hit {bounds[i].address, attachments[j].address, bounds[i].slot, bounds[i].mip,
				              bounds[i].storage, bounds[i].stage, attachments[j].depth,
				              static_cast<uint8_t>(j)};
			}
			n++;
			break;
		}
	}
	return n;
#endif
}

inline bool ShouldSplit(Mode mode, bool any_hit, bool blend_on) {
#if defined(KYTY_FB_MUTANT_NEVER)
	(void)mode;
	(void)any_hit;
	(void)blend_on;
	return false;
#else
	if (mode == Mode::Split || mode == Mode::Fix) return any_hit;
	if (mode == Mode::Blend) return blend_on;
	return false;
#endif
}

} // namespace Libs::Graphics::GalaxyFeedback

#endif
