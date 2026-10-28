#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_LANDSYNC_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_LANDSYNC_H_

// S495: one switch for the landing sync package. Default off.
// KYTY_LAND_SYNC=0|1|shadow. Sub-switches KYTY_LAND_SYNC_DCC / _GUEST / _RF override one part
// (unset = follow the master; "0" turns that part off while the others stay on).
// (a) DCC metadata is checked on the GPU. ObtainBuffer is a read (is_written=false) so the check
//     does not RecordWrite. (b) A guest read whose bytes are not GPU-dirty is emulated from the
//     backing instead of waiting for the whole page. (c) The SRT walk uses SrtProtAvoid.
// shadow: guest emulation and the SRT walk compare against the old read; the DCC GPU check still runs.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::LandSync {

inline constexpr uint32_t kColorClearSlots = 8;

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode Parse(const char* text) {
	if (text == nullptr || text[0] == '\0' || std::strcmp(text, "0") == 0) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::On: return "1";
		case Mode::Shadow: return "shadow";
		default: return "0";
	}
}

// Unset sub-switch follows `master`. An explicit value replaces it.
inline Mode Effective(Mode master, const char* sub) {
#if defined(KYTY_LANDSYNC_MUTANT_IGNORE_MASTER)
	(void)master;
	return Parse(sub);
#else
	if (sub == nullptr || sub[0] == '\0') return master;
	return Parse(sub);
#endif
}

inline Mode Master() {
	static const Mode mode = Parse(std::getenv("KYTY_LAND_SYNC"));
	return mode;
}

inline Mode DccMode() {
	static const Mode mode = Effective(Master(), std::getenv("KYTY_LAND_SYNC_DCC"));
	return mode;
}
inline Mode GuestMode() {
	static const Mode mode = Effective(Master(), std::getenv("KYTY_LAND_SYNC_GUEST"));
	return mode;
}
inline Mode RfMode() {
	static const Mode mode = Effective(Master(), std::getenv("KYTY_LAND_SYNC_RF"));
	return mode;
}

inline bool EnablesGuest(Mode mode) {
#if defined(KYTY_LANDSYNC_MUTANT_GUEST_OFF)
	(void)mode;
	return false;
#else
	return mode == Mode::On || mode == Mode::Shadow;
#endif
}

inline bool DccGpu() {
	const auto mode = DccMode();
	return mode == Mode::On || mode == Mode::Shadow;
}

// The metadata buffer is obtained as a read. A write would RecordWrite and hide a known fill.
inline constexpr bool DccMetadataObtainIsWrite() {
#if defined(KYTY_LANDSYNC_MUTANT_OBTAIN_WRITES)
	return true;
#else
	return false;
#endif
}

// CPU twin of gpu_tiler_color_clear.comp for one slice: every dword equals the first byte
// repeated, that byte is a multiple of 0x20, and `keys` selects its slot.
struct ClearHit {
	bool     hit  = false;
	uint32_t slot = 0;
};
inline ClearHit ColorClearHit(const uint32_t* words, uint32_t nwords, uint32_t keys) {
	ClearHit result;
	if (words == nullptr || nwords == 0) return result;
	const uint32_t key     = words[0] & 0xffu;
	const uint32_t pattern = key * 0x01010101u;
	result.slot            = key >> 5u;
	for (uint32_t i = 0; i < nwords; i++) {
		if (words[i] != pattern) return result;
	}
	result.hit = (key & 0x1fu) == 0u && result.slot < kColorClearSlots && (keys & (1u << result.slot)) != 0u;
	return result;
}

inline void Announce() {
	static const bool once = [] {
		std::printf("Environment: KYTY_LAND_SYNC=%s dcc=%d guest=%d rf=%s\n", ModeName(Master()), DccGpu() ? 1 : 0,
		            EnablesGuest(GuestMode()) ? 1 : 0, ModeName(RfMode()));
		std::fflush(stdout);
		return true;
	}();
	(void)once;
}

} // namespace Libs::Graphics::LandSync

#endif
