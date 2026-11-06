#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_FRAMEDUMPFLIPS_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_FRAMEDUMPFLIPS_H_

// KYTY_FRAME_DUMP_FLIPS=5000,5800: 1-based present indexes (frames.csv rows) to dump. Pure helpers.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

inline std::vector<uint64_t> ParseFrameDumpFlips(const char* text) {
	std::vector<uint64_t> flips;
	if (text == nullptr) {
		return flips;
	}
	const char* p = text;
	while (*p != '\0') {
		char*      end = nullptr;
		const auto n   = std::strtoull(p, &end, 10);
		if (end == p) {
			break;
		}
#if defined(KYTY_FRAMEDUMP_MUTANT_KEEP_ZERO)
		flips.push_back(n);
#else
		if (n != 0) {
			flips.push_back(n);
		}
#endif
		p = end;
		if (*p == ',') {
			++p;
		}
	}
	return flips;
}

inline bool FrameDumpFlipWanted(const std::vector<uint64_t>& flips, uint64_t flip) {
#if defined(KYTY_FRAMEDUMP_MUTANT_OFF_BY_ONE)
	flip += 1;
#endif
	return std::find(flips.begin(), flips.end(), flip) != flips.end();
}

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_PRESENTATION_WINDOW_FRAMEDUMPFLIPS_H_ */
