#ifndef KYTY_LIBS_EXITAFTER_H_
#define KYTY_LIBS_EXITAFTER_H_

#include <cstdint>
#include <string>
#include <string_view>

// Pure decisions for KYTY_EXIT_AFTER / KYTY_EXIT_AT_FLIP and pad-replay mark matching.
// The running emulator calls these; the CPU tests compile this header (and the mutants).

namespace Libs::BenchExit {

struct ParsedExitAfter {
	bool        ok   = false;
	std::string name;
	uint64_t    plus = 0;
};

inline bool ParseUint(std::string_view text, uint64_t& out) {
	if (text.empty()) {
		return false;
	}
	uint64_t value = 0;
	for (const char c: text) {
		if (c < '0' || c > '9') {
			return false;
		}
		const uint64_t digit = static_cast<uint64_t>(c - '0');
		if (value > (UINT64_MAX - digit) / 10) {
			return false;
		}
		value = value * 10 + digit;
	}
	out = value;
	return true;
}

// "underwater_aerial_garden+1400" → name + N. The '+' is the last one. Empty name or a non-number is rejected.
inline ParsedExitAfter ParseExitAfter(std::string_view spec) {
	ParsedExitAfter parsed;
	const auto plus = spec.rfind('+');
	if (plus == std::string_view::npos || plus == 0 || plus + 1 >= spec.size()) {
		return parsed;
	}
	uint64_t n = 0;
	if (!ParseUint(spec.substr(plus + 1), n)) {
		return parsed;
	}
	parsed.ok   = true;
	parsed.name = std::string(spec.substr(0, plus));
	parsed.plus = n;
#ifdef KYTY_EXIT_MUTANT_IGNORE_N
	parsed.plus = 0; // mutant: the threshold ignores N, so the level flip itself already exits
#endif
	return parsed;
}

inline bool GuestFlipReached(uint64_t flips, uint64_t threshold) {
	return flips >= threshold;
}

enum class MarkAction { Consume, Mismatch };

// Empty live name matches an empty recorded name. A mismatch must not consume the mark.
inline MarkAction ReplayMarkAction(std::string_view live, std::string_view recorded) {
#ifdef KYTY_REPLAY_MUTANT_ALWAYS_CONSUME
	(void)live;
	(void)recorded;
	return MarkAction::Consume;
#else
	return live == recorded ? MarkAction::Consume : MarkAction::Mismatch;
#endif
}

} // namespace Libs::BenchExit

#endif
