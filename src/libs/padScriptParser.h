#ifndef KYTY_LIBS_PADSCRIPTPARSER_H_
#define KYTY_LIBS_PADSCRIPTPARSER_H_

// Parser for scripted pad input (KYTY_PAD_SCRIPT). Kept free of emulator dependencies so it
// can be unit-tested on its own.
//
// Grammar: entries separated by ';' or newlines, '#' starts a comment.
//   entry   := anchor ':' action ('+' action)*
//   anchor  := start | start '-' end | start '+' hold
//   start   := seconds | 'f' frames
//   end     := number in the start unit, exclusive ('f' prefix optional)
//   hold    := length in the start unit, 'f' prefix required for frames
// A bare start holds for the default point length of its unit.
// Actions: cross circle square triangle options touchpad l1 r1 l2 r2 l3 r3 up down left right,
//          left-stick-{left,right,up,down}, right-stick-{left,right,up,down}.

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace Libs::Controller::PadScript {

enum class Unit : uint8_t { Seconds, Frames };

struct Entry {
	Unit     start_unit  = Unit::Seconds;
	double   start       = 0.0;
	Unit     end_unit    = Unit::Seconds;
	double   end         = 0.0;   // Exclusive, in the start unit, measured from the origin.
	bool     end_is_hold = false; // end is a length measured from start.
	uint32_t buttons     = 0;
	// Stick deflection per axis: -1, 0 or +1 (LeftX, LeftY, RightX, RightY).
	int8_t sticks[4] = {0, 0, 0, 0};
};

struct Defaults {
	double point_seconds = 0.3;
	double point_frames  = 8.0;
};

// Button bits match PAD_BUTTON_* in controller.h.
inline bool LookupAction(std::string_view name, Entry& entry) {
	struct Button {
		std::string_view name;
		uint32_t         bit;
	};
	static constexpr Button BUTTONS[] = {
	    {"l3", 0x00000002u},       {"r3", 0x00000004u},    {"options", 0x00000008u},
	    {"up", 0x00000010u},       {"right", 0x00000020u}, {"down", 0x00000040u},
	    {"left", 0x00000080u},     {"l2", 0x00000100u},    {"r2", 0x00000200u},
	    {"l1", 0x00000400u},       {"r1", 0x00000800u},    {"triangle", 0x00001000u},
	    {"circle", 0x00002000u},   {"cross", 0x00004000u}, {"square", 0x00008000u},
	    {"touchpad", 0x00100000u},
	};
	for (const auto& button: BUTTONS) {
		if (button.name == name) {
			entry.buttons |= button.bit;
			return true;
		}
	}
	struct Stick {
		std::string_view name;
		int              axis;
		int8_t           direction;
	};
	static constexpr Stick STICKS[] = {
	    {"left-stick-left", 0, -1}, {"left-stick-right", 0, 1},  {"left-stick-up", 1, -1},
	    {"left-stick-down", 1, 1},  {"right-stick-left", 2, -1}, {"right-stick-right", 2, 1},
	    {"right-stick-up", 3, -1},  {"right-stick-down", 3, 1},
	};
	for (const auto& stick: STICKS) {
		if (stick.name == name) {
			entry.sticks[stick.axis] = stick.direction;
			return true;
		}
	}
	return false;
}

inline std::string_view Trim(std::string_view text) {
	while (!text.empty() && (text.front() == ' ' || text.front() == '\t' || text.front() == '\r')) {
		text.remove_prefix(1);
	}
	while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\r')) {
		text.remove_suffix(1);
	}
	return text;
}

inline bool ParseNumber(std::string_view text, Unit& unit, double& value) {
	unit = Unit::Seconds;
	if (!text.empty() && text.front() == 'f') {
		unit = Unit::Frames;
		text.remove_prefix(1);
	}
	if (text.empty()) {
		return false;
	}
	const std::string copy(text);
	char*             end = nullptr;
	value                 = std::strtod(copy.c_str(), &end);
	return end == copy.c_str() + copy.size() && value >= 0.0;
}

inline bool ParseEntry(std::string_view text, const Defaults& defaults, Entry& entry,
                       std::string& error) {
	const auto colon = text.find(':');
	if (colon == std::string_view::npos) {
		error = "missing ':'";
		return false;
	}
	const auto anchor  = Trim(text.substr(0, colon));
	auto       actions = Trim(text.substr(colon + 1));

	const auto separator = anchor.find_first_of("-+");
	if (!ParseNumber(Trim(anchor.substr(0, separator)), entry.start_unit, entry.start)) {
		error = "bad start";
		return false;
	}
	if (separator == std::string_view::npos) {
		entry.end_unit    = entry.start_unit;
		entry.end_is_hold = true;
		entry.end =
		    entry.start_unit == Unit::Frames ? defaults.point_frames : defaults.point_seconds;
	} else {
		if (!ParseNumber(Trim(anchor.substr(separator + 1)), entry.end_unit, entry.end)) {
			error = "bad end";
			return false;
		}
		entry.end_is_hold = anchor[separator] == '+';
		// "fA-B" keeps the frame unit for B; a hold must spell its unit ("f100+f12").
		if (!entry.end_is_hold && entry.start_unit == Unit::Frames) {
			entry.end_unit = Unit::Frames;
		}
		if (entry.end_unit != entry.start_unit) {
			error = "anchor mixes seconds and frames";
			return false;
		}
		if (entry.end_is_hold ? entry.end <= 0.0 : entry.end <= entry.start) {
			error = "empty range";
			return false;
		}
	}

	if (actions.empty()) {
		error = "no action";
		return false;
	}
	while (!actions.empty()) {
		const auto plus = actions.find('+');
		const auto name = Trim(actions.substr(0, plus));
		if (!LookupAction(name, entry)) {
			error = "unknown action '" + std::string(name) + "'";
			return false;
		}
		if (plus == std::string_view::npos) {
			break;
		}
		actions.remove_prefix(plus + 1);
	}
	return true;
}

// Returns false and sets error (prefixed with the entry text) on the first bad entry.
inline bool Parse(std::string_view script, const Defaults& defaults, std::vector<Entry>& entries,
                  std::string& error) {
	entries.clear();
	while (!script.empty()) {
		const auto end  = script.find_first_of(";\n");
		auto       item = script.substr(0, end);
		script.remove_prefix(end == std::string_view::npos ? script.size() : end + 1);

		const auto hash = item.find('#');
		if (hash != std::string_view::npos) {
			item = item.substr(0, hash);
		}
		item = Trim(item);
		if (item.empty()) {
			continue;
		}
		Entry entry;
		if (!ParseEntry(item, defaults, entry, error)) {
			error = std::string(item) + ": " + error;
			return false;
		}
		entries.push_back(entry);
	}
	return true;
}

// Whether the entry is held at the given position (both measured from the script origin).
inline bool IsActive(const Entry& entry, double seconds, double frames) {
	const double now = entry.start_unit == Unit::Frames ? frames : seconds;
	const double end = entry.end_is_hold ? entry.start + entry.end : entry.end;
	return now >= entry.start && now < end;
}

} // namespace Libs::Controller::PadScript

#endif /* KYTY_LIBS_PADSCRIPTPARSER_H_ */
