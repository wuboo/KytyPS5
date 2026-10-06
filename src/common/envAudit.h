#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_ENVAUDIT_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_ENVAUDIT_H_

#include <algorithm>
#include <span>
#include <string>
#include <string_view>

namespace Common {

// Lists the KYTY_* variables of an environment block, one per line, marking the ones the build
// does not read. A misspelled or ported-from-another-branch switch is otherwise silently ignored.
// `env` is a null-terminated array of "NAME=value" strings.
inline std::string FormatKytyEnvironment(const char* const* env,
                                         std::span<const std::string_view> known) {
	std::string text;
	for (; env != nullptr && *env != nullptr; ++env) {
		const std::string_view entry(*env);
		if (!entry.starts_with("KYTY_")) continue;
		const auto name  = entry.substr(0, entry.find('='));
		const bool match = std::ranges::find(known, name) != known.end();
		text += "Environment: ";
		text += entry;
		if (!match) text += "  <-- unknown to this build, ignored";
		text += '\n';
	}
	return text;
}

} // namespace Common

#endif /* EMULATOR_INCLUDE_EMULATOR_COMMON_ENVAUDIT_H_ */
