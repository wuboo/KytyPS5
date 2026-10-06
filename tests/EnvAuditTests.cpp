#include "common/envAudit.h"

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "EnvAuditTests: failed: %s\n", text);
		std::abort();
	}
}

} // namespace

int main() {
	static constexpr std::string_view known[] = {"KYTY_RT_SKIP", "KYTY_OPT_OFF"};
	const char* const env[] = {"PATH=/usr/bin", "KYTY_RT_SKIP=1", "KYTY_RT_SKIPP=1",
	                           "KYTY_NOT_A_SWITCH=2", "KYTY_OPT_OFF=", nullptr};
	const auto text = Common::FormatKytyEnvironment(env, known);
	Check(text.find("PATH") == std::string::npos, "non-KYTY variable listed");
	Check(text.find("Environment: KYTY_RT_SKIP=1\n") != std::string::npos, "known variable lost");
	Check(text.find("KYTY_RT_SKIP=1  <--") == std::string::npos, "known variable marked unknown");
	Check(text.find("KYTY_RT_SKIPP=1  <-- unknown") != std::string::npos,
	      "prefix of a known name accepted");
	Check(text.find("KYTY_NOT_A_SWITCH=2  <-- unknown") != std::string::npos,
	      "unknown variable not marked");
	Check(text.find("KYTY_OPT_OFF=\n") != std::string::npos, "empty value lost or marked unknown");
	Check(Common::FormatKytyEnvironment(nullptr, known).empty(), "null environment not empty");
	return 0;
}
