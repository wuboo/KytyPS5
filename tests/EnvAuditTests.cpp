#include "common/envAudit.h"
#include "common/knownEnv.h"

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "EnvAuditTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
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
	Check(Common::CountUnknownKytyEnvironment(env, known) == 2, "unknown count wrong");
	Check(Common::CountUnknownKytyEnvironment(nullptr, known) == 0, "null environment counted");
	// The diagnostic switches of this build are known: a run with KYTY_ENV_STRICT=1 must not abort on them.
	const char* const diag[] = {"KYTY_CONTROL_FILE=/tmp/c.txt", "KYTY_DIAG_CS_TRACE_FLIP=6580", "KYTY_DIAG_DIR=/tmp/d",
	                            "KYTY_DIAG_PROTO_SKIP=1", "KYTY_DIAG_TLAS=30", "KYTY_RT_MISS=1", "KYTY_PV_UNIQUE=1", nullptr};
	Check(Common::CountUnknownKytyEnvironment(diag, Common::kKnownKytyEnv) == 0,
	      "a diagnostic switch of this build is unknown to the env audit");
	const char* const typo[] = {"KYTY_RT_MISSS=1", "KYTY_DIAG_CS_TRACE=1", nullptr};
	Check(Common::CountUnknownKytyEnvironment(typo, Common::kKnownKytyEnv) == 2, "misspelled switch accepted");
	return 0;
}
