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
	const char* const diag[] = {"KYTY_CONTROL_FILE=/tmp/c.txt", "KYTY_DIAG_CP_PHASES=1", "KYTY_DIAG_DIR=/tmp/d",
	                            "KYTY_DIAG_PROTO_SKIP=1", "KYTY_DIAG_SIGCENSUS=1", "KYTY_DIAG_SYNC=1", "KYTY_DIAG_CP_SAMPLE=1", "KYTY_DIAG_CP_SAMPLE_WINDOW=6300:6800", "KYTY_DIAG_GPU_DONE=1", "KYTY_DIAG_STARVE=1", "KYTY_EAGER_MID_US=800", "KYTY_EAGER_MID_MAX=8", "KYTY_EAGER_LOWWATER=1", "KYTY_EAGER_LOWWATER_US=1000", "KYTY_DIAG_MATMEMO=1", "KYTY_PV_RANGE_GEN=shadow", "KYTY_SPEC_COND=superset", "KYTY_PV_INDIRECT=shadow", "KYTY_PV_INDIRECT_CAP=4096", "KYTY_DIAG_INDIRECT=1", "KYTY_DCC_KNOWN=shadow", "KYTY_DCC_EAGER=1", "KYTY_INDIRECT_GPU=shadow", "KYTY_RT_MISS=1", "KYTY_PV_UNIQUE=1", "KYTY_PV_VSSTORE=1", nullptr};
	Check(Common::CountUnknownKytyEnvironment(diag, Common::kKnownKytyEnv) == 0,
	      "a diagnostic switch of this build is unknown to the env audit");
	const char* const typo[] = {"KYTY_RT_MISSS=1", "KYTY_DIAG_CS_TRACE=1", nullptr};
	Check(Common::CountUnknownKytyEnvironment(typo, Common::kKnownKytyEnv) == 2, "misspelled switch accepted");
	return 0;
}
