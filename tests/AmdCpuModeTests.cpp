#include "common/amdCpuMode.h"
#include "common/knownEnv.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "AmdCpuModeTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

} // namespace

int main() {
	using Common::AmdCpuMode;
	using Common::AmdCpuHostSupport;

	// Parsing of KYTY_AMD_CPU and of the launcher setting.
	Check(Common::ParseAmdCpuMode("auto") == AmdCpuMode::Auto, "auto not parsed");
	Check(Common::ParseAmdCpuMode("1") == AmdCpuMode::On, "1 not parsed as on");
	Check(Common::ParseAmdCpuMode("on") == AmdCpuMode::On, "on not parsed");
	Check(Common::ParseAmdCpuMode("0") == AmdCpuMode::Off, "0 not parsed as off");
	Check(Common::ParseAmdCpuMode("off") == AmdCpuMode::Off, "off not parsed");
	Check(!Common::ParseAmdCpuMode("").has_value(), "empty accepted");
	Check(!Common::ParseAmdCpuMode("maybe").has_value(), "garbage accepted");
	Check(!Common::ParseAmdCpuMode("AUTO").has_value(), "case-insensitive parse not intended");

	// Precedence: command line flag > KYTY_AMD_CPU > auto; a bad or unset variable leaves auto.
	Check(Common::ResolveAmdCpuMode(std::nullopt, nullptr) == AmdCpuMode::Auto, "default is not auto");
	Check(Common::ResolveAmdCpuMode(std::nullopt, "1") == AmdCpuMode::On, "env on ignored");
	Check(Common::ResolveAmdCpuMode(std::nullopt, "0") == AmdCpuMode::Off, "env off ignored");
	Check(Common::ResolveAmdCpuMode(std::nullopt, "auto") == AmdCpuMode::Auto, "env auto ignored");
	Check(Common::ResolveAmdCpuMode(std::nullopt, "garbage") == AmdCpuMode::Auto, "bad env not auto");
	Check(Common::ResolveAmdCpuMode(AmdCpuMode::On, "0") == AmdCpuMode::On, "--amd-cpu lost to env");
	Check(Common::ResolveAmdCpuMode(AmdCpuMode::Off, "1") == AmdCpuMode::Off, "--no-amd-cpu lost to env");

	// Auto with an injected CPUID result. Rosetta reports an Intel CPU: no SSE4a (RDPID/CLWB vary).
	const AmdCpuHostSupport amd {true, true, true};
	auto                    d = Common::DecideAmdCpu(AmdCpuMode::Auto, amd);
	Check(!d.enabled, "auto patches a host that supports everything");
	Check(d.reason.find("auto") != std::string::npos, "auto reason missing");
	d = Common::DecideAmdCpu(AmdCpuMode::Auto, {false, true, true});
	Check(d.enabled && d.reason.find("SSE4a") != std::string::npos, "no SSE4a: not enabled / not named");
	Check(d.reason.find("RDPID") == std::string::npos && d.reason.find("CLWB") == std::string::npos,
	      "supported instruction named as missing");
	d = Common::DecideAmdCpu(AmdCpuMode::Auto, {true, false, true});
	Check(d.enabled && d.reason.find("RDPID") != std::string::npos, "no RDPID: not enabled / not named");
	d = Common::DecideAmdCpu(AmdCpuMode::Auto, {true, true, false});
	Check(d.enabled && d.reason.find("CLWB") != std::string::npos, "no CLWB: not enabled / not named");
	d = Common::DecideAmdCpu(AmdCpuMode::Auto, {false, false, false});
	Check(d.enabled && d.reason.find("SSE4a, RDPID, CLWB") != std::string::npos, "nothing supported");

	// On/off do not look at the host.
	Check(Common::DecideAmdCpu(AmdCpuMode::On, amd).enabled, "forced on ignored");
	Check(!Common::DecideAmdCpu(AmdCpuMode::Off, {false, false, false}).enabled, "forced off ignored");
	Check(!Common::DecideAmdCpu(AmdCpuMode::Off, {false, false, false}).reason.empty(), "no reason");

	Check(std::string_view(Common::AmdCpuModeName(AmdCpuMode::Auto)) == "auto" &&
	          std::string_view(Common::AmdCpuModeName(AmdCpuMode::On)) == "on" &&
	          std::string_view(Common::AmdCpuModeName(AmdCpuMode::Off)) == "off",
	      "mode names");

	// KYTY_ENV_STRICT=1 must not abort on the new variable.
	Check(std::ranges::find(Common::kKnownKytyEnv, std::string_view("KYTY_AMD_CPU")) !=
	          std::end(Common::kKnownKytyEnv),
	      "KYTY_AMD_CPU unknown to the env audit");

	// KYTY_AMD_PATCH (S280): which instruction groups the patcher rewrites.
	using namespace Common;
	Check(ParseAmdPatchKinds("all") == AmdPatchAll && ParseAmdPatchKinds("auto") == AmdPatchAll, "all/auto");
	Check(ParseAmdPatchKinds("none") == 0u, "none");
	Check(ParseAmdPatchKinds("sse4a") == AmdPatchSse4a, "sse4a alone");
	Check(ParseAmdPatchKinds("rsqrt") == AmdPatchRsqrt, "rsqrt alone");
	Check(ParseAmdPatchKinds("cpuext") == AmdPatchCpuExt, "cpuext alone");
	Check(ParseAmdPatchKinds("sse4a,cpuext") == (AmdPatchSse4a | AmdPatchCpuExt), "list");
	Check(!ParseAmdPatchKinds("").has_value() && !ParseAmdPatchKinds("sse4a,").has_value() &&
	          !ParseAmdPatchKinds("sse4a,bogus").has_value() && !ParseAmdPatchKinds("ALL").has_value(),
	      "garbage accepted");
	Check(ResolveAmdPatchKinds(nullptr) == AmdPatchAll && ResolveAmdPatchKinds("bogus") == AmdPatchAll,
	      "unset/bad value keeps the default set");
	Check(ResolveAmdPatchKinds("none") == 0u && ResolveAmdPatchKinds("rsqrt") == AmdPatchRsqrt, "env value used");
	Check(AmdPatchKindsName(AmdPatchAll) == "all" && AmdPatchKindsName(0) == "none" &&
	          AmdPatchKindsName(AmdPatchSse4a | AmdPatchRsqrt) == "sse4a,rsqrt",
	      "kinds name");
	Check(std::ranges::find(Common::kKnownKytyEnv, std::string_view("KYTY_AMD_PATCH")) !=
	          std::end(Common::kKnownKytyEnv),
	      "KYTY_AMD_PATCH unknown to the env audit");

	std::puts("AmdCpuModeTests: ok");
	return 0;
}
