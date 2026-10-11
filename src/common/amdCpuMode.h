#ifndef EMULATOR_INCLUDE_EMULATOR_COMMON_AMDCPUMODE_H_
#define EMULATOR_INCLUDE_EMULATOR_COMMON_AMDCPUMODE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace Common {

// The guest is built for an AMD CPU. Without the patcher every SSE4a/RDPID/CLWB instruction the host lacks
// raises SIGILL and is emulated in a signal handler (about ten thousand signals per frame on Rosetta, which
// reports an Intel CPU without SSE4a). Auto turns the patcher on only when the host lacks one of them.
enum class AmdCpuMode { Auto, On, Off };

// Instructions the AMD patcher rewrites, as reported by CPUID of the host.
struct AmdCpuHostSupport {
	bool sse4a = false;
	bool rdpid = false;
	bool clwb  = false;
};

struct AmdCpuDecision {
	bool        enabled = false;
	std::string reason;
};

// "auto", "on"/"1"/"true", "off"/"0"/"false"; anything else (including empty) is not a mode.
inline std::optional<AmdCpuMode> ParseAmdCpuMode(std::string_view text) {
	if (text == "auto") return AmdCpuMode::Auto;
	if (text == "on" || text == "1" || text == "true") return AmdCpuMode::On;
	if (text == "off" || text == "0" || text == "false") return AmdCpuMode::Off;
	return std::nullopt;
}

inline const char* AmdCpuModeName(AmdCpuMode mode) {
	return mode == AmdCpuMode::On ? "on" : mode == AmdCpuMode::Off ? "off" : "auto";
}

// An explicit command line flag wins over KYTY_AMD_CPU; an unset or unparsable variable leaves auto.
inline AmdCpuMode ResolveAmdCpuMode(std::optional<AmdCpuMode> command_line, const char* env) {
	if (command_line) return *command_line;
	if (env != nullptr) {
		if (const auto mode = ParseAmdCpuMode(env)) return *mode;
	}
	return AmdCpuMode::Auto;
}

inline AmdCpuDecision DecideAmdCpu(AmdCpuMode mode, AmdCpuHostSupport host) {
	if (mode == AmdCpuMode::On) return {true, "forced on (--amd-cpu or KYTY_AMD_CPU=1)"};
	if (mode == AmdCpuMode::Off) return {false, "forced off (--no-amd-cpu or KYTY_AMD_CPU=0)"};
	std::string missing;
	const auto  add = [&missing](bool supported, const char* name) {
		if (supported) return;
		if (!missing.empty()) missing += ", ";
		missing += name;
	};
	add(host.sse4a, "SSE4a");
	add(host.rdpid, "RDPID");
	add(host.clwb, "CLWB");
	if (missing.empty()) return {false, "auto: host supports SSE4a, RDPID and CLWB"};
	return {true, "auto: host lacks " + missing};
}

// Which instruction groups the patcher rewrites once it is on (KYTY_AMD_PATCH, S280). Splitting them tells
// apart picture defects caused by one rewrite from the others: sse4a = EXTRQ/INSERTQ, rsqrt = VRSQRTPS
// (which the patcher replaces by an exact reciprocal square root even where the host has the instruction),
// cpuext = RDPID/CLWB.
enum AmdPatchKind : uint32_t {
	AmdPatchSse4a  = 1u << 0,
	AmdPatchRsqrt  = 1u << 1,
	AmdPatchCpuExt = 1u << 2,
	AmdPatchAll    = AmdPatchSse4a | AmdPatchRsqrt | AmdPatchCpuExt,
};

// "all"/"auto" (the default set), "none", or a comma separated list of sse4a, rsqrt, cpuext; anything else
// (including empty and an empty list item) is not a selection.
inline std::optional<uint32_t> ParseAmdPatchKinds(std::string_view text) {
	if (text == "all" || text == "auto") return AmdPatchAll;
	if (text == "none") return 0u;
	uint32_t kinds = 0;
	while (true) {
		const size_t      comma = text.find(',');
		const std::string_view item = text.substr(0, comma);
		if (item == "sse4a") kinds |= AmdPatchSse4a;
		else if (item == "rsqrt") kinds |= AmdPatchRsqrt;
		else if (item == "cpuext") kinds |= AmdPatchCpuExt;
		else return std::nullopt;
		if (comma == std::string_view::npos) return kinds;
		text.remove_prefix(comma + 1);
	}
}

// An unset or unparsable variable keeps the default set.
inline uint32_t ResolveAmdPatchKinds(const char* env) {
	if (env != nullptr) {
		if (const auto kinds = ParseAmdPatchKinds(env)) return *kinds;
	}
	return AmdPatchAll;
}

inline std::string AmdPatchKindsName(uint32_t kinds) {
	if (kinds == AmdPatchAll) return "all";
	if (kinds == 0) return "none";
	std::string name;
	const auto  add = [&](uint32_t bit, const char* text) {
		if ((kinds & bit) == 0) return;
		if (!name.empty()) name += ",";
		name += text;
	};
	add(AmdPatchSse4a, "sse4a");
	add(AmdPatchRsqrt, "rsqrt");
	add(AmdPatchCpuExt, "cpuext");
	return name;
}

} // namespace Common

#endif /* EMULATOR_INCLUDE_EMULATOR_COMMON_AMDCPUMODE_H_ */
