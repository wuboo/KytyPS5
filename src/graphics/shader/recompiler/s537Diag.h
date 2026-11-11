#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_S537DIAG_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_S537DIAG_H_

// S537: exact F32 min/max on AMD (KYTY_EXACT_MINMAX=1) and a DIAG trace of the card shader 5e7524f281e7d196.
//
// KYTY_DIAG_S537_DIR=<existing dir>: the card shader's RDNA2 listing, IR and SPIR-V go to that dir when it is
// compiled, and its stored SPIR-V when the shader store loads it.
// KYTY_DIAG_S537_PROBE=1 (with the dir): the card pixel shader keeps every scalar IR value in a function array and,
// at its MRT0 export, copies the array of an invocation whose colour has |c| > 1000, NaN or Inf (and of a few other
// invocations) to a host-visible buffer; a host thread writes the records to <dir>/s537-probe.txt. The store neither
// loads nor saves the probed modules (their SPIR-V holds this run's buffer address).

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace Libs::Graphics::ShaderRecompiler::S537 {

inline constexpr uint64_t kCardHash = 0x5e7524f281e7d196ull;

// Buffer layout, in dwords: [0] hot count, [1] cold count, records from kHeaderDwords on, each kRecordDwords long:
// magic, kind (1 hot, 2 cold), ordinal, frag x bits, frag y bits, r, g, b, a bits, trace slots, then the trace.
inline constexpr uint32_t kHeaderDwords   = 16;
inline constexpr uint32_t kRecordHead     = 16;
inline constexpr uint32_t kMaxTrace       = 2048;
inline constexpr uint32_t kRecordDwords   = kRecordHead + kMaxTrace;
inline constexpr uint32_t kHotRecords     = 48;
inline constexpr uint32_t kColdRecords    = 16;
inline constexpr uint32_t kMagic          = 0x53353337u;
inline constexpr float    kHotThreshold   = 1000.0f;
inline constexpr uint64_t kBufferBytes    = uint64_t {4} * (kHeaderDwords + uint64_t {kRecordDwords} * (kHotRecords + kColdRecords));

inline bool ExactMinMax() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_EXACT_MINMAX");
		return value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return on;
}

inline const char* Dir() {
	static const char* dir = [] {
		const char* value = std::getenv("KYTY_DIAG_S537_DIR");
		return value != nullptr && value[0] != '\0' ? value : nullptr;
	}();
	return dir;
}

inline bool Probe() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_S537_PROBE");
		return Dir() != nullptr && value != nullptr && std::strcmp(value, "1") == 0;
	}();
	return on;
}

// Device address of the probe buffer; 0 until the pipeline cache creates it.
inline std::atomic<uint64_t>& ProbeAddress() {
	static std::atomic<uint64_t> address {0};
	return address;
}

inline void WriteFile(const std::string& name, const void* data, size_t size) {
	if (Dir() == nullptr) {
		return;
	}
	const std::string path = std::string(Dir()) + "/" + name;
	if (FILE* file = std::fopen(path.c_str(), "wb"); file != nullptr) {
		std::fwrite(data, 1, size, file);
		std::fclose(file);
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::S537

#endif /* KYTY_GRAPHICS_SHADER_RECOMPILER_S537DIAG_H_ */
