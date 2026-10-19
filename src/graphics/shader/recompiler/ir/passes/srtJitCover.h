#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTJITCOVER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTJITCOVER_H_

// S442: KYTY_SRT_JIT_COVER=0|1|shadow (default 0 = the pre-S442 walk bit for bit). Widens what the SRT walk does without the interpreter:
//  - plans with no SRT reads and no block conditions return their (empty) flat buffer without entering RefreshFlatBuffer's walk (srt_cover_noreads_n);
//  - the per-slot JIT also compiles ReadConstBuffer slots (a V# whose words are user data or compiled reads) (srt_cover_cb_n);
//  - srt_cover_capture_n counts the slots the per-slot JIT served in walks of plans that capture specialization reads.
// shadow: both paths run and the flat buffers are compared; the old result is the one used. srt_jit_cover_ne_n must stay 0, `[srt-cover-ne]` lines <= 20.
// Mutants (tests): KYTY_SRTCOVER_MUTANT_NOREADS_IGNORES_USERDATA (the no-reads result depends on user data), KYTY_SRTCOVER_MUTANT_CB_WRONG_OFFSET
// (the compiled constant-buffer read uses the wrong byte offset).

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler::IR::SrtJitCover {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode Parse(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

// Tests set this (-1: use the environment).
inline std::atomic<int>& Override() {
	static std::atomic<int> value {-1};
	return value;
}

inline Mode Current() {
	static const Mode env = Parse(std::getenv("KYTY_SRT_JIT_COVER"));
	const int forced = Override().load(std::memory_order_relaxed);
	return forced >= 0 ? static_cast<Mode>(forced) : env;
}

enum class Cell : uint8_t { NoReads, ConstBuffer, Capture, Ne, Count };
constexpr size_t kCellCount = static_cast<size_t>(Cell::Count);

inline std::array<std::atomic<uint64_t>, kCellCount>& Cells() {
	static std::array<std::atomic<uint64_t>, kCellCount> cells {};
	return cells;
}

inline void Note(Cell cell, uint64_t n = 1) {
	Cells()[static_cast<size_t>(cell)].fetch_add(n, std::memory_order_relaxed);
}

// Takes the counts accumulated since the last Drain.
inline std::array<uint64_t, kCellCount> Drain() {
	std::array<uint64_t, kCellCount> out {};
	for (size_t i = 0; i < kCellCount; ++i) out[i] = Cells()[i].exchange(0, std::memory_order_relaxed);
	return out;
}

constexpr uint32_t kMaxNeLines = 20;

// Where `[srt-cover-ne]` lines go: stdout until the pipeline cache installs the emulator log (the walker is linked into test binaries without it).
inline void (*&LogSink())(const char*) {
	static void (*sink)(const char*) = +[](const char* text) { std::fputs(text, stdout); };
	return sink;
}

// True for the first kMaxNeLines mismatches (the caller prints a `[srt-cover-ne]` line).
inline bool ShouldLogNe() {
	static std::atomic<uint32_t> lines {0};
	return lines.fetch_add(1, std::memory_order_relaxed) < kMaxNeLines;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::SrtJitCover

#endif
