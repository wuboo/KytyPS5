#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTSLOTBATCH_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTSLOTBATCH_H_

// S445 (W1): KYTY_SRT_SLOT_BATCH=0|1|shadow (default 0 = the per-slot walk of S442 bit for bit, nothing extra compiled).
// The per-slot JIT enters generated code once per SRT slot (~40 ns each under Rosetta, ~87 000 slots per flip); the plans that matter all have block
// conditions. 1: the slots of one control-flow block that are compiled and follow each other in block.srt_reads form a run; a run is ONE entry into
// generated code with the pointer chains inlined (a node already ensured earlier in the run is read from the memo without a check). Same flat buffer,
// same memo, same order of the addresses in the capture. A slot that fails in the middle of a run goes to the interpreter exactly as before and the rest
// of the run goes through the per-slot entries. shadow: every run first runs on a scratch copy of the memo and the result is compared with the per-slot
// path that is then used (srt_batch_mismatch_n must stay 0, `[srt-batch-shadow]` lines <= 16).
// Mutants (tests): KYTY_SRTBATCH_MUTANT_SKIP_MEMO_CHECK, _NO_FAIL_POS, _ORDER_SWAP, _IGNORE_CLEAN, _SHADOW_REAL_MEMO, _ALL_BLOCKS.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler::IR::SrtSlotBatch {

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
	static const Mode env = Parse(std::getenv("KYTY_SRT_SLOT_BATCH"));
	const int forced = Override().load(std::memory_order_relaxed);
	return forced >= 0 ? static_cast<Mode>(forced) : env;
}

// A run shorter than this keeps the per-slot entry (one slot cannot get cheaper).
constexpr uint32_t kMinRun = 2;

// Runs: batch entries executed; Slots: slots they filled; Single: compiled slots left to the per-slot entry (run of one); Fail: runs that stopped at a
// failing slot; Mismatch: shadow results that differ from the per-slot path (must be 0).
enum class Cell : uint8_t { Runs, Slots, Single, Fail, Mismatch, Count };
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

constexpr uint32_t kMaxLogLines = 16;

// Where `[srt-batch-shadow]` lines go: stdout until the pipeline cache installs the emulator log (the walker is linked into test binaries without it).
inline void (*&LogSink())(const char*) {
	static void (*sink)(const char*) = +[](const char* text) { std::fputs(text, stdout); };
	return sink;
}

// True for the first kMaxLogLines mismatches (the caller prints a `[srt-batch-shadow]` line).
inline bool ShouldLog() {
	static std::atomic<uint32_t> lines {0};
	return lines.fetch_add(1, std::memory_order_relaxed) < kMaxLogLines;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::SrtSlotBatch

#endif
