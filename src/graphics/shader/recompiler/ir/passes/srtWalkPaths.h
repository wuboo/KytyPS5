#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTWALKPATHS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTWALKPATHS_H_

// S430 DIAG: which path SrtWalker::RefreshFlatBuffer took. Counts only while Active() (KYTY_DIAG_CP_SAMPLE / KYTY_DIAG_CP_PHASES),
// otherwise Note() is one relaxed load. PipelineCache drains the cells into FrameTiming counters (srt_walk_*) once per Get().

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR::SrtWalkPaths {

enum class Path : uint8_t {
	Tree,        // srt_code (JIT of the whole tree) filled the flat buffer
	TreeFail,    // srt_code ran and returned false (empty table): the walk goes on in the interpreter
	Capture,     // srt_code exists but the runtime captures reads (read_memory != nullptr): no tree JIT
	SlotJit,     // slots served by the per-slot JIT
	InterpWalk,  // walks not served by the tree JIT
	InterpSlots, // slots evaluated by the interpreter (clean slots, failed slot JIT, no slot JIT)
	Count,
};
constexpr size_t kPathCount = static_cast<size_t>(Path::Count);

inline std::atomic<bool>& Active() {
	static std::atomic<bool> on {false};
	return on;
}

inline std::array<std::atomic<uint64_t>, kPathCount>& Cells() {
	static std::array<std::atomic<uint64_t>, kPathCount> cells {};
	return cells;
}

inline void Note(Path path, uint64_t n = 1) {
#ifndef KYTY_SRTWALKPATHS_MUTANT_NO_GATE
	if (!Active().load(std::memory_order_relaxed)) return;
#endif
	Cells()[static_cast<size_t>(path)].fetch_add(n, std::memory_order_relaxed);
}

// Takes the counts accumulated since the last Drain.
inline std::array<uint64_t, kPathCount> Drain() {
	std::array<uint64_t, kPathCount> out {};
	for (size_t i = 0; i < kPathCount; ++i) out[i] = Cells()[i].exchange(0, std::memory_order_relaxed);
	return out;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::SrtWalkPaths

#endif
