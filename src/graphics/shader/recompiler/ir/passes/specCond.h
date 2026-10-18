#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SPECCOND_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SPECCOND_H_

// S408: KYTY_SPEC_COND=0|superset|stale (unset = 0). The condition of a control-flow block of the SRT plan chooses the reachable
// blocks, so the descriptor sources that are active. When the word it reads is GPU-written and the GPU has not finished writing it,
// ReadShaderGuestMemory drains the queue to see the fresh value (S406: ~2 drains per flip on 0x571b2da7c, 8.7 ms/flip).
//   0         the old path, bit for bit.
//   superset  a read for purpose Cond of GPU-dirty bytes fails WITHOUT draining. The walker answers a failed condition read by
//             following both successors (SrtWalker.cpp, RefreshFlatBuffer), so the sources of both sides are active and bound; the
//             GPU takes the branch it reads itself. Only on the hit / refresh path (a program that is already compiled): a miss
//             (compile, capture_specialization_reads) keeps the old read. Every other purpose reads as before.
//   stale     (not implemented: needs the verification of the used bytes after GPU-done; the process exits at startup.)
// The failure must reach the walker as a failure: the reader installed as read_specialization_memory is the one the condition
// evaluator calls; SrtReader::Apply only fills read_memory (ReadOrZero, which would turn the failure into zeros).
// Pure part: tests/SpecCondTests.cpp and tests/SpecShadowTests.cpp pin it.

#include "graphics/shader/recompiler/ir/passes/specShadow.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

namespace Libs::Graphics::ShaderRecompiler::IR::SpecCond {

enum class Mode : uint8_t { Off, Superset, Stale };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "superset") == 0) return Mode::Superset;
	if (std::strcmp(text, "stale") == 0) return Mode::Stale;
	return Mode::Off;
}
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_SPEC_COND"));
	return mode;
}

// Which lookup path the reader serves (SyncCensus::ReaderScope::path: "hit", "refresh", "miss").
inline bool IsMissPath(const char* path) {
	return path != nullptr && std::strcmp(path, "miss") == 0;
}

enum class Action : uint8_t {
	Read,       // the old read (drains when the bytes are GPU-dirty)
	Fail,       // superset: report a failed read, nothing is read and nothing drains
	SkipMiss,   // superset would have failed the read, but the program is being compiled: the old read
};

// `dirty()` is asked only for the one purpose that can change the answer, so the other reads pay nothing.
template <typename DirtyFn>
inline Action Decide(Mode mode, SpecShadow::Purpose purpose, bool miss_path, DirtyFn&& dirty) {
	if (mode != Mode::Superset) return Action::Read;
#if defined(KYTY_SPECCOND_MUTANT_SUPERSET_ALL_PURPOSES)
	if (purpose != SpecShadow::Purpose::Cond && purpose != SpecShadow::Purpose::Slot) return Action::Read;
#else
	if (purpose != SpecShadow::Purpose::Cond) return Action::Read;
#endif
#if defined(KYTY_SPECCOND_MUTANT_SUPERSET_CLEAN)
	(void)dirty;
#else
	if (!dirty()) return Action::Read;
#endif
	return miss_path ? Action::SkipMiss : Action::Fail;
}

// What the reader reports to its owner (counters).
struct Result {
	bool fail      = false; // the read was failed on purpose (superset)
	bool skip_miss = false;
};

// The whole reader on injectable hooks (tests); PipelineCache calls Decide with the real ones.
struct Hooks {
	bool (*dirty)(uint64_t address, uint64_t size)               = nullptr;
	bool (*read)(uint64_t address, void* data, uint64_t size)    = nullptr; // the old read (drains)
};

inline bool Read(Mode mode, bool miss_path, const Hooks& hooks, uint64_t address, std::span<uint32_t> values, Result* result = nullptr) {
	if (values.empty()) return false;
	const auto action = Decide(mode, SpecShadow::CurrentPurpose(), miss_path, [&] {
		return hooks.dirty != nullptr && hooks.dirty(address, values.size_bytes());
	});
	if (result != nullptr) {
		result->fail      = action == Action::Fail;
		result->skip_miss = action == Action::SkipMiss;
	}
	if (action == Action::Fail) return false;
	return hooks.read != nullptr && hooks.read(address, values.data(), values.size_bytes());
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::SpecCond

#endif
