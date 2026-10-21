#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_WALKPROBE_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_WALKPROBE_H_

// S451: counters of the SRT walk (where GetGraphicsPrograms spends its time, the control-flow memo). The walker lives in a library the CPU tests link without the
// frame timing, so it reports through two function pointers that PipelineCache points at FrameTiming; without them every call is a null check.

#include <cstdint>

namespace Libs::Graphics::ShaderRecompiler::IR::WalkProbe {

enum class Id : uint8_t {
	MatWalkNs, WalkTreeNs, WalkCfNs, WalkSlotsNs,
	CfWalkN, CfIterN, CfBlocksN, CfCondN, CfEvalN, CfRflN, CfReadN, CfReadNs, CfLoopNs,
	MemoHitN, MemoMissN, MemoNeN, MemoSkipN, MemoMissColdN, MemoMissInputsN, MemoMissReadsN,
	// S466 DIAG: the memo lookup (time, ways looked at, ways tried, guest words read again) and shadow: fast lookup differs from the plain one (must be 0).
	CfFindNs, CfFindScanN, CfFindTriesN, CfFindReadsN, CfFastNeN,
	Count,
};

struct Sink {
	void (*add)(Id id, uint64_t value) = nullptr;
	int64_t (*now_ns)()                = nullptr;
};

inline Sink& Hook() {
	static Sink sink;
	return sink;
}

inline void Add(Id id, uint64_t value = 1) {
	if (const auto fn = Hook().add) fn(id, value);
}

inline int64_t Now() {
	const auto fn = Hook().now_ns;
	return fn != nullptr ? fn() : 0;
}

// Adds the time between construction and destruction to a Ns counter.
class Timer {
public:
	explicit Timer(Id id): m_id(id), m_begin(Now()) {}
	~Timer() {
		if (Hook().now_ns != nullptr) Add(m_id, static_cast<uint64_t>(Now() - m_begin));
	}
	Timer(const Timer&)            = delete;
	Timer& operator=(const Timer&) = delete;

private:
	Id      m_id;
	int64_t m_begin;
};

} // namespace Libs::Graphics::ShaderRecompiler::IR::WalkProbe

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_WALKPROBE_H_ */
