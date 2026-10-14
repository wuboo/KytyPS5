#ifndef KYTY_GRAPHICS_PRESENTATION_CPPHASES_H_
#define KYTY_GRAPHICS_PRESENTATION_CPPHASES_H_

// S323: where the command-processor thread spends its time that is neither a known counter nor a wait.
// Pure bookkeeping (no clock, no globals, no Vulkan): a stack of phases that charges every nanosecond to
// exactly one phase (the innermost open one). A phase's SELF time is its span minus its children's spans, so the
// phase columns never count a nanosecond twice and, with the wait slot, they add up to the span of the outermost
// phase (<= cp_busy). Waits (MasterSemaphore::Wait) are the Wait phase: they are carved out of whichever phase
// was open and never reach a cp_* column. Switch: KYTY_DIAG_CP_PHASES=1 (cpPhaseScope.h).

#include "graphics/presentation/cpCounters.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace Libs::Graphics::CpPhases {

// Order = the Counter::CpParseNs.. columns of KYTY_FRAME_LOG (frameTiming.h asserts it); Wait is last.
enum class Phase : uint8_t {
	Parse,        // ProcessPm4 loop: packet walk, backlog, dispatch to the handler (handler body excluded)
	Handler,      // a PM4 handler's own time, outside any phase below (register writes, state, flush decisions)
	DrawShaders,  // RefreshShaders / GetGraphicsPrograms
	DrawBind,     // PrepareBindings + PrepareGraphicsBindings
	DrawPipeline, // GetGraphicsPipeline
	DrawTargets,  // AcquireRenderTargets
	DrawDesc,     // CommitBindings: key hash, heap commit, updateDescriptorSets, bind
	DrawVkcmd,    // Commit* / dynamic state / BeginRendering / draw call
	PvStore,      // TryPvVsStoreDraw
	Dispatch,     // compute dispatch recording
	Eop,          // WriteAtEndOfPipe + EopStoreLabel
	Barrier,      // Image::Transit pipelineBarrier2
	ReadbackCpu,  // DownloadBufferMemory without its wait: copies, Unmark / mprotect
	Gc,           // RunGarbageCollector
	SubmitCpu,    // CommandScheduler::Submit without the encode (submit thread)
	Cmds,         // ProcessCommands: commands queued by other threads, without their waits
	Wait,         // blocked in MasterSemaphore::Wait
	Count,
};
constexpr size_t kPhaseCount = static_cast<size_t>(Phase::Count);

// Phase stack. Enter/Leave take the clock reading, so a test drives it with a fake clock.
template <size_t Capacity = 32>
class Accountant {
public:
	// False when the stack is full: the caller must not call Leave for that Enter.
	bool Enter(Phase phase, int64_t now_ns) {
		if (m_depth >= Capacity) return false;
		m_stack[m_depth++] = Frame {phase, now_ns, 0};
		return true;
	}

	// Closes the innermost phase; returns its self ns (also added to Self(phase)) and tells which phase it was.
	uint64_t Leave(int64_t now_ns, Phase* closed = nullptr) {
		const Frame frame = m_stack[--m_depth];
		const uint64_t span = FrameTiming::ElapsedNs(frame.begin_ns, now_ns);
#if defined(KYTY_CPPHASES_MUTANT_KEEP_CHILDREN)
		const uint64_t self = span;
#else
		const uint64_t self = span > frame.child_ns ? span - frame.child_ns : 0;
#endif
		if (m_depth > 0) m_stack[m_depth - 1].child_ns += span;
		if (closed != nullptr) *closed = frame.phase;
#if defined(KYTY_CPPHASES_MUTANT_DROP_PARSE)
		if (frame.phase == Phase::Parse) return 0;
#endif
		m_self[static_cast<size_t>(frame.phase)] += self;
		return self;
	}

	size_t   Depth() const { return m_depth; }
	uint64_t Self(Phase phase) const { return m_self[static_cast<size_t>(phase)]; }
	uint64_t Total() const {
		uint64_t sum = 0;
		for (const auto v: m_self) sum += v;
		return sum;
	}

private:
	struct Frame {
		Phase    phase;
		int64_t  begin_ns;
		uint64_t child_ns;
	};
	std::array<Frame, Capacity>           m_stack {};
	size_t                             m_depth = 0;
	std::array<uint64_t, kPhaseCount>  m_self {};
};

} // namespace Libs::Graphics::CpPhases

#endif
