#pragma once

#include <cstdint>

// S327: bookkeeping of the GPU idle gaps between command buffers (KYTY_DIAG_GPUTRUTH=1). Pure, no Vulkan, so it has a CPU test.
// - drain gap: end of the last buffer before a drain -> start of the first buffer recorded after it (gap_ns / gap_n);
// - mid gap:   end of a buffer -> start of the next one with no drain between them (mid_ns).
// Intervals arrive in tick order from the timestamp harvest. A drain is noted with the tick of the buffer opened after it; the
// intervals still pending at that moment must be harvested BEFORE the new gap is armed (NoteDrain), otherwise the gap that is
// still armed is overwritten and its segment is not counted (S326 lost part of the gaps this way).
namespace Libs::Graphics::GpuGap {

struct Totals {
	uint64_t gap_ns = 0; // sum of the drain gaps
	uint64_t gap_n  = 0; // drain gaps counted
	uint64_t mid_ns = 0; // sum of the gaps in the middle of a segment
};

class Ledger {
public:
	void Arm(uint64_t next_tick) {
		m_armed = true;
		m_tick  = next_tick;
	}
	void Interval(uint64_t tick, uint64_t begin, uint64_t end, Totals& out) {
		if (m_armed && tick >= m_tick) {
			if (m_have_prev) {
				out.gap_ns += begin > m_prev_end ? begin - m_prev_end : 0;
				++out.gap_n;
			}
			m_armed = false;
		} else if (m_have_prev && begin > m_prev_end) {
			out.mid_ns += begin - m_prev_end;
		}
		if (!m_have_prev || end > m_prev_end) m_prev_end = end;
		m_have_prev = true;
	}

private:
	bool     m_armed     = false;
	uint64_t m_tick      = 0;
	uint64_t m_prev_end  = 0;
	bool     m_have_prev = false;
};

// A drain ended. `harvest` reads the intervals of the buffers the GPU has finished (all of them after a drain); it runs first.
template <class Harvest>
void NoteDrain(Ledger& ledger, uint64_t next_tick, Harvest&& harvest) {
#ifdef KYTY_GAP_MUTANT_ARM_FIRST
	ledger.Arm(next_tick);
	harvest();
#else
	harvest();
	ledger.Arm(next_tick);
#endif
}

} // namespace Libs::Graphics::GpuGap
