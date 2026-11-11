#ifndef KYTY_GRAPHICS_PRESENTATION_DIAGCPCOUNTERS_H_
#define KYTY_GRAPHICS_PRESENTATION_DIAGCPCOUNTERS_H_

// S543 DIAG (KYTY_DIAG_CP_COUNTERS=1, default off): cheap per-draw counters of the command processor, logged as KYTY_FRAME_LOG columns (no phases, no
// Observe). Every site is `if (DiagCp::Enabled())`, so with the switch off nothing changes on the recording path. Counters are in FrameTiming::Counter
// (ObtainN ... GuestEqWaitN).

#include "graphics/presentation/frameTiming.h"

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::DiagCp {

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_CP_COUNTERS");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

inline void Add(FrameTiming::Counter counter, uint64_t value = 1) {
	if (Enabled()) FrameTiming::Add(counter, value);
}

// Adds the lifetime of the scope in ns to a *Ns counter; no clock read when the switch is off.
class Scope {
public:
	explicit Scope(FrameTiming::Counter counter): m_counter(counter), m_on(Enabled()), m_begin(m_on ? FrameTiming::SteadyNowNs() : 0) {}
	~Scope() { Stop(); }
	void Stop() {
		if (m_on) FrameTiming::Add(m_counter, FrameTiming::ElapsedNs(m_begin, FrameTiming::SteadyNowNs()));
		m_on = false;
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	FrameTiming::Counter m_counter;
	bool                 m_on;
	int64_t              m_begin;
};

} // namespace Libs::Graphics::DiagCp

#endif /* KYTY_GRAPHICS_PRESENTATION_DIAGCPCOUNTERS_H_ */
