#ifndef KYTY_GRAPHICS_PRESENTATION_CPPHASESCOPE_H_
#define KYTY_GRAPHICS_PRESENTATION_CPPHASESCOPE_H_

// S323: RAII phase marker for the CP thread (see cpPhases.h). Off unless KYTY_DIAG_CP_PHASES=1: then a scope costs
// one thread_local flag test. On, it costs two steady_clock reads and one counter add (~7 k scopes per flip in the
// level scene, so ~0.6 ms/flip by estimate: not on by default; measured in the task, not here).

#include "graphics/presentation/cpPhases.h"
#include "graphics/presentation/frameTiming.h"

#include <cstdlib>

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace Libs::Graphics::CpPhases {

struct ThreadState {
	Accountant<> accountant;
	bool         bound = false; // only the CP thread counts
};
inline ThreadState& State() {
	thread_local ThreadState state;
	return state;
}

inline bool Enabled() {
	static const bool on = FrameTiming::DiagFlagOn(std::getenv("KYTY_DIAG_CP_PHASES"));
	return on;
}
// Called once by the CP thread.
inline void BindCpThread() {
	if (Enabled()) State().bound = true;
}

class Scope {
public:
	explicit Scope(Phase phase) {
		auto& state = State();
		if (!state.bound) return;
		m_active = state.accountant.Enter(phase, FrameTiming::SteadyNowNs());
	}
	~Scope() {
		if (!m_active) return;
		Phase      closed;
		const auto self = State().accountant.Leave(FrameTiming::SteadyNowNs(), &closed);
		FrameTiming::Add(static_cast<FrameTiming::Counter>(static_cast<uint32_t>(FrameTiming::Counter::CpParseNs) +
		                                                  static_cast<uint32_t>(closed)),
		                 self);
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool m_active = false;
};

// Thread name visible to `sample` / Instruments (the Tracy name alone is not).
inline void NameCurrentThread(const char* name) {
#if defined(__APPLE__)
	pthread_setname_np(name);
#else
	(void)name;
#endif
}

// For counters that only mean something on the CP thread (mprotect, packets).
inline bool OnCpThread() { return State().bound; }

} // namespace Libs::Graphics::CpPhases

#endif
