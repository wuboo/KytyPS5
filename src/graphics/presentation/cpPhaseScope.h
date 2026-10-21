#ifndef KYTY_GRAPHICS_PRESENTATION_CPPHASESCOPE_H_
#define KYTY_GRAPHICS_PRESENTATION_CPPHASESCOPE_H_

// S323: RAII phase marker for the CP thread (see cpPhases.h). Off unless KYTY_DIAG_CP_PHASES=1: then a scope costs
// one thread_local flag test. On, it costs two steady_clock reads and one counter add (~7 k scopes per flip in the
// level scene, so ~0.6 ms/flip by estimate: not on by default; measured in the task, not here).

#include "graphics/host_gpu/renderer/cpWaitDiag.h"
#include "graphics/host_gpu/renderer/rfOpDiag.h"
#include "graphics/presentation/cpPhases.h"
#include "graphics/presentation/cpSampler.h"
#include "graphics/presentation/cpPcSampler.h"
#include "graphics/presentation/frameTiming.h"

#include <cstdlib>
#include <optional>

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace Libs::Graphics::CpPhases {

struct ThreadState {
	Accountant<> accountant;
	bool         bound = false; // only the CP thread counts
	bool         sampler = false; // S406: this thread writes the sampler breadcrumbs
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
	if (CpSampler::Enabled()) {
		State().sampler = true;
#if defined(__APPLE__)
		if (CpSampler::Pc::Enabled()) CpSampler::Pc::RegisterCurrentThread(); // S439 DIAG: the sampler thread suspends this thread to read its PC
#endif
		CpSampler::StartProduction();
	} else if (RfOp::Enabled() || CpWait::Enabled()) {
		// S413: the breadcrumbs (site, opcode) without the 10 kHz sampler thread; the same single relaxed store per site as with the sampler.
		State().sampler = true;
		CpSampler::Active().store(true);
	}
}

// S463 DIAG: wall ns of a region (children included) added to `counter`; one flag test when KYTY_DIAG_CP_PHASES is off or off the CP thread.
class SubTimer {
public:
	explicit SubTimer(FrameTiming::Counter counter): m_counter(counter), m_on(State().bound) {
		if (m_on) m_begin = FrameTiming::SteadyNowNs();
	}
	~SubTimer() {
		if (m_on) FrameTiming::Add(m_counter, FrameTiming::ElapsedNs(m_begin, FrameTiming::SteadyNowNs()));
	}
	SubTimer(const SubTimer&)            = delete;
	SubTimer& operator=(const SubTimer&) = delete;

private:
	FrameTiming::Counter m_counter;
	bool                 m_on;
	int64_t              m_begin = 0;
};

class Scope {
public:
	explicit Scope(Phase phase) {
		// S406: the breadcrumb works without KYTY_DIAG_CP_PHASES (it is a single relaxed store), and only on the CP thread.
		if (CpSampler::Active().load(std::memory_order_relaxed) && State().sampler) m_site.emplace(CpSampler::PhaseSite(static_cast<uint32_t>(phase)));
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
		if (m_count != FrameTiming::Counter::Count) {
			FrameTiming::Add(m_sub, self);
			FrameTiming::Add(m_count);
		}
	}
	// S463 DIAG: also charge this scope's self ns to `sub` and count the packet in `count`.
	void SetSub(FrameTiming::Counter sub, FrameTiming::Counter count) {
		m_sub = sub;
		m_count = count;
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool                          m_active = false;
	FrameTiming::Counter          m_sub   = FrameTiming::Counter::Count;
	FrameTiming::Counter          m_count = FrameTiming::Counter::Count;
	std::optional<CpSampler::SiteScope> m_site;
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
