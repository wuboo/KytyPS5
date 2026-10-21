#ifndef KYTY_COMMON_CPTUNING_H_
#define KYTY_COMMON_CPTUNING_H_

// S467 (Q4, F1): priority / affinity of the command-processor (CP) thread, Windows only (no-ops elsewhere).
//   KYTY_CP_PRIORITY=0|1|2   0 = untouched (default), 1 = THREAD_PRIORITY_ABOVE_NORMAL, 2 = THREAD_PRIORITY_HIGHEST for Thread_Gpu.
//   KYTY_CP_AFFINITY=<lp>    Thread_Gpu pinned to logical processor <lp>; every guest thread (kernel pthread create) is kept off the whole
//                            physical core of <lp> (both SMT siblings), so the CP has a core to itself. Host threads (audio, present, driver) are untouched.

#include <cstdint>
#include <cstdlib>

namespace Common::CpTuning {

// Guest threads may use everything except the core of the CP thread; if that leaves nothing, no restriction.
inline uint64_t GuestMask(uint64_t all, uint64_t cp_core) {
#ifdef KYTY_CPTUNING_MUTANT_KEEP_SIBLING
	cp_core &= ~(cp_core & (cp_core - 1)); // keeps only the lowest LP: the sibling stays usable by the guest
#endif
	const uint64_t m = all & ~cp_core;
	return m != 0 ? m : all;
}

inline int EnvInt(const char* name, int fallback) {
	const char* v = std::getenv(name);
	if (v == nullptr || v[0] == 0) return fallback;
	return std::atoi(v);
}

inline int Priority() {
	static const int v = EnvInt("KYTY_CP_PRIORITY", 0);
	return v;
}

inline int Affinity() {
	static const int v = EnvInt("KYTY_CP_AFFINITY", -1);
	return v;
}

} // namespace Common::CpTuning

// Implemented in kernel/pthread.cpp (Windows: SetThreadPriority / SetThreadAffinityMask; elsewhere no-ops).
namespace Common::CpTuning {
void ApplyToCpThread();    // once, on Thread_Gpu (also names the thread for the thread-cpu snapshot)
void ApplyToGuestThread(); // at the start of every guest (kernel pthread) thread
} // namespace Common::CpTuning

#endif /* KYTY_COMMON_CPTUNING_H_ */
