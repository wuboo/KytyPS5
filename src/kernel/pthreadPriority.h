#ifndef KYTY_KERNEL_PTHREAD_PRIORITY_H_
#define KYTY_KERNEL_PTHREAD_PRIORITY_H_

// KYTY_HAVOK_PRIO=0|1 (default 0). On Windows the host priority +2 band covers guest
// priorities <= 478. ASTRO BOT's Havok workers sit at 478, spin on a lock and starve
// DrawThread/Physics (PR #2022). With the switch the +2 band ends at 460, so 461..478
// stay at normal host priority. The kernel priority class stays on 478/733 either way.

namespace Libs::LibKernel {

inline bool HavokPrioEnabled(const char* text) {
	return text != nullptr && text[0] == '1';
}

inline int HavokHostBand(bool havok_prio) {
#if defined(KYTY_HAVOK_PRIO_MUTANT_OLD_BAND)
	(void)havok_prio;
	return 478;
#else
	return havok_prio ? 460 : 478;
#endif
}

// Windows pthread sched_priority: +2 above normal, 0 normal, -2 below.
inline int HostSchedPriority(int guest, bool havok_prio) {
	if (guest <= HavokHostBand(havok_prio)) {
		return +2;
	}
	if (guest >= 733) {
		return -2;
	}
	return 0;
}

// scePthread priority class the kernel reports. Independent of the host band.
inline int KernelPriorityFromGuest(int guest) {
	if (guest >= 733) {
		return 767;
	}
	if (guest <= 478) {
		return 256;
	}
	return 700;
}

} // namespace Libs::LibKernel

#endif /* KYTY_KERNEL_PTHREAD_PRIORITY_H_ */
