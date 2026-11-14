#include "kernel/pthreadPriority.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::LibKernel;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "HavokPrioTests: failed: %s\n", message);
		std::exit(1);
	}
}

void TestEnv() {
	Check(!HavokPrioEnabled(nullptr) && !HavokPrioEnabled("0") && !HavokPrioEnabled(""), "unset/0/empty = off");
	Check(HavokPrioEnabled("1"), "1 = on");
}

// Off keeps the historical band (guest 478 -> host +2). On ends +2 at 460 (PR #2022).
void TestHostBand() {
	Check(HostSchedPriority(460, false) == +2, "off: 460 is +2");
	Check(HostSchedPriority(461, false) == +2, "off: 461 is still +2");
	Check(HostSchedPriority(478, false) == +2, "off: Havok 478 is +2");
	Check(HostSchedPriority(479, false) == 0, "off: 479 is normal");
	Check(HostSchedPriority(732, false) == 0, "off: 732 is normal");
	Check(HostSchedPriority(733, false) == -2, "off: 733 is -2");

	Check(HostSchedPriority(460, true) == +2, "on: 460 stays +2");
	Check(HostSchedPriority(461, true) == 0, "on: 461 drops to normal");
	Check(HostSchedPriority(478, true) == 0, "on: Havok 478 drops to normal");
	Check(HostSchedPriority(479, true) == 0, "on: 479 stays normal");
	Check(HostSchedPriority(732, true) == 0, "on: 732 stays normal");
	Check(HostSchedPriority(733, true) == -2, "on: 733 stays -2");
}

// Kernel class does not move with the host band: 461..478 remain the low class.
void TestKernelClass() {
	Check(KernelPriorityFromGuest(460) == 256, "460 kernel low");
	Check(KernelPriorityFromGuest(461) == 256, "461 kernel low");
	Check(KernelPriorityFromGuest(478) == 256, "478 kernel low");
	Check(KernelPriorityFromGuest(479) == 700, "479 kernel normal");
	Check(KernelPriorityFromGuest(732) == 700, "732 kernel normal");
	Check(KernelPriorityFromGuest(733) == 767, "733 kernel high");
}

} // namespace

int main() {
	TestEnv();
	TestHostBand();
	TestKernelClass();
	std::puts("HavokPrioTests: ok");
	return 0;
}
