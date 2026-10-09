#include "graphics/presentation/cpCounters.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::FrameTiming;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "CpCountersTests: failed: %s\n", message);
		std::exit(1);
	}
}

// S132: disp_proto_capture vs disp_proto_capture_nonnative follow the program's native flag.
void TestCaptureClass() {
	Check(ProtoCaptureClass(true) == DispatchClass::ProtoCapture, "native capture is disp_proto_capture");
	Check(ProtoCaptureClass(false) == DispatchClass::ProtoCaptureNonNative, "non-native capture is its own class");
	Check(DispatchClass::Other == static_cast<DispatchClass>(static_cast<uint32_t>(DispatchClass::Count) - 1u),
	      "disp_other is the last class");
}

// S132: the *_ns counters never go backwards.
void TestElapsed() {
	Check(ElapsedNs(100, 350) == 250u, "elapsed time");
	Check(ElapsedNs(100, 100) == 0u, "no time passed");
	Check(ElapsedNs(500, 100) == 0u, "a clock step back is zero, not a huge unsigned value");
}

// S132: KYTY_DIAG_PROTO_SKIP is on for "1" only.
void TestFlag() {
	Check(DiagFlagOn("1"), "1 turns the switch on");
	Check(!DiagFlagOn(nullptr), "unset is off");
	Check(!DiagFlagOn("0"), "0 is off");
	Check(!DiagFlagOn(""), "empty is off");
	Check(!DiagFlagOn("11"), "only the exact value 1");
}

void TestDecision() {
	Check(ProtoDrawDecision(true, false) == ProtoDrawAction::Execute, "supported draw runs by default");
	Check(ProtoDrawDecision(true, true) == ProtoDrawAction::DiagSkipped, "KYTY_DIAG_PROTO_SKIP=1 skips a supported draw");
	Check(ProtoDrawDecision(false, false) == ProtoDrawAction::Skip, "rejected draw stays skipped");
	Check(ProtoDrawDecision(false, true) == ProtoDrawAction::Skip, "rejected draw is not counted as diag-skipped");
}

} // namespace

int main() {
	TestCaptureClass();
	TestElapsed();
	TestFlag();
	TestDecision();
	std::puts("CpCountersTests: ok");
	return 0;
}
