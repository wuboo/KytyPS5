#include "graphics/host_gpu/gpuLabels.h"

#include <cstring>

namespace Libs::Graphics::GpuLabel {

namespace {

std::atomic<int>            g_mode {-1};
std::atomic<const Backend*> g_test_backend {nullptr};
thread_local Cause          t_cause = Cause::None;

constexpr const char* kClassNames[kClassCount] = {"G.cs", "G.rp", "I.tiler", "I.up", "I.fault", "I.pvind", "I.vsstore", "I.rb", "G.fill", "G.clr", "I.tex", "I.present"};

int ReadMode() {
	const char* value = std::getenv("KYTY_GPU_LABELS");
	if (value == nullptr || value[0] == '\0' || value[0] == '0') return 0;
	return value[0] == '2' ? 2 : 1;
}

} // namespace

Counters& Stats() {
	static Counters counters;
	return counters;
}

int Mode() {
	int mode = g_mode.load(std::memory_order_relaxed);
	if (mode < 0) {
		mode = ReadMode();
		g_mode.store(mode, std::memory_order_relaxed);
	}
	return mode;
}

void SetModeForTest(int mode) {
	g_mode.store(mode, std::memory_order_relaxed);
}

void SetBackendForTest(const Backend* backend) {
	g_test_backend.store(backend, std::memory_order_relaxed);
}

const Backend* TestBackend() {
	return g_test_backend.load(std::memory_order_relaxed);
}

const char* ClassName(Class c) {
	return kClassNames[static_cast<uint32_t>(c)];
}

const char* CauseName(Cause c) {
	switch (c) {
		case Cause::Upload: return "up";
		case Cause::GuestBarrier: return "gb";
		case Cause::Cs: return "cs";
		case Cause::Rt: return "rt";
		case Cause::Submit: return "sub";
		case Cause::Eop: return "eop";
		case Cause::Other: return "other";
		default: return "";
	}
}

void NoteCause(Cause cause) {
	if (Mode() != 0) t_cause = cause;
}

Cause TakeCause() {
	const Cause cause = t_cause;
	t_cause           = Cause::None;
	return cause;
}

bool ClassEnabled(Class c) {
#ifdef KYTY_GPULABEL_MUTANT_SKIP_TILER
	if (c == Class::Tiler) return false;
#endif
	return static_cast<uint32_t>(c) < kClassCount;
}

} // namespace Libs::Graphics::GpuLabel
