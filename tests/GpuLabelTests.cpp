#include "graphics/host_gpu/gpuLabels.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// S446: KYTY_GPU_LABELS on a recording backend (no Vulkan). Off = zero calls; on = the command that opens an encoder is preceded by exactly one
// label of its class (Begin ... End around it, or one Insert before it); a cause is consumed by the next label only.
// Mutants (must FAIL, WILL_FAIL): GPULABEL_SKIP_TILER (the tiler class is silent), GPULABEL_AFTER_CMD (the label comes after the command).

namespace GL = Libs::Graphics::GpuLabel;

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "GpuLabelTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

std::vector<std::string> g_events;

void RecBegin(void*, const char* label) { g_events.push_back(std::string("B:") + label); }
void RecEnd(void*) { g_events.push_back("E"); }
void RecInsert(void*, const char* label) { g_events.push_back(std::string("I:") + label); }
const GL::Backend kRecorder {RecBegin, RecEnd, RecInsert};
const GL::Backend kNever {nullptr, nullptr, nullptr}; // production backend: must not be used while the test backend is set

void Cmd(const char* name) { g_events.push_back(std::string("C:") + name); }

} // namespace

int main() {
	GL::SetBackendForTest(&kRecorder);

	// 1. Off: the command runs, nothing else is called, nothing is counted, a pending cause is not even stored.
	GL::SetModeForTest(0);
	for (uint32_t c = 0; c < GL::kClassCount; ++c) {
		GL::Run(kNever, nullptr, static_cast<GL::Class>(c), [&] { Cmd("x"); }, "L.%u", c);
	}
	GL::NoteCause(GL::Cause::Upload);
	Check(g_events.size() == GL::kClassCount, "off: only the commands may be recorded");
	for (const auto& e: g_events) Check(e == "C:x", "off: a label call was recorded");
	Check(GL::Stats().labels.load() == 0 && GL::Stats().ns.load() == 0, "off: counters moved");
	Check(GL::TakeCause() == GL::Cause::None, "off: a cause was stored");

	// 2. On (Begin/End): every class gets exactly B, C, E in this order, and the label names the class.
	GL::SetModeForTest(1);
	for (uint32_t c = 0; c < GL::kClassCount; ++c) {
		g_events.clear();
		const auto cls = static_cast<GL::Class>(c);
		Check(GL::Active(cls), "on: class not active (a class is silent)");
		GL::Run(kNever, nullptr, cls, [&] { Cmd("dispatch"); }, "%s.%08x.a%d.%ux%ux%u", GL::ClassName(cls), 0x999c5a2du, 1, 120u, 135u, 1u);
		Check(g_events.size() == 3, "on: not exactly three events per command");
		Check(g_events[0].rfind("B:", 0) == 0, "on: the label does not come first (GPULABEL_AFTER_CMD)");
		Check(g_events[1] == "C:dispatch", "on: the command is not in the middle");
		Check(g_events[2] == "E", "on: the label is not closed right after the command");
		Check(g_events[0].find(GL::ClassName(cls)) == 2, "on: the label does not carry its class");
		Check(g_events[0].size() - 2 <= 63, "on: the label is longer than 63 characters");
	}
	Check(GL::Stats().labels.load() == GL::kClassCount, "on: label count");

	// 3. Nesting: a labelled command inside another stays balanced.
	g_events.clear();
	GL::Run(kNever, nullptr, GL::Class::GuestRp, [&] { GL::Run(kNever, nullptr, GL::Class::Upload, [&] { Cmd("copy"); }, "I.up.1kb"); }, "G.rp.1x1.1.0");
	Check(g_events.size() == 5 && g_events[0] == "B:G.rp.1x1.1.0" && g_events[1] == "B:I.up.1kb" && g_events[2] == "C:copy" && g_events[3] == "E" &&
	          g_events[4] == "E",
	      "nesting: Begin/End not balanced");

	// 4. Cause: appended to the next label only.
	g_events.clear();
	GL::NoteCause(GL::Cause::Upload);
	GL::Run(kNever, nullptr, GL::Class::GuestCs, [&] { Cmd("d"); }, "G.cs.%08x", 0x12345678u);
	GL::Run(kNever, nullptr, GL::Class::GuestCs, [&] { Cmd("d"); }, "G.cs.%08x", 0x12345678u);
	Check(g_events[0] == "B:G.cs.12345678.e=up", "cause: missing on the first label");
	Check(g_events[3] == "B:G.cs.12345678", "cause: not consumed by the first label");

	// 5. A label that does not fit is cut at 63 characters.
	g_events.clear();
	GL::NoteCause(GL::Cause::GuestBarrier);
	GL::Run(kNever, nullptr, GL::Class::GuestCs, [&] { Cmd("d"); }, "G.cs.%08x.a0.%ux%ux%u", 0xffffffffu, 4294967295u, 4294967295u, 4294967295u);
	Check(g_events[0].size() - 2 <= 63, "long: label over 63 characters");

	// 6. Insert mode: one Insert before the command, no End.
	GL::SetModeForTest(2);
	g_events.clear();
	GL::Run(kNever, nullptr, GL::Class::Tiler, [&] { Cmd("dispatch"); }, "I.tiler.px.%ux%u", 4u, 4u);
	Check(g_events.size() == 2 && g_events[0] == "I:I.tiler.px.4x4" && g_events[1] == "C:dispatch", "insert: label not exactly before the command");

	// 7. Cost of the label calls with the recorder (CPU only; the Vulkan calls are measured by G through gpu_label_ns).
	GL::SetModeForTest(1);
	g_events.clear();
	g_events.reserve(8000000);
	const auto start = std::chrono::steady_clock::now();
	constexpr int kRuns = 100000;
	for (int i = 0; i < kRuns; ++i) GL::Run(kNever, nullptr, GL::Class::GuestCs, [&] {}, "G.cs.%08x.a0.%ux%ux%u", 0x999c5a2du, 120u, 135u, 1u);
	const double per_label_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / kRuns;
	std::printf("GpuLabelTests: format + recorder cost %.0f ns per label (includes the std::string of the recorder)\n", per_label_ns);
	Check(GL::Stats().ns.load() > 0, "ns not counted");

	GL::SetBackendForTest(nullptr);
	std::puts("GpuLabelTests: OK");
	return 0;
}
