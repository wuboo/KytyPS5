#ifndef KYTY_GRAPHICS_HOST_GPU_GPULABELS_H_
#define KYTY_GRAPHICS_HOST_GPU_GPULABELS_H_

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// S446: KYTY_GPU_LABELS=0|1|2. Names the Metal encoder of every piece of GPU work (guest dispatch, guest render pass, our own
// tiler / upload / fault / pv-indirect / VS-store / present passes) so a Metal System Trace can be summed per class.
//   0 (default): no debug-utils call at all.
//   1: the command that opens the encoder is wrapped in vkCmdBeginDebugUtilsLabelEXT / vkCmdEndDebugUtilsLabelEXT.
//   2: vkCmdInsertDebugUtilsLabelEXT right before that command (for a MoltenVK that turns inserts into encoder labels).
// This header has no Vulkan types, so the CPU tests can drive it with a recording backend.
namespace Libs::Graphics::GpuLabel {

enum class Class : uint8_t {
	GuestCs, GuestRp, Tiler, Upload, Fault, PvIndirect, VsStore, Readback, Fill, Clear, Tex, Present, Count
};
constexpr uint32_t kClassCount = static_cast<uint32_t>(Class::Count);

// Why the previous encoder / pass ended (suffix `.e=<name>`).
enum class Cause : uint8_t { None, Upload, GuestBarrier, Cs, Rt, Submit, Eop, Other };

struct Backend {
	void (*begin)(void* cb, const char* label);
	void (*end)(void* cb);
	void (*insert)(void* cb, const char* label);
};

struct Counters {
	std::atomic<uint64_t> ns {0};
	std::atomic<uint64_t> labels {0};
};

Counters&      Stats();
int            Mode();                      // cached from KYTY_GPU_LABELS
void           SetModeForTest(int mode);
void           SetBackendForTest(const Backend* backend); // nullptr = production
const Backend* TestBackend();
const char*    ClassName(Class c);
const char*    CauseName(Cause c);
void           NoteCause(Cause cause);      // thread-local; consumed by the next label
Cause          TakeCause();
bool           ClassEnabled(Class c);

[[nodiscard]] inline bool Active(Class c) {
	return Mode() != 0 && ClassEnabled(c);
}

// printf into a 64 byte buffer (63 characters), then ".e=<cause>" when a cause is pending and fits.
struct Text {
	char s[64];
	explicit Text(const char* format, ...) {
		va_list args;
		va_start(args, format);
		std::vsnprintf(s, sizeof(s), format, args);
		va_end(args);
		AppendCause();
	}
	void AppendCause() {
		const auto cause = TakeCause();
		if (cause == Cause::None) return;
		size_t used = 0;
		while (used < sizeof(s) && s[used] != '\0') ++used;
		std::snprintf(s + used, sizeof(s) - used, ".e=%s", CauseName(cause));
	}
};

inline uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}

// Puts `text` around / before `record` (the command that opens the encoder). The time spent in the label calls (not in
// `record`) is added to Stats().ns. A test backend replaces `production` when set.
// Mutant GPULABEL_AFTER_CMD: the label comes after the command, so it names the next encoder.
template <class F>
void RunText(const Backend& production, void* cb, const char* text, F&& record) {
	const auto*    test_backend = TestBackend();
	const Backend& backend      = test_backend != nullptr ? *test_backend : production;
	const bool     insert       = Mode() == 2;
	uint64_t       spent        = 0;
	const auto     mark         = [&] {
        const auto t0 = NowNs();
        if (insert) backend.insert(cb, text);
        else backend.begin(cb, text);
        spent += NowNs() - t0;
	};
	const auto unmark = [&] {
		if (insert) return;
		const auto t0 = NowNs();
		backend.end(cb);
		spent += NowNs() - t0;
	};
#ifdef KYTY_GPULABEL_MUTANT_AFTER_CMD
	record();
	mark();
	unmark();
#else
	mark();
	record();
	unmark();
#endif
	auto& stats = Stats();
	stats.labels.fetch_add(1, std::memory_order_relaxed);
	stats.ns.fetch_add(spent, std::memory_order_relaxed);
}

// The one call the sites use: no label work at all when the class is off; otherwise format and wrap.
template <class F, class... Args>
void Run(const Backend& production, void* cb, Class c, F&& record, const char* format, Args... args) {
	if (!Active(c)) {
		record();
		return;
	}
	const auto t0 = NowNs();
	const Text text(format, args...);
	Stats().ns.fetch_add(NowNs() - t0, std::memory_order_relaxed);
	RunText(production, cb, text.s, record);
}

} // namespace Libs::Graphics::GpuLabel

#endif
