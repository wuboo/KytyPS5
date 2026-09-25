#include "graphics/presentation/frameTiming.h"

#include "common/profiler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics::FrameTiming {

namespace {

std::atomic<uint64_t> g_presented_frames {0};

FILE* OpenFrameLog() {
	const char* path = std::getenv("KYTY_FRAME_LOG");
	if (path == nullptr || *path == '\0') {
		return nullptr;
	}
	FILE* file = std::fopen(path, "w");
	if (file != nullptr) {
		std::fputs("frame,host_ns\n", file);
	}
	return file;
}

} // namespace

void OnFramePresented() {
	// Tracy runs with a manual lifetime and is only started by --profile.
	if (tracy::ProfilerAvailable()) {
		FrameMark;
	}

	// Presentation happens on a single thread, so the log needs no locking.
	static FILE* log = OpenFrameLog();

	const auto frame = g_presented_frames.fetch_add(1, std::memory_order_relaxed) + 1;
	if (log != nullptr) {
		const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		                    std::chrono::steady_clock::now().time_since_epoch())
		                    .count();
		std::fprintf(log, "%llu,%lld\n", static_cast<unsigned long long>(frame),
		             static_cast<long long>(ns));
		// The emulator usually exits through a hard stop, so keep the file current.
		std::fflush(log);
	}
}

uint64_t PresentedFrames() {
	return g_presented_frames.load(std::memory_order_relaxed);
}

} // namespace Libs::Graphics::FrameTiming
