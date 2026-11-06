#include "graphics/presentation/benchFrameLog.h"

#include "graphics/guest_gpu/command_processor/cpPipe.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

namespace {

std::atomic<uint64_t> g_guest_flips {0};
std::atomic<uint64_t> g_presented {0};

void PrintEnvOnce() {
	static bool printed = false;
	if (printed) {
		return;
	}
	printed = true;
	const auto env = [](const char* name) {
		const char* value = std::getenv(name);
		return value != nullptr && value[0] != '\0' ? value : "(unset)";
	};
	std::printf("Environment: KYTY_CP_PIPE=%s\n", env("KYTY_CP_PIPE"));
	std::printf("Environment: KYTY_PAD_SCRIPT=%s\n", env("KYTY_PAD_SCRIPT"));
	std::printf("Environment: KYTY_SHADER_LOOP_LIMIT=%s\n", env("KYTY_SHADER_LOOP_LIMIT"));
	std::printf("Environment: KYTY_FRAME_LOG=%s\n", env("KYTY_FRAME_LOG"));
	std::printf("Environment: KYTY_FRAME_DUMP_FLIPS=%s\n", env("KYTY_FRAME_DUMP_FLIPS"));
	std::fflush(stdout);
}

} // namespace

void BenchNotePresented() {
	g_presented.fetch_add(1, std::memory_order_relaxed);
}

uint64_t BenchGuestFlips() {
	return g_guest_flips.load(std::memory_order_relaxed);
}

uint64_t BenchPresentedFrames() {
	return g_presented.load(std::memory_order_relaxed);
}

void BenchNoteGuestFlip() {
	PrintEnvOnce();
	const char* path = std::getenv("KYTY_FRAME_LOG");
	const auto  flip = g_guest_flips.fetch_add(1, std::memory_order_relaxed) + 1;
	if (path == nullptr || path[0] == '\0') {
		return;
	}
	static FILE* file   = nullptr;
	static bool  opened = false;
	if (!opened) {
		opened = true;
		file   = std::fopen(path, "w");
		if (file != nullptr) {
			std::fputs("frame,host_ns,flips,cp_busy_ns,seg_record_ns,get_graphics_programs_ns,"
			           "cp_wait_gpu_ns,readback_wait_ns,pipe_accepted\n",
			           file);
			std::fflush(file);
		}
	}
	if (file == nullptr) {
		return;
	}
	const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                    std::chrono::steady_clock::now().time_since_epoch())
	                    .count();
	std::fprintf(file, "%llu,%lld,%llu,0,0,0,0,0,%llu\n", static_cast<unsigned long long>(flip),
	             static_cast<long long>(ns), static_cast<unsigned long long>(flip),
	             static_cast<unsigned long long>(CpPipe::AcceptedJobs().load(std::memory_order_relaxed)));
	std::fflush(file);
}

} // namespace Libs::Graphics
