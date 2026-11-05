#include "graphics/presentation/benchFrameLog.h"

#include "graphics/guest_gpu/command_processor/cpPipe.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace Libs::Graphics {

void BenchNoteGuestFlip() {
	const char* path = std::getenv("KYTY_FRAME_LOG");
	if (path == nullptr || path[0] == '\0') return;
	static FILE*    file   = nullptr;
	static bool     opened = false;
	static uint64_t frames = 0;
	static uint64_t flips  = 0;
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
	if (file == nullptr) return;
	++frames;
	++flips;
	const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                    std::chrono::steady_clock::now().time_since_epoch())
	                    .count();
	std::fprintf(file, "%llu,%lld,%llu,0,0,0,0,0,%llu\n", static_cast<unsigned long long>(frames),
	             static_cast<long long>(ns), static_cast<unsigned long long>(flips),
	             static_cast<unsigned long long>(CpPipe::AcceptedJobs().load(std::memory_order_relaxed)));
	std::fflush(file);
}

} // namespace Libs::Graphics
