#ifndef KYTY_COMMON_DIAGOUT_H_
#define KYTY_COMMON_DIAGOUT_H_

// S477: the printf direction of the benchmark runs is Silent, so LOGF lines of the diagnostics never reach a file. DiagOut::Write appends a line to
// "<KYTY_FRAME_LOG>.diag.txt" (and to LOGF when KYTY_FRAME_LOG is not set). Thread safe, flushed per line (the emulator ends with a hard stop).

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "common/logging/log.h"

namespace Common::DiagOut {

inline void Write(const char* line) {
	static std::mutex mutex;
	static FILE*      file = []() -> FILE* {
        const char* base = std::getenv("KYTY_FRAME_LOG");
        if (base == nullptr || base[0] == '\0') return nullptr;
        return std::fopen((std::string(base) + ".diag.txt").c_str(), "a");
	}();
	std::lock_guard lock(mutex);
	if (file == nullptr) {
		LOGF("%s", line);
		return;
	}
	std::fputs(line, file);
	std::fflush(file);
}

} // namespace Common::DiagOut

#endif
