#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_S459LOG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_S459LOG_H_

// S521 DIAG. KYTY_DIAG_S459_FLIP gates every line written here.
// Unset: one getenv on the first call, then one bool check, no file and no frame counter.
// A single number keeps the S519 window present..present+2. A comma list logs those
// present frames only (frame_flip_<n> uses the same index).

#include "graphics/presentation/frameTiming.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::S459Log {

struct Gate {
	bool on   = false;
	bool list = false;
	long base = -1;
	long frames[8] {};
	int  n = 0;

	[[nodiscard]] bool Hit(long present) const {
		if (!on) return false;
		if (!list) return present >= base && present < base + 3;
		for (int i = 0; i < n; i++) {
			if (frames[i] == present) return true;
		}
		return false;
	}
};

inline Gate Parse(const char* value) {
	Gate g;
	if (value == nullptr || value[0] == '\0') return g;
	if (std::strchr(value, ',') == nullptr) {
		g.base = std::atol(value);
		g.on   = g.base >= 0;
		return g;
	}
	g.list         = true;
	const char* p  = value;
	while (*p != '\0' && g.n < 8) {
		char* end = nullptr;
		const long frame = std::strtol(p, &end, 10);
		if (end == p) break;
		g.frames[g.n++] = frame;
		p               = end;
		if (*p == ',') p++;
	}
	g.on = g.n > 0;
	return g;
}

inline const Gate& Get() {
	static const Gate gate = Parse(std::getenv("KYTY_DIAG_S459_FLIP"));
	return gate;
}

inline bool Hit() {
	const Gate& gate = Get();
	if (!gate.on) return false;
	return gate.Hit(static_cast<long>(FrameTiming::PresentedFrames() + 1));
}

inline long Present() { return static_cast<long>(FrameTiming::PresentedFrames() + 1); }
inline long Guest() { return static_cast<long>(FrameTiming::GuestFlips() + 1); }

struct LogState {
	FILE*  file   = nullptr;
	bool   opened = false;
	size_t bytes  = 0;
};

inline LogState& State() {
	static LogState state;
	return state;
}

inline constexpr size_t kCap = 40u * 1024u * 1024u;

// One fwrite of an already formatted line. Shared by every TU (function-local static).
inline void Write(const char* text, size_t n) {
	if (!Hit() || text == nullptr || n == 0) return;
	LogState& state = State();
	if (!state.opened) {
		state.opened        = true;
		const char* path    = std::getenv("KYTY_DIAG_S459_FILE");
		const char* use     = (path != nullptr && path[0] != '\0') ? path : "s459-draws.txt";
		state.file          = std::fopen(use, "a");
	}
	if (state.file == nullptr || state.bytes >= kCap) return;
	if (n > kCap - state.bytes) n = kCap - state.bytes;
	std::fwrite(text, 1, n, state.file);
	std::fflush(state.file);
	state.bytes += n;
}

inline void Linef(const char* fmt, ...) {
	if (!Hit()) return;
	char    buf[2048];
	va_list ap;
	va_start(ap, fmt);
	const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (n <= 0) return;
	const size_t take = static_cast<size_t>(n) < sizeof(buf) ? static_cast<size_t>(n) : sizeof(buf) - 1;
	Write(buf, take);
}

} // namespace Libs::Graphics::S459Log

#endif
