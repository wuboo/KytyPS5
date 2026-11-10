#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_S459LOG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_S459LOG_H_

// S521 DIAG. KYTY_DIAG_S459_FLIP gates every line written here.
// Unset: one getenv on the first call, then one bool check, no file and no frame counter.
// A single number keeps the S519 window present..present+2. A comma list logs those
// present frames only (frame_flip_<n> uses the same index).

#include "graphics/presentation/frameTiming.h"

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::S459Log {

struct Gate {
	bool on    = false;
	bool list  = false;
	bool guest = false;
	long base  = -1;
	long glo   = 0;
	long ghi   = -1;
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
	// g4240-4400: the card moves in present flips. Watch that guest span without logging every draw.
	if (value[0] == 'g') {
		char* end = nullptr;
		const long lo = std::strtol(value + 1, &end, 10);
		if (end == value + 1 || *end != '-') return g;
		char* end2 = nullptr;
		const long hi = std::strtol(end + 1, &end2, 10);
		if (end2 == end + 1 || hi < lo || hi - lo > 400) return g;
		g.guest = true;
		g.glo   = lo;
		g.ghi   = hi;
		return g;
	}
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

// Guest span from KYTY_DIAG_S459_FLIP=g<lo>-<hi>. Present lists stay on Hit().
inline bool Card() {
	const Gate& gate = Get();
	if (!gate.guest) return false;
	const long guest = static_cast<long>(FrameTiming::GuestFlips() + 1);
	return guest >= gate.glo && guest <= gate.ghi;
}

inline bool Watch() { return Hit() || Card(); }

// Up to four present flips asked for while a card frame is still being drawn.
inline std::atomic<long>* DumpSlots() {
	static std::atomic<long> slots[4] {};
	return slots;
}

inline void RequestDump(long present) {
	if (present <= 0) return;
	auto* slots = DumpSlots();
	for (int i = 0; i < 4; i++) {
		if (slots[i].load(std::memory_order_acquire) == present) return;
	}
	for (int i = 0; i < 4; i++) {
		long expected = 0;
		if (slots[i].compare_exchange_strong(expected, present, std::memory_order_release, std::memory_order_relaxed)) return;
		if (expected == present) return;
	}
}

inline bool DumpRequested(long present) {
	auto* slots = DumpSlots();
	for (int i = 0; i < 4; i++) {
		if (slots[i].load(std::memory_order_acquire) == present) return true;
	}
	return false;
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
inline void Emit(const char* text, size_t n) {
	if (text == nullptr || n == 0) return;
	LogState& state = State();
	if (!state.opened) {
		state.opened     = true;
		const char* path = std::getenv("KYTY_DIAG_S459_FILE");
		const char* use  = (path != nullptr && path[0] != '\0') ? path : "s459-draws.txt";
		state.file       = std::fopen(use, "a");
	}
	if (state.file == nullptr || state.bytes >= kCap) return;
	if (n > kCap - state.bytes) n = kCap - state.bytes;
	std::fwrite(text, 1, n, state.file);
	std::fflush(state.file);
	state.bytes += n;
}

inline void Write(const char* text, size_t n) {
	if (!Hit()) return;
	Emit(text, n);
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
	Emit(buf, take);
}

// Present list or the card's guest span. The guest span does not turn Hit() on.
inline void LineWatch(const char* fmt, ...) {
	if (!Watch()) return;
	char    buf[2048];
	va_list ap;
	va_start(ap, fmt);
	const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (n <= 0) return;
	const size_t take = static_cast<size_t>(n) < sizeof(buf) ? static_cast<size_t>(n) : sizeof(buf) - 1;
	Emit(buf, take);
}

// S522. The file must be set. Not gated on the present list: a handful of metadata
// writes are recorded for the whole run, and every write while the present list hits.
inline void LineAlways(const char* fmt, ...) {
	const char* path = std::getenv("KYTY_DIAG_S459_FILE");
	if (path == nullptr || path[0] == '\0' || fmt == nullptr) return;
	static int lines = 0;
	if (lines >= 800) return;
	char    buf[768];
	va_list ap;
	va_start(ap, fmt);
	const int n = std::vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	if (n <= 0) return;
	const size_t take = static_cast<size_t>(n) < sizeof(buf) ? static_cast<size_t>(n) : sizeof(buf) - 1;
	LogState& state = State();
	if (!state.opened) {
		state.opened = true;
		state.file   = std::fopen(path, "a");
	}
	if (state.file == nullptr || state.bytes >= kCap) return;
	std::fwrite(buf, 1, take, state.file);
	std::fflush(state.file);
	state.bytes += take;
	lines++;
}

// Guest metadata of the 960 and 1920 views of 0x53aa00000 sits in this window (S521).
inline bool HitsMetaWatch(uint64_t addr, uint64_t size) {
	constexpr uint64_t lo = 0x570520000ull;
	constexpr uint64_t hi = 0x570540000ull;
	if (size == 0) return false;
	if (addr > UINT64_MAX - size) return addr < hi;
	return addr < hi && addr + size > lo;
}

// True when the half-open range covers the first byte of the 1920 metadata slice.
inline bool Hits1920Byte(uint64_t addr, uint64_t size) {
	constexpr uint64_t byte = 0x570526000ull;
	if (size == 0 || addr > byte) return false;
	if (addr > UINT64_MAX - size) return true;
	return addr + size > byte;
}

// First four writes of each metadata address, plus every write while the present list hits.
// hit1920=1 means this write stores over the 1920 clear byte.
inline void NoteMetaWrite(const char* kind, uint64_t color, uint64_t addr, uint64_t size, uint32_t value, uint32_t w,
                          uint32_t h) {
	const bool color_hit = color == 0x53aa00000ull;
	if (!color_hit && !HitsMetaWatch(addr, size)) return;
	const bool window = Hit();
	static uint64_t seen_addr[16] {};
	static int      seen_n[16] {};
	static int      seen_count = 0;
	int             slot       = -1;
	for (int i = 0; i < seen_count; i++) {
		if (seen_addr[i] == addr) slot = i;
	}
	if (slot < 0 && seen_count < 16) {
		slot                 = seen_count;
		seen_addr[seen_count++] = addr;
	}
	if (!window) {
		if (slot < 0 || seen_n[slot] >= 4) return;
		seen_n[slot]++;
	}
	LineAlways("[s522-meta] present=%ld guest=%ld kind=%s color=0x%016" PRIx64 " meta=0x%016" PRIx64
	           " size=0x%" PRIx64 " value=0x%08" PRIx32 " hit1920=%d %ux%u\n",
	           Present(), Guest(), kind != nullptr ? kind : "?", color, addr, size, value, Hits1920Byte(addr, size) ? 1 : 0, w, h);
}

} // namespace Libs::Graphics::S459Log

#endif
