#ifndef KYTY_GRAPHICS_PRESENTATION_RECCOUNTERS_H_
#define KYTY_GRAPHICS_PRESENTATION_RECCOUNTERS_H_

// S613 (KYTY_REC_COUNTERS=0|buffered, default 0): the recorder (Thread_Gpu) adds its frames.csv counters into a table of its own, as the CP front
// thread does since S549, instead of a relaxed fetch_add on the global array per FrameTiming::Add. The global array is shared by every thread that
// counts (the workers' AddAlways per job, the emitter, guest-thread faults, the front's flushes, the presentation thread's Set): a lock-prefixed add
// on a line another core wrote last is a cache miss on the recorder, about 5 745 draws x ~40 adds per flip (ETW S611a: FrameTiming::Add 0,89 ms/flip
// exclusive on the recorder). The table goes to the global array at the top of every Thread_Gpu loop iteration (after each stream, command or
// wait), so a frames.csv row may show a stream's counts one row later; sums over a window do not change.
// FrameTiming::Get on a buffered thread adds its own pending part (the recorder computes seg_record from Get(cp_wait_*) at every submit, so the
// columns stay exact). Get from another thread sees the global array only: the diagnostics that compare a recorder counter across a guest-thread
// call (KYTY_DIAG_RFOP / KYTY_DIAG_GUEST_FAULT / KYTY_DIAG_GUEST_STALE, ReadbackEagerHits) lag by up to one loop iteration; production does not use them.
// Off: unchanged (Add as before, nothing buffered on Thread_Gpu).
//
// Pure part (the table helpers and the mode); tests/RecDietTests.cpp pins it. Mutants (WILL_FAIL): KYTY_RECCNT_MUTANT_GET_IGNORES_LOCAL (Get misses the
// thread's pending part), KYTY_RECCNT_MUTANT_FLUSH_NO_ZERO (a flush leaves the table: counted twice), KYTY_RECCNT_MUTANT_FLUSH_SKIPS_LAST (the last
// counter never moves), KYTY_RECCNT_MUTANT_DEFAULT_ON (an unset variable buffers).

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::RecCounters {

enum class Mode : uint8_t { Off = 0, Buffered = 1 };

[[nodiscard]] inline Mode ParseMode(const char* text) {
#if defined(KYTY_RECCNT_MUTANT_DEFAULT_ON)
	if (text == nullptr || text[0] == '\0') return Mode::Buffered;
#else
	if (text == nullptr || text[0] == '\0') return Mode::Off;
#endif
	if (std::strcmp(text, "1") == 0 || std::strcmp(text, "buffered") == 0) return Mode::Buffered;
	return Mode::Off;
}

[[nodiscard]] inline const char* ModeName(Mode mode) { return mode == Mode::Buffered ? "buffered" : "off"; }

// The first call logs `Environment: KYTY_REC_COUNTERS=<value> mode=<off|buffered>`.
inline Mode EnvMode() {
	static const Mode mode = [] {
		const char* value = std::getenv("KYTY_REC_COUNTERS");
		const Mode  m     = ParseMode(value);
		std::fprintf(stderr, "Environment: KYTY_REC_COUNTERS=%s mode=%s\n", value != nullptr ? value : "", ModeName(m));
		std::fflush(stderr);
		return m;
	}();
	return mode;
}

// Moves every non-zero entry of a thread's table into the global array (relaxed adds) and zeroes it; returns how many entries moved.
template <class Global>
uint32_t FlushInto(uint64_t* local, Global& global, size_t count) {
	uint32_t moved = 0;
#if defined(KYTY_RECCNT_MUTANT_FLUSH_SKIPS_LAST)
	if (count != 0) count--;
#endif
	for (size_t i = 0; i < count; i++) {
		if (local[i] == 0) continue;
		global[i].fetch_add(local[i], std::memory_order_relaxed);
#if !defined(KYTY_RECCNT_MUTANT_FLUSH_NO_ZERO)
		local[i] = 0;
#endif
		moved++;
	}
	return moved;
}

// The value a thread sees: the global entry plus its own pending part (local == nullptr: not a buffered thread).
template <class Global>
[[nodiscard]] uint64_t Read(const Global& global, const uint64_t* local, size_t index) {
	const uint64_t value = global[index].load(std::memory_order_relaxed);
#if defined(KYTY_RECCNT_MUTANT_GET_IGNORES_LOCAL)
	(void)local;
	return value;
#else
	return local != nullptr ? value + local[index] : value;
#endif
}

} // namespace Libs::Graphics::RecCounters

#endif /* KYTY_GRAPHICS_PRESENTATION_RECCOUNTERS_H_ */
