#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMAREADFENCE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMAREADFENCE_H_

#include <cstddef>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

// UMA (KYTY_UMA=1|shadow): rules for CPU reads of guest memory the GPU may write in place. Kept free of
// Vulkan so CPU tests can pin them (mutants in tests/UmaReadFenceTests.cpp).
namespace Libs::Graphics::UmaReadFence {

// "The backing may be older than what the GPU holds": a tracked GPU-modified range (copy path) or, in
// stage 3, a writer of the range that has not retired (the GPU writes the arena itself, nothing is tracked).
[[nodiscard]] constexpr bool GpuDirty(bool stage3, bool pending_writer, bool tracked_range) {
#ifdef KYTY_UMA_MUTANT_IGNORE_PENDING_WRITER
	(void)stage3;
	(void)pending_writer;
	return tracked_range;
#else
	return (stage3 && pending_writer) || tracked_range;
#endif
}

// S447: the page-quiet shortcuts of the CP/SRT reads (SRT pass, spec page cache) look only at the buffer cache's dirty and in-flight
// state. With UMA active a GPU store leaves no trace there, so the shortcuts must not be used (the read goes through the writer fence).
[[nodiscard]] constexpr bool ReadShortcutsAllowed(bool uma_active) {
#ifdef KYTY_UMA_MUTANT_SHORTCUT_IGNORES_UMA
	(void)uma_active;
	return true;
#else
	return !uma_active;
#endif
}

// Data shadow-compare (KYTY_UMA=shadow): every `every`-th read (1-based call number) is repeated after a full drain.
[[nodiscard]] constexpr bool ShouldCheck(uint64_t call_number, uint64_t every, uint64_t size, uint64_t max_size) {
#ifdef KYTY_UMA_MUTANT_CHECK_ALWAYS_BIG
	(void)every;
	(void)max_size;
	return size != 0;
#else
	return every != 0 && size != 0 && size <= max_size && call_number % every == 0;
#endif
}

// True and the first differing offset when the bytes read before and after the drain differ.
[[nodiscard]] inline bool Differs(const uint8_t* before, const uint8_t* after, size_t size, size_t* first) {
#ifdef KYTY_UMA_MUTANT_DIFF_IGNORES_TAIL
	if (size > 0) {
		--size;
	}
#endif
	if (size == 0 || std::memcmp(before, after, size) == 0) {
		return false;
	}
	size_t i = 0;
	while (before[i] == after[i]) {
		++i;
	}
	if (first != nullptr) {
		*first = i;
	}
	return true;
}

// The data shadow-compare counters as they appear at the end of the periodic [uma] line (S447: in the 426 port they were only in the
// final summary). Written even when the mode is off (uma_shadow=0), so a log parser always finds the keys.
inline int FormatShadowCounters(char* out, size_t capacity, bool shadow, uint64_t calls, uint64_t eq, uint64_t ne) {
#ifdef KYTY_UMA_MUTANT_SHADOW_LINE_OMITS_NE
	return std::snprintf(out, capacity, " uma_shadow=%d uma_dshadow_calls=%" PRIu64 " uma_dshadow_eq=%" PRIu64, shadow ? 1 : 0, calls, eq);
#else
	return std::snprintf(out, capacity, " uma_shadow=%d uma_dshadow_calls=%" PRIu64 " uma_dshadow_eq=%" PRIu64 " uma_dshadow_ne=%" PRIu64,
	                     shadow ? 1 : 0, calls, eq, ne);
#endif
}

} // namespace Libs::Graphics::UmaReadFence

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMAREADFENCE_H_
