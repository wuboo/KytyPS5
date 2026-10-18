#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PVINDEXBACKING_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PVINDEXBACKING_H_

// S417 (KYTY_PV_IDX_BACKING=0|1|shadow, unset = 0): the VS-store scan of draw A (TryPvVsStoreDraw) hashes and scans the index bytes through the
// guest address. When the page of those bytes is write-protected because the GPU wrote something ELSE on it (RESULT-415: `dirty=0 page_dirty=1`),
// the read faults and drains the queue (~9 ms/flip). The bytes themselves are clean (the byte tracker has no GPU write in the range), so the
// physical backing holds them and Memory::TryReadBacking returns them without the page protection and without a wait.
//   0       the read through the guest address, bit for bit as before
//   1       clean bytes are copied from the backing (the copy is hashed and scanned); dirty bytes, host copies, an unreadable backing: as before
//   shadow  as 1 plus the old read of the same bytes; the two are compared (eq / ne counters); the draw uses the old read
// Pure part; tests/PvIndexBackingTests.cpp pins it.

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::PvIndexBacking {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_PV_IDX_BACKING"));
	return mode;
}

enum class Source : uint8_t {
	Raw,     // read through the guest address (drains the queue when the page is protected)
	Backing, // copy from the physical backing; the copy is what the scan sees
	Shadow,  // both; the draw uses Raw, the copy is compared with it
};

// host_data: the indices are a host copy (no guest memory to read). dirty: the byte tracker has a GPU write in the range.
constexpr Source Decide(Mode mode, bool host_data, bool dirty) {
	if (host_data || mode == Mode::Off) return Source::Raw;
#if defined(KYTY_PVINDEXBACKING_MUTANT_BACKING_IGNORES_DIRTY)
	dirty = false;
#endif
	if (dirty) return Source::Raw;
#if defined(KYTY_PVINDEXBACKING_MUTANT_SHADOW_SKIPS_RAW)
	return Source::Backing;
#else
	return mode == Mode::Shadow ? Source::Shadow : Source::Backing;
#endif
}

// The first byte where the two reads differ; `size` when they are equal.
inline uint64_t FirstDiff(const void* a, const void* b, uint64_t size) {
	const auto* pa = static_cast<const uint8_t*>(a);
	const auto* pb = static_cast<const uint8_t*>(b);
	for (uint64_t i = 0; i < size; i++) {
#if defined(KYTY_PVINDEXBACKING_MUTANT_FIRSTDIFF_SKIPS_LAST)
		if (i + 1 == size) break;
#endif
		if (pa[i] != pb[i]) return i;
	}
	return size;
}

} // namespace Libs::Graphics::PvIndexBacking

#endif
