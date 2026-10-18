#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CPREADBACKING_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CPREADBACKING_H_

// S419 (KYTY_CP_READ_BACKING=0|1|shadow, unset = 0): the CP thread reads small guest tables itself (the V# table and the attribute table of
// ShaderGetStaticVertexInputInfo, RESULT-418: `site=draw_shaders dirty=0 page_dirty=1`). When the page of the table is write-protected because
// the GPU wrote something ELSE on it, the read faults and drains the queue (wait_us up to 8 ms). The bytes themselves are current in the physical
// backing when the byte tracker has no GPU write in the range AND no GPU->host copy of the range is still in flight (S418).
//   0       the read through the guest address, bit for bit as before
//   1       a readable range is copied from the backing; dirty / in-flight ranges, a non-GPU thread, an unreadable backing: as before
//   shadow  as 1 plus the old read of the same bytes; the two are compared (eq / ne counters); the caller uses the old read
// Pure part; tests/CpReadBackingTests.cpp pins it.

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::CpReadBacking {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_CP_READ_BACKING"));
	return mode;
}

enum class Source : uint8_t {
	Raw,     // read through the guest address (drains the queue when the page is protected)
	Backing, // copy from the physical backing
	Shadow,  // both; the caller uses Raw, the copy is compared with it
};

// dirty: the byte tracker has a GPU write in the range. inflight: a GPU->host copy of the range is recorded but not published.
constexpr Source Decide(Mode mode, bool dirty, bool inflight) {
	if (mode == Mode::Off) return Source::Raw;
#if defined(KYTY_CPREADBACKING_MUTANT_READ_BACKING_IGNORES_DIRTY)
	dirty = false;
#endif
#if defined(KYTY_CPREADBACKING_MUTANT_READ_BACKING_IGNORES_INFLIGHT)
	inflight = false;
#endif
	if (dirty || inflight) return Source::Raw;
#if defined(KYTY_CPREADBACKING_MUTANT_SHADOW_SKIPS_RAW)
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
#if defined(KYTY_CPREADBACKING_MUTANT_FIRSTDIFF_SKIPS_LAST)
		if (i + 1 == size) break;
#endif
		if (pa[i] != pb[i]) return i;
	}
	return size;
}

// Call sites of CpGuestRead (indexes of the per-site statistics).
enum class Site : uint8_t { VertexAttribTable, VertexBufferTable, Count };

constexpr const char* SiteName(Site site) {
	switch (site) {
		case Site::VertexAttribTable: return "vertex_attrib_table";
		case Site::VertexBufferTable: return "vertex_buffer_table";
		default: return "?";
	}
}

} // namespace Libs::Graphics::CpReadBacking

#endif
