#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_DEFERWRITEBARRIER_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_DEFERWRITEBARRIER_H_

// S511 L1 (KYTY_DEFER_WRITE_BARRIER=0|1, unset = 0). Port of bryan/main 08c3b6c0.
// A draw that writes buffers used to end the pass and record two full barriers around itself.
// PS5 orders those writes with the game's own sync packets, so the "after" barrier stays pending
// on the command buffer until EndRendering (every non-draw command and the buffer end call it)
// or until a later draw may read the written bytes. Two draws that touch a range only with
// atomics need no barrier. Not codegen: the recorded draws are the same, the barriers move.
// Mutant KYTY_DEFER_BARRIER_MUTANT_SKIP never reports a read, so the barrier is skipped entirely.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

namespace Libs::Graphics::DeferBarrier {

struct Range {
	uint64_t begin       = 0;
	uint64_t end         = 0;
	bool     atomic_only = false;
};

// One buffer binding of the draw. atomic_only: the shader touches it with atomics and neither a
// plain load (BufferResource::loaded) nor a plain store (BufferResource::stored).
struct Touch {
	uint64_t begin       = 0;
	uint64_t end         = 0;
	bool     atomic_only = false;
};

inline bool Enabled() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_DEFER_WRITE_BARRIER");
		return text != nullptr && std::strcmp(text, "1") == 0;
	}();
	return on;
}

// True when `draw` must flush the pending write barrier before it records.
// plain: vertex, index and argument ranges (they are ordinary reads). uses_dma: the shader
// reads through addresses, so the range is unknown and any pending write is a conflict.
inline bool ReadsPending(std::span<const Range> pending, std::span<const Touch> touches, bool uses_dma,
                         std::span<const std::pair<uint64_t, uint64_t>> plain) {
#if defined(KYTY_DEFER_BARRIER_MUTANT_SKIP)
	(void)pending;
	(void)touches;
	(void)uses_dma;
	(void)plain;
	return false;
#endif
	if (pending.empty()) {
		return false;
	}
	if (uses_dma) {
		return true;
	}
	const auto hits = [&](uint64_t begin, uint64_t end, bool atomic_only) {
		if (begin == 0 || end <= begin) {
			return false;
		}
		for (const auto& write: pending) {
			if (begin < write.end && write.begin < end && !(atomic_only && write.atomic_only)) {
				return true;
			}
		}
		return false;
	};
	for (const auto& touch: touches) {
		if (hits(touch.begin, touch.end, touch.atomic_only)) {
			return true;
		}
	}
	for (const auto& range: plain) {
		if (hits(range.first, range.second, false)) {
			return true;
		}
	}
	return false;
}

// Reuses an equal range. A later plain store clears atomic_only. Returns false when the caller
// must flush (the list is at the cap and this range is new).
inline bool NoteWrite(std::span<Range> pending, uint32_t& count, uint64_t begin, uint64_t end, bool atomic_only) {
	if (begin == 0 || end <= begin) {
		return true;
	}
	for (uint32_t i = 0; i < count; i++) {
		if (pending[i].begin == begin && pending[i].end == end) {
			pending[i].atomic_only = pending[i].atomic_only && atomic_only;
			return true;
		}
	}
	if (count >= pending.size()) {
		return false;
	}
	pending[count++] = Range {begin, end, atomic_only};
	return true;
}

} // namespace Libs::Graphics::DeferBarrier

#endif
