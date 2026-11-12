#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_NARROWBARRIER_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_NARROWBARRIER_H_

// S556 K-GPU (KYTY_NARROW_BARRIER=0|1|2, unset = 0 = the barriers of f99fe774 byte for byte).
// Header-only decisions; the Vulkan masks are in narrowBarrierVk.h.
//
// Level 1 (spec-safe, every hazard of the old barrier is kept):
//  - SynchronizeBuffer: the before/after pair covers the bytes the copy writes (the span of the copy
//    regions) instead of the whole cache buffer, and the stage/access masks are the ones a cache
//    buffer can be used with (it is created with transfer/uniform/index/vertex/indirect/storage/BDA
//    usage only: never an attachment, a conditional-rendering predicate or host-written), instead of
//    eAllCommands + eMemoryRead|eMemoryWrite. eAllCommands as the source means "wait for the end of the
//    pipe" (colour/depth export included) and eMemoryWrite asks for the colour/depth caches to be
//    flushed; neither can be relevant to a VkBuffer.
//  - WriteDataBuffer and the readback copy: the same masks (WriteDataBuffer was already ranged; the
//    readback "before" barrier is ranged to its copy regions).
//  - A guest dispatch that writes only buffers (and LDS, a buffer): the write-hazard barrier before it
//    keeps the full execution dependency on every stage that can touch a buffer and a memory
//    dependency on buffer writes (shader / transfer) instead of eAllCommands + eMemoryRead|Write.
//    Image hazards are not this barrier's job for such a dispatch: it writes no image, and its image
//    reads are ordered by Image::Transit exactly as for a read-only dispatch, which records no
//    barrier before itself at all. A dispatch that writes a storage image keeps the old barrier.
//  - The same for a draw that records the immediate (non-deferred) buffer-write hazard barrier:
//    shader_write_stages there come from buffer writes only.
//  - DispatchIndirect: the argument barrier's source eAllGraphics (end of pipe) becomes the stages
//    that can write a buffer (shaders and transfer); colour/depth output cannot write the arguments.
// Level 2 (adds what rests on an invariant that is not local to the barrier):
//  - the write-hazard barrier of a dispatch that writes storage images is narrowed too (relies on
//    Image::Transit recording every image write hazard, which it does: a state with write access
//    always records a barrier);
//  - the barrier after a read-only dispatch keeps only its execution dependency (no access masks):
//    with no writes in the first scope there is nothing to make available, but its visibility
//    operation could hide a writer elsewhere whose own barrier is too narrow.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

namespace Libs::Graphics::NarrowBarrier {

inline int ParseLevel(const char* text) {
	if (text == nullptr) {
		return 0;
	}
	if (std::strcmp(text, "1") == 0) {
		return 1;
	}
	if (std::strcmp(text, "2") == 0) {
		return 2;
	}
	return 0;
}

inline int Level() {
	static const int level = ParseLevel(std::getenv("KYTY_NARROW_BARRIER"));
	return level;
}

struct Span {
	uint64_t offset = 0;
	uint64_t size   = 0;
};

// The destination bytes a list of copy regions writes, as one range [min dstOffset, max end).
// Copy regions need not be sorted (UploadCopies keeps the tracker's order, which is ascending today,
// but a span that assumed it would miss bytes the day it is not). Empty list -> {0, 0}: the caller
// keeps the whole-buffer range.
template <typename Copy>
inline Span CopySpan(std::span<const Copy> copies) {
	if (copies.empty()) {
		return {};
	}
#if defined(KYTY_NARROW_MUTANT_FIRST_ONLY)
	return {copies.front().dstOffset, copies.front().size};
#endif
	uint64_t begin = copies.front().dstOffset;
	uint64_t end   = copies.front().dstOffset + copies.front().size;
#if defined(KYTY_NARROW_MUTANT_ASSUME_SORTED)
	end = copies.back().dstOffset + copies.back().size;
	return {begin, end - begin};
#endif
	for (const auto& copy: copies) {
		begin = copy.dstOffset < begin ? copy.dstOffset : begin;
		end   = copy.dstOffset + copy.size > end ? copy.dstOffset + copy.size : end;
	}
	return {begin, end - begin};
}

// Same, for readback regions (srcOffset is the cache-buffer side).
template <typename Copy>
inline Span ReadSpan(std::span<const Copy> copies) {
	if (copies.empty()) {
		return {};
	}
	uint64_t begin = copies.front().srcOffset;
	uint64_t end   = copies.front().srcOffset + copies.front().size;
	for (const auto& copy: copies) {
		begin = copy.srcOffset < begin ? copy.srcOffset : begin;
		end   = copy.srcOffset + copy.size > end ? copy.srcOffset + copy.size : end;
	}
	return {begin, end - begin};
}

// Upload / readback / WriteDataBuffer / indirect-argument barriers.
inline bool NarrowBufferSync(int level) {
	return level >= 1;
}

// The write-hazard barrier before a guest dispatch (or an immediate-barrier draw) that writes memory.
inline bool NarrowWriteHazard(int level, bool writes_storage_images) {
#if defined(KYTY_NARROW_MUTANT_IMAGES)
	(void)writes_storage_images;
	return level >= 1;
#endif
	if (level >= 2) {
		return true;
	}
	return level == 1 && !writes_storage_images;
}

// The barrier after a read-only dispatch: execution dependency only.
inline bool ExecOnlyAfterReadOnlyDispatch(int level) {
#if defined(KYTY_NARROW_MUTANT_RO_LEVEL1)
	return level >= 1;
#endif
	return level >= 2;
}

} // namespace Libs::Graphics::NarrowBarrier

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_NARROWBARRIER_H_ */
