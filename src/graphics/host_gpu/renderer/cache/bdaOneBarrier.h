#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDAONEBARRIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDAONEBARRIER_H_

// S603 (KYTY_BDA_ONE_BARRIER=0|1, default 0): port of KytyPS5 upstream commit b510053e8 by nmzik ("graphics: upload BDA sync ranges
// behind one barrier pair").
//
// Before every draw or dispatch with DMA (BDA) access, RenderContext::PrepareBda synchronizes the pages queued since the previous one.
// Without the switch every (queued range, mapped range, buffer) piece goes through SynchronizeBuffer on its own: one staging allocation,
// one EndRendering, a whole-buffer barrier before and after and one vkCmdCopyBuffer per piece. With the switch
// BufferCache::SynchronizeBda collects the CPU-dirty bytes of every piece first (MemoryTracker::ForEachUploadRange with an empty upload
// callback: the reads follow every range's protection, and a write in between faults and queues its page again), then uploads them from
// one staging allocation behind one global barrier pair, one copy command per run of copies to the same buffer.
//
// Copies are grouped as uploads: upload i owns copies [upload i-1's end, upload i's end) and goes to upload i's buffer. A piece of the
// same buffer as the previous one extends that upload (the m_buffers map is walked in address order, so the pieces of one queued range
// that fall into one buffer become one copy command).
//
// Off: PrepareBda takes the old path; only the counters bda_sync_n / bda_sync_buf_n / bda_sync_bytes are added (both paths count them).
//
// Mutants (tests/BdaOneBarrierTests.cpp, WILL_FAIL): NO_SPLIT (a piece of another buffer extends the previous upload, so its bytes land
// in the wrong buffer), FIRST_FROM_ZERO (every upload replays the copies from index 0), NO_BASE_OFFSET (the copies keep offsets relative
// to the mapped pointer, not to the staging buffer).

#include <cstdint>
#include <cstdlib>
#include <span>
#include <vector>

namespace Libs::Graphics::BdaOneBarrier {

inline bool Parse(const char* text) {
	return text != nullptr && text[0] == '1' && text[1] == 0;
}

// KYTY_BDA_ONE_BARRIER (read once; RenderContext's constructor reads it first, on the main thread).
inline bool EnvOn() {
	static const bool on = Parse(std::getenv("KYTY_BDA_ONE_BARRIER"));
	return on;
}

// Copies [previous upload's end, end) go to `buffer`.
template <typename B>
struct Upload {
	B*     buffer;
	size_t end;
};

// Appends one copy of `bytes` guest bytes to buffer offset `dst_offset`; the staging offset is the running total.
template <typename B, typename Copy>
void Append(std::vector<Upload<B>>& uploads, std::vector<Copy>& copies, uint64_t& total_size, B* buffer, uint64_t dst_offset,
            uint64_t bytes) {
#ifdef KYTY_BDAONEBARRIER_MUTANT_NO_SPLIT
	if (uploads.empty()) {
#else
	if (uploads.empty() || uploads.back().buffer != buffer) {
#endif
		uploads.push_back({buffer, copies.size()});
	}
	copies.emplace_back(total_size, dst_offset, bytes);
	uploads.back().end = copies.size();
	total_size += bytes;
}

// func(buffer, copies of that upload) per upload, in order.
template <typename B, typename Copy, typename Func>
void ForEachUpload(std::span<const Upload<B>> uploads, std::span<Copy> copies, Func&& func) {
	size_t first = 0;
	for (const auto& upload: uploads) {
#ifdef KYTY_BDAONEBARRIER_MUTANT_FIRST_FROM_ZERO
		first = 0;
#endif
		func(*upload.buffer, copies.subspan(first, upload.end - first));
		first = upload.end;
	}
}

// Fills the staging memory: read(guest address, destination, bytes) per copy, then rebases every copy's source offset from the mapped
// pointer to the staging buffer (`base_offset` = offset of `mapped` in it). B needs CpuAddress().
template <typename B, typename Copy, typename Read>
void Fill(std::span<const Upload<B>> uploads, std::span<Copy> copies, uint8_t* mapped, uint64_t base_offset, Read&& read) {
	ForEachUpload<B, Copy>(uploads, copies, [&](B& buffer, std::span<Copy> part) {
		for (auto& copy: part) {
			read(buffer.CpuAddress() + copy.dstOffset, mapped + copy.srcOffset, copy.size);
#ifndef KYTY_BDAONEBARRIER_MUTANT_NO_BASE_OFFSET
			copy.srcOffset += base_offset;
#else
			(void)base_offset;
#endif
		}
	});
}

} // namespace Libs::Graphics::BdaOneBarrier

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_BDAONEBARRIER_H_ */
