#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_INFLIGHTRANGES_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_INFLIGHTRANGES_H_

// S418: guest ranges whose GPU->host copy is recorded but whose bytes are not in the physical backing yet.
// DownloadBufferMemory<true> subtracts the range from the GPU-modified set when it RECORDS the copy, and WriteBacking runs later in the
// deferred publish, once the GPU is done. In between HasGpuDirtyBytes is false and the backing is stale (S417 review, F3).
// Two copies of one range can be in flight at once (dirty, download, dirty again, download), so entries are counted, not merged:
// Remove(a, n) takes away exactly one Add(a, n). Add runs on the GPU thread, Remove on the priority-operation thread, hence the mutex.
// Pure part; tests/InflightRangesTests.cpp pins it.

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

namespace Libs::Graphics {

class InflightRanges final {
public:
	void Add(uint64_t address, uint64_t size) {
		if (size == 0) return;
		std::lock_guard lock(m_mutex);
		m_entries.push_back({address, address + size});
	}

	void Remove(uint64_t address, uint64_t size) {
		if (size == 0) return;
#if defined(KYTY_INFLIGHTRANGES_MUTANT_REMOVE_ALL_OVERLAPPING)
		std::lock_guard lock(m_mutex);
		std::erase_if(m_entries, [&](const Entry& e) { return e.begin < address + size && address < e.end; });
#else
		std::lock_guard lock(m_mutex);
		const auto      it = std::find_if(m_entries.begin(), m_entries.end(),
		                                  [&](const Entry& e) { return e.begin == address && e.end == address + size; });
		if (it != m_entries.end()) m_entries.erase(it);
#endif
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		if (size == 0) return false;
		std::lock_guard lock(m_mutex);
		return std::any_of(m_entries.begin(), m_entries.end(),
		                   [&](const Entry& e) { return e.begin < address + size && address < e.end; });
	}

	// S428: for a signal handler of a guest thread; false when Add / Remove holds the lock (*out untouched), never waits.
	[[nodiscard]] bool TryIntersects(uint64_t address, uint64_t size, bool* out) const {
		std::unique_lock lock(m_mutex, std::try_to_lock);
		if (!lock.owns_lock()) return false;
		*out = size != 0 && std::any_of(m_entries.begin(), m_entries.end(),
		                                [&](const Entry& e) { return e.begin < address + size && address < e.end; });
		return true;
	}

	[[nodiscard]] size_t Count() const {
		std::lock_guard lock(m_mutex);
		return m_entries.size();
	}

private:
	struct Entry {
		uint64_t begin;
		uint64_t end;
	};
	mutable std::mutex m_mutex;
	std::vector<Entry> m_entries;
};

// The backing holds the current bytes of a range when the GPU has not written it (gpu_dirty) and no copy of it is still on its way (inflight).
constexpr bool CanReadBackingDecision(bool gpu_dirty, bool inflight) {
#if defined(KYTY_INFLIGHTRANGES_MUTANT_BACKING_IGNORES_INFLIGHT)
	inflight = false;
#endif
	return !gpu_dirty && !inflight;
}

} // namespace Libs::Graphics

#endif
