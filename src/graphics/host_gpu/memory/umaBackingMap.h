#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMABACKINGMAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMABACKINGMAP_H_

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <map>

namespace Libs::Graphics {

// Mirror of the guest-address -> direct-memory-backing mappings, as seen by the GPU (UMA).
// A cache page is usable for direct GPU access only when the whole page lies inside one run that
// is contiguous both in guest addresses and in the backing; everything else keeps the shadow-buffer
// path. Not thread safe: the owner serializes access.
template <uint32_t PageBits>
class UmaBackingMap {
public:
	static constexpr uint64_t PageSize = uint64_t {1} << PageBits;

	struct Run {
		uint64_t size           = 0;
		uint64_t backing_offset = 0;
	};

	// Maps [vaddr, vaddr + size) onto backing_offset; replaces whatever was there.
	void Map(uint64_t vaddr, uint64_t size, uint64_t backing_offset) {
		if (size == 0) {
			return;
		}
		Unmap(vaddr, size);
		auto it = m_runs.emplace(vaddr, Run {size, backing_offset}).first;
		if (it != m_runs.begin()) {
			auto prev = std::prev(it);
			if (Joins(prev, it)) {
				prev->second.size += it->second.size;
				m_runs.erase(it);
				it = prev;
			}
		}
		auto next = std::next(it);
		if (next != m_runs.end() && Joins(it, next)) {
			it->second.size += next->second.size;
			m_runs.erase(next);
		}
	}

	void Unmap(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return;
		}
		const auto end = vaddr + size;
		auto       it  = m_runs.upper_bound(vaddr);
		if (it != m_runs.begin()) {
			--it;
		}
		while (it != m_runs.end() && it->first < end) {
			const auto start     = it->first;
			const auto run       = it->second;
			const auto run_end   = start + run.size;
			if (run_end <= vaddr) {
				++it;
				continue;
			}
			it = m_runs.erase(it);
			if (start < vaddr) {
				it = std::next(m_runs.emplace(start, Run {vaddr - start, run.backing_offset}).first);
			}
			if (end < run_end) {
				it = std::next(m_runs
				                   .emplace(end, Run {run_end - end,
				                                      run.backing_offset + (end - start)})
				                   .first);
			}
		}
	}

	// Backing offset of the page that starts at page_vaddr, when the whole page is directly usable.
	[[nodiscard]] bool PageOffset(uint64_t page_vaddr, uint64_t* backing_offset) const {
		auto it = m_runs.upper_bound(page_vaddr);
		if (it == m_runs.begin()) {
			return false;
		}
		--it;
		if (page_vaddr + PageSize > it->first + it->second.size) {
			return false;
		}
		if (backing_offset != nullptr) {
			*backing_offset = it->second.backing_offset + (page_vaddr - it->first);
		}
		return true;
	}

	// Backing offset of vaddr when every byte of [vaddr, vaddr + size) lies in one contiguous run
	// (a descriptor view needs bytes, not whole pages).
	[[nodiscard]] bool RangeOffset(uint64_t vaddr, uint64_t size, uint64_t* backing_offset) const {
		if (size == 0 || vaddr + size < vaddr) {
			return false;
		}
		auto it = m_runs.upper_bound(vaddr);
		if (it == m_runs.begin()) {
			return false;
		}
		--it;
		if (vaddr + size > it->first + it->second.size) {
			return false;
		}
		if (backing_offset != nullptr) {
			*backing_offset = it->second.backing_offset + (vaddr - it->first);
		}
		return true;
	}

	// Calls fn(page_vaddr, backing_offset) for every directly usable page that intersects the range.
	template <typename Fn>
	void ForEachPage(uint64_t vaddr, uint64_t size, Fn&& fn) const {
		if (size == 0) {
			return;
		}
		const auto end = vaddr + size;
		auto       it  = m_runs.upper_bound(vaddr);
		if (it != m_runs.begin()) {
			--it;
		}
		for (; it != m_runs.end() && it->first < end; ++it) {
			const auto run_end = it->first + it->second.size;
			auto       first   = std::max(vaddr, it->first) & ~(PageSize - 1);
			if (first < it->first) {
				first += PageSize;
			}
			const auto last = std::min(end + PageSize - 1, run_end) & ~(PageSize - 1);
			for (auto page = first; page < last; page += PageSize) {
				fn(page, it->second.backing_offset + (page - it->first));
			}
		}
	}

	// Calls fn(start, end) for the sub-ranges of [vaddr, vaddr + size) that are NOT directly usable.
	template <typename Fn>
	void ForEachUncovered(uint64_t vaddr, uint64_t size, Fn&& fn) const {
		if (size == 0) {
			return;
		}
		const auto end    = vaddr + size;
		auto       cursor = vaddr;
		auto       it     = m_runs.upper_bound(vaddr);
		if (it != m_runs.begin()) {
			--it;
		}
		for (; it != m_runs.end() && it->first < end; ++it) {
			auto cover_begin = (it->first + PageSize - 1) & ~(PageSize - 1);
			auto cover_end   = (it->first + it->second.size) & ~(PageSize - 1);
			if (cover_end <= cover_begin || cover_end <= cursor) {
				continue;
			}
			cover_begin = std::max(cover_begin, cursor);
			if (cover_begin >= end) {
				break;
			}
			if (cursor < cover_begin) {
				fn(cursor, cover_begin);
			}
			cursor = std::min(cover_end, end);
		}
		if (cursor < end) {
			fn(cursor, end);
		}
	}

	// Calls fn(vaddr, size, backing_offset) for every run.
	template <typename Fn>
	void ForEachRun(Fn&& fn) const {
		for (const auto& [vaddr, run]: m_runs) {
			fn(vaddr, run.size, run.backing_offset);
		}
	}

	// Calls fn(vaddr, size, backing_offset) for the parts of the runs inside [vaddr, vaddr + size).
	template <typename Fn>
	void ForEachRunIn(uint64_t vaddr, uint64_t size, Fn&& fn) const {
		if (size == 0) {
			return;
		}
		const auto end = vaddr + size;
		auto       it  = m_runs.upper_bound(vaddr);
		if (it != m_runs.begin()) {
			--it;
		}
		for (; it != m_runs.end() && it->first < end; ++it) {
			const auto lo = std::max(vaddr, it->first);
			const auto hi = std::min(end, it->first + it->second.size);
			if (lo < hi) {
				fn(lo, hi - lo, it->second.backing_offset + (lo - it->first));
			}
		}
	}

	[[nodiscard]] size_t RunCount() const { return m_runs.size(); }

private:
	using Runs = std::map<uint64_t, Run>;

	static bool Joins(typename Runs::const_iterator a, typename Runs::const_iterator b) {
		return a->first + a->second.size == b->first &&
		       a->second.backing_offset + a->second.size == b->second.backing_offset;
	}

	Runs m_runs;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMABACKINGMAP_H_
