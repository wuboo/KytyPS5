#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READHOT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_READHOT_H_

// S497, port of upstream PR #2037 (29cd9624, 4fd72a87, 346f4a85) behind KYTY_READ_HOT=1 (default off).
// A guest read remembers the page. A later GPU write to that page is copied into guest memory when the
// command buffer is flushed or an end-of-pipe / flip is recorded, and the copy lands after the GPU has
// finished that submission. The next read of a page that was not written again does not download.
// A newer GPU write, an image write, or a shader with unknown address writes supersedes a copy that is
// still in flight, so that copy does not mark the page clean.

#include "common/alignment.h"
#include "graphics/host_gpu/rangeSet.h"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace Libs::Graphics::ReadHot {

inline constexpr uint64_t kPage         = 4ull * 1024ull;
inline constexpr uint64_t kHotFlips     = 8;
inline constexpr size_t   kHotPagesMax  = 1024;
inline constexpr uint64_t kHotNotePages = 16;
inline constexpr uint64_t kPublishBudget = 1ull * 1024ull * 1024ull;

inline bool EnvOn(const char* value) {
	return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline bool Enabled() {
	static const bool on = EnvOn(std::getenv("KYTY_READ_HOT"));
	return on;
}

struct Entry {
	uint64_t vaddr      = 0;
	uint64_t size       = 0;
	bool     superseded = false;
};

using Batch = std::vector<Entry>;

inline bool Clears(const Entry& entry) {
#if defined(KYTY_READHOT_MUTANT_CONFIRM_SUPERSEDED)
	(void)entry;
	return true;
#else
	return !entry.superseded;
#endif
}

// A publish copy has taken the bytes out of the dirty set, but the tracker stays GPU-modified
// until that copy is confirmed. A read in that window finds nothing to download. Treating it as
// a noop leaves the page protected, so the fault repeats and flips stop (S497 b2).
enum class InflightRead { Absent, Wait };

inline InflightRead ReadWhilePublished(bool tracker_gpu, bool dirty, bool inflight) {
#if defined(KYTY_READHOT_MUTANT_NOOP_INFLIGHT)
	(void)tracker_gpu;
	(void)dirty;
	(void)inflight;
	return InflightRead::Absent;
#else
	if (tracker_gpu && !dirty && inflight) {
		return InflightRead::Wait;
	}
	return InflightRead::Absent;
#endif
}

class Book {
public:
	// Pages of a guest read. A span wider than 16 pages is not a hot read and is ignored.
	uint32_t Note(uint64_t vaddr, uint64_t size) {
#if defined(KYTY_READHOT_MUTANT_NO_NOTE)
		(void)vaddr;
		(void)size;
		return 0;
#else
		if (size == 0 || vaddr > UINT64_MAX - size) {
			return 0;
		}
		const auto first = Common::AlignDown(vaddr, kPage);
		const auto last  = Common::AlignUp(vaddr + size, kPage);
		if (last - first > kHotNotePages * kPage) {
			return 0;
		}
		uint32_t noted = 0;
		for (auto page = first; page < last; page += kPage) {
			if (const auto it = m_hot_pages.find(page); it != m_hot_pages.end()) {
				it->second = m_hot_flip;
				noted++;
			} else if (m_hot_pages.size() < kHotPagesMax) {
				m_hot_pages.emplace(page, m_hot_flip);
				m_hot_ranges.Add(page, kPage);
				noted++;
			}
		}
		return noted;
#endif
	}

	void Decay() {
		m_hot_flip++;
		std::erase_if(m_hot_pages, [this](const auto& page) {
			if (m_hot_flip - page.second <= kHotFlips) {
				return false;
			}
			m_hot_ranges.Subtract(page.first, kPage);
			return true;
		});
	}

	void Supersede(uint64_t vaddr, uint64_t size) {
		if (size == 0 || vaddr > UINT64_MAX - size) {
			return;
		}
		const auto first = Common::AlignDown(vaddr, kPage);
		const auto last  = Common::AlignUp(vaddr + size, kPage);
		for (auto& batch: m_inflight) {
			for (auto& entry: *batch) {
				const auto entry_first = Common::AlignDown(entry.vaddr, kPage);
				const auto entry_last  = Common::AlignUp(entry.vaddr + entry.size, kPage);
				entry.superseded       = entry.superseded || (entry_first < last && first < entry_last);
			}
		}
	}

	void SupersedeAll() {
		for (auto& batch: m_inflight) {
			for (auto& entry: *batch) {
				entry.superseded = true;
			}
		}
	}

	// A GPU write: copies of these bytes already in flight are stale, and the hot pages inside the
	// write are queued for the next publish.
	void QueueWrite(uint64_t vaddr, uint64_t size) {
		Supersede(vaddr, size);
		if (size == 0 || vaddr > UINT64_MAX - size) {
			return;
		}
		m_hot_ranges.ForEachInRange(vaddr, size, [this](uint64_t start, uint64_t end) {
			m_publish.Add(start, end - start);
		});
	}

	void Remember(const std::shared_ptr<Batch>& batch) { m_inflight.push_back(batch); }

	void Forget(const std::shared_ptr<Batch>& batch) { std::erase(m_inflight, batch); }

	void BeginPlan() { m_plan_used = 0; }

	// False: the piece does not fit in this batch and stays queued (unless the drop mutant).
	// The returned offset is the destination offset inside the download allocation.
	std::optional<uint64_t> Accept(uint64_t address, uint64_t bytes, RangeSet* keep) {
		if (bytes == 0 || m_plan_used + bytes > kPublishBudget) {
#if !defined(KYTY_READHOT_MUTANT_DROP_OVER_BUDGET)
			if (bytes != 0 && keep != nullptr) {
				keep->Add(address, bytes);
			}
#endif
			return std::nullopt;
		}
		const auto at = m_plan_used;
		m_plan_used += Common::AlignUp(bytes, 64);
		return at;
	}

	void FinishPlan(RangeSet retained, bool commit) {
		if (commit) {
			m_publish = std::move(retained);
		}
	}

	template <typename Func>
	void ForEachPublish(Func&& func) const {
		m_publish.ForEach(static_cast<Func&&>(func));
	}

	[[nodiscard]] bool   PublishEmpty() const { return m_publish.Empty(); }
	[[nodiscard]] bool   PublishIntersects(uint64_t address, uint64_t size) const { return m_publish.Intersects(address, size); }
	[[nodiscard]] size_t HotPages() const { return m_hot_pages.size(); }
	[[nodiscard]] uint64_t PlanUsed() const { return m_plan_used; }

private:
	std::map<uint64_t, uint64_t>        m_hot_pages;
	RangeSet                            m_hot_ranges;
	RangeSet                            m_publish;
	std::vector<std::shared_ptr<Batch>> m_inflight;
	uint64_t                            m_hot_flip  = 0;
	uint64_t                            m_plan_used = 0;
};

} // namespace Libs::Graphics::ReadHot

#endif
