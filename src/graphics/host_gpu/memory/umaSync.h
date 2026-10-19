#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMASYNC_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMASYNC_H_

#include "graphics/host_gpu/memory/umaBackingMap.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <vector>

namespace Libs::Graphics {

// The CPU-side mirror of the guest -> backing mappings plus the rules for publishing changes to
// the GPU page table. A new mapping is published at once (the GPU cannot depend on an entry that
// did not exist). An unmapped range keeps its old entries until every submit recorded so far has
// retired: those submits may still read the old mapping, and the kernel may reuse the backing.
template <uint32_t PageBits>
class UmaPageSync {
public:
	static constexpr uint64_t PageSize = uint64_t {1} << PageBits;

	struct Event {
		uint64_t vaddr;
		uint64_t size;
		uint64_t backing_offset;
		bool     mapped;
	};

	[[nodiscard]] const UmaBackingMap<PageBits>& Map() const { return m_map; }

	// emit(first_page_vaddr, end_page_vaddr) must rewrite the page-table entries of those pages
	// from the current Map().
	template <typename Emit>
	void Apply(const std::vector<Event>& events, uint64_t current_tick, Emit&& emit) {
		for (const auto& event: events) {
			// A mapping change can complete or break neighbouring partial pages.
			const auto first = event.vaddr & ~(PageSize - 1);
			const auto last  = (event.vaddr + event.size + PageSize - 1) & ~(PageSize - 1);
			if (event.mapped) {
				m_map.Map(event.vaddr, event.size, event.backing_offset);
				emit(first, last);
				continue;
			}
			m_map.Unmap(event.vaddr, event.size);
#ifdef KYTY_UMA_MUTANT_IMMEDIATE_UNMAP
			emit(first, last);
#else
			m_pending.push_back({first, last, current_tick});
			++m_deferred_total;
#endif
		}
	}

	// Publishes the deferred unmaps whose tick has retired.
	template <typename IsRetired, typename Emit>
	void Retire(IsRetired&& is_retired, Emit&& emit) {
		while (!m_pending.empty() && is_retired(m_pending.front().tick)) {
			const auto pending = m_pending.front();
			m_pending.pop_front();
			emit(pending.first, pending.last);
		}
	}

	[[nodiscard]] size_t   PendingCount() const { return m_pending.size(); }
	[[nodiscard]] uint64_t DeferredTotal() const { return m_deferred_total; }
	// True while the page-table entries of the page may still show a deferred unmap.
	[[nodiscard]] bool IsPending(uint64_t page_vaddr) const {
		for (const auto& pending: m_pending) {
			if (page_vaddr >= pending.first && page_vaddr < pending.last) {
				return true;
			}
		}
		return false;
	}

private:
	struct Pending {
		uint64_t first;
		uint64_t last;
		uint64_t tick;
	};

	UmaBackingMap<PageBits> m_map;
	std::deque<Pending>     m_pending; // ticks are non-decreasing
	uint64_t                m_deferred_total = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMASYNC_H_
