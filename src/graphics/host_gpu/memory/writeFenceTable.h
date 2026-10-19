#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_WRITEFENCETABLE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_WRITEFENCETABLE_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace Libs::Graphics {

// Guest range -> (timeline, tick) of the last GPU submit that may have written it in place (UMA).
// The CPU must not read the range before that tick retires on that timeline. Timeline 0 is the
// graphics master semaphore; further timelines are for async compute queues. Not thread safe: the
// owner serializes access.
class WriteFenceTable {
public:
	static constexpr uint32_t MaxTimelines = 4;
	// More entries than this collapse into one conservative entry per timeline.
	static constexpr size_t MaxEntries = 4096;

	using Ticks = std::array<uint64_t, MaxTimelines>; // newest writer tick per timeline, 0 = none

	void Record(uint64_t vaddr, uint64_t size, uint32_t timeline, uint64_t tick) {
		if (size == 0 || timeline >= MaxTimelines) {
			return;
		}
		const auto end = vaddr + size;
		// Ticks are monotonic per timeline: a newer writer of the same bytes supersedes the older.
		for (auto& entry: m_entries) {
			if (entry.timeline == timeline && entry.start >= vaddr && entry.end <= end &&
			    entry.tick <= tick) {
				entry.tick = 0;
			}
		}
		m_entries.push_back({vaddr, end, timeline, tick});
		if (m_entries.size() > MaxEntries) {
			Collapse();
		}
	}

	// Newest tick per timeline that wrote any byte of the range.
	[[nodiscard]] Ticks MaxTicks(uint64_t vaddr, uint64_t size) const {
		const auto end = vaddr + size;
		Ticks      result {};
		for (const auto& entry: m_entries) {
			if (entry.tick > result[entry.timeline] && entry.start < end && vaddr < entry.end) {
#ifdef KYTY_UMA_MUTANT_FENCE_TIMELINE
				result[0] = std::max(result[0], entry.tick); // wrong: waits on the wrong timeline
#else
				result[entry.timeline] = entry.tick;
#endif
			}
		}
		return result;
	}

	// Drops the entries whose tick has retired: is_retired(timeline, tick).
	template <typename IsRetired>
	void Prune(IsRetired&& is_retired) {
		m_entries.erase(std::remove_if(m_entries.begin(), m_entries.end(),
		                               [&](const Entry& entry) {
			                               return entry.tick == 0 ||
			                                      is_retired(entry.timeline, entry.tick);
		                               }),
		                m_entries.end());
	}

	[[nodiscard]] size_t Size() const { return m_entries.size(); }

private:
	struct Entry {
		uint64_t start;
		uint64_t end;
		uint32_t timeline;
		uint64_t tick;
	};

	void Collapse() {
		std::array<Entry, MaxTimelines> all {};
		for (uint32_t t = 0; t < MaxTimelines; ++t) {
			all[t] = {UINT64_MAX, 0, t, 0};
		}
		for (const auto& entry: m_entries) {
			if (entry.tick == 0) {
				continue;
			}
			auto& a = all[entry.timeline];
			a.start = std::min(a.start, entry.start);
			a.end   = std::max(a.end, entry.end);
			a.tick  = std::max(a.tick, entry.tick);
		}
		m_entries.clear();
		for (const auto& a: all) {
			if (a.tick != 0) {
				m_entries.push_back(a);
			}
		}
	}

	std::vector<Entry> m_entries;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_WRITEFENCETABLE_H_
