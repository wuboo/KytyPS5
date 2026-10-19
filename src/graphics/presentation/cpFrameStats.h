#ifndef KYTY_GRAPHICS_PRESENTATION_CPFRAMESTATS_H_
#define KYTY_GRAPHICS_PRESENTATION_CPFRAMESTATS_H_

// S441 (step 0 of verify-442): what a frame really costs the command processor (CP) and how far the guest runs ahead of it, so that a -X ms saving
// on the CP can be turned into a frame-rate estimate. Pure bookkeeping, no Vulkan and no FrameTiming dependency (tests/CpFrameStatsTests.cpp):
//   FrameClock   interval between consecutive marks (a mark = the CP prepares a flip, Sync::PrepareVideoOutFlip); the first mark gives nothing.
//   Samples      the intervals of one reporting segment with p50/p90/p99/max (nearest rank).
//   PerFlipMin   the smallest value noted since the last flip: guest lead = (CP picks up a submission) - (the guest enqueued it); its minimum per
//                flip is the slack of the guest, the time it could have been late without the CP idling.
//   ThreadCpu    per-thread CPU time deltas between two snapshots of the process threads, split into the CP thread and the others by name.
// Mutants (KYTY_CPFRAME_MUTANT_*): PERCENTILE_LOW (p99 reads one rank too low), MIN_IS_MAX (PerFlipMin keeps the largest), FIRST_MARK_COUNTS (the first mark
// gives an interval from time 0), THREAD_ABSOLUTE (ThreadCpu reports the cumulative time, not the delta).

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::CpFrameStats {

class FrameClock {
public:
	std::optional<uint64_t> Mark(uint64_t now_ns) {
		std::optional<uint64_t> interval;
#ifdef KYTY_CPFRAME_MUTANT_FIRST_MARK_COUNTS
		interval = now_ns - m_last;
#else
		if (m_have && now_ns >= m_last) interval = now_ns - m_last;
#endif
		m_last = now_ns;
		m_have = true;
		return interval;
	}

private:
	uint64_t m_last = 0;
	bool     m_have = false;
};

class Samples {
public:
	void Add(uint64_t value) {
		m_values.push_back(value);
		m_sum += value;
	}
	[[nodiscard]] size_t   Size() const { return m_values.size(); }
	[[nodiscard]] uint64_t Sum() const { return m_sum; }
	[[nodiscard]] uint64_t Max() const { return m_values.empty() ? 0 : *std::max_element(m_values.begin(), m_values.end()); }
	// Nearest rank: the smallest value with at least p percent of the samples at or below it; 0 with no samples.
	[[nodiscard]] uint64_t Percentile(unsigned p) const {
		if (m_values.empty()) return 0;
		std::vector<uint64_t> sorted = m_values;
		std::sort(sorted.begin(), sorted.end());
		size_t rank = (static_cast<size_t>(p) * sorted.size() + 99) / 100; // ceil(p * n / 100), 1-based
		if (rank == 0) rank = 1;
#ifdef KYTY_CPFRAME_MUTANT_PERCENTILE_LOW
		rank = rank > 1 ? rank - 1 : rank;
#endif
		return sorted[std::min(rank, sorted.size()) - 1];
	}
	void Clear() {
		m_values.clear();
		m_sum = 0;
	}

private:
	std::vector<uint64_t> m_values;
	uint64_t              m_sum = 0;
};

class PerFlipMin {
public:
	void Note(uint64_t value) {
#ifdef KYTY_CPFRAME_MUTANT_MIN_IS_MAX
		m_value = m_value ? std::max(*m_value, value) : value;
#else
		m_value = m_value ? std::min(*m_value, value) : value;
#endif
	}
	// The minimum since the previous call (nothing when no value was noted); starts a new flip.
	std::optional<uint64_t> TakeFlip() {
		auto out = m_value;
		m_value.reset();
		return out;
	}

private:
	std::optional<uint64_t> m_value;
};

struct ThreadSample {
	uint64_t    id = 0;
	std::string name;
	uint64_t    cpu_ns = 0; // cumulative user + system time of the thread
};

class ThreadCpu {
public:
	struct Delta {
		uint64_t                                      cp    = 0; // threads called cp_name
		uint64_t                                      other = 0; // all the rest
		std::vector<std::pair<std::string, uint64_t>> by_name;   // every name, largest first
	};

	// The first snapshot of a thread gives 0 (it is only the baseline); a thread that left is forgotten.
	Delta Update(std::span<const ThreadSample> snapshot, const std::string& cp_name) {
		Delta                                    delta;
		std::unordered_map<uint64_t, uint64_t>   now;
		std::unordered_map<std::string, uint64_t> names;
		for (const auto& t: snapshot) {
			now[t.id]       = t.cpu_ns;
			const auto prev = m_last.find(t.id);
			(void)prev;
			uint64_t   d    = 0;
#ifdef KYTY_CPFRAME_MUTANT_THREAD_ABSOLUTE
			d = t.cpu_ns;
#else
			if (prev != m_last.end() && t.cpu_ns >= prev->second) d = t.cpu_ns - prev->second;
#endif
			(t.name == cp_name ? delta.cp : delta.other) += d;
			names[t.name] += d;
		}
		m_last.swap(now);
		delta.by_name.assign(names.begin(), names.end());
		std::sort(delta.by_name.begin(), delta.by_name.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
		return delta;
	}

private:
	std::unordered_map<uint64_t, uint64_t> m_last;
};

} // namespace Libs::Graphics::CpFrameStats

#endif /* KYTY_GRAPHICS_PRESENTATION_CPFRAMESTATS_H_ */
