#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_WFDIAG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_WFDIAG_H_

// S440 (DIAG, default off): which pages take the ~1200 write faults per flip of the guest, and does the GPU read them again in the same flip?
//   KYTY_DIAG_WRITE_FAULT=1   frames.csv columns wf_n (write faults), wf_handler_ns (time of the handler's write branch), wf_pages_distinct (pages that
//                             faulted in the last finished flip), wf_pages_every_flip (pages that faulted in >= 90 % of the last 60 flips);
//                             every kTopEveryFlips flips (and at the first flip of the log) one [wf] line per page for the 20 pages with most faults:
//                             address, owner (buf: the buffer tracker watched it, img: a texture-cache image watched it), image size, faults,
//                             flips with a fault, flips in which the same page was uploaded to the GPU after a fault.
// S443 (step 0, same switch): [wf-runs] line per finished flip every kTopEveryFlips: how many runs of consecutive faulting pages of each length the flip had
// (1, 2, 3-4, 5-8, 9-16, 17+), pages with >= 2 faults in the flip and the faults beyond the first; [wf-thread] lines: write faults and their handler time, read
// faults of guest threads and the time ReadMemory took, per thread (mach id and name), cumulative.
// This header is the pure part (tests/WfDiagTests.cpp pins it). Nothing here feeds a shader or pipeline cache key.

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::WriteFault {

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_WRITE_FAULT");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

constexpr uint64_t kPageSize      = 4096;
constexpr unsigned kWindowFlips   = 60;
constexpr unsigned kEveryMinFlips = 54; // 90 % of 60
constexpr uint64_t kTopEveryFlips = 1000;
constexpr size_t   kTopPages      = 20;
constexpr uint64_t kForgetFlips   = 256; // a page unseen for this many flips leaves the table

enum class Owner : uint8_t { None, Buffer, Image, Both };

constexpr const char* OwnerName(Owner owner) {
	switch (owner) {
		case Owner::Buffer: return "buf";
		case Owner::Image: return "img";
		case Owner::Both: return "buf+img";
		default: return "none";
	}
}

constexpr Owner MakeOwner(bool buffer, bool image) {
	return buffer ? (image ? Owner::Both : Owner::Buffer) : (image ? Owner::Image : Owner::None);
}

constexpr size_t kRunBuckets = 6;

constexpr size_t RunBucket(uint64_t length) {
	return length <= 2 ? static_cast<size_t>(length - 1) : length <= 4 ? 2 : length <= 8 ? 3 : length <= 16 ? 4 : 5;
}

struct Finished {
	uint64_t flip     = 0;
	uint64_t distinct = 0; // pages with a fault in the flip
	uint64_t every    = 0; // pages with a fault in >= kEveryMinFlips of the last kWindowFlips flips (this one included)
	uint64_t runs[kRunBuckets] = {}; // S443: runs of consecutive faulting pages by length: 1, 2, 3-4, 5-8, 9-16, 17+
	uint64_t refault_pages = 0;      // S443: pages with >= 2 faults in the flip
	uint64_t refaults      = 0;      // S443: faults beyond the first one of a page in the flip
};

struct TopEntry {
	uint64_t page = 0;
	uint64_t faults = 0;
	uint64_t flips = 0;
	uint64_t upload_flips = 0;
	uint64_t image_size = 0;
	Owner    owner = Owner::None;
};

// Faults per page; thread-safe (the handler runs on guest threads, uploads on the CP thread).
class Aggregator {
public:
	// A write fault on `page` in guest flip `flip`. Returns the statistics of the flip(s) finished by this call (flip moved on), if any.
	bool Note(uint64_t page, uint64_t flip, Owner owner, uint64_t image_size, Finished* finished) {
		std::lock_guard lock(m_lock);
		bool            rolled = false;
		if (m_started && flip > m_flip) {
			*finished = Finish();
			rolled    = true;
		}
		if (!m_started || flip > m_flip) {
			m_flip    = flip;
			m_started = true;
		}
		auto& entry = m_pages[page & ~(kPageSize - 1)];
		Advance(entry, flip);
		if (entry.last_flip != flip || entry.flips == 0) {
			entry.flips++;
			entry.flip_faults = 0;
		}
		entry.flip_faults++;
		entry.history |= 1;
		entry.last_flip = flip;
		entry.faults++;
		entry.owner      = owner;
		entry.image_size = image_size != 0 ? image_size : entry.image_size;
		entry.fault_flip = flip;
		return rolled;
	}

	// The GPU took [address, address + size) from the guest in `flip`; a page that faulted in that flip is counted once.
	void NoteUpload(uint64_t address, uint64_t size, uint64_t flip) {
		std::lock_guard lock(m_lock);
		for (uint64_t page = address & ~(kPageSize - 1); page < address + size; page += kPageSize) {
			const auto it = m_pages.find(page);
			if (it != m_pages.end() && it->second.fault_flip == flip && it->second.upload_flip != flip) {
				it->second.upload_flip = flip;
				it->second.upload_flips++;
			}
		}
	}

	[[nodiscard]] std::vector<TopEntry> Top(size_t count) const {
		std::lock_guard lock(m_lock);
		std::vector<TopEntry> all;
		all.reserve(m_pages.size());
		for (const auto& [page, entry]: m_pages) {
			all.push_back({page, entry.faults, entry.flips, entry.upload_flips, entry.image_size, entry.owner});
		}
		const auto n = std::min(count, all.size());
		std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(n), all.end(), [](const TopEntry& a, const TopEntry& b) {
			return a.faults != b.faults ? a.faults > b.faults : a.page < b.page;
		});
		all.resize(n);
		return all;
	}

	[[nodiscard]] size_t Pages() const {
		std::lock_guard lock(m_lock);
		return m_pages.size();
	}

private:
	struct Entry {
		uint64_t history     = 0; // bit i: a fault in flip (last_flip - i)
		uint64_t last_flip   = 0;
		uint64_t fault_flip  = 0;
		uint64_t upload_flip = 0;
		uint64_t faults      = 0;
		uint64_t flips       = 0;
		uint64_t upload_flips = 0;
		uint64_t image_size  = 0;
		uint64_t flip_faults = 0; // faults in last_flip
		Owner    owner       = Owner::None;
	};

	static void Advance(Entry& entry, uint64_t flip) {
		if (entry.flips == 0) {
			entry.history = 0;
			return;
		}
#if defined(KYTY_WFDIAG_MUTANT_NO_SHIFT)
		(void)flip; // the history never moves on: every page looks like it faulted in one flip only
#else
		const uint64_t shift = flip - entry.last_flip;
		entry.history        = shift >= 64 ? 0 : entry.history << shift;
#endif
	}

	// Called with the lock held, when the first fault of a later flip arrives: statistics of m_flip.
	Finished Finish() {
		Finished result;
		result.flip = m_flip;
		const uint64_t window_mask = (uint64_t {1} << kWindowFlips) - 1;
		std::vector<uint64_t> faulted;
		for (auto it = m_pages.begin(); it != m_pages.end();) {
			auto& entry = it->second;
			uint64_t history = entry.history;
			if (entry.last_flip != m_flip) {
				const uint64_t shift = m_flip - entry.last_flip;
#if !defined(KYTY_WFDIAG_MUTANT_NO_SHIFT)
				history              = shift >= 64 ? 0 : history << shift;
#else
				(void)shift;
#endif
			}
			if (entry.last_flip == m_flip) {
				result.distinct++;
				faulted.push_back(it->first);
				if (entry.flip_faults >= 2) {
					result.refault_pages++;
					result.refaults += entry.flip_faults - 1;
				}
			}
			if (static_cast<unsigned>(std::popcount(history & window_mask)) >= kEveryMinFlips) {
				result.every++;
			}
			if (m_flip - entry.last_flip > kForgetFlips) {
				it = m_pages.erase(it);
			} else {
				++it;
			}
		}
		std::sort(faulted.begin(), faulted.end());
		for (size_t i = 0; i < faulted.size();) {
			size_t j = i + 1;
#if defined(KYTY_WFDIAG_MUTANT_RUNS_SPLIT)
			(void)j; // every page is a run of its own
#else
			while (j < faulted.size() && faulted[j] == faulted[j - 1] + kPageSize) {
				j++;
			}
#endif
			result.runs[RunBucket(j - i)]++;
			i = j;
		}
		return result;
	}

	mutable std::mutex                       m_lock;
	std::unordered_map<uint64_t, Entry>      m_pages;
	uint64_t                                 m_flip    = 0;
	bool                                     m_started = false;
};

inline Aggregator& Pages() {
	static Aggregator aggregator;
	return aggregator;
}

// Write and read faults per thread (the handler runs on the faulting thread). Thread-safe.
struct ThreadEntry {
	uint64_t    tid      = 0;
	std::string name;
	uint64_t    wf_n     = 0;
	uint64_t    wf_ns    = 0;
	uint64_t    rf_n     = 0;
	uint64_t    rf_ns    = 0;
};

class ThreadStats {
public:
	void NoteWrite(uint64_t tid, const char* name, uint64_t ns) { Note(tid, name, true, ns); }
	void NoteRead(uint64_t tid, const char* name, uint64_t ns) { Note(tid, name, false, ns); }
	[[nodiscard]] std::vector<ThreadEntry> Snapshot() const {
		std::lock_guard lock(m_lock);
		std::vector<ThreadEntry> all;
		for (const auto& [tid, entry]: m_threads) {
			all.push_back(entry);
		}
		std::sort(all.begin(), all.end(), [](const ThreadEntry& a, const ThreadEntry& b) {
			return a.wf_ns + a.rf_ns != b.wf_ns + b.rf_ns ? a.wf_ns + a.rf_ns > b.wf_ns + b.rf_ns : a.tid < b.tid;
		});
		return all;
	}

private:
	void Note(uint64_t tid, const char* name, bool write, uint64_t ns) {
		std::lock_guard lock(m_lock);
		auto& entry = m_threads[tid];
		if (entry.tid == 0) {
			entry.tid  = tid;
			entry.name = name != nullptr ? name : "";
		}
#if defined(KYTY_WFDIAG_MUTANT_THREAD_MIXED)
		(void)write;
		entry.wf_n++;
		entry.wf_ns += ns;
#else
		if (write) {
			entry.wf_n++;
			entry.wf_ns += ns;
		} else {
			entry.rf_n++;
			entry.rf_ns += ns;
		}
#endif
	}

	mutable std::mutex                       m_lock;
	std::unordered_map<uint64_t, ThreadEntry> m_threads;
};

inline ThreadStats& Threads() {
	static ThreadStats stats;
	return stats;
}

inline std::string FormatRuns(const Finished& finished) {
	char text[256];
	std::snprintf(text, sizeof(text), "[wf-runs] flip=%llu pages=%llu runs_1=%llu runs_2=%llu runs_3_4=%llu runs_5_8=%llu runs_9_16=%llu runs_17_up=%llu refault_pages=%llu refaults=%llu",
	              static_cast<unsigned long long>(finished.flip), static_cast<unsigned long long>(finished.distinct),
	              static_cast<unsigned long long>(finished.runs[0]), static_cast<unsigned long long>(finished.runs[1]),
	              static_cast<unsigned long long>(finished.runs[2]), static_cast<unsigned long long>(finished.runs[3]),
	              static_cast<unsigned long long>(finished.runs[4]), static_cast<unsigned long long>(finished.runs[5]),
	              static_cast<unsigned long long>(finished.refault_pages), static_cast<unsigned long long>(finished.refaults));
	return text;
}

inline std::string FormatThread(const ThreadEntry& entry, uint64_t flip) {
	char text[256];
	std::snprintf(text, sizeof(text), "[wf-thread] flip=%llu tid=%llu name=%s wf_n=%llu wf_ms=%.2f rf_n=%llu rf_ms=%.2f", static_cast<unsigned long long>(flip),
	              static_cast<unsigned long long>(entry.tid), entry.name.empty() ? "-" : entry.name.c_str(), static_cast<unsigned long long>(entry.wf_n),
	              static_cast<double>(entry.wf_ns) / 1e6, static_cast<unsigned long long>(entry.rf_n), static_cast<double>(entry.rf_ns) / 1e6);
	return text;
}

inline std::string FormatLine(const TopEntry& entry, uint64_t flip) {
	char text[256];
	std::snprintf(text, sizeof(text), "[wf] flip=%llu page=0x%016llx owner=%s image_size=%llu faults=%llu flips=%llu upload_flips=%llu",
	              static_cast<unsigned long long>(flip), static_cast<unsigned long long>(entry.page), OwnerName(entry.owner),
	              static_cast<unsigned long long>(entry.image_size), static_cast<unsigned long long>(entry.faults),
	              static_cast<unsigned long long>(entry.flips), static_cast<unsigned long long>(entry.upload_flips));
	return text;
}

} // namespace Libs::Graphics::WriteFault

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_WFDIAG_H_ */
