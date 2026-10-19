#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_WFDIAG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_WFDIAG_H_

// S440 (DIAG, default off): which pages take the ~1200 write faults per flip of the guest, and does the GPU read them again in the same flip?
//   KYTY_DIAG_WRITE_FAULT=1   frames.csv columns wf_n (write faults), wf_handler_ns (time of the handler's write branch), wf_pages_distinct (pages that
//                             faulted in the last finished flip), wf_pages_every_flip (pages that faulted in >= 90 % of the last 60 flips);
//                             every kTopEveryFlips flips (and at the first flip of the log) one [wf] line per page for the 20 pages with most faults:
//                             address, owner (buf: the buffer tracker watched it, img: a texture-cache image watched it), image size, faults,
//                             flips with a fault, flips in which the same page was uploaded to the GPU after a fault.
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

struct Finished {
	uint64_t flip     = 0;
	uint64_t distinct = 0; // pages with a fault in the flip
	uint64_t every    = 0; // pages with a fault in >= kEveryMinFlips of the last kWindowFlips flips (this one included)
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
		}
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
