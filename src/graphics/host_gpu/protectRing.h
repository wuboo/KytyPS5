#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_PROTECTRING_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_PROTECTRING_H_

// S444 DIAG (KYTY_DIAG_WRITE_FAULT=1): ring of the last host protection changes (PageManager::Impl::Protect and the guest's own mprotect), each with the tag of the
// code that asked for it. A write fault on a page the tracker keeps unprotected for its flip ("sticky") prints the newest entries covering the page: the last
// caller that put the protection back. Pure part; tests/MemoryTrackerTests.cpp (TestStickyRefault) pins it. Nothing here feeds a shader or pipeline cache key.

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>

namespace Libs::Graphics::ProtectRing {

inline std::atomic<bool>& EnabledFlag() {
	static std::atomic<bool> on {false};
	return on;
}

inline bool Enabled() { return EnabledFlag().load(std::memory_order_relaxed); }

// The code asking for a protection change names itself through a thread-local tag; the innermost scope wins.
inline const char*& CurrentTag() {
	static thread_local const char* tag = nullptr;
	return tag;
}

class TagScope final {
public:
	explicit TagScope(const char* tag): m_previous(CurrentTag()) { CurrentTag() = tag; }
	~TagScope() { CurrentTag() = m_previous; }
	TagScope(const TagScope&)            = delete;
	TagScope& operator=(const TagScope&) = delete;

private:
	const char* m_previous;
};

// Sets the tag only when no caller named itself yet (the generic name of the tracker's own update).
class DefaultTagScope final {
public:
	explicit DefaultTagScope(const char* tag): m_previous(CurrentTag()) {
		if (m_previous == nullptr) {
			CurrentTag() = tag;
		}
	}
	~DefaultTagScope() { CurrentTag() = m_previous; }
	DefaultTagScope(const DefaultTagScope&)            = delete;
	DefaultTagScope& operator=(const DefaultTagScope&) = delete;

private:
	const char* m_previous;
};

struct Entry {
	uint64_t    seq  = 0; // 0: empty slot
	uint64_t    addr = 0;
	uint64_t    size = 0;
	uint32_t    flip = 0;
	int         mode = 0; // Common::VirtualMemory::Mode as the caller passes it
	const char* tag  = nullptr;
};

constexpr size_t kRingSize = 8192;
constexpr size_t kMaxWhy   = 3;
constexpr uint64_t kMaxLines = 40;

inline std::array<Entry, kRingSize>& Slots() {
	static std::array<Entry, kRingSize> slots {};
	return slots;
}

inline std::atomic<uint64_t>& Cursor() {
	static std::atomic<uint64_t> next {0};
	return next;
}

// Record the change of [addr, addr + size) to `mode`. The slot is written without a lock: a reader may see a torn entry, which a diagnostic accepts.
inline void Record(uint64_t addr, uint64_t size, int mode, uint32_t flip, const char* fallback_tag) {
	if (!Enabled()) {
		return;
	}
	const auto seq = Cursor().fetch_add(1, std::memory_order_relaxed) + 1;
	const char* tag = CurrentTag() != nullptr ? CurrentTag() : fallback_tag;
#if defined(KYTY_MEMTRACKER_MUTANT_RING_NO_TAG)
	tag = fallback_tag;
#endif
	Slots()[seq % kRingSize] = Entry {seq, addr, size, flip, mode, tag};
}

// The newest entries (newest first) whose range covers the 4 KiB page at `page`.
inline size_t Find(uint64_t page, Entry* out, size_t max) {
	const auto last = Cursor().load(std::memory_order_relaxed);
	size_t     n    = 0;
	for (uint64_t seq = last; seq != 0 && last - seq < kRingSize && n < max; --seq) {
		const auto& e = Slots()[seq % kRingSize];
#if defined(KYTY_MEMTRACKER_MUTANT_RING_NO_MATCH)
		const bool covers = e.seq == seq;
#else
		const bool covers = e.seq == seq && page >= e.addr && page - e.addr < e.size;
#endif
		if (covers) {
			out[n++] = e;
		}
	}
	return n;
}

inline std::string FormatRefault(uint64_t page, uint32_t flip, uint32_t sticky_flip, const Entry* why, size_t why_n) {
	char head[200];
	std::snprintf(head, sizeof(head), "[wf-sticky-refault] page=0x%016" PRIx64 " host_page16k=0x%016" PRIx64 " flip=%u sticky_flip=%u why=", page,
	              page & ~uint64_t {0x3fff}, flip, sticky_flip);
	std::string text = head;
	if (why_n == 0) {
		text += "none-in-ring";
	}
	for (size_t i = 0; i < why_n; ++i) {
		char part[200];
		std::snprintf(part, sizeof(part), "%s#%" PRIu64 "@flip%u:%s:mode%d:0x%" PRIx64 "+0x%" PRIx64, i == 0 ? "" : " | ", why[i].seq, why[i].flip,
		              why[i].tag != nullptr ? why[i].tag : "untagged", why[i].mode, why[i].addr, why[i].size);
		text += part;
	}
	return text;
}

inline std::atomic<uint64_t>& LinesPrinted() {
	static std::atomic<uint64_t> n {0};
	return n;
}

} // namespace Libs::Graphics::ProtectRing

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_PROTECTRING_H_
