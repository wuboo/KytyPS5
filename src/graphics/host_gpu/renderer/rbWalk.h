#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RBWALK_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RBWALK_H_

// S502 (KYTY_RB_WALK=0|1|shadow, unset = 0): the upstream SRT walker loads guest memory with
// mov eax,[r10] (SrtWalker::LoadGuest). A 4 KiB page that holds any GPU-written byte is
// protected, so the load faults and ReadMemory drains the queue even when this dword is not
// GPU-dirty and no copy of it is in flight (false sharing; the same rule as KYTY_CP_READ_BACKING).
//   0       the fault, as before
//   1       a walker load whose 4 bytes are known clean is completed from the backing (no drain)
//   shadow  the fault still runs; the backing dword is compared with the drained bytes (ne = 0)
// A query that cannot answer (lock busy, range not tracked) stays on the fault. Dirty or
// in-flight bytes stay on the fault: the backing would be stale.
// A dirty dword is not served (the backing is stale). Mode 1 remembers the page and, on the next GPU
// write of at most one tracker page, queues that page for the existing pre-submit download
// (FlushEagerReadbacks). The copy is recorded with the write, so a later walker load in the frame
// finds the bytes already published instead of draining the queue behind every draw since the write.
// Mutants (tests/RbWalkTests.cpp): KYTY_RBWALK_MUTANT_SERVE_DIRTY, _SERVE_INFLIGHT, _SERVE_UNKNOWN,
// _SHADOW_SKIPS_FAULT, _PREFETCH_ANY_SIZE.

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::RbWalk {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_RB_WALK"));
	return mode;
}

enum class Source : uint8_t {
	Fault,   // the page fault drains, as before
	Backing, // complete mov eax from the backing
	Shadow,  // the fault runs; the caller compares the backing dword with the drained bytes
};

// dirty / inflight: the 4 bytes the load reads. unknown: the tracker could not be queried.
constexpr Source Decide(Mode mode, bool dirty, bool inflight, bool unknown) {
	if (mode == Mode::Off) return Source::Fault;
#if defined(KYTY_RBWALK_MUTANT_SERVE_DIRTY)
	dirty = false;
#endif
#if defined(KYTY_RBWALK_MUTANT_SERVE_INFLIGHT)
	inflight = false;
#endif
#if defined(KYTY_RBWALK_MUTANT_SERVE_UNKNOWN)
	unknown = false;
#endif
	if (dirty || inflight || unknown) return Source::Fault;
#if defined(KYTY_RBWALK_MUTANT_SHADOW_SKIPS_FAULT)
	return Source::Backing;
#else
	return mode == Mode::Shadow ? Source::Shadow : Source::Backing;
#endif
}

// Queue a pre-submit download of a write that touches a page a walker load already faulted on.
// Only mode 1. Shadow and off leave the fault path alone. Writes larger than one 4 KiB page stay
// on the fault (a whole-buffer download is not this gap).
constexpr bool PrefetchWrite(Mode mode, bool learned, uint64_t size) {
	if (mode != Mode::On || !learned) return false;
#if defined(KYTY_RBWALK_MUTANT_PREFETCH_ANY_SIZE)
	(void)size;
	return true;
#else
	return size > 0 && size <= 4096;
#endif
}

// Pages whose walker load faulted on GPU-dirty bytes. The GPU thread only (the fault and the write).
class LearnedPages {
public:
	static constexpr size_t kCap = 8;

	bool Note(uint64_t page) {
		if (page == 0 || Contains(page)) return false;
		if (m_count >= kCap) return false;
		m_pages[m_count++] = page;
		return true;
	}
	bool Contains(uint64_t page) const {
		for (size_t i = 0; i < m_count; ++i) {
			if (m_pages[i] == page) return true;
		}
		return false;
	}
	bool Intersects(uint64_t address, uint64_t size) const {
		if (size == 0 || m_count == 0 || address > ~uint64_t {0} - (size - 1)) return false;
		const uint64_t last = address + size - 1;
		for (size_t i = 0; i < m_count; ++i) {
			const uint64_t page = m_pages[i];
			if (address <= page + 4095 && page <= last) return true;
		}
		return false;
	}

private:
	uint64_t m_pages[kCap] {};
	size_t   m_count = 0;
};

inline LearnedPages& Learned() {
	static LearnedPages pages;
	return pages;
}

} // namespace Libs::Graphics::RbWalk

#endif
