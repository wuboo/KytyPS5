#ifndef EMULATOR_INCLUDE_EMULATOR_KERNEL_SPECPAGECACHE_H_
#define EMULATOR_INCLUDE_EMULATOR_KERNEL_SPECPAGECACHE_H_

// S430 KYTY_SPEC_READ_FAST=0|1|shadow: the strict reader of the shader walk (ReadShaderGuestMemory) remembers which 4 KiB pages were quiet
// (no GPU-dirty bytes, no GPU->host copy in flight) and where their backing is, so a later read of the page is one lookup, two loads of the map
// sequence and a memcpy, instead of two range-set queries, a shared_lock and a std::map walk per dword.
//
// The cache holds a verdict and a pointer, never bytes: a CPU write of the guest is visible to the next read. A verdict dies when
//  - GpuDirtyRanges::Add / InflightRanges::Add touch the page (OnRangeBecameBusy; Subtract only makes pages quieter, no entry for that),
//  - the guest address space changes its mappings (m_map_seq, a seqlock: odd while a change is under way).
// Everything runs on the command processor thread (the only thread that adds dirty / in-flight ranges and the only caller of the reader).
// Pure part; tests/SpecPageCacheTests.cpp pins it. Mutants: KYTY_SPECFAST_MUTANT_{FAST_IGNORES_DIRTY, FAST_NO_INVALIDATE_ON_WRITE,
// NO_INFLIGHT_INVALIDATE, NO_MAP_SEQ, CACHE_BYTES, CROSS_PAGE}; S437: BUSY_NOT_GATED (tests/SpecPageCacheOffTests.cpp).

#include "kernel/srtReadPass.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::LibKernel::Memory {

enum class SpecFastMode : uint8_t { Off, On, Shadow };

// null/""/"0" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
[[nodiscard]] inline SpecFastMode ParseSpecFastMode(const char* text) {
	if (text == nullptr || text[0] == '\0') return SpecFastMode::Off;
	if (std::strcmp(text, "1") == 0) return SpecFastMode::On;
	if (std::strcmp(text, "shadow") == 0) return SpecFastMode::Shadow;
	return SpecFastMode::Off;
}

[[nodiscard]] inline SpecFastMode SpecFastEnvMode() {
	static const SpecFastMode mode = ParseSpecFastMode(std::getenv("KYTY_SPEC_READ_FAST"));
	return mode;
}

class SpecPageCache {
public:
	static constexpr uint64_t PageSize = SrtReadPass::PageSize;
	static constexpr uint64_t PageMask = ~(PageSize - 1u);
	static constexpr uint64_t Slots    = 256;
	static constexpr uint64_t NoPage   = ~uint64_t {0};

	static SpecPageCache& Global() {
		static SpecPageCache cache;
		return cache;
	}

	// The backing of `page` if it was filled quiet at the same map sequence, else nullptr.
	[[nodiscard]] const uint8_t* Lookup(uint64_t page, uint64_t map_seq) const {
		const Slot& slot = m_slots[(page / PageSize) % Slots];
#ifdef KYTY_SPECFAST_MUTANT_NO_MAP_SEQ
		(void)map_seq;
		return slot.page == page ? slot.host : nullptr;
#else
		return slot.page == page && slot.map_seq == map_seq && (map_seq & 1u) == 0 ? slot.host : nullptr;
#endif
	}

	void Fill(uint64_t page, const uint8_t* host, uint64_t map_seq) {
		Slot& slot   = m_slots[(page / PageSize) % Slots];
		slot.page    = page;
		slot.host    = host;
		slot.map_seq = map_seq;
#ifdef KYTY_SPECFAST_MUTANT_CACHE_BYTES
		std::memcpy(slot.snapshot.data(), host, PageSize);
#endif
	}

	// A range stopped being quiet. More than Slots pages: everything goes.
	void InvalidateRange(uint64_t address, uint64_t size) {
		if (size == 0) return;
		const uint64_t first = address & PageMask;
		const uint64_t last  = (address + size - 1u) & PageMask;
		if ((last - first) / PageSize >= Slots) {
			Clear();
			return;
		}
		for (uint64_t page = first;; page += PageSize) {
			Slot& slot = m_slots[(page / PageSize) % Slots];
			if (slot.page == page) slot.page = NoPage;
			if (page == last) break;
		}
	}

	void Clear() {
		for (auto& slot: m_slots) slot.page = NoPage;
	}

#ifdef KYTY_SPECFAST_MUTANT_CACHE_BYTES
	[[nodiscard]] const uint8_t* Snapshot(uint64_t page) const { return m_slots[(page / PageSize) % Slots].snapshot.data(); }
#endif

	// Called by GpuDirtyRanges::Add and InflightRanges::Add (the only transitions quiet -> busy).
	// S437: with KYTY_SPEC_READ_FAST off (the default) nothing reads the cache, so the hot path of every Add() pays one load and no invalidation.
	static void OnRangeBecameBusy(uint64_t address, uint64_t size) {
#ifndef KYTY_SPECFAST_MUTANT_BUSY_NOT_GATED
		if (SpecFastEnvMode() == SpecFastMode::Off) return;
#endif
		Global().InvalidateRange(address, size);
	}

private:
	struct Slot {
		uint64_t       page    = NoPage;
		const uint8_t* host    = nullptr;
		uint64_t       map_seq = 0;
#ifdef KYTY_SPECFAST_MUTANT_CACHE_BYTES
		std::array<uint8_t, PageSize> snapshot {};
#endif
	};
	std::array<Slot, Slots> m_slots {};
};

// One strict read of [vaddr, vaddr + size) through the cache.
//  h.MapSeq()                        -> uint64_t   the sequence of the guest mappings (odd: a change is under way)
//  h.Dirty(page) / h.Inflight(page)  -> bool       GPU-dirty bytes / unpublished GPU->host copy anywhere in the 4 KiB page
//  h.ResolvePage(page, &host, &seq)  -> bool       the backing of the whole page (one mapping), and the (even) sequence it was resolved at
//  h.Full(vaddr, data, size)         -> bool       the ordinary reader (drains, waits, TryReadBacking)
// *hit is true when the read was served from an entry without asking Dirty / Inflight.
template <typename Hooks>
bool ReadViaPageCache(SpecPageCache& cache, Hooks& h, uint64_t vaddr, void* data, uint64_t size, bool* hit) {
	*hit = false;
	constexpr uint64_t mask = SpecPageCache::PageMask;
#ifdef KYTY_SPECFAST_MUTANT_CROSS_PAGE
	const bool crosses = false;
#else
	const bool crosses = size != 0 && ((vaddr ^ (vaddr + size - 1u)) & mask) != 0;
#endif
	if (size == 0 || crosses || UINT64_MAX - vaddr < size) return h.Full(vaddr, data, size);
	const uint64_t page   = vaddr & mask;
	const uint64_t offset = vaddr - page;
	const uint64_t seq1   = h.MapSeq();
	if (const uint8_t* host = cache.Lookup(page, seq1)) {
#ifdef KYTY_SPECFAST_MUTANT_CACHE_BYTES
		std::memcpy(data, cache.Snapshot(page) + offset, size);
		*hit = true;
		return true;
#else
		std::memcpy(data, host + offset, size);
		if (h.MapSeq() == seq1) {
			*hit = true;
			return true;
		}
#endif
	}
	bool dirty    = h.Dirty(page);
	bool inflight = h.Inflight(page);
#ifdef KYTY_SPECFAST_MUTANT_FAST_IGNORES_DIRTY
	dirty = false;
#endif
	const uint8_t* host = nullptr;
	uint64_t       seq  = 0;
	if (SrtPageQuiet(dirty, inflight) && h.ResolvePage(page, &host, &seq)) {
		cache.Fill(page, host, seq);
		std::memcpy(data, host + offset, size);
		if (h.MapSeq() == seq) return true;
	}
	return h.Full(vaddr, data, size);
}

} // namespace Libs::LibKernel::Memory

#endif /* EMULATOR_INCLUDE_EMULATOR_KERNEL_SPECPAGECACHE_H_ */
