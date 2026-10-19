#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMA3POLICY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMA3POLICY_H_

#include "graphics/host_gpu/memory/writeFenceTable.h"

#include <atomic>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>

// Decisions of UMA stage 3 (KYTY_UMA=3) that need no GPU, kept apart so CPU tests can pin them.
namespace Libs::Graphics::Uma3 {

// Where a buffer request goes. Window: a window of the imported arena (no copy, no readback, no
// buffer page watchers). The other two keep the copy path and are counted by reason.
enum class Route { Window, LegacyUncovered, LegacyImageSync };

// covered: one contiguous run of the arena backs the whole range. A read of memory that holds a
// GPU-written image (texel buffer, DMA source) needs the image-to-buffer copy, which only fills a
// copy-path buffer; a write never does (the image is invalidated instead). The overlap test is
// lazy: it asks the texture cache, which is not free.
template <typename OverlapsImage>
[[nodiscard]] Route RouteBuffer(bool covered, bool texel_or_dma_source, bool written,
                                OverlapsImage&& overlaps_image) {
	if (!covered) {
		return Route::LegacyUncovered;
	}
#ifdef KYTY_UMA_MUTANT_TEXEL_WINDOW
	(void)texel_or_dma_source;
	(void)written;
	(void)overlaps_image;
	return Route::Window; // wrong: image memory is read through the window without the image copy
#else
	if (texel_or_dma_source && !written && overlaps_image()) {
		return Route::LegacyImageSync;
	}
	return Route::Window;
#endif
}

// A window starts at an arena offset; descriptors want an offset aligned to the storage-buffer
// alignment, the shader gets the remainder.
struct WindowSplit {
	uint64_t aligned    = 0;
	uint64_t adjustment = 0;
	bool     ok         = false;
};

[[nodiscard]] inline WindowSplit SplitWindow(uint64_t offset, uint64_t alignment, uint64_t size,
                                             uint64_t max_range, uint64_t max_adjustment = 256) {
	if (alignment == 0) {
		alignment = 1;
	}
	WindowSplit split;
#ifdef KYTY_UMA_MUTANT_WINDOW_NO_ADJUST
	split.aligned    = offset; // wrong: an unaligned descriptor offset
	split.adjustment = 0;
	split.ok         = true;
	(void)size;
	(void)max_range;
	(void)max_adjustment;
#else
	split.aligned    = offset - offset % alignment;
	split.adjustment = offset - split.aligned;
	split.ok         = split.adjustment < max_adjustment && split.adjustment <= max_range &&
	           size <= max_range - split.adjustment;
#endif
	return split;
}

// Guest ranges where a copy-path buffer was ever registered. Only those pages can carry buffer
// tracker state; everywhere else a write fault needs no readback and a read fault needs no
// download. Additive on purpose (a deleted buffer leaves its pages tracked until they are
// untracked), so skipping is always safe.
class LegacyRanges {
public:
	void Add(uint64_t address, uint64_t size) {
		if (size == 0) {
			return;
		}
		auto start = address;
		auto end   = address + size;
		auto it    = m_runs.upper_bound(start);
		if (it != m_runs.begin()) {
			auto prev = std::prev(it);
			if (prev->second >= start) {
				start = prev->first;
				end   = end > prev->second ? end : prev->second;
				it    = m_runs.erase(prev);
			}
		}
		while (it != m_runs.end() && it->first <= end) {
			end = end > it->second ? end : it->second;
			it  = m_runs.erase(it);
		}
		m_runs[start] = end;
	}

	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		if (size == 0 || m_runs.empty()) {
			return false;
		}
		auto it = m_runs.upper_bound(address);
		if (it != m_runs.begin() && std::prev(it)->second > address) {
			return true;
		}
		return it != m_runs.end() && it->first < address + size;
	}

	[[nodiscard]] bool   Empty() const { return m_runs.empty(); }
	[[nodiscard]] size_t Count() const { return m_runs.size(); }

private:
	std::map<uint64_t, uint64_t> m_runs; // start -> end
};

// True when a buffer-tracker call (InvalidateMemory / ReadMemory) has nothing to do for the range.
[[nodiscard]] inline bool TrackerCanSkip(const LegacyRanges& legacy, uint64_t address,
                                         uint64_t size) {
#ifdef KYTY_UMA_MUTANT_SKIP_LEGACY
	(void)legacy;
	(void)address;
	(void)size;
	return true; // wrong: a copy-path buffer's write fault would be ignored
#else
	return !legacy.Intersects(address, size);
#endif
}

// The newest submit that may have written guest memory at addresses nobody knows in advance
// (stores through guest pointers). Stages 1-2 treat every submit with direct access that way;
// stage 3 only the submits holding a shader that stores through pointers, because windows and
// fills record their exact ranges.
class UnknownWriterTick {
public:
	void Submit(uint64_t tick, bool writes_through_pointers, bool stage3) {
#ifdef KYTY_UMA_MUTANT_GLOBAL_IGNORES_WRITES
		(void)writes_through_pointers;
		(void)stage3;
		const bool record = true; // wrong: every submit keeps the global writer tick alive
#else
		const bool record = writes_through_pointers || !stage3;
#endif
		if (record && tick > m_tick) {
			m_tick = tick;
		}
	}
	[[nodiscard]] uint64_t Value() const { return m_tick; }

private:
	uint64_t m_tick = 0;
};

// Which timelines a reader of a range must wait on: bit t is set when ticks[t] has not retired.
// `ticks` are the range's writers; the unknown-range writer applies to the graphics timeline.
// effective_ticks (optional) receives the ticks to wait for, unknown writer included.
template <typename IsRetired>
[[nodiscard]] uint32_t PendingTimelines(WriteFenceTable::Ticks ticks, uint64_t unknown_writer_tick,
                                        IsRetired&& is_retired,
                                        WriteFenceTable::Ticks* effective_ticks = nullptr) {
	if (unknown_writer_tick > ticks[0]) {
		ticks[0] = unknown_writer_tick;
	}
	if (effective_ticks != nullptr) {
		*effective_ticks = ticks;
	}
	uint32_t mask = 0;
	for (uint32_t timeline = 0; timeline < WriteFenceTable::MaxTimelines; ++timeline) {
		if (ticks[timeline] != 0 && !is_retired(timeline, ticks[timeline])) {
			mask |= 1u << timeline;
		}
	}
	return mask;
}

// Who waits (WaitWriters caller) and why the wait was needed. Reason Range: a writer recorded for
// the range itself has not retired. Reason Global: only the conservative "unknown pointer writer"
// tick keeps the range pending. Names feed the uma_wait_<source>_<reason>_ms counters.
enum class WaitSource : uint32_t { Other, ReadMemory, Obtain, Dma, Cp, Cpu, Count };
enum class WaitReason : uint32_t { Range, Global, Count };

[[nodiscard]] constexpr const char* WaitSourceName(WaitSource source) {
	switch (source) {
		case WaitSource::ReadMemory: return "readmem";
		case WaitSource::Obtain: return "obtain";
		case WaitSource::Dma: return "dma";
		case WaitSource::Cp: return "cp";
		case WaitSource::Cpu: return "cpu";
		default: return "other";
	}
}

[[nodiscard]] constexpr const char* WaitReasonName(WaitReason reason) {
	return reason == WaitReason::Range ? "range" : "global";
}

// KYTY_OPT_ON=bda_wait_narrow. Guest ranges (1 MiB cells) that were bound at least once as a window
// to write or recorded as a GPU writer (fills, copies, labels). Heuristic: a range in a cell that
// was never written that way is not waited for on the global pointer-writer tick alone; a shader
// that stores through a pointer into such a cell is NOT covered (the option is off by default
// and must be validated on the game). Marked by any writer recorder, read by any thread (relaxed).
class WrittenCells {
public:
	static constexpr uint32_t CellBits    = 20;
	static constexpr uint32_t AddressBits = 44; // guest address space bits

	WrittenCells(): m_words(new std::atomic<uint64_t>[WordCount()]()) {}

	void Mark(uint64_t vaddr, uint64_t size) {
		if (size == 0) {
			return;
		}
		for (uint64_t cell = Cell(vaddr); cell <= Cell(vaddr + size - 1); ++cell) {
			m_words[cell >> 6].fetch_or(uint64_t {1} << (cell & 63), std::memory_order_relaxed);
		}
	}
	// True when any cell of the range was marked (size 0 = everything: true).
	[[nodiscard]] bool Any(uint64_t vaddr, uint64_t size) const {
		if (size == 0) {
			return true;
		}
		for (uint64_t cell = Cell(vaddr); cell <= Cell(vaddr + size - 1); ++cell) {
			if ((m_words[cell >> 6].load(std::memory_order_relaxed) >> (cell & 63) & 1u) != 0) {
				return true;
			}
		}
		return false;
	}

private:
	static constexpr uint64_t WordCount() { return (uint64_t {1} << (AddressBits - CellBits)) / 64; }
	static constexpr uint64_t Cell(uint64_t vaddr) {
		return (vaddr >> CellBits) & ((uint64_t {1} << (AddressBits - CellBits)) - 1);
	}
	std::unique_ptr<std::atomic<uint64_t>[]> m_words;
};

// The pointer-writer tick a reader of [vaddr, size) must honour. size 0 = every range: always the
// global tick. With narrow, ranges outside every written cell ignore it.
[[nodiscard]] inline uint64_t EffectiveUnknownTick(uint64_t unknown_tick, bool narrow,
                                                   const WrittenCells& written, uint64_t vaddr,
                                                   uint64_t size) {
#ifdef KYTY_UMA_MUTANT_NARROW_IGNORES_WRITTEN
	(void)written;
	return narrow && size != 0 ? 0 : unknown_tick; // wrong: a range written through a window is not waited for
#else
	return narrow && size != 0 && !written.Any(vaddr, size) ? 0 : unknown_tick;
#endif
}

// Reason of a wait that is needed: Range when the range's own writers are pending without the
// unknown-range tick, else Global.
template <typename IsRetired>
[[nodiscard]] WaitReason ClassifyWait(WriteFenceTable::Ticks range_ticks, IsRetired&& is_retired) {
	return PendingTimelines(range_ticks, 0, is_retired) != 0 ? WaitReason::Range : WaitReason::Global;
}

} // namespace Libs::Graphics::Uma3

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMA3POLICY_H_
