#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUDIRTYRANGES_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUDIRTYRANGES_H_

// S428 (B2): the GPU-modified byte ranges of the BufferCache, readable from a signal handler of a guest thread (KYTY_GF_EMULATE).
// Only the command processor thread changes the set. Its own reads need no lock; every change takes m_mutex, and the handler asks with TryIntersects,
// which never waits: when the CP holds the lock the answer is "busy" and the fault takes the ordinary drain (gf_emu_fb_lockbusy_n).
// BeginDownloadCopy fixes the order of the hand-over of a range to a GPU->host copy: first it becomes "in flight", then it stops being "dirty", so
// a reader that looks at dirty first and at in-flight second never sees a range that is neither (S418/S419: the backing is stale in between).
// Pure part; tests/GuestFaultSyncTests.cpp pins it.

#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/inflightRanges.h"

#include <cstdint>
#include <mutex>

namespace Libs::Graphics {

class GpuDirtyRanges final {
public:
	void Add(uint64_t address, uint64_t size) {
		std::lock_guard lock(m_mutex);
		m_set.Add(address, size);
	}

	void Subtract(uint64_t address, uint64_t size) {
		std::lock_guard lock(m_mutex);
		m_set.Subtract(address, size);
	}

	// Command processor thread only (the thread that changes the set).
	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const { return m_set.Intersects(address, size); }
	[[nodiscard]] bool Empty() const { return m_set.Empty(); }
	template <typename Func>
	void ForEachInRange(uint64_t address, uint64_t size, Func&& func) const {
		m_set.ForEachInRange(address, size, static_cast<Func&&>(func));
	}
	[[nodiscard]] const RangeSet& Set() const { return m_set; }

	// Any thread, never waits: false when the command processor is changing the set (*out untouched).
	[[nodiscard]] bool TryIntersects(uint64_t address, uint64_t size, bool* out) const {
#if defined(KYTY_GUESTFAULTSYNC_MUTANT_BLOCKING_QUERY)
		std::unique_lock lock(m_mutex);
#else
		std::unique_lock lock(m_mutex, std::try_to_lock);
		if (!lock.owns_lock()) {
			return false;
		}
#endif
		*out = m_set.Intersects(address, size);
		return true;
	}

	// The GPU-modified pieces of [address, address + size) are handed to a recorded GPU->host copy: `piece(start, end)` is called for each (the caller
	// builds its copy list there), every piece is added to `inflight` BEFORE the whole range leaves the dirty set. `between` runs after the in-flight
	// entries exist and before the dirty set changes (tests look at the state in the window).
	template <typename Piece, typename Between>
	void BeginDownloadCopy(InflightRanges& inflight, uint64_t address, uint64_t size, Piece&& piece, Between&& between) {
		std::lock_guard lock(m_mutex);
#if defined(KYTY_GUESTFAULTSYNC_MUTANT_SUBTRACT_FIRST)
		std::vector<std::pair<uint64_t, uint64_t>> pieces;
		m_set.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) { pieces.emplace_back(start, end); });
		m_set.Subtract(address, size);
		between();
		for (const auto& [start, end]: pieces) {
			piece(start, end);
			inflight.Add(start, end - start);
		}
#else
		m_set.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) {
			piece(start, end);
			inflight.Add(start, end - start);
		});
		between();
		m_set.Subtract(address, size);
#endif
	}
	template <typename Piece>
	void BeginDownloadCopy(InflightRanges& inflight, uint64_t address, uint64_t size, Piece&& piece) {
		BeginDownloadCopy(inflight, address, size, static_cast<Piece&&>(piece), [] {});
	}

	// Tests: the lock the command processor holds while it changes the set.
	[[nodiscard]] std::mutex& MutexForTest() const { return m_mutex; }

private:
	mutable std::mutex m_mutex;
	RangeSet           m_set;
};

} // namespace Libs::Graphics

#endif
