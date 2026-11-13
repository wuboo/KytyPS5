#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUDIRTYRANGES_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUDIRTYRANGES_H_

// S428 (B2): the GPU-modified byte ranges of the BufferCache, readable from a signal handler of a guest thread (KYTY_GF_EMULATE).
// Only the command processor thread changes the set. Its own reads need no lock; every change takes m_mutex, and the handler asks with TryIntersects,
// which never waits: when the CP holds the lock the answer is "busy" and the fault takes the ordinary drain (gf_emu_fb_lockbusy_n).
// BeginDownloadCopy fixes the order of the hand-over of a range to a GPU->host copy: first it becomes "in flight", then it stops being "dirty", so
// a reader that looks at dirty first and at in-flight second never sees a range that is neither (S418/S419: the backing is stale in between).
// Pure part; tests/GuestFaultSyncTests.cpp pins it.
//
// S570 micro (CP-only memos, the set is the CP's): KYTY_DIRTY_ADD_MEMO=0|1|shadow: ObtainBuffer adds the range of every written binding
// (~per draw), almost always one already in the set. A range added since the last removal of anything (Subtract / BeginDownloadCopy bump
// m_sub_gen) is still contained, so the Add (mutex, map walk, SpecPageCache invalidation of pages that cannot be quiet: they are dirty) is
// skipped. KYTY_DIRTY_QUERY_MEMO=0|1|shadow: Intersects (HasGpuDirtyBytes, ~10 k/flip) answers from a memo of the same range while the set
// has not changed (m_change_gen: every Add that ran, every removal). shadow: compute as before, count disagreements (*_ne_n must be 0).
// Direct-mapped, 1024 slots each. Mutants (tests/DirtyMemoTests.cpp, WILL_FAIL): KYTY_DIRTYMEMO_MUTANT_NO_SUB_GEN (a removal does not
// invalidate the add memo), _QUERY_IGNORES_ADD (an Add does not invalidate the query memo).

#include "graphics/host_gpu/rangeSet.h"
#include "kernel/specPageCache.h"
#include "graphics/host_gpu/renderer/cache/guestWait.h"
#include "graphics/host_gpu/renderer/cache/inflightRanges.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace Libs::Graphics {

namespace DirtyMemo {
enum class Mode : uint8_t { Off, On, Shadow };
inline Mode Parse(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}
inline Mode AddEnv() {
	static const Mode mode = Parse(std::getenv("KYTY_DIRTY_ADD_MEMO"));
	return mode;
}
inline Mode QueryEnv() {
	static const Mode mode = Parse(std::getenv("KYTY_DIRTY_QUERY_MEMO"));
	return mode;
}
struct Stats {
	uint64_t add_hit = 0, add_miss = 0, add_ne = 0, query_hit = 0, query_miss = 0, query_ne = 0;
};
} // namespace DirtyMemo

class GpuDirtyRanges final {
public:
	GpuDirtyRanges(): m_add_mode(DirtyMemo::AddEnv()), m_query_mode(DirtyMemo::QueryEnv()) {}
	// Tests.
	void SetMemoModes(DirtyMemo::Mode add, DirtyMemo::Mode query) {
		m_add_mode   = add;
		m_query_mode = query;
		m_add_memo   = {};
		m_query_memo = {};
	}
	[[nodiscard]] DirtyMemo::Mode AddMode() const { return m_add_mode; }
	// The counts since the last call (CP thread).
	DirtyMemo::Stats TakeMemoStats() {
		const auto out = m_stats;
		m_stats        = {};
		return out;
	}

	void Add(uint64_t address, uint64_t size) {
		if (m_add_mode != DirtyMemo::Mode::Off) {
			auto& slot = m_add_memo[Slot(address, size)];
			if (slot.size == size && slot.address == address && slot.gen == m_sub_gen) {
				m_stats.add_hit++;
				if (m_add_mode == DirtyMemo::Mode::On) return;
				if (!m_set.Contains(address, size)) m_stats.add_ne++;
			} else {
				m_stats.add_miss++;
			}
			slot = {address, size, m_sub_gen};
		}
		std::lock_guard lock(m_mutex);
		m_set.Add(address, size);
#ifndef KYTY_DIRTYMEMO_MUTANT_QUERY_IGNORES_ADD
		m_change_gen++;
#endif
		// S430 (KYTY_SPEC_READ_FAST): pages of this range are no longer quiet.
#ifndef KYTY_SPECFAST_MUTANT_FAST_NO_INVALIDATE_ON_WRITE
		Libs::LibKernel::Memory::SpecPageCache::OnRangeBecameBusy(address, size);
#endif
	}

	void Subtract(uint64_t address, uint64_t size) {
		std::lock_guard lock(m_mutex);
		m_set.Subtract(address, size);
		Removed();
	}

	// Command processor thread only (the thread that changes the set).
	[[nodiscard]] bool Intersects(uint64_t address, uint64_t size) const {
		if (m_query_mode == DirtyMemo::Mode::Off) return m_set.Intersects(address, size);
		auto& slot = m_query_memo[Slot(address, size)];
		if (slot.size == size && slot.address == address && slot.gen == m_change_gen) {
			m_stats.query_hit++;
			if (m_query_mode == DirtyMemo::Mode::On) return slot.result;
			const bool real = m_set.Intersects(address, size);
			if (real != slot.result) m_stats.query_ne++;
			slot.result = real;
			return real;
		}
		m_stats.query_miss++;
		const bool result = m_set.Intersects(address, size);
		slot              = {address, size, m_change_gen, result};
		return result;
	}
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
	// S585 (KYTY_GUEST_WAIT): TryIntersects with a bounded spin on the lock (GuestWait::SpinTryLock): a lock the CP holds for an Add or a
	// hand-over is waited out instead of answering "busy" at once; still never blocks without bound. False when the spin gave up.
	[[nodiscard]] bool TryIntersectsSpin(uint64_t address, uint64_t size, bool* out, int tries = GuestWait::kSpinTries) const {
		int spins = 0;
		if (!GuestWait::SpinTryLock(m_mutex, tries, &spins)) {
			GuestWait::ThreadSpin().busy++;
#if defined(KYTY_GUESTWAIT_MUTANT_BUSY_AS_CLEAN)
			*out = false;
			return true;
#else
			return false;
#endif
		}
		std::lock_guard lock(m_mutex, std::adopt_lock);
		if (spins != 0) GuestWait::ThreadSpin().spun++;
		*out = m_set.Intersects(address, size);
		return true;
	}

	// The GPU-modified pieces of [address, address + size) are handed to a recorded GPU->host copy: `piece(start, end)` is called for each (the caller
	// builds its copy list there), every piece is added to `inflight` BEFORE the whole range leaves the dirty set. `between` runs after the in-flight
	// entries exist and before the dirty set changes (tests look at the state in the window).
	// S585: `tick` = the submission the copy is recorded into (InflightRanges::Add), 0 = unknown.
	template <typename Piece, typename Between>
	void BeginDownloadCopy(InflightRanges& inflight, uint64_t address, uint64_t size, Piece&& piece, Between&& between, uint64_t tick = 0) {
		std::lock_guard lock(m_mutex);
#if defined(KYTY_GUESTFAULTSYNC_MUTANT_SUBTRACT_FIRST)
		std::vector<std::pair<uint64_t, uint64_t>> pieces;
		m_set.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) { pieces.emplace_back(start, end); });
		m_set.Subtract(address, size);
		between();
		for (const auto& [start, end]: pieces) {
			piece(start, end);
			inflight.Add(start, end - start, tick);
		}
#else
		m_set.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) {
			piece(start, end);
			inflight.Add(start, end - start, tick);
		});
		between();
		m_set.Subtract(address, size);
#endif
		Removed();
	}
	template <typename Piece>
	void BeginDownloadCopy(InflightRanges& inflight, uint64_t address, uint64_t size, Piece&& piece) {
		BeginDownloadCopy(inflight, address, size, static_cast<Piece&&>(piece), [] {});
	}
	template <typename Piece>
	void BeginDownloadCopyAt(InflightRanges& inflight, uint64_t tick, uint64_t address, uint64_t size, Piece&& piece) {
		BeginDownloadCopy(inflight, address, size, static_cast<Piece&&>(piece), [] {}, tick);
	}

	// Tests: the lock the command processor holds while it changes the set.
	[[nodiscard]] std::mutex& MutexForTest() const { return m_mutex; }

private:
	struct AddSlot {
		uint64_t address = 0;
		uint64_t size    = 0; // 0 = empty (the set never holds an empty range)
		uint64_t gen     = 0;
	};
	struct QuerySlot {
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t gen     = 0;
		bool     result  = false;
	};
	static constexpr size_t kSlots = 1024;
	static size_t Slot(uint64_t address, uint64_t size) {
		uint64_t h = (address >> 4u) ^ (size * 0x9e3779b97f4a7c15ull);
		h ^= h >> 29u;
		return static_cast<size_t>(h % kSlots);
	}
	// Under m_mutex, after any removal from the set.
	void Removed() {
#ifndef KYTY_DIRTYMEMO_MUTANT_NO_SUB_GEN
		m_sub_gen++;
#endif
		m_change_gen++;
	}

	mutable std::mutex m_mutex;
	RangeSet           m_set;
	DirtyMemo::Mode    m_add_mode   = DirtyMemo::Mode::Off;
	DirtyMemo::Mode    m_query_mode = DirtyMemo::Mode::Off;
	uint64_t           m_sub_gen    = 1; // slots start at gen 0: never a hit before the first store
	uint64_t           m_change_gen = 1;
	std::array<AddSlot, kSlots>           m_add_memo {};
	mutable std::array<QuerySlot, kSlots> m_query_memo {};
	mutable DirtyMemo::Stats              m_stats;
};

} // namespace Libs::Graphics

#endif
