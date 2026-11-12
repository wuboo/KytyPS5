#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUWRITELEDGER_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUWRITELEDGER_H_

#include "graphics/host_gpu/renderer/pipeline/texResolveMemo.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>

namespace Libs::Graphics {

// S562 KYTY_LEDGER_MEMO=0|1|shadow: GpuWriteLedger::LatestIsFill answered from a per-thread direct-mapped memo of (ledger, range) -> answer, valid
// while the ledger's epoch (bumped under its mutex by every Push, Forget and Trim) is the one the answer was computed at (read under the same mutex).
// MaterializeColorClear asks it for every DCC/CMASK slice of every colour target of every draw (~10 k/flip in the garden), and the ledger changes
// only on GPU buffer writes, fills, CPU writes and downloads of GPU-written ranges. A lookup reads the epoch without the mutex: a write that is not
// yet visible orders that lookup before the write, as the mutex would. Off: unchanged. shadow: scans as ever and compares (ledger_memo_ne_n).
// Mutants (tests, WILL_FAIL): KYTY_LEDGER_MUTANT_MEMO_{NO_EPOCH, NO_OWNER}.
namespace LedgerMemo {
using Mode = TexMemo::Mode;
[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = TexMemo::ParseMode(std::getenv("KYTY_LEDGER_MEMO"));
	return mode;
}
struct Stats {
	uint64_t hit  = 0;
	uint64_t miss = 0;
	uint64_t ne   = 0;
};
} // namespace LedgerMemo

// What the ledger knows about the bytes of a range (QueryFill).
enum class FillKind : uint32_t {
	Partial,    // no single fill covers the range (never written, forgotten, partly filled, evicted)
	NewerWrite, // a fill covered the range, but a newer write touched it afterwards
	Known,      // the newest write touching the range is one fill covering all of it
};

struct FillQuery {
	FillKind kind  = FillKind::Partial;
	uint32_t value = 0;     // Known: the fill value; otherwise the value of the newest fill seen (diagnostic only)
	bool     seen  = false; // a fill with a value was seen
};

// Newest-first record of GPU writes to guest ranges. A uniform fill carries its value, so the bytes of a range whose newest
// write is one fill are known without reading them back (KYTY_DCC_KNOWN). Fixed size, lossy: eviction drops the OLDEST
// entries first, so a fill that is still present is never hidden by an evicted newer write (the answer is "unknown", never a
// wrong value). Any other GPU write, a CPU write or a download of the range hides the fill.
class GpuWriteLedger {
public:
	static constexpr uint32_t Size = 64;

	void RecordWrite(uint64_t vaddr, uint64_t size) { Push({vaddr, size, false, 0}); }
	void RecordFill(uint64_t vaddr, uint64_t size, uint32_t value) { Push({vaddr, size, true, value}); }

	// The range left the GPU-modified state (downloaded, or the CPU wrote it): fills touching it no longer describe its
	// bytes. The entries stay as writers, so they still hide older fills.
	void Forget(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		Bump();
		for (auto& entry: m_entries) {
			if (entry.size != 0 && Overlaps(entry, vaddr, size)) {
				entry.has_value = false;
			}
		}
	}

	// The range left the GPU-modified state because a known fill was handed to the guest. Unlike Forget, the part of a fill
	// beyond the range stays known: an entry is cut down to the single side that remains; when both sides remain (cannot be
	// represented in place) or none does, it stops describing bytes, as Forget does.
	void Trim(uint64_t vaddr, uint64_t size) {
		std::lock_guard lock(m_mutex);
		Bump();
		for (auto& entry: m_entries) {
			if (entry.size == 0 || !entry.has_value || !Overlaps(entry, vaddr, size)) {
				continue;
			}
			const uint64_t end   = entry.vaddr + entry.size;
			const bool     left  = entry.vaddr < vaddr;
			const bool     right = vaddr + size < end;
			if (left && !right) {
				entry.size = vaddr - entry.vaddr;
			} else if (right && !left) {
				entry.vaddr = vaddr + size;
				entry.size  = end - entry.vaddr;
			} else {
				entry.has_value = false;
			}
		}
	}

	// Walks the entries newest first. Known = the newest entry touching the range is one fill covering all of it.
	[[nodiscard]] FillQuery QueryFill(uint64_t vaddr, uint64_t size) const {
		std::lock_guard lock(m_mutex);
		return QueryFillLocked(vaddr, size);
	}

	[[nodiscard]] bool LatestIsFill(uint64_t vaddr, uint64_t size, uint32_t* value) const {
		const auto query = QueryFill(vaddr, size);
		if (query.kind != FillKind::Known) {
			return false;
		}
		*value = query.value;
		return true;
	}

	// S562 (KYTY_LEDGER_MEMO): LatestIsFill through the per-thread memo (mode Off: LatestIsFill).
	[[nodiscard]] bool LatestIsFillMemo(uint64_t vaddr, uint64_t size, uint32_t* value, LedgerMemo::Mode mode, LedgerMemo::Stats& stats) const {
		if (mode == LedgerMemo::Mode::Off) {
			return LatestIsFill(vaddr, size, value);
		}
		auto&      slot  = MemoSlot(vaddr, size);
		const auto epoch = m_epoch.load(std::memory_order_acquire);
		FillQuery  query {};
#if defined(KYTY_LEDGER_MUTANT_MEMO_NO_EPOCH)
		(void)epoch;
		const bool same_epoch = true;
#else
		const bool same_epoch = slot.epoch == epoch;
#endif
#if defined(KYTY_LEDGER_MUTANT_MEMO_NO_OWNER)
		const bool same_owner = slot.owner != 0;
#else
		const bool same_owner = slot.owner == m_id;
#endif
		if (same_owner && same_epoch && slot.vaddr == vaddr && slot.size == size) {
			query = slot.query;
			if (mode == LedgerMemo::Mode::Shadow) {
				uint64_t  at = 0;
				FillQuery fresh {};
				{
					std::lock_guard lock(m_mutex);
					at    = m_epoch.load(std::memory_order_relaxed);
					fresh = QueryFillLocked(vaddr, size);
				}
				if (at != slot.epoch) {
					stats.miss++; // the ledger changed after the epoch was read: a miss now, not a disagreement
					slot.epoch = at;
					slot.query = fresh;
				} else if (fresh.kind != query.kind || (fresh.kind == FillKind::Known && fresh.value != query.value)) {
					stats.ne++;
				} else {
					stats.hit++;
				}
				query = fresh;
			} else {
				stats.hit++;
			}
		} else {
			uint64_t at = 0;
			{
				std::lock_guard lock(m_mutex);
				at    = m_epoch.load(std::memory_order_relaxed);
				query = QueryFillLocked(vaddr, size);
			}
			slot = {m_id, vaddr, size, at, query};
			stats.miss++;
		}
		if (query.kind != FillKind::Known) {
			return false;
		}
		*value = query.value;
		return true;
	}

private:
	[[nodiscard]] FillQuery QueryFillLocked(uint64_t vaddr, uint64_t size) const {
		FillQuery       query {};
		bool            newer_seen = false;
		for (uint32_t i = 0; i < Size; ++i) {
			const auto& entry = m_entries[(m_next + Size - 1 - i) % Size];
			if (entry.size == 0 || !Overlaps(entry, vaddr, size)) {
				continue;
			}
			if (!entry.has_value) {
#ifdef KYTY_LEDGER_MUTANT_IGNORE_NEWER_WRITE
				continue; // mutant: a later non-fill write (or a forgotten fill) no longer hides an older fill
#else
				newer_seen = true; // a newer write (or a forgotten fill) hides what is older
				continue;
#endif
			}
			if (!query.seen) {
				query.value = entry.value;
				query.seen  = true;
			}
#ifdef KYTY_LEDGER_MUTANT_PARTIAL_COVER
			// mutant: a fill that covers only part of the range counts as covering it
			query.kind  = newer_seen ? FillKind::NewerWrite : FillKind::Known;
			query.value = entry.value;
			return query;
#endif
			if (vaddr >= entry.vaddr && vaddr + size <= entry.vaddr + entry.size) {
				if (newer_seen) {
					query.kind = FillKind::NewerWrite;
				} else {
					query.kind  = FillKind::Known;
					query.value = entry.value;
				}
				return query;
			}
			newer_seen = true; // a fill of only part of the range hides what is older
		}
		return query;
	}

	struct MemoEntry {
		uint64_t  owner = 0; // m_id of the ledger (0: empty)
		uint64_t  vaddr = 0;
		uint64_t  size  = 0;
		uint64_t  epoch = 0;
		FillQuery query {};
	};
	static constexpr size_t kMemoSlots = 64;
	[[nodiscard]] static MemoEntry& MemoSlot(uint64_t vaddr, uint64_t size) {
		thread_local std::array<MemoEntry, kMemoSlots> table {};
		auto hash = (vaddr >> 12u) ^ (size * 0x9e3779b97f4a7c15ull);
		hash ^= hash >> 29u;
		return table[hash % kMemoSlots];
	}
	[[nodiscard]] static uint64_t NextId() {
		static std::atomic<uint64_t> next {1};
		return next.fetch_add(1, std::memory_order_relaxed);
	}
	// Under m_mutex: the entries change.
	void Bump() { m_epoch.store(m_epoch.load(std::memory_order_relaxed) + 1, std::memory_order_release); }

	struct Entry {
		uint64_t vaddr     = 0;
		uint64_t size      = 0;
		bool     has_value = false;
		uint32_t value     = 0;
	};
	static bool Overlaps(const Entry& entry, uint64_t vaddr, uint64_t size) {
		return entry.vaddr < vaddr + size && vaddr < entry.vaddr + entry.size;
	}
	void Push(const Entry& entry) {
		if (entry.size == 0) {
			return;
		}
		std::lock_guard lock(m_mutex);
		Bump();
		m_entries[m_next++ % Size] = entry;
		m_next %= Size;
	}

	mutable std::mutex      m_mutex;
	std::array<Entry, Size> m_entries {};
	uint32_t                m_next = 0;
	// S562 (KYTY_LEDGER_MEMO): grows with every change of the entries (under m_mutex); the memo's answers are valid at their epoch.
	std::atomic<uint64_t> m_epoch {1};
	// Tells memo entries of a destroyed ledger from this one's.
	const uint64_t m_id = NextId();
};

} // namespace Libs::Graphics

#endif
