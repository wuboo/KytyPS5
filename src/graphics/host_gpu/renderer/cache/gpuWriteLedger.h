#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUWRITELEDGER_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_GPUWRITELEDGER_H_

#include <array>
#include <cstdint>
#include <mutex>

namespace Libs::Graphics {

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

	[[nodiscard]] bool LatestIsFill(uint64_t vaddr, uint64_t size, uint32_t* value) const {
		const auto query = QueryFill(vaddr, size);
		if (query.kind != FillKind::Known) {
			return false;
		}
		*value = query.value;
		return true;
	}

private:
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
		m_entries[m_next++ % Size] = entry;
		m_next %= Size;
	}

	mutable std::mutex      m_mutex;
	std::array<Entry, Size> m_entries {};
	uint32_t                m_next = 0;
};

} // namespace Libs::Graphics

#endif
