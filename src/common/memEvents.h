#ifndef KYTY_COMMON_MEMEVENTS_H_
#define KYTY_COMMON_MEMEVENTS_H_

// S125 watch_tex (DIAG): a ring of guest-memory events kept from the start of the run, so that when a texture address becomes
// interesting (control file `watch_tex=`) the history of that address range can be listed: who mapped / unmapped / committed it,
// which file read or GPU write targeted it, when the CPU last wrote into it.
//
// Metadata only: addresses, sizes, flips, kernel range names and guest file paths. Never contents. Recording is on only when
// KYTY_CONTROL_FILE is set (otherwise Record() is one relaxed atomic load and a return).

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace Common::MemEvents {

enum class Kind : uint8_t {
	Reserve,         // virtual range added as Reserved / PoolReserved
	Map,             // virtual range added (direct / flexible / pooled / ...); aux = direct-memory offset
	Remap,           // ReplaceSpan: a part of a reservation became a committed range; aux = direct-memory offset
	Unmap,           // range removed
	ReleaseReserved, // reservation released
	Rename,          // range renamed; tag = new name
	Protect,         // protection changed; aux = new protection
	PoolCommit,      // sceKernelMemoryPoolCommit; result = return code
	PoolDecommit,    // sceKernelMemoryPoolDecommit; result = return code
	PoolMoveEinval,  // POOL_OP_MOVE (not implemented -> EINVAL); tag "dst" / "src" (aux = the other end)
	FileRead,        // AMPR/APR read of a file into guest memory; aux = file offset, result = return code, tag = path
	GpuWriteData,    // PM4 WRITE_DATA into guest memory
	GpuDma,          // PM4 DMA_DATA destination; aux = source (or fill value)
	CpuWrite,        // the texture cache's CPU-write tracker saw a write into the range
	Count
};

inline const char* KindName(Kind kind) {
	static const char* const names[] = {"reserve", "map",         "remap",        "unmap",     "release-reserved",
	                                    "rename",  "protect",     "pool-commit",  "pool-decommit", "pool-move-EINVAL",
	                                    "file-read", "gpu-write-data", "gpu-dma", "cpu-write"};
	const auto               index   = static_cast<size_t>(kind);
	return index < sizeof(names) / sizeof(names[0]) ? names[index] : "?";
}

struct Event {
	uint64_t seq    = 0;
	uint64_t flip   = 0; // guest flip in progress (last completed + 1)
	Kind     kind   = Kind::Map;
	uint64_t addr   = 0;
	uint64_t len    = 0;
	uint64_t aux    = 0;
	int64_t  result = 0;
	uint32_t count  = 1; // identical consecutive events are merged
	char     tag[56] = {};
};

// [a, a+alen) and [b, b+blen) share at least one byte (empty ranges share none; no overflow).
inline bool Intersects(uint64_t a, uint64_t alen, uint64_t b, uint64_t blen) {
	if (alen == 0 || blen == 0) return false;
	const uint64_t a_end = (UINT64_MAX - a < alen) ? UINT64_MAX : a + alen;
	const uint64_t b_end = (UINT64_MAX - b < blen) ? UINT64_MAX : b + blen;
#ifdef KYTY_MEVENTS_MUTANT_INCLUSIVE_END
	return a <= b_end && b <= a_end;
#else
	return a < b_end && b < a_end;
#endif
}

class Ring {
public:
	explicit Ring(size_t capacity) : m_events(capacity == 0 ? 1 : capacity) {}

	void Record(Kind kind, uint64_t flip, uint64_t addr, uint64_t len, uint64_t aux = 0, int64_t result = 0, const char* tag = nullptr) {
		std::lock_guard lock(m_mutex);
		m_totals[static_cast<size_t>(kind)]++;
		if (m_next != 0) {
			Event& last = m_events[(m_next - 1) % m_events.size()];
#ifdef KYTY_MEVENTS_MUTANT_MERGE_ANY_ADDR
			const bool same_addr = true;
#else
			const bool same_addr = last.addr == addr && last.len == len;
#endif
			if (last.kind == kind && last.flip == flip && same_addr && last.aux == aux && last.result == result &&
			    (tag == nullptr ? last.tag[0] == 0 : std::strncmp(last.tag, tag, sizeof(last.tag) - 1) == 0)) {
				last.count++;
				return;
			}
		}
		Event& e = m_events[m_next % m_events.size()];
		e        = Event {};
		e.seq    = m_next;
		e.flip   = flip;
		e.kind   = kind;
		e.addr   = addr;
		e.len    = len;
		e.aux    = aux;
		e.result = result;
		if (tag != nullptr) std::snprintf(e.tag, sizeof(e.tag), "%s", tag);
		m_next++;
	}

	// The newest `max_events` events that intersect [addr, addr+len), oldest first. An event is "in" the range when its own
	// [addr, addr+len) intersects it (a zero-length event counts as the single byte at addr).
	std::vector<Event> Query(uint64_t addr, uint64_t len, size_t max_events) const {
		std::lock_guard    lock(m_mutex);
		std::vector<Event> out;
		const uint64_t     have = std::min<uint64_t>(m_next, m_events.size());
		for (uint64_t i = 0; i < have && out.size() < max_events; i++) {
			const Event& e = m_events[(m_next - 1 - i) % m_events.size()];
			if (Intersects(addr, len, e.addr, e.len == 0 ? 1 : e.len)) out.push_back(e);
		}
		std::reverse(out.begin(), out.end());
		return out;
	}

	// Events of `kind` still in the ring that intersect the range, counting merged repeats.
	uint64_t CountInRange(Kind kind, uint64_t addr, uint64_t len) const {
		std::lock_guard lock(m_mutex);
		uint64_t        n    = 0;
		const uint64_t  have = std::min<uint64_t>(m_next, m_events.size());
		for (uint64_t i = 0; i < have; i++) {
			const Event& e = m_events[(m_next - 1 - i) % m_events.size()];
			if (e.kind == kind && Intersects(addr, len, e.addr, e.len == 0 ? 1 : e.len)) n += e.count;
		}
		return n;
	}

	// Oldest and newest event of `kind` still in the ring that intersects the range; false when there is none.
	bool FirstLast(Kind kind, uint64_t addr, uint64_t len, Event* first, Event* last) const {
		std::lock_guard lock(m_mutex);
		bool            found = false;
		const uint64_t  have  = std::min<uint64_t>(m_next, m_events.size());
		for (uint64_t i = 0; i < have; i++) { // newest first
			const Event& e = m_events[(m_next - 1 - i) % m_events.size()];
			if (e.kind != kind || !Intersects(addr, len, e.addr, e.len == 0 ? 1 : e.len)) continue;
			if (!found) *last = e;
			*first = e;
			found  = true;
		}
		return found;
	}

	// Lifetime counters (not limited by the ring size).
	uint64_t Total(Kind kind) const {
		std::lock_guard lock(m_mutex);
		return m_totals[static_cast<size_t>(kind)];
	}
	uint64_t Recorded() const {
		std::lock_guard lock(m_mutex);
		return m_next;
	}
	size_t Capacity() const { return m_events.size(); }
	bool   Wrapped() const {
        std::lock_guard lock(m_mutex);
        return m_next > m_events.size();
	}

private:
	mutable std::mutex m_mutex;
	std::vector<Event> m_events;
	uint64_t           m_next = 0;
	uint64_t           m_totals[static_cast<size_t>(Kind::Count)] = {};
};

// ---- process-wide instance ---------------------------------------------------------------------------------------------------

inline std::atomic<bool>& EnabledFlag() {
	static std::atomic<bool> flag {std::getenv("KYTY_CONTROL_FILE") != nullptr};
	return flag;
}
inline std::atomic<uint64_t>& FlipCounter() {
	static std::atomic<uint64_t> flips {0};
	return flips;
}
inline Ring& Global() {
	static Ring ring(1u << 17);
	return ring;
}

inline bool Enabled() {
	return EnabledFlag().load(std::memory_order_relaxed);
}
// Called when a guest flip completes (FrameTiming::OnGuestFlip); events recorded afterwards belong to flip n + 1.
inline void SetCompletedFlips(uint64_t n) {
	FlipCounter().store(n, std::memory_order_relaxed);
}
inline void Record(Kind kind, uint64_t addr, uint64_t len, uint64_t aux = 0, int64_t result = 0, const char* tag = nullptr) {
	if (!Enabled()) return;
	Global().Record(kind, FlipCounter().load(std::memory_order_relaxed) + 1, addr, len, aux, result, tag);
}

// One text line per event, for [watch-tex] output.
inline std::string Format(const Event& e) {
	char buf[320];
	std::snprintf(buf, sizeof(buf), "flip=%llu %s addr=0x%llx len=0x%llx aux=0x%llx result=%lld x%u %s", static_cast<unsigned long long>(e.flip),
	              KindName(e.kind), static_cast<unsigned long long>(e.addr), static_cast<unsigned long long>(e.len),
	              static_cast<unsigned long long>(e.aux), static_cast<long long>(e.result), e.count, e.tag);
	return buf;
}

// The [watch-tex] history of [addr, addr+len): ring state, per-kind totals / in-range counts / first and last event, then the newest
// `max_events` events in the range (oldest first).
inline std::vector<std::string> WatchLines(const Ring& ring, uint64_t addr, uint64_t len, size_t max_events) {
	std::vector<std::string> lines;
	char                     buf[400];
	std::snprintf(buf, sizeof(buf), "history range=[0x%llx, 0x%llx) ring: recorded=%llu capacity=%zu wrapped=%d (events older than the ring are lost)",
	              static_cast<unsigned long long>(addr), static_cast<unsigned long long>(addr + len), static_cast<unsigned long long>(ring.Recorded()),
	              ring.Capacity(), ring.Wrapped() ? 1 : 0);
	lines.emplace_back(buf);
	std::snprintf(buf, sizeof(buf), "counters: MOVE->EINVAL global=%llu in_range=%llu", static_cast<unsigned long long>(ring.Total(Kind::PoolMoveEinval)),
	              static_cast<unsigned long long>(ring.CountInRange(Kind::PoolMoveEinval, addr, len)));
	lines.emplace_back(buf);
	for (uint8_t k = 0; k < static_cast<uint8_t>(Kind::Count); k++) {
		const auto kind = static_cast<Kind>(k);
		Event      first, last;
		const auto in_range = ring.CountInRange(kind, addr, len);
		std::snprintf(buf, sizeof(buf), "kind %-16s total=%llu in_range=%llu", KindName(kind), static_cast<unsigned long long>(ring.Total(kind)),
		              static_cast<unsigned long long>(in_range));
		std::string line = buf;
		if (ring.FirstLast(kind, addr, len, &first, &last)) {
			std::snprintf(buf, sizeof(buf), " first_flip=%llu last_flip=%llu", static_cast<unsigned long long>(first.flip),
			              static_cast<unsigned long long>(last.flip));
			line += buf;
		}
		lines.push_back(line);
	}
	for (const auto& e: ring.Query(addr, len, max_events)) lines.push_back("event seq=" + std::to_string(e.seq) + " " + Format(e));
	return lines;
}

} // namespace Common::MemEvents

#endif /* KYTY_COMMON_MEMEVENTS_H_ */
