#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPECHECKED_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPECHECKED_H

// S609 (KYTY_CP_PIPE_SIGNALS=checked, default fifo): a draw's job starts at parse time instead of after the newest ordered item before it, and the recorder's
// join checks what the gate used to order.
//
// Why the fifo gate is mostly order the join already sees (pm4Handlers.cpp, graphicsRun.cpp): a worker reads guest memory (the walker's loads, the
// specialization and fetch reads: every page is logged, GfpJob::pages / cw_pages). Between the parse and the draw the recorder changes those bytes in
// three ways only:
//   1. GPU work it records (dispatches, DMA/COPY_DATA on the GPU path): the written ranges become GPU-dirty at recording, so the join's page check refuses
//      the job (the guest bytes do not change at recording; they change when a download publishes, which the backing write log refuses);
//   2. the publish of a download (BackingWriteLog, refused since S479);
//   3. CPU stores of the ordered packets themselves: WRITE_DATA, the CPU paths of DMA_DATA/COPY_DATA (BufferCache::FillBuffer/CopyBuffer on a range that is
//      not GPU-modified), the RELEASE_MEM label, the synthetic occlusion dump of EVENT_WRITE 0x39 (16 DB pairs = 256 bytes). The join cannot see these.
// So: the front notes the CPU-write range of every ordered item (ClassifyItem); a draw parsed while such items are still in the FIFO takes a copy of their
// ranges (Tracker::Capture), and the join refuses the job when a page it read lies in one of them (Overlaps). Only an item whose CPU writes have no known
// extent (a flip, an EVENT_WRITE that writes an address other than the synthetic dump, any write when the host occlusion queries defer their dumps to
// later items) still holds the jobs behind it, as fifo does. Dispatches never hold a job (case 1).
// The guest's own writes are ordered as before: WAIT_REG_MEM is answered by the front (drains), signals' interrupts reach the guest only after the GPU ran them.
//
// Mutants (WILL_FAIL, tests/CpPipeCheckedTests.cpp): KYTY_CPCHK_MUTANT_NO_CAPTURE (a free job carries no ranges), _UNKNOWN_NO_GATE (an item of unknown extent
// holds nothing), _JOIN_IGNORES_WRITES (Overlaps is always false), _PRUNE_NEXT (the writes of item done + 1 are dropped as if it had run), _RELEASE_NONE
// (RELEASE_MEM writes nothing), _OCCLUSION_SMALL (the synthetic dump is one qword), _HOST_OCCLUSION_KNOWN (deferred dumps are not unknown), _LAST_PAGE
// (a range's last page is not checked), _DMA_NONE (DMA_DATA writes nothing).

#include "graphics/guest_gpu/command_processor/cpPipe.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace Libs::Graphics::CpPipe::Checked {

enum class Kind : uint8_t { None, Known, Unknown };

struct ItemWrite {
	Kind     kind    = Kind::None;
	uint64_t address = 0;
	uint64_t size    = 0;
};

// The synthetic occlusion dump (CommandProcessor::TriggerEvent, event 0x39 without --occlusion-queries): 16 DBs, one begin/end pair of qwords each.
inline constexpr uint64_t kOcclusionDumpBytes = 16u * 2u * sizeof(uint64_t);

// What the recorder will store into guest memory from the CPU when it runs the item (cls/opcode/r as the front classified it, payload = the dwords after the
// header, packet_dw with the header). host_occlusion: OcclusionQueries::Enabled(): dumps are queued and written by a later flush, label or flip.
inline ItemWrite ClassifyItem(OpClass cls, uint32_t opcode, uint32_t r, const uint32_t* payload, uint32_t packet_dw, bool async_queue, bool host_occlusion) {
	if (cls != OpClass::Ordered) return {}; // draws and dispatches: no CPU stores into guest memory (GPU writes are case 1)
#if !defined(KYTY_CPCHK_MUTANT_HOST_OCCLUSION_KNOWN)
	if (host_occlusion) return {Kind::Unknown, 0, 0}; // any of them may publish a queued dump (RELEASE_MEM, a flush, a flip)
#else
	(void)host_occlusion;
#endif
	switch (opcode) {
		case Pm4::IT_WRITE_DATA: {
			if (packet_dw < 5u) return {Kind::Unknown, 0, 0};
			const auto w = DecodeWriteData(payload, packet_dw);
			return w.size == 0 ? ItemWrite {} : ItemWrite {Kind::Known, w.address, w.size};
		}
		case Pm4::IT_DMA_DATA: {
			if (packet_dw < 7u) return {Kind::Unknown, 0, 0};
#if defined(KYTY_CPCHK_MUTANT_DMA_NONE)
			return {};
#else
			const auto w = DecodeDmaData(payload);
			return w.size == 0 ? ItemWrite {} : ItemWrite {Kind::Known, w.address, w.size};
#endif
		}
		case Pm4::IT_COPY_DATA: {
			if (packet_dw < 6u) return {Kind::Unknown, 0, 0};
			const auto w = DecodeCopyData(payload);
			return {Kind::Known, w.address, w.size};
		}
		case Pm4::IT_RELEASE_MEM: {
			if (packet_dw < 8u) return {Kind::Unknown, 0, 0};
			uint64_t   address = 0;
			const auto w       = DecodeReleaseMem(payload, async_queue, &address);
#if defined(KYTY_CPCHK_MUTANT_RELEASE_NONE)
			(void)w;
			return {};
#else
			return w.writes ? ItemWrite {Kind::Known, address, w.size} : ItemWrite {};
#endif
		}
		case Pm4::IT_EVENT_WRITE: {
			if (packet_dw <= 3u) return {}; // the cache and partial-flush events: no address, no store (TriggerEvent)
			const uint32_t event_type = payload[0] & 0x3fu;
			if (event_type != 0x39u || packet_dw != 4u) return {Kind::Unknown, 0, 0};
			const uint64_t address = payload[1] | (static_cast<uint64_t>(payload[2]) << 32u);
#if defined(KYTY_CPCHK_MUTANT_OCCLUSION_SMALL)
			return {Kind::Known, address, 8u};
#else
			return {Kind::Known, address, kOcclusionDumpBytes};
#endif
		}
		case Pm4::IT_NOP:
			(void)r; // a flip (labels, the video-out event) or anything else the front orders: no known extent
			return {Kind::Unknown, 0, 0};
		default: return {Kind::Unknown, 0, 0};
	}
}

// Front thread only. The CPU-write ranges of the ordered items still in the FIFO, oldest first.
class Tracker {
public:
	void Note(uint64_t seq, const ItemWrite& write) {
		if (write.kind == Kind::Known) {
			m_writes.push_back({seq, {write.address, write.size}});
		} else if (write.kind == Kind::Unknown) {
#if !defined(KYTY_CPCHK_MUTANT_UNKNOWN_NO_GATE)
			m_unknown = seq;
#endif
		}
	}

	// A draw parsed now, with `done` = the newest item the recorder has executed (a stale, smaller value only adds ranges and holds longer). Puts the pages the
	// items after `done` store into as sorted, disjoint, page-aligned ranges into `out` (replacing it; the same label or dump buffer written again and again is
	// one range, so the join asks few) and returns the item the job has to wait for (0: it may start now).
	uint64_t Capture(uint64_t done, std::vector<WriteRange>* out) {
		Prune(done);
		out->clear();
#if !defined(KYTY_CPCHK_MUTANT_NO_CAPTURE)
		constexpr uint64_t kMask = ~uint64_t {4095};
		for (size_t i = m_head; i < m_writes.size(); i++) {
			const auto& w = m_writes[i].range;
			if (w.size == 0) continue;
			const uint64_t first = w.address & kMask;
			const uint64_t last  = (UINT64_MAX - w.address < w.size ? UINT64_MAX : w.address + w.size - 1u) & kMask;
			out->push_back({first, last - first}); // size = the distance to the last page (the page itself is implied), no overflow at the top
		}
		std::sort(out->begin(), out->end(), [](const WriteRange& a, const WriteRange& b) { return a.address < b.address; });
		size_t n = 0;
		for (size_t i = 0; i < out->size(); i++) {
			const auto& w = (*out)[i];
			if (n != 0 && w.address <= (*out)[n - 1].address + (*out)[n - 1].size + 4096u) { // overlapping or the next page
				auto&          prev = (*out)[n - 1];
				const uint64_t last = std::max(prev.address + prev.size, w.address + w.size);
				prev.size           = last - prev.address;
			} else {
				(*out)[n++] = w;
			}
		}
		out->resize(n);
		for (auto& w: *out) w.size = w.size > UINT64_MAX - 4096u ? UINT64_MAX : w.size + 4096u; // back to a byte count: through the end of the last page
#endif
		return m_unknown > done ? m_unknown : 0u;
	}

	// Ranges still held (tests).
	[[nodiscard]] size_t Pending() const { return m_writes.size() - m_head; }

	// A new stream: nothing of the old one is in the FIFO any more (every item before the End item ran).
	void Clear() {
		m_writes.clear();
		m_head    = 0;
		m_unknown = 0;
	}

private:
	struct Entry {
		uint64_t   seq;
		WriteRange range;
	};

	void Prune(uint64_t done) {
#if defined(KYTY_CPCHK_MUTANT_PRUNE_NEXT)
		done++;
#endif
		while (m_head < m_writes.size() && m_writes[m_head].seq <= done) m_head++;
		if (m_head == m_writes.size()) {
			m_writes.clear();
			m_head = 0;
		} else if (m_head >= 256u) {
			m_writes.erase(m_writes.begin(), m_writes.begin() + static_cast<std::ptrdiff_t>(m_head));
			m_head = 0;
		}
	}

	std::vector<Entry> m_writes;
	size_t             m_head    = 0;
	uint64_t           m_unknown = 0;
};

// The join: true when one of the 4 KiB pages in `pages` (sorted, page-aligned addresses) lies in one of the write ranges.
inline bool Overlaps(const std::vector<WriteRange>& writes, const std::vector<uint64_t>& pages) {
#if defined(KYTY_CPCHK_MUTANT_JOIN_IGNORES_WRITES)
	(void)writes;
	(void)pages;
	return false;
#else
	constexpr uint64_t kMask = ~uint64_t {4095};
	if (pages.empty()) return false;
	for (const auto& w: writes) {
		if (w.size == 0) continue;
		const uint64_t first = w.address & kMask;
		const uint64_t end   = UINT64_MAX - w.address < w.size ? UINT64_MAX : w.address + w.size;
#if defined(KYTY_CPCHK_MUTANT_LAST_PAGE)
		const uint64_t last = first;
#else
		const uint64_t last = (end - 1u) & kMask;
#endif
		const auto it = std::lower_bound(pages.begin(), pages.end(), first);
		if (it != pages.end() && *it <= last) return true;
	}
	return false;
#endif
}

} // namespace Libs::Graphics::CpPipe::Checked

#endif
