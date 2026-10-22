#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPE_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPE_H

// S479 (R13 stage 1): KYTY_CP_PIPE=0|1|shadow (default 0). The pure part of the CP pipeline: which PM4 packets the front runs at parse time, which it defers
// to the FIFO (recorder runs them later, in stream order), the FIFO ring, the map of the writes that are still in the FIFO (what WAIT_REG_MEM sees), and the
// thread gate for the state that belongs to the CP.
//
// Mutants (WILL_FAIL, tests/CpPipeTests.cpp): KYTY_CPPIPE_MUTANT_SIGNAL_AT_PARSE (signals run at parse), KYTY_CPPIPE_MUTANT_VIRTUAL_WAIT_IGNORES_MASK,
// KYTY_CPPIPE_MUTANT_DRAIN_SKIPS_UNKNOWN (an unknown opcode runs inline without a drain), KYTY_CPPIPE_MUTANT_WORKER_WRITES_CP (the CP-state gate lets a worker
// through), KYTY_CPPIPE_MUTANT_RETIRE_ALL (retiring one FIFO item forgets every pending write).

#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::CpPipe {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

// What the front does with a packet.
//   Inline   - touches only CP register state (or nothing): runs at parse time, exactly as before.
//   Ordered  - touches the command buffer or guest memory (signals, barriers, CPU writes, flips): the packet is copied into the FIFO and the recorder runs the
//              handler at the packet's place in the stream.
//   Draw     - a draw: the registers are snapshotted into the FIFO item, a worker computes GetGraphicsPrograms on the snapshot, the recorder draws.
//   Dispatch - like Draw without the worker job.
//   Wait     - WAIT_REG_MEM: the front evaluates it against the writes still in the FIFO (PendingWrites) and drains only when it has to.
//   Drain    - anything else: the FIFO is emptied first, then the handler runs inline (the safe default).
enum class OpClass : uint8_t { Inline, Ordered, Draw, Dispatch, Wait, Drain };

inline constexpr uint32_t kMarkerTagMask = 0xffff0000u;
inline constexpr uint32_t kMarkerTag     = 0x68750000u;

// payload0: the first dword after the header (0 when the packet has none). The marker NOP carries a tag there.
inline OpClass Classify(uint32_t opcode, uint32_t r, uint32_t payload0, uint32_t packet_dw) {
	using namespace Pm4;
	switch (opcode) {
		case IT_SET_BASE:
		case IT_CLEAR_STATE:
		case IT_INDEX_BASE:
		case IT_INDEX_BUFFER_SIZE:
		case IT_INDEX_TYPE:
		case IT_NUM_INSTANCES:
		case IT_SET_CONTEXT_REG:
		case IT_SET_SH_REG:
		case IT_SET_UCONFIG_REG:
		case IT_SET_UCONFIG_REG_INDEX:
		case IT_PFP_SYNC_ME:
		case IT_ACQUIRE_MEM:
		case IT_INCREMENT_CE_COUNTER:
		case IT_INCREMENT_DE_COUNTER:
		case IT_WRITE_CONST_RAM:
		case IT_INDIRECT_BUFFER: return OpClass::Inline;
		case IT_WRITE_DATA:
		case IT_DMA_DATA:
		case IT_COPY_DATA:
		case IT_EVENT_WRITE:
		case IT_EVENT_WRITE_EOP:
		case IT_EVENT_WRITE_EOS:
#ifdef KYTY_CPPIPE_MUTANT_SIGNAL_AT_PARSE
			return OpClass::Inline;
#else
			return OpClass::Ordered;
#endif
		case IT_DRAW_INDEX_2:
		case IT_DRAW_INDEX_AUTO:
		case IT_DRAW_INDEX_OFFSET_2:
		case IT_DISPATCH_DRAW_PREAMBLE: return OpClass::Draw;
		case IT_DISPATCH_DIRECT:
		case IT_DISPATCH_INDIRECT: return OpClass::Dispatch;
		case IT_WAIT_REG_MEM:
		case IT_WAIT_REG_MEM_64: return OpClass::Wait;
		case IT_NOP:
			switch (r) {
				case R_ZERO:
					if (packet_dw >= 2u && (payload0 & kMarkerTagMask) == kMarkerTag) {
						switch (payload0) {
							case 0x68750777: // flip
							case 0x68750778:
							case 0x68750781: return OpClass::Ordered;
							default: return OpClass::Inline;
						}
					}
					return OpClass::Inline;
				case R_PUSH_MARKER:
				case R_POP_MARKER:
				case R_CONTEXT_STATE:
				case R_DISPATCH_RESET: return OpClass::Inline;
				case R_FLIP:
				case R_RELEASE_MEM:
#ifdef KYTY_CPPIPE_MUTANT_SIGNAL_AT_PARSE
					return OpClass::Inline;
#else
					return OpClass::Ordered;
#endif
				default: break;
			}
			break;
		default: break;
	}
#ifdef KYTY_CPPIPE_MUTANT_DRAIN_SKIPS_UNKNOWN
	return OpClass::Inline;
#else
	return OpClass::Drain;
#endif
}

// A draw that goes to a worker (the rest of the Draw class runs through the FIFO without a job).
inline bool IsGraphicsDraw(uint32_t opcode) {
	return opcode == Pm4::IT_DRAW_INDEX_2 || opcode == Pm4::IT_DRAW_INDEX_AUTO || opcode == Pm4::IT_DRAW_INDEX_OFFSET_2;
}

inline const char* ClassName(OpClass cls) {
	switch (cls) {
		case OpClass::Inline: return "inline";
		case OpClass::Ordered: return "ordered";
		case OpClass::Draw: return "draw";
		case OpClass::Dispatch: return "dispatch";
		case OpClass::Wait: return "wait";
		case OpClass::Drain: return "drain";
	}
	return "?";
}

// The wait compare of PM4 WAIT_REG_MEM (the same table as TestWaitRegMemValue in graphicsRun.cpp).
inline bool WaitCompare(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
#ifdef KYTY_CPPIPE_MUTANT_VIRTUAL_WAIT_IGNORES_MASK
	mask = ~uint64_t {0};
#endif
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: return false;
	}
}

// A write that sits in the FIFO and has not reached guest memory yet. `known` = the packet fixes the bytes (an immediate value); a timestamp or a GDS read does not.
struct PendingWrite {
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t value   = 0; // little endian, the low `size` bytes (size <= 8)
	bool     known   = false;
	uint64_t seq     = 0; // the FIFO item that will perform it
};

class PendingWrites {
public:
	enum class Verdict : uint8_t {
		NoPending, // nothing in the FIFO touches the bytes: read guest memory as before
		Value,     // the newest pending write covers the bytes exactly and its value is known: *value is what memory will hold
		Unknown,   // a pending write overlaps but the value is not known (or not exactly the same range): drain, then read
	};

	void Note(uint64_t address, uint64_t size, bool known, uint64_t value, uint64_t seq) {
		if (size == 0) return;
		m_writes.push_back({address, size, value, known && size <= 8u, seq});
	}

	[[nodiscard]] Verdict Resolve(uint64_t address, uint64_t size, uint64_t* value) const {
		for (size_t i = m_writes.size(); i-- > 0;) {
			const auto& w = m_writes[i];
			if (address >= w.address + w.size || w.address >= address + size) continue; // disjoint
			if (w.address == address && w.size == size && w.known) {
				*value = w.value;
				return Verdict::Value;
			}
			return Verdict::Unknown;
		}
		return Verdict::NoPending;
	}

	// The FIFO item `seq` has run: its writes are in guest memory now.
	void Retire(uint64_t seq) {
#ifdef KYTY_CPPIPE_MUTANT_RETIRE_ALL
		(void)seq;
		m_writes.clear();
#else
		size_t kept = 0;
		for (const auto& w: m_writes) {
			if (w.seq != seq) m_writes[kept++] = w;
		}
		m_writes.resize(kept);
#endif
	}

	void Clear() { m_writes.clear(); }
	[[nodiscard]] size_t Size() const { return m_writes.size(); }

private:
	std::vector<PendingWrite> m_writes;
};

// The ring of FIFO slots: indices only. `seq` grows with every Commit, so a slot's sequence number is stable while it is in the ring.
class Ring {
public:
	explicit Ring(size_t capacity) : m_capacity(capacity) {}

	[[nodiscard]] size_t Capacity() const { return m_capacity; }
	[[nodiscard]] bool   Empty() const { return m_count == 0; }
	[[nodiscard]] bool   Full() const { return m_count >= m_capacity; }
	[[nodiscard]] size_t Size() const { return m_count; }

	// The slot to fill; call Commit() after it is filled (the slot is not visible to Front() before).
	[[nodiscard]] size_t Reserve() const { return (m_head + m_count) % m_capacity; }
	uint64_t             Commit() {
        ++m_count;
        return m_next_seq++;
	}
	[[nodiscard]] size_t   Front() const { return m_head; }
	[[nodiscard]] uint64_t FrontSeq() const { return m_next_seq - m_count; }
	void                   Pop() {
        m_head = (m_head + 1) % m_capacity;
        --m_count;
	}
	// i-th pending slot from the front (0 = oldest).
	[[nodiscard]] size_t At(size_t i) const { return (m_head + i) % m_capacity; }

private:
	size_t   m_capacity;
	size_t   m_head     = 0;
	size_t   m_count    = 0;
	uint64_t m_next_seq = 1;
};

// The state of the CP (index type, instance count, bases, registers) belongs to the CP thread. A worker gets a snapshot and never writes the CP: every setter
// goes through this gate (S478: the scout's replay wrote m_index_type_and_size and the GPU ran twice as long).
inline bool& WorkerThreadFlag() {
	static thread_local bool on_worker = false;
	return on_worker;
}

class WorkerScope {
public:
	WorkerScope() { WorkerThreadFlag() = true; }
	~WorkerScope() { WorkerThreadFlag() = false; }
	WorkerScope(const WorkerScope&)            = delete;
	WorkerScope& operator=(const WorkerScope&) = delete;
};

// True when the calling thread may write CP state.
inline bool MayWriteCpState() {
#ifdef KYTY_CPPIPE_MUTANT_WORKER_WRITES_CP
	return true;
#else
	return !WorkerThreadFlag();
#endif
}

template <class T, class V>
bool WriteCpMember(T& member, V value) {
	if (!MayWriteCpState()) return false;
	member = static_cast<T>(value);
	return true;
}

// Signals: what an EOP/RELEASE_MEM packet writes to guest memory (the part WAIT_REG_MEM can be answered from). Mirrors WriteAtEndOfPipe in graphicsRun.cpp; when in
// doubt the value is `known = false`, which makes a wait on the range drain the FIFO first (never a wrong answer, only a shorter window).
struct SignalWrite {
	bool     writes = false;
	uint64_t size   = 0;
	uint64_t value  = 0;
	bool     known  = false;
};

// Interrupt selector 1 on the graphics queue only raises the EOP interrupt (WriteAtEndOfPipe returns before the write).
inline bool InterruptOnly(uint32_t interrupt_selector, bool async_queue) {
	return interrupt_selector == 1u && !async_queue;
}

// NOP R_RELEASE_MEM payload (the dwords after the header): [1] = data_sel / interrupt selector, [2..3] = address, [4..5] = value.
inline SignalWrite DecodeReleaseMem(const uint32_t* payload, bool async_queue, uint64_t* address) {
	SignalWrite    out;
	const uint32_t data_sel  = (payload[1] >> 29u) & 0x7u;
	const uint32_t interrupt = (payload[1] >> 24u) & 0x7u;
	*address                 = payload[2] | (static_cast<uint64_t>(payload[3]) << 32u);
	const uint64_t value     = payload[4] | (static_cast<uint64_t>(payload[5]) << 32u);
	if (data_sel == 0 || interrupt == 4u || *address == 0 || InterruptOnly(interrupt, async_queue)) return out;
	switch (data_sel) {
		case 1: out = {true, 4u, static_cast<uint32_t>(value), true}; return out;
		case 2: out = {true, 8u, value, true}; return out;
		case 3: out = {true, 8u, 0u, false}; return out; // timestamp
		case 5: out = {true, 4u + 4u * (static_cast<uint32_t>(value) >> 16u), 0u, false}; return out; // GDS read
		default: out = {true, 8u, 0u, false}; return out;
	}
}

// IT_EVENT_WRITE_EOP payload: [1] = address low, [2] = address high + selectors, [3..4] = value.
inline SignalWrite DecodeEventWriteEop(const uint32_t* payload, bool async_queue, uint64_t* address) {
	SignalWrite    out;
	const uint32_t source    = (payload[2] >> 29u) & 0x7u;
	const uint32_t interrupt = (payload[2] >> 24u) & 0x7u;
	const uint64_t value     = payload[3] | (static_cast<uint64_t>(payload[4]) << 32u);
	*address                 = payload[1] | (static_cast<uint64_t>(payload[2] & 0xffffu) << 32u);
	if (*address == 0 || InterruptOnly(interrupt, async_queue)) return out;
	switch (source) {
		case 1: out = {true, 4u, static_cast<uint32_t>(value), true}; return out;
		case 2: out = {true, 8u, value, true}; return out;
		default: out = {true, 8u, 0u, false}; return out; // 4 = reference clock
	}
}

// IT_EVENT_WRITE_EOS payload: [1] = address low, [2] = address high + selectors, [3] = 32-bit value.
inline SignalWrite DecodeEventWriteEos(const uint32_t* payload, bool async_queue, uint64_t* address) {
	SignalWrite    out;
	const uint32_t source    = (payload[2] >> 29u) & 0x7u;
	const uint32_t interrupt = (payload[2] >> 24u) & 0x7u;
	*address                 = payload[1] | (static_cast<uint64_t>(payload[2] & 0xffffu) << 32u);
	if (*address == 0 || InterruptOnly(interrupt, async_queue)) return out;
	out = {true, 4u, payload[3], source == 2u};
	return out;
}

// CPU-visible writes of the other ordered packets: the range only (the bytes are not needed, a wait on it drains).
struct WriteRange {
	uint64_t address = 0;
	uint64_t size    = 0;
};

// IT_WRITE_DATA payload: [0] = control, [1..2] = address, [3..] = data; packet_dw includes the header.
inline WriteRange DecodeWriteData(const uint32_t* payload, uint32_t packet_dw) {
	const bool     one   = ((payload[0] >> 16u) & 0x1u) != 0;
	const uint32_t count = packet_dw >= 4u ? packet_dw - 4u : 0u;
	return {payload[1] | (static_cast<uint64_t>(payload[2]) << 32u), one ? 4u : static_cast<uint64_t>(count) * 4u};
}

// IT_DMA_DATA payload: [3..4] = destination, [5] low 26 bits = byte count.
inline WriteRange DecodeDmaData(const uint32_t* payload) {
	return {payload[3] | (static_cast<uint64_t>(payload[4]) << 32u), payload[5] & 0x03ffffffu};
}

// IT_COPY_DATA payload: [0] bit 16 = 64-bit, [3..4] = destination.
inline WriteRange DecodeCopyData(const uint32_t* payload) {
	return {payload[3] | (static_cast<uint64_t>(payload[4]) << 32u), ((payload[0] >> 16u) & 0x1u) != 0 ? 8u : 4u};
}

} // namespace Libs::Graphics::CpPipe

#endif
