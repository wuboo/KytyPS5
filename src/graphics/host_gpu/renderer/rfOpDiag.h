#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RFOPDIAG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RFOPDIAG_H_

// S412 (DIAG v4, default off): who reads a GPU-modified page on the command-processor thread.
//   KYTY_DIAG_RFOP=1   a page read fault raised on the GPU thread (SyncCensus site rf_gpu) logs one [rf-op] line: the PM4 opcode being
//                      handled, the first four dwords of its packet, the fault address, the faulting PC (symbol + offset), the opcode of
//                      the last GPU writer of the page, and 16 bytes at the fault address before and after the drain. It also counts
//                      the fault per opcode in the frames.csv columns rf_op_<name>_n.
// This header is the pure part (tests/RfOpDiagTests.cpp pins it). Nothing here feeds a shader or pipeline cache key.

#include "graphics/host_gpu/renderer/syncCensus.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include <dlfcn.h>

namespace Libs::Graphics::RfOp {

inline bool Enabled() {
	static const bool on = SyncCensus::FlagOn("KYTY_DIAG_RFOP");
	return on;
}

constexpr uint32_t kNoOpcode = 0xFFFFFFFFu;

// The opcodes with their own column; every other opcode counts in "other", a fault outside any packet handler in "none".
// The eight are the ones a handler reads guest memory in, or that wait on it (predication, indirect args, wait_reg_mem).
constexpr uint32_t kTrackedOpcodes[] = {0x3C, 0x22, 0x2C, 0x38, 0x16, 0x20, 0x47, 0x3F};
constexpr uint32_t kTrackedCount     = static_cast<uint32_t>(sizeof(kTrackedOpcodes) / sizeof(kTrackedOpcodes[0]));
constexpr uint32_t kSlotOther        = kTrackedCount;
constexpr uint32_t kSlotNone         = kTrackedCount + 1;
constexpr uint32_t kSlotCount        = kTrackedCount + 2;

constexpr uint32_t SlotOf(uint32_t opcode) {
	if (opcode == kNoOpcode) {
		return kSlotNone;
	}
	for (uint32_t slot = 0; slot < kTrackedCount; slot++) {
		if (kTrackedOpcodes[slot] == opcode) {
			return slot;
		}
	}
	return kSlotOther;
}

// frames.csv column of a slot (FrameTiming::Counter::RfOpWaitRegMemN + slot; frameTiming.cpp checks the names against this).
constexpr const char* ColumnName(uint32_t slot) {
	constexpr const char* names[] = {"rf_op_wait_reg_mem_n",    "rf_op_cond_exec_n",   "rf_op_draw_indirect_multi_n",
	                                 "rf_op_draw_index_indirect_multi_n", "rf_op_dispatch_indirect_n", "rf_op_set_predication_n",
	                                 "rf_op_event_write_eop_n", "rf_op_indirect_buffer_n", "rf_op_other_n", "rf_op_none_n"};
	return slot < kSlotCount ? names[slot] : "?";
}

// The packet the command processor thread is handling (breadcrumb written by the PM4 loop, read by the fault handler on the same
// thread, hence thread-local and plain).
struct Packet {
	uint32_t opcode = kNoOpcode;
	uint32_t dw[4] {};
};
inline Packet& CurrentPacket() {
	static thread_local Packet packet;
	return packet;
}
// packet = header of the PM4 packet, remaining_dw = dwords left in the buffer (the first four are copied, fewer if the buffer ends).
inline void BeginPacket(uint32_t opcode, const uint32_t* packet, uint32_t remaining_dw) {
	auto& current = CurrentPacket();
#if defined(KYTY_RFOP_MUTANT_NO_OPCODE)
	(void)opcode;
	current.opcode = kNoOpcode;
#else
	current.opcode = opcode;
#endif
	for (uint32_t i = 0; i < 4; i++) {
		current.dw[i] = i < remaining_dw && packet != nullptr ? packet[i] : 0;
	}
}
inline void EndPacket() {
	CurrentPacket().opcode = kNoOpcode;
}

// PC of the access violation being handled (set by the exception handler on the faulting thread just before HandleGpuFault).
inline uint64_t& FaultPcRef() {
	static thread_local uint64_t pc = 0;
	return pc;
}

// "symbol+0xoff" through dladdr; "image+0xoff" when the symbol is not exported, "?" for an address outside every image.
inline std::string ResolvePc(uint64_t pc) {
	Dl_info info {};
	if (pc == 0 || dladdr(reinterpret_cast<void*>(pc), &info) == 0) {
		return "?";
	}
	char text[512];
	if (info.dli_sname != nullptr) {
		std::snprintf(text, sizeof(text), "%.300s+0x%llx", info.dli_sname,
		              static_cast<unsigned long long>(pc - reinterpret_cast<uint64_t>(info.dli_saddr)));
	} else {
		const char* name = info.dli_fname != nullptr ? std::strrchr(info.dli_fname, '/') : nullptr;
		name             = name != nullptr ? name + 1 : (info.dli_fname != nullptr ? info.dli_fname : "image");
		std::snprintf(text, sizeof(text), "%.200s+0x%llx", name,
		              static_cast<unsigned long long>(pc - reinterpret_cast<uint64_t>(info.dli_fbase)));
	}
	return text;
}

// 16 bytes at the fault address, read through the backing (not the protected mapping).
struct Bytes16 {
	bool    ok = false;
	uint8_t data[16] {};
};

struct Event {
	uint64_t    flip = 0;
	Packet      packet;
	uint64_t    addr = 0;
	std::string pc_text;
	uint32_t    writer_op   = kNoOpcode; // kNoOpcode = no GPU writer known
	uint64_t    writer_hash = 0;         // 0 = unknown
	Bytes16     stale;
	Bytes16     fresh;
};

inline std::string FormatLine(const Event& event) {
	char op[16];
	if (event.packet.opcode == kNoOpcode) {
		std::snprintf(op, sizeof(op), "none");
	} else {
		std::snprintf(op, sizeof(op), "0x%02x", event.packet.opcode);
	}
	char writer[16];
	if (event.writer_op == kNoOpcode) {
		std::snprintf(writer, sizeof(writer), "-");
	} else {
		std::snprintf(writer, sizeof(writer), "0x%02x", event.writer_op);
	}
	char hash[24];
	if (event.writer_hash == 0) {
		std::snprintf(hash, sizeof(hash), "-");
	} else {
		std::snprintf(hash, sizeof(hash), "%016llx", static_cast<unsigned long long>(event.writer_hash));
	}
	char text[1024];
	std::snprintf(text, sizeof(text),
	              "[rf-op] flip=%llu op=%s pkt=%08x,%08x,%08x,%08x addr=0x%016llx pc=%s writer_op=%s writer_hash=%s stale=%s fresh=%s",
	              static_cast<unsigned long long>(event.flip), op, event.packet.dw[0], event.packet.dw[1], event.packet.dw[2],
	              event.packet.dw[3], static_cast<unsigned long long>(event.addr), event.pc_text.c_str(), writer, hash,
	              SyncCensus::ToHex(event.stale.data, event.stale.ok ? 16 : 0).c_str(),
	              SyncCensus::ToHex(event.fresh.data, event.fresh.ok ? 16 : 0).c_str());
	return text;
}

// At most 400 lines, plus every line of flips 6300-6310.
constexpr bool ShouldLog(uint64_t index, uint64_t flip) {
	return index < 400 || (flip >= 6300 && flip <= 6310);
}

// Opcode of the newest GPU writer per 4 KiB page (the PM4 handler that recorded the write; the writing shader's hash is not known at
// the recording site). Touched on the GPU thread only; the lock is for the diagnostics' sake.
class WriterOps {
public:
	static constexpr uint64_t kPageSize = 4096;
	void Note(uint64_t address, uint64_t size, uint32_t opcode) {
		if (size == 0) {
			return;
		}
		std::lock_guard lock(m_lock);
		for (auto page = address / kPageSize; page <= (address + size - 1) / kPageSize; page++) {
			m_pages[page] = {++m_seq, opcode};
		}
		if (m_pages.size() > kMaxPages) {
			m_pages.clear();
		}
	}
	// Opcode of the newest writer over the pages the range touches; kNoOpcode when none is known.
	[[nodiscard]] uint32_t Find(uint64_t address, uint64_t size) const {
		if (size == 0) {
			return kNoOpcode;
		}
		std::lock_guard lock(m_lock);
		uint64_t        best   = 0;
		uint32_t        result = kNoOpcode;
		for (auto page = address / kPageSize; page <= (address + size - 1) / kPageSize; page++) {
			const auto found = m_pages.find(page);
			if (found != m_pages.end() && found->second.seq > best) {
				best   = found->second.seq;
				result = found->second.opcode;
			}
		}
		return result;
	}

private:
	static constexpr size_t kMaxPages = 1u << 18;
	struct Entry {
		uint64_t seq    = 0;
		uint32_t opcode = kNoOpcode;
	};
	mutable std::mutex                     m_lock;
	std::unordered_map<uint64_t, Entry>    m_pages;
	uint64_t                               m_seq = 0;
};

} // namespace Libs::Graphics::RfOp

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_RFOPDIAG_H_ */
