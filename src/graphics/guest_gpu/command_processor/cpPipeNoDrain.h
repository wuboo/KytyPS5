#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPENODRAIN_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPENODRAIN_H

// S604 (KYTY_PIPE_NODRAIN=0|1|shadow, default 0; needs KYTY_CP_PIPE): the front of the CP pipeline drains the FIFO before a SET_*_REG_INDIRECT whose table a write
// still in the FIFO overlaps, and before a WAIT_REG_MEM the writes in the FIFO cannot answer. S598: 30 of the garden's 35.7 drains per flip are those two. The
// S479 map of the pending writes (PendingWrites, cpPipe.h) is coarse in three places, each of which turns a read the FIFO cannot touch into a drain:
//   - an EVENT_WRITE with an address (the occlusion counter dump, type 0x39: 256 bytes) is noted as a write of all memory, so every table read and every wait
//     after it drains until it has run (the game uses occlusion queries: "Warning: game uses occlusion queries" in every log);
//   - a DMA_DATA to nowhere (an L2 prefetch), to GDS or to the ignored register 0x3022c, and a COPY_DATA to GDS, are noted as writes of guest memory;
//   - a wait answers only from a write of exactly its bytes with a known value: a 32-bit wait on a 64-bit label, or a label set by WRITE_DATA, drains.
// The precise view of the same map (cpPipe.h kModelPrecise) mirrors what the handlers write (pm4Handlers.cpp CpOpEventWrite / CpOpDmaData / CpOpCopyData /
// CpOpWriteData, graphicsRun.cpp TriggerEvent / DmaData / WriteData). Off: the S479 entries only, byte for byte. 1: the drains follow the precise view (the S479
// view still counts what it would have drained). shadow: the drains follow the S479 view; where the precise view would not drain, the front reads the bytes before
// the drain and again after it and counts a difference (nodrain_ne_n).
//
// Mutants (WILL_FAIL, tests/CpPipeNoDrainTests.cpp): KYTY_PIPEND_MUTANT_EVENT_NO_WRITE (the occlusion dump writes nothing), KYTY_PIPEND_MUTANT_DUMP_SHORT (the dump
// is noted as 8 bytes), KYTY_PIPEND_MUTANT_DMA_REAL_DROPPED (a DMA to memory writes nothing), KYTY_PIPEND_MUTANT_WRITE_DATA_FIRST (WRITE_DATA to one address
// keeps the first dword), KYTY_PIPEND_MUTANT_S479_VIEW_CHANGED (the precise mode drops the S479 view's all-memory entry of the dump), KYTY_PIPEND_MUTANT_SHADOW_BLIND
// (the shadow compare never sees a difference), and in cpPipe.h KYTY_PIPEND_MUTANT_SUBRANGE_NO_SHIFT, KYTY_PIPEND_MUTANT_PARTIAL_COVER.

#include "graphics/guest_gpu/command_processor/cpPipe.h"
#include "graphics/guest_gpu/pm4.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::CpPipe {

enum class NoDrainMode : uint8_t { Off, On, Shadow };

inline NoDrainMode ParseNoDrainMode(const char* text) {
	if (text == nullptr) return NoDrainMode::Off;
	if (std::strcmp(text, "1") == 0) return NoDrainMode::On;
	if (std::strcmp(text, "shadow") == 0) return NoDrainMode::Shadow;
	return NoDrainMode::Off;
}

inline const char* NoDrainModeName(NoDrainMode mode) {
	switch (mode) {
		case NoDrainMode::Off: return "0";
		case NoDrainMode::On: return "1";
		case NoDrainMode::Shadow: return "shadow";
	}
	return "?";
}

// KYTY_PIPE_NODRAIN (read once).
inline NoDrainMode NoDrainEnvMode() {
	static const NoDrainMode mode = ParseNoDrainMode(std::getenv("KYTY_PIPE_NODRAIN"));
	return mode;
}

inline constexpr uint64_t kEverything = ~uint64_t {0} >> 1u;
// The bytes of an occlusion counter dump: 16 DBs x {begin, end} x 8 bytes from the address (TriggerEvent's synthetic counters write results[db * 2], the host
// queries' WriteZPassCounters results[db * 2] and results[db * 2 + 1]; an end dump starts 8 bytes in, so it ends at the same 256 from its own address).
inline constexpr uint64_t kOcclusionDumpBytes = 16u * 2u * 8u;
inline constexpr uint32_t kEventOcclusionDump = 0x39u;

inline uint16_t SrcTag(uint32_t opcode, uint32_t sub) {
	return static_cast<uint16_t>(((opcode & 0xffu) << 8u) | (sub & 0xffu));
}

// The writes a FIFO item will do, for the reads that follow it in the stream. `precise` = false: the S479 entries only, exactly as cpPipe.cpp noted them before
// S604 (both views; the tests replay the old switch against this one). `precise` = true: the same S479 entries plus the precise view; an entry the two views agree
// on is noted once for both.
inline void NoteItemWrites(PendingWrites& pending, uint32_t opcode, const uint32_t* packet, uint32_t packet_dw, bool async, uint64_t seq, bool precise) {
	const auto* payload = packet + 1;
	const auto  r       = opcode == Pm4::IT_NOP ? KYTY_PM4_R(packet[0]) : 0u;
	uint64_t    address = 0;
	switch (opcode) {
		case Pm4::IT_NOP:
			if (r == Pm4::R_FLIP || r == Pm4::R_ZERO) pending.NoteViews(0, kEverything, false, false, 0, seq, kModelBoth, SrcTag(opcode, r)); // a flip writes labels
			break;
		case Pm4::IT_RELEASE_MEM:
			if (packet_dw >= 8u) {
				const auto w = DecodeReleaseMem(payload, async, &address);
				if (w.writes) pending.NoteViews(address, w.size, w.known, w.known, w.value, seq, kModelBoth, SrcTag(opcode, (payload[1] >> 29u) & 0x7u));
			}
			break;
		case Pm4::IT_EVENT_WRITE:
			if (packet_dw > 3u) {
				const uint32_t type = payload[0] & 0x3fu;
				const uint16_t src  = SrcTag(opcode, type);
				if (!precise) {
					pending.NoteViews(0, kEverything, false, false, 0, seq, kModelBoth, src); // event types that write an address
					break;
				}
#ifndef KYTY_PIPEND_MUTANT_S479_VIEW_CHANGED
				pending.NoteViews(0, kEverything, false, false, 0, seq, kModelS479, src);
#endif
				if (type == kEventOcclusionDump) {
					if (packet_dw == 4u) {
						const uint64_t dump = payload[1] | (static_cast<uint64_t>(payload[2]) << 32u);
#if defined(KYTY_PIPEND_MUTANT_EVENT_NO_WRITE)
						(void)dump;
#elif defined(KYTY_PIPEND_MUTANT_DUMP_SHORT)
						pending.NoteViews(dump, 8u, false, false, 0, seq, kModelPrecise, src);
#else
						pending.NoteViews(dump, kOcclusionDumpBytes, false, false, 0, seq, kModelPrecise, src);
#endif
					} else {
						pending.NoteViews(0, kEverything, false, false, 0, seq, kModelPrecise, src); // a dump of another length (the handler stops on it): all memory
					}
				}
				// Every other type TriggerEvent accepts only orders GPU work or is ignored: no guest write.
			}
			break;
		case Pm4::IT_WRITE_DATA:
			if (packet_dw >= 5u) {
				const auto     w       = DecodeWriteData(payload, packet_dw);
				const uint32_t control = payload[0];
				const uint16_t src     = SrcTag(opcode, ((control >> 30u) & 0x1u) | ((control >> 7u) & 0x1eu));
				if (!precise) {
					pending.NoteViews(w.address, w.size, false, false, 0, seq, kModelBoth, src);
					break;
				}
				// CpOpWriteData: dw_num - 2 = packet_dw - 4 >= 1 dwords from payload[3]; one address: every dword to the same address (the last one stays).
				const uint32_t count = packet_dw - 4u;
				const bool one   = ((control >> 16u) & 0x1u) != 0;
				uint64_t   value = 0;
				bool       known = false;
				if (one) {
#ifdef KYTY_PIPEND_MUTANT_WRITE_DATA_FIRST
					value = payload[3];
#else
					value = payload[3 + count - 1u];
#endif
					known = true;
				} else if (count <= 2u) {
					value = payload[3] | (count == 2u ? static_cast<uint64_t>(payload[4]) << 32u : 0u);
					known = true;
				}
				pending.NoteViews(w.address, w.size, false, known, value, seq, kModelBoth, src);
			}
			break;
		case Pm4::IT_DMA_DATA:
			if (packet_dw >= 7u) {
				const auto     w        = DecodeDmaData(payload);
				const uint32_t control  = payload[0];
				const uint32_t control2 = payload[5];
				const uint32_t dst_sel  = ((control >> 20u) & 0x3u) | ((control2 >> 25u) & 0x4u) | ((control2 >> 26u) & 0x8u);
				const uint16_t src      = SrcTag(opcode, dst_sel);
				// CommandProcessor::DmaData: no bytes, the register 0x3022c, nowhere (an L2 prefetch) and GDS (1) are no write of guest memory.
				const bool no_guest_write = w.size == 0 || static_cast<uint32_t>(w.address) == 0x3022cu || dst_sel == 2u || dst_sel == 1u;
#ifdef KYTY_PIPEND_MUTANT_DMA_REAL_DROPPED
				const bool dropped = true;
#else
				const bool dropped = no_guest_write;
#endif
				pending.NoteViews(w.address, w.size, false, false, 0, seq, precise && dropped ? kModelS479 : kModelBoth, src);
			}
			break;
		case Pm4::IT_COPY_DATA:
			if (packet_dw >= 6u) {
				const auto     w         = DecodeCopyData(payload);
				const uint32_t dst_field = ((payload[0] >> 8u) & 0xfu) << 1u;
				// CpOpCopyData: destination 6 is GDS (CopyDataDstToDma); every other accepted destination is guest memory.
				pending.NoteViews(w.address, w.size, false, false, 0, seq, precise && dst_field == 6u ? kModelS479 : kModelBoth, SrcTag(opcode, dst_field));
			}
			break;
		default: break;
	}
}

// The S479 rule of a wait answered from the FIFO: a known value that passes is the answer; a known value that fails or an unknown one drains; nothing pending reads
// guest memory (no drain).
inline bool WaitNeedsDrain(PendingWrites::Verdict verdict, uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (verdict) {
		case PendingWrites::Verdict::NoPending: return false;
		case PendingWrites::Verdict::Value: return !WaitCompare(value, ref, mask, func);
		case PendingWrites::Verdict::Unknown: return true;
	}
	return true;
}

// Shadow: the precise view would not have drained (NoPending: the wait reads `before`, the memory ahead of the drain; Value: the wait passes on `value`). True when
// the wait after the drain (`after`) decides the same.
inline bool ShadowWaitAgrees(PendingWrites::Verdict precise, uint64_t value, uint64_t before, uint64_t after, uint64_t ref, uint64_t mask, uint32_t func) {
#ifdef KYTY_PIPEND_MUTANT_SHADOW_BLIND
	(void)precise;
	(void)value;
	(void)before;
	(void)after;
	(void)ref;
	(void)mask;
	(void)func;
	return true;
#else
	const bool predicted = precise == PendingWrites::Verdict::Value ? WaitCompare(value, ref, mask, func) : WaitCompare(before, ref, mask, func);
	return predicted == WaitCompare(after, ref, mask, func);
#endif
}

// S604 DIAG histogram key: the drain key (0x63/0x64/0x9f = the table's opcode, 0x1f1 = wait), the S479 view's culprit and the precise view's (or `avoided`).
inline uint64_t NoDrainHistKey(uint32_t drain_key, Culprit s479, bool avoided, Culprit precise) {
	return (static_cast<uint64_t>(drain_key & 0x1ffu) << 40u) | (avoided ? uint64_t {1} << 36u : 0u) | (static_cast<uint64_t>(precise.reason) << 34u) |
	       (static_cast<uint64_t>(precise.src) << 18u) | (static_cast<uint64_t>(s479.reason) << 16u) | s479.src;
}

inline char ReasonLetter(Reason reason) {
	switch (reason) {
		case Reason::None: return '-';
		case Reason::Unknown: return 'U';
		case Reason::Partial: return 'P';
		case Reason::Fail: return 'F';
	}
	return '?';
}

} // namespace Libs::Graphics::CpPipe

#endif
