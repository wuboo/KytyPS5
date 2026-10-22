#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSCOUT_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSCOUT_H

// S454 (B2): KYTY_CP_SCOUT=0|1|shadow (default 0). A lookahead walk of a PM4 buffer records the SH
// register words at each DRAW and the guest dwords those user-data words point at. CP later
// Matches() those dwords before using a precomputed GetGraphicsPrograms result. shadow: CP still
// computes the programs itself and counts mismatches. Mutant (WILL_FAIL): KYTY_CPSCOUT_MUTANT_SKIP_VALIDATE.

#include "graphics/guest_gpu/pm4.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

namespace Libs::Graphics::CpScout {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode Parse(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline Mode& TestOverride() {
	static Mode mode = Mode::Off;
	static bool set  = false;
	if (!set) {
		mode = Parse(std::getenv("KYTY_CP_SCOUT"));
		set  = true;
	}
	return mode;
}

inline Mode Current() { return TestOverride(); }

inline constexpr uint32_t kShNum = Pm4::SH_NUM;

struct GuestRead {
	uint64_t address = 0;
	uint32_t word    = 0;
};

struct DrawSnap {
	std::array<uint32_t, kShNum> sh {};
	std::vector<GuestRead>       reads;
	bool                         stopped_early = false;
};

struct WalkResult {
	std::vector<DrawSnap> draws;
	bool                  stopped = false;
};

using ReadGuest = bool (*)(uint64_t address, uint32_t& word);

inline uint64_t PointerFromUserData(uint32_t low, uint32_t high) {
	return (static_cast<uint64_t>(high) << 32u | low) & 0x0000fffffffffffcull;
}

inline void NoteUserDataReads(DrawSnap& snap, ReadGuest read) {
	if (read == nullptr) return;
	const auto note_range = [&](uint32_t first, uint32_t last) {
		for (uint32_t slot = first; slot + 1u <= last; slot += 2u) {
			const uint32_t low  = snap.sh[slot];
			const uint32_t high = snap.sh[slot + 1u];
			if (low == 0 && high == 0) continue;
			const uint64_t address = PointerFromUserData(low, high);
			if (address < 0x40000ull) continue;
			uint32_t word = 0;
			if (!read(address, word)) continue;
			snap.reads.push_back({address, word});
		}
	};
	note_range(Pm4::SPI_SHADER_USER_DATA_PS_0, Pm4::SPI_SHADER_USER_DATA_PS_31);
	note_range(Pm4::SPI_SHADER_USER_DATA_GS_0, Pm4::SPI_SHADER_USER_DATA_GS_31);
}

inline bool IsDrawOpcode(uint32_t op) {
	return op == Pm4::IT_DRAW_INDEX_2 || op == Pm4::IT_DRAW_INDEX_AUTO ||
	       op == Pm4::IT_DRAW_INDEX_OFFSET_2 || op == Pm4::IT_DRAW_INDIRECT ||
	       op == Pm4::IT_DRAW_INDEX_INDIRECT;
}

inline bool IsStopOpcode(uint32_t op) {
	return op == Pm4::IT_WAIT_REG_MEM || op == Pm4::IT_WAIT_REG_MEM_64 || op == Pm4::IT_COND_EXEC ||
	       op == Pm4::IT_DRAW_INDIRECT_MULTI || op == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI ||
	       op == Pm4::IT_WRITE_CONST_RAM || op == Pm4::IT_DUMP_CONST_RAM ||
	       op == Pm4::IT_WAIT_ON_CE_COUNTER || op == Pm4::IT_WAIT_ON_DE_COUNTER_DIFF;
}

inline bool IsIgnoreOpcode(uint32_t op) {
	return op == Pm4::IT_NOP || op == Pm4::IT_NUM_INSTANCES || op == Pm4::IT_INDEX_TYPE ||
	       op == Pm4::IT_INDEX_BASE || op == Pm4::IT_INDEX_BUFFER_SIZE || op == Pm4::IT_PFP_SYNC_ME ||
	       op == Pm4::IT_EVENT_WRITE || op == Pm4::IT_EVENT_WRITE_EOP || op == Pm4::IT_EVENT_WRITE_EOS ||
	       op == Pm4::IT_ACQUIRE_MEM || op == Pm4::IT_RELEASE_MEM || op == Pm4::IT_CONTEXT_CONTROL ||
	       op == Pm4::IT_SURFACE_SYNC || op == Pm4::IT_SET_PREDICATION;
}

inline WalkResult Walk(std::span<const uint32_t> commands, ReadGuest read = nullptr) {
	WalkResult result;
	if (commands.empty()) return result;

	struct Cursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset = 0;
	};
	std::array<uint32_t, kShNum> sh {};
	std::vector<Cursor>          stack;
	stack.push_back({commands, 0});

	auto apply_sh = [&](uint32_t offset, uint32_t value) {
		if (offset < kShNum) sh[offset] = value;
	};

	while (!stack.empty()) {
		auto& cursor = stack.back();
		if (cursor.offset >= cursor.commands.size()) {
			stack.pop_back();
			continue;
		}
		const auto* packet       = cursor.commands.data() + cursor.offset;
		const auto  remaining_dw = static_cast<uint32_t>(cursor.commands.size() - cursor.offset);
		const auto  header       = packet[0];
		const auto  opcode       = (header >> 8u) & 0xffu;
		if (header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u) {
			cursor.offset++;
			continue;
		}
		if (remaining_dw < 2) {
			result.stopped = true;
			break;
		}
		const auto packet_dw = Pm4::PacketSizeDw(header);
		if (packet_dw == 0 || packet_dw > remaining_dw) {
			result.stopped = true;
			break;
		}

		if (IsStopOpcode(opcode)) {
			result.stopped = true;
			break;
		}
		if (opcode == Pm4::IT_SET_SH_REG) {
			const uint32_t cmd_offset = packet[1];
			if (cmd_offset != Pm4::SH_NOP) {
				const uint32_t n = packet_dw - 2u;
				for (uint32_t i = 0; i < n; i++) apply_sh(cmd_offset + i, packet[2 + i]);
			}
		} else if (opcode == Pm4::IT_SET_SH_REG_INDIRECT && packet_dw >= 5u) {
			auto*       indirect = reinterpret_cast<uint32_t*>(packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
			const auto  n        = packet[4] & 0x3fffu;
			if (indirect == nullptr && n != 0) {
				result.stopped = true;
				break;
			}
			for (uint32_t i = 0; i < n; i++, indirect += 2) {
				if (indirect[0] == 0xffffffffu || indirect[0] == Pm4::SH_NOP) continue;
				apply_sh(indirect[0], indirect[1]);
			}
		} else if (opcode == Pm4::IT_INDIRECT_BUFFER && packet_dw >= 4u) {
			const uint32_t control = packet[3];
			const auto     num_dw  = control & 0xfffffu;
			auto*          ib = reinterpret_cast<const uint32_t*>(packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
			cursor.offset += packet_dw;
			if (num_dw != 0 && ib != nullptr) {
				const bool chain = (control & (1u << 20u)) != 0;
				if (chain) {
					cursor = {{ib, num_dw}, 0};
				} else {
					stack.push_back({{ib, num_dw}, 0});
				}
			}
			continue;
		} else if (IsDrawOpcode(opcode)) {
			DrawSnap snap;
			snap.sh = sh;
			NoteUserDataReads(snap, read);
			result.draws.push_back(std::move(snap));
		} else if (!IsIgnoreOpcode(opcode) && opcode != Pm4::IT_SET_CONTEXT_REG &&
		           opcode != Pm4::IT_SET_UCONFIG_REG && opcode != Pm4::IT_SET_CONTEXT_REG_INDIRECT &&
		           opcode != Pm4::IT_SET_UCONFIG_REG_INDIRECT && opcode != Pm4::IT_SET_CONFIG_REG) {
			result.stopped = true;
			break;
		}
		cursor.offset += packet_dw;
	}
	return result;
}

// S478: the worker applies the SET_* packets of its lookahead buffer through the CP's own handlers, with the register files redirected to its copies. A handler that
// writes a member of the CommandProcessor itself (index type / base / buffer size, instance count, indirect bases) must not do it from the worker: the CP would draw with
// the values of a draw that is still ahead. KYTY_SCOUT_RACE_FIX=0 brings the old behaviour back. Mutant (WILL_FAIL): KYTY_CPSCOUT_MUTANT_RACE_WRITES.
template <typename T>
inline bool WriteCpMember(T& member, T value, bool on_scout_worker, bool race_fix) {
#ifndef KYTY_CPSCOUT_MUTANT_RACE_WRITES
	if (on_scout_worker && race_fix) return false;
#else
	(void)on_scout_worker;
	(void)race_fix;
#endif
	member = value;
	return true;
}

inline bool Matches(const DrawSnap& snap, ReadGuest read) {
#ifdef KYTY_CPSCOUT_MUTANT_SKIP_VALIDATE
	(void)snap;
	(void)read;
	return true;
#else
	if (read == nullptr) return snap.reads.empty();
	for (const auto& guest: snap.reads) {
		uint32_t word = 0;
		if (!read(guest.address, word) || word != guest.word) return false;
	}
	return true;
#endif
}

} // namespace Libs::Graphics::CpScout

#endif
