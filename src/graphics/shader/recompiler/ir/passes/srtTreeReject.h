#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTTREEREJECT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTTREEREJECT_H_

// S441 DIAG (KYTY_DIAG_JIT_TREE_REJECT=1, no change of behaviour): why a plan did not get the whole-tree JIT of the SRT walk (CompileTreeWalker) and
// how many walks / slot evaluations that costs per flip. CompileTreeWalker stores the first reason in ResourcePlan::tree_reject (a byte, always); the
// walker notes (plan hash, reason, detail) per walk into the census while it is active; FrameTiming dumps the top rows every 100 flips as
// `[jit-tree-reject]` lines. Mutant (tests): KYTY_TREEREJECT_MUTANT_NO_RECORD (CompileTreeWalker does not store the reason),
// KYTY_TREEREJECT_MUTANT_ALL_ONE_ROW (the census merges rows of different reasons).

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR::SrtTreeReject {

enum class Reason : uint8_t {
	None,             // the tree JIT was compiled
	NotTracked,       // resource tracking did not complete
	PlanIncomplete,   // the SRT plan is not complete
	CaptureSpecReads, // the plan captures specialization reads
	ControlFlow,      // block conditions (control_flow is not empty): the whole-plan walker is for straight-line plans
	NoReads,          // no SRT reads
	SlotOpcode,       // a slot is not a LoadAddressU32 raw read (detail: its opcode) or it has fewer than 2 arguments
	SlotNotRaw,       // LoadAddressU32 that IsRawRead refuses (memory kind, index)
	SlotFlatOob,      // flat offset outside the table
	OffsetNotImm,     // the slot's byte offset is not an immediate U32
	OffsetRange,      // offset out of int32 range, or two slots share an instruction
	HandleShape,      // the address handle is not GetAddressResource(lo, hi)
	HandleWords,      // a handle word is not an instruction (immediate)
	UserDataRegs,     // the pointer words are user data registers below the base, or not adjacent
	PointerOffsets,   // the pointer words are loads that are not adjacent / not the same table
	DuplicateOffset,  // two slots read the same offset of one table
	PointerCycle,     // a pointer table is reached twice (or from itself)
	PointerUnreached, // a pointer table nobody reaches from user data
	CaptureRuntime,   // walk-time: the tree JIT exists but the runtime captures reads (read_memory != nullptr)
	NojitEnv,         // walk-time: the tree JIT exists but KYTY_SRT_READER=nojit
	TreeReturnedFalse, // walk-time: the tree JIT ran and returned false (table below guest memory, short user data)
	SlotInterp,       // walk-time, per slot: the slot was evaluated by the interpreter (detail: why)
	Count,
};

inline const char* Name(Reason reason) {
	static constexpr const char* names[] = {"none",        "not_tracked",   "plan_incomplete", "capture_spec_reads", "control_flow",        "no_reads",
	                                        "slot_opcode", "slot_not_raw",  "slot_flat_oob",   "offset_not_imm",     "offset_range",        "handle_shape",
	                                        "handle_words", "user_data_regs", "pointer_offsets", "duplicate_offset",   "pointer_cycle",       "pointer_unreached",
	                                        "capture_runtime", "nojit_env", "tree_returned_false", "slot_interp"};
	static_assert(sizeof(names) / sizeof(names[0]) == static_cast<size_t>(Reason::Count));
	return static_cast<size_t>(reason) < static_cast<size_t>(Reason::Count) ? names[static_cast<size_t>(reason)] : "?";
}

inline std::atomic<bool>& Active() {
	static std::atomic<bool> on {[] {
		const char* v = std::getenv("KYTY_DIAG_JIT_TREE_REJECT");
		return v != nullptr && v[0] != 0 && v[0] != '0';
	}()};
	return on;
}

struct Row {
	uint64_t    plan   = 0;
	Reason      reason = Reason::None;
	std::string detail;
	uint64_t    n = 0; // walks (slot evaluations for SlotInterp)
};

class Census {
public:
	void Note(uint64_t plan, Reason reason, const std::string& detail, uint64_t n = 1) {
		std::scoped_lock lock(m_mutex);
#ifdef KYTY_TREEREJECT_MUTANT_ALL_ONE_ROW
		reason = Reason::None;
#endif
		m_rows[std::make_tuple(plan, static_cast<uint8_t>(reason), detail)] += n;
	}
	// All rows, largest first (ties: plan, reason); empties the census.
	std::vector<Row> Take() {
		std::scoped_lock lock(m_mutex);
		std::vector<Row> out;
		out.reserve(m_rows.size());
		for (const auto& [key, n]: m_rows) out.push_back({std::get<0>(key), static_cast<Reason>(std::get<1>(key)), std::get<2>(key), n});
		m_rows.clear();
		std::sort(out.begin(), out.end(), [](const Row& a, const Row& b) {
			if (a.n != b.n) return a.n > b.n;
			if (a.plan != b.plan) return a.plan < b.plan;
			return a.reason < b.reason;
		});
		return out;
	}
	// Sum per reason of Take()'s rows.
	static std::map<Reason, uint64_t> ByReason(const std::vector<Row>& rows) {
		std::map<Reason, uint64_t> sums;
		for (const auto& row: rows) sums[row.reason] += row.n;
		return sums;
	}

private:
	std::mutex                                                         m_mutex;
	std::map<std::tuple<uint64_t, uint8_t, std::string>, uint64_t>     m_rows;
};

inline Census& Global() {
	static Census census;
	return census;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::SrtTreeReject

#endif
