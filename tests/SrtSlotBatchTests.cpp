// S445 (W1): KYTY_SRT_SLOT_BATCH=0|1|shadow (srtSlotBatch.h). Plans with block conditions: the per-slot walk (mode 0) and the batch entries (1, shadow) must give
// the same flat buffer, the same captured reads in the same order and the same memo. Mutants (WILL_FAIL):
//   KYTY_SRTBATCH_MUTANT_SKIP_MEMO_CHECK (a shared table is read twice), _NO_FAIL_POS (the failing position is lost), _ORDER_SWAP (the reads change order),
//   _IGNORE_CLEAN (a clean slot is dropped from its run), _SHADOW_REAL_MEMO (shadow works on the real memo), _ALL_BLOCKS (the slots of unvisited blocks are filled).

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/srtSlotBatch.h"
#include "graphics/shader/recompiler/ir/passes/srtWalkPaths.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <set>
#include <span>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
namespace Batch = Libs::Graphics::ShaderRecompiler::IR::SrtSlotBatch;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SrtSlotBatchTests: failed: %s\n", text);
		std::exit(1); // not abort(): the WILL_FAIL mutants need a plain nonzero exit code
	}
}

std::array<uint32_t, 2> PointerWords(const void* pointer) {
	const auto address = reinterpret_cast<uint64_t>(pointer);
	return {static_cast<uint32_t>(address) | 3u, static_cast<uint32_t>(address >> 32u) | 0xabcd0000u};
}

// Guest memory of the plan: a root table (user data 4, 5) that points at tables A, B and a null table; plain DWORDs everywhere.
struct Guest {
	std::array<uint32_t, 64> root {};
	std::array<uint32_t, 16> a {}, b {};
	std::array<uint32_t, 3>  user_data {};
	Guest() {
		const auto pa = PointerWords(a.data()), pb = PointerWords(b.data());
		root[0] = pa[0], root[1] = pa[1], root[2] = pb[0], root[3] = pb[1]; // byte offsets 0, 4 / 8, 12
		// byte offsets 16, 20 stay zero: the null table
		for (uint32_t i = 0; i < 16; ++i) a[i] = 0xa0000000u | i, b[i] = 0xb0000000u | i;
		for (uint32_t i = 8; i < 64; ++i) root[i] = 0xc0000000u | i;
		root[40] = 1u; // the branch DWORD (byte offset 160)
		root[44] = 1u, root[48] = 0u; // the two incoming conditions of the Revisit shape's Phi (byte offsets 176, 192)
		const auto pr = PointerWords(root.data());
		user_data = {pr[0], pr[1], 0u};
	}
};

enum class Shape { Plain, NullMidRun, CleanMidRun, Revisit };

// entry block: 5 slots (a run), condition on root[40]; true block: 3 slots (NullMidRun: 4 with a null-table slot at position 1); false block: 2 slots.
// CleanMidRun: the third slot of the entry block is a clean read.
struct Built {
	std::unique_ptr<Program> program;
	uint32_t                 slots = 0;
};

// Shape::Revisit (T8): b0 -> b1 | b2 -> b3 (Phi of two conditions: b1 true, b2 false) -> b4 | b5. With root[40] = 1 the first pass keeps b2 and both ends (the
// Phi differs between its predecessors), the second pass drops b2 and the Phi becomes true: only b0, b1, b3, b4 are visited at the end.
// Slots: b0 {0: root+32, 1: A+0}, b2 {2: B+0, 3: B+8}, b4 {4: A+8, 5: B+4, 6: A+12}, b5 {7: root+40, 8: B+12}.
Built BuildRevisit() {
	Built built;
	built.program = std::make_unique<Program>();
	auto& program = *built.program;
	program.stage                 = Libs::Graphics::ShaderType::Compute;
	program.user_data_base        = 4;
	program.user_data_count       = 3;
	program.srt_plan_complete = program.resource_tracking_complete = true;
	program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
	std::array<Block*, 6> b {};
	for (auto& block: b) {
		auto owned = std::make_unique<Block>();
		block      = owned.get();
		program.blocks.push_back(block);
		block->id = static_cast<uint32_t>(program.blocks.size() - 1u);
		program.block_storage.push_back(std::move(owned));
	}
	const auto emit = [](Block& block, ValueOpcode opcode, std::initializer_list<Value> args) { return Value(&block.AppendNewInst(opcode, args)); };
	const auto branch = [](Block& block, Block& target, Value condition = {}, Block* other = nullptr) {
		block.terminator.kind        = other ? TerminatorKind::ConditionalBranch : TerminatorKind::Branch;
		block.terminator.true_block  = &target;
		block.terminator.false_block = other;
		block.condition              = condition;
		block.AddBranch(&target);
		if (other) block.AddBranch(other);
	};
	auto& entry = *b[0];
	const auto low  = emit(entry, ValueOpcode::GetUserData, {Value(ScalarReg(4))});
	const auto high = emit(entry, ValueOpcode::GetUserData, {Value(ScalarReg(5))});
	const auto root = emit(entry, ValueOpcode::GetAddressResource, {low, high});
	const auto load = [&](Value handle, uint32_t offset) {
		const auto value = emit(entry, ValueOpcode::LoadAddressU32, {handle, Value(offset), Value(0u), Value(true)});
		value.Instruction()->SetFlags(MemoryFlags {.index = 0});
		return value;
	};
	const auto table = [&](uint32_t offset) { return emit(entry, ValueOpcode::GetAddressResource, {load(root, offset), load(root, offset + 4u)}); };
	const auto ta = table(0u), tb = table(8u);
	const auto srt = emit(entry, ValueOpcode::GetSrtResource, {});
	std::vector<Value>                       values;
	std::vector<std::pair<Block*, uint32_t>> uses;
	const auto add = [&](Block& block, Value value) {
		uses.emplace_back(&block, static_cast<uint32_t>(values.size()));
		values.push_back(value);
	};
	add(*b[0], load(root, 32u));
	add(*b[0], load(ta, 0u));
	add(*b[2], load(tb, 0u));
	add(*b[2], load(tb, 8u));
	add(*b[4], load(ta, 8u));
	add(*b[4], load(tb, 4u));
	add(*b[4], load(ta, 12u));
	add(*b[5], load(root, 40u));
	add(*b[5], load(tb, 12u));
	built.slots = static_cast<uint32_t>(values.size());
	for (uint32_t slot = 0; slot < values.size(); ++slot) program.srt_reads.push_back({values[slot], built.slots - 1u - slot});
	for (const auto& [block, slot] : uses) emit(*block, ValueOpcode::ReadConst, {srt, Value(slot)});
	const auto incoming_true  = emit(*b[1], ValueOpcode::INotEqual32, {load(root, 176u), Value(0u)});
	const auto incoming_false = emit(*b[2], ValueOpcode::INotEqual32, {load(root, 192u), Value(0u)});
	auto&      phi            = b[3]->AppendNewInst(ValueOpcode::Phi);
	phi.SetFlags(Type::U1);
	phi.AddPhiOperand(b[1], incoming_true);
	phi.AddPhiOperand(b[2], incoming_false);
	branch(entry, *b[1], emit(entry, ValueOpcode::INotEqual32, {load(root, 160u), Value(0u)}), b[2]);
	branch(*b[1], *b[3]);
	branch(*b[2], *b[3]);
	branch(*b[3], *b[4], Value(&phi), b[5]);
	b[4]->terminator.kind = b[5]->terminator.kind = TerminatorKind::Return;
	return built;
}

Built Build(Shape shape) {
	if (shape == Shape::Revisit) return BuildRevisit();
	Built built;
	built.program = std::make_unique<Program>();
	auto& program = *built.program;
	program.stage                 = Libs::Graphics::ShaderType::Compute;
	program.user_data_base        = 4;
	program.user_data_count       = 3;
	program.srt_plan_complete = program.resource_tracking_complete = true;
	program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
	const auto add_block = [&]() -> Block& {
		auto block   = std::make_unique<Block>();
		auto* result = block.get();
		program.blocks.push_back(result);
		result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
		program.block_storage.push_back(std::move(block));
		return *result;
	};
	auto& entry = add_block();
	auto& taken = add_block();
	auto& other = add_block();
	const auto emit = [](Block& block, ValueOpcode opcode, std::initializer_list<Value> args) { return Value(&block.AppendNewInst(opcode, args)); };
	const auto low  = emit(entry, ValueOpcode::GetUserData, {Value(ScalarReg(4))});
	const auto high = emit(entry, ValueOpcode::GetUserData, {Value(ScalarReg(5))});
	const auto root = emit(entry, ValueOpcode::GetAddressResource, {low, high});
	const auto load = [&](Value handle, uint32_t offset) {
		const auto value = emit(entry, ValueOpcode::LoadAddressU32, {handle, Value(offset), Value(0u), Value(true)});
		value.Instruction()->SetFlags(MemoryFlags {.index = 0});
		return value;
	};
	const auto table = [&](uint32_t offset) { return emit(entry, ValueOpcode::GetAddressResource, {load(root, offset), load(root, offset + 4u)}); };
	const auto ta = table(0u), tb = table(8u), tn = table(16u);
	const auto srt = emit(entry, ValueOpcode::GetSrtResource, {});
	std::vector<Value>                  values;
	std::vector<std::pair<Block*, uint32_t>> uses; // block, slot
	const auto                          add = [&](Block& block, Value value) {
        const auto slot = static_cast<uint32_t>(values.size());
        values.push_back(value);
        uses.emplace_back(&block, slot);
	};
	// entry: root plain, A+0, A+4 (shares A's pointer words), B+0, B+8
	add(entry, load(root, 32u));
	add(entry, load(ta, 0u));
	add(entry, shape == Shape::CleanMidRun ? load(root, 36u) : load(ta, 4u));
	add(entry, load(tb, 0u));
	add(entry, load(tb, 8u));
	// taken: A+8, [null table +0], B+4, A+12
	add(taken, load(ta, 8u));
	if (shape == Shape::NullMidRun) add(taken, load(tn, 0u));
	add(taken, load(tb, 4u));
	add(taken, load(ta, 12u));
	// other: root+40 (the branch DWORD), B+12
	add(other, load(root, 40u));
	add(other, load(tb, 12u));
	built.slots = static_cast<uint32_t>(values.size());
	// Flat offsets run backwards so that a slot index is never taken for its flat offset.
	for (uint32_t slot = 0; slot < values.size(); ++slot) program.srt_reads.push_back({values[slot], static_cast<uint32_t>(values.size()) - 1u - slot});
	for (const auto& [block, slot] : uses) emit(*block, ValueOpcode::ReadConst, {srt, Value(slot)});
	if (shape == Shape::CleanMidRun) {
		// A written buffer keeps its descriptor reads on the strict reader: slot 2 is a clean read.
		DescriptorSource source;
		source.dwords      = {values[2], Value(0u), Value(4u), Value(0u)};
		source.dword_count = 4;
		program.descriptor_sources.push_back(source);
		program.info.buffers.push_back({.source = 0, .written = true});
	}
	entry.condition           = emit(entry, ValueOpcode::INotEqual32, {load(root, 160u), Value(0u)});
	entry.terminator.kind     = TerminatorKind::ConditionalBranch;
	entry.terminator.true_block  = &taken;
	entry.terminator.false_block = &other;
	entry.AddBranch(&taken);
	entry.AddBranch(&other);
	taken.terminator.kind = other.terminator.kind = TerminatorKind::Return;
	return built;
}

struct Walk {
	bool                                                ok = false;
	std::vector<uint32_t>                               flat;
	std::vector<std::pair<uint64_t, uint64_t>>          reads;
	std::vector<ResourcePlan::EvaluationContext::Entry> memo;
	std::array<uint64_t, Batch::kCellCount>             cells {};
	uint64_t                                            slot_jit = 0;
};

struct Strict {
	uint32_t reads = 0;
};

bool ReadStrict(void* data, uint64_t address, std::span<uint32_t> out) {
	++static_cast<Strict*>(data)->reads;
	std::memcpy(out.data(), reinterpret_cast<const void*>(address), out.size_bytes());
	return true;
}

// capture: the reads of the walker are recorded like MaterializeResourcesWalk does for a plan that captures specialization reads.
Walk Run(Shape shape, Batch::Mode mode, std::span<const uint32_t> user_data, bool capture) {
	Batch::Override().store(static_cast<int>(mode));
	(void)Batch::Drain();
	SrtWalkPaths::Active().store(true);
	(void)SrtWalkPaths::Drain();
	auto  built = Build(shape);
	auto  plan  = ExtractResourcePlan(*built.program);
	Walk  walk;
	Strict strict;
	SrtRuntime runtime {.user_data = user_data, .userdata = &strict, .read_specialization_memory = ReadStrict};
	SrtReadCapture observed_source(runtime, walk.reads);
	const auto     observed = capture ? observed_source.ObservedRuntime() : runtime;
	{
		SrtWalker clean(plan, CleanRuntime(observed));
		SrtWalker walker(plan, observed, &clean);
		walk.ok = walker.RefreshFlatBuffer(walk.flat);
	}
	for (const auto& context: plan.evaluation_contexts) {
		walk.memo.insert(walk.memo.end(), context.values.begin(), context.values.end());
		walk.memo.push_back({context.generation, context.generation});
	}
	walk.cells    = Batch::Drain();
	walk.slot_jit = SrtWalkPaths::Drain()[static_cast<size_t>(SrtWalkPaths::Path::SlotJit)];
	return walk;
}

uint64_t Cell(const Walk& walk, Batch::Cell cell) {
	return walk.cells[static_cast<size_t>(cell)];
}

bool SameMemo(const Walk& x, const Walk& y) {
	if (x.memo.size() != y.memo.size()) return false;
	for (size_t i = 0; i < x.memo.size(); ++i)
		if (x.memo[i].value != y.memo[i].value || x.memo[i].generation != y.memo[i].generation) return false;
	return true;
}

// Two walks must agree on everything the batch may not change.
void CheckSame(const Walk& want, const Walk& got, const char* what) {
	char text[160];
	std::snprintf(text, sizeof(text), "%s: result", what);
	Check(want.ok == got.ok, text);
	std::snprintf(text, sizeof(text), "%s: flat buffer", what);
	Check(want.flat == got.flat, text);
	std::snprintf(text, sizeof(text), "%s: captured reads (values and order)", what);
	Check(want.reads == got.reads, text);
	std::snprintf(text, sizeof(text), "%s: memo (values and generation)", what);
	Check(SameMemo(want, got), text);
}

// T1 / T2 / T6 / T7 / T8: both branches of the block condition, with and without capture.
void TestBatchEqualsSlots() {
	Guest guest;
	for (const bool capture: {true, false}) {
		for (const uint32_t branch: {1u, 0u}) {
			guest.root[40] = branch;
			const auto off    = Run(Shape::Plain, Batch::Mode::Off, guest.user_data, capture);
			const auto on     = Run(Shape::Plain, Batch::Mode::On, guest.user_data, capture);
			const auto shadow = Run(Shape::Plain, Batch::Mode::Shadow, guest.user_data, capture);
			Check(off.ok && off.flat.size() == 10u, "mode 0 walk");
			// Slot s has flat offset 9 - s. Visited: entry (0..4) and the taken block (5..7) or the other block (8, 9).
			std::vector<uint32_t> expect(10, 0u);
			const auto            set = [&](uint32_t slot, uint32_t value) { expect[9u - slot] = value; };
			set(0, 0xc0000000u | 8u);
			set(1, 0xa0000000u | 0u);
			set(2, 0xa0000000u | 1u);
			set(3, 0xb0000000u | 0u);
			set(4, 0xb0000000u | 2u);
			if (branch) {
				set(5, 0xa0000000u | 2u);
				set(6, 0xb0000000u | 1u);
				set(7, 0xa0000000u | 3u);
			} else {
				set(8, 0xc0000000u | 10u);
				set(9, 0xb0000000u | 3u);
			}
			if (off.flat != expect) {
				for (size_t i = 0; i < expect.size(); ++i) std::fprintf(stderr, "flat[%zu] got %08x want %08x\n", i, i < off.flat.size() ? off.flat[i] : 0u, expect[i]);
			}
			Check(off.flat == expect, "mode 0 flat buffer is not the expected one (unvisited block filled or value lost)");
			CheckSame(off, on, "mode 1");
			CheckSame(off, shadow, "mode shadow");
			const uint64_t runs   = 2u;
			const uint64_t slots  = branch ? 8u : 7u;
			Check(Cell(off, Batch::Cell::Runs) == 0u && Cell(off, Batch::Cell::Slots) == 0u, "mode 0 entered a batch");
			Check(Cell(on, Batch::Cell::Runs) == runs && Cell(on, Batch::Cell::Slots) == slots, "mode 1 run / slot counts");
			Check(Cell(shadow, Batch::Cell::Runs) == runs && Cell(shadow, Batch::Cell::Slots) == slots, "shadow run / slot counts");
			Check(Cell(on, Batch::Cell::Fail) == 0u && Cell(on, Batch::Cell::Single) == 0u && Cell(on, Batch::Cell::Mismatch) == 0u, "mode 1 failures");
			Check(Cell(shadow, Batch::Cell::Mismatch) == 0u && Cell(shadow, Batch::Cell::Fail) == 0u, "shadow found a difference");
			Check(on.slot_jit == slots && off.slot_jit == slots, "slots served by the JIT");
			if (capture) {
				// T6: A's pointer words and B's are read once (the plan reads them for four slots), every address once.
				std::set<uint64_t> distinct;
				for (const auto& read: off.reads) distinct.insert(read.first);
				Check(distinct.size() == off.reads.size() && on.reads.size() == off.reads.size(), "a table pointer was read more than once");
			}
		}
	}
}

// T3: a null table in the middle of a run: the batch stops there, the slot goes to the interpreter (zero), the rest of the run is read.
void TestNullTableMidRun() {
	Guest guest;
	guest.root[40] = 1u;
	const auto off = Run(Shape::NullMidRun, Batch::Mode::Off, guest.user_data, true);
	const auto on  = Run(Shape::NullMidRun, Batch::Mode::On, guest.user_data, true);
	const auto shadow = Run(Shape::NullMidRun, Batch::Mode::Shadow, guest.user_data, true);
	Check(off.ok && on.ok && shadow.ok, "a null table failed the walk");
	CheckSame(off, on, "null table, mode 1");
	CheckSame(off, shadow, "null table, shadow");
	// entry run: 5 slots; taken run: A+8 (position 1) fills, the null-table slot (position 2) fails: 1 slot filled
	Check(Cell(on, Batch::Cell::Fail) == 1u, "the failing run was not counted");
	Check(Cell(on, Batch::Cell::Slots) == 5u + 1u, "the batch did not stop at the failing slot (the position of the failure was lost)");
	Check(Cell(shadow, Batch::Cell::Mismatch) == 0u && Cell(shadow, Batch::Cell::Fail) == 1u, "shadow on a failing run");
	Check(on.flat.size() == 11u && on.flat[11u - 1u - 6u] == 0u, "the null-table slot is not zero");
}

// T4: user data shorter than the pointer word: the walk fails as before.
void TestShortUserData() {
	Guest guest;
	for (const size_t words: {size_t {0}, size_t {1}}) {
		const auto user = std::span<const uint32_t>(guest.user_data).first(words);
		const auto off  = Run(Shape::Plain, Batch::Mode::Off, user, true);
		const auto on   = Run(Shape::Plain, Batch::Mode::On, user, true);
		const auto shadow = Run(Shape::Plain, Batch::Mode::Shadow, user, true);
		Check(!off.ok, "mode 0 accepted truncated user data");
		CheckSame(off, on, "short user data, mode 1");
		CheckSame(off, shadow, "short user data, shadow");
	}
}

// T5: a clean slot ends its run; the two runs around it are batched, the clean read goes through the strict reader in its place.
void TestCleanSplitsRun() {
	Guest guest;
	guest.root[40] = 0u;
	const auto off = Run(Shape::CleanMidRun, Batch::Mode::Off, guest.user_data, true);
	const auto on  = Run(Shape::CleanMidRun, Batch::Mode::On, guest.user_data, true);
	const auto shadow = Run(Shape::CleanMidRun, Batch::Mode::Shadow, guest.user_data, true);
	Check(off.ok && on.ok, "clean split walk");
	CheckSame(off, on, "clean slot, mode 1");
	CheckSame(off, shadow, "clean slot, shadow");
	Check(on.flat[9u - 2u] == (0xc0000000u | 9u), "the clean slot lost its value");
	// entry: [0,1] run, clean slot 2, [3,4] run; other block: [8, 9] run
	Check(Cell(on, Batch::Cell::Runs) == 3u && Cell(on, Batch::Cell::Slots) == 6u && Cell(on, Batch::Cell::Single) == 0u, "the clean slot did not split the run");
}

// T8: the visited set shrinks in the second pass of the fixed point (a Phi of block conditions): only the slots of the final visited blocks are filled, in the
// order of block.srt_reads.
void TestRevisit() {
	Guest guest;
	guest.root[40] = 1u;
	{
		auto built = Build(Shape::Revisit);
		Check(!ExtractResourcePlan(*built.program).control_phi_predecessors.empty(), "revisit: the plan has no control Phi");
	}
	const auto off    = Run(Shape::Revisit, Batch::Mode::Off, guest.user_data, true);
	const auto on     = Run(Shape::Revisit, Batch::Mode::On, guest.user_data, true);
	const auto shadow = Run(Shape::Revisit, Batch::Mode::Shadow, guest.user_data, true);
	Check(off.ok && off.flat.size() == 9u, "revisit plan walk");
	std::vector<uint32_t> expect(9, 0u);
	expect[8u - 0u] = 0xc0000008u; // b0
	expect[8u - 1u] = 0xa0000000u;
	expect[8u - 4u] = 0xa0000002u; // b4
	expect[8u - 5u] = 0xb0000001u;
	expect[8u - 6u] = 0xa0000003u;
	Check(off.flat == expect, "revisit: mode 0 filled a block that the fixed point dropped (or lost a value)");
	CheckSame(off, on, "revisit, mode 1");
	CheckSame(off, shadow, "revisit, shadow");
	Check(Cell(on, Batch::Cell::Runs) == 2u && Cell(on, Batch::Cell::Slots) == 5u, "revisit: runs of the final visited blocks only");
	Check(Cell(shadow, Batch::Cell::Mismatch) == 0u && Cell(shadow, Batch::Cell::Runs) == 2u, "revisit: shadow");
}

} // namespace

int main() {
	TestBatchEqualsSlots();
	TestNullTableMidRun();
	TestShortUserData();
	TestCleanSplitsRun();
	TestRevisit();
	Batch::Override().store(-1);
	std::puts("SrtSlotBatchTests: all cases passed");
	return 0;
}

// The IR library is not linked into the walker tests: compile what the plan needs into this binary.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
