// S406: purpose tags of the spec reads and the stale / superset shadow runs (specShadow.h) on a plan with a conditional block.
// Mutants (each must fail): KYTY_SPECSHADOW_MUTANT_TAG_IGNORED (the purpose scope sets nothing), SHADOW_FRESH (the stale shadow is
// taken from the fresh run), SUPERSET_READS_STALE (the superset reader does not fail a dirty condition read).

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/specShadow.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
namespace SS = Libs::Graphics::ShaderRecompiler::IR::SpecShadow;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SpecShadowTests: failed: %s\n", text);
		std::exit(1);
	}
}

Block& AddValueBlock(Program& program) {
	auto  block  = std::make_unique<Block>();
	auto* result = block.get();
	program.blocks.push_back(result);
	result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
	program.block_storage.push_back(std::move(block));
	return *result;
}

// "Guest memory": the root table holds the branch word at index 3 (offset 12). Stale = what the CPU sees before the drain.
constexpr uint32_t kStaleBranch = 0;
constexpr uint32_t kFreshBranch = 1;
std::array<uint32_t, 6> g_root {};
std::array<uint32_t, 4> g_child {0u, 0x11111111u, 0x22222222u, 0x33333333u};
uint32_t                g_drains = 0;

bool Covers(uint64_t address, uint64_t size) {
	const auto branch = reinterpret_cast<uint64_t>(&g_root[3]);
	return address <= branch && branch < address + size;
}
bool TestBacking(uint64_t address, void* data, uint64_t size) {
	std::memcpy(data, reinterpret_cast<const void*>(address), size);
	return true;
}
bool TestDirty(uint64_t address, uint64_t size) {
	return Covers(address, size);
}
// The real reader: a read of GPU-dirty bytes drains (the GPU writes the fresh value), then reads.
bool RealRead(void*, uint64_t address, std::span<uint32_t> out) {
	if (Covers(address, out.size_bytes()) && g_root[3] != kFreshBranch) {
		g_drains++;
		g_root[3] = kFreshBranch;
	}
	std::memcpy(out.data(), reinterpret_cast<const void*>(address), out.size_bytes());
	return true;
}

std::array<uint32_t, 3> g_user_data {};

ResourcePlan MakePlan() {
	using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
	const auto pointer_words = [](const void* pointer) {
		const auto address = reinterpret_cast<uint64_t>(pointer);
		return std::array<uint32_t, 2> {static_cast<uint32_t>(address) | 3u, static_cast<uint32_t>(address >> 32u) | 0xabcd0000u};
	};
	const auto child_words = pointer_words(g_child.data());
	g_root                 = {child_words[0], child_words[1], 0x44444444u, kStaleBranch, 0x55555555u, 0u};
	const auto root_words  = pointer_words(g_root.data());
	g_user_data            = {root_words[0], root_words[1], 0u};

	Program program;
	program.stage             = Libs::Graphics::ShaderType::Compute;
	program.user_data_base    = 4;
	program.user_data_count   = 3;
	program.srt_plan_complete = program.resource_tracking_complete = true;
	program.memory_info.push_back({.kind = ResourceKind::ScalarAddress});
	auto&      entry = AddValueBlock(program);
	const auto emit  = [](Block& block, ValueOpcode opcode, std::initializer_list<Value> args) {
        return Value(&block.AppendNewInst(opcode, args));
	};
	const auto low         = emit(entry, ValueOpcode::GetUserData, {Value(ScalarReg(4))});
	const auto high        = emit(entry, ValueOpcode::GetUserData, {Value(ScalarReg(5))});
	const auto root_handle = emit(entry, ValueOpcode::GetAddressResource, {low, high});
	const auto load        = [&](Value handle, uint32_t offset) {
        const auto value = emit(entry, ValueOpcode::LoadAddressU32, {handle, Value(offset), Value(0u), Value(true)});
        value.Instruction()->SetFlags(MemoryFlags {.index = 0});
        return value;
	};
	const auto pointer_low  = load(root_handle, 0u);
	const auto pointer_high = load(root_handle, 4u);
	const auto child_handle = emit(entry, ValueOpcode::GetAddressResource, {pointer_low, pointer_high});
	const auto srt          = emit(entry, ValueOpcode::GetSrtResource, {});
	// Slots: 0 plain, 1 pointer word, 2 clean, 3 nested (true edge), 4 nested (false edge).
	const std::array slot_values {load(root_handle, 8u), pointer_high, load(root_handle, 16u), load(child_handle, 4u),
	                              load(child_handle, 8u)};
	for (uint32_t slot = 0; slot < slot_values.size(); ++slot) program.srt_reads.push_back({slot_values[slot], 4u - slot});
	for (uint32_t slot = 0; slot < 3; ++slot) emit(entry, ValueOpcode::ReadConst, {srt, Value(slot)});
	DescriptorSource source;
	source.dwords      = {slot_values[2], Value(0u), Value(4u), Value(0u)};
	source.dword_count = 4;
	program.descriptor_sources.push_back(source);
	program.info.buffers.push_back({.source = 0, .written = true});
	const auto condition = emit(entry, ValueOpcode::INotEqual32, {load(root_handle, 12u), Value(0u)});
	for (uint32_t slot = 3; slot < 5; ++slot) emit(AddValueBlock(program), ValueOpcode::ReadConst, {srt, Value(slot)});
	entry.condition                     = condition;
	entry.terminator.kind               = TerminatorKind::ConditionalBranch;
	entry.terminator.true_block         = program.blocks[1];
	entry.terminator.false_block        = program.blocks[2];
	program.blocks[1]->terminator.kind  = TerminatorKind::Return;
	program.blocks[2]->terminator.kind  = TerminatorKind::Return;
	return ExtractResourcePlan(program);
}

SrtRuntime Runtime(SrtMemoryReader reader, void* userdata) {
	return SrtRuntime {.user_data = g_user_data, .userdata = userdata, .read_specialization_memory = reader};
}

void TestPurposeScopes() {
	Check(SS::CurrentPurpose() == SS::Purpose::Other, "no scope: other");
	{
		SS::PurposeScope outer(SS::Purpose::Desc);
		Check(SS::CurrentPurpose() == SS::Purpose::Desc, "outer scope");
		{
			SS::PurposeScope inner(SS::Purpose::Cond);
			Check(SS::CurrentPurpose() == SS::Purpose::Cond, "innermost scope wins");
		}
		Check(SS::CurrentPurpose() == SS::Purpose::Desc, "leaving the inner scope restores the outer tag");
	}
	Check(SS::CurrentPurpose() == SS::Purpose::Other, "outside every scope: other");
}

void TestShadowDecision() {
	auto plan = MakePlan();
	Check(plan.capture_specialization_reads && plan.control_flow.size() == 3, "plan has the conditional blocks");

	// Baseline: the real run alone (the fresh value is seen after the drain).
	g_root[3] = kStaleBranch;
	g_drains  = 0;
	SS::Outcome alone;
	SS::Materialize(plan, Runtime(RealRead, nullptr), alone);
	Check(alone.ok && g_drains == 1, "baseline real run drains once");
	Check(alone.snapshot.flattened_srt[0] == 0u && alone.snapshot.flattened_srt[1] == g_child[1], "baseline took the true edge");

	// Production order: stale shadow, superset shadow, then the real run.
	g_root[3] = kStaleBranch;
	g_drains  = 0;
	SS::ShadowReader stale_reader {.mode = SS::ReadMode::Stale, .backing = TestBacking, .dirty = TestDirty};
	SS::ShadowReader super_reader {.mode = SS::ReadMode::Superset, .backing = TestBacking, .dirty = TestDirty};
	SS::Outcome      stale, superset, fresh;
	SS::Materialize(plan, Runtime(SS::ShadowRead, &stale_reader), stale);
	SS::Materialize(plan, Runtime(SS::ShadowRead, &super_reader), superset);
	Check(g_drains == 0, "the shadow runs never drain");
	Check(g_root[3] == kStaleBranch, "the shadow runs left the memory alone");
	Check(stale_reader.dirty_reads[static_cast<size_t>(SS::Purpose::Cond)] >= 1, "the dirty read was a condition read");
	Check(stale_reader.DirtyTotal() == stale_reader.dirty_reads[static_cast<size_t>(SS::Purpose::Cond)],
	      "nothing but the condition read the dirty word");
	Check(stale.ok && stale.snapshot.flattened_srt[0] == g_child[2] && stale.snapshot.flattened_srt[1] == 0u,
	      "stale shadow took the false edge");
	Check(superset.ok && superset.snapshot.flattened_srt[0] == g_child[2] && superset.snapshot.flattened_srt[1] == g_child[1],
	      "superset shadow followed both edges");

	SS::Materialize(plan, Runtime(RealRead, nullptr), fresh);
	Check(fresh.ok && g_drains == 1, "the real run drained once");
	Check(fresh.snapshot.flattened_srt == alone.snapshot.flattened_srt && fresh.specialization == alone.specialization &&
	          fresh.snapshot.buffers == alone.snapshot.buffers && fresh.active == alone.active,
	      "the shadows left nothing behind in the plan: the real run equals the baseline");

	const auto verdict = SS::Compare(stale, superset, fresh);
	Check(!verdict.StaleEqual(), "stale and fresh differ in the condition word (ne = 1)");
	Check(verdict.stale_part == SS::Part::Flat, "... in the flattened SRT");
	Check(verdict.super_ok, "the superset materialization succeeded");
	Check(SS::FirstDifference(fresh, fresh) == SS::Part::None, "a run equals itself");
	const auto diff = SS::FirstWordDiff(verdict.stale_part, stale, fresh);
	Check(diff.valid && diff.index == 0 && diff.stale == g_child[2] && diff.fresh == 0u, "first differing word reported");
}

void TestStaleEqualWhenBranchUnchanged() {
	auto plan = MakePlan();
	// The GPU writes the same value: stale == fresh.
	g_root[3] = kFreshBranch;
	g_drains  = 0;
	SS::ShadowReader stale_reader {.mode = SS::ReadMode::Stale, .backing = TestBacking, .dirty = TestDirty};
	SS::ShadowReader super_reader {.mode = SS::ReadMode::Superset, .backing = TestBacking, .dirty = TestDirty};
	SS::Outcome      stale, superset, fresh;
	SS::Materialize(plan, Runtime(SS::ShadowRead, &stale_reader), stale);
	SS::Materialize(plan, Runtime(SS::ShadowRead, &super_reader), superset);
	SS::Materialize(plan, Runtime(RealRead, nullptr), fresh);
	const auto verdict = SS::Compare(stale, superset, fresh);
	Check(verdict.StaleEqual() && verdict.super_ok, "same bytes: eq = 1");
}

} // namespace

namespace Common {

int DbgExitHandler(const char*, int, std::string_view) {
	std::abort();
}
int DbgExitHandler(const char*, int, fmt::text_style, std::string_view) {
	std::abort();
}
int DbgExitIfHandler(const char*, const char*, int) {
	return 1;
}
void DbgExit(int) {
	std::abort();
}

} // namespace Common

int main() {
	TestPurposeScopes();
	TestShadowDecision();
	TestStaleEqualWhenBranchUnchanged();
	std::fprintf(stderr, "SpecShadowTests: ok\n");
	return 0;
}

// Keep this focused standalone target self-contained by amalgamating the small typed-IR implementation set
// (as ResourceMaterializationTests.cpp does).
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
