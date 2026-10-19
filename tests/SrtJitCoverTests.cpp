// S442: KYTY_SRT_JIT_COVER (srtJitCover.h): walks of plans with no reads, constant-buffer slots in the per-slot JIT.
// Mutants: KYTY_SRTCOVER_MUTANT_NOREADS_IGNORES_USERDATA, KYTY_SRTCOVER_MUTANT_CB_WRONG_OFFSET.

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/srtJitCover.h"
#include "graphics/shader/shader.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SrtJitCoverTests: failed: %s\n", text);
		std::exit(1);
	}
}

using namespace Libs::Graphics::ShaderRecompiler::IR;
namespace Cover = Libs::Graphics::ShaderRecompiler::IR::SrtJitCover;

Block& AddValueBlock(Program& program) {
	auto block = std::make_unique<Block>();
	auto* result = block.get();
	program.blocks.push_back(result);
	result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
	program.block_storage.push_back(std::move(block));
	return *result;
}

void SetMode(Cover::Mode mode) {
	Cover::Override().store(static_cast<int>(mode));
	(void)Cover::Drain();
}

std::array<uint64_t, Cover::kCellCount> Cells() {
	return Cover::Drain();
}

uint64_t Cell(const std::array<uint64_t, Cover::kCellCount>& cells, Cover::Cell cell) {
	return cells[static_cast<size_t>(cell)];
}

// ---- part 1: plans with no reads ----

ResourcePlan NoReadsPlan() {
	Program program;
	program.stage = Libs::Graphics::ShaderType::Compute;
	program.srt_plan_complete = true;
	program.resource_tracking_complete = true;
	AddValueBlock(program);
	return ExtractResourcePlan(program);
}

void TestNoReads() {
	auto plan = NoReadsPlan();
	Check(plan.srt_plan_complete && plan.srt_reads.empty() && plan.control_flow.empty(), "the test plan is not a no-reads plan");
	const std::vector<std::vector<uint32_t>> user_data {{}, {7u}, {0xdeadbeefu, 1u, 2u, 3u}};
	for (const auto mode: {Cover::Mode::Off, Cover::Mode::On, Cover::Mode::Shadow}) {
		SetMode(mode);
		for (const auto& words: user_data) {
			SrtRuntime runtime;
			runtime.user_data = words;
			SrtWalker walker(plan, runtime);
			std::vector<uint32_t> flat {1u, 2u, 3u};
			plan.active_sources.assign(3, 1u);
			Check(walker.RefreshFlatBuffer(flat), "no-reads refresh failed");
			Check(flat.empty(), "no-reads refresh produced a flat buffer (it must not depend on user data)");
			Check(plan.active_sources.empty(), "no-reads refresh left descriptor sources switched on or off");
		}
		const auto cells = Cells();
		if (mode == Cover::Mode::Off) {
			Check(Cell(cells, Cover::Cell::NoReads) == 0u, "the off mode counted a no-reads walk");
		} else {
			Check(Cell(cells, Cover::Cell::NoReads) == user_data.size(), "no-reads walks were not counted");
		}
		Check(Cell(cells, Cover::Cell::Ne) == 0u, "the shadow comparison of a no-reads walk differs");
	}
}


// ---- part 2: ReadConstBuffer in the per-slot JIT ----

alignas(16) uint32_t g_data[16];

// The constant-buffer read as the hardware describes it (the model the interpreter and the JIT must both match): V# = {low, high | stride << 16,
// records, word 3}; the DWORD at base + (imm & ~3) + (offset & ~3) when it fits in the buffer (size = stride ? stride * records : records), else 0.
uint32_t ModelRead(const uint32_t* v, uint32_t imm, uint32_t offset) {
	const uint64_t base   = ((static_cast<uint64_t>(v[1]) << 32u) | v[0]) & 0x0000fffffffffffcull;
	const uint32_t stride = (v[1] >> 16u) & 0x3fffu;
	const uint64_t size   = stride == 0u ? v[2] : static_cast<uint64_t>(stride) * v[2];
	const uint64_t at     = (imm & ~3u) + (offset & ~3u);
	if (at > size || size - at < 4u) return 0u;
	if (base < 0x40000u) return 0u;
	return *reinterpret_cast<const uint32_t*>(base + at);
}

// regs 4..7: V# words of slots 0 and 1 (offset args 8 and 12, memory immediates 0 and 4); regs 8, 9: a table {low, high | stride << 16} whose words are
// the first two V# words of slot 2 (words 2 and 3 from regs 6, 7; offset arg 12).
Program CbProgram() {
	Program program;
	program.stage = Libs::Graphics::ShaderType::Compute;
	program.srt_plan_complete = true;
	program.resource_tracking_complete = true;
	program.user_data_base = 4;
	program.user_data_count = 6;
	auto& block = AddValueBlock(program);
	for (const uint32_t imm: {0u, 4u}) {
		MemoryInfo memory;
		memory.kind = ResourceKind::ScalarBuffer;
		memory.planning_only = true;
		memory.offset = imm;
		program.memory_info.push_back(memory);
	}
	{
		MemoryInfo memory;
		memory.kind = ResourceKind::ScalarAddress;
		memory.planning_only = true;
		program.memory_info.push_back(memory);
	}
	const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) { return Value(&block.AppendNewInst(opcode, args)); };
	const auto user = [&](uint32_t reg) { return emit(ValueOpcode::GetUserData, {Value(ScalarReg(reg))}); };
	const auto read = [&](ValueOpcode opcode, Value handle, uint32_t offset, uint32_t index) {
		const auto value = opcode == ValueOpcode::ReadConstBuffer ? emit(opcode, {handle, Value(offset)})
		                                                          : emit(opcode, {handle, Value(offset), Value(0u), Value(true)});
		value.Instruction()->SetFlags(MemoryFlags {.index = index});
		program.srt_reads.push_back({value, static_cast<uint32_t>(program.srt_reads.size())});
		return value;
	};
	const auto vsharp = emit(ValueOpcode::GetBufferResource, {user(4), user(5), user(6), user(7)});
	read(ValueOpcode::ReadConstBuffer, vsharp, 8u, 0u);
	read(ValueOpcode::ReadConstBuffer, vsharp, 12u, 1u);
	const auto table = emit(ValueOpcode::GetAddressResource, {user(8), user(9)});
	const auto lo = emit(ValueOpcode::LoadAddressU32, {table, Value(0u), Value(0u), Value(true)});
	lo.Instruction()->SetFlags(MemoryFlags {.index = 2});
	const auto hi = emit(ValueOpcode::LoadAddressU32, {table, Value(4u), Value(0u), Value(true)});
	hi.Instruction()->SetFlags(MemoryFlags {.index = 2});
	const auto nested = emit(ValueOpcode::GetBufferResource, {lo, hi, user(6), user(7)});
	read(ValueOpcode::ReadConstBuffer, nested, 12u, 0u);
	return program;
}

void TestConstBuffer() {
	for (uint32_t i = 0; i < 16u; ++i) g_data[i] = 0xC0DE0000u + i;
	const auto address = reinterpret_cast<uint64_t>(g_data);
	struct Case {
		const char* name;
		uint32_t    stride;
		uint32_t    records;
		uint64_t    base;
	};
	const Case cases[] = {
	    {"bytes", 0u, 64u, address},          {"strided", 16u, 4u, address},  {"too-small", 0u, 8u, address},
	    {"exact-fit", 0u, 12u, address},      {"null-table", 0u, 64u, 0x1000u}, {"empty", 0u, 0u, address},
	    {"stride-over", 4u, 5u, address},     {"strided-small", 1u, 1u, address},
	};
	for (const auto mode: {Cover::Mode::Off, Cover::Mode::On, Cover::Mode::Shadow}) {
		SetMode(mode);
		auto plan = ExtractResourcePlan(CbProgram());
		Check(plan.srt_reads.size() == 3u, "the test plan lost a slot");
		Check(plan.srt_code == nullptr, "a constant-buffer plan compiled a whole-tree walker");
		Check((plan.srt_read_code != nullptr) == (mode != Cover::Mode::Off), "the per-slot JIT of constant-buffer slots follows the mode");
		uint64_t served = 0;
		for (const auto& c: cases) {
			const uint32_t hi_word = static_cast<uint32_t>(c.base >> 32u) | (c.stride << 16u);
			const uint32_t lo_word = static_cast<uint32_t>(c.base);
			uint32_t       table[2] = {lo_word, hi_word};
			const auto     table_address = reinterpret_cast<uint64_t>(table);
			std::vector<uint32_t> ud {lo_word, hi_word, c.records, 0x12345u, static_cast<uint32_t>(table_address),
			                          static_cast<uint32_t>(table_address >> 32u)};
			SrtRuntime runtime;
			runtime.user_data = ud;
			SrtWalker walker(plan, runtime);
			std::vector<uint32_t> flat;
			Check(walker.RefreshFlatBuffer(flat) && flat.size() == 3u, c.name);
			const uint32_t v[4] = {lo_word, hi_word, c.records, 0x12345u};
			Check(flat[0] == ModelRead(v, 0u, 8u), "slot 0 differs from the model");
			Check(flat[1] == ModelRead(v, 4u, 12u), "slot 1 (memory immediate 4) differs from the model");
			Check(flat[2] == ModelRead(v, 0u, 12u), "slot 2 (V# read through a table) differs from the model");
			if (c.base == address) served += 3;
		}
		const auto cells = Cells();
		Check(Cell(cells, Cover::Cell::Ne) == 0u, "constant-buffer JIT differs from the interpreter");
		if (mode == Cover::Mode::Off) {
			Check(Cell(cells, Cover::Cell::ConstBuffer) == 0u, "the off mode counted constant-buffer slots");
		} else {
			// every case with a mapped table is served by the JIT (an empty or too small buffer is a zero result of the JIT itself); the null table falls back
			Check(Cell(cells, Cover::Cell::ConstBuffer) == served, "constant-buffer slots were not served by the JIT (or not compared in shadow)");
		}
	}
}

void TestNeLineCap() {
	uint32_t logged = 0;
	for (uint32_t i = 0; i < 50u; ++i) logged += Cover::ShouldLogNe() ? 1u : 0u;
	Check(logged == Cover::kMaxNeLines && Cover::kMaxNeLines <= 20u, "[srt-cover-ne] lines are not capped at 20");
}

} // namespace

int main() {
	TestNoReads();
	TestConstBuffer();
	TestNeLineCap();
	Cover::Override().store(-1);
	std::printf("SrtJitCoverTests: ok\n");
	return 0;
}

#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
