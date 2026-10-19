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

} // namespace

int main() {
	TestNoReads();
	Cover::Override().store(-1);
	std::printf("SrtJitCoverTests: ok\n");
	return 0;
}

#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
