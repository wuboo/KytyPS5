// S441 DIAG: the census of walks without the tree JIT (srtTreeReject.h). Mutant: KYTY_TREEREJECT_MUTANT_ALL_ONE_ROW (rows of different reasons merge).

#include "graphics/shader/recompiler/ir/passes/srtTreeReject.h"

#include <cstdio>
#include <cstring>

namespace TR = Libs::Graphics::ShaderRecompiler::IR::SrtTreeReject;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	Check(!TR::Active().load(), "off without KYTY_DIAG_JIT_TREE_REJECT");
	for (size_t i = 0; i < static_cast<size_t>(TR::Reason::Count); i++) Check(std::strcmp(TR::Name(static_cast<TR::Reason>(i)), "?") != 0, "every reason has a name");
	Check(std::strcmp(TR::Name(TR::Reason::ControlFlow), "control_flow") == 0, "name of control_flow");

	TR::Census census;
	census.Note(0xAA, TR::Reason::ControlFlow, "");
	census.Note(0xAA, TR::Reason::ControlFlow, "");
	census.Note(0xAA, TR::Reason::ControlFlow, "", 3);
	census.Note(0xBB, TR::Reason::SlotOpcode, "ReadConstBuffer");
	census.Note(0xBB, TR::Reason::SlotOpcode, "ReadConst", 2);
	census.Note(0xCC, TR::Reason::CaptureRuntime, "", 7);
	census.Note(0xAA, TR::Reason::SlotInterp, "opcode:ReadConst", 4);
	const auto rows = census.Take();
	Check(rows.size() == 5, "five distinct (plan, reason, detail) rows");
	Check(rows[0].plan == 0xCC && rows[0].reason == TR::Reason::CaptureRuntime && rows[0].n == 7, "largest row first");
	Check(rows[1].plan == 0xAA && rows[1].reason == TR::Reason::ControlFlow && rows[1].n == 5, "same key adds up (1 + 1 + 3)");
	Check(rows[2].reason == TR::Reason::SlotInterp && rows[2].n == 4, "a reason of the same plan is a separate row");
	Check(rows[3].detail == "ReadConst" && rows[3].n == 2 && rows[4].detail == "ReadConstBuffer" && rows[4].n == 1, "detail splits rows");
	auto       sums = TR::Census::ByReason(rows);
	Check(sums[TR::Reason::SlotOpcode] == 3 && sums[TR::Reason::ControlFlow] == 5 && sums.size() == 4, "per-reason sums");
	Check(census.Take().empty(), "Take empties the census");
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("SrtTreeRejectTests: OK");
	return 0;
}
