// S486 (KYTY_SRT_PROT_AVOID, srtProtAvoid.h): the walk of a program whose hot page is protected reads its ordinary loads through the strict reader instead of the guest's
// mapping (backing and mapping differ in this test on purpose, so the result shows which one was read). Mutants (WILL_FAIL):
//   KYTY_SRTPA_MUTANT_ALWAYS_AVOID (avoids without a protected page), _NEVER_PROTECTED (never avoids), _NOTE_WHEN_AVOIDING (an avoiding walk teaches pages),
//   _NO_SCOPE_CHECK (a fault outside a walk teaches pages), _SHARE_PROGRAMS (one table for all programs), _NO_CAP (unbounded pages per program).

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/srtProtAvoid.h"
#include "graphics/shader/recompiler/ir/passes/srtWalkPaths.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <vector>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderRecompiler::CFG::TerminatorKind;
namespace Pa = Libs::Graphics::ShaderRecompiler::IR::SrtProtAvoid;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SrtProtAvoidTests: failed: %s\n", text);
		std::exit(1); // not abort(): the WILL_FAIL mutants need a plain nonzero exit code
	}
}

constexpr uint64_t kProgram = 0x5eed0001u;

std::array<uint32_t, 2> PointerWords(const void* pointer) {
	const auto address = reinterpret_cast<uint64_t>(pointer);
	return {static_cast<uint32_t>(address) | 3u, static_cast<uint32_t>(address >> 32u) | 0xabcd0000u};
}

// The guest's mapping (raw) and the backing: same pointer words, different data words.
struct Memory {
	std::array<uint32_t, 64> root {}, root_back {};
	std::array<uint32_t, 16> a {}, a_back {}, b {}, b_back {};
	std::array<uint32_t, 3>  user_data {};
	Memory() {
		const auto pa = PointerWords(a.data()), pb = PointerWords(b.data());
		root[0] = pa[0], root[1] = pa[1], root[2] = pb[0], root[3] = pb[1];
		for (uint32_t i = 0; i < 16; ++i) a[i] = 0xa0000000u | i, b[i] = 0xb0000000u | i;
		for (uint32_t i = 8; i < 64; ++i) root[i] = 0xc0000000u | i;
		root_back = root, a_back = a, b_back = b;
		for (uint32_t i = 8; i < 64; ++i) root_back[i] = 0xd0000000u | i;
		for (uint32_t i = 0; i < 16; ++i) a_back[i] = 0xe0000000u | i, b_back[i] = 0xf0000000u | i;
		const auto pr = PointerWords(root.data());
		user_data = {pr[0], pr[1], 0u};
	}
	// The backing byte of a mapping address.
	bool Backing(uint64_t address, uint32_t* out, size_t dwords) const {
		const auto in = [&](const auto& raw, const auto& back) {
			const auto base = reinterpret_cast<uint64_t>(raw.data());
			if (address < base || address + dwords * 4 > base + raw.size() * 4) return false;
			std::memcpy(out, reinterpret_cast<const uint8_t*>(back.data()) + (address - base), dwords * 4);
			return true;
		};
		return in(root, root_back) || in(a, a_back) || in(b, b_back);
	}
};

struct Strict {
	const Memory* memory = nullptr;
	uint32_t      reads  = 0;
};

bool ReadStrict(void* data, uint64_t address, std::span<uint32_t> out) {
	auto& strict = *static_cast<Strict*>(data);
	++strict.reads;
	return strict.memory->Backing(address, out.data(), out.size());
}

bool g_protected = false;
bool HookProtected(uint64_t) {
	return g_protected;
}

// One block, four slots: root+32, A+0, B+0, A+4. Flat offsets run backwards (slot s -> 3 - s).
std::unique_ptr<Program> Build() {
	auto program = std::make_unique<Program>();
	program->stage                 = Libs::Graphics::ShaderType::Compute;
	program->shader_hash           = kProgram;
	program->user_data_base        = 4;
	program->user_data_count       = 3;
	program->srt_plan_complete = program->resource_tracking_complete = true;
	program->memory_info.push_back({.kind = ResourceKind::ScalarAddress});
	auto block   = std::make_unique<Block>();
	auto* entry  = block.get();
	program->blocks.push_back(entry);
	entry->id = 0;
	program->block_storage.push_back(std::move(block));
	const auto emit = [&](ValueOpcode opcode, std::initializer_list<Value> args) { return Value(&entry->AppendNewInst(opcode, args)); };
	const auto low  = emit(ValueOpcode::GetUserData, {Value(ScalarReg(4))});
	const auto high = emit(ValueOpcode::GetUserData, {Value(ScalarReg(5))});
	const auto root = emit(ValueOpcode::GetAddressResource, {low, high});
	const auto load = [&](Value handle, uint32_t offset) {
		const auto value = emit(ValueOpcode::LoadAddressU32, {handle, Value(offset), Value(0u), Value(true)});
		value.Instruction()->SetFlags(MemoryFlags {.index = 0});
		return value;
	};
	const auto table = [&](uint32_t offset) { return emit(ValueOpcode::GetAddressResource, {load(root, offset), load(root, offset + 4u)}); };
	const auto ta = table(0u), tb = table(8u);
	const auto srt = emit(ValueOpcode::GetSrtResource, {});
	const std::vector<Value> values {load(root, 32u), load(ta, 0u), load(tb, 0u), load(ta, 4u)};
	for (uint32_t slot = 0; slot < values.size(); ++slot) program->srt_reads.push_back({values[slot], static_cast<uint32_t>(values.size()) - 1u - slot});
	for (uint32_t slot = 0; slot < values.size(); ++slot) emit(ValueOpcode::ReadConst, {srt, Value(slot)});
	entry->terminator.kind = TerminatorKind::Return;
	return program;
}

struct Walk {
	bool                                       ok = false;
	std::vector<uint32_t>                      flat;
	std::vector<std::pair<uint64_t, uint64_t>> reads;
	std::array<uint64_t, static_cast<size_t>(Pa::Cell::Count)> cells {};
	uint64_t                                   slot_jit = 0;
	uint32_t                                   strict_reads = 0;
};

Walk Run(const Memory& memory, Pa::Mode mode, bool protected_page) {
	Pa::Override().store(static_cast<int>(mode));
	g_protected = protected_page;
	(void)Pa::Drain();
	SrtWalkPaths::Active().store(true);
	(void)SrtWalkPaths::Drain();
	auto   program = Build();
	auto   plan    = ExtractResourcePlan(*program);
	Walk   walk;
	Strict strict {&memory};
	SrtRuntime runtime {.user_data = memory.user_data, .userdata = &strict, .read_specialization_memory = ReadStrict};
	SrtReadCapture capture(runtime, walk.reads);
	const auto     observed = capture.ObservedRuntime();
	{
		SrtWalker clean(plan, CleanRuntime(observed));
		SrtWalker walker(plan, observed, &clean);
		walk.ok = walker.RefreshFlatBuffer(walk.flat);
	}
	walk.cells        = Pa::Drain();
	walk.slot_jit     = SrtWalkPaths::Drain()[static_cast<size_t>(SrtWalkPaths::Path::SlotJit)];
	walk.strict_reads = strict.reads;
	return walk;
}

uint64_t Cell(const Walk& walk, Pa::Cell cell) {
	return walk.cells[static_cast<size_t>(cell)];
}

std::vector<uint32_t> Expect(uint32_t root8, uint32_t a0, uint32_t b0, uint32_t a1) {
	return {a1, b0, a0, root8}; // slot s at flat 3 - s
}

void LearnPages(const Memory& memory) {
	Pa::Global().Clear();
	Pa::Global().Note(kProgram, Pa::PageOf(reinterpret_cast<uint64_t>(memory.root.data())));
	Pa::Global().Note(kProgram, Pa::PageOf(reinterpret_cast<uint64_t>(memory.a.data())));
	Pa::Global().Note(kProgram, Pa::PageOf(reinterpret_cast<uint64_t>(memory.b.data())));
}

const std::vector<uint32_t> kRaw  = Expect(0xc0000008u, 0xa0000000u, 0xb0000000u, 0xa0000001u);
const std::vector<uint32_t> kBack = Expect(0xd0000008u, 0xe0000000u, 0xf0000000u, 0xe0000001u);

// T1: mode 0 reads the mapping, whatever the table says.
void TestOff() {
	Memory memory;
	LearnPages(memory);
	const auto walk = Run(memory, Pa::Mode::Off, true);
	Check(walk.ok && walk.flat == kRaw, "mode 0 must read the guest's mapping");
	Check(walk.strict_reads == 0u && Cell(walk, Pa::Cell::WalkAvoided) == 0u, "mode 0 touched the avoid path");
	Check(walk.slot_jit > 0u, "mode 0 did not use the compiled slots");
}

// T2: a hot page that is protected: the strict reader's bytes, no compiled slot.
void TestAvoid() {
	Memory memory;
	LearnPages(memory);
	const auto walk = Run(memory, Pa::Mode::On, true);
	Check(walk.ok, "avoid walk failed");
	Check(walk.flat == kBack, "an avoiding walk must read through the strict reader (the backing)");
	Check(Cell(walk, Pa::Cell::WalkAvoided) == 1u, "walk not counted as avoided");
	Check(Cell(walk, Pa::Cell::ReadBacking) > 0u && Cell(walk, Pa::Cell::ReadFallback) == 0u, "reads not served by the strict reader");
	Check(walk.slot_jit == 0u, "an avoiding walk entered compiled code (a compiled load faults on a protected page)");
	Check(!walk.reads.empty(), "the avoiding walk did not capture its reads");
}

// T3 / T4 / T7: no avoid without a protected hot page of THIS program.
void TestNoAvoid() {
	Memory memory;
	LearnPages(memory);
	auto walk = Run(memory, Pa::Mode::On, false);
	Check(walk.ok && walk.flat == kRaw && walk.strict_reads == 0u && walk.slot_jit > 0u, "hot pages that are not protected must not avoid");
	Pa::Global().Clear();
	walk = Run(memory, Pa::Mode::On, true);
	Check(walk.ok && walk.flat == kRaw && walk.strict_reads == 0u, "a program with no hot page must not avoid");
	Pa::Global().Clear();
	Pa::Global().Note(kProgram + 1u, Pa::PageOf(reinterpret_cast<uint64_t>(memory.root.data())));
	walk = Run(memory, Pa::Mode::On, true);
	Check(walk.ok && walk.flat == kRaw && walk.strict_reads == 0u, "pages learned for another program must not avoid");
}

// T6: shadow: the strict read first, the mapping is the result; equal bytes -> 0 differences, different bytes are counted.
void TestShadow() {
	Memory memory;
	LearnPages(memory);
	auto walk = Run(memory, Pa::Mode::Shadow, true);
	Check(walk.ok && walk.flat == kRaw, "shadow must return the mapping's bytes");
	Check(Cell(walk, Pa::Cell::ShadowNe) > 0u, "shadow did not see the (deliberate) difference");
	Check(walk.strict_reads > 0u, "shadow did not read the backing");
	memory.root_back = memory.root;
	memory.a_back    = memory.a;
	memory.b_back    = memory.b;
	walk = Run(memory, Pa::Mode::Shadow, true);
	Check(walk.ok && walk.flat == kRaw && Cell(walk, Pa::Cell::ShadowNe) == 0u && Cell(walk, Pa::Cell::ReadBacking) == 0u, "equal bytes: shadow count must be 0");
}

// T5: what teaches the table.
void TestLearning() {
	Pa::Override().store(static_cast<int>(Pa::Mode::On));
	Pa::Global().Clear();
	Pa::NoteFault(0x50740a5b0u);
	Check(Pa::Global().Programs() == 0u, "a fault outside any walk taught the table");
	{
		const Pa::WalkScope scope(7u, false);
		Pa::NoteFault(0x50740a5b0u);
		Pa::NoteFault(0x50740a5c0u); // same page
		Pa::NoteFault(0x50740b000u);
	}
	Check(Pa::Global().PagesOf(7u) == 2u, "pages of a walk's faults (two distinct pages)");
	{
		const Pa::WalkScope scope(8u, true);
		Pa::NoteFault(0x50740a5b0u);
	}
	Check(Pa::Global().PagesOf(8u) == 0u, "a walk that avoids must not teach the table");
	{
		const Pa::WalkScope outer(9u, false);
		{
			const Pa::WalkScope inner(10u, true);
			Check(Pa::Avoiding() && Pa::CurrentWalk().program == 10u, "inner scope");
		}
		Check(!Pa::Avoiding() && Pa::CurrentWalk().program == 9u, "scope not restored");
	}
	Check(!Pa::Avoiding() && Pa::CurrentWalk().program == 0u, "scope leaked");
	Pa::Override().store(static_cast<int>(Pa::Mode::Off));
	Pa::NoteFault(0x50740a5b0u);
	{
		const Pa::WalkScope scope(11u, false);
		Pa::NoteFault(0x50740a5b0u);
	}
	Check(Pa::Global().PagesOf(11u) == 0u, "mode 0 taught the table");
	Pa::Override().store(static_cast<int>(Pa::Mode::On));
	for (uint64_t i = 0; i < 20; ++i) (void)Pa::Global().Note(12u, 0x1000u * (i + 1));
	Check(Pa::Global().PagesOf(12u) == Pa::HotPages::kPerProgram, "pages per program are capped");
	Check(!Pa::Global().Note(12u, 0x99000u), "a ninth page was accepted");
	Check(!Pa::Global().Note(12u, 0x1000u), "a known page was reported as new");
	Check(Pa::Parse("1") == Pa::Mode::On && Pa::Parse("shadow") == Pa::Mode::Shadow && Pa::Parse("0") == Pa::Mode::Off && Pa::Parse(nullptr) == Pa::Mode::Off, "Parse");
}

} // namespace

int main() {
	Pa::ProtectedHook().store(&HookProtected);
	TestOff();
	TestAvoid();
	TestNoAvoid();
	TestShadow();
	TestLearning();
	Pa::Override().store(-1);
	std::puts("SrtProtAvoidTests: all cases passed");
	return 0;
}

// The IR library is not linked into the walker tests: compile what the plan needs into this binary.
#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
