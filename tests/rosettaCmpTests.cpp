// S318: Rosetta 2 computes a wrong result for VCMPSS with a predicate other than 0, 1, 2, 11, 27 and for the legacy
// CMPSS with a predicate 3..7 when the result register is then overwritten by a full VEX load (shadPS4 #5297).
// The AMD patcher kind "rosettacmp" appends a move of the result register onto itself after such a compare.
// This test runs for real on the CPU (x86-64, under Rosetta on Apple silicon):
//  - for all 32 VCMPSS predicates and CMPSS 0..7, register and memory second operand, a destination below and
//    above xmm8, the unpatched code is run (under Rosetta it MUST show the bug: the test fails if it does not),
//    then the code patched with rosettacmp must match a reference model (the result of the compare and the value
//    after the VEX reload) for every input;
//  - on any host, the patcher output is checked structurally: the trampoline holds the compare followed by the
//    self move, compares outside the affected set are not touched, and the kind selection is honoured.
#include "common/amdCpuMode.h"
#include "loader/guestInstructionPatcher.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <sys/mman.h>
#include <vector>

namespace {

constexpr size_t kPage = 4096;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "rosettaCmpTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

using Bytes = std::vector<uint8_t>;
using Fn    = void (*)(uint32_t* out, const float* in);

struct Mapped {
	void*                               area = nullptr;
	Loader::GuestInstructionPatchResult result {};
	Fn                                  fn() const { return reinterpret_cast<Fn>(area); }
};

// Code in the first page, trampolines in the second, like a mapped guest module.
Mapped MapPatched(const Bytes& blob, bool patch, uint32_t kinds) {
	Mapped mapped;
	mapped.area = ::mmap(nullptr, 2 * kPage, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0);
	Check(mapped.area != MAP_FAILED, "map RWX area");
	std::memcpy(mapped.area, blob.data(), blob.size());
	if (patch) {
		auto* trampoline = static_cast<uint8_t*>(mapped.area) + kPage;
		Loader::RegisterGuestInstructionPatchModule(mapped.area, kPage, trampoline, kPage);
		const std::array<uintptr_t, 1> starts {reinterpret_cast<uintptr_t>(mapped.area)};
		mapped.result = Loader::PatchGuestInstructions(reinterpret_cast<uintptr_t>(mapped.area), blob.size(),
		                                               starts, false, true, {false, false, false}, kinds);
		Loader::UnregisterGuestInstructionPatchModule(mapped.area);
	}
	return mapped;
}

void Unmap(const Mapped& mapped) {
	::munmap(mapped.area, 2 * kPage);
}

enum class Form { VexReg, VexMem, LegacyReg, LegacyMem };

const char* FormName(Form form) {
	switch (form) {
		case Form::VexReg: return "vcmpss xmm,xmm,xmm";
		case Form::VexMem: return "vcmpss xmm,xmm,[mem]";
		case Form::LegacyReg: return "cmpss xmm,xmm";
		default: return "cmpss xmm,[mem]";
	}
}

bool IsLegacy(Form form) {
	return form == Form::LegacyReg || form == Form::LegacyMem;
}

// What the patcher is expected to fix (the Rosetta 2 bug, shadPS4 #5297).
bool IsAffected(Form form, unsigned predicate) {
	if (IsLegacy(form)) return predicate >= 3 && predicate <= 7;
	const unsigned low = predicate & 0x1f;
	return low != 0 && low != 1 && low != 2 && low != 11 && low != 27;
}

// in[0] = a, in[1] = b, in[2] = 1.0, in[4..7] = 0.5 (16-byte aligned). out[0] = 1.0 - 0.5 after the VEX reload,
// out[1] = the compare result of a second compare (destination xmm9, no reload) as bits.
Bytes BuildBlob(Form form, unsigned predicate) {
	Bytes                buffer(512, 0xcc);
	Xbyak::CodeGenerator g(buffer.size(), buffer.data());
	using namespace Xbyak::util;
	const auto compare = [&](const Xbyak::Xmm& destination, const Xbyak::Xmm& source) {
		const auto imm = static_cast<uint8_t>(predicate);
		if (form == Form::VexReg) g.vcmpss(destination, destination, source, imm);
		else if (form == Form::VexMem) g.vcmpss(destination, destination, g.ptr[rsi + 4], imm);
		else if (form == Form::LegacyReg) g.cmpss(destination, source, imm);
		else g.cmpss(destination, g.ptr[rsi + 4], imm);
	};
	g.vmovss(xmm5, g.ptr[rsi]);
	g.vmovss(xmm3, g.ptr[rsi + 4]);
	g.vmovss(xmm7, g.ptr[rsi + 8]);
	compare(xmm5, xmm3);
	g.vmovaps(xmm5, g.ptr[rsi + 16]);
	g.vsubss(xmm4, xmm7, xmm5);
	g.vmovss(g.ptr[rdi], xmm4);
	g.vmovss(xmm9, g.ptr[rsi]);
	g.vmovss(xmm10, g.ptr[rsi + 4]);
	compare(xmm9, xmm10);
	g.vmovd(g.ptr[rdi + 4], xmm9);
	g.ret();
	return Bytes(g.getCode(), g.getCode() + g.getSize());
}

// The architectural result (SDM, CMPSS predicate table). Predicates 16..31 differ from 0..15 only in the
// signalling behaviour on QNaN, which is masked here.
uint32_t Model(unsigned predicate, float a, float b) {
	const bool unordered = std::isnan(a) || std::isnan(b);
	bool       result    = false;
	switch (predicate & 15u) {
		case 0: result = !unordered && a == b; break;
		case 1: result = !unordered && a < b; break;
		case 2: result = !unordered && a <= b; break;
		case 3: result = unordered; break;
		case 4: result = unordered || a != b; break;
		case 5: result = unordered || !(a < b); break;
		case 6: result = unordered || !(a <= b); break;
		case 7: result = !unordered; break;
		case 8: result = unordered || a == b; break;
		case 9: result = unordered || !(a >= b); break;
		case 10: result = unordered || !(a > b); break;
		case 11: result = false; break;
		case 12: result = !unordered && a != b; break;
		case 13: result = !unordered && a >= b; break;
		case 14: result = !unordered && a > b; break;
		default: result = true; break;
	}
	return result ? 0xffffffffu : 0u;
}

constexpr uint32_t kAfterReload = 0x3f000000u; // 1.0 - 0.5

struct Inputs {
	float a, b;
};

constexpr float kNaN = __builtin_nanf("");
constexpr float kInf = __builtin_inff();
const Inputs kInputs[] = {{3.0f, 2.0f}, {2.0f, 3.0f}, {2.0f, 2.0f}, {kNaN, 2.0f}, {2.0f, kNaN}, {kInf, -kInf}};

// Runs the function for one input; returns the number of wrong words (0..2).
unsigned RunOnce(const Mapped& mapped, unsigned predicate, const Inputs& input) {
	alignas(16) float in[8] = {input.a, input.b, 1.0f, 0.0f, 0.5f, 0.5f, 0.5f, 0.5f};
	uint32_t          out[4] {0xdeadbeefu, 0xdeadbeefu, 0, 0};
	mapped.fn()(out, in);
	return (out[0] != kAfterReload) + (out[1] != Model(predicate, input.a, input.b));
}

// First occurrence of `needle` in the trampoline page, or nullptr.
bool TrampolineContains(const Mapped& mapped, const Bytes& needle) {
	const auto* begin = static_cast<const uint8_t*>(mapped.area) + kPage;
	return std::search(begin, begin + kPage, needle.begin(), needle.end()) != begin + kPage;
}

Bytes SelfMove(bool vex, unsigned reg) {
	Bytes                buffer(32, 0);
	Xbyak::CodeGenerator g(buffer.size(), buffer.data());
	if (vex) g.vmovaps(Xbyak::Xmm(reg), Xbyak::Xmm(reg));
	else g.movaps(Xbyak::Xmm(reg), Xbyak::Xmm(reg));
	return Bytes(g.getCode(), g.getCode() + g.getSize());
}

} // namespace

int main() {
	const bool     rosetta   = Common::IsRunningUnderRosetta();
	const uint32_t kinds     = Common::AmdPatchRosettaCmp;
	unsigned       unpatched_bad = 0;
	unsigned       patched_sites = 0;
	std::printf("rosettaCmpTests: running %s\n", rosetta ? "under Rosetta 2" : "on a native x86-64 CPU");

	for (const Form form: {Form::VexReg, Form::VexMem, Form::LegacyReg, Form::LegacyMem}) {
		const unsigned count = IsLegacy(form) ? 8 : 32;
		unsigned       form_bad = 0;
		for (unsigned predicate = 0; predicate < count; ++predicate) {
			const Bytes blob     = BuildBlob(form, predicate);
			const bool  affected = IsAffected(form, predicate);

			// Unpatched: the bug, where it is expected.
			const auto plain = MapPatched(blob, false, 0);
			unsigned   bad   = 0;
			for (const auto& input: kInputs) bad += RunOnce(plain, predicate, input) != 0;
			Unmap(plain);
			form_bad += bad != 0;
			if (bad != 0) {
				std::printf("  unpatched %-22s predicate %2u: %u of %zu inputs wrong\n", FormName(form), predicate,
				            bad, std::size(kInputs));
			}
			Check(rosetta || bad == 0, "the host computes the compare correctly");
			Check(bad == 0 || affected, "a wrong result outside the affected set (the set is wrong)");

			// Patched.
			const auto mapped = MapPatched(blob, true, kinds);
			const auto& r     = mapped.result;
			Check(r.rosettacmp.found == (affected ? 2u : 0u), "the number of affected compares found");
			Check(r.rosettacmp.native == r.rosettacmp.found && r.rosettacmp.trapped == 0 &&
			          r.rosettacmp.Skipped() == 0,
			      "every affected compare is patched natively");
			patched_sites += static_cast<unsigned>(r.rosettacmp.native);
			if (affected) {
				const bool vex = !IsLegacy(form);
				Check(TrampolineContains(mapped, SelfMove(vex, 5)), "self move after the compare (xmm5)");
				Check(TrampolineContains(mapped, SelfMove(vex, 9)), "self move after the compare (xmm9, REX)");
			} else {
				Check(std::memcmp(mapped.area, blob.data(), blob.size()) == 0, "an unaffected compare is untouched");
			}
			for (const auto& input: kInputs) {
				Check(RunOnce(mapped, predicate, input) == 0, "patched code matches the reference model");
			}
			Unmap(mapped);
		}
		std::printf("  %-22s: %u predicates wrong without the patch\n", FormName(form), form_bad);
		unpatched_bad += form_bad;
	}
	// Under Rosetta the bug has to be visible, otherwise the test proves nothing about the fix.
	Check(!rosetta || unpatched_bad > 0, "Rosetta shows the bug without the patch");

	// Kind selection: the default set without rosettacmp leaves the compares alone.
	{
		const Bytes blob   = BuildBlob(Form::VexReg, 3);
		const auto  mapped = MapPatched(blob, true, Common::AmdPatchAll);
		Check(mapped.result.rosettacmp.found == 0, "rosettacmp is not part of the plain default set");
		Check(std::memcmp(mapped.area, blob.data(), blob.size()) == 0, "kinds=all leaves the compare untouched");
		Unmap(mapped);
	}
	std::printf("rosettaCmpTests: all passed (%u sites patched, %u predicate/form pairs wrong unpatched)\n",
	            patched_sites, unpatched_bad);
	return 0;
}
