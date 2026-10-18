// S426: KYTY_GF_EMULATE. The decoder / executor of guest read faults (src/loader/guestReadEmulator.cpp), checked two ways:
//  - on a fabricated saved context: every supported form leaves the register the instruction would, RIP moves by the instruction length,
//    every unsupported form is refused with its reason;
//  - with real SIGSEGVs on a PROT_NONE page: the handler serves the load from a separate "backing" buffer (value in the register == backing,
//    not the page), the page stays protected (the second run faults again), dirty bytes and unsupported forms take the ordinary path.
// Mutants (WILL_FAIL): EMU_IGNORES_DIRTY (emulates GPU-dirty bytes), EMU_WRONG_RIP (RIP skips one byte too many).
// S428: B1 (the 8-bit destinations SPL/BPL/SIL/DIL/R8B..R15B: Zydis ids 8..19 are encoding numbers 4..15) with all 16 REX byte registers on a fabricated
// context and on a real SIGSEGV, the reasons of a fallback (ClassifyFallback, BackingResult -> Why), and a parity check of every form against the
// native instruction on the same bytes. Mutant (WILL_FAIL): GPR8_RAW_ID.
// S427: the VEX.128 forms (vmovss/sd/d/q/ups/upd/dqu/aps/apd/dqa). Bits 128..255 of the YMM register are zeroed (legacy SSE leaves them), vvvv must be
// 1111b, L = 1 / EVEX are fallbacks, a context without the AVX layout is a fallback. Mutants (WILL_FAIL): VEX_NO_ZERO_UPPER, VEX_IGNORE_VVVV.

#include "common/hostException.h"
#include "loader/guestReadEmulator.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using namespace Loader::GuestReadEmulator;
using Common::HostException::AccessViolationType;
using Common::HostException::ExceptionInfo;
using Common::HostException::ExceptionType;

constexpr uint64_t kPageSize = 4096;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "GuestReadEmulatorTests: failed: %s\n", text);
		std::exit(1); // ctest WILL_FAIL does not invert a signal exit
	}
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// The backing: 64 bytes at g_base. Reads outside are refused; g_dirty / g_inflight play the buffer cache.
std::array<uint8_t, 64> g_backing {};
uint64_t                g_base = 0;
std::atomic<bool>       g_dirty {false};
std::atomic<bool>       g_inflight {false};

BackingResult TestRead(uint64_t address, uint64_t size, void* dst) {
	const Gate gate {Mode::On, true, false, g_dirty.load(), g_inflight.load()};
	if (address < g_base || address + size > g_base + g_backing.size()) {
		return BackingResult::Unavailable;
	}
	if (!MayEmulate(gate)) {
		return g_dirty.load() ? BackingResult::Dirty : BackingResult::Inflight;
	}
	std::memcpy(dst, g_backing.data() + (address - g_base), size);
	return BackingResult::Ok;
}

void FillBacking() {
	for (size_t i = 0; i < g_backing.size(); ++i) {
		g_backing[i] = static_cast<uint8_t>(0x88 + i * 0x0b);
	}
}

uint64_t LoadLe(const uint8_t* p, unsigned n) {
	uint64_t v = 0;
	std::memcpy(&v, p, n);
	return v;
}

uint64_t Sx(const uint8_t* p, unsigned n) {
	uint64_t v = LoadLe(p, n);
	if (n < 8 && (v >> (n * 8 - 1)) != 0) {
		v |= ~uint64_t {0} << (n * 8);
	}
	return v;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Fabricated context.
struct Fake {
	_STRUCT_MCONTEXT_AVX64 mc {};
	ucontext_t             uc {};
	// `avx` = false plays a signal context without the upper YMM halves (the plain _STRUCT_MCONTEXT64 size).
	explicit Fake(bool avx = true) {
		uc.uc_mcontext = reinterpret_cast<_STRUCT_MCONTEXT64*>(&mc);
		uc.uc_mcsize   = avx ? sizeof(_STRUCT_MCONTEXT_AVX64) : sizeof(_STRUCT_MCONTEXT64);
	}
	void* Ymmh(uint8_t id) {
		void* x[] = {&mc.__fs.__fpu_ymmh0, &mc.__fs.__fpu_ymmh1, &mc.__fs.__fpu_ymmh2,  &mc.__fs.__fpu_ymmh3,
		             &mc.__fs.__fpu_ymmh4, &mc.__fs.__fpu_ymmh5, &mc.__fs.__fpu_ymmh6,  &mc.__fs.__fpu_ymmh7,
		             &mc.__fs.__fpu_ymmh8, &mc.__fs.__fpu_ymmh9, &mc.__fs.__fpu_ymmh10, &mc.__fs.__fpu_ymmh11,
		             &mc.__fs.__fpu_ymmh12, &mc.__fs.__fpu_ymmh13, &mc.__fs.__fpu_ymmh14, &mc.__fs.__fpu_ymmh15};
		return x[id];
	}
	uint64_t& Reg(uint8_t id) {
		uint64_t* r[] = {&mc.__ss.__rax, &mc.__ss.__rcx, &mc.__ss.__rdx, &mc.__ss.__rbx, &mc.__ss.__rsp, &mc.__ss.__rbp,
		                 &mc.__ss.__rsi, &mc.__ss.__rdi, &mc.__ss.__r8,  &mc.__ss.__r9,  &mc.__ss.__r10, &mc.__ss.__r11,
		                 &mc.__ss.__r12, &mc.__ss.__r13, &mc.__ss.__r14, &mc.__ss.__r15};
		return *r[id];
	}
	void* Xmm(uint8_t id) {
		void* x[] = {&mc.__fs.__fpu_xmm0, &mc.__fs.__fpu_xmm1, &mc.__fs.__fpu_xmm2,  &mc.__fs.__fpu_xmm3,
		             &mc.__fs.__fpu_xmm4, &mc.__fs.__fpu_xmm5, &mc.__fs.__fpu_xmm6,  &mc.__fs.__fpu_xmm7,
		             &mc.__fs.__fpu_xmm8, &mc.__fs.__fpu_xmm9, &mc.__fs.__fpu_xmm10, &mc.__fs.__fpu_xmm11,
		             &mc.__fs.__fpu_xmm12, &mc.__fs.__fpu_xmm13, &mc.__fs.__fpu_xmm14, &mc.__fs.__fpu_xmm15};
		return x[id];
	}
};

struct Case {
	const char*          name;
	std::vector<uint8_t> code;
	Kind                 kind;
	uint8_t              dst;
	bool                 xmm;
	unsigned             size;      // bytes read
	unsigned             offset;    // of the operand from rdi (and from g_base)
	unsigned             dst_bytes; // GPR width, or 0
	bool                 sign;
	bool                 vex = false; // VEX.128: the upper YMM half is zeroed
};

constexpr uint64_t kSentinel = 0xfedcba9876543210ull;

void RunFake(const Case& c) {
	FillBacking();
	// The code sits in its own buffer so that the decoder reads real bytes; it is followed by a marker.
	std::vector<uint8_t> code = c.code;
	code.resize(code.size() + 16, 0x90);
	Fake fake;
	fake.mc.__ss.__rip = reinterpret_cast<uint64_t>(code.data());
	g_base             = 0x100000;
	fake.Reg(7)        = g_base; // rdi
	fake.Reg(1)        = 2;      // rcx
	for (uint8_t i = 0; i < 16; ++i) {
		if (i != 7 && i != 1 && i != 4) {
			fake.Reg(i) = kSentinel;
		}
		std::memset(fake.Xmm(i), 0xee, 16);
		std::memset(fake.Ymmh(i), 0xee, 16);
	}
	const uint64_t fault = g_base + c.offset;
	Attempt        attempt;
	char           text[160];
	std::snprintf(text, sizeof(text), "%s: Prepare accepts (why=%d)", c.name, static_cast<int>(attempt.plan.why));
	const bool ok = Prepare(&fake.uc, fault, TestRead, &attempt);
	std::snprintf(text, sizeof(text), "%s: Prepare accepts (why=%d)", c.name, static_cast<int>(attempt.plan.why));
	Check(ok, text);
	Check(attempt.plan.kind == c.kind, c.name);
	Check(attempt.plan.size == c.size && attempt.plan.address == fault, c.name);
	Commit(&fake.uc, attempt.plan, attempt.bytes);
	std::snprintf(text, sizeof(text), "%s: RIP moves by the instruction length", c.name);
	Check(fake.mc.__ss.__rip == reinterpret_cast<uint64_t>(code.data()) + c.code.size(), text);
	const uint8_t* src = g_backing.data() + c.offset;
	if (c.xmm) {
		uint8_t expect[16];
		std::memset(expect, 0, sizeof(expect));
		if (c.kind == Kind::Xmm128) {
			std::memcpy(expect, src, 16);
		} else {
			std::memcpy(expect, src, c.size); // scalar loads zero the rest
		}
		std::snprintf(text, sizeof(text), "%s: xmm%d holds the backing bytes", c.name, c.dst);
		Check(std::memcmp(fake.Xmm(c.dst), expect, 16) == 0, text);
		for (uint8_t i = 0; i < 16; ++i) {
			if (i != c.dst) {
				uint8_t untouched[16];
				std::memset(untouched, 0xee, 16);
				Check(std::memcmp(fake.Xmm(i), untouched, 16) == 0, "other xmm registers untouched");
				Check(std::memcmp(fake.Ymmh(i), untouched, 16) == 0, "other ymm upper halves untouched");
			}
		}
		uint8_t upper[16];
		std::memset(upper, c.vex ? 0x00 : 0xee, 16);
		std::snprintf(text, sizeof(text), "%s: the upper half of ymm%d is %s", c.name, c.dst, c.vex ? "zeroed (VEX)" : "untouched (legacy SSE)");
		Check(std::memcmp(fake.Ymmh(c.dst), upper, 16) == 0, text);
		return;
	}
	uint64_t expect = 0;
	if (c.kind == Kind::Gpr) {
		expect = LoadLe(src, c.size);
	} else {
		expect = c.sign ? Sx(src, c.size) : LoadLe(src, c.size);
	}
	switch (c.dst_bytes) {
		case 1: expect = (kSentinel & ~uint64_t {0xff}) | (expect & 0xff); break;
		case 2: expect = (kSentinel & ~uint64_t {0xffff}) | (expect & 0xffff); break;
		case 4: expect &= 0xffffffffull; break;
		default: break;
	}
	std::snprintf(text, sizeof(text), "%s: r%d == 0x%llx (got 0x%llx)", c.name, c.dst, static_cast<unsigned long long>(expect),
	              static_cast<unsigned long long>(fake.Reg(c.dst)));
	Check(fake.Reg(c.dst) == expect, text);
	for (uint8_t i = 0; i < 16; ++i) {
		if (i != c.dst && i != 7 && i != 1 && i != 4) {
			Check(fake.Reg(i) == kSentinel, "other registers untouched");
		}
	}
}

void TestSupportedForms() {
	const std::vector<Case> cases = {
	    {"mov eax,[rdi]", {0x8b, 0x07}, Kind::Gpr, 0, false, 4, 0, 4, false},
	    {"mov rax,[rdi]", {0x48, 0x8b, 0x07}, Kind::Gpr, 0, false, 8, 0, 8, false},
	    {"mov ax,[rdi]", {0x66, 0x8b, 0x07}, Kind::Gpr, 0, false, 2, 0, 2, false},
	    {"mov al,[rdi]", {0x8a, 0x07}, Kind::Gpr, 0, false, 1, 0, 1, false},
	    {"mov r10d,[rdi]", {0x44, 0x8b, 0x17}, Kind::Gpr, 10, false, 4, 0, 4, false},
	    {"mov r15,[rdi+8]", {0x4c, 0x8b, 0x7f, 0x08}, Kind::Gpr, 15, false, 8, 8, 8, false},
	    {"mov eax,[rdi+rcx*4+8]", {0x8b, 0x44, 0x8f, 0x08}, Kind::Gpr, 0, false, 4, 16, 4, false},
	    {"movzx eax,byte [rdi]", {0x0f, 0xb6, 0x07}, Kind::GprZext, 0, false, 1, 0, 4, false},
	    {"movzx rax,word [rdi]", {0x48, 0x0f, 0xb7, 0x07}, Kind::GprZext, 0, false, 2, 0, 8, false},
	    {"movzx ax,byte [rdi]", {0x66, 0x0f, 0xb6, 0x07}, Kind::GprZext, 0, false, 1, 0, 2, false},
	    {"movsx eax,byte [rdi]", {0x0f, 0xbe, 0x07}, Kind::GprSext, 0, false, 1, 0, 4, true},
	    {"movsx rax,word [rdi]", {0x48, 0x0f, 0xbf, 0x07}, Kind::GprSext, 0, false, 2, 0, 8, true},
	    {"movsxd rax,dword [rdi+4]", {0x48, 0x63, 0x47, 0x04}, Kind::GprSext, 0, false, 4, 4, 8, true},
	    {"movss xmm0,[rdi]", {0xf3, 0x0f, 0x10, 0x07}, Kind::XmmScalar, 0, true, 4, 0, 0, false},
	    {"movsd xmm0,[rdi]", {0xf2, 0x0f, 0x10, 0x07}, Kind::XmmScalar, 0, true, 8, 0, 0, false},
	    {"movd xmm0,[rdi]", {0x66, 0x0f, 0x6e, 0x07}, Kind::XmmScalar, 0, true, 4, 0, 0, false},
	    {"movq xmm0,[rdi]", {0xf3, 0x0f, 0x7e, 0x07}, Kind::XmmScalar, 0, true, 8, 0, 0, false},
	    {"movss xmm12,[rdi+4]", {0xf3, 0x44, 0x0f, 0x10, 0x67, 0x04}, Kind::XmmScalar, 12, true, 4, 4, 0, false},
	    {"movups xmm0,[rdi]", {0x0f, 0x10, 0x07}, Kind::Xmm128, 0, true, 16, 0, 0, false},
	    {"movupd xmm1,[rdi+16]", {0x66, 0x0f, 0x10, 0x4f, 0x10}, Kind::Xmm128, 1, true, 16, 16, 0, false},
	    {"movdqu xmm0,[rdi]", {0xf3, 0x0f, 0x6f, 0x07}, Kind::Xmm128, 0, true, 16, 0, 0, false},
	    {"movaps xmm0,[rdi]", {0x0f, 0x28, 0x07}, Kind::Xmm128, 0, true, 16, 0, 0, false},
	    {"movapd xmm0,[rdi]", {0x66, 0x0f, 0x28, 0x07}, Kind::Xmm128, 0, true, 16, 0, 0, false},
	    {"movdqa xmm0,[rdi+32]", {0x66, 0x0f, 0x6f, 0x47, 0x20}, Kind::Xmm128, 0, true, 16, 32, 0, false},
	    {"movups xmm9,[rdi]", {0x44, 0x0f, 0x10, 0x0f}, Kind::Xmm128, 9, true, 16, 0, 0, false},
	    // S427: VEX.128
	    {"vmovss xmm0,[rdi]", {0xc5, 0xfa, 0x10, 0x07}, Kind::XmmScalar, 0, true, 4, 0, 0, false, true},
	    {"vmovsd xmm0,[rdi]", {0xc5, 0xfb, 0x10, 0x07}, Kind::XmmScalar, 0, true, 8, 0, 0, false, true},
	    {"vmovd xmm0,[rdi]", {0xc5, 0xf9, 0x6e, 0x07}, Kind::XmmScalar, 0, true, 4, 0, 0, false, true},
	    {"vmovq xmm0,[rdi] (F3 0F 7E)", {0xc5, 0xfa, 0x7e, 0x07}, Kind::XmmScalar, 0, true, 8, 0, 0, false, true},
	    {"vmovq xmm0,[rdi] (VEX.W1 6E)", {0xc4, 0xe1, 0xf9, 0x6e, 0x07}, Kind::XmmScalar, 0, true, 8, 0, 0, false, true},
	    {"vmovups xmm0,[rdi]", {0xc5, 0xf8, 0x10, 0x07}, Kind::Xmm128, 0, true, 16, 0, 0, false, true},
	    {"vmovupd xmm1,[rdi+16]", {0xc5, 0xf9, 0x10, 0x4f, 0x10}, Kind::Xmm128, 1, true, 16, 16, 0, false, true},
	    {"vmovdqu xmm0,[rdi]", {0xc5, 0xfa, 0x6f, 0x07}, Kind::Xmm128, 0, true, 16, 0, 0, false, true},
	    {"vmovaps xmm0,[rdi]", {0xc5, 0xf8, 0x28, 0x07}, Kind::Xmm128, 0, true, 16, 0, 0, false, true},
	    {"vmovapd xmm0,[rdi+16]", {0xc5, 0xf9, 0x28, 0x47, 0x10}, Kind::Xmm128, 0, true, 16, 16, 0, false, true},
	    {"vmovdqa xmm0,[rdi+32]", {0xc5, 0xf9, 0x6f, 0x47, 0x20}, Kind::Xmm128, 0, true, 16, 32, 0, false, true},
	    {"vmovss xmm12,[rdi+4] (3-byte VEX)", {0xc4, 0x61, 0x7a, 0x10, 0x67, 0x04}, Kind::XmmScalar, 12, true, 4, 4, 0, false, true},
	    {"vmovdqu xmm9,[rdi]", {0xc5, 0x7a, 0x6f, 0x0f}, Kind::Xmm128, 9, true, 16, 0, 0, false, true},
	};
	for (const auto& c: cases) {
		RunFake(c);
	}
}

void TestRipRelative() {
	FillBacking();
	// mov eax,[rip+0x20]: the operand is 0x20 bytes behind the end of the instruction.
	std::vector<uint8_t> code = {0x8b, 0x05, 0x20, 0x00, 0x00, 0x00, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
	Fake                 fake;
	const uint64_t       rip = reinterpret_cast<uint64_t>(code.data());
	fake.mc.__ss.__rip       = rip;
	g_base                   = rip + 6 + 0x20;
	Attempt attempt;
	Check(Prepare(&fake.uc, g_base, TestRead, &attempt), "rip-relative load accepted");
	Check(attempt.plan.address == g_base, "rip-relative address uses the end of the instruction");
	Commit(&fake.uc, attempt.plan, attempt.bytes);
	Check(fake.mc.__ss.__rax == LoadLe(g_backing.data(), 4) && fake.mc.__ss.__rip == rip + 6, "rip-relative load result");
}

struct Refusal {
	const char*          name;
	std::vector<uint8_t> code;
	Why                  why;
	uint64_t             offset; // fault offset from rdi
	uint64_t             rdi_extra;
};

void TestRefusals() {
	FillBacking();
	const std::vector<Refusal> cases = {
	    {"cmp eax,[rdi]", {0x3b, 0x07}, Why::NotALoad, 0, 0},
	    {"add eax,[rdi]", {0x03, 0x07}, Why::NotALoad, 0, 0},
	    {"test [rdi],eax", {0x85, 0x07}, Why::NotALoad, 0, 0},
	    {"mov [rdi],eax", {0x89, 0x07}, Why::NotALoad, 0, 0},
	    {"movups [rdi],xmm0", {0x0f, 0x11, 0x07}, Why::NotALoad, 0, 0},
	    {"mov ah,[rdi]", {0x8a, 0x27}, Why::HighByteReg, 0, 0},
	    {"vmovdqu ymm0,[rdi] (VEX.L = 1)", {0xc5, 0xfe, 0x6f, 0x07}, Why::Vex, 0, 0},
	    {"vmovups ymm0,[rdi] (VEX.L = 1)", {0xc5, 0xfc, 0x10, 0x07}, Why::Vex, 0, 0},
	    {"vmovss xmm0,[rdi] with VEX.L = 1", {0xc5, 0xfe, 0x10, 0x07}, Why::Vex, 0, 0},
	    // vvvv != 1111b is #UD; Zydis refuses to decode it (DecodeFailed), the decoder's own check is the second line (VEX_IGNORE_VVVV hides it from both).
	    {"vmovss xmm0,[rdi] with vvvv != 1111b", {0xc5, 0xf2, 0x10, 0x07}, Why::DecodeFailed, 0, 0},
	    {"vmovdqu xmm0,[rdi] with vvvv != 1111b (3-byte)", {0xc4, 0xe1, 0x72, 0x6f, 0x07}, Why::DecodeFailed, 0, 0},
	    {"evex vmovups xmm0,[rdi]", {0x62, 0xf1, 0x7c, 0x08, 0x10, 0x07}, Why::Vex, 0, 0},
	    {"vmovups [rdi],xmm0 (store)", {0xc5, 0xf8, 0x11, 0x07}, Why::NotALoad, 0, 0},
	    {"vpxor xmm0,xmm0,[rdi]", {0xc5, 0xf9, 0xef, 0x07}, Why::NotALoad, 0, 0},
	    {"vmovaps xmm0,[rdi+4] misaligned", {0xc5, 0xf8, 0x28, 0x47, 0x04}, Why::Misaligned, 4, 0},
	    {"vmovdqa xmm0,[rdi+8] misaligned", {0xc5, 0xf9, 0x6f, 0x47, 0x08}, Why::Misaligned, 8, 0},
	    {"vmovdqu xmm0,[rdi+4088] crosses the page", {0xc5, 0xfa, 0x6f, 0x87, 0xf8, 0x0f, 0x00, 0x00}, Why::PageCross, 4088, 0},
	    {"mov eax,fs:[rdi]", {0x64, 0x8b, 0x07}, Why::Prefix, 0, 0},
	    {"mov eax,[edi]", {0x67, 0x8b, 0x07}, Why::Prefix, 0, 0},
	    {"mov rax,[rdi+4092] crosses the page", {0x48, 0x8b, 0x87, 0xfc, 0x0f, 0x00, 0x00}, Why::PageCross, 4092, 0},
	    {"movaps xmm0,[rdi+4] misaligned", {0x0f, 0x28, 0x47, 0x04}, Why::Misaligned, 4, 0},
	    {"movdqa xmm0,[rdi+8] misaligned", {0x66, 0x0f, 0x6f, 0x47, 0x08}, Why::Misaligned, 8, 0},
	    {"mov eax,[rdi] but the fault is elsewhere", {0x8b, 0x07}, Why::WrongAddress, 8, 0},
	    {"movsb string", {0xa4}, Why::NotALoad, 0, 0},
	    {"movsd string", {0xa5}, Why::NotALoad, 0, 0},
	    {"movlps xmm0,[rdi]", {0x0f, 0x12, 0x07}, Why::NotALoad, 0, 0},
	    {"pxor xmm0,[rdi]", {0x66, 0x0f, 0xef, 0x07}, Why::NotALoad, 0, 0},
	    {"lddqu xmm0,[rdi]", {0xf2, 0x0f, 0xf0, 0x07}, Why::NotALoad, 0, 0},
	};
	for (const auto& c: cases) {
		std::vector<uint8_t> code = c.code;
		code.resize(code.size() + 16, 0x90);
		Fake fake;
		fake.mc.__ss.__rip = reinterpret_cast<uint64_t>(code.data());
		g_base             = 0x200000; // 16-byte aligned, the backing covers only [g_base, g_base + 64)
		fake.mc.__ss.__rdi = g_base;
		Attempt attempt;
		char    text[160];
		const bool ok = Prepare(&fake.uc, g_base + c.offset, TestRead, &attempt);
		std::snprintf(text, sizeof(text), "%s refused (why=%d, want %d)", c.name, static_cast<int>(attempt.plan.why), static_cast<int>(c.why));
		Check(!ok && attempt.plan.why == c.why, text);
		Check(fake.mc.__ss.__rip == reinterpret_cast<uint64_t>(code.data()), "a refused load leaves RIP alone");
		Check(fake.mc.__ss.__rax == 0, "a refused load leaves the registers alone");
	}
	Check(Decode(nullptr, 0).why == Why::NoContext, "a null context is refused");
	ucontext_t no_mcontext {};
	Check(Decode(&no_mcontext, 0).why == Why::NoContext, "a context without mcontext is refused");
}

// S427: a saved context without the AVX layout cannot take the zeroed upper half; the VEX load is a fallback, the legacy one is not.
void TestVexNeedsAvxContext() {
	FillBacking();
	const std::vector<uint8_t> vex_code = {0xc5, 0xfa, 0x10, 0x07, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
	const std::vector<uint8_t> sse_code = {0xf3, 0x0f, 0x10, 0x07, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
	g_base                              = 0x400000;
	for (const bool vex: {true, false}) {
		Fake fake(false);
		fake.mc.__ss.__rip = reinterpret_cast<uint64_t>((vex ? vex_code : sse_code).data());
		fake.mc.__ss.__rdi = g_base;
		Attempt attempt;
		const bool ok = Prepare(&fake.uc, g_base, TestRead, &attempt);
		if (vex) {
			Check(!ok && attempt.plan.why == Why::VexNoAvx, "VEX load without the AVX context layout: refused with VexNoAvx");
			Check(fake.mc.__ss.__rip == reinterpret_cast<uint64_t>(vex_code.data()), "refused VEX load leaves RIP alone");
		} else {
			Check(ok, "a legacy SSE load does not need the AVX context layout");
		}
	}
}

void TestDiagText() {
	FillBacking();
	std::vector<uint8_t> code = {0x3b, 0x07, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
	Fake                 fake;
	fake.mc.__ss.__rip = reinterpret_cast<uint64_t>(code.data());
	fake.mc.__ss.__rdi = 0x300000;
	const auto plan    = Decode(&fake.uc, 0x300000);
	Check(std::strcmp(plan.mnemonic, "cmp") == 0, "the DIAG line names the mnemonic of a refused form");
	Check(std::strcmp(plan.operands, "reg32,mem32") == 0, "the DIAG line names the operand shape");
}

void TestGate() {
	const auto may = [](Mode mode, bool read, bool gpu, bool dirty, bool inflight) {
		return MayEmulate(Gate {mode, read, gpu, dirty, inflight});
	};
	Check(may(Mode::On, true, false, false, false), "clean bytes, guest read: emulate");
	Check(may(Mode::Shadow, true, false, false, false), "shadow takes the same decision");
	Check(!may(Mode::Off, true, false, false, false), "off never emulates");
	Check(!may(Mode::On, false, false, false, false), "a write is never emulated");
	Check(!may(Mode::On, true, true, false, false), "the command processor thread is never emulated");
	Check(!may(Mode::On, true, false, true, false), "GPU-dirty bytes are not emulated");
	Check(!may(Mode::On, true, false, false, true), "bytes with a GPU->host copy in flight are not emulated");
	Check(ParseMode(nullptr) == Mode::Off && ParseMode("") == Mode::Off && ParseMode("0") == Mode::Off, "0 / unset / empty is off");
	Check(ParseMode("1") == Mode::On && ParseMode("shadow") == Mode::Shadow && ParseMode("bogus") == Mode::Off, "1 / shadow / junk");
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Real SIGSEGVs.
std::atomic<int> g_emulated {0};
std::atomic<int> g_fallback {0};
uint8_t*         g_page = nullptr;

bool Handler(const ExceptionInfo& info) {
	if (info.type != ExceptionType::AccessViolation || info.access_violation_type != AccessViolationType::Read) {
		return false;
	}
	Attempt attempt;
	if (Prepare(info.native_context, info.access_violation_vaddr, TestRead, &attempt)) {
		Commit(info.native_context, attempt.plan, attempt.bytes);
		g_emulated.fetch_add(1);
		return true;
	}
	// The ordinary path: the drain made the bytes current and released the page.
	g_fallback.fetch_add(1);
	return ::mprotect(g_page, kPageSize, PROT_READ) == 0;
}

// rax = -1, xmm0 = all ones, CF = 1; <insn>; store rax, xmm0 and CF.
std::vector<uint8_t> Wrap(const std::vector<uint8_t>& insn) {
	std::vector<uint8_t> code = {0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax,-1
	                             0x66, 0x0f, 0x76, 0xc0,                   // pcmpeqd xmm0,xmm0
	                             0xf9};                                    // stc
	code.insert(code.end(), insn.begin(), insn.end());
	const std::vector<uint8_t> tail = {0x0f, 0x92, 0xc1,             // setc cl
	                                   0x48, 0x89, 0x06,             // mov [rsi],rax
	                                   0xf3, 0x0f, 0x7f, 0x46, 0x10, // movdqu [rsi+16],xmm0
	                                   0x88, 0x4e, 0x20,             // mov [rsi+32],cl
	                                   0xc3};
	code.insert(code.end(), tail.begin(), tail.end());
	return code;
}

// Same, with a 256-bit view: ymm0 = all ones before (VEX vpcmpeqd), vmovdqu ymm0 stored after. Out layout: rax at 0, ymm0 at 16..47, CF at 48.
std::vector<uint8_t> WrapYmm(const std::vector<uint8_t>& insn) {
	std::vector<uint8_t> code = {0x48, 0xc7, 0xc0, 0xff, 0xff, 0xff, 0xff, // mov rax,-1
	                             0xc5, 0xfd, 0x76, 0xc0,                   // vpcmpeqd ymm0,ymm0,ymm0
	                             0xf9};                                    // stc
	code.insert(code.end(), insn.begin(), insn.end());
	const std::vector<uint8_t> tail = {0x0f, 0x92, 0xc1,             // setc cl
	                                   0x48, 0x89, 0x06,             // mov [rsi],rax
	                                   0xc5, 0xfe, 0x7f, 0x46, 0x10, // vmovdqu [rsi+16],ymm0
	                                   0x88, 0x4e, 0x30,             // mov [rsi+48],cl
	                                   0xc5, 0xf8, 0x77,             // vzeroupper
	                                   0xc3};
	code.insert(code.end(), tail.begin(), tail.end());
	return code;
}

using Fn = void (*)(const void* data, uint8_t* out);

Fn MapCode(const std::vector<uint8_t>& code) {
	void* page = ::mmap(nullptr, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	Check(page != MAP_FAILED, "map code");
	std::memcpy(page, code.data(), code.size());
	Check(::mprotect(page, kPageSize, PROT_READ | PROT_EXEC) == 0, "code executable");
	return reinterpret_cast<Fn>(page);
}

void ArmPage() {
	// The page holds the "raw" bytes (what a guest would read after the drain): the complement of the backing.
	Check(::mprotect(g_page, kPageSize, PROT_READ | PROT_WRITE) == 0, "page writable");
	for (size_t i = 0; i < g_backing.size(); ++i) {
		g_page[i] = static_cast<uint8_t>(~g_backing[i]);
	}
	Check(::mprotect(g_page, kPageSize, PROT_NONE) == 0, "page protected");
}

struct Real {
	const char*          name;
	std::vector<uint8_t> insn;
	Kind                 kind;
	unsigned             size;
	unsigned             dst_bytes;
	bool                 sign;
};

void TestRealFaults() {
	FillBacking();
	g_base = reinterpret_cast<uint64_t>(g_page);
	const std::vector<Real> reals = {
	    {"mov eax,[rdi]", {0x8b, 0x07}, Kind::Gpr, 4, 4, false},
	    {"mov rax,[rdi]", {0x48, 0x8b, 0x07}, Kind::Gpr, 8, 8, false},
	    {"mov ax,[rdi]", {0x66, 0x8b, 0x07}, Kind::Gpr, 2, 2, false},
	    {"mov al,[rdi]", {0x8a, 0x07}, Kind::Gpr, 1, 1, false},
	    {"movzx eax,byte [rdi]", {0x0f, 0xb6, 0x07}, Kind::GprZext, 1, 4, false},
	    {"movsx rax,word [rdi]", {0x48, 0x0f, 0xbf, 0x07}, Kind::GprSext, 2, 8, true},
	    {"movsxd rax,dword [rdi]", {0x48, 0x63, 0x07}, Kind::GprSext, 4, 8, true},
	    {"movss xmm0,[rdi]", {0xf3, 0x0f, 0x10, 0x07}, Kind::XmmScalar, 4, 0, false},
	    {"movsd xmm0,[rdi]", {0xf2, 0x0f, 0x10, 0x07}, Kind::XmmScalar, 8, 0, false},
	    {"movd xmm0,[rdi]", {0x66, 0x0f, 0x6e, 0x07}, Kind::XmmScalar, 4, 0, false},
	    {"movq xmm0,[rdi]", {0xf3, 0x0f, 0x7e, 0x07}, Kind::XmmScalar, 8, 0, false},
	    {"movups xmm0,[rdi]", {0x0f, 0x10, 0x07}, Kind::Xmm128, 16, 0, false},
	    {"movdqu xmm0,[rdi]", {0xf3, 0x0f, 0x6f, 0x07}, Kind::Xmm128, 16, 0, false},
	    {"movaps xmm0,[rdi]", {0x0f, 0x28, 0x07}, Kind::Xmm128, 16, 0, false},
	    {"movdqa xmm0,[rdi]", {0x66, 0x0f, 0x6f, 0x07}, Kind::Xmm128, 16, 0, false},
	};
	for (const auto& r: reals) {
		Fn fn = MapCode(Wrap(r.insn));
		for (int round = 0; round < 3; ++round) {
			ArmPage();
			g_emulated = 0;
			g_fallback = 0;
			alignas(16) uint8_t out[48] = {};
			fn(g_page, out);
			char text[160];
			std::snprintf(text, sizeof(text), "%s round %d: served by the emulator (emulated=%d fallback=%d)", r.name, round, g_emulated.load(),
			              g_fallback.load());
			Check(g_emulated == 1 && g_fallback == 0, text);
			// Not unprotected: the next run of the same load faults again.
			alignas(16) uint8_t again[48] = {};
			fn(g_page, again);
			std::snprintf(text, sizeof(text), "%s: the page stays protected (second run emulated=%d)", r.name, g_emulated.load());
			Check(g_emulated == 2 && g_fallback == 0, text);
			Check(std::memcmp(out, again, sizeof(out)) == 0, r.name);
			uint64_t rax = 0;
			std::memcpy(&rax, out, 8);
			uint64_t expect = ~uint64_t {0};
			uint8_t  xmm[16];
			std::memset(xmm, 0xff, sizeof(xmm));
			if (r.kind == Kind::XmmScalar) {
				std::memset(xmm, 0, 16);
				std::memcpy(xmm, g_backing.data(), r.size);
			} else if (r.kind == Kind::Xmm128) {
				std::memcpy(xmm, g_backing.data(), 16);
			} else {
				uint64_t v = r.sign ? Sx(g_backing.data(), r.size) : LoadLe(g_backing.data(), r.size);
				switch (r.dst_bytes) {
					case 1: expect = (expect & ~uint64_t {0xff}) | (v & 0xff); break;
					case 2: expect = (expect & ~uint64_t {0xffff}) | (v & 0xffff); break;
					case 4: expect = v & 0xffffffffull; break;
					default: expect = v; break;
				}
			}
			std::snprintf(text, sizeof(text), "%s: rax == backing-derived value (got 0x%llx want 0x%llx)", r.name,
			              static_cast<unsigned long long>(rax), static_cast<unsigned long long>(expect));
			Check(rax == expect, text);
			Check(std::memcmp(out + 16, xmm, 16) == 0, r.name);
			Check(out[32] == 1, "the flags (CF) are untouched by the emulated load");
			// And the value is the backing, not what the page holds.
			Check(std::memcmp(g_backing.data(), g_page, 4) != 0, "page and backing differ");
		}
	}
}

// S427: the VEX.128 loads on a real fault: the xmm half holds the backing, the upper half of ymm0 (all ones before) is ZERO; the legacy SSE
// load of the same bytes leaves the upper half alone. RIP, flags and the protected page as in TestRealFaults.
void TestRealVexFaults() {
	FillBacking();
	g_base = reinterpret_cast<uint64_t>(g_page);
	struct Item {
		const char*          name;
		std::vector<uint8_t> insn;
		unsigned             size;
		bool                 scalar;
		bool                 vex;
	};
	const std::vector<Item> items = {
	    {"vmovss xmm0,[rdi]", {0xc5, 0xfa, 0x10, 0x07}, 4, true, true},
	    {"vmovsd xmm0,[rdi]", {0xc5, 0xfb, 0x10, 0x07}, 8, true, true},
	    {"vmovd xmm0,[rdi]", {0xc5, 0xf9, 0x6e, 0x07}, 4, true, true},
	    {"vmovq xmm0,[rdi]", {0xc5, 0xfa, 0x7e, 0x07}, 8, true, true},
	    {"vmovups xmm0,[rdi]", {0xc5, 0xf8, 0x10, 0x07}, 16, false, true},
	    {"vmovupd xmm0,[rdi]", {0xc5, 0xf9, 0x10, 0x07}, 16, false, true},
	    {"vmovdqu xmm0,[rdi]", {0xc5, 0xfa, 0x6f, 0x07}, 16, false, true},
	    {"vmovaps xmm0,[rdi]", {0xc5, 0xf8, 0x28, 0x07}, 16, false, true},
	    {"vmovapd xmm0,[rdi]", {0xc5, 0xf9, 0x28, 0x07}, 16, false, true},
	    {"vmovdqa xmm0,[rdi]", {0xc5, 0xf9, 0x6f, 0x07}, 16, false, true},
	    {"movss xmm0,[rdi] (legacy keeps the upper half)", {0xf3, 0x0f, 0x10, 0x07}, 4, true, false},
	    {"movdqu xmm0,[rdi] (legacy keeps the upper half)", {0xf3, 0x0f, 0x6f, 0x07}, 16, false, false},
	};
	for (const auto& item: items) {
		Fn fn = MapCode(WrapYmm(item.insn));
		for (int round = 0; round < 2; ++round) {
			ArmPage();
			g_emulated = 0;
			g_fallback = 0;
			alignas(32) uint8_t out[64] = {};
			fn(g_page, out);
			char text[160];
			std::snprintf(text, sizeof(text), "%s round %d: served by the emulator (emulated=%d fallback=%d)", item.name, round,
			              g_emulated.load(), g_fallback.load());
			Check(g_emulated == 1 && g_fallback == 0, text);
			uint8_t lo[16];
			uint8_t hi[16];
			std::memset(lo, 0xff, sizeof(lo));
			std::memset(hi, item.vex ? 0x00 : 0xff, sizeof(hi));
			if (item.scalar) {
				std::memset(lo, 0, sizeof(lo));
				std::memcpy(lo, g_backing.data(), item.size);
			} else {
				std::memcpy(lo, g_backing.data(), 16);
			}
			std::snprintf(text, sizeof(text), "%s: low half of ymm0 == backing", item.name);
			Check(std::memcmp(out + 16, lo, 16) == 0, text);
			std::snprintf(text, sizeof(text), "%s: bits 128..255 of ymm0 are %s", item.name, item.vex ? "zero" : "untouched");
			Check(std::memcmp(out + 32, hi, 16) == 0, text);
			uint64_t rax = 0;
			std::memcpy(&rax, out, 8);
			Check(rax == ~uint64_t {0}, "rax untouched by an xmm load");
			Check(out[48] == 1, "the flags (CF) are untouched by the emulated load");
		}
	}
	// VEX.L = 1 (a 256-bit load) is a fallback: the instruction re-executes against the released page and loads the page bytes.
	{
		Fn fn = MapCode(WrapYmm({0xc5, 0xfe, 0x6f, 0x07})); // vmovdqu ymm0,[rdi]
		ArmPage();
		g_emulated = 0;
		g_fallback = 0;
		alignas(32) uint8_t out[64] = {};
		fn(g_page, out);
		Check(g_emulated == 0 && g_fallback == 1, "vmovdqu ymm0,[rdi]: the 256-bit form takes the ordinary path");
		uint8_t raw[16];
		for (size_t i = 0; i < 16; ++i) {
			raw[i] = static_cast<uint8_t>(~g_backing[i]);
		}
		Check(std::memcmp(out + 16, raw, 16) == 0, "vmovdqu ymm0: the page bytes (ordinary path), not the backing");
	}
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// S428 (B1): `mov <byte reg>,[rdi]` with a REX prefix for every encoding number 0..15: AL CL DL BL SPL BPL SIL DIL R8B..R15B.
std::vector<uint8_t> MovByteRegInsn(unsigned n) {
	return {static_cast<uint8_t>(0x40 | (n >= 8 ? 0x04 : 0x00)), 0x8a, static_cast<uint8_t>(((n & 7u) << 3) | 0x07)};
}

uint64_t Pattern(unsigned r) {
	return 0x5a5a5a5a5a5a5a00ull + (r * 3u + 1u);
}

void TestGpr8Fake() {
	for (unsigned n = 0; n < 16; ++n) {
		FillBacking();
		std::vector<uint8_t> code = MovByteRegInsn(n);
		code.resize(code.size() + 16, 0x90);
		Fake fake;
		fake.mc.__ss.__rip = reinterpret_cast<uint64_t>(code.data());
		g_base             = 0x100000;
		uint64_t before[16];
		for (unsigned r = 0; r < 16; ++r) {
			fake.Reg(static_cast<uint8_t>(r)) = Pattern(r);
		}
		fake.Reg(7) = g_base; // rdi is the base register
		for (unsigned r = 0; r < 16; ++r) {
			before[r] = fake.Reg(static_cast<uint8_t>(r));
		}
		Attempt attempt;
		char    text[160];
		const bool ok = Prepare(&fake.uc, g_base, TestRead, &attempt);
		std::snprintf(text, sizeof(text), "mov byte reg #%u: Prepare accepts (why=%d)", n, static_cast<int>(attempt.plan.why));
		Check(ok, text);
		std::snprintf(text, sizeof(text), "mov byte reg #%u: the destination is register %u (got %u)", n, n, attempt.plan.dst);
		Check(attempt.plan.dst == n && attempt.plan.dst_bytes == 1 && attempt.plan.size == 1, text);
		Commit(&fake.uc, attempt.plan, attempt.bytes);
		for (unsigned r = 0; r < 16; ++r) {
			const uint64_t expect = r == n ? ((before[r] & ~uint64_t {0xff}) | g_backing[0]) : before[r];
			std::snprintf(text, sizeof(text), "mov byte reg #%u: r%u == 0x%llx (got 0x%llx)", n, r, static_cast<unsigned long long>(expect),
			              static_cast<unsigned long long>(fake.Reg(static_cast<uint8_t>(r))));
			Check(fake.Reg(static_cast<uint8_t>(r)) == expect, text);
		}
		Check(fake.mc.__ss.__rip == reinterpret_cast<uint64_t>(code.data()) + 3, "mov byte reg: RIP moves by the instruction length");
	}
	// Without REX the numbers 4..7 are AH..BH: still a fallback.
	{
		std::vector<uint8_t> code = {0x8a, 0x27, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90};
		Fake                 fake;
		fake.mc.__ss.__rip = reinterpret_cast<uint64_t>(code.data());
		fake.mc.__ss.__rdi = 0x100000;
		g_base             = 0x100000;
		Attempt attempt;
		Check(!Prepare(&fake.uc, g_base, TestRead, &attempt) && attempt.plan.why == Why::HighByteReg, "mov ah,[rdi] stays a fallback");
	}
}

// A function that saves all 16 registers, loads a pattern into all but rsp (rdi = the protected page), runs `insn`, stores all 16, restores the
// callee-saved ones and rsp. Two pages: the code, then the data (A = saved at entry, B = after `insn`).
struct RegProbe {
	uint8_t*  mem = nullptr;
	uint64_t* a   = nullptr;
	uint64_t* b   = nullptr;
	void (*fn)() = nullptr;
};

void Emit32(std::vector<uint8_t>& c, int32_t v) {
	for (int i = 0; i < 4; ++i) {
		c.push_back(static_cast<uint8_t>(static_cast<uint32_t>(v) >> (8 * i)));
	}
}

RegProbe BuildRegProbe(const std::vector<uint8_t>& insn, uint64_t rdi_value) {
	RegProbe probe;
	void*    mem = ::mmap(nullptr, 2 * kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	Check(mem != MAP_FAILED, "map probe");
	probe.mem = static_cast<uint8_t*>(mem);
	probe.a   = reinterpret_cast<uint64_t*>(probe.mem + kPageSize);
	probe.b   = probe.a + 16;
	std::vector<uint8_t> c;
	const auto           rip_rel = [&](uint8_t rex, uint8_t opcode, unsigned reg, uint64_t* slot) {
        c.push_back(static_cast<uint8_t>(rex | (reg >= 8 ? 0x04 : 0x00)));
        c.push_back(opcode);
        c.push_back(static_cast<uint8_t>(((reg & 7u) << 3) | 0x05));
        const int64_t disp = reinterpret_cast<uint8_t*>(slot) - (probe.mem + c.size() + 4);
        Emit32(c, static_cast<int32_t>(disp));
	};
	for (unsigned r = 0; r < 16; ++r) {
		rip_rel(0x48, 0x89, r, probe.a + r); // mov [A+r],reg
	}
	for (unsigned r = 0; r < 16; ++r) {
		if (r == 4) {
			continue; // rsp keeps the real stack: a signal handler runs on it
		}
		c.push_back(static_cast<uint8_t>(0x48 | (r >= 8 ? 0x01 : 0x00)));
		c.push_back(static_cast<uint8_t>(0xb8 + (r & 7u)));
		const uint64_t value = r == 7 ? rdi_value : Pattern(r);
		for (int i = 0; i < 8; ++i) {
			c.push_back(static_cast<uint8_t>(value >> (8 * i)));
		}
	}
	c.insert(c.end(), insn.begin(), insn.end());
	for (unsigned r = 0; r < 16; ++r) {
		rip_rel(0x48, 0x89, r, probe.b + r); // mov [B+r],reg
	}
	for (const unsigned r: {4u, 3u, 5u, 12u, 13u, 14u, 15u}) {
		rip_rel(0x48, 0x8b, r, probe.a + r); // mov reg,[A+r]: rsp first, then the callee-saved registers
	}
	c.push_back(0xc3);
	Check(c.size() < kPageSize, "probe fits a page");
	std::memcpy(probe.mem, c.data(), c.size());
	Check(::mprotect(probe.mem, kPageSize, PROT_READ | PROT_EXEC) == 0, "probe code executable");
	probe.fn = reinterpret_cast<void (*)()>(probe.mem);
	return probe;
}

void TestGpr8Real() {
	FillBacking();
	g_base = reinterpret_cast<uint64_t>(g_page);
	for (unsigned n = 0; n < 16; ++n) {
		const RegProbe probe = BuildRegProbe(MovByteRegInsn(n), reinterpret_cast<uint64_t>(g_page));
		for (const bool protect: {true, false}) {
			if (protect) {
				ArmPage();
			} else { // native: the page holds the backing bytes, no fault
				Check(::mprotect(g_page, kPageSize, PROT_READ | PROT_WRITE) == 0, "page writable");
				std::memcpy(g_page, g_backing.data(), g_backing.size());
				Check(::mprotect(g_page, kPageSize, PROT_READ) == 0, "page readable");
			}
			g_emulated = 0;
			g_fallback = 0;
			probe.fn();
			char text[200];
			std::snprintf(text, sizeof(text), "register #%u %s: served by the emulator (emulated=%d fallback=%d)", n, protect ? "emulated" : "native",
			              g_emulated.load(), g_fallback.load());
			Check(protect ? (g_emulated == 1 && g_fallback == 0) : (g_emulated == 0 && g_fallback == 0), text);
			for (unsigned r = 0; r < 16; ++r) {
				uint64_t start = r == 4 ? probe.a[4] : (r == 7 ? reinterpret_cast<uint64_t>(g_page) : Pattern(r));
				uint64_t expect = r == n ? ((start & ~uint64_t {0xff}) | g_backing[0]) : start;
				std::snprintf(text, sizeof(text), "register #%u %s: r%u == 0x%llx (got 0x%llx)", n, protect ? "emulated" : "native", r,
				              static_cast<unsigned long long>(expect), static_cast<unsigned long long>(probe.b[r]));
				Check(probe.b[r] == expect, text);
			}
		}
		::munmap(probe.mem, 2 * kPageSize);
	}
}

// S428: the reasons of a fallback, and the mapping of the backing result onto them.
void TestFallbackReasons() {
	Check(ClassifyFallback(Why::BackingDirty) == Fallback::Dirty, "dirty bytes -> dirty");
	Check(ClassifyFallback(Why::BackingInflight) == Fallback::Inflight, "in-flight bytes -> inflight");
	Check(ClassifyFallback(Why::LockBusy) == Fallback::LockBusy, "busy lock -> lockbusy");
	Check(ClassifyFallback(Why::Vex) == Fallback::Vex, "VEX.L = 1 / EVEX -> vex");
	for (const Why why : {Why::NotALoad, Why::Prefix, Why::HighByteReg, Why::Size, Why::Misaligned, Why::PageCross, Why::WrongAddress, Why::DecodeFailed,
	                      Why::BackingRefused, Why::VexNoAvx, Why::NoContext, Why::CodeUnreadable}) {
		Check(ClassifyFallback(why) == Fallback::Form, "everything else -> form");
	}
	static BackingResult g_result;
	g_result = BackingResult::Ok;
	const auto fake_read = [](uint64_t, uint64_t, void* dst) {
		std::memset(dst, 0, 1);
		return g_result;
	};
	const std::pair<BackingResult, Why> expectations[] = {{BackingResult::Dirty, Why::BackingDirty},       {BackingResult::Inflight, Why::BackingInflight},
	                                                      {BackingResult::LockBusy, Why::LockBusy},         {BackingResult::Unavailable, Why::BackingRefused}};
	for (const auto& [result, why]: expectations) {
		std::vector<uint8_t> code = MovByteRegInsn(0);
		code.resize(code.size() + 16, 0x90);
		Fake fake;
		fake.mc.__ss.__rip = reinterpret_cast<uint64_t>(code.data());
		fake.mc.__ss.__rdi = 0x100000;
		g_result           = result;
		Attempt attempt;
		Check(!Prepare(&fake.uc, 0x100000, fake_read, &attempt) && attempt.plan.why == why, "a backing result maps onto its reason");
		Check(fake.mc.__ss.__rip == reinterpret_cast<uint64_t>(code.data()), "a refused backing read leaves RIP alone");
	}
	Attempt attempt;
	Check(!Prepare(nullptr, 0, nullptr, &attempt) && attempt.plan.why == Why::NoContext, "no context");
}

// S428: every form gives the same result as the native instruction on the same bytes (no protection, the page holds the backing).
void TestNativeParity() {
	FillBacking();
	g_base = reinterpret_cast<uint64_t>(g_page);
	struct Item {
		const char*          name;
		std::vector<uint8_t> insn;
		bool                 ymm;
	};
	const std::vector<Item> items = {
	    {"mov eax,[rdi]", {0x8b, 0x07}, false},
	    {"mov rax,[rdi]", {0x48, 0x8b, 0x07}, false},
	    {"mov ax,[rdi]", {0x66, 0x8b, 0x07}, false},
	    {"mov al,[rdi]", {0x8a, 0x07}, false},
	    {"movzx eax,byte [rdi]", {0x0f, 0xb6, 0x07}, false},
	    {"movzx rax,word [rdi]", {0x48, 0x0f, 0xb7, 0x07}, false},
	    {"movzx ax,byte [rdi]", {0x66, 0x0f, 0xb6, 0x07}, false},
	    {"movsx eax,byte [rdi]", {0x0f, 0xbe, 0x07}, false},
	    {"movsx rax,word [rdi]", {0x48, 0x0f, 0xbf, 0x07}, false},
	    {"movsxd rax,dword [rdi]", {0x48, 0x63, 0x07}, false},
	    {"movss xmm0,[rdi]", {0xf3, 0x0f, 0x10, 0x07}, false},
	    {"movsd xmm0,[rdi]", {0xf2, 0x0f, 0x10, 0x07}, false},
	    {"movd xmm0,[rdi]", {0x66, 0x0f, 0x6e, 0x07}, false},
	    {"movq xmm0,[rdi]", {0xf3, 0x0f, 0x7e, 0x07}, false},
	    {"movups xmm0,[rdi]", {0x0f, 0x10, 0x07}, false},
	    {"movupd xmm0,[rdi]", {0x66, 0x0f, 0x10, 0x07}, false},
	    {"movdqu xmm0,[rdi]", {0xf3, 0x0f, 0x6f, 0x07}, false},
	    {"movaps xmm0,[rdi]", {0x0f, 0x28, 0x07}, false},
	    {"movapd xmm0,[rdi]", {0x66, 0x0f, 0x28, 0x07}, false},
	    {"movdqa xmm0,[rdi]", {0x66, 0x0f, 0x6f, 0x07}, false},
	    {"vmovss xmm0,[rdi]", {0xc5, 0xfa, 0x10, 0x07}, true},
	    {"vmovsd xmm0,[rdi]", {0xc5, 0xfb, 0x10, 0x07}, true},
	    {"vmovd xmm0,[rdi]", {0xc5, 0xf9, 0x6e, 0x07}, true},
	    {"vmovq xmm0,[rdi]", {0xc5, 0xfa, 0x7e, 0x07}, true},
	    {"vmovups xmm0,[rdi]", {0xc5, 0xf8, 0x10, 0x07}, true},
	    {"vmovupd xmm0,[rdi]", {0xc5, 0xf9, 0x10, 0x07}, true},
	    {"vmovdqu xmm0,[rdi]", {0xc5, 0xfa, 0x6f, 0x07}, true},
	    {"vmovaps xmm0,[rdi]", {0xc5, 0xf8, 0x28, 0x07}, true},
	    {"vmovapd xmm0,[rdi]", {0xc5, 0xf9, 0x28, 0x07}, true},
	    {"vmovdqa xmm0,[rdi]", {0xc5, 0xf9, 0x6f, 0x07}, true},
	};
	for (const auto& item: items) {
		Fn fn = MapCode(item.ymm ? WrapYmm(item.insn) : Wrap(item.insn));
		alignas(32) uint8_t native[64] = {};
		alignas(32) uint8_t emulated[64] = {};
		Check(::mprotect(g_page, kPageSize, PROT_READ | PROT_WRITE) == 0, "page writable");
		std::memcpy(g_page, g_backing.data(), g_backing.size());
		Check(::mprotect(g_page, kPageSize, PROT_READ) == 0, "page readable");
		g_emulated = 0;
		g_fallback = 0;
		fn(g_page, native);
		Check(g_emulated == 0 && g_fallback == 0, "native run does not fault");
		ArmPage();
		fn(g_page, emulated);
		char text[160];
		std::snprintf(text, sizeof(text), "%s: emulated registers == native (emulated=%d fallback=%d)", item.name, g_emulated.load(), g_fallback.load());
		Check(g_emulated == 1 && g_fallback == 0 && std::memcmp(native, emulated, sizeof(native)) == 0, text);
	}
}

void TestRealFallbacks() {
	FillBacking();
	g_base = reinterpret_cast<uint64_t>(g_page);
	// GPU-dirty bytes: the ordinary path, the guest reads the (drained) page.
	{
		Fn fn = MapCode(Wrap({0x8b, 0x07}));
		ArmPage();
		g_emulated = 0;
		g_fallback = 0;
		g_dirty    = true;
		uint8_t out[48] = {};
		fn(g_page, out);
		g_dirty = false;
		Check(g_emulated == 0 && g_fallback == 1, "dirty bytes: the load takes the ordinary path");
		uint32_t raw = 0;
		std::memcpy(&raw, g_page, 4);
		uint64_t rax = 0;
		std::memcpy(&rax, out, 8);
		Check(rax == raw, "dirty bytes: the register holds what the page holds, not the backing");
		// Same load, clean: emulated again.
		ArmPage();
		g_emulated = 0;
		g_fallback = 0;
		fn(g_page, out);
		std::memcpy(&rax, out, 8);
		Check(g_emulated == 1 && g_fallback == 0 && rax == LoadLe(g_backing.data(), 4), "clean again: emulated");
	}
	// In-flight copy: ordinary path.
	{
		Fn fn = MapCode(Wrap({0x48, 0x8b, 0x07}));
		ArmPage();
		g_emulated = 0;
		g_fallback = 0;
		g_inflight = true;
		uint8_t out[48] = {};
		fn(g_page, out);
		g_inflight = false;
		Check(g_emulated == 0 && g_fallback == 1, "in-flight bytes: the load takes the ordinary path");
	}
	// An unsupported form (cmp eax,[rdi]): the ordinary path, the instruction re-executes and the flags are the real ones.
	{
		Fn fn = MapCode(Wrap({0x3b, 0x07}));
		ArmPage();
		g_emulated = 0;
		g_fallback = 0;
		uint8_t out[48] = {};
		fn(g_page, out);
		Check(g_emulated == 0 && g_fallback == 1, "cmp eax,[rdi]: unsupported form takes the ordinary path");
		uint64_t rax = 0;
		std::memcpy(&rax, out, 8);
		Check(rax == ~uint64_t {0}, "cmp does not change rax");
	}
}

} // namespace

int main() {
	FillBacking();
	TestGate();
	TestSupportedForms();
	TestRipRelative();
	TestRefusals();
	TestVexNeedsAvxContext();
	TestDiagText();
	TestGpr8Fake();
	TestFallbackReasons();
	void* page = ::mmap(nullptr, kPageSize, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	Check(page != MAP_FAILED, "map data page");
	g_page = static_cast<uint8_t*>(page);
	Check(Common::HostException::InstallHandler(Handler), "install host exception handler");
	TestRealFaults();
	TestRealVexFaults();
	TestGpr8Real();
	TestNativeParity();
	TestRealFallbacks();
	std::printf("GuestReadEmulatorTests: all passed\n");
	return 0;
}
