#include "loader/guestReadEmulator.h"

#include "common/common.h"

#include <Zydis/Zydis.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__APPLE__)
#include <sys/ucontext.h>
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <ucontext.h>
#endif

namespace Loader::GuestReadEmulator {

#if defined(KYTY_GUESTREAD_MUTANT_EMU_WRONG_RIP)
constexpr int kLengthSkew = 1;
#else
constexpr int kLengthSkew = 0;
#endif

Mode ParseMode(const char* text) {
	if (text == nullptr) {
		return Mode::Off;
	}
	if (std::strcmp(text, "1") == 0) {
		return Mode::On;
	}
	if (std::strcmp(text, "shadow") == 0) {
		return Mode::Shadow;
	}
	return Mode::Off;
}

Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_GF_EMULATE"));
	return mode;
}

bool MayEmulate(const Gate& gate) {
	if (gate.mode == Mode::Off || !gate.read || gate.gpu_thread) {
		return false;
	}
#if defined(KYTY_GUESTREAD_MUTANT_EMU_IGNORES_DIRTY)
	return !gate.inflight;
#else
	return !gate.dirty && !gate.inflight;
#endif
}

namespace {

#if defined(__APPLE__) || KYTY_PLATFORM == KYTY_PLATFORM_LINUX
struct Regs {
#if defined(__APPLE__)
	ucontext_t* native;

	[[nodiscard]] bool     Valid() const { return native != nullptr && native->uc_mcontext != nullptr; }
	[[nodiscard]] uint64_t Rip() const { return native->uc_mcontext->__ss.__rip; }
	void                   Advance(uint64_t length) const { native->uc_mcontext->__ss.__rip += length; }
	[[nodiscard]] uint64_t& Gpr(uint8_t index) const {
		auto*     state       = &native->uc_mcontext->__ss;
		uint64_t* registers[] = {&state->__rax, &state->__rcx, &state->__rdx, &state->__rbx, &state->__rsp, &state->__rbp,
		                         &state->__rsi, &state->__rdi, &state->__r8,  &state->__r9,  &state->__r10, &state->__r11,
		                         &state->__r12, &state->__r13, &state->__r14, &state->__r15};
		return *registers[index];
	}
	[[nodiscard]] void* Xmm(uint8_t index) const {
		auto* fs = &native->uc_mcontext->__fs;
		switch (index) {
			case 0: return &fs->__fpu_xmm0;
			case 1: return &fs->__fpu_xmm1;
			case 2: return &fs->__fpu_xmm2;
			case 3: return &fs->__fpu_xmm3;
			case 4: return &fs->__fpu_xmm4;
			case 5: return &fs->__fpu_xmm5;
			case 6: return &fs->__fpu_xmm6;
			case 7: return &fs->__fpu_xmm7;
			case 8: return &fs->__fpu_xmm8;
			case 9: return &fs->__fpu_xmm9;
			case 10: return &fs->__fpu_xmm10;
			case 11: return &fs->__fpu_xmm11;
			case 12: return &fs->__fpu_xmm12;
			case 13: return &fs->__fpu_xmm13;
			case 14: return &fs->__fpu_xmm14;
			case 15: return &fs->__fpu_xmm15;
			default: return nullptr;
		}
	}
#else
	ucontext_t* native;

	[[nodiscard]] bool     Valid() const { return native != nullptr; }
	[[nodiscard]] uint64_t Rip() const { return static_cast<uint64_t>(native->uc_mcontext.gregs[REG_RIP]); }
	void                   Advance(uint64_t length) const { native->uc_mcontext.gregs[REG_RIP] += static_cast<greg_t>(length); }
	[[nodiscard]] uint64_t& Gpr(uint8_t index) const {
		constexpr int registers[] = {REG_RAX, REG_RCX, REG_RDX, REG_RBX, REG_RSP, REG_RBP, REG_RSI, REG_RDI,
		                             REG_R8,  REG_R9,  REG_R10, REG_R11, REG_R12, REG_R13, REG_R14, REG_R15};
		return reinterpret_cast<uint64_t&>(native->uc_mcontext.gregs[registers[index]]);
	}
	[[nodiscard]] void* Xmm(uint8_t index) const {
		return native->uc_mcontext.fpregs == nullptr ? nullptr : native->uc_mcontext.fpregs->_xmm[index].element;
	}
#endif
};
constexpr bool kHaveRegs = true;
#else
struct Regs {
	void*                   native;
	[[nodiscard]] bool      Valid() const { return false; }
	[[nodiscard]] uint64_t  Rip() const { return 0; }
	void                    Advance(uint64_t) const {}
	[[nodiscard]] uint64_t& Gpr(uint8_t) const { static uint64_t dummy; return dummy; }
	[[nodiscard]] void*     Xmm(uint8_t) const { return nullptr; }
};
constexpr bool kHaveRegs = false;
#endif

Regs MakeRegs(void* native_context) {
	return Regs {static_cast<decltype(Regs::native)>(native_context)};
}

constexpr uint64_t kPageSize = 4096;

Plan Refuse(Plan plan, Why why) {
	plan.ok  = false;
	plan.why = why;
	return plan;
}

// "reg32,mem64" style shape of the visible operands: types and widths only, no register numbers, no addresses, no code bytes.
void SetShape(Plan& plan, const ZydisDecodedInstruction& instruction, const ZydisDecodedOperand* operands) {
	size_t used = 0;
	for (uint8_t i = 0; i < instruction.operand_count_visible && used + 2 < sizeof(plan.operands); ++i) {
		const char* type = "?";
		switch (operands[i].type) {
			case ZYDIS_OPERAND_TYPE_REGISTER:
				type = ZydisRegisterGetClass(operands[i].reg.value) == ZYDIS_REGCLASS_XMM ? "xmm" : "reg";
				break;
			case ZYDIS_OPERAND_TYPE_MEMORY: type = "mem"; break;
			case ZYDIS_OPERAND_TYPE_IMMEDIATE: type = "imm"; break;
			default: break;
		}
		const int n = std::snprintf(plan.operands + used, sizeof(plan.operands) - used, "%s%s%u", i == 0 ? "" : ",", type,
		                            static_cast<unsigned>(operands[i].size));
		if (n < 0) {
			break;
		}
		used += static_cast<size_t>(n);
	}
}

} // namespace

Plan Decode(void* native_context, uint64_t fault_address) {
	Plan plan;
	if (!kHaveRegs) {
		return Refuse(plan, Why::NoContext);
	}
	const Regs regs = MakeRegs(native_context);
	if (!regs.Valid()) {
		return Refuse(plan, Why::NoContext);
	}
	const uint64_t rip = regs.Rip();
	// The code is readable (it is executing), but the 15 bytes Zydis may look at can run into the next page, which may not be mapped.
	size_t available = ZYDIS_MAX_INSTRUCTION_LENGTH;
	if ((rip & (kPageSize - 1)) + available > kPageSize) {
		available = kPageSize - (rip & (kPageSize - 1));
	}
	ZydisDecoder            decoder {};
	ZydisDecodedInstruction instruction {};
	ZydisDecodedOperand     operands[ZYDIS_MAX_OPERAND_COUNT] {};
	if (!ZYAN_SUCCESS(ZydisDecoderInit(&decoder, ZYDIS_MACHINE_MODE_LONG_64, ZYDIS_STACK_WIDTH_64)) ||
	    !ZYAN_SUCCESS(ZydisDecoderDecodeFull(&decoder, reinterpret_cast<const void*>(rip), available, &instruction, operands))) {
		return Refuse(plan, Why::DecodeFailed);
	}
	const char* name = ZydisMnemonicGetString(instruction.mnemonic);
	std::snprintf(plan.mnemonic, sizeof(plan.mnemonic), "%s", name != nullptr ? name : "?");
	SetShape(plan, instruction, operands);

	if (instruction.encoding != ZYDIS_INSTRUCTION_ENCODING_LEGACY) {
		return Refuse(plan, Why::Vex);
	}
	if ((instruction.attributes & ZYDIS_ATTRIB_HAS_LOCK) != 0 || instruction.address_width != 64) {
		return Refuse(plan, Why::Prefix);
	}
	if (instruction.operand_count_visible != 2 || operands[0].type != ZYDIS_OPERAND_TYPE_REGISTER ||
	    operands[1].type != ZYDIS_OPERAND_TYPE_MEMORY || operands[1].mem.type != ZYDIS_MEMOP_TYPE_MEM) {
		return Refuse(plan, Why::NotALoad);
	}
	if (operands[1].mem.segment == ZYDIS_REGISTER_FS || operands[1].mem.segment == ZYDIS_REGISTER_GS) {
		return Refuse(plan, Why::Prefix);
	}

	const auto dst_reg   = operands[0].reg.value;
	const auto reg_class = ZydisRegisterGetClass(dst_reg);
	const int  reg_id    = ZydisRegisterGetId(dst_reg);
	if (reg_id < 0 || reg_id > 15) {
		return Refuse(plan, Why::NotALoad);
	}
	const bool dst_is_gpr = reg_class == ZYDIS_REGCLASS_GPR8 || reg_class == ZYDIS_REGCLASS_GPR16 || reg_class == ZYDIS_REGCLASS_GPR32 ||
	                        reg_class == ZYDIS_REGCLASS_GPR64;
	const bool dst_is_xmm = reg_class == ZYDIS_REGCLASS_XMM;
	if (dst_reg == ZYDIS_REGISTER_AH || dst_reg == ZYDIS_REGISTER_BH || dst_reg == ZYDIS_REGISTER_CH || dst_reg == ZYDIS_REGISTER_DH) {
		return Refuse(plan, Why::HighByteReg);
	}
	const uint32_t mem_bits = operands[1].size;
	if (mem_bits == 0 || mem_bits > 128 || (mem_bits & 7u) != 0) {
		return Refuse(plan, Why::Size);
	}
	plan.size   = static_cast<uint8_t>(mem_bits / 8);
	plan.dst    = static_cast<uint8_t>(reg_id);
	plan.length = static_cast<uint8_t>(instruction.length + kLengthSkew);

	bool aligned_only = false;
	switch (instruction.mnemonic) {
		case ZYDIS_MNEMONIC_MOV:
			if (!dst_is_gpr || mem_bits != operands[0].size) {
				return Refuse(plan, Why::NotALoad);
			}
			plan.kind = Kind::Gpr;
			break;
		case ZYDIS_MNEMONIC_MOVZX:
		case ZYDIS_MNEMONIC_MOVSX:
		case ZYDIS_MNEMONIC_MOVSXD:
			if (!dst_is_gpr || mem_bits >= operands[0].size) {
				return Refuse(plan, Why::NotALoad);
			}
			plan.kind = instruction.mnemonic == ZYDIS_MNEMONIC_MOVZX ? Kind::GprZext : Kind::GprSext;
			break;
		case ZYDIS_MNEMONIC_MOVSS:
		case ZYDIS_MNEMONIC_MOVSD:
		case ZYDIS_MNEMONIC_MOVD:
		case ZYDIS_MNEMONIC_MOVQ:
			if (!dst_is_xmm || (mem_bits != 32 && mem_bits != 64)) {
				return Refuse(plan, Why::NotALoad);
			}
			plan.kind = Kind::XmmScalar;
			break;
		case ZYDIS_MNEMONIC_MOVAPS:
		case ZYDIS_MNEMONIC_MOVAPD:
		case ZYDIS_MNEMONIC_MOVDQA: aligned_only = true; [[fallthrough]];
		case ZYDIS_MNEMONIC_MOVUPS:
		case ZYDIS_MNEMONIC_MOVUPD:
		case ZYDIS_MNEMONIC_MOVDQU:
			if (!dst_is_xmm || mem_bits != 128) {
				return Refuse(plan, Why::NotALoad);
			}
			plan.kind = Kind::Xmm128;
			break;
		default: return Refuse(plan, Why::NotALoad);
	}
	plan.dst_bytes = dst_is_gpr ? static_cast<uint8_t>(operands[0].size / 8) : 16;

	ZydisRegisterContext registers {};
	for (uint8_t i = 0; i < 16; ++i) {
		const uint64_t value                     = regs.Gpr(i);
		registers.values[ZYDIS_REGISTER_RAX + i] = value;
		registers.values[ZYDIS_REGISTER_EAX + i] = static_cast<uint32_t>(value);
	}
	uint64_t address = 0;
	if (!ZYAN_SUCCESS(ZydisCalcAbsoluteAddressEx(&instruction, &operands[1], rip, &registers, &address))) {
		return Refuse(plan, Why::DecodeFailed);
	}
	plan.address = address;
	if (aligned_only && (address & 15u) != 0) {
		return Refuse(plan, Why::Misaligned);
	}
	if ((address & (kPageSize - 1)) + plan.size > kPageSize) {
		return Refuse(plan, Why::PageCross);
	}
	if (fault_address < address || fault_address >= address + plan.size) {
		return Refuse(plan, Why::WrongAddress);
	}
	plan.ok  = true;
	plan.why = Why::None;
	return plan;
}

uint64_t GprResult(const Plan& plan, const uint8_t* bytes, uint64_t old) {
	uint64_t value = 0;
	std::memcpy(&value, bytes, plan.size < 8 ? plan.size : 8);
	if (plan.kind == Kind::GprSext && plan.size < 8) {
		const uint64_t sign = uint64_t {1} << (plan.size * 8 - 1);
		if ((value & sign) != 0) {
			value |= ~((sign << 1) - 1);
		}
	}
	switch (plan.dst_bytes) {
		case 1: return (old & ~uint64_t {0xff}) | (value & 0xff);
		case 2: return (old & ~uint64_t {0xffff}) | (value & 0xffff);
		case 4: return value & 0xffffffffull; // a 32-bit destination zeroes the upper half
		default: return value;
	}
}

void Commit(void* native_context, const Plan& plan, const uint8_t* bytes) {
	const Regs regs = MakeRegs(native_context);
	if (!regs.Valid() || !plan.ok) {
		return;
	}
	switch (plan.kind) {
		case Kind::Gpr:
		case Kind::GprZext:
		case Kind::GprSext: regs.Gpr(plan.dst) = GprResult(plan, bytes, regs.Gpr(plan.dst)); break;
		case Kind::XmmScalar:
		case Kind::Xmm128:
			if (void* xmm = regs.Xmm(plan.dst); xmm != nullptr) {
				uint8_t value[16] = {};
				std::memcpy(value, bytes, plan.size);
				std::memcpy(xmm, value, sizeof(value));
			}
			break;
		case Kind::None: return;
	}
	regs.Advance(plan.length);
}

bool Prepare(void* native_context, uint64_t fault_address, ReadBackingFn read, Attempt* attempt) {
	attempt->plan = Decode(native_context, fault_address);
	if (!attempt->plan.ok) {
		return false;
	}
	if (read == nullptr || !read(attempt->plan.address, attempt->plan.size, attempt->bytes)) {
		attempt->plan = Refuse(attempt->plan, Why::BackingRefused);
		return false;
	}
	return true;
}

} // namespace Loader::GuestReadEmulator
