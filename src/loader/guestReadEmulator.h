#ifndef KYTY_LOADER_GUEST_READ_EMULATOR_H_
#define KYTY_LOADER_GUEST_READ_EMULATOR_H_

// S426 (KYTY_GF_EMULATE=0|1|shadow): a guest read fault on a protected page whose bytes are clean in the backing is served by executing the
// faulting load in the signal handler: the bytes come from the backing, go to the destination register of the saved context, RIP moves past
// the instruction, and the page stays protected (no drain). Only plain loads of <= 16 bytes that do not cross a page are handled; everything
// else is a fallback to the ordinary fault path. S427: the VEX.128 forms of the xmm loads (vmovss/sd/d/q/ups/upd/dqu/aps/apd/dqa, VEX.L = 0,
// vvvv = 1111b) are handled too; they also zero bits 128..255 of the register, which needs the AVX layout of the saved context (under Rosetta the
// signal context is always _STRUCT_MCONTEXT_AVX64, uc_mcsize = 1032); without it the load is a fallback (Why::VexNoAvx).
//
// This file is the decoder / executor. It knows nothing about the GPU: the caller supplies the function that reads the backing (and refuses
// when the bytes are GPU-dirty or in flight). tests/GuestReadEmulatorTests.cpp pins it with real SIGSEGVs.

#include <cstddef>
#include <cstdint>

namespace Loader::GuestReadEmulator {

enum class Mode : uint8_t { Off, On, Shadow };

// Pure parse of the KYTY_GF_EMULATE value: null/""/"0" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
[[nodiscard]] Mode ParseMode(const char* text);
// KYTY_GF_EMULATE, read once.
[[nodiscard]] Mode EnvMode();

enum class Kind : uint8_t {
	None,
	Gpr,        // MOV r, m: 8/16-bit keep the rest of the register, 32-bit zero-extends, 64-bit
	GprZext,    // MOVZX
	GprSext,    // MOVSX / MOVSXD
	XmmScalar,  // MOVSS / MOVSD / MOVD / MOVQ xmm, m: the rest of the register is zeroed
	Xmm128,     // MOVUPS / MOVUPD / MOVDQU / MOVAPS / MOVAPD / MOVDQA xmm, m128
};

enum class Why : uint8_t {
	None,
	NoContext,
	CodeUnreadable,
	DecodeFailed,
	NotALoad,       // not one of the supported mnemonics / operand shapes
	Prefix,         // LOCK, REP, a segment override (fs/gs), 32-bit address size
	Vex,            // EVEX / XOP, or a VEX form that is not VEX.128 with vvvv = 1111b (VEX.L = 1 is a fallback, vvvv != 1111b is #UD)
	VexNoAvx,       // VEX.128 load, but the saved context has no upper YMM halves to zero (S427, gf_emu_vex_noavx_n)
	HighByteReg,    // AH/BH/CH/DH
	Size,           // memory operand wider than 16 bytes or zero
	Misaligned,     // aligned-only form on a misaligned address
	PageCross,      // the access crosses the 4 KiB page of the fault
	WrongAddress,   // the effective address does not contain the fault address
	BackingRefused, // the caller refused to read the backing (dirty / in flight / unreadable)
};

struct Plan {
	bool     ok        = false;
	Why      why       = Why::None;
	Kind     kind      = Kind::None;
	uint8_t  length    = 0; // instruction length
	uint8_t  size      = 0; // bytes read from memory
	uint8_t  dst       = 0; // register index (GPR 0..15 in encoding order, XMM 0..15)
	uint8_t  dst_bytes = 0; // GPR kinds: width of the destination register
	bool     vex       = false; // VEX.128 form: bits 128..255 of the YMM register are zeroed (S427)
	uint64_t address   = 0; // effective address of the memory operand
	char     mnemonic[16] = {};
	char     operands[48] = {}; // "r32, m32" style shape for the DIAG line (no code bytes)
};

// Decode the instruction at the saved RIP and compute its memory operand. Does not touch the context.
[[nodiscard]] Plan Decode(void* native_context, uint64_t fault_address);

// Write the loaded bytes into the destination register and move RIP past the instruction.
void Commit(void* native_context, const Plan& plan, const uint8_t* bytes);

// For the tests: the value of the destination as the instruction would leave it (GPR kinds: 64-bit register value given `old`).
[[nodiscard]] uint64_t GprResult(const Plan& plan, const uint8_t* bytes, uint64_t old);

// The decision to take the emulation path at all, before any decoding. A read of a guest (non-CP) thread, and the bytes are clean in the backing
// (not GPU-dirty, no GPU->host copy in flight); any other combination keeps the ordinary drain.
struct Gate {
	Mode mode       = Mode::Off;
	bool read       = false; // a read access (not write / execute)
	bool gpu_thread = false; // the faulting thread is the command processor
	bool dirty      = false; // the bytes are GPU-dirty
	bool inflight   = false; // a GPU->host copy of the bytes has not been published yet
};
[[nodiscard]] bool MayEmulate(const Gate& gate);

// Callback: read [address, address + size) from the backing into dst; false when the bytes may not be taken from the backing.
using ReadBackingFn = bool (*)(uint64_t address, uint64_t size, void* dst);

struct Attempt {
	Plan    plan;
	uint8_t bytes[16] = {};
};

// Decode + read the backing. On true, `attempt` holds everything Commit needs; on false attempt.plan.why says why (BackingRefused when only the
// backing read failed).
[[nodiscard]] bool Prepare(void* native_context, uint64_t fault_address, ReadBackingFn read, Attempt* attempt);

} // namespace Loader::GuestReadEmulator

#endif /* KYTY_LOADER_GUEST_READ_EMULATOR_H_ */
