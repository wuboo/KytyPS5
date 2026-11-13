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
	BackingRefused, // the caller could not read the backing (range not tracked / unreadable)
	BackingDirty,    // S428: the bytes are GPU-dirty (the ordinary drain has to fetch them)
	BackingInflight, // S428: a GPU->host copy of the bytes has not been published yet
	LockBusy,        // S428: the buffer cache state is being changed by the command processor (try_lock failed); a signal handler never waits
};

// S428: the five reasons a fault is handed to the ordinary drain, as the frames.csv columns gf_emu_fb_{dirty,inflight,vex,form,lockbusy}_n.
enum class Fallback : uint8_t { Dirty, Inflight, Vex, Form, LockBusy };
// Pure: Vex = VEX.L = 1 / EVEX / vvvv != 1111b (Why::Vex), VexNoAvx stays Form (it has its own column); everything the decoder refuses is Form.
[[nodiscard]] Fallback ClassifyFallback(Why why);

// S585 (KYTY_GUEST_WAIT=clean|1): the upper YMM halves of a saved context.
//  - Present: writable in the context (the restore loads them);
//  - InitZero: the context carries the AVX area (Windows: CONTEXT_XSTATE, LocateXStateFeature(XSTATE_AVX) != null) but its XSTATE_BV bit is
//    clear: the component is in its init state, every upper half is zero, the restore re-initialises it; a VEX.128 load writes nothing there;
//  - Hardware: the context has no extended area at all (Windows CONTEXT without CONTEXT_XSTATE): the restore (NtContinue) does not load the
//    AVX state from the context, the thread keeps its current upper halves, so the load zeroes the destination's upper half in the hardware
//    register right before the handler returns (a VEX.128 vmovaps xmmD, xmmD: the low 128 bits, the only part the Win64 ABI preserves, are
//    unchanged). Legacy SSE code never writes upper halves, VEX.128 code zeroes them, 256-bit code ends with vzeroupper: nothing between the
//    handler and NtContinue makes it non-zero again. tests/GuestWaitWinTests.cpp checks it on a real fault with a stripped CONTEXT_XSTATE;
//  - Unknown: Why::VexNoAvx (extended area present but its mask unreadable or without the AVX area: a guess).
// Pure: `xstate` = the context has an extended area, `mask_ok` = its feature mask could be read, `avx_bit` = the AVX bit of that mask,
// `located` = the AVX area exists in the context, `allow` = the caller allows InitZero / Hardware (KYTY_GUEST_WAIT), `hw_ok` = the hardware
// zeroing is available (Windows, the stub page exists).
enum class YmmhState : uint8_t { Present, InitZero, Hardware, Unknown };
// Mutant (tests/GuestWaitTests.cpp, WILL_FAIL): KYTY_GUESTREAD_MUTANT_VEX_MASK_GUESS (an unreadable XSTATE mask counts as init state).
[[nodiscard]] constexpr YmmhState ClassifyYmmh(bool xstate, bool mask_ok, bool avx_bit, bool located, bool allow, bool hw_ok) {
	if (!xstate) {
		return allow && hw_ok ? YmmhState::Hardware : YmmhState::Unknown;
	}
#if defined(KYTY_GUESTREAD_MUTANT_VEX_MASK_GUESS)
	if (!mask_ok) return allow ? YmmhState::InitZero : YmmhState::Unknown;
#endif
	if (!mask_ok || !located) {
		return YmmhState::Unknown;
	}
	if (avx_bit) {
		return YmmhState::Present;
	}
	return allow ? YmmhState::InitZero : YmmhState::Unknown;
}

struct Options {
	bool vex_init_zero = false; // S585: serve VEX.128 loads in YmmhState::InitZero / Hardware
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
	bool     vex_init  = false; // S585: ... and they are already zero (YmmhState::InitZero), Commit leaves them
	bool     vex_hw    = false; // S585: ... the context has no extended area (YmmhState::Hardware): Commit zeroes the hardware register's upper half
	uint64_t address   = 0; // effective address of the memory operand
	char     mnemonic[16] = {};
	char     operands[48] = {}; // "r32, m32" style shape for the DIAG line (no code bytes)
};

// Decode the instruction at the saved RIP and compute its memory operand. Does not touch the context.
// S585: a VexNoAvx refusal still carries `address` / `size` when they could be computed (the DIAG of KYTY_DIAG_GUEST_STALE reads them).
[[nodiscard]] Plan Decode(void* native_context, uint64_t fault_address, const Options& options = {});

// S585 DIAG: the saved context's extended-state facts behind YmmhState (Windows: flags, XSTATE mask, enabled features, AVX area), one line.
int DescribeContext(void* native_context, char* text, size_t size);

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

// Callback: read [address, address + size) from the backing into dst; anything but Ok when the bytes may not be taken from the backing.
enum class BackingResult : uint8_t { Ok, Unavailable, Dirty, Inflight, LockBusy };
using ReadBackingFn = BackingResult (*)(uint64_t address, uint64_t size, void* dst);

struct Attempt {
	Plan    plan;
	uint8_t bytes[16] = {};
};

// Decode + read the backing. On true, `attempt` holds everything Commit needs; on false attempt.plan.why says why (BackingRefused / BackingDirty /
// BackingInflight / LockBusy when only the backing read failed).
[[nodiscard]] bool Prepare(void* native_context, uint64_t fault_address, ReadBackingFn read, Attempt* attempt, const Options& options = {});

} // namespace Loader::GuestReadEmulator

#endif /* KYTY_LOADER_GUEST_READ_EMULATOR_H_ */
