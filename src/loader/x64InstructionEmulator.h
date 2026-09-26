#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <Zydis/DecoderTypes.h>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool IsReciprocalSquareRoot(const ZydisDecodedInstruction& instruction,
                                         const ZydisDecodedOperand* operands);
uint64_t           PatchReciprocalSquareRoots(uint64_t address, uint64_t size);
[[nodiscard]] bool TryEmulate(void* native_context);
struct WideStoreSplitResult {
	uint64_t candidates = 0;  // plain 256-bit vector stores found
	uint64_t patched    = 0;  // rewritten into two 128-bit stores
	uint64_t too_short  = 0;  // shorter than a rel32 jump; left in place
	uint64_t unsupported = 0; // segment override, encoder failure or trampoline area full
	uint64_t trampoline_bytes = 0;
};

// Rosetta aborts the process when a 256-bit store that crosses into a write-protected page
// faults after writing its first half. 128-bit stores fault cleanly. Rewrite each plain
// 256-bit vector store in [address, address + size) into a jump to a trampoline that performs
// the store as two 128-bit halves and jumps back. Trampolines are written from *cursor up to
// trampoline_end, which must be within +/-2 GiB of the code.
WideStoreSplitResult SplitWideStores(uint64_t address, uint64_t size, uint64_t* cursor,
                                     uint64_t trampoline_end);

// Bench diagnostic: log 256-bit memory stores in a code range by length and addressing.
void LogWideStores(uint64_t address, uint64_t size, const char* module_name);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
