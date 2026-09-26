#ifndef KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_
#define KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_

#include <Zydis/DecoderTypes.h>
#include <cstdint>

namespace Loader::X64InstructionEmulator {

[[nodiscard]] bool IsReciprocalSquareRoot(const ZydisDecodedInstruction& instruction,
                                         const ZydisDecodedOperand* operands);
uint64_t           PatchReciprocalSquareRoots(uint64_t address, uint64_t size);
[[nodiscard]] bool TryEmulate(void* native_context);
// Bench diagnostic: log 256-bit memory stores in a code range by length and addressing.
void LogWideStores(uint64_t address, uint64_t size, const char* module_name);

} // namespace Loader::X64InstructionEmulator

#endif /* KYTY_LOADER_X64_INSTRUCTION_EMULATOR_H_ */
