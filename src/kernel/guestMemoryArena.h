#ifndef EMULATOR_SRC_KERNEL_GUESTMEMORYARENA_H_
#define EMULATOR_SRC_KERNEL_GUESTMEMORYARENA_H_

#include <cstdint>

namespace Libs::LibKernel::Memory {

// Identifies a shared-memory arena that backs guest mappings. Only the direct-memory arena exists
// today; a flexible arena gets the next id.
using GuestArenaId = uint32_t;
inline constexpr GuestArenaId GUEST_ARENA_DIRECT = 0;

// Host-side placement of one arena: the writable alias the GPU may import.
struct GuestArenaInfo {
	GuestArenaId id   = GUEST_ARENA_DIRECT;
	uint64_t     base = 0; // host address of the alias, 0 when the arena is unavailable
	uint64_t     size = 0;
};

// One contiguous piece of a guest range inside an arena (a guest range can span several mappings).
struct GuestArenaSpan {
	uint64_t     vaddr  = 0;
	uint64_t     size   = 0;
	GuestArenaId arena  = GUEST_ARENA_DIRECT;
	uint64_t     offset = 0; // byte offset of vaddr inside the arena
};

// Receives every change of the guest -> arena mappings, under the address-space lock, on the
// thread that changed the mapping. Implementations must only queue the event and must not call
// back into the kernel (the lock is held). Registering replays the existing mappings as OnMapped.
class IGuestMappingListener {
public:
	virtual ~IGuestMappingListener() = default;

	virtual void OnMapped(uint64_t vaddr, uint64_t size, GuestArenaId arena, uint64_t offset) = 0;
	virtual void OnUnmapped(uint64_t vaddr, uint64_t size)                                   = 0;
};

} // namespace Libs::LibKernel::Memory

#endif // EMULATOR_SRC_KERNEL_GUESTMEMORYARENA_H_
