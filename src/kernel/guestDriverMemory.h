#pragma once

#include <cstdint>

namespace Libs::LibKernel::GuestLayout {

// Highest guest address of the system-managed region that the host reserves up front. Rosetta
// leaves the area above 0xfc0000000 to the host, so macOS x86_64 stops short of the other hosts.
#if defined(__APPLE__) && (defined(__x86_64__) || defined(_M_X64))
inline constexpr uint64_t SystemReservedMax = 0x0fbfffffffull;
#else
inline constexpr uint64_t SystemReservedMax = 0x0fffffffffull;
#endif

// The native driver keeps its direct memory at a fixed guest address. The preferred address is used
// when it lies inside the reserved region; otherwise the block moves to the top of that region, so
// the mapping never depends on an address the host does not hand out.
// `size` must be a power of two.
constexpr uint64_t ChooseDriverDmemBase(uint64_t preferred, uint64_t size, uint64_t reserved_max) {
	if (size == 0 || size > reserved_max) {
		return 0;
	}
	if (preferred <= reserved_max && size - 1u <= reserved_max - preferred) {
		return preferred;
	}
	return (reserved_max + 1u - size) & ~(size - 1u);
}

} // namespace Libs::LibKernel::GuestLayout
