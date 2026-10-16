#include "kernel/guestDriverMemory.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::LibKernel::GuestLayout::ChooseDriverDmemBase;
using Libs::LibKernel::GuestLayout::SystemReservedMax;

constexpr uint64_t kPreferred = 0x0fe0000000ull;
constexpr uint64_t kSize      = 0x200000;
constexpr uint64_t kMacMax    = 0x0fbfffffffull;
constexpr uint64_t kOtherMax  = 0x0fffffffffull;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "GuestDriverMemoryTests: failed: %s\n", message);
		std::exit(1);
	}
}

bool Fits(uint64_t base, uint64_t size, uint64_t reserved_max) {
	return base != 0 && size - 1u <= reserved_max - base && base <= reserved_max;
}

} // namespace

int main() {
	// Hosts that reserve up to 0xfffffffff keep the upstream address.
	Check(ChooseDriverDmemBase(kPreferred, kSize, kOtherMax) == kPreferred, "preferred address kept when reserved");

	// macOS x86_64 reserves only up to 0xfbfffffff: 0xfe0000000 is not guest memory there.
	const auto mac = ChooseDriverDmemBase(kPreferred, kSize, kMacMax);
	Check(mac != kPreferred, "preferred address outside the reserved region is not used");
	Check(Fits(mac, kSize, kMacMax), "moved block lies inside the reserved region");
	Check(mac % kSize == 0, "moved block keeps the size alignment");
	Check(mac == kMacMax + 1u - kSize, "moved block sits at the top of the reserved region");

	// A reserved region whose end is not size-aligned still yields an aligned block inside it.
	const auto odd = ChooseDriverDmemBase(kPreferred, kSize, 0x1234ffffull + 0x100000ull);
	Check(odd % kSize == 0 && Fits(odd, kSize, 0x1234ffffull + 0x100000ull), "unaligned region end gives an aligned block inside");

	// The block touching the last reserved byte is accepted; one byte past it is not.
	Check(ChooseDriverDmemBase(kMacMax + 1u - kSize, kSize, kMacMax) == kMacMax + 1u - kSize, "block ending on the last byte is kept");
	// Told apart from a move to the aligned top only when the preferred end is not size-aligned.
	Check(ChooseDriverDmemBase(0x0f00100000ull, kSize, 0x0f002fffffull) == 0x0f00100000ull, "block ending on an unaligned last byte is kept");
	Check(ChooseDriverDmemBase(kMacMax + 2u - kSize, kSize, kMacMax) != kMacMax + 2u - kSize, "block ending one past the region is moved");

	// Degenerate input never yields a block.
	Check(ChooseDriverDmemBase(kPreferred, 0, kMacMax) == 0, "zero size has no block");
	Check(ChooseDriverDmemBase(kPreferred, kSize, kSize - 1u) == 0, "region smaller than the block has no block");

	// The host this binary targets must be able to place the driver block.
	const auto host = ChooseDriverDmemBase(kPreferred, kSize, SystemReservedMax);
	Check(Fits(host, kSize, SystemReservedMax), "driver block fits the host's reserved region");

	std::printf("GuestDriverMemoryTests: ok\n");
	return 0;
}
