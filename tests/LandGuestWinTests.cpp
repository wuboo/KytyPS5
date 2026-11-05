// S495: KYTY_LAND_SYNC guest reads on Windows. The fault handler hands over a CONTEXT, not a
// ucontext. A fabricated CONTEXT must decode `mov eax, [rdi]`, write the backing into Rax and
// advance Rip. Mutant WIN_NO_REGS makes Regs::Valid() false, so Prepare refuses with NoContext
// and this test exits 1.

#include "loader/guestReadEmulator.h"

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

int g_fails = 0;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "land_guest_win: %s\n", what);
		g_fails++;
	}
}

Loader::GuestReadEmulator::BackingResult ReadBacking(uint64_t /*address*/, uint64_t size, void* dst) {
	static const uint8_t bytes[4] = {0x78, 0x56, 0x34, 0x12};
	if (size > sizeof(bytes)) {
		return Loader::GuestReadEmulator::BackingResult::Unavailable;
	}
	std::memcpy(dst, bytes, static_cast<size_t>(size));
	return Loader::GuestReadEmulator::BackingResult::Ok;
}

} // namespace

int main() {
	using namespace Loader::GuestReadEmulator;
	Check(Decode(nullptr, 0).why == Why::NoContext, "null context is refused");

	alignas(16) uint8_t code[32];
	std::memset(code, 0x90, sizeof(code));
	code[0] = 0x8b;
	code[1] = 0x07; // mov eax, [rdi]

	CONTEXT ctx {};
	ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
	ctx.Rip          = reinterpret_cast<DWORD64>(code);
	ctx.Rdi          = 0x200000;
	ctx.Rax          = 0xffffffffffffffffull;

	Attempt    attempt;
	const bool ok = Prepare(&ctx, 0x200000, ReadBacking, &attempt);
	Check(ok, "mov eax,[rdi] prepares from a CONTEXT");
	if (ok) {
		Commit(&ctx, attempt.plan, attempt.bytes);
		Check(ctx.Rax == 0x12345678ull, "eax is the backing dword and the upper half is zero");
		Check(ctx.Rip == reinterpret_cast<DWORD64>(code) + 2, "rip advanced past the mov");
	}
	return g_fails == 0 ? 0 : 1;
}
