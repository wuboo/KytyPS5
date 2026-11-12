// S552 KYTY_RT_PASS_MEMO (rtPassMemo.h): the slot memo hits only on the same register bytes, mask nibble and call flags.
// Mutants: NO_MASK (a target-mask change still hits), PREFIX (only the first register word is compared).

#include "graphics/host_gpu/renderer/rtPassMemo.h"

#include <cstdio>
#include <cstring>

namespace {

int g_failures = 0;
void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

struct FakeRegs {
	uint64_t base   = 0;
	uint32_t view   = 0;
	uint32_t info   = 0;
	uint32_t attrib = 0;
	uint32_t size   = 0;
	uint64_t dcc    = 0;
};

struct FakeValue {
	int id = 0;
};

using Libs::Graphics::RtPassMemo::Flags;
using Libs::Graphics::RtPassMemo::Mode;
using Libs::Graphics::RtPassMemo::ParseMode;
using Slot = Libs::Graphics::RtPassMemo::Slot<FakeRegs, FakeValue>;

} // namespace

int main() {
	Check(ParseMode(nullptr) == Mode::Off && ParseMode("") == Mode::Off && ParseMode("0") == Mode::Off, "off");
	Check(ParseMode("1") == Mode::On && ParseMode("shadow") == Mode::Shadow && ParseMode("2") == Mode::Off, "modes");

	FakeRegs regs {};
	regs.base   = 0x10000000;
	regs.view   = 3;
	regs.info   = 0x1234;
	regs.attrib = 7;
	regs.size   = (1919u << 16u) | 1079u;
	regs.dcc    = 0x20000000;

	Slot slot {};
	Check(!slot.Matches(regs, 0xf, Flags(false, false)), "empty slot misses");
	slot.Store(regs, 0xf, Flags(false, false), {42});
	Check(slot.Matches(regs, 0xf, Flags(false, false)) && slot.value.id == 42, "same key hits");
	Check(!slot.Matches(regs, 0x7, Flags(false, false)), "mask change misses");
	Check(!slot.Matches(regs, 0xf, Flags(true, false)), "ignore flag misses");
	Check(!slot.Matches(regs, 0xf, Flags(false, true)), "exact flag misses");

	auto other = regs;
	other.size = (959u << 16u) | 539u; // a different extent at the same address
	Check(!slot.Matches(other, 0xf, Flags(false, false)), "extent change misses");
	other     = regs;
	other.dcc = 0; // DCC turned off
	Check(!slot.Matches(other, 0xf, Flags(false, false)), "dcc change misses");
	other      = regs;
	other.view = 4; // another slice
	Check(!slot.Matches(other, 0xf, Flags(false, false)), "view change misses");

	slot.Clear();
	Check(!slot.Matches(regs, 0xf, Flags(false, false)), "cleared slot misses");

	if (g_failures == 0) std::printf("rt_pass_memo: OK\n");
	return g_failures == 0 ? 0 : 1;
}
