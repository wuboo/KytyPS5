#include "graphics/host_gpu/renderer/cache/landSync.h"

#include <cstdint>
#include <cstdio>

using Libs::Graphics::LandSync::ClearHit;
using Libs::Graphics::LandSync::ColorClearHit;
using Libs::Graphics::LandSync::DccMetadataObtainIsWrite;
using Libs::Graphics::LandSync::Effective;
using Libs::Graphics::LandSync::EnablesGuest;
using Libs::Graphics::LandSync::Mode;
using Libs::Graphics::LandSync::Parse;
using Libs::Graphics::LandSync::ResolveGuestEmu;

// Release builds define NDEBUG, which strips assert(). The mutant must still
// exit non-zero there, and the exit must be a normal failure (not abort):
// ctest WILL_FAIL does not invert crashes.
static int g_fails = 0;

static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "land_sync: %s\n", what);
		g_fails++;
	}
}

int main() {
	Check(Parse(nullptr) == Mode::Off, "parse null");
	Check(Parse("0") == Mode::Off, "parse 0");
	Check(Parse("1") == Mode::On, "parse 1");
	Check(Parse("shadow") == Mode::Shadow, "parse shadow");
	Check(Parse("yes") == Mode::Off, "parse yes");

	Check(Effective(Mode::On, nullptr) == Mode::On, "effective unset follows master");
	Check(Effective(Mode::On, "") == Mode::On, "effective empty follows master");
	Check(Effective(Mode::On, "0") == Mode::Off, "effective 0 overrides");
	Check(Effective(Mode::Off, "1") == Mode::On, "effective 1 overrides");
	Check(Effective(Mode::On, "shadow") == Mode::Shadow, "effective shadow overrides");

	Check(EnablesGuest(Mode::On), "guest on");
	Check(EnablesGuest(Mode::Shadow), "guest shadow");
	Check(!EnablesGuest(Mode::Off), "guest off");
	Check(ResolveGuestEmu(0, Mode::On) == 1, "land gate turns emulation on");
	Check(ResolveGuestEmu(0, Mode::Shadow) == 2, "land shadow stays shadow");
	Check(ResolveGuestEmu(0, Mode::Off) == 0, "land off leaves emulation off");
	Check(ResolveGuestEmu(2, Mode::Off) == 2, "explicit gf emulate wins");
	Check(ResolveGuestEmu(1, Mode::On) == 1, "explicit gf on wins");
	Check(!DccMetadataObtainIsWrite(), "obtain is a read");

	const uint32_t uniform = 0x20202020u;
	const uint32_t words[] {uniform, uniform, uniform, uniform};
	const ClearHit hit = ColorClearHit(words, 4, 1u << 1u);
	Check(hit.hit, "clear hit");
	Check(hit.slot == 1u, "clear slot");

	const uint32_t mixed[] {uniform, 0x21212121u};
	Check(!ColorClearHit(mixed, 2, 1u << 1u).hit, "mixed bytes miss");
	Check(!ColorClearHit(words, 4, 1u << 0u).hit, "unselected slot misses");
	const uint32_t unaligned = 0x01010101u;
	Check(!ColorClearHit(&unaligned, 1, 0xffu).hit, "unaligned key misses");
	Check(!ColorClearHit(words, 0, 0xffu).hit, "empty range misses");
	return g_fails == 0 ? 0 : 1;
}
