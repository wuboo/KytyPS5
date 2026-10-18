// S417: KYTY_PV_IDX_BACKING decisions (pvIndexBacking.h).
// Mutants (each must fail): BACKING_IGNORES_DIRTY (the backing is read although the byte tracker has a GPU write in the range),
// SHADOW_SKIPS_RAW (shadow draws from the copy and does not do the old read), FIRSTDIFF_SKIPS_LAST (a difference in the last byte is missed).

#include "graphics/host_gpu/renderer/pvIndexBacking.h"

#include <cstdio>

using namespace Libs::Graphics::PvIndexBacking;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	Check(ParseMode(nullptr) == Mode::Off, "unset = off");
	Check(ParseMode("0") == Mode::Off, "0 = off");
	Check(ParseMode("1") == Mode::On, "1 = on");
	Check(ParseMode("shadow") == Mode::Shadow, "shadow");
	Check(ParseMode("yes") == Mode::Off, "anything else = off");
	Check(EnvMode() == Mode::Off, "without the variable the switch is off");

	Check(Decide(Mode::Off, false, false) == Source::Raw, "off: raw");
	Check(Decide(Mode::On, false, false) == Source::Backing, "on, clean bytes: backing");
	Check(Decide(Mode::On, false, true) == Source::Raw, "on, GPU-dirty bytes: raw (the backing is stale)");
	Check(Decide(Mode::On, true, false) == Source::Raw, "on, host copy: raw");
	Check(Decide(Mode::Shadow, false, false) == Source::Shadow, "shadow, clean bytes: both reads");
	Check(Decide(Mode::Shadow, false, true) == Source::Raw, "shadow, GPU-dirty bytes: raw only");
	Check(Decide(Mode::Shadow, true, false) == Source::Raw, "shadow, host copy: raw only");

	const unsigned char a[4] = {1, 2, 3, 4};
	unsigned char       b[4] = {1, 2, 3, 4};
	Check(FirstDiff(a, b, 4) == 4, "equal reads");
	b[3] = 9;
	Check(FirstDiff(a, b, 4) == 3, "a difference in the last byte");
	b[1] = 9;
	Check(FirstDiff(a, b, 4) == 1, "the first of two differences");
	Check(FirstDiff(a, b, 0) == 0, "empty");

	if (g_failures != 0) return 1;
	std::fprintf(stderr, "PvIndexBackingTests: ok\n");
	return 0;
}
