// S507: KYTY_RB_CLEAN_WALK serves a clean walker load from the backing (cleanWalk.h).
// Mutant KYTY_CLEANWALK_MUTANT_SERVE_DIRTY must fail: a GPU-dirty range is served.

#include "graphics/host_gpu/renderer/cleanWalk.h"

#include <cstdint>
#include <cstdio>

using namespace Libs::Graphics::CleanWalk;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	Check(Parse(nullptr) == Mode::Off, "unset = off");
	Check(Parse("0") == Mode::Off, "0 = off");
	Check(Parse("1") == Mode::On, "1 = on");
	Check(Parse("shadow") == Mode::Shadow, "shadow");
	Check(Parse("yes") == Mode::Off, "anything else = off");

	Check(Decide(Mode::Off, false, false, false) == Action::Drain, "off drains");
	Check(Decide(Mode::On, false, false, false) == Action::Serve, "on, clean bytes: backing");
	Check(Decide(Mode::On, true, false, false) == Action::Drain, "on, GPU-dirty bytes: drain");
	Check(Decide(Mode::On, false, true, false) == Action::Drain, "on, copy in flight: drain");
	Check(Decide(Mode::On, false, false, true) == Action::Drain, "on, tracker busy: drain");
	Check(Decide(Mode::Shadow, false, false, false) == Action::Compare, "shadow, clean bytes: compare");
	Check(Decide(Mode::Shadow, true, false, false) == Action::Drain, "shadow, GPU-dirty bytes: drain only");
	Check(Decide(Mode::Shadow, false, true, false) == Action::Drain, "shadow, copy in flight: drain only");

	const uint8_t a[4] = {1, 2, 3, 4};
	uint8_t       b[4] = {1, 2, 3, 4};
	Check(Equal(a, b, 4), "equal reads");
	b[3] = 9;
	Check(!Equal(a, b, 4), "a different last byte");
	return g_failures == 0 ? 0 : 1;
}
