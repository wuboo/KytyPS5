// S502: KYTY_RB_WALK decisions (rbWalk.h).
// Mutants (each must fail): SERVE_DIRTY, SERVE_INFLIGHT, SERVE_UNKNOWN (a clean-looking answer is served
// although the bytes are GPU-dirty, a copy is in flight, or the tracker could not be queried),
// SHADOW_SKIPS_FAULT (shadow completes from the backing and does not do the old read).

#include "graphics/host_gpu/renderer/rbWalk.h"

#include <cstdio>

using namespace Libs::Graphics::RbWalk;

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

	Check(Decide(Mode::Off, false, false, false) == Source::Fault, "off: fault");
	Check(Decide(Mode::Off, true, true, true) == Source::Fault, "off, every danger: fault");
	Check(Decide(Mode::On, false, false, false) == Source::Backing, "on, known clean: backing");
	Check(Decide(Mode::On, true, false, false) == Source::Fault, "on, GPU-dirty bytes: fault");
	Check(Decide(Mode::On, false, true, false) == Source::Fault, "on, copy in flight: fault");
	Check(Decide(Mode::On, false, false, true) == Source::Fault, "on, tracker unknown: fault");
	Check(Decide(Mode::On, true, true, true) == Source::Fault, "on, all three: fault");
	Check(Decide(Mode::Shadow, false, false, false) == Source::Shadow, "shadow, known clean: both");
	Check(Decide(Mode::Shadow, true, false, false) == Source::Fault, "shadow, GPU-dirty: fault only");
	Check(Decide(Mode::Shadow, false, true, false) == Source::Fault, "shadow, in flight: fault only");
	Check(Decide(Mode::Shadow, false, false, true) == Source::Fault, "shadow, unknown: fault only");

	Check(!PrefetchWrite(Mode::Off, true, 32), "off does not prefetch");
	Check(!PrefetchWrite(Mode::Shadow, true, 32), "shadow does not prefetch");
	Check(!PrefetchWrite(Mode::On, false, 32), "an unlearned page is not prefetched");
	Check(PrefetchWrite(Mode::On, true, 32), "on, learned, 32 B: prefetch");
	Check(PrefetchWrite(Mode::On, true, 4096), "on, learned, one page: prefetch");
	Check(!PrefetchWrite(Mode::On, true, 8192), "on, learned, two pages: no prefetch");
	Check(!PrefetchWrite(Mode::On, true, 0), "empty write: no prefetch");
	Check(SplitAfterWrite(Mode::On, true), "on, armed: submit the copy");
	Check(!SplitAfterWrite(Mode::On, false), "on, not armed: no submit");
	Check(!SplitAfterWrite(Mode::Shadow, true), "shadow does not submit");
	Check(!SplitAfterWrite(Mode::Off, true), "off does not submit");

	LearnedPages learned;
	Check(learned.Note(0x1000), "first page is new");
	Check(!learned.Note(0x1000), "the same page is not new");
	Check(learned.Contains(0x1000), "noted page is found");
	Check(learned.Intersects(0x101c, 4), "a dword inside the page intersects");
	Check(!learned.Intersects(0x2000, 4), "another page does not");
	for (uint64_t i = 1; i < LearnedPages::kCap; ++i) Check(learned.Note(0x1000 + i * 0x1000), "fill");
	Check(!learned.Note(0x9000), "the table stops at its cap");

	if (g_failures != 0) {
		std::fprintf(stderr, "RbWalkTests: %d failed\n", g_failures);
		return 1;
	}
	std::puts("RbWalkTests: ok");
	return 0;
}
