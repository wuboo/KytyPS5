// S323: CP phase accounting (cpPhases.h) against a fake clock.
// Mutants (KYTY_CPPHASES_MUTANT_*, each must fail): KEEP_CHILDREN (a phase is not reduced by its children),
// DROP_PARSE (the Parse phase is never counted).

#include "graphics/presentation/cpPhases.h"

#include <cstdio>

using namespace Libs::Graphics::CpPhases;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	// ---- the model of one CP submission: Parse{ Handler{ DrawBind, DrawDesc{ Wait }, Eop }, Gc } ----
	{
		Accountant<> a;
		int64_t      t = 1000;
		const int64_t begin = t;
		a.Enter(Phase::Parse, t);
		t += 10; // parse
		a.Enter(Phase::Handler, t);
		t += 5; // handler
		a.Enter(Phase::DrawBind, t);
		t += 20;
		a.Leave(t);
		a.Enter(Phase::DrawDesc, t);
		t += 7;
		a.Enter(Phase::Wait, t);
		t += 100; // blocked on the GPU inside the descriptor phase
		a.Leave(t);
		t += 3;
		a.Leave(t);
		a.Enter(Phase::Eop, t);
		t += 4;
		a.Leave(t);
		t += 6; // handler tail
		a.Leave(t);
		a.Enter(Phase::Gc, t);
		t += 9;
		a.Leave(t);
		t += 2; // parse tail
		a.Leave(t);
		const uint64_t busy = static_cast<uint64_t>(t - begin);

		Check(a.Depth() == 0, "stack empty after the submission");
		Check(a.Self(Phase::Parse) == 12, "parse = 10 + 2 (children excluded)");
		Check(a.Self(Phase::Handler) == 11, "handler = 5 + 6 (children excluded)");
		Check(a.Self(Phase::DrawBind) == 20, "draw_bind");
		Check(a.Self(Phase::DrawDesc) == 10, "draw_desc = 7 + 3 without the wait");
		Check(a.Self(Phase::Wait) == 100, "wait slot");
		Check(a.Self(Phase::Eop) == 4, "eop");
		Check(a.Self(Phase::Gc) == 9, "gc");
		// no nanosecond twice: phases + wait == the span of the outermost phase <= cp_busy
		Check(a.Total() == busy, "sum of phases + wait equals the outer span (no double count)");
		Check(a.Total() - a.Self(Phase::Wait) <= busy - a.Self(Phase::Wait), "phases + waits <= cp_busy");
	}

	// ---- repeated phases accumulate; sibling order does not matter ----
	{
		Accountant<> a;
		int64_t      t = 0;
		a.Enter(Phase::Parse, t);
		for (int i = 0; i < 3; ++i) {
			a.Enter(Phase::Barrier, t);
			t += 5;
			a.Leave(t);
			t += 1;
		}
		a.Leave(t);
		Check(a.Self(Phase::Barrier) == 15, "barrier accumulates over three scopes");
		Check(a.Self(Phase::Parse) == 3, "parse keeps only the gaps");
	}

	// ---- a clock step back counts as zero, never as a huge value ----
	{
		Accountant<> a;
		a.Enter(Phase::Gc, 100);
		a.Leave(50);
		Check(a.Self(Phase::Gc) == 0, "negative span is zero");
	}

	// ---- a full stack refuses Enter, and the refused Enter is not paired with a Leave ----
	{
		Accountant<2> a;
		Check(a.Enter(Phase::Parse, 0), "first");
		Check(a.Enter(Phase::Handler, 1), "second");
		Check(!a.Enter(Phase::Gc, 2), "third refused");
		a.Leave(5);
		a.Leave(9);
		Check(a.Depth() == 0 && a.Total() == 9, "bookkeeping intact after overflow");
	}

	return g_failures == 0 ? 0 : 1;
}
