// S323: CP phase accounting (cpPhases.h) against a fake clock.
// Mutants (KYTY_CPPHASES_MUTANT_*, each must fail): KEEP_CHILDREN (a phase is not reduced by its children),
// DROP_PARSE (the Parse phase is never counted).

#include "graphics/presentation/cpPhases.h"
#include "graphics/host_gpu/renderer/drawRepeatDiag.h"

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

	// Draw phases: self times add up to the outer span, children excluded.
	{
		Accountant<> a;
		int64_t      t = 0;
		a.Enter(Phase::Handler, t);
		a.Enter(Phase::DrawShaders, t);
		t += 4;
		a.Leave(t);
		a.Enter(Phase::DrawBind, t);
		t += 6;
		a.Leave(t);
		a.Enter(Phase::DrawPipeline, t);
		t += 3;
		a.Leave(t);
		a.Enter(Phase::DrawTargets, t);
		t += 5;
		a.Leave(t);
		a.Enter(Phase::DrawDesc, t);
		t += 2;
		a.Leave(t);
		a.Enter(Phase::DrawVkcmd, t);
		t += 8;
		a.Leave(t);
		t += 1;
		a.Leave(t);
		Check(a.Self(Phase::DrawShaders) == 4 && a.Self(Phase::DrawBind) == 6 && a.Self(Phase::DrawPipeline) == 3, "draw phase selves");
		Check(a.Self(Phase::DrawTargets) == 5 && a.Self(Phase::DrawDesc) == 2 && a.Self(Phase::DrawVkcmd) == 8, "draw phase selves, rest");
		Check(a.Self(Phase::Handler) == 1, "handler keeps only the gap");
		Check(a.Total() == 29, "draw phases plus handler equal the outer span");
	}

	{
		using Libs::Graphics::DrawRepeat::Finger;
		using Libs::Graphics::DrawRepeat::SameRt;
		using Libs::Graphics::DrawRepeat::SameState;
		Finger a;
		a.valid = true;
		a.cb = 1;
		a.pipeline = 9;
		a.generation = 3;
		a.color_n = 1;
		a.colors[0] = 4;
		a.depth = 5;
		a.images = {7};
		a.samplers = {8};
		a.buffers = {2};
		a.shader_data = {1, 2, 3};
		Finger b = a;
		Check(SameState(a, b) && SameRt(a, b), "identical draw repeats state and targets");
		b.shader_data[2] = 9;
		Check(!SameState(a, b) && SameRt(a, b), "shader_data is part of the state");
		b = a;
		b.generation = 4;
		Check(SameState(a, b) && !SameRt(a, b), "texture generation is part of the targets");
		b = a;
		b.cb = 2;
		Check(!SameState(a, b) && !SameRt(a, b), "a new command buffer is not a repeat");
		Finger empty;
		Check(!SameState(empty, a) && !SameRt(empty, a), "the first draw is not a repeat");
	}

	return g_failures == 0 ? 0 : 1;
}
