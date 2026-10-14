#include "graphics/host_gpu/renderer/eagerSubmit.h"

#include <cstdio>
#include <cstdlib>

// S326: policy of the eager submit (ShouldEagerSplit). Built four times: normally and with KYTY_EAGER_MUTANT_PASS_OPEN / _BUSY / _NO_AGE / _NO_CLOSING (S327).
// S328: + mid-pass split (MidPassAllowed) and the reopen without clears; mutants KYTY_EAGER_MUTANT_MID_ANY_CB / _MID_NO_LIMIT / _CLEAR_REOPEN.
// The mutant builds must FAIL (ctest WILL_FAIL).

namespace {

using Libs::Graphics::Sync::EagerQuery;
using Libs::Graphics::Sync::ShouldEagerSplit;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "EagerSubmitTests: failed: %s\n", message);
		std::exit(1); // not abort(): ctest WILL_FAIL does not invert signals
	}
}

EagerQuery Ready() {
	EagerQuery q;
	q.enabled    = true;
	q.has_work   = true;
	q.gpu_idle   = true;
	q.pass_open  = false;
	q.age_us     = 500;
	q.min_age_us = 300;
	return q;
}

struct FakeAttachment {
	bool is_clear = false, depth_clear = false, stencil_clear = false;
};
struct FakeState {
	FakeAttachment color_attachments[2];
	FakeAttachment depth_stencil_attachment;
	uint32_t       num_color_attachments = 2;
};

EagerQuery MidReady() {
	EagerQuery q = Ready();
	q.pass_open            = true;
	q.pass_closing         = false;
	q.mid_mode             = true;
	q.first_cb_after_drain = true;
	q.mid_splits           = 0;
	q.mid_max              = 1;
	return q;
}

} // namespace

int main() {
	Check(ShouldEagerSplit(Ready()), "idle GPU, closed pass, work, old enough: split");
	{
		auto q = Ready();
		q.enabled = false;
		Check(!ShouldEagerSplit(q), "disabled never splits");
	}
	{
		auto q = Ready();
		q.has_work = false;
		Check(!ShouldEagerSplit(q), "empty buffer never splits");
	}
	{
		auto q = Ready();
		q.pass_open = true;
		Check(!ShouldEagerSplit(q), "open pass: no split");
		q.pass_closing = true;
		Check(ShouldEagerSplit(q), "S327: open pass that is being closed at a pass boundary: split");
		q.has_work = false;
		Check(!ShouldEagerSplit(q), "S327: pass boundary but empty buffer: no split");
	}
	{
		auto q = Ready();
		q.pass_open    = true;
		q.pass_closing = true;
		q.gpu_idle     = false;
		Check(!ShouldEagerSplit(q), "S327: pass boundary, busy GPU: no split");
		q.gpu_idle = true;
		q.age_us   = 10;
		Check(!ShouldEagerSplit(q), "S327: pass boundary, young buffer: no split");
	}
	{
		auto q = Ready();
		q.gpu_idle = false;
		Check(!ShouldEagerSplit(q), "busy GPU: no split");
	}
	{
		auto q = Ready();
		q.age_us = 299;
		Check(!ShouldEagerSplit(q), "younger than the threshold: no split");
		q.age_us = 300;
		Check(ShouldEagerSplit(q), "exactly the threshold: split");
	}
	// S328: split in the middle of an open pass, first buffer after a drain only, at most mid_max per segment.
	Check(ShouldEagerSplit(MidReady()), "S328: first buffer after a drain, open pass, mode 2: mid-pass split");
	{
		auto q = MidReady();
		q.mid_mode = false;
		Check(!ShouldEagerSplit(q), "S328: mode 1 never splits an open pass in the middle");
	}
	{
		auto q = MidReady();
		q.first_cb_after_drain = false;
		Check(!ShouldEagerSplit(q), "S328: not the first buffer after the drain: no mid-pass split");
	}
	{
		auto q = MidReady();
		q.mid_splits = 1;
		Check(!ShouldEagerSplit(q), "S328: one mid-pass split per segment already made: no second");
		q.mid_max = 2;
		Check(ShouldEagerSplit(q), "S328: limit raised to 2: second allowed");
	}
	{
		auto q = MidReady();
		q.gpu_idle = false;
		Check(!ShouldEagerSplit(q), "S328: mid-pass, busy GPU: no split");
		q = MidReady();
		q.age_us = 100;
		Check(!ShouldEagerSplit(q), "S328: mid-pass, young buffer: no split");
		q = MidReady();
		q.has_work = false;
		Check(!ShouldEagerSplit(q), "S328: mid-pass, empty buffer: no split");
	}
	{
		// The reopened part loads what the first part stored: no clear is repeated, everything else is untouched.
		FakeState s;
		s.color_attachments[0].is_clear      = true;
		s.color_attachments[1].is_clear      = true;
		s.depth_stencil_attachment.depth_clear   = true;
		s.depth_stencil_attachment.stencil_clear = true;
		Libs::Graphics::Sync::StripClearsOnReopen(s);
		Check(!s.color_attachments[0].is_clear && !s.color_attachments[1].is_clear, "S328: reopen: colour attachments load");
		Check(!s.depth_stencil_attachment.depth_clear, "S328: reopen: depth loads");
		Check(!s.depth_stencil_attachment.stencil_clear, "S328: reopen: stencil loads");
		Check(s.num_color_attachments == 2, "S328: reopen keeps the attachment count");
	}
	std::printf("EagerSubmitTests: ok\n");
	return 0;
}
