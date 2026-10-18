#include "graphics/host_gpu/renderer/eagerSubmit.h"

#include <cstdio>
#include <cstdlib>

// S326: policy of the eager submit (ShouldEagerSplit). Built four times: normally and with KYTY_EAGER_MUTANT_PASS_OPEN / _BUSY / _NO_AGE / _NO_CLOSING (S327).
// S328: + mid-pass split (MidPassAllowed) and the reopen without clears; mutants KYTY_EAGER_MUTANT_MID_ANY_CB / _MID_NO_LIMIT / _CLEAR_REOPEN.
// S410: mode 3 (MidPassAllowed in any buffer, per flip limit), LowWaterAllowed, ShouldEagerFlush3, EagerDeclineReason, ReopenAfterFlush;
// mutants KYTY_EAGER_MUTANT_EAGER3_NO_LIMIT / _LOWWATER_ANY_BACKLOG (+ the existing _CLEAR_REOPEN for ReopenAfterFlush).
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
	bool operator==(const FakeAttachment&) const = default;
};
struct FakeState {
	FakeAttachment color_attachments[2];
	FakeAttachment depth_stencil_attachment;
	uint32_t       num_color_attachments = 2;
	bool operator==(const FakeState& o) const {
		return color_attachments[0] == o.color_attachments[0] && color_attachments[1] == o.color_attachments[1] &&
		       depth_stencil_attachment == o.depth_stencil_attachment && num_color_attachments == o.num_color_attachments;
	}
};

// S410: a buffer that is not the first after a drain, pass open and continued by the draw, mode 3, idle GPU, 900 us old.
EagerQuery Mid3Ready() {
	EagerQuery q = Ready();
	q.pass_open            = true;
	q.mid_mode             = true;
	q.mode3                = true;
	q.first_cb_after_drain = false;
	q.age_us               = 900;
	q.mid3_us              = 800;
	q.flip_mid             = 0;
	q.mid3_max             = 8;
	return q;
}

// S410: GPU busy with exactly the last submitted buffer, low-water on, buffer 1100 us old, no pass open.
EagerQuery LowWaterReady() {
	EagerQuery q = Ready();
	q.mode3       = true;
	q.gpu_idle    = false;
	q.backlog     = 1;
	q.lowwater    = true;
	q.lowwater_us = 1000;
	q.age_us      = 1100;
	return q;
}

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
	// S410: mode 3: mid-pass split in ANY buffer (first_cb_after_drain not required), old enough, at most mid3_max per flip.
	Check(ShouldEagerSplit(Mid3Ready()), "S410: mode 3: later buffer, open pass, old enough: mid-pass split");
	{
		auto q = Mid3Ready();
		q.age_us = 799;
		Check(!ShouldEagerSplit(q), "S410: mode 3: younger than KYTY_EAGER_MID_US: no mid-pass split");
		q.age_us = 800;
		Check(ShouldEagerSplit(q), "S410: mode 3: exactly KYTY_EAGER_MID_US: split");
	}
	{
		auto q = Mid3Ready();
		q.flip_mid = 8;
		Check(!ShouldEagerSplit(q), "S410: mode 3: per-flip limit reached: no mid-pass split");
		q.flip_mid = 7;
		Check(ShouldEagerSplit(q), "S410: mode 3: one below the limit: split");
		q.flip_mid = 0;
		q.mid3_max = 0;
		Check(!ShouldEagerSplit(q), "S410: mode 3: limit 0: never");
	}
	{
		auto q = Mid3Ready();
		q.mid_mode = false; // a call site that is not a draw continuing the pass (dispatch, pass boundary of another state)
		Check(!ShouldEagerSplit(q), "S410: mode 3: not a mid-pass call site: an open pass is not split");
		q = Mid3Ready();
		q.gpu_idle = false;
		Check(!ShouldEagerSplit(q), "S410: mode 3: busy GPU (low-water off): no split");
	}
	// S410 low-water: the GPU runs the last submitted buffer.
	Check(ShouldEagerSplit(LowWaterReady()), "S410: low-water: backlog 1, old enough: split");
	{
		auto q = LowWaterReady();
		q.backlog = 2;
		Check(!ShouldEagerSplit(q), "S410: low-water: backlog 2: no split");
		q = LowWaterReady();
		q.age_us = 999;
		Check(!ShouldEagerSplit(q), "S410: low-water: younger than KYTY_EAGER_LOWWATER_US: no split");
		q = LowWaterReady();
		q.lowwater = false;
		Check(!ShouldEagerSplit(q), "S410: low-water off: busy GPU: no split");
		q = LowWaterReady();
		q.mode3 = false;
		Check(!ShouldEagerSplit(q), "S410: low-water needs mode 3");
		q = LowWaterReady();
		q.pass_open = true; // open pass not being closed and not a mid-pass call site
		Check(!ShouldEagerSplit(q), "S410: low-water does not split an open pass that is not closing");
		q.pass_closing = true;
		Check(ShouldEagerSplit(q), "S410: low-water at a pass boundary: split");
		q = LowWaterReady();
		q.pass_open  = true;
		q.mid_mode   = true;
		Check(ShouldEagerSplit(q), "S410: low-water + mid-pass call site in mode 3 (age 1100 >= 800): split");
		q.flip_mid = 8;
		Check(!ShouldEagerSplit(q), "S410: low-water mid-pass: same per-flip limit");
	}
	// S410: RequestFlush with an idle GPU.
	{
		auto q = Ready();
		q.mode3 = true;
		Check(Libs::Graphics::Sync::ShouldEagerFlush3(q), "S410: flush: idle GPU, work, old enough: flush now");
		q.gpu_idle = false;
		Check(!Libs::Graphics::Sync::ShouldEagerFlush3(q), "S410: flush: busy GPU: keep batching");
		q = Ready();
		q.mode3 = true;
		q.age_us = 299;
		Check(!Libs::Graphics::Sync::ShouldEagerFlush3(q), "S410: flush: young buffer: keep batching");
		q = Ready();
		q.mode3 = true;
		q.has_work = false;
		Check(!Libs::Graphics::Sync::ShouldEagerFlush3(q), "S410: flush: empty buffer: nothing to flush");
		q = Ready();
		Check(!Libs::Graphics::Sync::ShouldEagerFlush3(q), "S410: flush: modes 0-2 never flush early");
	}
	// S410 DIAG: the reason of a decline on an idle GPU.
	{
		using Libs::Graphics::Sync::EagerDecline;
		using Libs::Graphics::Sync::EagerDeclineReason;
		auto q = Ready();
		q.has_work = false;
		Check(EagerDeclineReason(q) == EagerDecline::NoWork, "S410 decline: no work");
		q = Ready();
		q.pass_open = true;
		Check(EagerDeclineReason(q) == EagerDecline::PassOpen, "S410 decline: pass open, no mid-pass site");
		q = MidReady();
		q.first_cb_after_drain = false;
		Check(EagerDeclineReason(q) == EagerDecline::FirstCbOnly, "S410 decline: mode 2, not the first buffer");
		q = MidReady();
		q.mid_splits = 1;
		Check(EagerDeclineReason(q) == EagerDecline::MidLimit, "S410 decline: mode 2, limit");
		q = Mid3Ready();
		q.flip_mid = 8;
		Check(EagerDeclineReason(q) == EagerDecline::MidLimit, "S410 decline: mode 3, limit");
		q = Mid3Ready();
		q.age_us = 100;
		Check(EagerDeclineReason(q) == EagerDecline::Age, "S410 decline: mode 3, young");
		q = Ready();
		q.age_us = 10;
		Check(EagerDeclineReason(q) == EagerDecline::Age, "S410 decline: closed pass, young");
		Check(EagerDeclineReason(Ready()) == EagerDecline::None, "S410 decline: nothing to decline");
	}
	// S410: reopen after a flush that closed a pass: the continuation loads; a clear the guest asked for in between stays.
	{
		FakeState before;
		before.depth_stencil_attachment.depth_clear = true; // the first part opened with the clear
		FakeState same = before;
		Check(Libs::Graphics::Sync::ReopenAfterFlush(before, same) && !same.depth_stencil_attachment.depth_clear,
		      "S410: the same pass after a flush loads (no repeated clear)");
		FakeState before2;
		FakeState fresh = before2;
		fresh.depth_stencil_attachment.depth_clear = true; // a new clear on the same attachments
		Check(!Libs::Graphics::Sync::ReopenAfterFlush(before2, fresh) && fresh.depth_stencil_attachment.depth_clear,
		      "S410: a new clear after a flush is kept");
		FakeState other = before;
		other.num_color_attachments = 1;
		other.depth_stencil_attachment.depth_clear = true;
		Check(!Libs::Graphics::Sync::ReopenAfterFlush(before, other) && other.depth_stencil_attachment.depth_clear,
		      "S410: another pass state is opened as asked");
	}
	std::printf("EagerSubmitTests: ok\n");
	return 0;
}
