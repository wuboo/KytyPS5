#pragma once

#include <cstdint>

// S326 (KYTY_EAGER_SUBMIT=0|1, KYTY_EAGER_SUBMIT_US, default OFF): after a drain the command processor records for a long time into one
// command buffer while the GPU queue is empty; the buffer is submitted only at a PM4 / ReleaseMem boundary (lazy_flush). The eager policy
// splits the open buffer at a safe point (between draws / dispatches) once the queue is empty, no pass is open, the buffer holds work and
// has been open for at least min_age_us. Only buffer boundaries move; the order of the recorded commands does not change.
// S327: a pass that is open may be split off only at a pass boundary (pass_closing: the next draw needs another pass state, so the old
// pass is closed right here anyway; no extra load/store).
// S328 (KYTY_EAGER_SUBMIT=2, on top of 1): the first command buffer after a drain is one long pass (the pass is lazy), so there is no pass
// boundary before its first submit. In that buffer ONE open pass may be split in the middle: EndRendering + Flush before a draw, and the next
// draw reopens the pass with loadOp=LOAD (never a clear a second time) and storeOp=STORE. Not in any later buffer, at most mid_max per segment.
namespace Libs::Graphics::Sync {

struct EagerQuery {
	bool     enabled  = false;
	bool     pass_open = false; // a render pass is open in the buffer: ending it would change the pass structure
	bool     pass_closing = false; // the open pass is being closed by the caller because the next pass has another state
	bool     has_work = false;  // something was recorded since Begin
	bool     gpu_idle = false;  // the newest submitted tick has finished (KnownGpuTick == last submitted tick)
	uint64_t age_us   = 0;      // how long the buffer has been open
	uint64_t min_age_us = 300;
	bool     mid_mode = false;        // S328: KYTY_EAGER_SUBMIT=2
	bool     first_cb_after_drain = false; // S328: nothing was submitted since the last drain ended
	uint32_t mid_splits = 0;          // S328: mid-pass splits already made in this segment
	uint32_t mid_max    = 1;
};

// S328: may an open pass that is NOT being closed be split in the middle?
inline bool MidPassAllowed(const EagerQuery& q) {
	if (!q.mid_mode) return false;
#ifndef KYTY_EAGER_MUTANT_MID_ANY_CB
	if (!q.first_cb_after_drain) return false;
#endif
#ifndef KYTY_EAGER_MUTANT_MID_NO_LIMIT
	if (q.mid_splits >= q.mid_max) return false;
#endif
	return true;
}

// S328: the reopened part of a split pass loads what the first part stored. A clear must not be repeated (it would wipe the first part).
// S is RenderState (or anything with the same three flags).
template <class S>
inline void StripClearsOnReopen(S& state) {
#ifndef KYTY_EAGER_MUTANT_CLEAR_REOPEN
	for (uint32_t i = 0; i < state.num_color_attachments; i++) state.color_attachments[i].is_clear = false;
	state.depth_stencil_attachment.depth_clear   = false;
	state.depth_stencil_attachment.stencil_clear = false;
#else
	(void)state;
#endif
}

// S410 (DIAG): the first reason ShouldEagerSplit says no for a buffer on an idle GPU (None = it says yes), in the order it checks them.
enum class EagerDecline : uint8_t { None, NoWork, PassOpen, FirstCbOnly, MidLimit, Age };
inline EagerDecline EagerDeclineReason(const EagerQuery& q) {
	if (!q.enabled || !q.has_work) return EagerDecline::NoWork;
	if (q.pass_open && !q.pass_closing) {
		if (!q.mid_mode) return EagerDecline::PassOpen;
		if (!q.first_cb_after_drain) return EagerDecline::FirstCbOnly;
		if (q.mid_splits >= q.mid_max) return EagerDecline::MidLimit;
	}
	if (q.age_us < q.min_age_us) return EagerDecline::Age;
	return EagerDecline::None;
}

inline bool ShouldEagerSplit(const EagerQuery& q) {
	if (!q.enabled || !q.has_work) return false;
#ifndef KYTY_EAGER_MUTANT_PASS_OPEN
	bool closing = q.pass_closing;
#ifdef KYTY_EAGER_MUTANT_NO_CLOSING
	closing = true; // mutant: an open pass is split off even when it is not being closed
#endif
	if (q.pass_open && !closing && !MidPassAllowed(q)) return false;
#endif
#ifndef KYTY_EAGER_MUTANT_BUSY
	if (!q.gpu_idle) return false;
#endif
#ifndef KYTY_EAGER_MUTANT_NO_AGE
	if (q.age_us < q.min_age_us) return false;
#endif
	return true;
}

} // namespace Libs::Graphics::Sync
