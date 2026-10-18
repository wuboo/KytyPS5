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
// S410 (KYTY_EAGER_SUBMIT=3, on top of 2): (1) the mid-pass split is allowed in EVERY buffer once it is mid3_us old, at most mid3_max per flip;
// (2) RequestFlush submits at once when the GPU is idle and the buffer is old enough (ShouldEagerFlush3), instead of waiting for 8 requests / 1 ms;
// (3) KYTY_EAGER_LOWWATER=1: a split is also allowed while the GPU is not idle but runs the LAST submitted buffer (backlog 1) and the buffer is
// lowwater_us old (LowWaterAllowed). Modes 0/1/2 do not read any of the new fields.
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
	// S410
	bool     mode3        = false; // KYTY_EAGER_SUBMIT=3
	uint64_t mid3_us      = 800;   // age a buffer needs before a mid-pass split (mode 3)
	uint32_t flip_mid     = 0;     // mid-pass splits already made in this flip (mode 3)
	uint32_t mid3_max     = 8;     // per flip
	bool     lowwater     = false; // KYTY_EAGER_LOWWATER=1 (with mode 3)
	uint64_t lowwater_us  = 1000;
	uint32_t backlog      = 0;     // submitted buffers the GPU has not finished (0 = idle)
};

// S328: may an open pass that is NOT being closed be split in the middle?
inline bool MidPassAllowed(const EagerQuery& q) {
	if (q.mode3) { // S410: any buffer, old enough, limited per flip (mid_mode = the call site is a draw that continues the open pass)
		if (!q.mid_mode) return false;
		if (q.age_us < q.mid3_us) return false;
#ifndef KYTY_EAGER_MUTANT_EAGER3_NO_LIMIT
		if (q.flip_mid >= q.mid3_max) return false;
#endif
		return true;
	}
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

// S410: the GPU is busy but runs the last submitted buffer only, so what is recorded now still reaches it before it goes idle.
inline bool LowWaterAllowed(const EagerQuery& q) {
	if (!q.mode3 || !q.lowwater) return false;
#ifdef KYTY_EAGER_MUTANT_LOWWATER_ANY_BACKLOG
	if (q.backlog == 0) return false;
#else
	if (q.backlog != 1) return false;
#endif
	return q.age_us >= q.lowwater_us;
}

// S410: RequestFlush with an idle GPU: submit the batched flush now.
inline bool ShouldEagerFlush3(const EagerQuery& q) {
	return q.enabled && q.mode3 && q.has_work && q.gpu_idle && q.age_us >= q.min_age_us;
}

// S410 (DIAG): the first reason ShouldEagerSplit says no for a buffer on an idle GPU (None = it says yes), in the order it checks them.
enum class EagerDecline : uint8_t { None, NoWork, PassOpen, FirstCbOnly, MidLimit, Age };
inline EagerDecline EagerDeclineReason(const EagerQuery& q) {
	if (!q.enabled || !q.has_work) return EagerDecline::NoWork;
	const bool closing = q.pass_closing;
	if (q.pass_open && !closing) {
		if (!q.mid_mode) return EagerDecline::PassOpen;
		if (q.mode3) {
			if (q.age_us < q.mid3_us) return EagerDecline::Age;
			if (q.flip_mid >= q.mid3_max) return EagerDecline::MidLimit;
		} else if (!q.first_cb_after_drain) {
			return EagerDecline::FirstCbOnly;
		} else if (q.mid_splits >= q.mid_max) {
			return EagerDecline::MidLimit;
		}
	}
	if (q.age_us < q.min_age_us) return EagerDecline::Age;
	return EagerDecline::None;
}

// S410: a flush (not a draw) closed the pass `before`; `wanted` is the state of the pass opened next. Only the exact same state is its continuation
// (it loads); a state with a clear `before` did not have is a new clear the guest asked for and stays. Returns true when it was a continuation.
template <class S>
inline bool ReopenAfterFlush(const S& before, S& wanted) {
	if (!(wanted == before)) return false;
	StripClearsOnReopen(wanted);
	return true;
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
	if (!q.gpu_idle && !LowWaterAllowed(q)) return false;
#endif
#ifndef KYTY_EAGER_MUTANT_NO_AGE
	if (q.age_us < q.min_age_us) return false;
#endif
	return true;
}

} // namespace Libs::Graphics::Sync
