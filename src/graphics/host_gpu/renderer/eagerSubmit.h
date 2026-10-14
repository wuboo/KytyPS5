#pragma once

#include <cstdint>

// S326 (KYTY_EAGER_SUBMIT=0|1, KYTY_EAGER_SUBMIT_US, default OFF): after a drain the command processor records for a long time into one
// command buffer while the GPU queue is empty; the buffer is submitted only at a PM4 / ReleaseMem boundary (lazy_flush). The eager policy
// splits the open buffer at a safe point (between draws / dispatches) once the queue is empty, no pass is open, the buffer holds work and
// has been open for at least min_age_us. Only buffer boundaries move; the order of the recorded commands does not change.
namespace Libs::Graphics::Sync {

struct EagerQuery {
	bool     enabled  = false;
	bool     pass_open = false; // a render pass is open in the buffer: ending it would change the pass structure
	bool     has_work = false;  // something was recorded since Begin
	bool     gpu_idle = false;  // the newest submitted tick has finished (KnownGpuTick == last submitted tick)
	uint64_t age_us   = 0;      // how long the buffer has been open
	uint64_t min_age_us = 300;
};

inline bool ShouldEagerSplit(const EagerQuery& q) {
	if (!q.enabled || !q.has_work) return false;
#ifndef KYTY_EAGER_MUTANT_PASS_OPEN
	if (q.pass_open) return false;
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
