#pragma once

// A command-processor wait on a value in guest memory (WAIT_REG_MEM, predicated reads) can only
// end when whoever writes the value has run. With eop_on_completion the writer is a label (or
// another deferred write) that is queued on the open command buffer and stored once that buffer's
// tick retires; if the processor waits without submitting the buffer, nothing ever retires the
// tick and the wait is a self-deadlock (the class of hang from 145). So before waiting, the open
// buffer is submitted whenever a deferred write is queued at its tick.
namespace Libs::Graphics::Sync {

// `Scheduler` needs HasDeferredWriteAtOpenTick() (a label / publish is queued at the tick being
// recorded, or a label flush is due) and SubmitOpen(). Returns true when the buffer was submitted.
template <class Scheduler>
bool SubmitOpenBeforeWait(Scheduler& scheduler) {
#ifdef KYTY_CPWAIT_MUTANT_NO_SUBMIT
	(void)scheduler;
	return false; // wrong: the processor waits on a label that sits in the unsubmitted buffer
#else
	if (!scheduler.HasDeferredWriteAtOpenTick()) {
		return false;
	}
	scheduler.SubmitOpen();
	return true;
#endif
}

} // namespace Libs::Graphics::Sync
