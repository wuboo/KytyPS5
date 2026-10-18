#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_INTERRUPTEVENT_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_INTERRUPTEVENT_H_

// The trigger/reset functions of the GRAPHICS (interrupt) equeue event, header-only so that tests/EventQueueCoalesceTests.cpp runs the
// code the emulator runs (sync.cpp's AddEqEvent installs InterruptEventTriggerFunc, which calls these).

#include "kernel/eventQueue.h"

#include <cstddef>

namespace Libs::Graphics::Interrupt {

namespace EventQueue = Libs::LibKernel::EventQueue;

struct TriggerResult {
	bool   coalesced = false; // the trigger replaced the last pending entry (queue at its cap)
	bool   queued    = false; // the event was already triggered: the trigger went to the pending queue
	size_t pending   = 0;     // pending_events.size() after a queued trigger
};

inline void Reset(EventQueue::KernelEqueueEvent* event) {
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

inline TriggerResult Trigger(EventQueue::KernelEqueueEvent* event, void* trigger_data) {
	TriggerResult result;
	auto          triggered_event = event->event;
	triggered_event.fflags++;
	triggered_event.data = reinterpret_cast<intptr_t>(trigger_data);
	if (event->triggered) {
		result.queued    = true;
		result.coalesced = EventQueue::PushPendingEvent(*event, triggered_event);
		result.pending   = event->pending_events.size();
	} else {
		event->event     = triggered_event;
		event->triggered = true;
	}
	return result;
}

} // namespace Libs::Graphics::Interrupt

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_INTERRUPTEVENT_H_ */
