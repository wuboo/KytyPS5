// S420: a GRAPHICS-style equeue event that nobody waits for keeps a bounded pending queue.
#include "kernel/eventQueue.h"
#include "libs/errno.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

namespace EventQueue = Libs::LibKernel::EventQueue;

size_t g_coalesced = 0;
size_t g_max       = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "EventQueueCoalesceTests: failed: %s\n", text);
		std::exit(1);
	}
}

// Same shape as Sync::InterruptEventTriggerFunc / InterruptEventResetFunc.
void ResetFunc(EventQueue::KernelEqueueEvent* event) {
	event->triggered    = false;
	event->event.fflags = 0;
	event->event.data   = 0;
}

void TriggerFunc(EventQueue::KernelEqueueEvent* event, void* trigger_data) {
	auto triggered_event = event->event;
	triggered_event.fflags++;
	triggered_event.data = reinterpret_cast<intptr_t>(trigger_data);
	if (event->triggered) {
		if (EventQueue::PushPendingEvent(*event, triggered_event)) {
			g_coalesced++;
		}
		if (event->pending_events.size() > g_max) {
			g_max = event->pending_events.size();
		}
	} else {
		event->event     = triggered_event;
		event->triggered = true;
	}
}

void AddGraphics(EventQueue::KernelEqueue queue, int id) {
	EventQueue::KernelEqueueEvent event;
	event.event.ident         = static_cast<uintptr_t>(id);
	event.event.filter        = EventQueue::KERNEL_EVFILT_GRAPHICS;
	event.filter.reset_func   = ResetFunc;
	event.filter.trigger_func = TriggerFunc;
	Check(EventQueue::KernelAddEvent(queue, event) == OK, "add graphics event");
}

void Trigger(EventQueue::KernelEqueue queue, int id, uintptr_t data) {
	Check(EventQueue::KernelTriggerEvent(queue, static_cast<uintptr_t>(id),
	                                     EventQueue::KERNEL_EVFILT_GRAPHICS,
	                                     reinterpret_cast<void*>(data)) == OK, "trigger");
}

// Drains the queue with zero timeout; returns the number of entries and the data of the last one.
int Drain(EventQueue::KernelEqueue queue, intptr_t* last_data) {
	Libs::LibKernel::KernelUseconds timeout = 0;
	EventQueue::KernelEvent         events[32] {};
	int                             total = 0;
	for (;;) {
		int out = 0;
		const int result = EventQueue::KernelWaitEqueue(queue, events, 32, &out, &timeout);
		if (result != OK || out == 0) {
			return total;
		}
		total += out;
		*last_data = events[out - 1].data;
	}
}

void TestFloodIsBounded() {
	EventQueue::KernelEqueue queue = EventQueue::KERNEL_EQUEUE_INVALID;
	Check(EventQueue::KernelCreateEqueue(&queue, "coalesce-flood") == OK, "create");
	AddGraphics(queue, 7);
	const size_t flood = EventQueue::kMaxPendingEvents * 20;
	for (size_t i = 1; i <= flood; ++i)
		Trigger(queue, 7, i);
	Check(g_max == EventQueue::kMaxPendingEvents, "pending queue stops at the cap");
	Check(g_coalesced == flood - 1 - EventQueue::kMaxPendingEvents, "every trigger past the cap coalesced");
	intptr_t last = 0;
	const int delivered = Drain(queue, &last);
	Check(delivered == static_cast<int>(1 + EventQueue::kMaxPendingEvents), "head plus capped pending delivered");
	Check(last == static_cast<intptr_t>(flood), "the latest trigger data survives coalescing");
	Check(EventQueue::KernelDeleteEqueue(queue) == OK, "delete");
}

void TestBelowCapKeepsEveryEntry() {
	EventQueue::KernelEqueue queue = EventQueue::KERNEL_EQUEUE_INVALID;
	Check(EventQueue::KernelCreateEqueue(&queue, "coalesce-below") == OK, "create");
	AddGraphics(queue, 8);
	g_coalesced = 0;
	for (uintptr_t i = 1; i <= 5; ++i)
		Trigger(queue, 8, i * 100);
	Check(g_coalesced == 0, "below the cap nothing coalesces");
	intptr_t last = 0;
	Check(Drain(queue, &last) == 5 && last == 500, "five triggers, five distinct entries");
	Trigger(queue, 8, 9);
	Check(Drain(queue, &last) == 1 && last == 9, "trigger after a receive gives one entry");
	Check(EventQueue::KernelDeleteEqueue(queue) == OK, "delete");
}

void TestDistinctIdentsAndWake() {
	EventQueue::KernelEqueue queue = EventQueue::KERNEL_EQUEUE_INVALID;
	Check(EventQueue::KernelCreateEqueue(&queue, "coalesce-idents") == OK, "create");
	AddGraphics(queue, 1);
	AddGraphics(queue, 2);
	for (uintptr_t i = 0; i < EventQueue::kMaxPendingEvents * 3; ++i) {
		Trigger(queue, 1, i + 1);
		Trigger(queue, 2, i + 1);
	}
	static EventQueue::KernelEvent events[2048] {};
	Libs::LibKernel::KernelUseconds timeout = 0;
	int out = 0;
	Check(EventQueue::KernelWaitEqueue(queue, events, 2048, &out, &timeout) == OK, "wait");
	int per_ident[3] = {};
	for (int i = 0; i < out; ++i)
		per_ident[events[i].ident]++;
	const int expected = static_cast<int>(1 + EventQueue::kMaxPendingEvents);
	Check(out == 2 * expected && per_ident[1] == expected && per_ident[2] == expected,
	      "each ident keeps its own bounded queue");
	Check(EventQueue::KernelDeleteEqueue(queue) == OK, "delete");
}

} // namespace

int main() {
	TestFloodIsBounded();
	TestBelowCapKeepsEveryEntry();
	TestDistinctIdentsAndWake();
	std::puts("EventQueueCoalesceTests: ok");
	return 0;
}
