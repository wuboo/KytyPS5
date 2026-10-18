// S420: a GRAPHICS-style equeue event that nobody waits for keeps a bounded pending queue.
#include "graphics/host_gpu/renderer/interruptEvent.h"
#include "kernel/eventQueue.h"
#include "libs/errno.h"

#include <cstdint>
#include <cstdio>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <thread>

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

// The real trigger/reset of the GRAPHICS event (what sync.cpp installs); the counters are what sync.cpp feeds to FrameTiming.
void ResetFunc(EventQueue::KernelEqueueEvent* event) {
	Libs::Graphics::Interrupt::Reset(event);
}

void TriggerFunc(EventQueue::KernelEqueueEvent* event, void* trigger_data) {
	const auto result = Libs::Graphics::Interrupt::Trigger(event, trigger_data);
	if (result.queued) {
		if (result.coalesced) {
			g_coalesced++;
		}
		if (result.pending > g_max) {
			g_max = result.pending;
		}
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

void Trigger(EventQueue::KernelEqueue queue, int id, uintptr_t data, int16_t filter = EventQueue::KERNEL_EVFILT_GRAPHICS) {
	Check(EventQueue::KernelTriggerEvent(queue, static_cast<uintptr_t>(id), filter, reinterpret_cast<void*>(data)) == OK, "trigger");
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

void TestAmprFloodIsBounded() {
	EventQueue::KernelEqueue queue = EventQueue::KERNEL_EQUEUE_INVALID;
	Check(EventQueue::KernelCreateEqueue(&queue, "coalesce-ampr") == OK, "create");
	Check(EventQueue::KernelAddAmprEvent(queue, 5, nullptr) == OK, "add ampr event (real AmprEventTriggerFunc)");
	const size_t flood = EventQueue::kMaxPendingEvents * 10;
	for (size_t i = 1; i <= flood; ++i)
		Trigger(queue, 5, i, EventQueue::KERNEL_EVFILT_AMPR);
	intptr_t last = 0;
	const int delivered = Drain(queue, &last);
	Check(delivered == static_cast<int>(1 + EventQueue::kMaxPendingEvents), "ampr: head plus capped pending delivered");
	Check(last == static_cast<intptr_t>(flood), "ampr: the latest trigger data survives coalescing");
	Check(EventQueue::KernelDeleteEqueue(queue) == OK, "delete");
}

void TestReceivedCounters() {
	using Kind = EventQueue::ReceivedKind;
	EventQueue::KernelEqueue queue = EventQueue::KERNEL_EQUEUE_INVALID;
	Check(EventQueue::KernelCreateEqueue(&queue, "coalesce-recv") == OK, "create");
	AddGraphics(queue, 3);
	Check(EventQueue::KernelAddAmprEvent(queue, 4, nullptr) == OK, "add ampr");
	const auto all0 = EventQueue::ReceivedCount(Kind::All);
	const auto gfx0 = EventQueue::ReceivedCount(Kind::Graphics);
	const auto amp0 = EventQueue::ReceivedCount(Kind::Ampr);
	const auto vo0  = EventQueue::ReceivedCount(Kind::VideoOut);
	for (uintptr_t i = 1; i <= 7; ++i)
		Trigger(queue, 3, i);
	for (uintptr_t i = 1; i <= 3; ++i)
		Trigger(queue, 4, i, EventQueue::KERNEL_EVFILT_AMPR);
	intptr_t last = 0;
	Check(Drain(queue, &last) == 10, "ten entries delivered");
	Check(EventQueue::ReceivedCount(Kind::Graphics) - gfx0 == 7, "graphics receive counter counts what the guest received");
	Check(EventQueue::ReceivedCount(Kind::Ampr) - amp0 == 3, "ampr receive counter");
	Check(EventQueue::ReceivedCount(Kind::VideoOut) - vo0 == 0, "videoout receive counter untouched");
	Check(EventQueue::ReceivedCount(Kind::All) - all0 == 10, "total receive counter");
	Check(EventQueue::KernelDeleteEqueue(queue) == OK, "delete");
}

// A waiter that blocks on an emptied queue is woken by a trigger after the queue had been at its cap.
void TestWaiterWokenAfterFullQueue() {
	EventQueue::KernelEqueue queue = EventQueue::KERNEL_EQUEUE_INVALID;
	Check(EventQueue::KernelCreateEqueue(&queue, "coalesce-wake") == OK, "create");
	AddGraphics(queue, 9);
	for (uintptr_t i = 1; i <= EventQueue::kMaxPendingEvents * 4; ++i)
		Trigger(queue, 9, i);
	intptr_t last = 0;
	(void)Drain(queue, &last);
	int          out    = 0;
	int          result = -1;
	intptr_t     got    = 0;
	std::atomic<bool> waiting {false};
	std::thread waiter([&] {
		EventQueue::KernelEvent         event {};
		Libs::LibKernel::KernelUseconds timeout = 10'000'000; // 10 s
		waiting = true;
		result  = EventQueue::KernelWaitEqueue(queue, &event, 1, &out, &timeout);
		got     = event.data;
	});
	while (!waiting)
		std::this_thread::yield();
	std::this_thread::sleep_for(std::chrono::milliseconds(100));
	const auto begin = std::chrono::steady_clock::now();
	Trigger(queue, 9, 4242);
	waiter.join();
	const auto waited = std::chrono::steady_clock::now() - begin;
	Check(result == OK && out == 1 && got == 4242, "the waiter received the trigger made after the queue had been full");
	Check(waited < std::chrono::seconds(2), "the waiter was woken by the trigger, not by its timeout");
	Check(EventQueue::KernelDeleteEqueue(queue) == OK, "delete");
}

} // namespace

int main() {
	TestFloodIsBounded();
	TestBelowCapKeepsEveryEntry();
	TestDistinctIdentsAndWake();
	TestAmprFloodIsBounded();
	TestReceivedCounters();
	TestWaiterWokenAfterFullQueue();
	std::puts("EventQueueCoalesceTests: ok");
	return 0;
}
