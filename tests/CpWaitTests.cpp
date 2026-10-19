#include "graphics/host_gpu/renderer/cpWait.h"
#include "graphics/host_gpu/renderer/deferredLabel.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

// Built twice: normally, and with KYTY_CPWAIT_MUTANT_NO_SUBMIT (the processor waits on a label
// that sits in the unsubmitted buffer). The mutant hangs; the watchdog below turns the hang into a
// failure so ctest WILL_FAIL can see it.

using namespace Libs::Graphics::Sync;
using namespace std::chrono_literals;

namespace {

void Fail(const char* message) {
	std::fprintf(stderr, "CpWaitTests: failed: %s\n", message);
	std::fflush(stderr);
	std::_Exit(1); // a hung test cannot unwind; ctest WILL_FAIL needs a plain exit code
}

// The open command buffer, the label queued on it, and a GPU that completes a submitted tick a
// little later; the label writer stores the label once its tick has completed.
class FakeScheduler {
public:
	void DeferPriorityOperation(std::function<void()> operation) {
		DeferLabelOperation(std::move(operation));
	}
	void DeferLabelOperation(std::function<void()> operation) {
		std::lock_guard lock(m_mutex);
		m_labels.push_back({m_open_tick, std::move(operation)});
		m_label_flush_due = false; // (no eop_flush_now in this model: the CP has to ask)
	}
	[[nodiscard]] bool HasDeferredWriteAtOpenTick() {
		std::lock_guard lock(m_mutex);
		for (const auto& label: m_labels) {
			if (label.tick >= m_open_tick) {
				return true;
			}
		}
		return false;
	}
	void SubmitOpen() {
		std::lock_guard lock(m_mutex);
		++m_submits;
		const auto tick = m_open_tick++;
		// The GPU retires the tick shortly after; the writer then stores the labels of that tick.
		m_gpu.emplace_back([this, tick] {
			std::this_thread::sleep_for(2ms);
			std::vector<std::function<void()>> run;
			{
				std::lock_guard inner(m_mutex);
				for (auto it = m_labels.begin(); it != m_labels.end();) {
					if (it->tick <= tick) {
						run.push_back(std::move(it->operation));
						it = m_labels.erase(it);
					} else {
						++it;
					}
				}
			}
			for (auto& operation: run) {
				operation();
			}
		});
	}
	[[nodiscard]] uint32_t Submits() {
		std::lock_guard lock(m_mutex);
		return m_submits;
	}
	~FakeScheduler() {
		for (auto& gpu: m_gpu) {
			gpu.join();
		}
	}

private:
	struct Label {
		uint64_t                tick;
		std::function<void()>   operation;
	};
	std::mutex             m_mutex;
	std::vector<Label>     m_labels;
	std::vector<std::thread> m_gpu;
	uint64_t               m_open_tick = 1;
	uint32_t               m_submits   = 0;
	bool                   m_label_flush_due = false;
};

std::atomic<uint64_t> g_memory {0};

void StoreLabel(uint64_t, const void* data, uint64_t size) {
	uint64_t value = 0;
	std::memcpy(&value, data, size);
	g_memory.store(value, std::memory_order_release);
}

// WAIT_REG_MEM as the command processor runs it: test, submit the open buffer when a deferred
// write is queued, suspend and retry.
bool WaitRegMemUntil(FakeScheduler& scheduler, uint64_t expected, int max_polls) {
	for (int poll = 0; poll < max_polls; ++poll) {
		if (g_memory.load(std::memory_order_acquire) == expected) {
			return true;
		}
		(void)SubmitOpenBeforeWait(scheduler);
		std::this_thread::sleep_for(1ms); // SuspendPm4 + resume
	}
	return g_memory.load(std::memory_order_acquire) == expected;
}

void TestWaitOnLabelInOpenBuffer() {
	FakeScheduler scheduler;
	g_memory.store(0);
	// EOP label queued on the open buffer; nothing submits it (the CP is about to wait).
	DeferGuestLabelWrite(scheduler, StoreLabel, nullptr, 0, 7, 8);
	std::atomic<bool> done {false};
	std::thread       cp([&] {
        if (!WaitRegMemUntil(scheduler, 7, 1'000'000)) {
            Fail("WAIT_REG_MEM never saw the label");
        }
        done.store(true);
	});
	const auto deadline = std::chrono::steady_clock::now() + 2s;
	while (!done.load() && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::sleep_for(1ms);
	}
	if (!done.load()) {
		Fail("WAIT_REG_MEM on a label in the open buffer did not finish (self-deadlock)");
	}
	cp.join();
	if (scheduler.Submits() != 1) {
		Fail("the open buffer was submitted once, before the wait");
	}
}

void TestNoSubmitWhenNothingIsQueued() {
	FakeScheduler scheduler;
	if (SubmitOpenBeforeWait(scheduler) || scheduler.Submits() != 0) {
		Fail("an empty open buffer must not be submitted by a wait");
	}
}

} // namespace

int main() {
	TestNoSubmitWhenNothingIsQueued();
	TestWaitOnLabelInOpenBuffer();
	std::puts("CpWaitTests: ok");
	return 0;
}
