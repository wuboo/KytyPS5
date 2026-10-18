#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_GPUDONE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_GPUDONE_H_

// S406 (KYTY_DIAG_GPU_DONE=1, DIAG, default off): estimate of the time the GPU is busy, without timestamp queries.
// The submit side notes (timeline value, time after vkQueueSubmit returned); a waiter thread blocks on those values in order
// (vkWaitSemaphores) and notes the time each one completed. Busy time of one command buffer =
//     t_done(v) - max(t_submit(v), t_done(v-1))
// i.e. from the moment it could start (submitted, and the previous buffer finished) to its completion. The estimate includes the
// waiter's wake-up lag (tens of microseconds) and treats the queue as serial. Pure part: tests/GpuDoneTests.cpp pins it.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace Libs::Graphics::GpuDone {

// Serial-queue busy time from (submit, done) pairs fed in value order.
class Estimator {
public:
	// Adds the busy time of one completed buffer; returns it.
	uint64_t NoteDone(int64_t t_submit_ns, int64_t t_done_ns) {
#if defined(KYTY_GPUDONE_MUTANT_NO_MAX)
		const int64_t begin = t_submit_ns;
#else
		const int64_t begin = t_submit_ns > m_prev_done_ns ? t_submit_ns : m_prev_done_ns;
#endif
		const uint64_t busy = t_done_ns > begin ? static_cast<uint64_t>(t_done_ns - begin) : 0;
		if (t_done_ns > m_prev_done_ns) m_prev_done_ns = t_done_ns;
		m_active_ns += busy;
		m_count++;
		return busy;
	}
	[[nodiscard]] uint64_t ActiveNs() const { return m_active_ns; }
	[[nodiscard]] uint64_t Count() const { return m_count; }

private:
	int64_t  m_prev_done_ns = 0;
	uint64_t m_active_ns    = 0;
	uint64_t m_count        = 0;
};

// Queue of submitted values plus the waiter thread. `wait(value)` blocks until the GPU completed the value (or the tracker stops),
// `now()` is the clock, `on_busy(ns)` receives each increment (production: FrameTiming::Add(GpuActiveEstNs)).
class Tracker {
public:
	using Wait   = std::function<void(uint64_t value)>;
	using Clock  = std::function<int64_t()>;
	using OnBusy = std::function<void(uint64_t ns)>;

	Tracker(Wait wait, Clock now, OnBusy on_busy): m_wait(std::move(wait)), m_now(std::move(now)), m_on_busy(std::move(on_busy)) {
		m_thread = std::thread([this] { Run(); });
	}
	~Tracker() {
		{
			std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		m_cv.notify_all();
		if (m_thread.joinable()) m_thread.join();
	}
	Tracker(const Tracker&)            = delete;
	Tracker& operator=(const Tracker&) = delete;

	void NoteSubmit(uint64_t value) {
		const int64_t t = m_now();
		{
			std::lock_guard lock(m_mutex);
			m_pending.push_back({value, t});
		}
		m_cv.notify_one();
	}
	[[nodiscard]] uint64_t ActiveNs() {
		std::lock_guard lock(m_mutex);
		return m_estimator.ActiveNs();
	}
	[[nodiscard]] uint64_t Count() {
		std::lock_guard lock(m_mutex);
		return m_estimator.Count();
	}

private:
	struct Item {
		uint64_t value;
		int64_t  t_submit;
	};
	void Run() {
		for (;;) {
			Item item {};
			{
				std::unique_lock lock(m_mutex);
				m_cv.wait(lock, [this] { return m_stop || !m_pending.empty(); });
				if (m_stop) return;
				item = m_pending.front();
				m_pending.pop_front();
			}
			m_wait(item.value);
			const int64_t t_done = m_now();
			uint64_t      busy;
			{
				std::lock_guard lock(m_mutex);
				busy = m_estimator.NoteDone(item.t_submit, t_done);
			}
			if (m_on_busy) m_on_busy(busy);
		}
	}

	Wait                    m_wait;
	Clock                   m_now;
	OnBusy                  m_on_busy;
	std::mutex              m_mutex;
	std::condition_variable m_cv;
	std::deque<Item>        m_pending;
	Estimator               m_estimator;
	bool                    m_stop = false;
	std::thread             m_thread;
};

} // namespace Libs::Graphics::GpuDone

#endif
