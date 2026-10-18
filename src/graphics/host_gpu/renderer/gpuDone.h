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

// S410 (KYTY_DIAG_STARVE=1, DIAG): why a command buffer was handed to the queue. The caller of Flush/Submit sets it
// (CommandScheduler::SetSubmitReason); the starve columns are split by it. Order = order of the gpu_starve_work_ns_<reason> counters.
enum class SubmitReason : uint8_t {
	Drain,       // FlushAndWait / Finish / Wait(current tick) / readback: the CP waits for the result
	LazyMax,     // RequestFlush: KYTY_LAZY_FLUSH_MAX requests pending
	Lazy1ms,     // RequestFlush: first pending request older than 1 ms
	LazyIdle,    // FlushPending when the GPU thread is about to wait (nothing runnable)
	Eager,       // eager split at a draw / dispatch (KYTY_EAGER_SUBMIT>=1)
	EagerPass,   // eager split at a pass boundary
	EagerMid,    // KYTY_EAGER_SUBMIT=2: mid-pass split in the first buffer after a drain
	Eager3Mid,   // KYTY_EAGER_SUBMIT=3: mid-pass split in any buffer
	Eager3Flush, // KYTY_EAGER_SUBMIT=3: RequestFlush with an idle GPU
	LowWater,    // KYTY_EAGER_LOWWATER=1: split while the GPU runs the last submitted buffer
	Flip,        // flip / EOP flush of the CP
	Other,
	Count,
};
inline constexpr uint32_t kSubmitReasonCount = static_cast<uint32_t>(SubmitReason::Count);

// S410: set once by the CommandScheduler constructor (KYTY_DIAG_STARVE=1 with KYTY_DIAG_GPU_DONE=1); read on the recording path.
inline bool g_starve_diag = false;

// S410: the time the GPU queue sat empty before buffer v was submitted. `total` = max(0, t_submit(v) - t_done(v-1)); `work` leaves out
// the part in which the CP had not recorded anything for v yet: max(0, t_submit(v) - max(t_done(v-1), t_first_rec(v))).
struct Starve {
	uint64_t total_ns = 0;
	uint64_t work_ns  = 0;
};
inline Starve ComputeStarve(int64_t t_submit_ns, int64_t prev_done_ns, int64_t t_first_rec_ns) {
	Starve s;
	s.total_ns = t_submit_ns > prev_done_ns ? static_cast<uint64_t>(t_submit_ns - prev_done_ns) : 0;
#if defined(KYTY_GPUDONE_MUTANT_STARVE_NO_FIRSTREC)
	s.work_ns = s.total_ns;
	(void)t_first_rec_ns;
#else
	const int64_t begin = t_first_rec_ns > prev_done_ns ? t_first_rec_ns : prev_done_ns;
	s.work_ns           = t_submit_ns > begin ? static_cast<uint64_t>(t_submit_ns - begin) : 0;
#endif
	return s;
}

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
	// S410: like NoteDone, plus the starve of this buffer (zero for the first one: there is no previous completion).
	uint64_t NoteDoneStarve(int64_t t_submit_ns, int64_t t_first_rec_ns, int64_t t_done_ns, Starve* starve) {
		*starve = m_count == 0 ? Starve {} : ComputeStarve(t_submit_ns, m_prev_done_ns, t_first_rec_ns);
		return NoteDone(t_submit_ns, t_done_ns);
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
	using OnStarve = std::function<void(const Starve& starve, SubmitReason reason)>; // S410

	Tracker(Wait wait, Clock now, OnBusy on_busy, OnStarve on_starve = nullptr)
	    : m_wait(std::move(wait)), m_now(std::move(now)), m_on_busy(std::move(on_busy)), m_on_starve(std::move(on_starve)) {
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

	void NoteSubmit(uint64_t value, int64_t t_first_rec_ns = 0, SubmitReason reason = SubmitReason::Other) {
		const int64_t t = m_now();
		{
			std::lock_guard lock(m_mutex);
			m_pending.push_back({value, t, t_first_rec_ns, reason});
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
		uint64_t     value;
		int64_t      t_submit;
		int64_t      t_first_rec;
		SubmitReason reason;
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
			Starve        starve;
			{
				std::lock_guard lock(m_mutex);
				busy = m_estimator.NoteDoneStarve(item.t_submit, item.t_first_rec, t_done, &starve);
			}
			if (m_on_busy) m_on_busy(busy);
			if (m_on_starve) m_on_starve(starve, item.reason);
		}
	}

	Wait                    m_wait;
	Clock                   m_now;
	OnBusy                  m_on_busy;
	OnStarve                m_on_starve;
	std::mutex              m_mutex;
	std::condition_variable m_cv;
	std::deque<Item>        m_pending;
	Estimator               m_estimator;
	bool                    m_stop = false;
	std::thread             m_thread;
};

} // namespace Libs::Graphics::GpuDone

#endif
