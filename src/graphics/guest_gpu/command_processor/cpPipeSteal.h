#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPESTEAL_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPESTEAL_H

// S606 (KYTY_CP_PIPE_STEAL=0|1|count, default 0; acts with KYTY_CP_PIPE=1 and KYTY_CP_FRONT_THREAD=1 only). The recorder waits in JoinJob
// (pipe_stall_ns, 2.4 ms per garden flip on S605) for a worker that is still computing the programs of the draw it is about to record: a job
// whose gate (an ordered item: signal, CPU write, event) has just run is handed to a worker right before its draw, so the recorder waits for
// the whole job (~7-8 us on the workers) while the CP computes the same programs in ~1.5-2 us (S598 ETW: GetGraphicsPrograms 0.54 ms for
// ~395 fallbacks per flip). With 1 the recorder does not wait for a job that the cost model says finishes later than the CP's own
// computation would: it computes the programs itself (the fallback path, the reference computation, same result) and abandons the job. The
// worker finishes it (or skips it when it had not started: Cancelled); the front thread waits for it before it reuses the FIFO slot (the
// job's snapshot lives in the slot). count: the same decisions, the recorder still waits (what a steal would save, no behaviour change).
//
// Header only, no clock, no Vulkan, no FrameTiming: the state machine of a job, the cost model and the decision (tests/CpPipeStealTests.cpp).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::CpPipe::Steal {

// The states of GfpJob::state (cpPipeJob.h uses the same values). Running and Cancelled are used only while KYTY_CP_PIPE_STEAL is on.
enum JobState : uint8_t { kIdle = 0, kQueued = 1, kDone = 2, kHeld = 3, kRunning = 4, kCancelled = 5 };

enum class Mode : uint8_t { Off, On, Count };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "count") == 0) return Mode::Count;
	return Mode::Off;
}

inline const char* ModeName(Mode mode) {
	return mode == Mode::On ? "1" : (mode == Mode::Count ? "count" : "0");
}

inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_CP_PIPE_STEAL"));
	return mode;
}

inline int64_t ParseNs(const char* text, int64_t fallback, int64_t lo, int64_t hi) {
	if (text == nullptr || text[0] == '\0') return fallback;
	char*      end   = nullptr;
	const long value = std::strtol(text, &end, 10);
	if (end == text) return fallback;
	return value < lo ? lo : (value > hi ? hi : static_cast<int64_t>(value));
}

// What a steal must save on top of the CP's own computation: the worker's texture predecode is lost (S559: ~0.6 us per draw).
inline int64_t EnvMarginNs() {
	static const int64_t margin = ParseNs(std::getenv("KYTY_CP_PIPE_STEAL_MARGIN_NS"), 700, -20000, 1000000);
	return margin;
}

// A job running this much longer than its estimate is taken over whatever the estimate says (a preempted worker).
inline int64_t EnvOverdueNs() {
	static const int64_t overdue = ParseNs(std::getenv("KYTY_CP_PIPE_STEAL_OVERDUE_NS"), 15000, 0, 100000000);
	return overdue;
}

// --- the job's state machine ---------------------------------------------------------------------------------------------------------

// Worker, after popping a job: true = run it (Queued -> Running). False: the recorder cancelled it before it started; the worker only marks it Done.
inline bool WorkerStart(std::atomic<uint8_t>& state) {
	uint8_t expected = kQueued;
	if (state.compare_exchange_strong(expected, kRunning, std::memory_order_acq_rel, std::memory_order_acquire)) return true;
#ifdef KYTY_CPSTEAL_MUTANT_WORKER_RUNS_CANCELLED
	return true;
#else
	return expected != kCancelled;
#endif
}

// Recorder: take a job no worker has started (Queued -> Cancelled). False when a worker started it (or it is done).
inline bool TryCancel(std::atomic<uint8_t>& state) {
	uint8_t expected = kQueued;
	if (state.compare_exchange_strong(expected, kCancelled, std::memory_order_acq_rel, std::memory_order_acquire)) return true;
#ifdef KYTY_CPSTEAL_MUTANT_CANCEL_RUNNING
	if (expected == kRunning) {
		state.store(kCancelled, std::memory_order_release);
		return true;
	}
#endif
	return false;
}

// Recorder, the decision was Steal: true = the job is abandoned (the caller computes the programs itself; *cancelled: no worker had started it).
// False = it ended meanwhile: use it. A cancelled job turns Done as soon as a worker skips it, so it is never "ended meanwhile". `between` runs
// between the cancel and the re-check (a test hook: a worker that skips the job in that window).
template <typename Between>
inline bool Abandon(std::atomic<uint8_t>& state, bool* cancelled, Between&& between) {
	*cancelled = TryCancel(state);
	between();
#ifdef KYTY_CPSTEAL_MUTANT_USE_CANCELLED
	return state.load(std::memory_order_acquire) != kDone;
#else
	return *cancelled || state.load(std::memory_order_acquire) != kDone;
#endif
}

inline bool Abandon(std::atomic<uint8_t>& state, bool* cancelled) {
	return Abandon(state, cancelled, [] {});
}

// Front: an abandoned job's slot may be rewritten only when no worker can read it any more (Done: the worker's last store).
inline bool Reclaimable(const std::atomic<uint8_t>& state) {
	const uint8_t value = state.load(std::memory_order_acquire);
#ifdef KYTY_CPSTEAL_MUTANT_RECLAIM_RUNNING
	return value == kDone || value == kRunning;
#else
	return value == kDone;
#endif
}

// --- the cost model (recorder thread only) ---------------------------------------------------------------------------------------------

// Per shader (the job's shard key: the PS or VS address) an average of the worker's job time and of the CP's own GetGraphicsPrograms time;
// global averages for keys not seen yet. Integer EMA with weight 1/8.
class Model {
public:
	static constexpr int64_t kPriorJobNs = 7000;
	static constexpr int64_t kPriorCpNs  = 2000;
	static constexpr size_t  kSlots      = 1024;

	void NoteJob(uint64_t key, int64_t ns) {
		if (ns <= 0) return;
#ifndef KYTY_CPSTEAL_MUTANT_NO_LEARN
		Ema(m_job_global, m_job_seen, ns);
		auto& slot = Slot(key);
		if (slot.key != key) slot = Entry {key, 0, 0};
		Ema(slot.job_ns, ns);
#else
		(void)key;
#endif
	}

	void NoteCp(uint64_t key, int64_t ns) {
		if (ns <= 0) return;
		Ema(m_cp_global, m_cp_seen, ns);
		auto& slot = Slot(key);
		if (slot.key != key) slot = Entry {key, 0, 0};
		Ema(slot.cp_ns, ns);
	}

	[[nodiscard]] int64_t JobNs(uint64_t key) const {
		const auto& slot = Slot(key);
		return slot.key == key && slot.job_ns != 0 ? slot.job_ns : m_job_global;
	}

	[[nodiscard]] int64_t CpNs(uint64_t key) const {
#ifdef KYTY_CPSTEAL_MUTANT_MODEL_IGNORES_CP
		(void)key;
		return 0;
#else
		const auto& slot = Slot(key);
		return slot.key == key && slot.cp_ns != 0 ? slot.cp_ns : m_cp_global;
#endif
	}

private:
	struct Entry {
		uint64_t key    = ~uint64_t {0};
		int64_t  job_ns = 0;
		int64_t  cp_ns  = 0;
	};

	static void Ema(int64_t& value, int64_t sample) { value = value == 0 ? sample : value + (sample - value) / 8; }
	static void Ema(int64_t& value, bool& seen, int64_t sample) {
		value = seen ? value + (sample - value) / 8 : sample;
		seen  = true;
	}
	static size_t Index(uint64_t key) { return static_cast<size_t>((key >> 8u) ^ (key >> 20u)) & (kSlots - 1u); }
	Entry&        Slot(uint64_t key) { return m_slots[Index(key)]; }
	const Entry&  Slot(uint64_t key) const { return m_slots[Index(key)]; }

	Entry   m_slots[kSlots] {};
	int64_t m_job_global = kPriorJobNs;
	int64_t m_cp_global  = kPriorCpNs;
	bool    m_job_seen   = false;
	bool    m_cp_seen    = false;
};

enum class Action : uint8_t { Wait, Steal, WouldSteal };
enum class Why : uint8_t { None, Early, Overdue };

struct Decision {
	Action action = Action::Wait;
	Why    why    = Why::None;
};

// The job is not done. elapsed_ns: since a worker started it (0 when it is still queued). Steal when what is left of the job (its estimate
// minus what already ran) is longer than the CP's own computation plus the margin, or when it runs much longer than its estimate.
inline Decision Decide(Mode mode, const Model& model, uint64_t key, int64_t elapsed_ns, int64_t margin_ns, int64_t overdue_ns) {
	if (mode == Mode::Off) return {};
	const int64_t job = model.JobNs(key);
	const int64_t cp  = model.CpNs(key);
#ifdef KYTY_CPSTEAL_MUTANT_MODEL_IGNORES_ELAPSED
	const int64_t left = job;
#else
	const int64_t left = job - (elapsed_ns > 0 ? elapsed_ns : 0);
#endif
	Why why = Why::None;
	if (left > cp + margin_ns) {
		why = Why::Early;
	} else if (elapsed_ns - job > overdue_ns) {
		why = Why::Overdue;
	}
	if (why == Why::None) return {};
#ifdef KYTY_CPSTEAL_MUTANT_COUNT_STEALS
	return {Action::Steal, why};
#else
	return {mode == Mode::On ? Action::Steal : Action::WouldSteal, why};
#endif
}

} // namespace Libs::Graphics::CpPipe::Steal

#endif
