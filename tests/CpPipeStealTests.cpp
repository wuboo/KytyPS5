// S606 (KYTY_CP_PIPE_STEAL): the state machine of a pipeline job the recorder may abandon, the cost model and the decision (cpPipeSteal.h). Every check
// ends in exit(1) when it fails (no crash), so each mutant (CMake cp_pipe_steal_mutant_*, WILL_FAIL) is caught by a failed check:
//   WORKER_RUNS_CANCELLED - a worker runs a job the recorder cancelled before it started (it would read a slot the front may rewrite);
//   CANCEL_RUNNING        - the recorder "cancels" a job a worker is running (the worker would still be reading the slot it frees);
//   RECLAIM_RUNNING       - the front rewrites the slot of an abandoned job that a worker is still running;
//   MODEL_IGNORES_ELAPSED - the decision ignores how long the job has run (it steals a job that is about to end);
//   MODEL_IGNORES_CP      - the decision ignores the CP's own cost (it steals the draws whose programs the CP computes slowly);
//   NO_LEARN              - the model does not learn the job times (it keeps stealing a shader whose jobs are short);
//   COUNT_STEALS          - count mode steals (it must only count);
//   USE_CANCELLED         - the recorder uses the "result" of a job it cancelled when a worker has marked it Done meanwhile (nothing was computed).
// No sleeps: the interleaving test is ordered by gates (atomics); the oracle runs a front, a recorder and three workers over a ring of slots.

#include "graphics/guest_gpu/command_processor/cpPipeSteal.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::CpPipe::Steal;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpPipeStealTests: failed: %s\n", text);
		std::exit(1);
	}
}

constexpr int64_t kMargin  = 700;
constexpr int64_t kOverdue = 15000;

void ModeTests() {
	Check(ParseMode(nullptr) == Mode::Off && ParseMode("0") == Mode::Off && ParseMode("yes") == Mode::Off, "default off");
	Check(ParseMode("1") == Mode::On && ParseMode("count") == Mode::Count, "1 and count");
	Check(std::string_view(ModeName(Mode::Count)) == "count" && std::string_view(ModeName(Mode::On)) == "1", "names");
	Check(ParseNs("123", 7, 0, 1000) == 123 && ParseNs(nullptr, 7, 0, 1000) == 7 && ParseNs("x", 7, 0, 1000) == 7 && ParseNs("5000", 7, 0, 1000) == 1000,
	      "ns knobs: value, default, junk, clamp");
}

void StateTests() {
	std::atomic<uint8_t> s {kQueued};
	Check(WorkerStart(s) && s.load() == kRunning, "a queued job starts");
	Check(!TryCancel(s) && s.load() == kRunning, "a running job cannot be cancelled");
	Check(!Reclaimable(s), "a running job's slot is not reclaimable");
	s.store(kQueued);
	Check(TryCancel(s) && s.load() == kCancelled, "a queued job is cancelled");
	Check(!WorkerStart(s), "the worker does not run a cancelled job");
	Check(!Reclaimable(s), "a cancelled job's slot is reclaimable only after the worker's Done");
	s.store(kDone);
	Check(Reclaimable(s) && !TryCancel(s), "a done job: reclaimable, not cancellable");
	s.store(kHeld);
	Check(!TryCancel(s) && s.load() == kHeld, "a held job is not touched by the cancel");
	// Abandon: a queued job is cancelled and stays stolen even when a worker skips it (Done) before the re-check
	bool cancelled = false;
	s.store(kQueued);
	const bool abandoned = Abandon(s, &cancelled, [&] {
		Check(!WorkerStart(s), "the skipping worker sees the cancel");
		s.store(kDone, std::memory_order_release);
	});
	Check(abandoned && cancelled, "a cancelled job that a worker skipped is still abandoned (there is no result to use)");
	s.store(kRunning);
	Check(Abandon(s, &cancelled) && !cancelled, "a running job is abandoned, not cancelled");
	s.store(kRunning);
	Check(!Abandon(s, &cancelled, [&] { s.store(kDone, std::memory_order_release); }) && !cancelled, "a running job that ends before the re-check is used");
}

void ModelTests() {
	{
		const Model m;
		Check(Decide(Mode::Off, m, 1, 0, kMargin, kOverdue).action == Action::Wait, "off never steals");
		const auto d = Decide(Mode::On, m, 1, 0, kMargin, kOverdue);
		Check(d.action == Action::Steal && d.why == Why::Early, "a job that has not started (prior 7 us vs CP 2 us): steal");
		Check(Decide(Mode::On, m, 1, 5000, kMargin, kOverdue).action == Action::Wait, "a job that ran 5 of 7 us: wait");
		Check(Decide(Mode::On, m, 1, 4200, kMargin, kOverdue).action == Action::Steal, "4.2 of 7 us: 2.8 left > 2 + 0.7, steal");
		Check(Decide(Mode::On, m, 1, 7000 + kOverdue - 100, kMargin, kOverdue).action == Action::Wait, "late but not overdue: wait");
		const auto o = Decide(Mode::On, m, 1, 7000 + kOverdue + 100, kMargin, kOverdue);
		Check(o.action == Action::Steal && o.why == Why::Overdue, "overdue (a preempted worker): steal");
		const auto c = Decide(Mode::Count, m, 1, 0, kMargin, kOverdue);
		Check(c.action == Action::WouldSteal && c.why == Why::Early, "count only counts");
	}
	{
		Model m;
		for (int i = 0; i < 4; i++) m.NoteCp(0x1000, 9000);
		Check(m.CpNs(0x1000) == 9000, "the CP cost of a shader is learned");
		Check(Decide(Mode::On, m, 0x1000, 0, kMargin, kOverdue).action == Action::Wait, "a shader the CP computes slowly: wait for the worker");
	}
	{
		Model m;
		for (int i = 0; i < 4; i++) m.NoteJob(0x2000, 1500);
		Check(m.JobNs(0x2000) == 1500, "the job time of a shader is learned");
		Check(Decide(Mode::On, m, 0x2000, 0, kMargin, kOverdue).action == Action::Wait, "a shader with short jobs: wait");
		Check(Decide(Mode::On, m, 0x3000, 0, kMargin, kOverdue).action == Action::Wait, "an unseen shader uses the learned global job time");
	}
	{
		Model m;
		m.NoteJob(0x4000, 20000);
		for (int i = 0; i < 40; i++) m.NoteJob(0x4000, 4000);
		Check(m.JobNs(0x4000) < 4500 && m.JobNs(0x4000) >= 4000, "the average follows the recent job times");
		m.NoteJob(0x4000 + (Model::kSlots << 8u), 9000); // the same slot, another shader: the slot is re-keyed
		Check(m.JobNs(0x4000 + (Model::kSlots << 8u)) == 9000, "a colliding shader takes the slot");
	}
}

// Gates, not sleeps: a worker that started the job waits until the recorder has abandoned it and the front has looked at the slot.
void InterleavingTest() {
	std::atomic<uint8_t> state {kQueued};
	std::atomic<int>     gate {0};
	uint32_t             snapshot = 0xabcd;
	std::atomic<bool>    torn {false};
	std::thread          worker([&] {
        Check(WorkerStart(state), "the worker starts the job");
        const uint32_t before = snapshot;
        gate.store(1, std::memory_order_release);
        while (gate.load(std::memory_order_acquire) != 2) std::this_thread::yield();
        if (snapshot != before) torn.store(true);
        state.store(kDone, std::memory_order_release);
	});
	while (gate.load(std::memory_order_acquire) != 1) std::this_thread::yield();
	const Model m;
	const auto  d = Decide(Mode::On, m, 7, 0, kMargin, kOverdue);
	Check(d.action == Action::Steal, "the recorder steals the fresh job");
	Check(!TryCancel(state), "a started job is not cancelled, only abandoned");
	// the front wants the slot back: not before the worker's Done
	bool rewrote = false;
	if (Reclaimable(state)) {
		snapshot = 0x1234; // the next item's snapshot
		rewrote  = true;
	}
	gate.store(2, std::memory_order_release);
	worker.join();
	Check(!rewrote && !torn.load(), "the front does not rewrite the slot of a running abandoned job");
	Check(Reclaimable(state), "after the worker's Done the slot is reclaimable");
}

// The oracle: front -> ring of slots (job + snapshot) -> recorder, three workers. The recorder steals at random or by the model; the front reclaims
// abandoned slots. Every result the recorder uses must be the one of its item's snapshot, and no worker may see its snapshot change while it runs.
struct Job {
	std::atomic<uint8_t>    state {kIdle};
	bool                    abandoned = false;
	int64_t                 steal_ns  = 0;
	std::atomic<int64_t>    run_ns {0};
	std::atomic<int64_t>    done_ns {0};
	uint64_t                seq = 0;
	std::array<uint32_t, 64> payload {};
	uint64_t                result = 0;
};

uint32_t Word(uint64_t seq, size_t i) {
	return static_cast<uint32_t>(seq * 2654435761u + i * 40503u);
}

uint64_t Expected(uint64_t seq) {
	uint64_t sum = 0;
	for (size_t i = 0; i < 64; i++) sum += Word(seq, i) ^ static_cast<uint32_t>(i);
	return sum;
}

void OracleTest(uint64_t items, bool use_model) {
	constexpr size_t kDepth = 8;
	std::array<Job, kDepth> slots;
	std::atomic<uint64_t>   committed {0};
	std::atomic<uint64_t>   done {0};
	std::mutex              qmutex;
	std::deque<Job*>        queue;
	std::atomic<bool>       stop {false};
	std::atomic<uint64_t>   torn {0};
	std::atomic<uint64_t>   wrong {0};
	std::atomic<uint64_t>   steals {0};
	std::atomic<uint64_t>   reclaim_waits {0};
	std::atomic<int64_t>    clock {1};
	const auto              now = [&] { return clock.fetch_add(1, std::memory_order_relaxed); }; // a logical clock: no time in the test

	std::vector<std::thread> workers;
	for (int w = 0; w < 3; w++) {
		workers.emplace_back([&, w] {
			std::minstd_rand rng(static_cast<uint32_t>(w) + 11u);
			for (;;) {
				Job* job = nullptr;
				{
					const std::lock_guard lock(qmutex);
					if (!queue.empty()) {
						job = queue.front();
						queue.pop_front();
					}
				}
				if (job == nullptr) {
					if (stop.load(std::memory_order_acquire)) return;
					std::this_thread::yield();
					continue;
				}
				job->run_ns.store(now(), std::memory_order_relaxed);
				if (!WorkerStart(job->state)) {
					job->state.store(kDone, std::memory_order_release);
					continue;
				}
				const uint64_t seq = job->seq;
				uint64_t       sum = 0;
				for (size_t i = 0; i < 64; i++) {
					if (job->payload[i] != Word(seq, i)) torn.fetch_add(1);
					sum += job->payload[i] ^ static_cast<uint32_t>(i);
					if ((rng() & 15u) == 0) std::this_thread::yield();
				}
				if (job->seq != seq) torn.fetch_add(1);
				job->result = sum;
				job->done_ns.store(now(), std::memory_order_relaxed);
				job->state.store(kDone, std::memory_order_release);
			}
		});
	}

	std::thread front([&] {
		for (uint64_t seq = 1; seq <= items; seq++) {
			while (seq > kDepth && done.load(std::memory_order_acquire) < seq - kDepth) std::this_thread::yield();
			Job& job = slots[seq % kDepth];
			if (job.abandoned) { // ReclaimSlot
				if (!Reclaimable(job.state)) {
					reclaim_waits.fetch_add(1);
					while (!Reclaimable(job.state)) std::this_thread::yield();
				}
				job.abandoned = false;
				job.state.store(kIdle, std::memory_order_relaxed);
			}
			job.seq = seq;
			for (size_t i = 0; i < 64; i++) job.payload[i] = Word(seq, i);
			job.result = 0;
			job.state.store(kQueued, std::memory_order_relaxed);
			{
				const std::lock_guard lock(qmutex);
				queue.push_back(&job);
			}
			committed.store(seq, std::memory_order_release);
		}
	});

	std::minstd_rand rng(7);
	Model            model;
	for (uint64_t seq = 1; seq <= items; seq++) {
		while (committed.load(std::memory_order_acquire) < seq) std::this_thread::yield();
		Job& job = slots[seq % kDepth];
		const uint64_t expected = Expected(seq); // the recorder's own computation, from its own copy of the item
		bool          stolen   = false;
		for (uint32_t i = 0; job.state.load(std::memory_order_acquire) != kDone; i++) {
			if (i == 0 || (i & 63u) == 0) {
				const uint8_t state   = job.state.load(std::memory_order_acquire);
				const int64_t elapsed = state == kRunning ? (now() - job.run_ns.load(std::memory_order_relaxed)) * 300 : 0;
				const bool    steal   = use_model ? Decide(Mode::On, model, seq & 3u, elapsed, kMargin, kOverdue).action == Action::Steal : (rng() & 1u) != 0;
				if (steal) {
					bool cancelled = false; // as WaitOrSteal
					if (!Abandon(job.state, &cancelled)) break;
					job.abandoned = true;
					job.steal_ns  = now();
					stolen        = true;
					steals.fetch_add(1);
					break;
				}
			}
			std::this_thread::yield();
		}
		if (!stolen) {
			if (job.result != expected) wrong.fetch_add(1);
			model.NoteJob(seq & 3u, 1000 + static_cast<int64_t>(seq & 3u) * 4000);
			job.state.store(kIdle, std::memory_order_relaxed); // the end of PipeFrontExecute
		} else {
			model.NoteCp(seq & 3u, 2000);
		}
		done.store(seq, std::memory_order_release);
	}
	front.join();
	stop.store(true, std::memory_order_release);
	for (auto& t: workers) t.join();
	std::printf("CpPipeStealTests: oracle %s: %llu items, %llu steals, %llu reclaim waits, %llu torn, %llu wrong\n", use_model ? "model" : "random",
	            static_cast<unsigned long long>(items), static_cast<unsigned long long>(steals.load()), static_cast<unsigned long long>(reclaim_waits.load()),
	            static_cast<unsigned long long>(torn.load()), static_cast<unsigned long long>(wrong.load()));
	Check(torn.load() == 0, "no worker sees its snapshot rewritten");
	Check(wrong.load() == 0, "every result the recorder uses is its item's");
	Check(steals.load() > 0, "the oracle stole jobs");
}

} // namespace

int main() {
	ModeTests();
	StateTests();
	ModelTests();
	InterleavingTest();
	OracleTest(20000, false);
	OracleTest(20000, true);
	std::printf("CpPipeStealTests: OK\n");
	return 0;
}
