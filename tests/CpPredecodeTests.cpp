// S558 KYTY_CP_PREDECODE (texturePredecode.h): the protocol between the pipeline workers that predecode a draw's T#s and the recorder that binds them,
// run with fake jobs on two worker threads. A job carries its own snapshot (the worker's copy of the draw's values); a page of it may be GPU-dirty when
// the recorder joins it (JoinJob rejects the job and the recorder binds its own, newer snapshot). The recorder's reference decode of the snapshot it
// binds must equal every slot it takes from a worker (shadow: predecode_ne_n = 0), and a worker writes only the job it runs (S478).
// Mutants (each must fail, WILL_FAIL): STALE_SNAPSHOT (the worker decodes the snapshot of its previous job), NO_JOIN (the recorder uses a job JoinJob
// rejected, matching the stage by its module only).

#include "graphics/host_gpu/renderer/pipeline/texturePredecode.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::Predecode::Counts;
using Libs::Graphics::Predecode::kStages;
using Libs::Graphics::Predecode::Mode;
using Libs::Graphics::Predecode::ParseMode;
using Libs::Graphics::Predecode::RunStage;
using Libs::Graphics::Predecode::Stage;
using Libs::Graphics::Predecode::Status;
using Libs::Graphics::Predecode::Usable;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "CpPredecodeTests: failed: %s\n", what);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

struct Value { // a T# of the snapshot
	std::array<uint32_t, 8> dwords {};
	bool operator==(const Value&) const = default;
};

struct Slot {
	Status   status = Status::Decline;
	uint64_t key    = 0; // the "description"
	bool operator==(const Slot&) const = default;
};

// The pure decode both sides run: null T# -> Null, a T# with bit 31 of dword 1 -> Decline (the recorder resolves it), else a description that
// depends on every dword and on the image's resource (its index stands for the program's ImageResource).
Status Decode(size_t index, const Value& value, Slot& slot) {
	if (value.dwords[0] == 0) return Status::Null;
	if ((value.dwords[1] & 0x80000000u) != 0) return Status::Decline;
	uint64_t h = 0xcbf29ce484222325ull ^ index;
	for (const auto word: value.dwords) h = (h ^ word) * 0x100000001b3ull;
	slot.key = h;
	return Status::Ok;
}

struct Module {
	size_t images = 0;
};

struct Job {
	enum : uint8_t { Queued, Done };
	std::atomic<uint8_t>                     state {Queued};
	std::array<const Module*, kStages>       module {};
	std::array<std::vector<Value>, kStages>  snapshot; // the job's own copy (out.vs_res / out.ps_res)
	bool                                     answered = true;
	bool                                     dirty    = false; // a page the worker read is GPU-dirty at the join
	std::array<Stage<Slot>, kStages>         pre;
};

Value MakeValue(uint32_t job, uint32_t stage, uint32_t image) {
	Value v;
	const uint32_t seed = job * 131u + stage * 17u + image;
	v.dwords[0] = (seed % 7u) == 0 ? 0u : seed + 1u; // some null T#s
	v.dwords[1] = (seed % 11u) == 0 ? 0x80000000u : seed * 3u; // some declined
	for (uint32_t i = 2; i < 8; i++) v.dwords[i] = seed ^ (i * 0x9e3779b9u);
	return v;
}

// The bytes the GPU wrote into the page after the worker read it: the recorder's own snapshot differs in every non-null T#.
std::vector<Value> Rewritten(const std::vector<Value>& values) {
	auto out = values;
	for (auto& v: out) {
		if (v.dwords[0] != 0) v.dwords[2] ^= 0x5a5a5a5au;
	}
	return out;
}

class Workers {
public:
	explicit Workers(size_t count) {
		for (size_t i = 0; i < count; i++) m_threads.emplace_back([this] { Run(); });
	}
	~Workers() {
		{
			const std::lock_guard lock(m_mutex);
			m_stop = true;
		}
		for (auto& t: m_threads) t.join();
	}
	void Submit(Job* job) {
		const std::lock_guard lock(m_mutex);
		m_jobs.push_back(job);
	}

private:
	void Run() {
		for (;;) {
			Job* job = nullptr;
			{
				const std::lock_guard lock(m_mutex);
				if (!m_jobs.empty()) {
					job = m_jobs.front();
					m_jobs.pop_front();
				} else if (m_stop) {
					return;
				}
			}
			if (job == nullptr) {
				std::this_thread::yield();
				continue;
			}
			// RunGfpJob: clear, then (answered) decode each stage from the job's own snapshot, then publish.
			for (auto& stage: job->pre) stage.Clear();
			if (job->answered) {
				for (size_t s = 0; s < kStages; s++) {
					if (job->module[s] == nullptr) continue;
					RunStage(job->pre[s], s, job->module[s], &job->snapshot[s], job->snapshot[s], job->module[s]->images, Decode);
				}
			}
			job->state.store(Job::Done, std::memory_order_release);
		}
	}

	std::mutex               m_mutex;
	std::deque<Job*>         m_jobs;
	bool                     m_stop = false;
	std::vector<std::thread> m_threads;
};

struct Totals {
	Counts   counts;
	uint64_t images  = 0;
	uint64_t dirty_used = 0; // slots taken from a job JoinJob rejected
};

// The recorder: joins the jobs in order and binds both stages of each draw as PrepareBindings does.
void Record(std::vector<std::unique_ptr<Job>>& jobs, Totals& totals, std::vector<std::array<Stage<Slot>, kStages>>& consumed) {
	for (auto& job_ptr: jobs) {
		Job& job = *job_ptr;
		while (job.state.load(std::memory_order_acquire) != Job::Done) std::this_thread::yield();
		const bool joined = job.answered && !job.dirty;
		for (size_t s = 0; s < kStages; s++) {
			if (job.module[s] == nullptr) continue;
			// CP_PIPE=1: a joined job's stage runs on the job's snapshot; a rejected one on the recorder's own, newer one.
			const std::vector<Value>  own       = Rewritten(job.snapshot[s]);
			const std::vector<Value>& bound     = joined ? job.snapshot[s] : own;
			const void*               resources = joined ? static_cast<const void*>(&job.snapshot[s]) : static_cast<const void*>(&own);
			const auto*               pre       = Usable(job.pre, joined, job.module[s], resources, job.module[s]->images);
			for (size_t i = 0; i < job.module[s]->images; i++) {
				totals.images++;
				Slot       reference;
				const auto status = Decode(i, bound[i], reference);
				reference.status  = status;
				if (pre == nullptr) {
					totals.counts.nojob++;
					continue;
				}
				const auto& slot = pre->slots[i];
				if (slot.status == Status::Decline) {
					totals.counts.decline++;
					continue;
				}
				totals.counts.img++;
				if (!joined) totals.dirty_used++;
				const bool same = slot.status == reference.status && (slot.status != Status::Ok || slot.key == reference.key);
				if (!same) totals.counts.ne++;
			}
		}
		consumed.push_back(job.pre);
	}
}

void TestModes() {
	Check(ParseMode(nullptr) == Mode::Off, "unset is off");
	Check(ParseMode("") == Mode::Off, "empty is off");
	Check(ParseMode("0") == Mode::Off, "0 is off");
	Check(ParseMode("1") == Mode::On, "1 is on");
	Check(ParseMode("shadow") == Mode::Shadow, "shadow");
	Check(ParseMode("on") == Mode::Off, "junk is off");
}

void TestUsable() {
	Module a {3}, b {3};
	std::vector<Value> snap_a(3), other(3);
	std::array<Stage<Slot>, kStages> pre;
	RunStage(pre[0], 0, &a, &snap_a, snap_a, 3, Decode);
	Check(Usable(pre, true, &a, &snap_a, 3) == &pre[0], "joined stage on its snapshot is usable");
	Check(Usable(pre, true, &b, &snap_a, 3) == nullptr, "another module is not usable");
	Check(Usable(pre, true, &a, &snap_a, 4) == nullptr, "another image count is not usable");
#ifndef KYTY_PREDECODE_MUTANT_NO_JOIN
	Check(Usable(pre, true, &a, &other, 3) == nullptr, "another snapshot is not usable");
	Check(Usable(pre, false, &a, &snap_a, 3) == nullptr, "a rejected job is not usable");
#endif
	std::array<Stage<Slot>, kStages> empty;
	Check(Usable(empty, true, &a, &snap_a, 3) == nullptr, "an empty job is not usable");
}

void TestProtocol() {
	constexpr uint32_t kJobs = 4000;
	Module vs {5}, ps {9};
	std::vector<std::unique_ptr<Job>> jobs;
	jobs.reserve(kJobs);
	for (uint32_t j = 0; j < kJobs; j++) {
		auto job       = std::make_unique<Job>();
		job->module[0] = &vs;
		job->module[1] = (j % 5u) == 4u ? nullptr : &ps; // some draws without a pixel stage
		for (size_t s = 0; s < kStages; s++) {
			if (job->module[s] == nullptr) continue;
			for (uint32_t i = 0; i < job->module[s]->images; i++) job->snapshot[s].push_back(MakeValue(j, static_cast<uint32_t>(s), i));
		}
		job->answered = (j % 23u) != 0;  // the worker could not answer
		job->dirty    = (j % 13u) == 6u; // JoinJob rejects it
		jobs.push_back(std::move(job));
	}
	std::vector<std::array<Stage<Slot>, kStages>> consumed;
	Totals                                        totals;
	{
		Workers workers(2);
		for (auto& job: jobs) workers.Submit(job.get());
		Record(jobs, totals, consumed);
	}
	std::fprintf(stderr, "CpPredecodeTests: images=%llu img=%llu decline=%llu nojob=%llu ne=%llu dirty_used=%llu\n",
	             static_cast<unsigned long long>(totals.images), static_cast<unsigned long long>(totals.counts.img),
	             static_cast<unsigned long long>(totals.counts.decline), static_cast<unsigned long long>(totals.counts.nojob),
	             static_cast<unsigned long long>(totals.counts.ne), static_cast<unsigned long long>(totals.dirty_used));
	Check(totals.counts.img + totals.counts.decline + totals.counts.nojob == totals.images, "every image counted once");
	Check(totals.counts.img * 10 > totals.images * 7, "most images come from the workers");
	Check(totals.counts.decline > 0 && totals.counts.nojob > 0, "declines and rejected jobs are exercised");
	Check(totals.counts.ne == 0, "every slot taken equals the recorder's decode of the snapshot it binds (predecode_ne_n = 0)");
	Check(totals.dirty_used == 0, "no slot of a job JoinJob rejected is used");
	// S478: after the recorder consumed a job nobody wrote it again, and each job's stages name its own snapshot.
	for (uint32_t j = 0; j < kJobs; j++) {
		for (size_t s = 0; s < kStages; s++) {
			const auto& now    = jobs[j]->pre[s];
			const auto& before = consumed[j][s];
			Check(now.module == before.module && now.resources == before.resources && now.slots == before.slots, "a job changed after it was consumed");
			if (now.module != nullptr) Check(now.resources == &jobs[j]->snapshot[s], "a worker decoded another job's snapshot object");
		}
	}
}

} // namespace

int main() {
	TestModes();
	TestUsable();
	TestProtocol();
	std::fprintf(stderr, "CpPredecodeTests: OK\n");
	return 0;
}
