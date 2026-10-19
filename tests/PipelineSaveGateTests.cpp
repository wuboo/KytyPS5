// S437: when the driver pipeline cache is written (pipelineSaveGate.h). Mutants: KYTY_PSGATE_MUTANT_{HIT_DIRTIES, NO_DEBOUNCE, UNKNOWN_IS_HIT}.

#include "graphics/host_gpu/renderer/pipeline/pipelineSaveGate.h"

#include <cstdio>

using namespace Libs::Graphics;
using Clock = PipelineSaveGate::Clock;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

constexpr auto S(int seconds) { return std::chrono::seconds(seconds); }

void TestFeedbackClassification() {
	Check(ClassifyFeedback(0) == PipelineCacheUse::Unknown, "feedback without the VALID bit is unknown");
	Check(ClassifyFeedback(kFeedbackCacheHitBit) == PipelineCacheUse::Unknown, "a hit bit without VALID is not trusted");
	Check(ClassifyFeedback(kFeedbackValidBit) == PipelineCacheUse::Miss, "valid, no hit bit: the driver compiled it");
	Check(ClassifyFeedback(kFeedbackValidBit | kFeedbackCacheHitBit) == PipelineCacheUse::Hit, "valid + hit bit: the driver cache had it");
}

void TestHitDoesNotDirty() {
	PipelineSaveGate gate;
	const auto       t0 = Clock::now();
	for (int i = 0; i < 50; i++) Check(!gate.NoteCreated(PipelineCacheUse::Hit, t0 + S(i)), "a cache hit does not make the cache dirty");
	Check(!gate.Dirty() && !gate.Due(t0 + S(100)), "50 hits: nothing to write");
	Check(gate.Skipped() == 50 && gate.TakeSkippedDelta() == 50 && gate.TakeSkippedDelta() == 0, "skipped counter: 50, then delta 0");
}

void TestMissAndUnknownDirty() {
	PipelineSaveGate gate;
	const auto       t0 = Clock::now();
	Check(gate.NoteCreated(PipelineCacheUse::Miss, t0) && gate.Dirty(), "a miss makes the cache dirty");
	PipelineSaveGate other;
	Check(other.NoteCreated(PipelineCacheUse::Unknown, t0) && other.Dirty(), "unknown feedback is treated as a miss");
}

void TestDebounce() {
	PipelineSaveGate gate;
	const auto       t0 = Clock::now();
	gate.NoteCreated(PipelineCacheUse::Miss, t0);
	Check(!gate.Due(t0 + std::chrono::milliseconds(500)), "not due 0.5 s after the change");
	gate.NoteCreated(PipelineCacheUse::Miss, t0 + S(1));
	Check(!gate.Due(t0 + std::chrono::milliseconds(2500)), "a newer pipeline moves the debounce");
	Check(gate.Due(t0 + S(3)), "due 2 s after the last change");
	gate.Cleared();
	Check(!gate.Dirty() && !gate.Due(t0 + S(10)), "cleared: nothing to write");
}

void TestMaxDelay() {
	PipelineSaveGate gate;
	const auto       t0 = Clock::now();
	for (int i = 0; i < 40; i++) gate.NoteCreated(PipelineCacheUse::Miss, t0 + S(i)); // a new pipeline every second for 40 s
	Check(gate.Due(t0 + S(40)), "a continuous burst is written at the latest 30 s after it started");
	PipelineSaveGate burst;
	for (int i = 0; i < 20; i++) burst.NoteCreated(PipelineCacheUse::Miss, t0 + S(i));
	Check(!burst.Due(t0 + S(20)), "a burst of 20 s with a pipeline every second is not due yet");
}

void TestFailureRetries() {
	PipelineSaveGate gate;
	const auto       t0 = Clock::now();
	gate.NoteCreated(PipelineCacheUse::Miss, t0);
	gate.Cleared();
	gate.Failed(t0 + S(1));
	Check(gate.Dirty() && gate.Due(t0 + S(1)), "a failed write is retried at the next chance");
}

// The run that used to write 52 identical files: pipelines come in as driver cache hits, with a few real misses.
void TestWritesInARun() {
	PipelineSaveGate gate;
	const auto       t0     = Clock::now();
	int              writes = 0;
	for (int i = 0; i < 400; i++) {
		const auto now = t0 + std::chrono::milliseconds(i * 500); // 200 s
		if (gate.Due(now)) {
			writes++;
			gate.Cleared();
		}
		const bool real_miss = i == 40 || i == 41 || i == 200;
		gate.NoteCreated(real_miss ? PipelineCacheUse::Miss : PipelineCacheUse::Hit, now);
	}
	Check(writes == 2, "398-pipeline run with 3 real misses: 2 writes, one per burst");
}

} // namespace

int main() {
	TestFeedbackClassification();
	TestHitDoesNotDirty();
	TestMissAndUnknownDirty();
	TestDebounce();
	TestMaxDelay();
	TestFailureRetries();
	TestWritesInARun();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("pipeline save gate tests passed");
	return 0;
}
