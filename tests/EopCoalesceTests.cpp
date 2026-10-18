// S429: KYTY_EOP_COALESCE. The tracker that decides which end-of-pipe interrupts of a tick are deferred (src/graphics/guest_gpu/command_processor/
// eopCoalesce.h), driven against a model of the deferred queue: the label is written when the packet is recorded, the interrupt runs after the tick.
// Mutants: KYTY_EOPCOALESCE_MUTANT_COALESCE_ACROSS_TICKS, KYTY_EOPCOALESCE_MUTANT_DROP_CONTEXT.

#include "graphics/guest_gpu/command_processor/eopCoalesce.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

using namespace Libs::Graphics::EopCoalesce;

static int g_failed = 0;

#define CHECK(cond)                                                              \
	do {                                                                         \
		if (!(cond)) {                                                           \
			std::fprintf(stderr, "EopCoalesceTests: failed: %s (line %d)\n", #cond, __LINE__); \
			g_failed++;                                                          \
		}                                                                        \
	} while (0)

struct Delivered {
	int      event_id;
	uint32_t context_id;
	uint32_t label_seen;
};

// Stand-in for CommandProcessor::TriggerEopEventAtEndOfPipe + the priority queue: Record() writes the label at once, defers the interrupt unless
// the tracker says it is a duplicate of the tick; Complete() is the tick finishing and the deferred operations running in order.
struct Model {
	Mode                              mode;
	Tracker                           tracker;
	uint64_t                          tick = 1;
	uint32_t                          label = 0;
	std::vector<std::function<void()>> queue;
	std::vector<Delivered>            delivered;
	uint64_t                          deferred = 0, coalesced = 0, would = 0;

	void Record(int event_id, uint32_t context_id, uint32_t value) {
		label = value;
		if (mode != Mode::Off && tracker.AlreadyDeferred(tick, event_id, context_id)) {
			if (mode == Mode::On) {
				coalesced++;
				return;
			}
			would++;
		}
		deferred++;
		queue.push_back([this, event_id, context_id] { delivered.push_back({event_id, context_id, label}); });
	}
	void Complete() {
		for (auto& op: queue) {
			op();
		}
		queue.clear();
		tick++;
	}
};

static void TestParse() {
	CHECK(ParseMode(nullptr) == Mode::Off);
	CHECK(ParseMode("") == Mode::Off);
	CHECK(ParseMode("0") == Mode::Off);
	CHECK(ParseMode("1") == Mode::On);
	CHECK(ParseMode("shadow") == Mode::Shadow);
	CHECK(ParseMode("2") == Mode::Off);
}

static void TestManyInOneTick() {
	Model off {Mode::Off}, on {Mode::On}, shadow {Mode::Shadow};
	for (auto* m: {&off, &on, &shadow}) {
		for (uint32_t i = 1; i <= 32; i++) {
			m->Record(0, 7, i);
		}
		m->Complete();
	}
	CHECK(off.deferred == 32 && off.delivered.size() == 32 && off.coalesced == 0 && off.would == 0);
	CHECK(on.deferred == 1 && on.delivered.size() == 1 && on.coalesced == 31);
	CHECK(shadow.deferred == 32 && shadow.delivered.size() == 32 && shadow.would == 31 && shadow.coalesced == 0);
	// The one delivery carries the context and a label that is already the newest one (the labels are written at record time).
	CHECK(on.delivered[0].context_id == 7 && on.delivered[0].label_seen == 32);
	CHECK(off.delivered.back().context_id == on.delivered.back().context_id);
	CHECK(off.delivered.back().label_seen == on.delivered.back().label_seen);
}

static void TestDifferentContexts() {
	Model m {Mode::On};
	m.Record(0, 1, 10);
	m.Record(0, 2, 11);
	m.Record(0, 1, 12);
	m.Record(0, 2, 13);
	m.Record(0x20, 1, 14); // another event id with an already seen context
	m.Complete();
	CHECK(m.delivered.size() == 3);
	CHECK(m.coalesced == 2);
	CHECK(m.delivered[0].context_id == 1 && m.delivered[1].context_id == 2 && m.delivered[2].event_id == 0x20);
}

static void TestTicksStayApart() {
	Model m {Mode::On};
	for (int t = 0; t < 5; t++) {
		m.Record(0, 3, 100 + t);
		m.Record(0, 3, 200 + t);
		m.Complete();
	}
	CHECK(m.delivered.size() == 5);
	CHECK(m.coalesced == 5);
	for (int t = 0; t < 5; t++) {
		CHECK(m.delivered[t].label_seen == static_cast<uint32_t>(200 + t));
	}
}

static void TestLabelBeforeInterrupt() {
	Model m {Mode::On};
	m.Record(0, 0, 1);
	CHECK(m.label == 1 && m.delivered.empty()); // label written, interrupt not yet delivered
	m.Record(0, 0, 2);
	CHECK(m.label == 2 && m.delivered.empty());
	m.Complete();
	CHECK(m.delivered.size() == 1 && m.delivered[0].label_seen == 2);
}

int main() {
	TestParse();
	TestManyInOneTick();
	TestDifferentContexts();
	TestTicksStayApart();
	TestLabelBeforeInterrupt();
	if (g_failed != 0) {
		std::fprintf(stderr, "EopCoalesceTests: %d check(s) failed\n", g_failed);
		return 1;
	}
	std::printf("EopCoalesceTests: all passed\n");
	return 0;
}
