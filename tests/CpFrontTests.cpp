// S549 (KYTY_CP_FRONT_THREAD): the hand-over between the front thread and the recorder (cpPipe.h: FrontRing, the CP-state gate on two threads, SelectDrawState).
// Every check below fails under one of the mutants (CMake: cp_front_mutant_*, WILL_FAIL):
//   FRONT_LIVE_STATE      - the recorder reads the front's live draw state instead of its item's copy (the S478 race on two threads);
//   RECORDER_WRITES_FRONT - the gate lets the recorder write the front's state outside a hand-over;
//   FRONT_EARLY_REUSE     - the front fills a slot whose previous item has not run yet.

#include "graphics/guest_gpu/command_processor/cpPipe.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::CpPipe;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpFrontTests: failed: %s\n", text);
		std::exit(1);
	}
}

void GateTests() {
	Check(MayWriteCpState(), "a plain thread (the front, Thread_Gpu between streams) writes CP state");
	{
		const RecorderScope recorder;
		Check(!MayWriteCpState(), "the recorder may not write the front's state");
		{
			const HandoffScope handoff;
			Check(MayWriteCpState(), "inside a hand-over the recorder may");
		}
		Check(!MayWriteCpState(), "the gate shuts again after the hand-over");
		{
			const WorkerScope worker;
			const HandoffScope handoff;
			Check(!MayWriteCpState(), "a hand-over never opens the gate for a worker");
		}
	}
	Check(MayWriteCpState(), "the recorder scope ends with the stream");
	std::thread([] {
		const RecorderScope recorder;
		uint32_t            member = 7;
		Check(!WriteCpMember(member, 9u) && member == 7, "WriteCpMember refuses the recorder");
	}).join();
}

void SelectTests() {
	CpDrawState live;
	live.index_type_and_size = 1;
	CpDrawState item;
	item.index_type_and_size = 0;
	Check(SelectDrawState(&item, live).index_type_and_size == 0, "the recorder reads its item's copy");
	Check(SelectDrawState(nullptr, live).index_type_and_size == 1, "outside an item the live state");
}

void PendingRetireTests() {
	PendingWrites pending;
	for (uint64_t seq = 1; seq <= 5; seq++) pending.Note(0x1000u * seq, 4, true, seq, seq);
	pending.RetireUpTo(2);
	uint64_t value = 0;
	Check(pending.Resolve(0x1000u, 4, &value) == PendingWrites::Verdict::NoPending, "item 1 ran: its write is in memory");
	Check(pending.Resolve(0x3000u, 4, &value) == PendingWrites::Verdict::Value && value == 3, "item 3 has not run");
	pending.RetireUpTo(5);
	Check(pending.Size() == 0, "everything ran");
}

// One FIFO item of the simulation: a snapshot of the front's state taken at the "draw".
struct Item {
	uint64_t    seq  = 0;
	uint32_t    kind = 0; // 0 draw, 1 sync, 2 end
	CpDrawState ds;
	uint32_t    regs[256] {};
	uint64_t    hash = 0;
};

uint64_t Expected(uint64_t draw) {
	return draw * 0x9e3779b97f4a7c15ull;
}

// The front mutates its live state between draws (as SET_*_REG / INDEX_TYPE do) and snapshots it into the item; the recorder checks that every draw sees exactly the
// state of its own place in the stream. Sync items let the recorder change the live state while the front waits; the front checks the result.
void HandOverTests() {
	constexpr size_t   kDepth = 64;
	constexpr uint64_t kItems = 200000;
	FrontRing          ring(kDepth);
	std::vector<Item>  items(kDepth);
	CpDrawState        live; // owned by the front, by the recorder inside a sync
	std::atomic<bool>  failed {false};
	std::atomic<bool>  finished {false};
	std::atomic<int>   command {0}; // a guest command: 1 posted (the front waits for it), 2 done
	uint64_t           syncs = 0;

	std::thread watchdog([&finished] {
		for (int i = 0; i < 200 && !finished.load(); i++) std::this_thread::sleep_for(std::chrono::milliseconds(100));
		Check(finished.load(), "the hand-over finishes (no deadlock) within 20 s");
	});

	std::thread recorder([&] {
		const RecorderScope scope;
		std::this_thread::sleep_for(std::chrono::milliseconds(20)); // the front fills the FIFO first: it is ahead, as in the game
		const auto idle = [&command] {
			int posted = 1;
			command.compare_exchange_strong(posted, 2); // the recorder runs the command while it waits for an item
		};
		for (uint64_t seq = 1;; seq++) {
			ring.WaitItem(seq, idle);
			Item& it = items[ring.Slot(seq)];
			if (it.seq != seq) failed.store(true);
			if (it.kind == 2) {
				ring.MarkDone(seq);
				break;
			}
			if (it.kind == 1) {
				const HandoffScope handoff;
				if (!MayWriteCpState()) failed.store(true);
				live.num_instances = static_cast<uint32_t>(seq); // the front waits: the live state is the recorder's for now
			} else {
				if (MayWriteCpState()) failed.store(true); // the gate is shut outside a hand-over
				const CpDrawState& ds   = SelectDrawState(&it.ds, live);
				const uint64_t     want = Expected(it.regs[0]);
				if (ds.index_base_addr != want || ds.index_type_and_size != it.regs[0] % 3u) failed.store(true);
				if (HashBytes(it.regs, sizeof(it.regs)) != it.hash) failed.store(true);
				for (int spin = 0; spin < 200; spin++) FrontCpuRelax(); // the recorder is the slower side
			}
			ring.MarkDone(seq);
		}
	});

	uint32_t draw = 0;
	for (uint64_t n = 0; n < kItems; n++) {
		// register writes between draws
		live.index_type_and_size = ++draw % 3u;
		live.index_base_addr     = Expected(draw);
		const bool sync          = n % 997u == 0;
		ring.WaitSlot();
		Item& it = items[ring.Slot(ring.NextSeq())];
		it.seq   = ring.NextSeq();
		if (sync) {
			it.kind              = 1;
			const uint64_t seq   = ring.Commit();
			ring.WaitDone(seq);
			if (live.num_instances != static_cast<uint32_t>(seq)) failed.store(true);
			syncs++;
			continue;
		}
		it.kind = 0;
		it.ds   = live;
		for (uint32_t& r: it.regs) r = draw;
		it.hash = HashBytes(it.regs, sizeof(it.regs));
		ring.Commit();
		if (n % 50021u == 0) { // a front that blocks on a guest command (its own fault readback) with items still queued
			command.store(1);
			while (command.load() != 2) FrontCpuRelax();
		}
		if (n % 4099u == 0) { // a drain (WAIT_REG_MEM that has to read memory)
			ring.WaitDone(ring.LastCommitted());
			if (ring.Done() != ring.LastCommitted()) failed.store(true);
		}
	}
	ring.WaitSlot();
	Item& end = items[ring.Slot(ring.NextSeq())];
	end.seq   = ring.NextSeq();
	end.kind  = 2;
	const uint64_t last = ring.Commit();
	ring.WaitDone(last);
	recorder.join();
	finished.store(true);
	watchdog.join();
	Check(!failed.load(), "every draw saw its own snapshot, every sync the front's state, slots were not reused early");
	Check(syncs > 100, "the syncs ran");
}

} // namespace

int main() {
	GateTests();
	SelectTests();
	PendingRetireTests();
	HandOverTests();
	std::printf("CpFrontTests: ok\n");
	return 0;
}
