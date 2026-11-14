// S601 (KYTY_BDA_DEFER, port of KytyPS5#1933): the held-back BDA events of bdaDefer.h on the real RangeSet.
// Mutants (WILL_FAIL): KYTY_BDADEFER_MUTANT_PROMOTE_DROPS, KYTY_BDADEFER_MUTANT_GPU_THREAD_DEFERRED, KYTY_BDADEFER_MUTANT_NO_FLAG.

#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/bdaDefer.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace {

using Libs::Graphics::RangeSet;
namespace BdaDefer = Libs::Graphics::BdaDefer;

constexpr uint64_t kPage = 0x1000;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "BdaDeferTests: failed: %s\n", message);
		std::_Exit(1);
	}
}

std::vector<std::pair<uint64_t, uint64_t>> Ranges(const RangeSet& set) {
	std::vector<std::pair<uint64_t, uint64_t>> out;
	set.ForEach([&out](uint64_t first, uint64_t last) { out.emplace_back(first, last); });
	return out;
}

// The BufferCache side: two sets, one flag, one mutex (QueueBdaSync / PromoteBdaSync / TakeBdaSync).
struct Cache {
	std::mutex        mutex;
	RangeSet          immediate;
	RangeSet          held;
	std::atomic<bool> has_held {false};

	void Queue(uint64_t begin, uint64_t size, bool deferred) {
		std::lock_guard lock(mutex);
		BdaDefer::Queue(immediate, held, has_held, begin, size, deferred);
	}
	bool Promote() {
		if (!BdaDefer::MayHaveHeld(has_held)) {
			return false;
		}
		std::lock_guard lock(mutex);
		return BdaDefer::Promote(immediate, held, has_held);
	}
	RangeSet Take() {
		std::lock_guard lock(mutex);
		return std::exchange(immediate, {});
	}
};

void TestParse() {
	Check(!BdaDefer::Parse(nullptr) && !BdaDefer::Parse("") && !BdaDefer::Parse("0") && !BdaDefer::Parse("2") && !BdaDefer::Parse("10"),
	      "unset / empty / 0 / 2 / 10 = off");
	Check(BdaDefer::Parse("1"), "1 = on");
}

void TestDefer() {
	Check(!BdaDefer::Defer(false, false) && !BdaDefer::Defer(false, true), "switch off: nothing is held back");
	Check(BdaDefer::Defer(true, false), "switch on: a game-thread write is held back");
	Check(!BdaDefer::Defer(true, true), "switch on: a GPU-thread write stays immediate");
}

void TestImmediate() {
	Cache c;
	c.Queue(0x10000, kPage, false);
	Check(Ranges(c.immediate).size() == 1 && c.held.Empty(), "immediate event goes to the immediate set");
	Check(!c.has_held.load() && !c.Promote(), "immediate event raises no flag; promotion has nothing to do");
	const auto taken = c.Take();
	Check(Ranges(taken) == (std::vector<std::pair<uint64_t, uint64_t>> {{0x10000, 0x11000}}), "take returns the immediate event");
}

void TestHeldUntilPromoted() {
	Cache c;
	c.Queue(0x20000, kPage, true);
	Check(c.immediate.Empty() && !c.held.Empty(), "held-back event stays out of the immediate set");
	Check(BdaDefer::MayHaveHeld(c.has_held), "held-back event raises the flag");
	Check(c.Take().Empty(), "a BDA draw before the promotion does not see the held-back event");
	Check(c.Promote(), "promotion moves the held-back event");
	Check(!c.has_held.load() && c.held.Empty(), "promotion clears the flag and the held-back set");
	Check(Ranges(c.Take()) == (std::vector<std::pair<uint64_t, uint64_t>> {{0x20000, 0x21000}}), "after the promotion the next take has it");
	Check(!c.Promote() && c.Take().Empty(), "promoted exactly once");
}

void TestPromoteIntoNonEmpty() {
	Cache c;
	c.Queue(0x30000, kPage, false);
	c.Queue(0x40000, 2 * kPage, true);
	c.Queue(0x31000, kPage, true); // adjacent to the immediate range: the union merges
	Check(c.Promote(), "promotion into a non-empty immediate set");
	Check(Ranges(c.Take()) == (std::vector<std::pair<uint64_t, uint64_t>> {{0x30000, 0x32000}, {0x40000, 0x42000}}),
	      "immediate set is the union of both");
	Check(c.held.Empty(), "held-back set emptied");
}

void TestPromoteIntoEmpty() {
	Cache c;
	c.Queue(0x50000, kPage, true);
	c.Queue(0x60000, kPage, true);
	Check(c.Promote(), "promotion into an empty immediate set (swap)");
	Check(Ranges(c.Take()) == (std::vector<std::pair<uint64_t, uint64_t>> {{0x50000, 0x51000}, {0x60000, 0x61000}}),
	      "both held-back events moved whole");
	Check(c.held.Empty() && c.immediate.Empty(), "both sets empty afterwards");
}

// A game thread queues the event of its data, then writes the guard; the CP reads the guard (acquire) and promotes: it must see the event.
void TestGuardOrdering() {
	for (int round = 0; round < 200; round++) {
		Cache            c;
		std::atomic<int> guard {0};
		std::thread      game([&] {
			c.Queue(0x70000 + static_cast<uint64_t>(round) * kPage, kPage, true);
			guard.store(1, std::memory_order_release);
		});
		while (guard.load(std::memory_order_acquire) == 0) {
		}
		std::atomic_thread_fence(std::memory_order_acquire);
		const bool moved = c.Promote();
		game.join();
		Check(moved, "promotion after the guard sees the event queued before it");
		Check(!c.Take().Empty(), "the guarded data reaches the next BDA draw");
	}
}

} // namespace

int main() {
	TestParse();
	TestDefer();
	TestImmediate();
	TestHeldUntilPromoted();
	TestPromoteIntoNonEmpty();
	TestPromoteIntoEmpty();
	TestGuardOrdering();
	std::puts("BdaDeferTests: ok");
	return 0;
}
