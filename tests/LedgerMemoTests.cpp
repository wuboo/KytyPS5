// S562 KYTY_LEDGER_MEMO (gpuWriteLedger.h): GpuWriteLedger::LatestIsFillMemo answers, in modes 1 and shadow, what LatestIsFill answers, over writes,
// fills, Forget and Trim of overlapping ranges, two ledgers that hold different fills of the same range (the per-thread memo is shared), a ledger
// destroyed and another built in its place, and a writer thread racing with the reader (checked after it joins).
// Mutants (must fail, WILL_FAIL): MEMO_NO_EPOCH (the memo ignores ledger changes), MEMO_NO_OWNER (the memo ignores which ledger it is).

#include "graphics/host_gpu/renderer/cache/gpuWriteLedger.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

namespace {

using namespace Libs::Graphics;

void Check(bool ok, const char* what, int step) {
	if (!ok) {
		std::fprintf(stderr, "LedgerMemoTests: failed: %s (step %d)\n", what, step);
		std::exit(1);
	}
}

int g_step = 0;

void Same(const GpuWriteLedger& ledger, uint64_t vaddr, uint64_t size, LedgerMemo::Mode mode, LedgerMemo::Stats& stats) {
	uint32_t   a = 0xdead;
	uint32_t   b = 0xbeef;
	const bool known_a = ledger.LatestIsFill(vaddr, size, &a);
	const bool known_b = ledger.LatestIsFillMemo(vaddr, size, &b, mode, stats);
	Check(known_a == known_b && (!known_a || a == b), "memo answer differs from the scan", g_step);
	g_step++;
}

void Run(LedgerMemo::Mode mode) {
	LedgerMemo::Stats stats;
	GpuWriteLedger    ledger;
	const uint64_t    base = 0x100000;
	for (int round = 0; round < 3; round++) {
		Same(ledger, base, 0x1000, mode, stats);   // nothing recorded
		ledger.RecordFill(base, 0x4000, 0x11 + round);
		Same(ledger, base, 0x1000, mode, stats);   // known
		Same(ledger, base, 0x1000, mode, stats);   // again (memo)
		Same(ledger, base + 0x1000, 0x1000, mode, stats);
		ledger.RecordWrite(base + 0x800, 0x100);   // a newer write over part of it
		Same(ledger, base, 0x1000, mode, stats);
		Same(ledger, base + 0x1000, 0x1000, mode, stats);
		ledger.RecordFill(base, 0x1000, 0x22);
		Same(ledger, base, 0x1000, mode, stats);
		ledger.Forget(base, 0x10);
		Same(ledger, base, 0x1000, mode, stats);
		ledger.RecordFill(base, 0x4000, 0x33);
		Same(ledger, base + 0x2000, 0x1000, mode, stats);
		ledger.Trim(base + 0x3000, 0x1000);
		Same(ledger, base + 0x2000, 0x1000, mode, stats);
		Same(ledger, base + 0x3000, 0x1000, mode, stats);
	}
	// Two ledgers, the same range, different fills.
	GpuWriteLedger other;
	other.RecordFill(base, 0x1000, 0x77);
	ledger.RecordFill(base, 0x1000, 0x55);
	for (int i = 0; i < 4; i++) {
		Same(ledger, base, 0x1000, mode, stats);
		Same(other, base, 0x1000, mode, stats);
	}
	// A ledger destroyed and another built in its place.
	for (int i = 0; i < 4; i++) {
		auto fresh = std::make_unique<GpuWriteLedger>();
		if (i % 2 == 0) fresh->RecordFill(base, 0x1000, 0x90 + i);
		Same(*fresh, base, 0x1000, mode, stats);
		Same(*fresh, base, 0x1000, mode, stats);
	}
	// A writer racing with the reader: no check while it runs (an answer may be older), the memo must catch up after the join.
	{
		std::thread writer([&] {
			for (uint32_t i = 0; i < 20000; i++) {
				if (i % 3 == 0) ledger.RecordWrite(base + (i % 4) * 0x400, 0x400);
				else ledger.RecordFill(base, 0x1000, i);
			}
		});
		uint32_t value = 0;
		for (int i = 0; i < 20000; i++) (void)ledger.LatestIsFillMemo(base, 0x1000, &value, mode, stats);
		writer.join();
		Same(ledger, base, 0x1000, mode, stats);
		Same(ledger, base + 0x400, 0x400, mode, stats);
	}
	Check(stats.ne == 0, "shadow disagreements", g_step);
	Check(stats.hit >= 10, "too few memo hits", g_step);
	std::printf("LedgerMemoTests: mode %s: hit %llu miss %llu ne %llu\n", TexMemo::ModeName(mode), static_cast<unsigned long long>(stats.hit),
	            static_cast<unsigned long long>(stats.miss), static_cast<unsigned long long>(stats.ne));
}

} // namespace

int main() {
	Run(LedgerMemo::Mode::On);
	Run(LedgerMemo::Mode::Shadow);
	std::printf("LedgerMemoTests: OK\n");
	return 0;
}
