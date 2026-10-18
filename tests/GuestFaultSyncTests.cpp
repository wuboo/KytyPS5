// S428 (B2): the GPU-modified ranges as the BufferCache hands them to a signal handler (gpuDirtyRanges.h).
//  - hand-over order: a range becomes "in flight" before it stops being "dirty"; at no point of BeginDownloadCopy is it neither;
//  - TryIntersects never waits for the command processor: with the lock held by another thread it answers "busy" at once (the fault takes the ordinary drain).
// Mutants (WILL_FAIL): SUBTRACT_FIRST (dirty is removed before in-flight is added), BLOCKING_QUERY (the handler's query waits for the lock).

#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace Libs::Graphics;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

static bool Dirty(const GpuDirtyRanges& d, uint64_t a, uint64_t n) { return d.Intersects(a, n); }

int main() {
	// Order of the hand-over.
	{
		GpuDirtyRanges d;
		InflightRanges inflight;
		d.Add(0x10000, 0x3000);
		d.Add(0x20000, 0x100);
		std::vector<std::pair<uint64_t, uint64_t>> pieces;
		int                                        windows = 0;
		d.BeginDownloadCopy(
		    inflight, 0x10800, 0x2000, [&](uint64_t s, uint64_t e) { pieces.emplace_back(s, e); },
		    [&] {
			    ++windows;
			    // The window between the two steps: every byte is still dirty, every piece is already in flight.
			    for (const auto& [s, e]: pieces) {
				    Check(inflight.Intersects(s, e - s), "window: the piece is already in flight");
				    Check(Dirty(d, s, e - s), "window: the piece is still dirty");
			    }
			    for (uint64_t a = 0x10800; a < 0x12800; a += 0x100) {
				    Check(Dirty(d, a, 1) || inflight.Intersects(a, 1), "window: every byte is dirty or in flight (never neither)");
			    }
		    });
		Check(windows == 1, "the window callback ran once");
		Check(pieces.size() == 1 && pieces[0].first == 0x10800 && pieces[0].second == 0x12800, "one piece, clipped to the range");
		Check(!Dirty(d, 0x10800, 0x2000), "after: the range is no longer dirty");
		Check(Dirty(d, 0x10000, 0x800) && Dirty(d, 0x12800, 0x800), "after: the rest of the dirty range stays");
		Check(Dirty(d, 0x20000, 0x100), "after: an unrelated dirty range stays");
		Check(inflight.Intersects(0x10800, 0x2000), "after: the range is in flight until the publish removes it");
		inflight.Remove(0x10800, 0x2000);
		Check(!inflight.Intersects(0x10800, 0x2000), "published");
		Check(inflight.Count() == 0, "no entry left");
	}
	// Several pieces.
	{
		GpuDirtyRanges d;
		InflightRanges inflight;
		d.Add(0x1000, 0x100);
		d.Add(0x1400, 0x100);
		int pieces = 0;
		d.BeginDownloadCopy(inflight, 0x1000, 0x1000, [&](uint64_t, uint64_t) { ++pieces; });
		Check(pieces == 2 && inflight.Count() == 2, "two dirty pieces, two in-flight entries");
		Check(d.Empty(), "all of it handed over");
	}
	// The handler's query: answers while nobody holds the lock, "busy" at once while the command processor does.
	{
		GpuDirtyRanges d;
		d.Add(0x5000, 0x100);
		bool dirty = false;
		Check(d.TryIntersects(0x5000, 0x10, &dirty) && dirty, "free lock: dirty answer");
		Check(d.TryIntersects(0x6000, 0x10, &dirty) && !dirty, "free lock: clean answer");

		std::atomic<bool> held {false};
		std::thread       holder([&] {
            std::lock_guard lock(d.MutexForTest());
            held = true;
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        });
		while (!held) {
			std::this_thread::yield();
		}
		dirty              = false;
		const auto begin   = std::chrono::steady_clock::now();
		const bool answered = d.TryIntersects(0x5000, 0x10, &dirty);
		const auto waited   = std::chrono::steady_clock::now() - begin;
		Check(!answered, "locked: busy, no answer");
		Check(waited < std::chrono::milliseconds(100), "locked: the query returned without waiting for the lock");
		holder.join();
		Check(d.TryIntersects(0x5000, 0x10, &dirty) && dirty, "released: answers again");
	}
	// The same for the in-flight set.
	{
		InflightRanges inflight;
		inflight.Add(0x7000, 0x100);
		bool in = false;
		Check(inflight.TryIntersects(0x7000, 0x10, &in) && in, "in-flight: answer");
		Check(inflight.TryIntersects(0x8000, 0x10, &in) && !in, "in-flight: clean answer");
	}
	if (g_failures != 0) {
		std::fprintf(stderr, "GuestFaultSyncTests: %d failures\n", g_failures);
		return 1;
	}
	std::printf("GuestFaultSyncTests: all passed\n");
	return 0;
}
