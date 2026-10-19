// S439 DIAG: PC sampling of the CP thread (cpPcSampler.h). Covers the histogram, the JIT registry and symbolization, ReadPc on a spinning
// thread (under Rosetta the thread is translated: the PC must still land in the spinning function), thread CPU time vs wall time, and the
// walk tally. Mutants: KYTY_CPPC_MUTANT_NO_REGISTER (RegisterJit does nothing), KYTY_CPPC_MUTANT_NO_READS (NoteReads counts nothing).

#include "graphics/presentation/cpPcSampler.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

using namespace Libs::Graphics::CpSampler::Pc;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

static std::atomic<bool>     g_stop {false};
static std::atomic<uint32_t> g_port {0};
static std::atomic<uint64_t> g_sink {0};

extern "C" __attribute__((noinline)) void KytyPcTestSpinHere() {
	uint64_t x = 0;
	while (!g_stop.load(std::memory_order_relaxed)) {
		for (int i = 0; i < 1000; ++i) x += static_cast<uint64_t>(i) * 3u;
		g_sink.store(x, std::memory_order_relaxed);
	}
}
extern "C" __attribute__((noinline)) void KytyPcTestAfterSpin() {
	g_sink.fetch_add(1);
}

int main() {
	// Default state: nothing is on without the env / the explicit store.
	Check(!Enabled(), "Enabled() is false without KYTY_DIAG_CP_PC");
	Check(!WalkActive().load(), "walk tally gate is off by default");

	// Histogram.
	{
		PcHistogram hist;
		hist.Add(0x2000);
		hist.Add(0x1000);
		hist.Add(0x1000);
		hist.Add(0x3000);
		hist.Add(0x3000);
		hist.AddFail();
		Check(hist.Total() == 5 && hist.Fail() == 1 && hist.Distinct() == 3, "histogram counts");
		const auto top = hist.Top(2);
		Check(top.size() == 2 && top[0].first == 0x1000 && top[0].second == 2 && top[1].first == 0x3000, "top: count desc, ties by lower pc");
		hist.Clear();
		Check(hist.Total() == 0 && hist.Fail() == 0 && hist.Distinct() == 0, "clear");
	}

	// JIT registry and symbolization.
	{
		static uint8_t buffer[512];
		static uint8_t other[64];
		RegisterJit(buffer, sizeof(buffer), JitKind::Slots, 0xABCDEF0123456789ull);
		JitKind  kind;
		uint64_t hash, offset;
		Check(LookupJit(reinterpret_cast<uint64_t>(buffer) + 100, kind, hash, offset) && kind == JitKind::Slots && offset == 100 &&
		          hash == 0xABCDEF0123456789ull,
		      "pc inside a registered range resolves");
		Check(!LookupJit(reinterpret_cast<uint64_t>(buffer) + sizeof(buffer), kind, hash, offset), "one past the end is outside");
		Check(Symbolize(reinterpret_cast<uint64_t>(buffer) + 16).rfind("jit:slots:abcdef0123456789+0x10", 0) == 0, "jit symbol text");
		// Newest registration wins on overlap (address reuse after a free).
		RegisterJit(buffer + 128, 64, JitKind::Tree, 7);
		Check(LookupJit(reinterpret_cast<uint64_t>(buffer) + 130, kind, hash, offset) && kind == JitKind::Tree && hash == 7 && offset == 2,
		      "newest registration wins");
		Check(Symbolize(reinterpret_cast<uint64_t>(other)).rfind("jit:", 0) != 0, "unregistered address is not a jit symbol");
		Check(SymbolOnly("jit:slots:abc+0x10") == "jit:slots:abc" && SymbolOnly("plain") == "plain", "SymbolOnly drops the offset");
	}
	{
		const auto sym = Symbolize(reinterpret_cast<uint64_t>(&KytyPcTestAfterSpin));
		std::printf("dladdr symbol of KytyPcTestAfterSpin: %s\n", sym.c_str());
		Check(sym.find("KytyPcTestAfterSpin") != std::string::npos || sym.find("CpPcSamplerTests") != std::string::npos ||
		          sym.find("cp_pc_sampler") != std::string::npos,
		      "dladdr names the function or its image");
	}

	// ReadPc / ThreadCpuNs on a spinning thread, and a sleeping one.
#if defined(__APPLE__)
	{
		std::thread spinner([] {
			g_port.store(mach_thread_self());
			KytyPcTestSpinHere();
		});
		while (g_port.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		const auto port = static_cast<mach_port_t>(g_port.load());
		std::this_thread::sleep_for(std::chrono::milliseconds(20));

		uint64_t cpu0 = 0, cpu1 = 0;
		Check(ThreadCpuNs(port, cpu0), "ThreadCpuNs works on the spinner");
		const auto wall0 = std::chrono::steady_clock::now();
		int        ok = 0, in_fn = 0;
		PcHistogram hist;
		for (int i = 0; i < 300; ++i) {
			uint64_t pc = 0;
			if (ReadPc(port, pc)) {
				++ok;
				hist.Add(pc);
				if (Symbolize(pc).find("KytyPcTestSpinHere") != std::string::npos) ++in_fn;
			}
			std::this_thread::sleep_for(std::chrono::microseconds(500));
		}
		const double wall_ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - wall0).count();
		Check(ThreadCpuNs(port, cpu1), "ThreadCpuNs second read");
		const auto top = hist.Top(1);
		std::printf("ReadPc: ok=%d/300 in_spin_function=%d top_pc=0x%llx sym=%s\n", ok, in_fn,
		            top.empty() ? 0ull : static_cast<unsigned long long>(top[0].first), top.empty() ? "-" : Symbolize(top[0].first).c_str());
		Check(ok >= 290, "ReadPc succeeds on a running (translated) thread");
		Check(in_fn >= 0.9 * ok && ok > 0, "PC of the spinner lands in the spinning function");
		const double ratio = static_cast<double>(cpu1 - cpu0) / wall_ns;
		std::printf("spinner cpu/wall = %.3f\n", ratio);
		Check(ratio > 0.5 && ratio < 1.3, "spinner: cpu time ~ wall time");
		g_stop.store(true);
		spinner.join();

		std::atomic<uint32_t> sleeper_port {0};
		std::atomic<bool>     sleeper_stop {false};
		std::thread           sleeper([&] {
            sleeper_port.store(mach_thread_self());
            while (!sleeper_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        });
		while (sleeper_port.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(1));
		uint64_t s0 = 0, s1 = 0;
		ThreadCpuNs(static_cast<mach_port_t>(sleeper_port.load()), s0);
		std::this_thread::sleep_for(std::chrono::milliseconds(200));
		ThreadCpuNs(static_cast<mach_port_t>(sleeper_port.load()), s1);
		std::printf("sleeper cpu/200ms = %.3f\n", static_cast<double>(s1 - s0) / 200e6);
		Check(static_cast<double>(s1 - s0) / 200e6 < 0.2, "sleeper: cpu time << wall time");
		sleeper_stop.store(true);
		sleeper.join();
	}
#endif

	// CpuWall deltas.
	{
		CpuWall tracker;
		auto    d0 = tracker.Tick(1000, 5000);
		Check(d0.first == 0 && d0.second == 0, "first tick yields nothing");
		auto d1 = tracker.Tick(1600, 6000);
		Check(d1.first == 600 && d1.second == 1000, "tick deltas");
		auto d2 = tracker.Tick(1500, 6500);
		Check(d2.first == 0 && d2.second == 500, "a backwards cpu reading clamps to 0");
	}

	// Walk tally: distinct pages per walk, summed over walks.
	{
		(void)Drain();
		const uint64_t a[] = {0x10000, 0x10004, 0x10ffe, 0x20000}; // page 0x10 twice, a read at 0x10ffe (4 bytes) spans pages 0x10/0x11, page 0x20
		NoteReads(a, 4, 4);
		EndWalk();
		const uint64_t b[] = {0x10000, 0x10008};
		NoteReads(b, 2, 4);
		EndWalk();
		const auto t = Drain();
		Check(t.walks == 2, "walks counted");
		Check(t.reads == 6 && t.bytes == 24, "reads and bytes");
		Check(t.pages == 3 + 1, "distinct pages per walk: {0x10,0x11,0x20} + {0x10}");
		const auto again = Drain();
		Check(again.walks == 0 && again.reads == 0 && again.pages == 0, "Drain resets");
	}

	if (g_failures == 0) std::printf("CpPcSamplerTests: OK\n");
	return g_failures == 0 ? 0 : 1;
}
