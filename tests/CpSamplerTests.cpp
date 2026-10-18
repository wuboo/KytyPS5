// S406: CP breadcrumb sampler (cpSampler.h). A dummy thread cycles through sites in known time proportions; the sampler
// thread's histogram must show them within 3 percentage points. Mutant KYTY_CPSAMPLER_MUTANT_NO_RESTORE (a nested site
// does not restore its parent) must fail the proportions.

#include "graphics/presentation/cpSampler.h"

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

using namespace Libs::Graphics::CpSampler;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

static void Spin(std::chrono::microseconds duration) {
	const auto end = std::chrono::steady_clock::now() + duration;
	while (std::chrono::steady_clock::now() < end) {
	}
}

int main() {
	Active().store(true);
	// Pure histogram bookkeeping.
	{
		Histogram hist;
		hist.Add(SiteId(Site::ShWalk), 0x27);
		hist.Add(SiteId(Site::ShWalk), 0x27);
		hist.Add(kWaitSite, kNoOpcode);
		hist.Add(9999, 0x15); // out of range -> none
		Check(hist.Total() == 4, "total");
		Check(hist.Wait() == 1, "wait counted");
		Check(hist.Cell(SiteId(Site::ShWalk), 0x27) == 2, "cell");
		Check(hist.Cell(kWaitSite, Histogram::kOps - 1) == 1, "no opcode cell");
		Check(hist.SiteTotal(SiteId(Site::None)) == 1, "out-of-range site lands in none");
		Check(std::string(SiteName(SiteId(Site::PvIndexHash))) == "pv.index_hash", "name");
	}
	// Restore on leaving a nested scope.
	{
		CpSite().store(SiteId(Site::None));
		{
			SiteScope outer(SiteId(Site::DrawVertexBuffers));
			{
				SiteScope inner(SiteId(Site::PvPlace));
				Check(CpSite().load() == SiteId(Site::PvPlace), "inner site set");
			}
			Check(CpSite().load() == SiteId(Site::DrawVertexBuffers), "inner restores the parent");
		}
		Check(CpSite().load() == SiteId(Site::None), "outer restores none");
	}
	// Sampler against a dummy thread: A 15% + B 70% + A 15% per 1 ms cycle => A 30 %, B 70 %.
	Sampler sampler;
	sampler.Start(100);
	std::atomic<bool> stop {false};
	std::thread worker([&] {
		while (!stop.load()) {
			SiteScope a(SiteId(Site::BindSamplers));
			Spin(std::chrono::microseconds(150));
			{
				SiteScope b(SiteId(Site::BindFindBuffers));
				Spin(std::chrono::microseconds(700));
			}
			Spin(std::chrono::microseconds(150));
		}
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(1500));
	stop.store(true);
	worker.join();
	sampler.Stop();
	const auto& hist = sampler.Result();
	const double total = static_cast<double>(hist.SiteTotal(SiteId(Site::BindSamplers)) + hist.SiteTotal(SiteId(Site::BindFindBuffers)));
	Check(total > 3000, "enough samples");
	if (total > 0) {
		const double a = 100.0 * static_cast<double>(hist.SiteTotal(SiteId(Site::BindSamplers))) / total;
		const double b = 100.0 * static_cast<double>(hist.SiteTotal(SiteId(Site::BindFindBuffers))) / total;
		std::fprintf(stderr, "samples=%.0f A=%.1f%% B=%.1f%%\n", total, a, b);
		Check(a > 27.0 && a < 33.0, "site A is 30% +-3");
		Check(b > 67.0 && b < 73.0, "site B is 70% +-3");
	}
	if (g_failures != 0) return 1;
	std::fprintf(stderr, "CpSamplerTests: ok\n");
	return 0;
}
