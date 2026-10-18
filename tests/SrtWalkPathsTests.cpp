// S430 DIAG: the walk-path cells (srtWalkPaths.h) and the sampler names of the new sub-sites (cpSampler.h).
// Mutant: KYTY_SRTWALKPATHS_MUTANT_NO_GATE (counts while inactive).

#include "graphics/presentation/cpSampler.h"
#include "graphics/shader/recompiler/ir/passes/srtWalkPaths.h"

#include <cstdio>
#include <cstring>

using namespace Libs::Graphics;
namespace WP = Libs::Graphics::ShaderRecompiler::IR::SrtWalkPaths;

static int g_failed = 0;

#define CHECK(cond)                                                                    \
	do {                                                                               \
		if (!(cond)) {                                                                 \
			std::fprintf(stderr, "SrtWalkPathsTests: failed: %s (line %d)\n", #cond, __LINE__); \
			g_failed++;                                                                \
		}                                                                              \
	} while (0)

int main() {
	// Inactive: nothing is counted, Drain is all zeros.
	WP::Note(WP::Path::Tree);
	WP::Note(WP::Path::InterpSlots, 5);
	{
		const auto cells = WP::Drain();
		for (auto cell: cells) CHECK(cell == 0);
	}
	// Active: counted per path, Drain takes and clears.
	WP::Active().store(true);
	WP::Note(WP::Path::Tree);
	WP::Note(WP::Path::Tree);
	WP::Note(WP::Path::TreeFail);
	WP::Note(WP::Path::Capture);
	WP::Note(WP::Path::SlotJit, 7);
	WP::Note(WP::Path::InterpWalk, 3);
	WP::Note(WP::Path::InterpSlots, 4);
	{
		const auto cells = WP::Drain();
		CHECK(cells[static_cast<size_t>(WP::Path::Tree)] == 2);
		CHECK(cells[static_cast<size_t>(WP::Path::TreeFail)] == 1);
		CHECK(cells[static_cast<size_t>(WP::Path::Capture)] == 1);
		CHECK(cells[static_cast<size_t>(WP::Path::SlotJit)] == 7);
		CHECK(cells[static_cast<size_t>(WP::Path::InterpWalk)] == 3);
		CHECK(cells[static_cast<size_t>(WP::Path::InterpSlots)] == 4);
		const auto again = WP::Drain();
		for (auto cell: again) CHECK(cell == 0);
	}
	WP::Active().store(false);
	// Sampler names of the S430 sites.
	CHECK(std::strcmp(CpSampler::SiteName(CpSampler::SiteId(CpSampler::Site::ShSpecRead)), "sh.spec_read") == 0);
	CHECK(std::strcmp(CpSampler::SiteName(CpSampler::SiteId(CpSampler::Site::BindResolveDescCopy)), "bind.resolve.desc_copy") == 0);
	CHECK(std::strcmp(CpSampler::SiteName(CpSampler::SiteId(CpSampler::Site::BindResolveFindImage)), "bind.resolve.find_image") == 0);
	CHECK(std::strcmp(CpSampler::SiteName(CpSampler::SiteId(CpSampler::Site::BindResolveValidate)), "bind.resolve.validate") == 0);
	CHECK(std::strcmp(CpSampler::SiteName(CpSampler::SiteId(CpSampler::Site::PvIndexHash)), "pv.index_hash") == 0);
	if (g_failed != 0) {
		std::fprintf(stderr, "SrtWalkPathsTests: %d failed\n", g_failed);
		return 1;
	}
	std::printf("SrtWalkPathsTests: ok\n");
	return 0;
}
