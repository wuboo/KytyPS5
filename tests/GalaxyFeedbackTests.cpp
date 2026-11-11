#include "graphics/host_gpu/renderer/galaxyFeedback.h"

#include <cstdint>
#include <cstdio>

using Libs::Graphics::GalaxyFeedback::Attach;
using Libs::Graphics::GalaxyFeedback::Bound;
using Libs::Graphics::GalaxyFeedback::Find;
using Libs::Graphics::GalaxyFeedback::FindAddress;
using Libs::Graphics::GalaxyFeedback::Hit;
using Libs::Graphics::GalaxyFeedback::Mode;
using Libs::Graphics::GalaxyFeedback::Parse;
using Libs::Graphics::GalaxyFeedback::ShouldSplit;

static int g_fails = 0;

static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "galaxy_feedback: %s\n", what);
		g_fails++;
	}
}

int main() {
	constexpr uint64_t kImage = 0x1000;
	constexpr uint64_t kOther = 0x2000;
	constexpr uint64_t kAlias = 0x3000;
	constexpr uint64_t kAddr  = 0x514080000ull;
	constexpr uint64_t kAlt   = 0x53aa00000ull;
	constexpr uint64_t kView  = 0x600000000ull;

	const Attach attachments[] = {
	    {kImage, kAddr, 0},
	    {kOther, kAlt, 1},
	};
	// Another view of the color attachment: same VkImage, a guest base neither attachment uses.
	const Bound view[] = {{kImage, kView, 3, 2, 0, 1}};
	Hit          hits[4] {};
	Check(Find(view, 1, attachments, 2, hits, 4) == 1, "same VkImage is a feedback hit");
	Check(hits[0].slot == 3 && hits[0].mip == 2 && hits[0].stage == 1 && hits[0].depth == 0 &&
	          hits[0].attach == kAddr && hits[0].address == kView && hits[0].attach_index == 0,
	      "hit keeps the slot, mip and both addresses");
	Check(FindAddress(view, 1, attachments, 2, hits, 4) == 0, "same VkImage is not an address alias");

	// Alias: the card's guest address, a third VkImage. Not a feedback hit.
	const Bound alias[] = {{kAlias, kAddr, 1, 0, 1, 1}};
	Check(Find(alias, 1, attachments, 2, hits, 4) == 0, "different VkImage is not a feedback hit");
	Check(FindAddress(alias, 1, attachments, 2, hits, 4) == 1, "same address and a different image is an alias");
	Check(hits[0].storage == 1 && hits[0].attach_index == 0, "alias keeps the storage flag and the attachment");

	const Bound none[] = {{0, kAddr, 0, 0, 0, 1}};
	Check(Find(none, 1, attachments, 2, hits, 4) == 0, "a null image is not a hit");
	Check(FindAddress(none, 1, attachments, 2, hits, 4) == 0, "a null image is not an alias");

	Check(ShouldSplit(Mode::Split, true, false), "split acts on a feedback hit");
	Check(ShouldSplit(Mode::Fix, true, false), "the fix acts on a feedback hit");
	Check(!ShouldSplit(Mode::Split, false, true), "split ignores a blend draw without feedback");
	Check(!ShouldSplit(Mode::Log, true, true), "log does not split");
	Check(!ShouldSplit(Mode::Off, true, true), "off does not split");
	Check(ShouldSplit(Mode::Blend, false, true), "blend mode splits a blend-on card draw");
	Check(!ShouldSplit(Mode::Blend, true, false), "blend mode ignores feedback without the card blend");

	Check(Parse("1", nullptr, nullptr) == Mode::Log, "FB_LOOP=1 is log");
	Check(Parse(nullptr, "1", nullptr) == Mode::Split, "SPLIT_PASS=1 is split");
	Check(Parse(nullptr, "blend", nullptr) == Mode::Blend, "SPLIT_PASS=blend is the card fallback");
	Check(Parse("1", "1", "1") == Mode::Fix, "FIX wins over the diagnostics");
	Check(Parse(nullptr, nullptr, nullptr) == Mode::Off, "unset is off");
	Check(Parse("0", "0", "0") == Mode::Off, "zero is off");

	if (g_fails != 0) {
		std::fprintf(stderr, "galaxy_feedback: %d failed\n", g_fails);
		return 1;
	}
	return 0;
}
