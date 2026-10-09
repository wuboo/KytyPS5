#include "graphics/presentation/window/frameDumpFlips.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "FrameDumpFlipsTests: failed: %s\n", message);
		std::exit(1);
	}
}

// S129: KYTY_FRAME_DUMP_FLIPS list parsing and matching of the 1-based present index.
void TestParse() {
	const auto flips = ParseFrameDumpFlips("5000,5800,6600");
	Check(flips.size() == 3 && flips[0] == 5000 && flips[1] == 5800 && flips[2] == 6600, "three flips in order");
	Check(ParseFrameDumpFlips(nullptr).empty() && ParseFrameDumpFlips("").empty(), "unset or empty: nothing to dump");
	Check(ParseFrameDumpFlips("0,12,0").size() == 1, "zero is not a present index");
	const auto stop = ParseFrameDumpFlips("7,x,9");
	Check(stop.size() == 1 && stop[0] == 7, "parsing stops at garbage");
	Check(ParseFrameDumpFlips("3,4,").size() == 2, "trailing comma");
}

void TestMatch() {
	const auto flips = ParseFrameDumpFlips("100,250");
	Check(FrameDumpFlipWanted(flips, 100) && FrameDumpFlipWanted(flips, 250), "listed flips match");
	Check(!FrameDumpFlipWanted(flips, 99) && !FrameDumpFlipWanted(flips, 101) && !FrameDumpFlipWanted(flips, 0), "neighbours do not");
	Check(!FrameDumpFlipWanted({}, 100), "empty list matches nothing");
}

} // namespace

int main() {
	TestParse();
	TestMatch();
	std::puts("FrameDumpFlipsTests: ok");
	return 0;
}
