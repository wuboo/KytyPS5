#include "libs/exitAfter.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int g_checks = 0;

void Check(bool value, const char* message) {
	g_checks++;
	if (!value) {
		std::fprintf(stderr, "ExitAfterTests: failed: %s\n", message);
		std::exit(1);
	}
}

void TestParse() {
	using namespace Libs::BenchExit;
	const auto ok = ParseExitAfter("underwater_aerial_garden+1400");
	Check(ok.ok, "parses");
	Check(ok.name == "underwater_aerial_garden", "level name");
	Check(ok.plus == 1400, "N is 1400");
	const uint64_t start = 4862;
	Check(!GuestFlipReached(start + ok.plus - 1, start + ok.plus), "one flip early");
	Check(GuestFlipReached(start + ok.plus, start + ok.plus), "at the threshold");
	Check(!ParseExitAfter("underwater_aerial_garden").ok, "missing plus");
	Check(!ParseExitAfter("+1400").ok, "empty name");
	Check(!ParseExitAfter("underwater_aerial_garden+").ok, "empty N");
	Check(!ParseExitAfter("underwater_aerial_garden+12a").ok, "trailing junk");
	Check(!ParseExitAfter("").ok, "empty spec");
	uint64_t n = 0;
	Check(ParseUint("6995", n) && n == 6995, "absolute flip");
	Check(!ParseUint("06995x", n), "absolute flip junk");
}

void TestMismatch() {
	using namespace Libs::BenchExit;
	Check(ReplayMarkAction("", "") == MarkAction::Consume, "empty matches empty");
	Check(ReplayMarkAction("underwater_aerial_garden", "underwater_aerial_garden") == MarkAction::Consume, "same name");
	Check(ReplayMarkAction("underwater_aerial_garden", "worldmap") == MarkAction::Mismatch, "different name is not consumed");
}

} // namespace

int main() {
	TestParse();
	TestMismatch();
	std::printf("ExitAfterTests: ok (%d)\n", g_checks);
	return 0;
}
