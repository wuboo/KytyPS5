#include "common/controlFile.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace CF = Common::ControlFile;

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "ControlFileTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

const char* kFile =
    "# base: nothing skipped\n"
    "label=base\n"
    "skip_cs=56a15431999c5a2d  # trailing comment\n"
    "\n"
    "at=6600\n"
    "until=6609\n"
    "skip_cs=ea8e2d64,5323c4ef\n"
    "label=armA\n"
    "@6620\n"
    "skip_cs=\n"
    "dump_rt=2f46fd22b1de1575:6625,abcd\n"
    "force_tex=2f46fd22b1de1575:3:ffff00ff\n";

void TestParseAndResolve() {
	const auto file = CF::Parse(kFile);
	Check(file.errors.empty(), "valid file reported errors");
	Check(file.blocks.size() == 2 && file.base.size() == 2, "block / base split wrong");
	Check(file.blocks[0].at == 6600 && file.blocks[0].until == 6609 && file.blocks[1].until == 0,
	      "at / until not parsed");

	const auto base = CF::Resolve(file, 100);
	Check(base.skip_cs.size() == 1 && base.skip_cs[0] == 0x56a15431999c5a2dull, "base skip_cs wrong");
	Check(base.label == "base" && base.block.empty(), "base label / block wrong");
	Check(base.dump_rt.empty() && base.force_tex.empty(), "base has block-only keys");

	const auto in_a = CF::Resolve(file, 6605);
	Check(in_a.skip_cs.size() == 2 && in_a.SkipCs(0xea8e2d64) && in_a.SkipCs(0x5323c4ef) &&
	          !in_a.SkipCs(0x56a15431999c5a2dull),
	      "block does not replace the base list");
	Check(in_a.label == "armA" && in_a.block == "at=6600-6609", "block label / name wrong");
	Check(CF::Resolve(file, 6600).SkipCs(0xea8e2d64), "window start flip not active");
	Check(CF::Resolve(file, 6609).SkipCs(0xea8e2d64), "window end flip not active");

	// until closes the window: the keys fall back to the base.
	const auto after = CF::Resolve(file, 6610);
	Check(after.SkipCs(0x56a15431999c5a2dull) && !after.SkipCs(0xea8e2d64) && after.label == "base",
	      "until did not close the window");

	// `skip_cs=` clears the list; dump_rt / force_tex parsed with their fields.
	const auto late = CF::Resolve(file, 6700);
	Check(late.skip_cs.empty(), "empty value did not clear the key");
	Check(late.dump_rt.size() == 2 && late.dump_rt[0].ps == 0x2f46fd22b1de1575ull &&
	          late.dump_rt[0].flip == 6625 && late.dump_rt[1].ps == 0xabcd && late.dump_rt[1].flip == 0,
	      "dump_rt fields wrong");
	Check(late.force_tex.size() == 1 && late.force_tex[0].slot == 3 && late.force_tex[0].rgba == 0xffff00ffu,
	      "force_tex fields wrong");
	Check(late.block == "at=6620", "open block name wrong");

	uint32_t rgba = 0;
	Check(late.ForceTexFor(0x2f46fd22b1de1575ull, 3, &rgba) && rgba == 0xffff00ffu, "force_tex lookup failed");
	Check(!late.ForceTexFor(0x2f46fd22b1de1575ull, 4, &rgba), "force_tex matched another slot");
	Check(!late.ForceTexFor(0x1234, 3, &rgba), "force_tex matched another pixel shader");
	Check(!base.ForceTexFor(0x2f46fd22b1de1575ull, 3, &rgba), "force_tex active outside its block");
}

// S122: diag_tex=<ps>:<slot>[,...] lists (ps, slot) pairs and is bound to its block.
void TestDiagTex() {
	const auto file = CF::Parse("diag_tex=b96c2898f05637df:30,abc:4\n@100\ndiag_tex=\n@200\nuntil=210\ndiag_tex=ff:0\n");
	Check(file.errors.empty(), "diag_tex lines rejected");
	const auto base = CF::Resolve(file, 50);
	Check(base.diag_tex.size() == 2 && base.diag_tex[0].ps == 0xb96c2898f05637dfull && base.diag_tex[0].slot == 30 &&
	          base.diag_tex[1].slot == 4,
	      "diag_tex fields wrong");
	Check(base.DiagTexListed(0xb96c2898f05637dfull, 30) && base.DiagTexListed(0xabc, 4), "listed pair not found");
	Check(!base.DiagTexListed(0xb96c2898f05637dfull, 31), "diag_tex matched another slot");
	Check(!base.DiagTexListed(0x1234, 30), "diag_tex matched another pixel shader");
	Check(CF::Resolve(file, 150).diag_tex.empty(), "empty value did not clear diag_tex");
	Check(CF::Resolve(file, 205).DiagTexListed(0xff, 0) && CF::Resolve(file, 211).diag_tex.empty(), "diag_tex block window wrong");
	Check(!CF::Parse("diag_tex=abc\n").errors.empty() && !CF::Parse("diag_tex=abc:128\n").errors.empty() &&
	          !CF::Parse("diag_tex=0:1\n").errors.empty() && !CF::Parse("diag_tex=abc:x\n").errors.empty(),
	      "bad diag_tex accepted");
	CF::Snapshot a, b;
	b.diag_tex.push_back({1, 2});
	Check(!a.SameSettings(b), "diag_tex ignored by SameSettings");
}

void TestStrictParsing() {
	const auto file = CF::Parse("skip_cs=12zz\n"        // junk in a hex list
	                            "skip_cs=0\n"            // zero hash
	                            "force_tex=ab:3:ff00ff\n" // 6 digits
	                            "force_tex=ab:200:ffffffff\n" // slot > 127
	                            "dump_rt=ab:0\n"         // flip 0
	                            "label=bad label\n"
	                            "frobnicate=1\n"
	                            "no equals sign\n"
	                            "until=5\n"              // outside a block
	                            "at=x\n"                 // bad at swallows its block
	                            "skip_cs=99\n"
	                            "at=10\n"
	                            "until=5\n"              // before at
	                            "skip_cs=77\n");
	Check(file.errors.size() == 11, "every bad line must be reported once");
	Check(file.base.empty(), "a bad line reached the base");
	Check(file.blocks.size() == 1 && file.blocks[0].at == 10 && file.blocks[0].items.size() == 1,
	      "the block of a bad `at` line was not swallowed");
	Check(CF::Resolve(file, 10).SkipCs(0x77) && !CF::Resolve(file, 10).SkipCs(0x99), "wrong keys applied");

	uint64_t value = 0;
	Check(!CF::ParseNumber("12zz", 16, value) && !CF::ParseNumber("", 10, value) &&
	          !CF::ParseNumber("12345678901234567", 10, value) && !CF::ParseNumber("1a", 10, value),
	      "ParseNumber accepted junk");
	Check(CF::ParseNumber("ff", 16, value) && value == 255, "ParseNumber rejected hex");
}

void TestPublishAndPoll() {
	const auto file = CF::Parse(kFile);
	Check(CF::Current().gen == 0 && !CF::Current().SkipCs(0x56a15431999c5a2dull), "initial snapshot not empty");
	Check(CF::Publish(CF::Resolve(file, 100)), "first publish refused");
	const auto gen = CF::Current().gen;
	Check(gen == 1 && CF::Current().SkipCs(0x56a15431999c5a2dull), "published snapshot not visible");
	Check(!CF::Publish(CF::Resolve(file, 101)) && CF::Current().gen == gen, "unchanged settings bumped the generation");
	Check(CF::Publish(CF::Resolve(file, 6605)) && CF::Current().gen == gen + 1, "changed settings not published");

	const auto dir  = std::filesystem::temp_directory_path() / ("kyty-control-" + std::to_string(::getpid()));
	std::filesystem::create_directories(dir);
	const auto path = dir / "control.txt";
	{ std::ofstream(path) << "skip_cs=11\n"; }
	CF::Poller poller {.path = path};
	Check(poller.Refresh().empty() && poller.loaded && CF::Resolve(poller.file, 1).SkipCs(0x11),
	      "first refresh did not load the file");
	{ std::ofstream(path) << "skip_cs=11,22\nbogus\n"; } // different size
	const auto errors = poller.Refresh();
	Check(errors.size() == 1 && CF::Resolve(poller.file, 1).SkipCs(0x22), "changed file not re-read");
	Check(poller.Refresh().empty(), "unchanged file reported errors again");
	std::filesystem::remove_all(dir);
	CF::Poller missing {.path = dir / "nope.txt"};
	Check(missing.Refresh().empty() && !missing.loaded, "missing file counted as loaded");
}

} // namespace

int main() {
	TestParseAndResolve();
	TestDiagTex();
	TestStrictParsing();
	TestPublishAndPoll();
	return 0;
}
