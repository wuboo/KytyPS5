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

// S123: force_tex filter (@addr / @fmt=n,WxH) and diag_tex ...:all.
void TestForceFilterAndDiagAll() {
	const auto file = CF::Parse("force_tex=aa:30:ffff00ff@0x70d1770000,bb:5:ff000000@fmt=130,1x1,cc:7:ffffffff\n"
	                            "diag_tex=aa:30:all,bb:4\n");
	Check(file.errors.empty(), "filtered force_tex / diag_tex all rejected");
	const auto snap = CF::Resolve(file, 1);
	Check(snap.force_tex.size() == 3, "the comma of @fmt= split an item");
	Check(snap.force_tex[0].filter.kind == CF::TexFilter::Kind::Addr && snap.force_tex[0].filter.addr == 0x70d1770000ull,
	      "address filter fields");
	Check(snap.force_tex[1].filter.kind == CF::TexFilter::Kind::Fmt && snap.force_tex[1].filter.fmt == 130 &&
	          snap.force_tex[1].filter.width == 1 && snap.force_tex[1].filter.height == 1,
	      "format filter fields");
	Check(snap.force_tex[2].filter.kind == CF::TexFilter::Kind::None && snap.force_tex[2].slot == 7, "item after @fmt= parsed");
	Check(snap.ForceTexHasFilter(), "filter not reported");

	// A filtered entry binds only a T# that fits it; without the T# only unfiltered entries answer.
	uint32_t           rgba = 0;
	const CF::TexKey   hit {0x70d1770000ull, 181, 4, 4}, other_addr {0x1000, 181, 4, 4};
	Check(snap.ForceTexFor(0xaa, 30, &rgba, &hit) && rgba == 0xffff00ffu, "address filter missed its T#");
	Check(!snap.ForceTexFor(0xaa, 30, &rgba, &other_addr), "address filter matched another T#");
	Check(!snap.ForceTexFor(0xaa, 30, &rgba), "filtered entry answered without a T#");
	const CF::TexKey fmt_hit {0x5000, 130, 1, 1}, fmt_size {0x5000, 130, 4, 4}, fmt_fmt {0x5000, 131, 1, 1};
	Check(snap.ForceTexFor(0xbb, 5, &rgba, &fmt_hit) && rgba == 0xff000000u, "format filter missed its T#");
	Check(!snap.ForceTexFor(0xbb, 5, &rgba, &fmt_size), "format filter ignored the size");
	Check(!snap.ForceTexFor(0xbb, 5, &rgba, &fmt_fmt), "format filter ignored the format");
	Check(snap.ForceTexFor(0xcc, 7, &rgba) && snap.ForceTexFor(0xcc, 7, &rgba, &other_addr), "unfiltered entry lost");

	Check(snap.diag_tex.size() == 2 && snap.diag_tex[0].all && !snap.diag_tex[1].all, "diag_tex all flag");
	Check(snap.DiagTexAll(0xaa, 30) && !snap.DiagTexAll(0xaa, 31) && !snap.DiagTexAll(0xbb, 4), "DiagTexAll lookup");
	Check(!snap.DiagTexListed(0xaa, 30) && snap.DiagTexListed(0xbb, 4), "all entries must not also be once-per-block");

	Check(!CF::Parse("force_tex=aa:30:ffff00ff@\n").errors.empty() && !CF::Parse("force_tex=aa:30:ffff00ff@zz\n").errors.empty() &&
	          !CF::Parse("force_tex=aa:30:ffff00ff@fmt=130\n").errors.empty() &&
	          !CF::Parse("force_tex=aa:30:ffff00ff@fmt=130,0x1\n").errors.empty() && !CF::Parse("diag_tex=aa:30:some\n").errors.empty(),
	      "bad filter / all accepted");
	CF::Snapshot a, b;
	a.force_tex.push_back({1, 2, 3, {CF::TexFilter::Kind::Addr, 0x10, 0, 0, 0}});
	b.force_tex.push_back({1, 2, 3, {}});
	Check(!a.SameSettings(b), "filter ignored by SameSettings");
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

// S124: dump_tex=<ps>:<slot>[@addr|@fmt=n,WxH][,...] with the force_tex filters.
void TestDumpTex() {
	const auto file = CF::Parse("dump_tex=b96c2898f05637df:30@0x70cd470000,aa:4,bb:5@fmt=130,2048x2048\n"
	                            "@100\ndump_tex=\n");
	Check(file.errors.empty(), "dump_tex lines rejected");
	const auto snap = CF::Resolve(file, 1);
	Check(snap.dump_tex.size() == 3, "dump_tex item count (the comma of @fmt= split an item?)");
	Check(snap.dump_tex[0].ps == 0xb96c2898f05637dfull && snap.dump_tex[0].slot == 30 &&
	          snap.dump_tex[0].filter.kind == CF::TexFilter::Kind::Addr && snap.dump_tex[0].filter.addr == 0x70cd470000ull,
	      "dump_tex address filter fields");
	const CF::TexKey terrain {0x70cd470000ull, 134, 2048, 2048}, placeholder {0x70d1770000ull, 134, 4, 4};
	Check(snap.DumpTexListed(0xb96c2898f05637dfull, 30, terrain), "dump_tex did not list the filtered T#");
	Check(!snap.DumpTexListed(0xb96c2898f05637dfull, 30, placeholder), "dump_tex ignored its address filter");
	Check(!snap.DumpTexListed(0xb96c2898f05637dfull, 31, terrain), "dump_tex matched another slot");
	Check(!snap.DumpTexListed(0x1234, 30, terrain), "dump_tex matched another pixel shader");
	Check(snap.DumpTexListed(0xaa, 4, placeholder) && snap.DumpTexListed(0xaa, 4, terrain), "unfiltered dump_tex must match any T#");
	Check(snap.dump_tex[2].filter.kind == CF::TexFilter::Kind::Fmt && snap.dump_tex[2].filter.width == 2048, "dump_tex format filter");
	Check(CF::Resolve(file, 150).dump_tex.empty(), "empty value did not clear dump_tex");
	Check(!CF::Parse("dump_tex=abc\n").errors.empty() && !CF::Parse("dump_tex=abc:128\n").errors.empty() &&
	          !CF::Parse("dump_tex=0:1\n").errors.empty() && !CF::Parse("dump_tex=abc:1@zz\n").errors.empty(),
	      "bad dump_tex accepted");
	CF::Snapshot a, b;
	b.dump_tex.push_back({1, 2, {}});
	Check(!a.SameSettings(b), "dump_tex ignored by SameSettings");
}

// S125: watch_tex=<ps>:<slot>[@filter] parses like dump_tex but is its own key.
void TestWatchTex() {
	const auto file = CF::Parse("watch_tex=b96c2898f05637df:30@fmt=182,2048x2048\n@100\nwatch_tex=\n");
	Check(file.errors.empty(), "watch_tex line rejected");
	const auto snap = CF::Resolve(file, 1);
	Check(snap.watch_tex.size() == 1 && snap.dump_tex.empty(), "watch_tex must not fill dump_tex");
	const CF::TexKey terrain {0x70cd470000ull, 182, 2048, 2048}, other {0x70cd470000ull, 130, 64, 64};
	Check(snap.WatchTexListed(0xb96c2898f05637dfull, 30, terrain), "watch_tex did not list the T#");
	Check(!snap.WatchTexListed(0xb96c2898f05637dfull, 30, other), "watch_tex ignored its format filter");
	Check(!snap.WatchTexListed(0xb96c2898f05637dfull, 31, terrain), "watch_tex matched another slot");
	Check(!snap.DumpTexListed(0xb96c2898f05637dfull, 30, terrain), "dump_tex listed a watch_tex entry");
	Check(CF::Resolve(file, 150).watch_tex.empty(), "empty value did not clear watch_tex");
	Check(!CF::Parse("watch_tex=abc\n").errors.empty() && !CF::Parse("watch_tex=abc:128\n").errors.empty(), "bad watch_tex accepted");
	CF::Snapshot a, b;
	b.watch_tex.push_back({1, 2, {}});
	Check(!a.SameSettings(b), "watch_tex ignored by SameSettings");
}

} // namespace

int main() {
	TestParseAndResolve();
	TestDiagTex();
	TestForceFilterAndDiagAll();
	TestDumpTex();
	TestWatchTex();
	TestStrictParsing();
	TestPublishAndPoll();
	return 0;
}
