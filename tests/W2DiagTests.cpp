// S560 KYTY_W2_DIAG (w2Diag.h): the per-call-site table of CommandBuffer::Handle(). Sites are (file, line) with the file as a pointer; two pointers
// with the same text (two translation units) are one site in the snapshot, two lines of one file are two sites, counts from two threads add up
// exactly and stay split by thread kind, and the snapshot is ordered by calls.
// Mutant (must fail, WILL_FAIL): NO_LINE (the key drops the line, so the lines of a file merge).

#include "graphics/host_gpu/renderer/w2Diag.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

using Libs::Graphics::W2Diag::ParseFlag;
using Libs::Graphics::W2Diag::SiteTable;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "W2DiagTests: failed: %s\n", what);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

// Two distinct arrays with the same text stand for __builtin_FILE() of one header inlined in two translation units.
const char kFileA1[] = "C:\\src\\KytyPS5\\src\\graphics\\host_gpu\\renderer\\renderDraw.cpp";
const char kFileA2[] = "C:/src/KytyPS5/src/graphics/host_gpu/renderer/renderDraw.cpp";
const char kFileB[]  = "/home/u/KytyPS5/src/graphics/host_gpu/renderer/cache/textureCache.cpp";

} // namespace

int main() {
	Check(ParseFlag("1") && !ParseFlag("0") && !ParseFlag(nullptr) && !ParseFlag("11") && !ParseFlag(""), "ParseFlag");
	Check(SiteTable<>::ShortName(kFileA1) == "graphics/host_gpu/renderer/renderDraw.cpp", "ShortName backslashes");
	Check(SiteTable<>::ShortName(kFileA1) == SiteTable<>::ShortName(kFileA2), "ShortName same text");

	static SiteTable<64> table;
	constexpr uint64_t   kN = 200000;
	std::thread          cp([&] {
        for (uint64_t i = 0; i < kN; i++) {
            table.Add(kFileA1, 1331, true);
            if (i % 2 == 0) table.Add(kFileA1, 1500, true);
            if (i % 4 == 0) table.Add(kFileB, 77, true);
        }
    });
	std::thread other([&] {
		for (uint64_t i = 0; i < kN; i++) {
			table.Add(kFileA2, 1331, false);
			if (i % 8 == 0) table.Add(kFileB, 77, false);
		}
	});
	cp.join();
	other.join();

	const auto rows = table.Snapshot();
	Check(table.Dropped() == 0, "nothing dropped");
	Check(rows.size() == 3, "three sites (two lines of renderDraw.cpp, one of textureCache.cpp)");
	Check(rows[0].file == "graphics/host_gpu/renderer/renderDraw.cpp" && rows[0].line == 1331, "top site");
	Check(rows[0].cp == kN && rows[0].other == kN, "top site: both pointers merged, split by thread");
	Check(rows[1].line == 1500 && rows[1].cp == kN / 2 && rows[1].other == 0, "second site");
	Check(rows[2].file == "graphics/host_gpu/renderer/cache/textureCache.cpp" && rows[2].line == 77, "third site");
	Check(rows[2].cp == kN / 4 && rows[2].other == kN / 8, "third site counts");

	// A full table drops (and counts) instead of looping.
	static SiteTable<4> small;
	static const char   files[6][8] = {"a", "b", "c", "d", "e", "f"};
	for (const auto& f: files) small.Add(f, 1, true);
	Check(small.Snapshot().size() == 4 && small.Dropped() == 2, "full table drops");

	std::printf("W2DiagTests: ok\n");
	return 0;
}
