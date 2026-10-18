// S406: KYTY_PV_RANGE_GEN memo decisions (pvRangeGen.h).
// Mutants (each must fail): IGNORE_STAMP (a hit although the page stamp moved), DEFAULT_OFF (the switch is off without the variable).
// The page stamps themselves (bumped on CPU and GPU modification) are pinned in MemoryTrackerTests.cpp (mutants NO_BUMP_CPU / NO_BUMP_GPU).

#include "graphics/host_gpu/renderer/pvRangeGen.h"

#include <cstdio>

using namespace Libs::Graphics::PvRangeGen;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

int main() {
	Check(ParseMode(nullptr) == Mode::On, "unset = on (S408 default)");
	Check(ParseMode("0") == Mode::Off, "0 = off");
	Check(ParseMode("1") == Mode::On, "1 = on");
	Check(ParseMode("shadow") == Mode::Shadow, "shadow");
	Check(ParseMode("yes") == Mode::Off, "anything else = off");
	Check(EnvMode() == Mode::On, "without the variable the switch is on (default 1)");

	const Key   key {0x571b2da7c, 768, 2, 192, 0, 0x7fffffff};
	Entry       entry {/*max_stamp*/ 41, /*index_hash*/ 0xabc, /*derivable*/ true, {10, 200, false}};
	Memo        memo;
	Check(memo.Find(key) == nullptr, "empty memo");
	memo.Insert(key, entry);
	Check(memo.Find(key) != nullptr && memo.Find(key)->range.end == 200, "inserted entry found");
	Key other = key;
	other.vertex_offset = 7;
	Check(memo.Find(other) == nullptr, "another vertex offset is another entry");
	other = key;
	other.address += 4;
	Check(memo.Find(other) == nullptr, "another address is another entry");

	const Probe clean {.host_data = false, .registered = true, .clean = true, .max_stamp = 41};
	Check(Decide(clean, memo.Find(key)) == Outcome::Hit, "clean range, same stamp: hit");
	Probe moved = clean;
	moved.max_stamp = 42;
	Check(Decide(moved, memo.Find(key)) == Outcome::Miss, "a page took a newer stamp: miss");
	Check(Decide(clean, nullptr) == Outcome::Miss, "no entry: miss");
	Probe host = clean;
	host.host_data = true;
	Check(Decide(host, memo.Find(key)) == Outcome::IneligibleHost, "host data falls back");
	Probe unregistered = clean;
	unregistered.registered = false;
	unregistered.clean      = false;
	Check(Decide(unregistered, memo.Find(key)) == Outcome::IneligibleUnregistered, "unregistered range falls back");
	Probe dirty = clean;
	dirty.clean = false;
	Check(Decide(dirty, memo.Find(key)) == Outcome::IneligibleDirty, "CPU- or GPU-modified range falls back");

	// Shadow comparison.
	const Range same {10, 200, false};
	Check(!Compare(entry, 0xabc, true, same).Mismatch(), "identical answer: no mismatch");
	Check(Compare(entry, 0xabd, true, same).hash_differs && !Compare(entry, 0xabd, true, same).range_differs, "hash only");
	Check(Compare(entry, 0xabc, true, {10, 201, false}).range_differs, "range differs");
	Check(Compare(entry, 0xabc, false, same).range_differs, "derivability differs");
	Check(!Compare({0, 1, false, {}}, 1, false, {5, 6, true}).range_differs, "not derivable on both sides: the range is not used");

	// Bounded memo.
	Memo bounded;
	for (uint64_t i = 0; i < Memo::kMaxEntries + 10; i++) {
		Key k = key;
		k.address = 0x1000 + i * 0x1000;
		bounded.Insert(k, entry);
	}
	Check(bounded.Size() <= Memo::kMaxEntries, "memo stays bounded");

	if (g_failures != 0) return 1;
	std::fprintf(stderr, "PvRangeGenTests: ok\n");
	return 0;
}
