#include "graphics/host_gpu/renderer/sampledNote.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::SampledNote;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SampledNoteTests: failed: %s\n", message);
		std::exit(1);
	}
}

// R-T1: watch mode with nothing listed (the bench's control file without watch_tex): 1000 resolved T# leave the list empty.
void TestWatchEmptyNotesNothing() {
	List list;
	for (uint64_t i = 0; i < 1000; i++) {
		if (ShouldNote(Mode::Watch, false)) list.Note({1, 0x1000 + i * 0x100, 0, 9, 64, 64});
	}
	Check(list.Entries().empty(), "watch with no watch_tex: the list stays empty");
	Check(ShouldNote(Mode::Watch, true), "watch with watch_tex listed notes");
	Check(ShouldNote(Mode::All, false) && ShouldNote(Mode::All, true), "all always notes");
}

// R-T2: with watch_tex listed, watch gives the same list (and so the same FlipReport lines) as all.
void TestWatchArmedEqualsAll() {
	List watch, all;
	for (uint64_t i = 0; i < 50; i++) {
		const Entry e {0xabc, 0x10000 + (i % 20) * 0x1000, static_cast<uint32_t>(i % 3), 9, 128, 128};
		if (ShouldNote(Mode::Watch, true)) watch.Note(e);
		if (ShouldNote(Mode::All, true)) all.Note(e);
	}
	Check(!watch.Entries().empty() && watch.Entries().size() == all.Entries().size(), "same size");
	bool same = true;
	for (size_t i = 0; i < watch.Entries().size(); i++) {
		const auto &a = watch.Entries()[i], &b = all.Entries()[i];
		same = same && a.ps == b.ps && a.addr == b.addr && a.slot == b.slot;
	}
	Check(same, "same entries in the same order");
}

// R-T3: dedup by (ps, slot, addr): the same 3 keys x 100 give 3 entries; a different ps or slot with the same address is a new key; the cap holds.
void TestDedupAndCap() {
	List list;
	for (int r = 0; r < 100; r++) {
		for (uint32_t s = 0; s < 3; s++) list.Note({7, 0x2000, s, 9, 8, 8});
	}
	Check(list.Entries().size() == 3, "3 keys x 100 -> 3 entries");
	Check(list.Note({8, 0x2000, 0, 9, 8, 8}) && list.Entries().size() == 4, "other ps is a new key");
	list.Clear();
	Check(list.Entries().empty() && list.Note({7, 0x2000, 0, 9, 8, 8}), "Clear forgets the keys");
	for (uint64_t i = 0; i < List::kCap + 10; i++) list.Note({1, i, 0, 0, 0, 0});
	Check(list.Entries().size() == List::kCap, "cap");
}

void TestParse() {
	Check(ParseMode(nullptr) == Mode::Watch && ParseMode("watch") == Mode::Watch && ParseMode("all") == Mode::All && ParseMode("x") == Mode::Watch, "parse");
}

} // namespace

int main() {
	TestWatchEmptyNotesNothing();
	TestWatchArmedEqualsAll();
	TestDedupAndCap();
	TestParse();
	std::puts("SampledNoteTests: all cases passed");
	return 0;
}
