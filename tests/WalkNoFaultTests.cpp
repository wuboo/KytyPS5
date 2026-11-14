// S594 (KYTY_WALK_NOFAULT): the pure part of walkNoFault.h. The walker side (served / refused / cold page / worker, code identity when
// off) is in ResourceMaterializationTests.cpp (TestWalkNoFault and the walk_nofault_every_load build).
#include "graphics/shader/recompiler/ir/passes/walkNoFault.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace NF = Libs::Graphics::ShaderRecompiler::IR::WalkNoFault;

namespace {

int g_failures = 0;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "WalkNoFaultTests: failed: %s\n", text);
		++g_failures;
	}
}

void TestParse() {
	Check(NF::Parse(nullptr) == NF::Mode::Off, "unset = off");
	Check(NF::Parse("") == NF::Mode::Off, "empty = off");
	Check(NF::Parse("0") == NF::Mode::Off, "0 = off");
	Check(NF::Parse("1") == NF::Mode::On, "1 = on");
	Check(NF::Parse("on") == NF::Mode::Off && NF::Parse("10") == NF::Mode::Off, "anything else = off");
	const char* env = std::getenv("KYTY_WALK_NOFAULT");
	Check(NF::EnvMode() == NF::Parse(env), "EnvMode follows the environment");
	NF::Override().store(1);
	Check(NF::Current() == NF::Mode::On, "override on");
	NF::Override().store(0);
	Check(NF::Current() == NF::Mode::Off, "override off");
	NF::Override().store(-1);
	Check(NF::Current() == NF::EnvMode(), "no override = environment");
}

void TestTable() {
	NF::ClearHot();
	const uint64_t a = 0x5794bf8fcull;
	Check(!NF::IsHot(a), "empty table: nothing hot");
	Check(NF::NoteHot(a), "first note changes the slot");
	Check(!NF::NoteHot(a + 4), "same page: no change");
	Check(NF::IsHot(a & ~uint64_t {4095}) && NF::IsHot((a & ~uint64_t {4095}) + 4092), "the whole page is hot");
	Check(!NF::IsHot(a + 4096), "the next page is not");
	// A page sharing the slot replaces it (direct mapped): the old page is cold again, never a false hit of another page.
	const uint64_t alias = a + (uint64_t {NF::kSlots} << NF::kPageShift);
	Check(NF::SlotOf(NF::PageOf(alias)) == NF::SlotOf(NF::PageOf(a)), "alias shares the slot");
	Check(!NF::IsHot(alias), "alias is not hot while the page holds the slot");
	Check(NF::NoteHot(alias) && NF::IsHot(alias) && !NF::IsHot(a), "alias takes the slot");
	// Address zero (page 0) must not match an empty slot.
	NF::ClearHot();
	Check(!NF::IsHot(0) && !NF::IsHot(~uint64_t {0} >> 16), "empty slots match no page");
	NF::ClearHot();
}

void TestServed() {
	Check(NF::Served(0) == NF::kServed && (NF::Served(0) & 0xffffffffu) == 0, "served zero keeps the served bit");
	Check(NF::Served(0xdeadbeefu) == (NF::kServed | 0xdeadbeefu), "served dword in the low half");
	Check((uint64_t {0} & NF::kServed) == 0, "0 = not served");
}

// The fault handler (CP thread) notes pages while walkers on other threads read the table (TSan build).
void TestConcurrent() {
	NF::ClearHot();
	std::atomic<bool> stop {false};
	std::atomic<uint64_t> hits {0};
	std::vector<std::thread> readers;
	for (int t = 0; t < 3; ++t) {
		readers.emplace_back([&] {
			uint64_t local = 0;
			while (!stop.load(std::memory_order_relaxed)) {
				for (uint64_t p = 0; p < 512; ++p) local += NF::IsHot(p << NF::kPageShift) ? 1 : 0;
			}
			hits.fetch_add(local);
		});
	}
	for (int round = 0; round < 2000; ++round) NF::NoteHot(static_cast<uint64_t>(round % 512) << NF::kPageShift);
	stop = true;
	for (auto& reader: readers) reader.join();
	Check(NF::IsHot(uint64_t {1999 % 512} << NF::kPageShift), "last noted page is hot");
	NF::ClearHot();
}

} // namespace

int main() {
	TestParse();
	TestTable();
	TestServed();
	TestConcurrent();
	if (g_failures != 0) return 1;
	std::puts("WalkNoFaultTests: OK");
	return 0;
}
