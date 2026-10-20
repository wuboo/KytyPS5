#include "graphics/shader/recompiler/ir/SrtCfMemo.h"

#include <array>
#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::ShaderRecompiler::IR;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SrtCfMemoTests: failed: %s\n", text);
		std::exit(1); // not abort(): the WILL_FAIL mutants need a plain nonzero exit code
	}
}

struct Memory {
	uint32_t value = 0;
	bool     ok    = true;
	uint32_t calls = 0;
};

bool ReadStrict(void* userdata, uint64_t, std::span<uint32_t> out) {
	auto& memory = *static_cast<Memory*>(userdata);
	++memory.calls;
	if (!memory.ok) return false;
	out[0] = memory.value;
	return true;
}

bool ReadOther(void*, uint64_t, std::span<uint32_t>) {
	return false;
}

// One recorded loop: user data words 1 and 3 read, shader base read when `with_base`, one strict read of 0x100000 that returned `memory.value`.
SrtCfMemo Record(const std::array<uint32_t, 4>& user, Memory& memory, bool with_base, uint64_t base = 0x7000, bool read_ok = true) {
	SrtCfMemo memo;
	memo.Begin(4u, nullptr, ReadStrict);
	memo.NoteUserData(1u, user[1]);
	memo.NoteUserData(3u, user[3]);
	if (with_base) memo.NoteShaderBase(base);
	memo.NoteRead(ReadStrict, 0x100000, memory.value, read_ok);
	const std::array<uint8_t, 3> visited{1u, 0u, 1u};
	memo.End(visited);
	return memo;
}

void TestKey() {
	Memory memory {.value = 0x55u};
	const std::array<uint32_t, 4> user {10u, 11u, 12u, 13u};
	const auto memo = Record(user, memory, true);
	Check(memo.valid && memo.visited == std::vector<uint8_t>{1u, 0u, 1u}, "memo: not stored");
	Check(memo.Matches(user, 0x7000, nullptr, ReadStrict, &memory), "memo: the recorded inputs did not match");
	// A word nobody read does not matter.
	const std::array<uint32_t, 4> other_unread {99u, 11u, 98u, 13u};
	Check(memo.Matches(other_unread, 0x7000, nullptr, ReadStrict, &memory), "memo: an unread user data word broke the match");
	// A word that was read does.
	const std::array<uint32_t, 4> other_read {10u, 11u, 12u, 14u};
	Check(!memo.Matches(other_read, 0x7000, nullptr, ReadStrict, &memory), "memo: a changed user data word matched");
	// The shader base, when it was read.
	Check(!memo.Matches(user, 0x8000, nullptr, ReadStrict, &memory), "memo: a changed shader base matched");
	// ... and not when it was not.
	const auto no_base = Record(user, memory, false);
	Check(no_base.Matches(user, 0x8000, nullptr, ReadStrict, &memory), "memo: an unread shader base broke the match");
	// The size of the table (GetUserData fails beyond it) and the readers.
	const std::array<uint32_t, 3> shorter {10u, 11u, 12u};
	Check(!memo.Matches(shorter, 0x7000, nullptr, ReadStrict, &memory), "memo: a shorter user data table matched");
	Check(!memo.Matches(user, 0x7000, ReadOther, ReadStrict, &memory), "memo: another ordinary reader matched");
	Check(!memo.Matches(user, 0x7000, nullptr, ReadOther, &memory), "memo: another strict reader matched");
	// Nothing stored: no match.
	Check(!SrtCfMemo {}.Matches(user, 0x7000, nullptr, ReadStrict, &memory), "memo: an empty memo matched");
}

void TestReads() {
	Memory memory {.value = 0x55u};
	const std::array<uint32_t, 4> user {10u, 11u, 12u, 13u};
	const auto memo = Record(user, memory, false);
	memory.calls = 0;
	Check(memo.Matches(user, 0, nullptr, ReadStrict, &memory) && memory.calls == 1u, "memo: the recorded read was not asked again");
	memory.value = 0x56u;
	Check(!memo.Matches(user, 0, nullptr, ReadStrict, &memory), "memo: a changed guest word matched");
	memory.value = 0x55u;
	memory.ok = false;
	Check(!memo.Matches(user, 0, nullptr, ReadStrict, &memory), "memo: a read that now fails matched");
	// A read that failed during the loop is part of the key: when it starts to succeed the loop may end elsewhere.
	memory.ok = false;
	const auto failed = Record(user, memory, false, 0, false);
	Check(failed.Matches(user, 0, nullptr, ReadStrict, &memory), "memo: a failed read that fails again did not match");
	memory.ok = true;
	Check(!failed.Matches(user, 0, nullptr, ReadStrict, &memory), "memo: a failed read that now succeeds matched");
	// A direct host read below guest memory reads zero without touching memory.
	SrtCfMemo direct;
	direct.Begin(4u, nullptr, ReadStrict);
	direct.NoteRead(nullptr, 0x100, 0u, true);
	direct.End(std::array<uint8_t, 1> {1u});
	Check(direct.Matches(user, 0, nullptr, ReadStrict, &memory), "memo: a direct read below guest memory did not read zero");
	// A reader the key does not name cannot be asked again: the loop is not stored.
	SrtCfMemo foreign;
	foreign.Begin(4u, nullptr, ReadStrict);
	foreign.NoteRead(ReadOther, 0x100000, 0u, true);
	foreign.End(std::array<uint8_t, 1> {1u});
	Check(!foreign.valid, "memo: a loop that used a reader outside the key was stored");
	// A user data word beyond the table of 64 cannot be keyed.
	SrtCfMemo wide;
	wide.Begin(100u, nullptr, ReadStrict);
	wide.NoteUserData(70u, 1u);
	wide.End(std::array<uint8_t, 1> {1u});
	Check(!wide.valid, "memo: a loop that read user data word 70 was stored");
}

void TestSet() {
	Memory memory {.value = 0x55u};
	SrtCfMemoSet set;
	const auto store = [&](const std::array<uint32_t, 4>& user) {
		auto& way = set.Victim(user, 0, nullptr, ReadStrict);
		way.Begin(4u, nullptr, ReadStrict);
		way.NoteUserData(1u, user[1]);
		way.NoteRead(ReadStrict, 0x100000, memory.value, true);
		way.End(std::array<uint8_t, 1> {static_cast<uint8_t>(user[1] & 1u)});
	};
	// Three pointers in turn (a frame ring) fit; each answers with its own blocks.
	const std::array<std::array<uint32_t, 4>, 3> ring {{{0u, 100u, 0u, 0u}, {0u, 201u, 0u, 0u}, {0u, 300u, 0u, 0u}}};
	for (const auto& user: ring) store(user);
	for (const auto& user: ring) {
		const auto* way = set.Find(user, 0, nullptr, ReadStrict, &memory);
		Check(way != nullptr && way->visited[0] == (user[1] & 1u), "set: a ring entry was not found or answered for another entry");
	}
	// A way is replaced, not duplicated, when the same inputs read something else.
	memory.value = 0x56u;
	Check(set.Find(ring[0], 0, nullptr, ReadStrict, &memory) == nullptr, "set: a changed guest word was found");
	store(ring[0]);
	size_t same = 0;
	for (const auto& way: set.ways) same += way.valid && way.user_words[1] == 100u;
	Check(same == 1u, "set: the same inputs took two ways");
	// More entries than ways: the oldest goes, the newest stay.
	memory.value = 0x55u;
	for (uint32_t i = 0; i < SrtCfMemoSet::kWays + 2u; ++i) store(std::array<uint32_t, 4> {0u, 1000u + i, 0u, 0u});
	Check(set.Find(std::array<uint32_t, 4> {0u, 1000u + SrtCfMemoSet::kWays + 1u, 0u, 0u}, 0, nullptr, ReadStrict, &memory) != nullptr, "set: the newest entry was lost");
}

void TestMode() {
	Check(ParseCfMemoMode(nullptr) == CfMemoMode::Off, "mode: unset is not off");
	Check(ParseCfMemoMode("0") == CfMemoMode::Off, "mode: 0 is not off");
	Check(ParseCfMemoMode("1") == CfMemoMode::On, "mode: 1 is not on");
	Check(ParseCfMemoMode("shadow") == CfMemoMode::Shadow, "mode: shadow");
	Check(ParseCfMemoMode("on") == CfMemoMode::Off, "mode: an unknown value is not off");
}

} // namespace

int main() {
	TestKey();
	TestReads();
	TestSet();
	TestMode();
	std::puts("SrtCfMemoTests: all cases passed");
	return 0;
}
