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
		auto& way = set.Victim();
		way.Begin(4u, nullptr, ReadStrict);
		way.NoteUserData(1u, user[1]);
		way.NoteRead(ReadStrict, 0x100000, memory.value, true);
		way.End(std::array<uint8_t, 1> {static_cast<uint8_t>(user[1] & 1u)});
		set.Promote(&way);
	};
	// Three pointers in turn fit; each answers with its own blocks.
	const std::array<std::array<uint32_t, 4>, 3> ring {{{0u, 100u, 0u, 0u}, {0u, 201u, 0u, 0u}, {0u, 300u, 0u, 0u}}};
	for (const auto& user: ring) store(user);
	for (const auto& user: ring) {
		const auto* way = set.Find(user, 0, nullptr, ReadStrict, &memory);
		Check(way != nullptr && way->visited[0] == (user[1] & 1u), "set: a ring entry was not found or answered for another entry");
		Check(way == &set.ways[0], "set: a hit did not move to the front");
	}
	// The guest word under the recorded read changed: not found; recording it again keeps the old entry (another combination) and the new one hits first.
	memory.value = 0x56u;
	Check(set.Find(ring[0], 0, nullptr, ReadStrict, &memory) == nullptr, "set: a changed guest word was found");
	store(ring[0]);
	Check(set.Find(ring[0], 0, nullptr, ReadStrict, &memory) == &set.ways[0], "set: the new entry did not hit");
	memory.value = 0x55u;
	Check(set.Find(ring[1], 0, nullptr, ReadStrict, &memory) != nullptr, "set: the old entry was lost");
	// More entries than ways: the least recently used goes, the newest stay.
	for (uint32_t i = 0; i < SrtCfMemoSet::kWays + 2u; ++i) store(std::array<uint32_t, 4> {0u, 1000u + i, 0u, 0u});
	Check(set.Find(std::array<uint32_t, 4> {0u, 1000u + SrtCfMemoSet::kWays + 1u, 0u, 0u}, 0, nullptr, ReadStrict, &memory) != nullptr, "set: the newest entry was lost");
	Check(set.Find(std::array<uint32_t, 4> {0u, 1000u, 0u, 0u}, 0, nullptr, ReadStrict, &memory) == nullptr, "set: the oldest entry was kept");
}

// ---- nodes: scalar reads recorded as how their address follows from the handle words

uint32_t g_tables[4][16];

bool ReadHost(void*, uint64_t address, std::span<uint32_t> out) {
	std::memcpy(out.data(), reinterpret_cast<const void*>(address), out.size_bytes());
	return true;
}

// V# words of a buffer at `table` with `records` bytes and no stride.
std::array<uint32_t, 4> Vsharp(const uint32_t* table, uint32_t records) {
	const auto address = reinterpret_cast<uint64_t>(table);
	return {static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u), records, 0u};
}

// One ReadConstBuffer whose V# is in user data words 0..3, reading the DWORD at byte 8; compare as given.
SrtCfMemo RecordFlag(const std::array<uint32_t, 4>& vsharp, bool compare, uint32_t expect_word, bool expect_ok = true) {
	SrtCfMemo memo;
	memo.Begin(4u, ReadHost, nullptr);
	CfNode node;
	node.constbuf  = true;
	node.src_kind  = {CfNode::Src::User, CfNode::Src::User, CfNode::Src::User, CfNode::Src::User};
	node.src_index = {0u, 1u, 2u, 3u};
	node.mem_offset = 8;
	node.reader     = ReadHost;
	node.kind       = CfReadKind::Read;
	node.ok         = expect_ok;
	node.compare    = compare;
	node.word       = expect_word;
	(void)vsharp;
	memo.AddNode(node);
	memo.End(std::array<uint8_t, 2> {1u, 0u});
	return memo;
}

void TestNodes() {
	for (auto& table: g_tables) std::memset(table, 0, sizeof(table));
	g_tables[0][2] = 1u;
	g_tables[1][2] = 1u; // another buffer, the same flag
	g_tables[2][2] = 0u; // another buffer, another flag
	const auto a = Vsharp(g_tables[0], 64);
	const auto b = Vsharp(g_tables[1], 64);
	const auto c = Vsharp(g_tables[2], 64);
	const auto memo = RecordFlag(a, true, 1u);
	Check(memo.valid && memo.user_mask == 0, "nodes: the user data words of the handle are in the key");
	Check(memo.Matches(a, 0, ReadHost, nullptr, nullptr), "nodes: the recorded buffer did not match");
	Check(memo.Matches(b, 0, ReadHost, nullptr, nullptr), "nodes: another buffer with the same flag did not match");
	Check(!memo.Matches(c, 0, ReadHost, nullptr, nullptr), "nodes: another buffer with another flag matched");
	// The flag changed in place.
	g_tables[0][2] = 0u;
	Check(!memo.Matches(a, 0, ReadHost, nullptr, nullptr), "nodes: a changed flag matched");
	g_tables[0][2] = 1u;
	// The buffer got too small for the offset: another path (nothing is read), whatever the flag says.
	Check(!memo.Matches(Vsharp(g_tables[0], 8), 0, ReadHost, nullptr, nullptr), "nodes: a buffer too small for the offset matched");
	// A node that nobody but a handle uses is not compared: a different answer is fine.
	const auto structural = RecordFlag(a, false, 77u);
	Check(structural.Matches(a, 0, ReadHost, nullptr, nullptr), "nodes: a word only used as a handle word was compared");
	// ... but it has to succeed as before.
	const auto failed = RecordFlag(a, true, 0u, false);
	Check(!failed.Matches(a, 0, ReadHost, nullptr, nullptr), "nodes: a read that succeeds now matched a recorded failure");
	// A user data table that is shorter than the words the node takes.
	const std::array<uint32_t, 2> short_table {a[0], a[1]};
	SrtCfMemo shorter = RecordFlag(a, true, 1u);
	shorter.user_size = 2;
	Check(!shorter.Matches(short_table, 0, ReadHost, nullptr, nullptr), "nodes: a user data table too short for the node matched");
}

// A directory (LoadAddressU32 of user data 0,1 at +0..+12) holds the V# of the buffer that holds the flag: the V# words are the answers of four nodes that no
// condition uses; only the flag is compared.
void TestNodeChain() {
	for (auto& table: g_tables) std::memset(table, 0, sizeof(table));
	// g_tables[0] is the directory (V# at dword 0..3); the buffers are g_tables[1], g_tables[2].
	const auto set_directory = [&](const uint32_t* buffer) {
		const auto v = Vsharp(buffer, 64);
		for (size_t i = 0; i < 4; ++i) g_tables[0][i] = v[i];
	};
	g_tables[1][2] = 1u;
	g_tables[2][2] = 1u;
	g_tables[3][2] = 0u;
	set_directory(g_tables[1]);
	SrtCfMemo memo;
	memo.Begin(2u, ReadHost, nullptr);
	for (uint32_t i = 0; i < 4; ++i) {
		CfNode load;
		load.src_kind   = {CfNode::Src::User, CfNode::Src::User};
		load.src_index  = {0u, 1u};
		load.mem_offset = static_cast<int32_t>(i * 4u);
		load.reader     = ReadHost;
		load.kind       = CfReadKind::Read;
		load.ok         = true;
		load.compare    = false;
		load.word       = g_tables[0][i];
		memo.AddNode(load);
	}
	CfNode flag;
	flag.constbuf   = true;
	flag.src_kind   = {CfNode::Src::Node, CfNode::Src::Node, CfNode::Src::Node, CfNode::Src::Node};
	flag.src_index  = {0u, 1u, 2u, 3u};
	flag.mem_offset = 8;
	flag.reader     = ReadHost;
	flag.kind       = CfReadKind::Read;
	flag.ok         = true;
	flag.compare    = true;
	flag.word       = 1u;
	memo.AddNode(flag);
	memo.End(std::array<uint8_t, 1> {1u});
	const auto directory = Vsharp(g_tables[0], 64);
	const std::array<uint32_t, 2> user {directory[0], directory[1]};
	Check(memo.Matches(user, 0, ReadHost, nullptr, nullptr), "chain: the recorded directory did not match");
	set_directory(g_tables[2]);
	Check(memo.Matches(user, 0, ReadHost, nullptr, nullptr), "chain: another buffer with the same flag behind the directory did not match");
	set_directory(g_tables[3]);
	Check(!memo.Matches(user, 0, ReadHost, nullptr, nullptr), "chain: another buffer with another flag behind the directory matched");
	set_directory(g_tables[1]);
	g_tables[1][2] = 0u;
	Check(!memo.Matches(user, 0, ReadHost, nullptr, nullptr), "chain: a changed flag behind the directory matched");
}

void TestAddress() {
	const uint64_t heap = 0x7000'0000'1000ull;
	// ReadConstBuffer
	auto where = CfConstBufferAddress(static_cast<uint32_t>(heap), heap >> 32u, 64, 8, 0, false);
	Check(where.kind == CfReadKind::Read && where.address == heap + 8u, "address: constant buffer read");
	where = CfConstBufferAddress(static_cast<uint32_t>(heap), heap >> 32u, 64, 8, 3, false);
	Check(where.kind == CfReadKind::Read && where.address == heap + 8u, "address: the operand is DWORD aligned");
	where = CfConstBufferAddress(static_cast<uint32_t>(heap), heap >> 32u, 8, 8, 0, false);
	Check(where.kind == CfReadKind::Outside, "address: a read at the end of the buffer");
	where = CfConstBufferAddress(static_cast<uint32_t>(heap), heap >> 32u, 11, 8, 0, false);
	Check(where.kind == CfReadKind::Outside, "address: a DWORD that does not fit");
	where = CfConstBufferAddress(static_cast<uint32_t>(heap), (heap >> 32u) | (4u << 16u), 4, 8, 0, false);
	Check(where.kind == CfReadKind::Read, "address: the stride multiplies the records");
	where = CfConstBufferAddress(static_cast<uint32_t>(heap), heap >> 32u, 64, -4, 0, false);
	Check(where.kind == CfReadKind::Fail, "address: a negative immediate");
	where = CfConstBufferAddress(0x100, 0, 64, 8, 0, true);
	Check(where.kind == CfReadKind::NullTable, "address: a table below guest memory under the host rule");
	where = CfConstBufferAddress(0x100, 0, 64, 8, 0, false);
	Check(where.kind == CfReadKind::Read, "address: a table below guest memory without the host rule");
	// LoadAddressU32
	where = CfLoadAddress(static_cast<uint32_t>(heap), heap >> 32u, 12, 0, false);
	Check(where.kind == CfReadKind::Read && where.address == heap + 12u, "address: pointer load");
	where = CfLoadAddress(static_cast<uint32_t>(heap) | 3u, heap >> 32u, -8, 0, false);
	Check(where.kind == CfReadKind::Read && where.address == heap - 8u, "address: negative immediate of a pointer load");
	where = CfLoadAddress(0, 0, -8, 0, false);
	Check(where.kind == CfReadKind::Fail, "address: a pointer load below zero");
	where = CfLoadAddress(0xfffffffcu, 0xffffu, 8, 0, false);
	Check(where.kind == CfReadKind::Fail, "address: a pointer load past the address space");
	where = CfLoadAddress(0x100, 0, 4, 0, true);
	Check(where.kind == CfReadKind::NullTable, "address: a null table pointer under the host rule");
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
	TestNodes();
	TestNodeChain();
	TestAddress();
	TestMode();
	std::puts("SrtCfMemoTests: all cases passed");
	return 0;
}
