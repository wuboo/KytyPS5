#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCFMEMO_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCFMEMO_H_

// S451 KYTY_PROGS_MEMO=0|1|shadow (default 0): the reachability loop of SrtWalker::RefreshFlatBufferWalk (the block conditions of a plan with control flow) is a
// function of the user data words and the shader base it reads (GetUserData / GetShaderBase) and of the guest words it reads (the raw reads). The memo records
// these during one loop together with the visited blocks it ended with. A later walk of the same plan whose recorded inputs are equal ends with the same visited
// blocks, so the loop is skipped.
//
// The reads come in two kinds. A read whose handle words are user data words or the results of other such reads (a constant buffer whose V# sits in the user
// data, a table pointer in the user data) is recorded as a NODE: how its address follows from the handle words, not the address itself. Draws that differ in the
// pointer but not in the flags behind it then hit, and only the words that decide a condition are compared. Every other read is recorded as an absolute read
// (address and answer) and the user data words it was computed from are part of the key.
// Off = loop runs; On = skip on a hit; Shadow = loop runs anyway and the memo's answer is compared with it.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

enum class CfMemoMode : uint8_t { Off, On, Shadow };

inline CfMemoMode ParseCfMemoMode(const char* text) {
	if (text == nullptr) return CfMemoMode::Off;
	if (std::strcmp(text, "1") == 0) return CfMemoMode::On;
	if (std::strcmp(text, "shadow") == 0) return CfMemoMode::Shadow;
	return CfMemoMode::Off;
}

inline CfMemoMode CfMemoModeFromEnv() {
	static const CfMemoMode mode = ParseCfMemoMode(std::getenv("KYTY_PROGS_MEMO"));
	return mode;
}

// Tests set a mode without the environment.
inline std::optional<CfMemoMode>& CfMemoModeTestOverride() {
	static std::optional<CfMemoMode> mode;
	return mode;
}

inline CfMemoMode CfMemoModeCurrent() {
	if (const auto& forced = CfMemoModeTestOverride()) return *forced;
	return CfMemoModeFromEnv();
}

// A guest read as the loop issued it: through which reader (null = the direct host copy of the main walker, else the function).
using CfMemoReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

inline constexpr uint64_t kCfAddressMask    = 0x0000ffffffffffffull;
inline constexpr uint64_t kCfGuestMemoryMin = 0x40000; // the direct copy reads zero below it (SrtWalker.cpp GUEST_MEMORY_MIN)

// What a raw read does once the words of its handle are known. The interpreter (EvaluateRawRead) and the memo's replay both use these two functions.
enum class CfReadKind : uint8_t {
	Fail,      // the read fails (negative offset, address overflow)
	Outside,   // beyond the buffer: the answer is zero, nothing is read
	NullTable, // the table pointer is below guest memory: the answer is zero, nothing is read
	Read,      // the DWORD at `address`
};

struct CfAddress {
	CfReadKind kind    = CfReadKind::Fail;
	uint64_t   address = 0;
};

// ReadConstBuffer: V# words low, high (stride in bits 16..29), records, word 3 (unused); `mem_offset` is the immediate of the instruction, `offset_arg` its operand.
inline CfAddress CfConstBufferAddress(uint64_t low, uint64_t high, uint64_t records, int32_t mem_offset, uint32_t offset_arg, bool host_rule) {
	const uint64_t base      = ((high << 32u) | static_cast<uint32_t>(low)) & kCfAddressMask;
	const int64_t  immediate = mem_offset;
	if (immediate < 0) return {CfReadKind::Fail, 0};
	const uint64_t byte_offset = (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (offset_arg & ~3u);
	const uint32_t stride      = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
	const uint64_t size        = stride == 0 ? static_cast<uint64_t>(static_cast<uint32_t>(records)) : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
	if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) return {CfReadKind::Outside, 0};
	const uint64_t address = (base & ~uint64_t {3}) + byte_offset;
	// A host-memory read through a table below guest memory reads zero, whatever its offset.
	if (base < kCfGuestMemoryMin && host_rule) return {CfReadKind::NullTable, 0};
	return {CfReadKind::Read, address};
}

// LoadAddressU32: a pointer (low, high) plus the immediate and the operand.
inline CfAddress CfLoadAddress(uint64_t low, uint64_t high, int32_t mem_offset, uint32_t offset_arg, bool host_rule) {
	const uint64_t base      = ((high << 32u) | static_cast<uint32_t>(low)) & kCfAddressMask;
	const int64_t  immediate = mem_offset;
	const int64_t  relative  = (immediate & ~int64_t {3}) + static_cast<int64_t>(offset_arg & ~3u);
	const uint64_t start     = base & ~uint64_t {3};
	uint64_t       address   = 0;
	if (start > kCfAddressMask) return {CfReadKind::Fail, 0};
	if (relative < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(relative);
		if (magnitude > start) return {CfReadKind::Fail, 0};
		address = start - magnitude;
	} else {
		const auto magnitude = static_cast<uint64_t>(relative);
		if (magnitude > kCfAddressMask - start) return {CfReadKind::Fail, 0};
		address = start + magnitude;
	}
	if (base < kCfGuestMemoryMin && host_rule) return {CfReadKind::NullTable, 0};
	return {CfReadKind::Read, address};
}

// One raw read of the loop as a node: the sources of its handle words (a user data word or an earlier node), what it asked and what it got.
struct CfNode {
	enum class Src : uint8_t { User, Node };

	bool                    constbuf = false; // ReadConstBuffer (four handle words) else LoadAddressU32 (two)
	std::array<Src, 4>      src_kind {};
	std::array<uint32_t, 4> src_index {};     // User: index into the user data; Node: index into the nodes
	int32_t                 mem_offset = 0;
	uint32_t                offset_arg = 0;
	CfMemoReader            reader     = nullptr;
	bool                    host_rule  = false;
	// what happened
	CfReadKind              kind    = CfReadKind::Fail;
	bool                    ok      = false;
	bool                    compare = false; // the answer is used by something other than a handle word: it has to be equal
	uint32_t                word    = 0;
	const void*             inst    = nullptr; // recording only
};

struct SrtCfMemo {
	static constexpr uint32_t kMaxUserData = 64; // a user data index beyond this makes the loop unrecordable
	static constexpr uint32_t kMaxNodes    = 64;

	struct Read {
		uint64_t     address = 0;
		CfMemoReader reader  = nullptr;
		uint32_t     word    = 0;
		bool         ok      = false;
	};

	// ---- the stored answer
	bool                  valid = false;
	uint32_t              user_size = 0;
	bool                  base_used = false;
	uint64_t              shader_base = 0;
	uint64_t              user_mask = 0;     // bit i: user data word i was read
	std::array<uint32_t, kMaxUserData> user_words {};
	std::vector<Read>     reads;
	std::vector<CfNode>   nodes;
	std::vector<uint8_t>  visited;
	// The reader functions of the runtime the loop ran with (null = none): the loop's reads mean other things under other readers.
	CfMemoReader          ordinary_fn = nullptr;
	CfMemoReader          strict_fn   = nullptr;

	// ---- the recording of the loop in progress
	bool                  recording = false;
	bool                  overflow  = false;

	void Begin(uint32_t user_data_size, CfMemoReader ordinary, CfMemoReader strict) {
		valid = false;
		recording = true;
		overflow = false;
		user_size = user_data_size;
		base_used = false;
		shader_base = 0;
		user_mask = 0;
		reads.clear();
		nodes.clear();
		visited.clear();
		ordinary_fn = ordinary;
		strict_fn = strict;
	}

	void NoteUserData(uint32_t index, uint32_t value) {
		if (!recording) return;
		if (index >= kMaxUserData) {
			overflow = true;
			return;
		}
		user_mask |= uint64_t {1} << index;
		user_words[index] = value;
	}

	void NoteShaderBase(uint64_t base) {
		if (!recording) return;
		base_used = true;
		shader_base = base;
	}

	bool ReaderKnown(CfMemoReader reader) const {
		return reader == nullptr || reader == ordinary_fn || reader == strict_fn;
	}

	void NoteRead(CfMemoReader reader, uint64_t address, uint32_t word, bool ok) {
		if (!recording) return;
#ifdef KYTY_CFMEMO_MUTANT_NO_FAILED
		if (!ok) return;
#endif
		// Only the two readers the key names can be asked again; any other function (the failing stub of a clean runtime) makes the loop unrecordable.
		if (!ReaderKnown(reader)) overflow = true;
		reads.push_back({address, reader, word, ok});
	}

	// The latest node of this instruction, or UINT32_MAX.
	uint32_t NodeOf(const void* inst) const {
		for (size_t i = nodes.size(); i-- > 0;) {
			if (nodes[i].inst == inst) return static_cast<uint32_t>(i);
		}
		return UINT32_MAX;
	}

	// Returns the index of the new node, or UINT32_MAX (and the loop is unrecordable) when there are too many or the reader is unknown.
	uint32_t AddNode(const CfNode& node) {
		if (!recording) return UINT32_MAX;
		if (nodes.size() >= kMaxNodes || !ReaderKnown(node.reader)) {
			overflow = true;
			return UINT32_MAX;
		}
		nodes.push_back(node);
		return static_cast<uint32_t>(nodes.size() - 1);
	}

	// Something other than a handle word used the answer of this instruction (or of this node).
	void MarkConsumed(const void* inst) {
		if (const auto index = NodeOf(inst); index != UINT32_MAX) nodes[index].compare = true;
	}

	void MarkConsumedNode(uint32_t index) {
		if (index < nodes.size()) nodes[index].compare = true;
	}

	// The loop ended with `final_visited`. A loop that could not be recorded is not stored.
	void End(std::span<const uint8_t> final_visited) {
		recording = false;
		if (overflow) {
			valid = false;
			return;
		}
		for (auto& node: nodes) node.inst = nullptr;
		visited.assign(final_visited.begin(), final_visited.end());
		valid = true;
	}

	void Abort() {
		recording = false;
		valid = false;
	}

	// Equal user data words (the ones read), shader base (when read), table size and readers.
	bool InputsMatch(std::span<const uint32_t> user, uint64_t base, CfMemoReader ordinary, CfMemoReader strict) const {
		if (!valid || user.size() != user_size || ordinary != ordinary_fn || strict != strict_fn) return false;
#ifndef KYTY_CFMEMO_MUTANT_IGNORES_BASE
		if (base_used && base != shader_base) return false;
#endif
#ifndef KYTY_CFMEMO_MUTANT_IGNORES_USERDATA
		for (uint32_t i = 0; i < kMaxUserData; ++i) {
			if ((user_mask >> i & 1u) != 0 && user[i] != user_words[i]) return false;
		}
#endif
		return true;
	}

	// Reads every recorded address again; true when each answers as it did.
	bool ReadsUnchanged(void* userdata) const {
		for (const auto& read: reads) {
			uint32_t word = 0;
			bool     ok   = true;
			if (read.reader != nullptr) {
				ok = read.reader(userdata, read.address, std::span<uint32_t>(&word, 1));
			} else if (read.address >= kCfGuestMemoryMin) {
				std::memcpy(&word, reinterpret_cast<const void*>(read.address), sizeof(word));
			}
			if (ok != read.ok) return false;
#ifndef KYTY_CFMEMO_MUTANT_ADDR_ONLY
			if (ok && word != read.word) return false;
#endif
		}
		return true;
	}

	// Asks every node again with the current user data: the same address computation, the same reader. True when each node takes the same path (kind), succeeds
	// as before and, where something other than a handle word uses it, answers the same.
	bool NodesUnchanged(std::span<const uint32_t> user, void* userdata) const {
		std::array<uint64_t, kMaxNodes> result {};
		for (size_t i = 0; i < nodes.size(); ++i) {
			const auto&             node  = nodes[i];
			std::array<uint64_t, 4> words {};
			const size_t            count = node.constbuf ? 4 : 2;
			for (size_t k = 0; k < count; ++k) {
				if (node.src_kind[k] == CfNode::Src::User) {
					if (node.src_index[k] >= user.size()) return false;
					words[k] = user[node.src_index[k]];
				} else {
					words[k] = result[node.src_index[k]];
				}
			}
			const auto address = node.constbuf ? CfConstBufferAddress(words[0], words[1], words[2], node.mem_offset, node.offset_arg, node.host_rule)
			                                   : CfLoadAddress(words[0], words[1], node.mem_offset, node.offset_arg, node.host_rule);
			if (address.kind != node.kind) return false;
			uint32_t word = 0;
			bool     ok   = true;
			switch (address.kind) {
				case CfReadKind::Fail: ok = false; break;
				case CfReadKind::Outside:
				case CfReadKind::NullTable: break;
				case CfReadKind::Read:
					if (node.reader != nullptr) {
						ok = node.reader(userdata, address.address, std::span<uint32_t>(&word, 1));
					} else if (address.address >= kCfGuestMemoryMin) {
						std::memcpy(&word, reinterpret_cast<const void*>(address.address), sizeof(word));
					}
					break;
			}
			if (ok != node.ok) return false;
#ifndef KYTY_CFMEMO_MUTANT_NODE_VALUE_IGNORED
			if (ok && node.compare && word != node.word) return false;
#endif
			result[i] = ok ? word : 0;
		}
		return true;
	}

	bool Matches(std::span<const uint32_t> user, uint64_t base, CfMemoReader ordinary, CfMemoReader strict, void* userdata) const {
		return InputsMatch(user, base, ordinary, strict) && NodesUnchanged(user, userdata) && ReadsUnchanged(userdata);
	}
};

// The memos of one plan, most recently used first. Draws of a plan mostly differ in a few flags, so a few ways cover the combinations; a lookup tries at most
// kTries of them (each one asks the guest words again).
struct SrtCfMemoSet {
	static constexpr uint32_t kWays  = 16;
	static constexpr uint32_t kTries = 6;

	std::array<SrtCfMemo, kWays> ways;

	// The way whose answer is current for these inputs (inputs equal and every recorded read answers the same), or null. A hit moves to the front.
	// `why` (optional) tells a miss apart: 0 nothing stored, 1 no way has these inputs, 2 a way has them but a re-read answered differently.
	const SrtCfMemo* Find(std::span<const uint32_t> user, uint64_t base, CfMemoReader ordinary, CfMemoReader strict, void* userdata, uint32_t* why = nullptr) {
		uint32_t reason = 0;
		uint32_t tried  = 0;
		for (uint32_t i = 0; i < kWays; ++i) {
			auto& way = ways[i];
			if (way.valid && reason == 0) reason = 1;
			if (!way.InputsMatch(user, base, ordinary, strict)) continue;
			if (tried++ >= kTries) break;
			if (way.NodesUnchanged(user, userdata) && way.ReadsUnchanged(userdata)) {
				Promote(i);
				return &ways[0];
			}
			reason = 2;
		}
		if (why != nullptr) *why = reason;
		return nullptr;
	}

	// The way to record into after a miss: an unused one, else the least recently used.
	SrtCfMemo& Victim() {
		for (auto& way: ways) {
			if (!way.valid && !way.recording) return way;
		}
		return ways[kWays - 1];
	}

	// Moves a way to the front (after it was recorded or hit).
	void Promote(uint32_t index) {
		for (; index > 0; --index) std::swap(ways[index], ways[index - 1]);
	}
	void Promote(const SrtCfMemo* way) {
		Promote(static_cast<uint32_t>(way - ways.data()));
	}
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCFMEMO_H_ */
