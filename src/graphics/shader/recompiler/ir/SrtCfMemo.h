#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCFMEMO_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCFMEMO_H_

// S451 KYTY_PROGS_MEMO=0|1|shadow (default 0): the reachability loop of SrtWalker::RefreshFlatBufferWalk (the block conditions of a plan with control flow) is a
// function of exactly three inputs: the user data words and the shader base it reads (GetUserData / GetShaderBase) and the guest words it reads (the raw
// reads). The memo records these during one loop and the visited blocks it ended with. A later walk of the same plan whose recorded user data words and
// shader base are equal, and whose recorded reads all return what they returned (same ok, same word), ends with the same visited blocks, so the loop is skipped.
// The user data words that were not read do not matter: that is what makes it hit when every draw has other constants.
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

// A guest read as the loop issued it: through which reader (null = the direct host copy of the main walker, else the function), the address, and the answer.
using CfMemoReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

struct SrtCfMemo {
	static constexpr uint32_t kMaxUserData = 64; // a user data index beyond this makes the loop unrecordable
	static constexpr uint64_t kGuestMemoryMin = 0x40000; // the direct copy reads zero below it (SrtWalker.cpp GUEST_MEMORY_MIN)

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

	void NoteRead(CfMemoReader reader, uint64_t address, uint32_t word, bool ok) {
		if (!recording) return;
#ifdef KYTY_CFMEMO_MUTANT_NO_FAILED
		if (!ok) return;
#endif
		// Only the two readers the key names can be asked again; any other function (the failing stub of a clean runtime) makes the loop unrecordable.
		if (reader != nullptr && reader != ordinary_fn && reader != strict_fn) overflow = true;
		reads.push_back({address, reader, word, ok});
	}

	// The loop ended with `final_visited`. A loop that read a user data word beyond the table is not stored.
	void End(std::span<const uint8_t> final_visited) {
		recording = false;
		if (overflow) {
			valid = false;
			return;
		}
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
			} else if (read.address >= kGuestMemoryMin) {
				std::memcpy(&word, reinterpret_cast<const void*>(read.address), sizeof(word));
			}
			if (ok != read.ok) return false;
#ifndef KYTY_CFMEMO_MUTANT_ADDR_ONLY
			if (ok && word != read.word) return false;
#endif
		}
		return true;
	}

	bool Matches(std::span<const uint32_t> user, uint64_t base, CfMemoReader ordinary, CfMemoReader strict, void* userdata) const {
		return InputsMatch(user, base, ordinary, strict) && ReadsUnchanged(userdata);
	}
};

// The memos of one plan. The user data words that decide the loop are mostly pointers, and a frame ring hands out a few of them in turn (three constant
// buffers: the pointer of draw N is the one of draw N-3), so one memo per plan would miss every time; a few ways keep the recent ones.
struct SrtCfMemoSet {
	static constexpr uint32_t kWays = 256;

	std::array<SrtCfMemo, kWays> ways;
	uint32_t                     next = 0;

	// The way whose answer is current for these inputs (inputs equal and every recorded read answers the same), or null.
	// `why` (optional) tells a miss apart: 0 nothing stored, 1 no way has these inputs, 2 a way has them but a re-read answered differently.
	const SrtCfMemo* Find(std::span<const uint32_t> user, uint64_t base, CfMemoReader ordinary, CfMemoReader strict, void* userdata, uint32_t* why = nullptr) const {
		uint32_t reason = 0;
		for (const auto& way: ways) {
			if (way.valid && reason == 0) reason = 1;
			if (!way.InputsMatch(user, base, ordinary, strict)) continue;
			if (way.ReadsUnchanged(userdata)) return &way;
			reason = 2;
		}
		if (why != nullptr) *why = reason;
		return nullptr;
	}

	// The way to record into after a miss: the one whose inputs are equal (its reads changed), else an unused one, else the oldest.
	SrtCfMemo& Victim(std::span<const uint32_t> user, uint64_t base, CfMemoReader ordinary, CfMemoReader strict) {
		for (auto& way: ways) {
			if (way.InputsMatch(user, base, ordinary, strict)) return way;
		}
		for (auto& way: ways) {
			if (!way.valid && !way.recording) return way;
		}
		return ways[next++ % kWays];
	}
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTCFMEMO_H_ */
