#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RTPASSMEMO_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RTPASSMEMO_H_

// S552 KYTY_RT_PASS_MEMO=0|1|shadow: ResolveRenderColorTarget builds the image description of a colour slot from the slot's CB registers
// (CB_COLOR*_BASE/VIEW/INFO/ATTRIB/ATTRIB2/ATTRIB3/DCC_CONTROL/CMASK/FMASK/CLEAR_WORD0/1/DCC_BASE = HW::RenderTarget) and the slot's
// CB_TARGET_MASK nibble, then looks the image up (TextureCache::FindImage). Inside a pass every draw repeats both with the same registers.
//
// Key: the slot's HW::RenderTarget bytes, the mask nibble and the call flags (ignore_target_mask, exact_format). Value: the resolved
// RenderColorInfo plus a TexMemo entry (image id + registration generation of the lookup).
// A hit skips the description (format tables, pitch/DCC size, UpdateSize, the EXIT checks that passed when it was stored), ValidateImageDesc,
// the ExactImageKey hash and lookup. It still runs, through TextureCache::ReuseImage, what FindImage's `known` branch runs: lock, slot and
// registration checks (RegistrationsSince on the image's pages), tick + LRU touch and MaterializeColorClear (DCC/CMASK clear state lives in
// guest memory, not in registers). TouchMeta, the attachment layouts and the depth target stay per draw (AcquireRenderTargets).
// Only results FindImage itself would remember (TexMemo::CanStore: exact, no view adjustment, no null image) are stored.
//
// Off: unchanged. 1: as above. shadow: always resolves in full; when the memo would have hit, compares the result (rt_memo_ne_n).
//
// Header only and templated on the register and value types, so tests/RtPassMemoTests.cpp pins the key with fake types.
// Mutants (tests, WILL_FAIL): KYTY_RTMEMO_MUTANT_{NO_MASK, PREFIX}.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace Libs::Graphics::RtPassMemo {

enum class Mode : uint8_t { Off, On, Shadow };

// null/""/"0" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
[[nodiscard]] inline Mode ParseMode(const char* text) {
	if (text == nullptr || text[0] == '\0') return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_RT_PASS_MEMO"));
	return mode;
}

// Call flags in the key.
[[nodiscard]] constexpr uint32_t Flags(bool ignore_target_mask, bool exact_format) {
	return (ignore_target_mask ? 1u : 0u) | (exact_format ? 2u : 0u);
}

template <class Regs, class Value>
struct Slot {
	static_assert(std::is_trivially_copyable_v<Regs>);

	Regs     regs {};
	uint32_t mask  = 0;
	uint32_t flags = 0;
	bool     valid = false;
	Value    value {};

	// Byte comparison: equal bytes are equal registers. Padding that differs only costs a miss.
	[[nodiscard]] bool Matches(const Regs& now, uint32_t now_mask, uint32_t now_flags) const {
		if (!valid || flags != now_flags) return false;
#if !defined(KYTY_RTMEMO_MUTANT_NO_MASK)
		if (mask != now_mask) return false;
#endif
#if defined(KYTY_RTMEMO_MUTANT_PREFIX)
		return std::memcmp(&regs, &now, sizeof(uint64_t)) == 0;
#else
		return std::memcmp(&regs, &now, sizeof(Regs)) == 0;
#endif
	}
	void Store(const Regs& now, uint32_t now_mask, uint32_t now_flags, const Value& now_value) {
		std::memcpy(&regs, &now, sizeof(Regs));
		mask  = now_mask;
		flags = now_flags;
		value = now_value;
		valid = true;
	}
	void Clear() { valid = false; }
};

} // namespace Libs::Graphics::RtPassMemo

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_RTPASSMEMO_H_
