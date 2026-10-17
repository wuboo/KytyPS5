#ifndef KYTY_GRAPHICS_GUEST_GPU_LODSTATS_H_
#define KYTY_GRAPHICS_GUEST_GPU_LODSTATS_H_

// S393: IT_GET_LOD_STATS report (guest texture-streaming feedback). Pure helpers, no Vulkan.
// Layout (peer description, sharpemu #1029; not verified on hardware): dword 0 of the buffer = byte offset of the
// first counter (64); 256 slots of 8 bytes follow. Slot = T# word6[7:0], enabled by T# word5[25].
//   counter dword 0 = samples[23:0] | slot << 24
//   counter dword 1 = samples[23:0] | finest_mip_sampled << 24   (0 = finest, 15 = nothing sampled)
// KYTY_LOD_STATS=real (default when unset: slots sampled since the last report ask for mip 0, the rest report 15)
// | zero (the old stub) | full (every slot asks for mip 0). Any other value = zero.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::LodStats {

constexpr uint32_t kSlotCount     = 256;
constexpr uint32_t kFirstCounter  = 64;
constexpr uint32_t kCounterBytes  = 8;
constexpr uint32_t kReportBytes   = kFirstCounter + kSlotCount * kCounterBytes;
constexpr uint32_t kNoMipSampled  = 15;
constexpr uint32_t kSamplesPerUse = 0x10000;
constexpr uint32_t kMaxSamples    = 0xFFFFFF;

enum class Mode : uint8_t { Zero, Full, Real };

constexpr Mode ParseMode(const char* value) {
	if (value == nullptr) {
#ifdef KYTY_LODSTATS_MUTANT_UNSET_ZERO
		return Mode::Zero;
#else
		return Mode::Real;
#endif
	}
	const auto is = [value](const char* name) {
		uint32_t i = 0;
		for (; name[i] != 0; i++) {
			if (value[i] != name[i]) {
				return false;
			}
		}
		return value[i] == 0;
	};
	if (is("full")) {
		return Mode::Full;
	}
	if (is("real")) {
		return Mode::Real;
	}
	return Mode::Zero;
}

// `uses[slot]` = number of sampled bindings of that slot since the last report (Real only).
// Returns true when the full counter block was written; otherwise only the stub is written.
inline bool WriteReport(Mode mode, uint8_t* dst, uint32_t size, const uint32_t* uses) {
	if (dst == nullptr || size == 0) {
		return false;
	}
	std::memset(dst, 0, size);
	const bool full = mode != Mode::Zero && size >= kReportBytes;
	if (full) {
		const uint32_t header = kFirstCounter;
		std::memcpy(dst, &header, sizeof(header));
		for (uint32_t slot = 0; slot < kSlotCount; slot++) {
#ifdef KYTY_LODSTATS_MUTANT_REAL_AS_FULL
			const uint32_t used = 1;
#else
			const uint32_t used = mode == Mode::Full ? 1u : (uses != nullptr ? uses[slot] : 0u);
#endif
			const uint32_t samples = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(used) * kSamplesPerUse, kMaxSamples));
#ifdef KYTY_LODSTATS_MUTANT_NO_SLOT
			const uint32_t c0 = samples;
#else
			const uint32_t c0 = samples | (slot << 24u);
#endif
			const uint32_t c1 = samples | ((samples == 0 ? kNoMipSampled : 0u) << 24u);
			std::memcpy(dst + kFirstCounter + slot * kCounterBytes, &c0, 4);
			std::memcpy(dst + kFirstCounter + slot * kCounterBytes + 4, &c1, 4);
		}
	} else if (size >= sizeof(uint32_t)) {
		const uint32_t label = 1;
		std::memcpy(dst, &label, sizeof(label));
	}
	return full;
}

// Process-wide state: the mode is read once; Real counts sampled bindings per slot until the next report drains them.
inline Mode CurrentMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_LOD_STATS"));
	return mode;
}

inline std::atomic<uint32_t>* Uses() {
	static std::atomic<uint32_t> uses[kSlotCount] = {};
	return uses;
}

inline void NoteSampled(uint32_t slot) {
	if (slot < kSlotCount) {
		Uses()[slot].fetch_add(1, std::memory_order_relaxed);
	}
}

inline void Drain(uint32_t* out) {
	for (uint32_t i = 0; i < kSlotCount; i++) {
		out[i] = Uses()[i].exchange(0, std::memory_order_relaxed);
	}
}

} // namespace Libs::Graphics::LodStats

#endif /* KYTY_GRAPHICS_GUEST_GPU_LODSTATS_H_ */
