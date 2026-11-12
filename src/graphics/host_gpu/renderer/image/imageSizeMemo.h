#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGESIZEMEMO_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGESIZEMEMO_H_

// S546 KYTY_IMAGE_SIZE_MEMO=0|1|shadow: ImageInfo::UpdateSize is a pure function of a few description fields, and the CP thread
// recomputes it for every texture of every draw (ResolveTexture) and every colour target (ResolveRenderColorTarget). The result is
// remembered per thread in a direct-mapped table keyed by every field UpdateSize reads, and copied back on a hit.
//
// Read by UpdateSize (the key): data.address (CompactMipSize tests its alignment), data.size (0 = derive it), pixel_format (IsDepth,
// S8 planes), guest_format, type (volume), extent, resources, pitch (0 = derive it), bytes_per_block (0 = derive it), samples,
// tile_mode, metadata.kind (TiledMipRange). Written by UpdateSize (the value): data.size, pitch, bytes_per_block, mip_layout, tiling,
// tiled_slice_stride, linear_slice_stride, first_tail_level. Nothing else of the info is read or written.
//
// Off: UpdateSize as before. 1: a hit skips the computation (and its EXIT checks, which passed on the miss that stored it). shadow:
// always computes, and a hit whose remembered value differs counts a disagreement.
//
// Header only and templated on the info type, so tests/ImageSizeMemoTests.cpp pins it with a fake info and a fake computation.
// Mutants (tests, WILL_FAIL): KYTY_SIZEMEMO_MUTANT_{NO_ADDRESS, NO_SIZE, NO_MIPS}.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace Libs::Graphics::SizeMemo {

enum class Mode : uint8_t { Off, On, Shadow };

// null/""/"0" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
[[nodiscard]] inline Mode ParseMode(const char* text) {
	if (text == nullptr || text[0] == '\0') return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_IMAGE_SIZE_MEMO"));
	return mode;
}

enum class Result : uint8_t { Hit, Miss, Disagree };

inline constexpr uint32_t kSlotBits = 10;
inline constexpr uint32_t kSlots    = 1u << kSlotBits;

struct Key {
	static constexpr uint32_t Words = 9;
	uint64_t                  words[Words] {};

	[[nodiscard]] bool operator==(const Key& other) const noexcept {
		for (uint32_t i = 0; i < Words; i++) {
			if (words[i] != other.words[i]) return false;
		}
		return true;
	}
	[[nodiscard]] uint32_t Slot() const noexcept {
		uint64_t hash = 0x9e3779b97f4a7c15ull;
		for (const auto word: words) {
			hash = (hash ^ word) * 0xff51afd7ed558ccdull;
			hash ^= hash >> 32u;
		}
		return static_cast<uint32_t>(hash >> (64u - kSlotBits));
	}
};

[[nodiscard]] constexpr uint64_t Pack(uint64_t high, uint64_t low) noexcept {
	return (high << 32u) | (low & 0xffffffffull);
}

template <class Info>
[[nodiscard]] Key MakeKey(const Info& info) noexcept {
	Key key;
#if defined(KYTY_SIZEMEMO_MUTANT_NO_ADDRESS)
	key.words[0] = 0;
#else
	key.words[0] = info.data.address;
#endif
#if defined(KYTY_SIZEMEMO_MUTANT_NO_SIZE)
	key.words[1] = 0;
#else
	key.words[1] = info.data.size;
#endif
	key.words[2] = Pack(static_cast<uint64_t>(info.pixel_format), static_cast<uint64_t>(info.guest_format));
	key.words[3] = Pack(static_cast<uint64_t>(info.type), static_cast<uint64_t>(info.tile_mode));
	key.words[4] = Pack(info.extent.width, info.extent.height);
	key.words[5] = Pack(info.extent.depth, info.resources.levels);
	key.words[6] = Pack(info.resources.layers, info.pitch);
	key.words[7] = Pack(info.bytes_per_block, info.samples);
	key.words[8] = static_cast<uint64_t>(info.metadata.kind);
	return key;
}

template <class Info>
struct Value {
	decltype(Info::mip_layout) mip_layout {};
	decltype(Info::tiling)     tiling {};
	uint64_t                   data_size           = 0;
	uint64_t                   tiled_slice_stride  = 0;
	uint64_t                   linear_slice_stride = 0;
	uint32_t                   pitch               = 0;
	uint32_t                   bytes_per_block     = 0;
	uint32_t                   first_tail_level    = 0;

	static Value From(const Info& info) {
		Value value;
		value.mip_layout          = info.mip_layout;
		value.tiling              = info.tiling;
		value.data_size           = info.data.size;
		value.tiled_slice_stride  = info.tiled_slice_stride;
		value.linear_slice_stride = info.linear_slice_stride;
		value.pitch               = info.pitch;
		value.bytes_per_block     = info.bytes_per_block;
		value.first_tail_level    = info.first_tail_level;
		return value;
	}
	void To(Info& info) const {
#if !defined(KYTY_SIZEMEMO_MUTANT_NO_MIPS)
		info.mip_layout = mip_layout;
#endif
		info.tiling              = tiling;
		info.data.size           = data_size;
		info.tiled_slice_stride  = tiled_slice_stride;
		info.linear_slice_stride = linear_slice_stride;
		info.pitch               = pitch;
		info.bytes_per_block     = bytes_per_block;
		info.first_tail_level    = first_tail_level;
	}
	// The mip and tiling structs hold only integers without padding, so their bytes compare them.
	[[nodiscard]] bool Same(const Value& other) const noexcept {
		return std::memcmp(&mip_layout, &other.mip_layout, sizeof(mip_layout)) == 0 &&
		       std::memcmp(&tiling, &other.tiling, sizeof(tiling)) == 0 && data_size == other.data_size &&
		       tiled_slice_stride == other.tiled_slice_stride && linear_slice_stride == other.linear_slice_stride &&
		       pitch == other.pitch && bytes_per_block == other.bytes_per_block &&
		       first_tail_level == other.first_tail_level;
	}
};

template <class Info>
struct Slot {
	Key         key;
	Value<Info> value;
	bool        valid = false;
};

// One table per thread and info type, allocated on the first use of the thread.
template <class Info>
[[nodiscard]] Slot<Info>* Table() {
	thread_local std::unique_ptr<Slot<Info>[]> table;
	if (!table) {
		table = std::make_unique<Slot<Info>[]>(kSlots);
	}
	return table.get();
}

// mode != Off. compute(info) is the full UpdateSize.
template <class Info, class Compute>
Result Run(Info& info, Compute&& compute, Mode mode) {
	const Key key  = MakeKey(info);
	auto&     slot = Table<Info>()[key.Slot()];
	const bool hit = slot.valid && slot.key == key;
	if (hit && mode == Mode::On) {
		slot.value.To(info);
		return Result::Hit;
	}
	compute(info);
	const auto value = Value<Info>::From(info);
	if (hit) {
		const bool same = slot.value.Same(value);
		slot.value      = value;
		return same ? Result::Hit : Result::Disagree;
	}
	slot.key   = key;
	slot.value = value;
	slot.valid = true;
	return Result::Miss;
}

} // namespace Libs::Graphics::SizeMemo

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_IMAGE_IMAGESIZEMEMO_H_
