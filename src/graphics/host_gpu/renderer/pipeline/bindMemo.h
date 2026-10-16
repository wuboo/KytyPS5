#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDMEMO_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDMEMO_H_

// S336: KYTY_BIND_MEMO=0|1|shadow (default 0 = the code path as it was, bit for bit).
//  (a) RenderExecutor::ResolveTexture: two memo tables keyed by the raw T# dwords + the shader's image resource.
//      L1 (layout): the tiled-layout result (ImageInfo, pitch, size, view base) - a pure function of the key, no generation.
//      L2 (image):  the FindImage result (ImageId + the view base level/layer FindImage may rewrite) - valid only while the
//                   TextureCache generation (bumped on every insert/register/unregister/delete/metadata rewrite) is unchanged.
//  (b) CommitBindings: descriptor-set identity from a streamed hash of (binding, handle, offset, range) instead of copying
//      every write into a byte vector and hashing that; the overlap assertions run only in mode 0/shadow.
//  tex (S339): (a) only, (b) stays the code as it was.
//  shadow: both paths run, the memo answer is compared with the fresh one (counters bind_memo_*/desc_reuse_*, first 16 mismatches logged).
// No Vulkan or emulator dependency: unit-tested on the CPU (tests/BindMemoTests.cpp).

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace Libs::Graphics::BindMemo {

enum class Mode : uint8_t { Off, On, Shadow, Tex };

inline Mode ParseMode(const char* value) {
	if (value == nullptr) {
		return Mode::Off;
	}
	if (std::strcmp(value, "1") == 0) {
		return Mode::On;
	}
	if (std::strcmp(value, "shadow") == 0) {
		return Mode::Shadow;
	}
	if (std::strcmp(value, "tex") == 0) {
		return Mode::Tex;
	}
	return Mode::Off;
}

// S339 KYTY_BIND_MEMO=tex: only the ResolveTexture memo (a). In S337 the descriptor-set reuse (b) hit 0.003 % and had one mismatch.
// The two call sites read their own view of the mode: `tex` is `1` for ResolveTexture and `0` for CommitBindings.
inline Mode TextureMode(Mode mode) {
	return mode == Mode::Tex ? Mode::On : mode;
}

inline Mode DescriptorMode(Mode mode) {
#if defined(KYTY_BINDMEMO_MUTANT_TEX_DESC)
	return mode == Mode::Tex ? Mode::On : mode;
#else
	return mode == Mode::Tex ? Mode::Off : mode;
#endif
}

// 64-bit streaming hash (xxh64-style rounds over 8-byte words, avalanche at the end).
class Hash {
public:
	void Add(uint64_t word) {
		acc_ += word * kP2;
		acc_ = (acc_ << 31 | acc_ >> 33) * kP1;
		n_++;
	}
	void AddBytes(const void* data, size_t size) {
		const auto* bytes = static_cast<const uint8_t*>(data);
		for (; size >= 8; size -= 8, bytes += 8) {
			uint64_t word;
			std::memcpy(&word, bytes, 8);
			Add(word);
		}
		if (size != 0) {
			uint64_t word = 0;
			std::memcpy(&word, bytes, size);
			Add(word ^ (static_cast<uint64_t>(size) << 56));
		}
	}
	[[nodiscard]] uint64_t Digest() const {
		uint64_t h = acc_ ^ (n_ * kP3);
		h ^= h >> 33;
		h *= kP2;
		h ^= h >> 29;
		h *= kP3;
		h ^= h >> 32;
		return h;
	}

private:
	static constexpr uint64_t kP1 = 0x9E3779B185EBCA87ull;
	static constexpr uint64_t kP2 = 0xC2B2AE3D27D4EB4Full;
	static constexpr uint64_t kP3 = 0x165667B19E3779F9ull;
	uint64_t                  acc_ = 0x27D4EB2F165667C5ull;
	uint64_t                  n_   = 0;
};

// Descriptor-set identity (b): every word that reaches updateDescriptorSets goes through one of these.
class DescIdentity {
public:
	void Header(uint32_t binding, uint32_t array_element, uint32_t count, uint32_t type) {
		h_.Add(static_cast<uint64_t>(binding) << 32 | array_element);
		h_.Add(static_cast<uint64_t>(count) << 32 | type);
	}
	void Buffer(uint64_t handle, uint64_t offset, uint64_t range) {
		h_.Add(handle);
#ifndef KYTY_BINDMEMO_MUTANT_OFFSET_IGNORED
		h_.Add(offset);
#endif
		h_.Add(range);
	}
	void Image(uint64_t sampler, uint64_t view, uint32_t layout) {
		h_.Add(sampler);
		h_.Add(view);
		h_.Add(layout);
	}
	void TexelBuffer(uint64_t view) { h_.Add(view); }
	[[nodiscard]] uint64_t Digest() const { return h_.Digest(); }

private:
	Hash h_;
};

// The previous set of this layout may be bound again only if all of these agree.
[[nodiscard]] inline bool CanReuseSet(bool have_set, uint64_t cached_hash, uint64_t hash, uint64_t cached_pool, uint64_t pool,
                                      uint64_t cached_resources, uint64_t resources) {
	return have_set && cached_hash == hash && cached_pool == pool && cached_resources == resources;
}

// (a) memo key: the 8 T# dwords and everything of the shader's image resource that shapes the result.
struct Key {
	static constexpr size_t kWords = 20;
	std::array<uint32_t, kWords> w {};

	bool operator==(const Key& other) const { return w == other.w; }
};

struct KeyHash {
	size_t operator()(const Key& key) const {
		Hash h;
		h.AddBytes(key.w.data(), sizeof(key.w));
		return static_cast<size_t>(h.Digest());
	}
};

// `flags` / `resource_words` come from the caller (descriptors.cpp) - bools and enums of ImageResource, packed.
[[nodiscard]] inline Key MakeKey(const uint32_t (&dwords)[8], uint32_t flags, const uint32_t* resource_words, size_t resource_count) {
	Key key;
#ifdef KYTY_BINDMEMO_MUTANT_KEY_IGNORES_DWORD
	constexpr size_t dword_count = 7;
#else
	constexpr size_t dword_count = 8;
#endif
	for (size_t i = 0; i < dword_count; i++) {
		key.w[i] = dwords[i];
	}
	key.w[8] = flags;
	for (size_t i = 0; i < resource_count && 9 + i < Key::kWords; i++) {
		key.w[9 + i] = resource_words[i];
	}
	return key;
}

// Bounded table; an entry is a hit only while its generation equals the caller's (L1 passes a constant).
template <typename Value>
class Table {
public:
	explicit Table(size_t capacity = 16384): capacity_(capacity) {}

	[[nodiscard]] const Value* Find(const Key& key, uint64_t generation) {
		const auto it = map_.find(key);
#ifdef KYTY_BINDMEMO_MUTANT_NO_GEN
		(void)generation;
		if (it == map_.end()) {
#else
		if (it == map_.end() || it->second.generation != generation) {
#endif
			misses_++;
			return nullptr;
		}
		hits_++;
		return &it->second.value;
	}
	void Put(const Key& key, uint64_t generation, const Value& value) {
		if (map_.size() >= capacity_ && map_.find(key) == map_.end()) {
			map_.clear();
		}
		map_[key] = Slot {value, generation};
	}
	[[nodiscard]] size_t   Size() const { return map_.size(); }
	[[nodiscard]] uint64_t Hits() const { return hits_; }
	[[nodiscard]] uint64_t Misses() const { return misses_; }

private:
	struct Slot {
		Value    value {};
		uint64_t generation = 0;
	};
	std::unordered_map<Key, Slot, KeyHash> map_;
	size_t                                 capacity_;
	uint64_t                               hits_   = 0;
	uint64_t                               misses_ = 0;
};

} // namespace Libs::Graphics::BindMemo

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDMEMO_H_
