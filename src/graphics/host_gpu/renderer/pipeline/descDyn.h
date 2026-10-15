#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCDYN_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCDYN_H_

// S361: KYTY_DESC_DYN=0|1|shadow (default 0 = the code path as it was, bit for bit).
// RebindBuffers uploads `flattened_srt` and `shader_data` to the stream buffer on every draw, so their (buffer, offset) differ for every
// draw and the offset in the descriptor-set key made every key unique (desc_reuse 0.003 %). With 1 those two bindings are
// STORAGE_BUFFER_DYNAMIC: the set holds {stream buffer, 0, range}, the upload offset travels in vkCmdBindDescriptorSets, the key has no offsets,
// and the last kMinSetsPerLayout sets of a layout are kept (LRU) instead of one. shadow = the old path builds the set; the model below runs
// next to it and its content check must never differ (desc_dyn_mismatch_n = 0).
// No Vulkan or emulator dependency: unit-tested on the CPU (tests/DescDynTests.cpp).

#include "graphics/host_gpu/renderer/pipeline/bindMemo.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_set>
#include <utility>
#include <vector>

namespace Libs::Graphics::DescDyn {

enum class Mode : uint8_t { Off, On, Shadow };

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
	return Mode::Off;
}

constexpr size_t kMinSetsPerLayout = 16;

// Words that identify a descriptor set apart from the dynamic offsets: what updateDescriptorSets would be given when every dynamic buffer is
// written as {handle, 0, range}. The same stream feeds the hash (BindMemo::DescIdentity) and, in shadow, the content compared on a hit.
class SetContent {
public:
	void Header(uint32_t binding, uint32_t array_element, uint32_t count, uint32_t type) {
		id_.Header(binding, array_element, count, type);
		words_.push_back(static_cast<uint64_t>(binding) << 32 | array_element);
		words_.push_back(static_cast<uint64_t>(count) << 32 | type);
	}
	// A buffer bound with its own offset in the set (the guest buffers: their window start is packed into shader_data, not a dynamic offset).
	void Buffer(uint64_t handle, uint64_t offset, uint64_t range) {
		id_.Buffer(handle, offset, range);
		words_.push_back(handle);
		words_.push_back(offset);
		words_.push_back(range);
	}
	// A dynamic buffer: the upload offset `dynamic_offset` is NOT part of the set. (The mutant puts it into the key.)
	void DynamicBuffer(uint64_t handle, uint64_t range, uint64_t dynamic_offset) {
#ifdef KYTY_DESCDYN_MUTANT_OFFSET_IN_KEY
		Buffer(handle, dynamic_offset, range);
#else
		(void)dynamic_offset;
		Buffer(handle, 0, range);
#endif
	}
	void Image(uint64_t sampler, uint64_t view, uint32_t layout) {
		id_.Image(sampler, view, layout);
		words_.push_back(sampler);
		words_.push_back(view);
		words_.push_back(layout);
	}
	void TexelBuffer(uint64_t view) {
		id_.TexelBuffer(view);
		words_.push_back(view);
	}
	[[nodiscard]] uint64_t                     Digest() const { return id_.Digest(); }
	[[nodiscard]] const std::vector<uint64_t>& Words() const { return words_; }

private:
	BindMemo::DescIdentity id_;
	std::vector<uint64_t>  words_;
};

// One entry per set: valid only while the pool and the resources it references are the ones it was written against.
template <typename Handle>
struct CachedSet {
	uint64_t              hash                = 0;
	uint64_t              pool_generation     = 0;
	uint64_t              resource_generation = 0;
	Handle                set {};
	std::vector<uint64_t> content; // shadow only
};

// LRU of the sets of one layout. Newest first.
template <typename Handle>
class SetCache {
public:
	explicit SetCache(size_t capacity = kMinSetsPerLayout): capacity_(std::max(capacity, kMinSetsPerLayout)) {}

	// A hit moves the entry to the front.
	[[nodiscard]] const CachedSet<Handle>* Find(uint64_t hash, uint64_t pool_generation, uint64_t resource_generation) {
		for (size_t i = 0; i < entries_.size(); i++) {
			auto& entry = entries_[i];
			if (entry.hash != hash) {
				continue;
			}
#ifndef KYTY_DESCDYN_MUTANT_NO_GEN
			if (entry.pool_generation != pool_generation || entry.resource_generation != resource_generation) {
				return nullptr;
			}
#else
			(void)pool_generation;
			(void)resource_generation;
#endif
			std::rotate(entries_.begin(), entries_.begin() + static_cast<std::ptrdiff_t>(i), entries_.begin() + static_cast<std::ptrdiff_t>(i) + 1);
			return &entries_.front();
		}
		return nullptr;
	}
	void Put(CachedSet<Handle> entry) {
		// A new pool (reset) invalidates every set written from an older one; they would never be found again.
		std::erase_if(entries_, [&](const CachedSet<Handle>& old) {
			return old.hash == entry.hash || old.pool_generation != entry.pool_generation;
		});
		if (entries_.size() >= capacity_) {
			entries_.pop_back();
		}
		entries_.insert(entries_.begin(), std::move(entry));
	}
	[[nodiscard]] size_t Size() const { return entries_.size(); }

private:
	std::vector<CachedSet<Handle>> entries_;
	size_t                         capacity_;
};

// Shadow: the content the cached set was written with against what the old path just built (both without dynamic offsets).
// Returns the number of differing words (a different length counts as differing).
[[nodiscard]] inline size_t ContentMismatch(const std::vector<uint64_t>& cached, const std::vector<uint64_t>& fresh) {
	if (cached.size() != fresh.size()) {
		return std::max<size_t>(1, std::max(cached.size(), fresh.size()) - std::min(cached.size(), fresh.size()));
	}
	size_t diff = 0;
	for (size_t i = 0; i < cached.size(); i++) {
		diff += cached[i] != fresh[i] ? 1u : 0u;
	}
	return diff;
}

// Shadow: the buffer window a reused set + dynamic offset would give against the window the old path bound.
struct BufferWindow {
	uint64_t handle = 0;
	uint64_t offset = 0;
	uint64_t range  = 0;
};
[[nodiscard]] inline bool WindowsEqual(const BufferWindow& cached_base, uint64_t dynamic_offset, const BufferWindow& fresh) {
#ifdef KYTY_DESCDYN_MUTANT_SHADOW_IGNORES_RANGE
	return cached_base.handle == fresh.handle && cached_base.offset + dynamic_offset == fresh.offset;
#else
	return cached_base.handle == fresh.handle && cached_base.offset + dynamic_offset == fresh.offset && cached_base.range == fresh.range;
#endif
}

// Vulkan: dynamic offsets are given in ascending binding order across the whole set.
inline void SortDynamicOffsets(std::vector<std::pair<uint32_t, uint32_t>>& binding_and_offset, std::vector<uint32_t>& offsets) {
	std::sort(binding_and_offset.begin(), binding_and_offset.end(),
	          [](const auto& a, const auto& b) { return a.first < b.first; });
	offsets.clear();
	for (const auto& item: binding_and_offset) {
		offsets.push_back(item.second);
	}
}

// Overlap assertions run once per combination (hash of everything they read), not per draw.
class VerifiedSet {
public:
	explicit VerifiedSet(size_t capacity = 4096): capacity_(capacity) {}
	[[nodiscard]] bool Contains(uint64_t digest) const { return items_.contains(digest); }
	void               Add(uint64_t digest) {
		if (items_.size() >= capacity_) {
			items_.clear();
		}
		items_.insert(digest);
	}
	[[nodiscard]] size_t Size() const { return items_.size(); }

private:
	std::unordered_set<uint64_t> items_;
	size_t                       capacity_;
};

} // namespace Libs::Graphics::DescDyn

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCDYN_H_
