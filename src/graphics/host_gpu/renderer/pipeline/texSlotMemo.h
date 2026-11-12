#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXSLOTMEMO_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXSLOTMEMO_H_

// S562 KYTY_TEX_SLOT_MEMO=0|1|shadow (needs KYTY_CP_PREDECODE=1): a draw stage's binding slot (PreparedBindings::images[i]) that already holds the
// binding of the same program image with the same T# is kept as it is, instead of copying the worker's ~1.2 KB description into it again and looking
// it up (TextureCache::FindImage + FindTextureImageId). Consecutive draws bind the same textures in most slots (S463: ~75 % of stage binds equal the
// previous one with the same texture-cache generation).
//
// Key: the stage's program (ShaderStageRuntime::module, so the slot index names the same ImageResource), the decoded T# (ShaderTextureResource bytes,
// from which the worker's description is a pure function), shader_conversion (FindImage's exact_format), and the id the slot holds now (another path
// that wrote the slot leaves another id or clears the memo).
// Value: the binding left in the slot (its description and id, after FindImage and the sampled view format) and a TexMemo entry (FindImage's result +
// registration generation of the lookup), checked with TextureCache::ReuseImage: the `known` branch of FindImage (lock, the image exists, is
// registered and needs no rebind, RegistrationsSince on its pages, tick + LRU touch, MaterializeColorClear).
// Only stored (CanStore): results FindImage itself would remember (TexMemo::CanStore: exact, no view adjustment, no null/video-out image); no colour
// metadata (DCC/CMASK: MaterializeColorClear would need the description before the sampled view format; without metadata it returns at once); no
// depth image and no stencil association (their validation reads the image; the redirect reads it live).
// A hit leaves what FillBinding leaves for a found image: id and description kept, no image view, mip_views emptied, layout `found`.
//
// Off: unchanged. 1: as above (the miss path is the in-place fill of KYTY_CP_PREDECODE_SLIM). shadow: always resolves in full; when the memo would
// have hit, compares id + description with the full result (tex_slot_memo_ne_n).
//
// Header only and templated on the descriptor, id, binding and layout types, so tests/TexSlotMemoTests.cpp pins it with fakes.
// Mutants (tests, WILL_FAIL): KYTY_TEXSLOT_MUTANT_{IGNORE_MODULE, IGNORE_SLOT, PREFIX, KEEP_VIEW, STORE_METADATA}.

#include "graphics/host_gpu/renderer/pipeline/texResolveMemo.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>

namespace Libs::Graphics::TexSlotMemo {

using Mode = TexMemo::Mode;

[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = TexMemo::ParseMode(std::getenv("KYTY_TEX_SLOT_MEMO"));
	return mode;
}

// What the full resolve says about a result, beyond TexMemo::Found.
struct Store {
	TexMemo::Found found;
	bool           same_id        = false; // FindImage's id is the one bound (no stencil association redirect)
	bool           depth          = false; // the bound image is a depth image
	bool           color_metadata = false; // the description carries DCC/CMASK metadata
	bool           texture        = false; // a sampled texture binding (not storage)
};

[[nodiscard]] inline bool CanStore(const Store& s) {
#if defined(KYTY_TEXSLOT_MUTANT_STORE_METADATA)
	return TexMemo::CanStore(s.found) && s.same_id && !s.depth && s.texture;
#else
	return TexMemo::CanStore(s.found) && s.same_id && !s.depth && !s.color_metadata && s.texture;
#endif
}

template <class Descriptor, class Id>
struct Slot {
	static_assert(std::is_trivially_copyable_v<Descriptor>);

	const void*        module = nullptr;
	Descriptor         descriptor {};
	bool               shader_conversion = false;
	Id                 bound {};      // the id the slot was left with
	uint32_t           base_layer = 0; // the description's base layer before FindImage (MaterializeColorClear's metadata layer)
	TexMemo::Entry<Id> entry;          // FindImage's result + generation

	[[nodiscard]] bool Matches(const void* now_module, const Descriptor& now, bool now_conversion, const Id& slot_id) const {
		if (!entry.valid || shader_conversion != now_conversion) return false;
#if !defined(KYTY_TEXSLOT_MUTANT_IGNORE_MODULE)
		if (module != now_module) return false;
#else
		(void)now_module;
#endif
#if !defined(KYTY_TEXSLOT_MUTANT_IGNORE_SLOT)
		if (!(bound == slot_id)) return false;
#else
		(void)slot_id;
#endif
#if defined(KYTY_TEXSLOT_MUTANT_PREFIX)
		return std::memcmp(&descriptor, &now, sizeof(uint32_t)) == 0;
#else
		return std::memcmp(&descriptor, &now, sizeof(Descriptor)) == 0;
#endif
	}
	void Set(const void* now_module, const Descriptor& now, bool now_conversion, const Id& slot_id, uint32_t now_base_layer,
	         const TexMemo::Entry<Id>& now_entry) {
		module = now_module;
		std::memcpy(&descriptor, &now, sizeof(Descriptor));
		shader_conversion = now_conversion;
		bound             = slot_id;
		base_layer        = now_base_layer;
		entry             = now_entry;
	}
	void Clear() { entry.valid = false; }
};

// The slot after a hit: what Predecode::FillBinding leaves for a found image, the description and the id kept.
template <class Binding, class Layout>
void KeepBinding(Binding& out, Layout found) {
#if !defined(KYTY_TEXSLOT_MUTANT_KEEP_VIEW)
	out.image_view = decltype(out.image_view) {};
	out.mip_views.clear();
#endif
	out.layout = found;
}

// Per PrepareBindings call, added to the FrameTiming columns once.
struct Counts {
	uint64_t hit   = 0; // kept (shadow: would have been, and agreed)
	uint64_t miss  = 0; // eligible slots resolved in full (no entry, other key, or the entry was stale)
	uint64_t stale = 0; // the key matched but ReuseImage said Stale
	uint64_t ne    = 0; // shadow disagreements
};

} // namespace Libs::Graphics::TexSlotMemo

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXSLOTMEMO_H_
