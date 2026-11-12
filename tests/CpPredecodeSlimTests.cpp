// S560 KYTY_CP_PREDECODE_SLIM (texturePredecode.h FillBinding): the binding of a predecoded image built in place in the stage's binding slot equals,
// field by field, what the returned-value path leaves in that slot (copy the slot's description, find on the copy, build the binding, swap the slot's
// mip_views into it, clear them, move it into the slot), over a sequence of draws that reuse one slot: found / not found, a find that changes the
// description (FindImage, the sampled view format), a stencil association (find returns another id), and a previous draw whose binding had an image
// view, mip views and another layout.
// Mutant (must fail, WILL_FAIL): KEEP_STALE (the view and the mip views of the previous draw stay in the slot).

#include "graphics/host_gpu/renderer/pipeline/texturePredecode.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using Libs::Graphics::Predecode::FillBinding;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "CpPredecodeSlimTests: failed: %s\n", what);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

struct Id { // Common::SlotId: explicit bool, default = invalid
	uint32_t index = 0xFFFFFFFFu;
	explicit operator bool() const { return index != 0xFFFFFFFFu; }
	bool operator==(const Id&) const = default;
};

enum class Layout : uint8_t { Undefined, General, ShaderRead };

struct Desc {
	std::array<uint64_t, 16> mips {};
	uint32_t                 format = 0;
	bool operator==(const Desc&) const = default;
};

struct Binding { // TextureBinding
	Id                    image_id;
	const void*           image_view = nullptr;
	Desc                  desc;
	Layout                layout = Layout::Undefined;
	std::vector<uint32_t> mip_views;
	bool operator==(const Binding& o) const {
		return image_id == o.image_id && image_view == o.image_view && desc == o.desc && layout == o.layout && mip_views == o.mip_views;
	}
};

// What FindImage + the sampled view format do to the description, and the id the lookup ends with (a stencil association maps to its depth owner).
struct Case {
	uint32_t found_index;   // 0xFFFFFFFF = not found
	uint32_t format_after;  // the view format FindTextureImage sets (found only)
	uint32_t mip0_after;    // FindImage fills a field of the description
};

Id Find(const Case& c, Desc& desc) {
	desc.mips[0] = c.mip0_after;
	if (c.found_index == 0xFFFFFFFFu) return {};
	desc.format = c.format_after;
	return {c.found_index};
}

// The returned-value path of PrepareBindings (PredecodedTexture -> FindTextureImage, then the mip_views swap and the move into the slot).
void Returned(Binding& slot_binding, const Desc& slot_desc, const Case& c) {
	auto       desc = slot_desc;
	const auto id   = Find(c, desc);
	Binding    binding;
	if (!id) {
		binding = Binding {.desc = desc, .layout = Layout::General};
	} else {
		binding = Binding {id, nullptr, desc};
	}
	binding.mip_views.swap(slot_binding.mip_views);
	binding.mip_views.clear();
	slot_binding = std::move(binding);
}

void Slim(Binding& slot_binding, const Desc& slot_desc, const Case& c) {
	FillBinding(slot_binding, slot_desc, Layout::Undefined, Layout::General, [&](Desc& desc) { return Find(c, desc); });
}

// The previous draw left a binding with a view, mip views and another layout in the slot (RebindImages / CommitBindings fill those after PrepareBindings).
void Dirty(Binding& b, uint32_t seed) {
	b.image_view = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x1000u + seed));
	b.mip_views  = {seed, seed + 1, seed + 2};
	b.layout     = Layout::ShaderRead;
}

} // namespace

int main() {
	const std::array<Case, 5> cases = {{
	    {7, 0x11, 100},           // found
	    {0xFFFFFFFFu, 0, 200},    // not found: the heap retries later
	    {3, 0x22, 300},           // found, another format
	    {9, 0x33, 400},           // stencil association: find ends on the depth owner's id
	    {0xFFFFFFFFu, 0, 500},    // not found after found
	}};
	Binding returned_slot;
	Binding slim_slot;
	for (uint32_t draw = 0; draw < 40; draw++) {
		const auto& c = cases[draw % cases.size()];
		Desc        slot_desc;
		for (uint32_t m = 0; m < slot_desc.mips.size(); m++) slot_desc.mips[m] = draw * 100u + m;
		slot_desc.format = draw;
		Returned(returned_slot, slot_desc, c);
		Slim(slim_slot, slot_desc, c);
		Check(returned_slot == slim_slot, "in-place binding equals the returned one");
		Check(returned_slot.desc.mips[0] == c.mip0_after && returned_slot.desc.mips[1] == draw * 100u + 1u, "the description is the slot's, changed by find");
		Dirty(returned_slot, draw);
		Dirty(slim_slot, draw);
	}
	std::printf("CpPredecodeSlimTests: ok\n");
	return 0;
}
