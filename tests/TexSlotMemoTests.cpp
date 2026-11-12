// S562 KYTY_TEX_SLOT_MEMO (texSlotMemo.h): over a sequence of draws that reuse one binding slot, the slot kept or refilled by the memo protocol of
// RenderExecutor::BindPredecodedSlot equals, field by field, the slot the full in-place resolve (Predecode::FillBinding + FindImage) leaves, and the
// colour-metadata clears see the same descriptions. Events: same T#, another T#, a T# differing only past its first dword, another program with the
// same T#, a registration (the image is replaced), the previous draw's view and mip views, the slot reset by another writer, a DCC texture.
// Mutants (must fail, WILL_FAIL): IGNORE_MODULE, IGNORE_SLOT, PREFIX, KEEP_VIEW, STORE_METADATA.

#include "graphics/host_gpu/renderer/pipeline/texSlotMemo.h"
#include "graphics/host_gpu/renderer/pipeline/texturePredecode.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

namespace {

using namespace Libs::Graphics;

void Check(bool ok, const char* what, int draw) {
	if (!ok) {
		std::fprintf(stderr, "TexSlotMemoTests: failed: %s (draw %d)\n", what, draw);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

struct Id {
	uint32_t index = 0xFFFFFFFFu;
	explicit operator bool() const { return index != 0xFFFFFFFFu; }
	bool operator==(const Id&) const = default;
};

enum class Layout : uint8_t { Undefined, General, ShaderRead };

struct Descriptor {
	std::array<uint32_t, 8> fields {};
};

struct Desc {
	uint64_t address    = 0;
	uint32_t format     = 0; // the sampled view format FindTextureImageId sets after FindImage
	uint32_t base_layer = 0;
	bool     metadata   = false;
	bool operator==(const Desc&) const = default;
};

struct Binding {
	Id                    image_id;
	const void*           image_view = nullptr;
	Desc                  desc;
	Layout                layout = Layout::Undefined;
	std::vector<uint32_t> mip_views;
	bool operator==(const Binding& o) const {
		return image_id == o.image_id && image_view == o.image_view && desc == o.desc && layout == o.layout && mip_views == o.mip_views;
	}
};

// The cache: one image per address, replaced by a registration (new id, generation bump); colour clears are logged with the format they see.
struct Cache {
	std::map<uint64_t, uint32_t> images;
	std::map<uint32_t, bool>     registered;
	uint64_t                     generation = 1;
	uint32_t                     next_id    = 0;
	std::vector<uint32_t>        clears;

	Id Find(Desc& desc, TexMemo::Found& found, uint64_t& gen) {
		auto it = images.find(desc.address);
		if (it == images.end()) {
			it = images.emplace(desc.address, next_id++).first;
			registered[it->second] = true;
			generation++;
		}
		if (desc.metadata) clears.push_back(desc.format); // MaterializeColorClear inside FindImage: before the view format
		found = {.exact = true};
		gen   = generation;
		return Id {it->second};
	}
	void Register(uint64_t address) { // the image at the address is replaced
		auto it = images.find(address);
		if (it != images.end()) registered[it->second] = false;
		images[address] = next_id++;
		registered[images[address]] = true;
		generation++;
	}
	TexMemo::Verdict Reuse(const TexMemo::Entry<Id>& entry, const Desc& desc) {
		if (!registered[entry.id.index] || entry.generation != generation) return TexMemo::Verdict::Stale;
		if (desc.metadata) clears.push_back(desc.format);
		return TexMemo::Verdict::Hit;
	}
};

// The worker's description of a T# for a program (a pure function of both).
Desc Describe(const void* /*module*/, const Descriptor& d) {
	return {.address = d.fields[0] * 0x10000ull, .format = 0, .base_layer = 0, .metadata = (d.fields[1] & 1u) != 0};
}
uint32_t ViewFormat(const void* module, const Descriptor& d) {
	return 3u + d.fields[7] * 10u + static_cast<uint32_t>(reinterpret_cast<uintptr_t>(module) & 0xFu);
}

void FullResolve(Binding& out, Cache& cache, const void* module, const Descriptor& d, TexMemo::Found& found, Id& found_id, uint64_t& gen) {
	Predecode::FillBinding(out, Describe(module, d), Layout::Undefined, Layout::General, [&](Desc& desc) {
		found_id = cache.Find(desc, found, gen);
		desc.format = ViewFormat(module, d);
		return found_id;
	});
}

// RenderExecutor::BindPredecodedSlot with the fakes.
bool MemoBind(Binding& out, TexSlotMemo::Slot<Descriptor, Id>& memo, Cache& cache, const void* module, const Descriptor& d) {
	if (memo.Matches(module, d, false, out.image_id) && cache.Reuse(memo.entry, out.desc) == TexMemo::Verdict::Hit) {
		TexSlotMemo::KeepBinding(out, Layout::Undefined);
		return true;
	}
	TexMemo::Found found;
	Id             found_id;
	uint64_t       gen = 0;
	FullResolve(out, cache, module, d, found, found_id, gen);
	TexSlotMemo::Store store {.found = found, .same_id = found_id == out.image_id, .depth = false, .color_metadata = out.desc.metadata, .texture = true};
	if (out.image_id && TexSlotMemo::CanStore(store)) {
		memo.Set(module, d, false, out.image_id, 0, {found_id, gen, true});
	} else {
		memo.Clear();
	}
	return false;
}

struct Draw {
	int          module;   // 0 / 1
	uint32_t     tsharp;   // fields[0]
	uint32_t     tail = 0; // fields[7]
	bool         dcc  = false;
	int          event = 0; // 1: register (replace) the image of the T# before the draw; 2: another writer resets the slot
};

} // namespace

int main() {
	static int modules[2];
	const std::vector<Draw> draws = {
	    {0, 1}, {0, 1}, {0, 1}, {0, 2}, {0, 2}, {0, 1},         // repeats and a switch
	    {0, 1, 5}, {0, 1, 5}, {0, 1, 0},                        // a T# differing past its first dword
	    {1, 1}, {1, 1}, {0, 1},                                 // another program with the same T#
	    {0, 1, 0, false, 1}, {0, 1}, {0, 1},                    // a registration replaces the image
	    {0, 3, 0, false, 2}, {0, 3}, {0, 3, 0, false, 2}, {0, 3}, // another writer resets the slot
	    {0, 4, 0, true}, {0, 4, 0, true}, {0, 4, 0, true},      // a DCC texture
	    {0, 1}, {0, 1},
	};
	Cache                              cache_full;
	Cache                              cache_memo;
	Binding                            full;
	Binding                            memo_slot;
	TexSlotMemo::Slot<Descriptor, Id>  memo;
	int                                n = 0;
	int                                kept = 0;
	for (const auto& draw: draws) {
		const void* module = &modules[draw.module];
		Descriptor  d;
		d.fields[0] = draw.tsharp;
		d.fields[1] = draw.dcc ? 1u : 0u;
		d.fields[7] = draw.tail;
		if (draw.event == 1) {
			cache_full.Register(draw.tsharp * 0x10000ull);
			cache_memo.Register(draw.tsharp * 0x10000ull);
		}
		if (draw.event == 2) {
			full      = {};
			memo_slot = {};
		}
		TexMemo::Found found;
		Id             found_id;
		uint64_t       gen = 0;
		FullResolve(full, cache_full, module, d, found, found_id, gen);
		kept += MemoBind(memo_slot, memo, cache_memo, module, d) ? 1 : 0;
		Check(memo_slot == full, "slot differs from the full resolve", n);
		Check(cache_memo.clears == cache_full.clears, "colour clears saw other descriptions", n);
		// RebindImages: the views of this draw stay in the slot until the next draw.
		full.image_view      = &modules[0];
		memo_slot.image_view = &modules[0];
		full.mip_views       = {7, 8};
		memo_slot.mip_views  = {7, 8};
		n++;
	}
	Check(kept >= 8, "the memo kept too few slots", n);
	std::printf("TexSlotMemoTests: %d draws, %d kept, OK\n", n, kept);
	return 0;
}
