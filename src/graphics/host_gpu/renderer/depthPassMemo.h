#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_DEPTHPASSMEMO_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_DEPTHPASSMEMO_H_

// S562 KYTY_DEPTH_PASS_MEMO=0|1|shadow: the depth counterpart of KYTY_RT_PASS_MEMO (rtPassMemo.h). ResolveRenderDepthTarget builds the depth
// target's image description from the DB registers (MakeDepthTargetDesc: format policy, host format query, pitch, depth/stencil/HTile footprint,
// UpdateSize) and looks it up (TextureCache::FindImage) on every draw; inside a pass every draw repeats both with the same registers.
//
// Key: the HW::DepthRenderTarget bytes (MakeDepthTargetDesc is a pure function of them and of the device). Value: the description FindImage left
// and a TexMemo entry (image id + registration generation of the lookup). The register checks of ResolveRenderDepthTarget (DepthFatal) and the
// per-draw depth/stencil state still run on every draw; only the description and the lookup are taken from the memo.
// A hit runs, through TextureCache::ReuseImage, what FindImage's `known` branch runs: lock, the image exists, is registered and needs no rebind,
// RegistrationsSince on its pages, tick + LRU touch, MaterializeColorClear (a no-op for depth: HTile is not colour metadata).
// Only results FindImage itself would remember (TexMemo::CanStore) are stored, so the description is the one FindImage would leave (no view adjustment).
//
// Off: unchanged. 1: as above. shadow: always resolves in full; when the memo would have hit, compares id + description (depth_memo_ne_n).
//
// Header only and templated on the register, description and id types, so tests/DepthPassMemoTests.cpp runs the protocol with a fake cache.
// Mutants (tests, WILL_FAIL): KYTY_DEPTHMEMO_MUTANT_{NO_REUSE, STORE_ANY}.

#include "graphics/host_gpu/renderer/pipeline/texResolveMemo.h"
#include "graphics/host_gpu/renderer/rtPassMemo.h"

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::DepthPassMemo {

using Mode = TexMemo::Mode;

[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = TexMemo::ParseMode(std::getenv("KYTY_DEPTH_PASS_MEMO"));
	return mode;
}

template <class Desc, class Id>
struct Value {
	Desc               desc {};
	TexMemo::Entry<Id> entry;
};

template <class Regs, class Desc, class Id>
using Slot = RtPassMemo::Slot<Regs, Value<Desc, Id>>;

struct Probe {
	bool hit       = false; // 1: use the memo's description and id
	bool would_hit = false; // shadow: compare after the full resolve
};

// Before the description is built. hooks.Reuse(entry, desc, commit) -> TexMemo::Verdict (TextureCache::ReuseImage).
template <class Regs, class Desc, class Id, class H>
[[nodiscard]] Probe Lookup(Mode mode, const Slot<Regs, Desc, Id>* slot, const Regs& regs, H& hooks) {
	Probe probe;
	if (mode == Mode::Off || slot == nullptr || !slot->Matches(regs, 0, 0)) {
		return probe;
	}
	const bool commit = mode == Mode::On;
#if defined(KYTY_DEPTHMEMO_MUTANT_NO_REUSE)
	(void)hooks;
	const auto verdict = TexMemo::Verdict::Hit;
#else
	const auto verdict = hooks.Reuse(slot->value.entry, slot->value.desc, commit);
#endif
	if (verdict == TexMemo::Verdict::Hit) {
		probe.hit       = commit;
		probe.would_hit = !commit;
	}
	return probe;
}

struct Counts {
	uint64_t hit  = 0;
	uint64_t miss = 0;
	uint64_t ne   = 0;
};

// After a full resolve (desc, id = FindImage's result, found = what FindImage said). same(a, b): the descriptions agree.
template <class Regs, class Desc, class Id, class Same>
void Finish(Slot<Regs, Desc, Id>* slot, const Probe& probe, const Regs& regs, const Desc& desc, const Id& id, const TexMemo::Found& found,
            uint64_t generation, Same&& same, Counts& counts) {
	if (slot == nullptr) {
		return;
	}
	if (probe.would_hit) {
		if (slot->value.entry.id == id && same(slot->value.desc, desc)) {
			counts.hit++;
		} else {
			counts.ne++;
		}
	} else {
		counts.miss++;
	}
#if defined(KYTY_DEPTHMEMO_MUTANT_STORE_ANY)
	(void)found;
	const bool store = static_cast<bool>(id);
#else
	const bool store = static_cast<bool>(id) && TexMemo::CanStore(found);
#endif
	if (store) {
		slot->Store(regs, 0, 0, Value<Desc, Id> {desc, {id, generation, true}});
	} else {
		slot->Clear();
	}
}

} // namespace Libs::Graphics::DepthPassMemo

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_DEPTHPASSMEMO_H_
