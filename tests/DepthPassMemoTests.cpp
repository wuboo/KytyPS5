// S562 KYTY_DEPTH_PASS_MEMO (depthPassMemo.h): over a sequence of draws, the depth target (description + image id) that the memo protocol of
// RenderExecutor::ResolveRenderDepthTarget yields in mode 1 equals the full resolve (build the description from the registers, FindImage), and shadow
// counts only agreeing would-be hits. Events: the same registers, other registers, a registration that replaces the image, an inexact result
// (a block-compressed alias left behind, which a later lookup resolves to another image without a registration).
// Mutants (must fail, WILL_FAIL): NO_REUSE (a key hit is used without the ReuseImage check), STORE_ANY (inexact results are stored).

#include "graphics/host_gpu/renderer/depthPassMemo.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

namespace {

using namespace Libs::Graphics;

void Check(bool ok, const char* what, int draw) {
	if (!ok) {
		std::fprintf(stderr, "DepthPassMemoTests: failed: %s (draw %d)\n", what, draw);
		std::exit(1);
	}
}

struct Id {
	uint32_t index = 0xFFFFFFFFu;
	explicit operator bool() const { return index != 0xFFFFFFFFu; }
	bool operator==(const Id&) const = default;
};

struct Regs {
	uint32_t address = 0;
	uint32_t format  = 0;
};

struct Desc {
	uint64_t address = 0;
	uint32_t format  = 0;
	bool operator==(const Desc&) const = default;
};

Desc Build(const Regs& r) { return {r.address * 0x10000ull, r.format * 3u + 1u}; }

struct Cache {
	std::map<uint64_t, uint32_t> images;
	std::map<uint32_t, bool>     registered;
	uint64_t                     generation = 1;
	uint32_t                     next_id    = 0;
	bool                         alias_at_7 = true; // address 7: an alias is left until it can be downloaded

	Cache() { registered[1000] = true; } // the alias' owner

	Id Find(Desc& desc, TexMemo::Found& found, uint64_t& gen) {
		if (desc.address == 7 * 0x10000ull && alias_at_7) {
			found = {.exact = false};
			gen   = generation;
			return Id {1000}; // the alias' owner, not remembered by FindImage
		}
		auto it = images.find(desc.address);
		if (it == images.end()) {
			it = images.emplace(desc.address, next_id++).first;
			registered[it->second] = true;
			generation++;
		}
		found = {.exact = true};
		gen   = generation;
		return Id {it->second};
	}
	void Register(uint64_t address) {
		auto it = images.find(address);
		if (it != images.end()) registered[it->second] = false;
		images[address] = next_id++;
		registered[images[address]] = true;
		generation++;
	}
};

struct Hooks {
	Cache& cache;
	TexMemo::Verdict Reuse(const TexMemo::Entry<Id>& entry, const Desc&, bool) {
		const auto it = cache.registered.find(entry.id.index);
		if (it == cache.registered.end() || !it->second || entry.generation != cache.generation) return TexMemo::Verdict::Stale;
		return TexMemo::Verdict::Hit;
	}
};

struct Result {
	Desc desc;
	Id   id;
};

Result Full(Cache& cache, const Regs& regs) {
	Result         r {Build(regs), {}};
	TexMemo::Found found;
	uint64_t       gen = 0;
	r.id = cache.Find(r.desc, found, gen);
	return r;
}

Result Memo(TexMemo::Mode mode, DepthPassMemo::Slot<Regs, Desc, Id>& slot, Cache& cache, const Regs& regs, DepthPassMemo::Counts& counts) {
	Hooks      hooks {cache};
	const auto probe = DepthPassMemo::Lookup(mode, &slot, regs, hooks);
	Result     r {probe.hit ? slot.value.desc : Build(regs), {}};
	if (probe.hit) {
		r.id = slot.value.entry.id;
		counts.hit++;
		return r;
	}
	TexMemo::Found found;
	uint64_t       gen = 0;
	r.id = cache.Find(r.desc, found, gen);
	DepthPassMemo::Finish(&slot, probe, regs, r.desc, r.id, found, gen, [](const Desc& a, const Desc& b) { return a == b; }, counts);
	return r;
}

struct Draw {
	Regs regs;
	int  event = 0; // 1: the image at the address is replaced; 2: the alias at address 7 becomes downloadable (resolved)
};

} // namespace

int main() {
	const std::vector<Draw> draws = {
	    {{1, 1}}, {{1, 1}}, {{1, 1}}, {{2, 1}}, {{2, 1}}, {{1, 1}}, {{1, 2}}, {{1, 2}},
	    {{1, 2}, 1}, {{1, 2}}, {{1, 2}},                         // a registration replaces the target
	    {{7, 1}}, {{7, 1}}, {{7, 1}, 2}, {{7, 1}}, {{7, 1}},     // an inexact result, then resolved
	    {{1, 1}}, {{1, 1}},
	};
	for (const auto mode: {TexMemo::Mode::On, TexMemo::Mode::Shadow}) {
		Cache                               full_cache;
		Cache                               memo_cache;
		DepthPassMemo::Slot<Regs, Desc, Id> slot;
		DepthPassMemo::Counts               counts;
		int                                 n = 0;
		for (const auto& draw: draws) {
			if (draw.event == 1) {
				full_cache.Register(draw.regs.address * 0x10000ull);
				memo_cache.Register(draw.regs.address * 0x10000ull);
			}
			if (draw.event == 2) {
				full_cache.alias_at_7 = false;
				memo_cache.alias_at_7 = false;
			}
			const auto a = Full(full_cache, draw.regs);
			const auto b = Memo(mode, slot, memo_cache, draw.regs, counts);
			Check(a.desc == b.desc && a.id == b.id, mode == TexMemo::Mode::On ? "memo target differs" : "shadow result differs", n);
			n++;
		}
		Check(counts.ne == 0, "shadow disagreements", n);
		Check(counts.hit >= 8, "too few hits", n);
		std::printf("DepthPassMemoTests: mode %s: %d draws, hit %llu miss %llu ne %llu\n", TexMemo::ModeName(mode), n,
		            static_cast<unsigned long long>(counts.hit), static_cast<unsigned long long>(counts.miss),
		            static_cast<unsigned long long>(counts.ne));
	}
	std::printf("DepthPassMemoTests: OK\n");
	return 0;
}
