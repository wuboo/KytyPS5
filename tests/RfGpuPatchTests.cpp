// S570 K-rf (KYTY_RF_GPU_PATCH, rfGpuPatch.h): the pure part of the GPU patch of flattened SRT slots.
//   - PatchableLoads: a load is patchable only when nothing but its flat slots uses it (a structural root, any other node's dependency,
//     also the second one, or a folded result make it structural), it really loads guest memory and it is not cyclic;
//   - Decide: only armed, on the CP thread, patchable, not an address base, GPU-dirty and not in flight, with room;
//   - Pending: one entry per address, slots merged, the latest stale value, full at kMaxEntries; Scope arms / disarms / clears;
//   - SnapshotsAgree: only the patched slots may differ;
//   - a model of the two paths (drain: the CPU reads the GPU's value; patch: the walker gets the stale one, the copy writes the GPU's value
//     into the uploaded slot) ends with the same uploaded words.
// Mutants (WILL_FAIL): KYTY_RFPATCH_MUTANT_DIRECT_USES_ONLY, _IGNORE_DIRTY, _IGNORE_ADDRESS_BASE, _NO_DEDUPE.

#include "graphics/shader/recompiler/ir/passes/rfGpuPatch.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

using namespace Libs::Graphics::ShaderRecompiler::IR::RfGpuPatch;

namespace {

int g_failures = 0;
void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "RfGpuPatchTests: failed: %s\n", what);
		++g_failures;
	}
}

GraphNode Load(std::vector<uint32_t> deps = {}) {
	GraphNode n;
	n.deps       = std::move(deps);
	n.read       = true;
	n.emits_load = true;
	return n;
}
GraphNode Op(std::vector<uint32_t> deps) {
	GraphNode n;
	n.deps = std::move(deps);
	return n;
}

void TestPatchable() {
	// 0 load -> flat 0 only: patchable.
	// 1 load -> op 2 -> descriptor root: structural.
	// 3 load -> flat 1 and the address of load 4; 4 -> flat 2: 3 structural, 4 patchable (its address deps do not matter).
	// 5 op(6, 7) -> condition root; 6 load (first dep), 7 load (second dep) also -> flat 3: both structural.
	// 8 load -> flat 4 and an output root: structural.
	// 9 load that does not emit a load (folded to a constant) -> flat 5: not patchable.
	// 10 cyclic load -> flat 6: not patchable.
	// 11 load not stored anywhere and unused: not patchable (no slot to copy into).
	// 12 node folded into load 13 (dep through its result) -> condition root; 13 -> flat 7: structural.
	std::vector<GraphNode> g(14);
	g[0]  = Load();
	g[1]  = Load();
	g[2]  = Op({1});
	g[3]  = Load();
	g[4]  = Load({3});
	g[5]  = Op({6, 7});
	g[6]  = Load();
	g[7]  = Load();
	g[8]  = Load();
	g[9]  = Load();
	g[9].emits_load = false;
	g[10] = Load();
	g[10].cyclic = true;
	g[11] = Load();
	g[12] = Op({13});
	g[13] = Load();
	const std::vector<uint32_t> structural {2, 5, 8, 12};
	const std::vector<uint32_t> flat {0, 3, 4, 7, 8, 9, 10, 13, UINT32_MAX};
	const auto p = PatchableLoads(g, structural, flat);
	Check(p.size() == g.size(), "one answer per node");
	Check(p[0], "a load stored only in a flat slot is patchable");
	Check(!p[1] && !p[2], "a load feeding a descriptor is structural");
	Check(!p[3], "a load whose value is another load's address is structural");
	Check(p[4], "a leaf load is patchable whatever its address comes from");
	Check(!p[6], "the first dependency of a condition is structural");
	Check(!p[7], "the second dependency of a condition is structural (not only the first)");
	Check(!p[8], "a load that is also an output is structural");
	Check(!p[9], "a load that reads nothing is not patchable");
	Check(!p[10], "a cyclic load is not patchable");
	Check(!p[11], "a load without a flat slot is not patchable");
	Check(!p[13], "a load another node folded into is structural");
}

void TestDecide() {
	FaultQuery ok;
	ok.mode       = Mode::On;
	ok.armed      = true;
	ok.gpu_thread = true;
	ok.slots      = 1;
	ok.queried    = true;
	ok.dirty      = true;
	ok.inflight   = false;
	ok.room       = true;
	Check(Decide(ok) == Reject::None, "dirty, patchable, armed: defer");
	auto q = ok;
	q.mode = Mode::Shadow;
	Check(Decide(q) == Reject::None, "shadow defers too");
	q      = ok;
	q.mode = Mode::Off;
	Check(Decide(q) == Reject::NotArmed, "off: drain");
	q       = ok;
	q.armed = false;
	Check(Decide(q) == Reject::NotArmed, "outside a scope: drain");
	q            = ok;
	q.gpu_thread = false;
	Check(Decide(q) == Reject::NotArmed, "another thread: drain");
	q       = ok;
	q.slots = 0;
	Check(Decide(q) == Reject::NotPatchable, "a structural load: drain");
	q              = ok;
	q.address_base = true;
	Check(Decide(q) == Reject::AddressBase, "a global address base: drain (the CPU backs pages from it)");
	q       = ok;
	q.dirty = false;
	Check(Decide(q) == Reject::NotDirty, "clean bytes: the ordinary path");
	q         = ok;
	q.queried = false;
	Check(Decide(q) == Reject::NotDirty, "state unknown (lock busy): drain");
	q          = ok;
	q.inflight = true;
	Check(Decide(q) == Reject::Inflight, "bytes in flight to the backing: drain");
	q      = ok;
	q.room = false;
	Check(Decide(q) == Reject::Full, "no room: drain");
}

void TestPending() {
	auto& p = ThreadPending();
	{
		Scope scope(true);
		Check(p.armed && p.count == 0, "scope arms and clears");
		const uint32_t a[] {3};
		const uint32_t b[] {3, 4};
		Check(p.Add(0x1000, 7, a), "add");
		Check(p.Add(0x1000, 9, b), "same address again");
		Check(p.count == 1, "one entry per address");
		Check(p.entries[0].slot_count == 2 && p.entries[0].slots[0] == 3 && p.entries[0].slots[1] == 4, "slots merged without duplicates");
		Check(p.entries[0].stale == 9, "the latest stale value");
		for (uint64_t i = 1; i < kMaxEntries; i++) {
			const uint32_t s[] {static_cast<uint32_t>(i)};
			Check(p.Add(0x1000 + i * 4, 0, s), "fill");
		}
		Check(!p.Room(), "full");
		const uint32_t s[] {1};
		Check(!p.Add(0x9000, 0, s), "a new address does not fit");
		scope.Disarm();
		Check(!p.armed && p.count == kMaxEntries, "disarm keeps the entries");
	}
	Check(!p.armed, "destructor disarms");
	{
		Scope off(false);
		Check(!p.armed && p.count == kMaxEntries, "an inactive scope changes nothing");
	}
	{
		Scope again(true);
		Check(p.count == 0, "the next scope starts empty");
	}
}

void TestAgree() {
	Pending p;
	const uint32_t s[] {2};
	p.Add(0x100, 5, s);
	const std::vector<uint32_t> a {1, 2, 5, 4};
	std::vector<uint32_t>       b {1, 2, 6, 4};
	Check(SnapshotsAgree(a, b, p), "only the patched slot differs");
	b[0] = 9;
	Check(!SnapshotsAgree(a, b, p), "another slot differs");
	Check(!SnapshotsAgree(a, std::vector<uint32_t> {1, 2, 6}, p), "a size change");
}

// The two paths on a toy: guest memory where the GPU wrote `fresh` and the backing still holds `stale`.
void TestModel() {
	const uint64_t                 addr  = 0x5000;
	const uint32_t                 stale = 0x20, fresh = 0x10;
	std::map<uint64_t, uint32_t>   gpu_buffer {{addr, fresh}};
	std::map<uint64_t, uint32_t>   backing {{addr, stale}};
	const uint32_t                 slot = 1;
	// Drain: the walker reads the drained (fresh) value; the upload carries it.
	std::vector<uint32_t> drained {0xAA, fresh, 0xBB};
	// Patch: the walker got the stale value, the upload carries it, then the copy puts the GPU's value into the slot.
	Pending p;
	const uint32_t s[] {slot};
	FaultQuery     q;
	q.mode       = Mode::On;
	q.armed      = true;
	q.gpu_thread = true;
	q.slots      = 1;
	q.queried    = true;
	q.dirty      = true;
	q.room       = true;
	Check(Decide(q) == Reject::None, "model: the dirty dword defers");
	p.Add(addr, backing[addr], s);
	std::vector<uint32_t> uploaded {0xAA, p.entries[0].stale, 0xBB};
	for (uint32_t i = 0; i < p.count; i++) {
		for (uint32_t k = 0; k < p.entries[i].slot_count; k++) uploaded[p.entries[i].slots[k]] = gpu_buffer[p.entries[i].address];
	}
	Check(uploaded == drained, "model: the patched upload equals the drained one");
	Check(SnapshotsAgree(std::vector<uint32_t> {0xAA, stale, 0xBB}, drained, p), "model: the walks agree beyond the slot");
}

} // namespace

int main() {
	TestPatchable();
	TestDecide();
	TestPending();
	TestAgree();
	TestModel();
	if (g_failures != 0) {
		std::fprintf(stderr, "RfGpuPatchTests: %d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("RfGpuPatchTests: OK");
	return 0;
}
