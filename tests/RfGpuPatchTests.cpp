// S570 K-rf (KYTY_RF_GPU_PATCH, rfGpuPatch.h): the pure part of the GPU patch of flattened SRT slots.
//   - PatchableLoads: a load is patchable only when nothing but its flat slots uses it (a structural root, any other node's dependency,
//     also the second one, or a folded result make it structural), it really loads guest memory and it is not cyclic;
//   - Decide: only armed, on the CP thread, patchable, not an address base, GPU-dirty and not in flight, with room;
//   - Pending: one entry per address, slots merged, the latest stale value, full at kMaxEntries; Scope arms / disarms / clears;
//   - SnapshotsAgree: only the patched slots may differ;
//   - a model of the two paths (drain: the CPU reads the GPU's value; patch: the walker gets the stale one, the copy writes the GPU's value
//     into the uploaded slot) ends with the same uploaded words.
// Mutants (WILL_FAIL): KYTY_RFPATCH_MUTANT_DIRECT_USES_ONLY, _IGNORE_DIRTY, _IGNORE_ADDRESS_BASE, _NO_DEDUPE.
// S579 K-rf v2 (superset): a model of the SRT walker (three-valued nodes: known / unknown / deferred, monotonic reachability to a fixpoint, an
// unknown condition reaches SuccessorsWhenUnknown, a Phi over reached predecessors, flat slots and resources of reached blocks, a failed walk
// retried with the drain) driven by the production pieces (SupersetLoads, DecideSuperset, DeferredStateByte, SupersetEntrySlots, Pending,
// SupersetRetry, SupersetCover*). Plans: K2 simple (4928...), K2 through a control Phi with dead users (c5ee...), a load in an address, a
// load in a loop, K2 with its own flat slot (GPU copy), derived flat, descriptor, control-Phi address (K1), a failing superset walk, and 3000
// random plans: the superset (after the GPU copy) always holds what the synchronous walk does.
// Mutants (WILL_FAIL): _SKIP_SUCCESSOR, _COND_AS_TAKEN, _NO_FLAT_COPY, _DEFER_ADDRESS, _DEFER_LOOP, _COVER_MISSES_ONE, _CPHI_ADDRESS.
// S581 K-rf v3 (KYTY_KRF3): ParseKrf3; DecideSuperset defers dirty bytes with an older download in flight only with inflight_ok; the guest page
// guard (GuestPageGuard) on models of the S580 runs: the desert (the guest reads one page of the deferred dwords whenever it is left dirty) bans
// that page within a window and keeps the other page deferring, trials stay rare; the garden (a burst of guest reads at the level entry, then
// ~2 % of flips) defers again long before F10+330; a scene change (the guest stops reading) gets the page back; inflight only counts would-bans;
// eviction keeps banned pages; guest threads hammering NoteGuestRead while the CP defers.
// Mutants (WILL_FAIL): _KRF3_INFLIGHT_DRAINS, _KRF3_NO_GUEST_HITS, _KRF3_NO_EXPIRY, _KRF3_ADDRESS_KEY.

#include "graphics/shader/recompiler/ir/passes/rfGpuPatch.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <thread>
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

	// S573 DIAG: Explain agrees with PatchableLoads and names the reason.
	const auto e = Explain(g, structural, flat);
	Check(e.size() == g.size(), "Explain: one entry per node");
	for (size_t i = 0; i < g.size(); ++i) {
		Check((e[i].why == Why::Patchable) == p[i], "Explain: patchable exactly when PatchableLoads says so");
	}
	Check(e[1].why == Why::Node && e[1].users.size() == 1 && e[1].users[0] == 2, "Explain: load 1 is used by op 2");
	Check(e[2].why == Why::NotLoad && e[2].roots.size() == 1 && structural[e[2].roots[0]] == 2, "Explain: op 2 is a root");
	Check(e[3].why == Why::Node && e[3].flat_slots == 1, "Explain: load 3 is another load's address and has a flat slot");
	Check(e[7].why == Why::Node && e[7].users.size() == 1 && e[7].users[0] == 5, "Explain: the second dependency is a use");
	Check(e[8].why == Why::Root, "Explain: an output root");
	Check(e[9].why == Why::NoLoad, "Explain: a folded load");
	Check(e[10].why == Why::Cyclic, "Explain: a cyclic load");
	Check(e[11].why == Why::NotStored, "Explain: a load without a flat slot");
	Check(e[13].why == Why::Node, "Explain: a load another node folded into");
	// Too many slots: patchable for PatchableLoads, but the walker compiler drops it (kMaxSlots).
	std::vector<GraphNode> g2(1);
	g2[0] = Load();
	const std::vector<uint32_t> none;
	const std::vector<uint32_t> five(kMaxSlots + 1, 0u);
	Check(Explain(g2, none, five)[0].why == Why::TooManySlots, "Explain: more flat slots than an entry holds");
	Check(Explain(g2, none, std::vector<uint32_t>(kMaxSlots, 0u))[0].why == Why::Patchable, "Explain: kMaxSlots slots are fine");
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
	// S573 Oracle (DIAG only): patchability, address base and room do not matter; armed, CP thread and the bytes' state do.
	auto o = ok;
	o.mode = Mode::Oracle;
	o.slots = 0;
	o.address_base = true;
	o.room = false;
	Check(Decide(o) == Reject::None, "oracle: a structural load on dirty bytes is completed");
	auto o2 = o;
	o2.dirty = false;
	Check(Decide(o2) == Reject::NotDirty, "oracle: clean bytes take the ordinary path");
	o2 = o;
	o2.inflight = true;
	Check(Decide(o2) == Reject::Inflight, "oracle: bytes in flight drain");
	o2 = o;
	o2.armed = false;
	Check(Decide(o2) == Reject::NotArmed, "oracle: only inside DISPATCH_INDIRECT");
	Check(ParseMode("oracle") == Mode::Oracle && ParseMode("0") == Mode::Off, "oracle: parsed");
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


// ==== S579 K-rf v2: the walker model ===========================================================================================================
namespace krf2 {

constexpr uint32_t kNone = UINT32_MAX;

enum class Op : uint8_t { Const, User, Load, Add, Ule, Not, Phi };

struct MNode {
	Op                                        op    = Op::Const;
	uint32_t                                  a     = kNone; // operands (Load: its address operand, dword index added to base)
	uint32_t                                  b     = kNone;
	uint32_t                                  value = 0;     // Const value, User index
	uint64_t                                  base  = 0;     // Load
	std::vector<std::pair<uint32_t, uint32_t>> phi;           // (predecessor block, node)
	bool                                      cyclic = false; // as the walker compiler's node.referenced
};

struct MBlock {
	uint32_t              cond = kNone; // kNone: every successor
	std::vector<uint32_t> succ;
	std::vector<uint32_t> resources;
	std::vector<uint32_t> slots;
};

struct MPlan {
	std::vector<MNode>    nodes;
	std::vector<MBlock>   blocks;
	std::vector<uint32_t> resources; // descriptor node per resource; a resource no block lists is always active
	std::vector<uint32_t> slots;     // node per flat slot
	std::vector<uint32_t> user;
};

struct Cell {
	uint32_t stale = 0, fresh = 0;
	bool     dirty = false;
};
using Memory = std::map<uint64_t, Cell>;

struct MDesc {
	std::array<uint32_t, 8> dwords {};
	uint32_t                dword_count = 1;
	bool                    operator==(const MDesc&) const = default;
};

struct Result {
	bool                  ok = false;
	std::vector<bool>     reach;
	std::vector<MDesc>    resources;
	std::vector<uint32_t> flat;     // what the CPU uploads
	std::vector<uint32_t> uploaded; // after the GPU copies of the pending entries
	uint32_t              defers = 0, drains = 0, walkfail = 0, retry = 0;
	std::map<SupersetWhy, uint32_t> fallback;
};

std::vector<GraphNode> Graph(const MPlan& p) {
	std::vector<GraphNode> g(p.nodes.size());
	for (size_t i = 0; i < p.nodes.size(); ++i) {
		const auto& n = p.nodes[i];
		auto&       x = g[i];
		if (n.a != kNone) x.deps.push_back(n.a);
		if (n.b != kNone) x.deps.push_back(n.b);
		for (const auto& [block, node]: n.phi) x.deps.push_back(node);
		x.read        = n.op == Op::Load;
		x.emits_load  = x.read;
		x.cyclic      = n.cyclic;
		x.control_phi = n.op == Op::Phi;
	}
	return g;
}

std::vector<SupersetWhy> Classify(const MPlan& p) {
	std::vector<Root> roots;
	for (const auto& b: p.blocks) {
		if (b.cond != kNone) roots.push_back({b.cond, RootKind::Condition});
	}
	for (const auto r: p.resources) roots.push_back({r, RootKind::Descriptor});
	return SupersetLoads(Graph(p), roots, p.slots);
}

std::vector<uint32_t> SlotsOf(const MPlan& p, uint32_t node) {
	std::vector<uint32_t> out;
	for (uint32_t s = 0; s < p.slots.size(); ++s) {
		if (p.slots[s] == node) out.push_back(s);
	}
	return out;
}

class Walker {
public:
	Walker(const MPlan& plan, Memory& memory, bool superset): m_p(plan), m_mem(memory), m_superset(superset), m_why(Classify(plan)) {}

	Result Run() {
		Result r = Once();
		if (!r.ok && m_superset && SupersetRetry()) {
			const auto defers = r.defers;
			const auto fall   = r.fallback;
			r                 = Once();
			r.walkfail        = 1;
			r.defers += defers;
			for (const auto& [k, v]: fall) r.fallback[k] += v;
		}
		r.uploaded = r.flat;
		const auto& pending = ThreadPending();
		for (uint32_t i = 0; i < pending.count; ++i) {
			const auto& e = pending.entries[i];
			for (uint32_t k = 0; k < e.slot_count; ++k) {
				if (e.slots[k] < r.uploaded.size()) r.uploaded[e.slots[k]] = m_mem[e.address].fresh; // the GPU copy: the buffer's real bytes
			}
		}
		return r;
	}

private:
	Result Once() {
		Result r;
		m_r = &r;
		r.reach.assign(m_p.blocks.size(), false);
		if (!r.reach.empty()) r.reach[0] = true;
		for (bool changed = true; changed;) {
			changed = false;
			Reset();
			for (uint32_t b = 0; b < m_p.blocks.size(); ++b) {
				if (!r.reach[b]) continue;
				const auto& block = m_p.blocks[b];
				const auto  mark  = [&](uint32_t s) {
                    if (s < r.reach.size() && !r.reach[s]) r.reach[s] = changed = true;
				};
				if (block.cond != kNone) {
					Eval(block.cond);
					if (m_state[block.cond] == kKnownState) {
						mark(block.succ[m_value[block.cond] != 0 ? 0 : 1]);
						continue;
					}
				}
				for (const auto s: SuccessorsWhenUnknown(std::span<const uint32_t>(block.succ))) mark(s);
			}
		}
		r.ok = true;
		r.flat.assign(m_p.slots.size(), 0u);
		std::vector<bool> slot_reached(m_p.slots.size(), false);
		for (uint32_t b = 0; b < m_p.blocks.size(); ++b) {
			if (!r.reach[b]) continue;
			for (const auto s: m_p.blocks[b].slots) slot_reached[s] = true;
		}
		for (uint32_t s = 0; s < m_p.slots.size(); ++s) {
			if (!slot_reached[s]) continue;
			const auto node = m_p.slots[s];
			Eval(node);
			const bool direct_deferred = m_state[node] == DeferredStateByte() && m_why[node] == SupersetWhy::Superset;
			if (m_state[node] != kKnownState && !direct_deferred) {
				r.ok = false;
				continue;
			}
			r.flat[s] = m_value[node];
		}
		r.resources.assign(m_p.resources.size(), MDesc {});
		for (uint32_t res = 0; res < m_p.resources.size(); ++res) {
			bool listed = false, active = false;
			for (uint32_t b = 0; b < m_p.blocks.size(); ++b) {
				const auto& list = m_p.blocks[b].resources;
				if (std::find(list.begin(), list.end(), res) == list.end()) continue;
				listed = true;
				active |= r.reach[b];
			}
			if (listed && !active) continue;
			const auto node = m_p.resources[res];
			Eval(node);
			if (m_state[node] != kKnownState) {
				r.ok = false;
				continue;
			}
			r.resources[res].dwords[0] = m_value[node];
		}
		return r;
	}

	void Reset() {
		m_state.assign(m_p.nodes.size(), 0);
		m_value.assign(m_p.nodes.size(), 0);
	}

	bool Known(uint32_t i) {
		Eval(i);
		return m_state[i] == kKnownState;
	}

	void Eval(uint32_t i) {
		if (m_state[i] != 0) return;
		m_state[i] = 1;
		const auto& n = m_p.nodes[i];
		const auto  known = [&](uint32_t v) {
            m_value[i] = v;
            m_state[i] = kKnownState;
		};
		switch (n.op) {
			case Op::Const: known(n.value); return;
			case Op::User: known(m_p.user[n.value]); return;
			case Op::Add:
				if (Known(n.a) && Known(n.b)) known(m_value[n.a] + m_value[n.b]);
				return;
			case Op::Not:
				if (Known(n.a)) known(m_value[n.a] == 0 ? 1u : 0u);
				return;
			case Op::Ule:
				// As the walker: a zero bound decides alone (0 <= x).
				if (Known(n.a) && m_value[n.a] == 0) {
					known(1u);
					return;
				}
				if (Known(n.a) && Known(n.b)) known(m_value[n.a] <= m_value[n.b] ? 1u : 0u);
				return;
			case Op::Phi: {
				bool     any = false;
				uint32_t v   = 0;
				for (const auto& [block, node]: n.phi) {
					if (!m_r->reach[block]) continue;
					if (!Known(node)) return;
					if (any && m_value[node] != v) return;
					any = true;
					v   = m_value[node];
				}
				if (any) known(v);
				return;
			}
			case Op::Load: {
				uint64_t address = n.base;
				if (n.a != kNone) {
					if (!Known(n.a)) return;
					address += uint64_t {m_value[n.a]} * 4u;
				}
				auto& cell = m_mem[address];
				if (cell.dirty) {
					// The exception handler (runtimeLinker.cpp TrySupersetDefer).
					auto&            pending = ThreadPending();
					SupersetQuery    q;
					const auto       slots = SlotsOf(m_p, i);
					q.armed                = m_superset && pending.armed;
					q.gpu_thread           = true;
					q.site                 = true;
					q.why                  = m_why[i];
					q.slots                = static_cast<uint32_t>(slots.size());
					q.queried              = true;
					q.dirty                = true;
					q.room                 = pending.Room();
					q.disabled             = pending.superset_off;
					const auto decision    = DecideSuperset(q);
					if (decision == SupersetReject::None &&
					    pending.Add(address, cell.stale, SupersetEntrySlots(std::span<const uint32_t>(slots)))) {
						++pending.superset_defers;
						++m_r->defers;
						m_value[i] = cell.stale;
						m_state[i] = DeferredStateByte();
						return;
					}
					if (decision == SupersetReject::Static) ++m_r->fallback[q.why];
					if (decision == SupersetReject::Retry) ++m_r->retry;
					// The drain: the whole queue, every GPU-written dword lands in the backing.
					++m_r->drains;
					for (auto& [a, c]: m_mem) {
						if (c.dirty) {
							c.stale = c.fresh;
							c.dirty = false;
						}
					}
				}
				known(cell.dirty ? cell.stale : cell.fresh);
				return;
			}
		}
	}

	const MPlan&             m_p;
	Memory&                  m_mem;
	bool                     m_superset;
	std::vector<SupersetWhy> m_why;
	std::vector<uint8_t>     m_state;
	std::vector<uint32_t>    m_value;
	Result*                  m_r = nullptr;
};

// The synchronous walk (every fault drains) and the superset walk on the same memory; each in its own Scope.
struct Pair {
	Result                  sync, superset;
	Pending                 pending; // the superset's entries
	std::vector<uint32_t>   fresh;   // per entry, the drained dword
};

Pair RunBoth(const MPlan& plan, const Memory& memory) {
	Pair   out;
	Memory a = memory, b = memory;
	{
		Scope scope(true);
		out.sync = Walker(plan, a, false).Run();
	}
	{
		Scope scope(true);
		out.superset = Walker(plan, b, true).Run();
		out.pending  = ThreadPending();
	}
	for (uint32_t i = 0; i < out.pending.count; ++i) out.fresh.push_back(memory.at(out.pending.entries[i].address).fresh);
	return out;
}

// The superset holds every resource and flat word of the synchronous walk, after the GPU copy; reachability only grows.
bool Covers(const Pair& r, uint32_t* extra = nullptr) {
	if (!r.sync.ok || !r.superset.ok) return false;
	for (size_t b = 0; b < r.sync.reach.size(); ++b) {
		if (r.sync.reach[b] && !r.superset.reach[b]) return false;
	}
	const auto res  = SupersetCoverResources(std::span<const MDesc>(r.sync.resources), std::span<const MDesc>(r.superset.resources));
	const auto flat = SupersetCoverFlat(r.sync.flat, r.superset.flat, r.pending, r.fresh);
	if (extra != nullptr) *extra = res.extra;
	bool words = true;
	for (size_t s = 0; s < r.sync.flat.size(); ++s) {
		if (r.sync.flat[s] != 0 && r.superset.uploaded[s] != r.sync.flat[s]) words = false;
	}
	for (size_t i = 0; i < r.sync.resources.size(); ++i) {
		if (r.sync.resources[i].dwords[0] != 0 && !(r.superset.resources[i] == r.sync.resources[i])) words = false;
	}
	return res.agree && flat.agree && words;
}

uint32_t Add(MPlan& p, MNode n) {
	p.nodes.push_back(std::move(n));
	return static_cast<uint32_t>(p.nodes.size() - 1);
}
MNode Const(uint32_t v) {
	return {.op = Op::Const, .value = v};
}
MNode User(uint32_t k) {
	return {.op = Op::User, .value = k};
}
MNode LoadAt(uint64_t base, uint32_t address = kNone) {
	return {.op = Op::Load, .a = address, .base = base};
}
MNode Ule(uint32_t a, uint32_t b) {
	return {.op = Op::Ule, .a = a, .b = b};
}
MNode Not(uint32_t a) {
	return {.op = Op::Not, .a = a};
}
MNode Sum(uint32_t a, uint32_t b) {
	return {.op = Op::Add, .a = a, .b = b};
}

constexpr uint64_t kA = 0x57954027c, kB = 0x5794bf8fc;

// 4928273de21f1aa0: N2 ReadConstBuffer@+0x1c -> N1 ULessThanEqual32 -> N0 LogicalNot -> cond1. A: stale 0x20, fresh 4 (S574).
// Block 0: cond, true -> 1 (resource 1), false -> 2 (resource 2); both -> 3. Resource 0 is always active.
MPlan PlanK2Simple(uint32_t* load) {
	MPlan p;
	p.user          = {8};
	const auto u    = Add(p, User(0));
	*load           = Add(p, LoadAt(kA));
	const auto ule  = Add(p, Ule(*load, u));
	const auto cond = Add(p, Not(ule));
	p.resources     = {Add(p, Const(0x11)), Add(p, Const(0x22)), Add(p, Const(0x33))};
	p.blocks        = {{.cond = cond, .succ = {1, 2}}, {.succ = {3}, .resources = {1}}, {.succ = {3}, .resources = {2}}, {}};
	return p;
}

void TestK2Simple() {
	uint32_t   load = 0;
	const auto plan = PlanK2Simple(&load);
	Check(Classify(plan)[load] == SupersetWhy::Superset, "krf2 K2 simple: the condition-only load is a superset load");
	const Memory mem {{kA, {.stale = 0x20, .fresh = 4, .dirty = true}}};
	const auto   r = RunBoth(plan, mem);
	uint32_t     extra = 0;
	Check(r.sync.drains == 1 && r.sync.defers == 0, "krf2 K2 simple: the synchronous walk drains");
	Check(r.superset.defers >= 1 && r.superset.drains == 0, "krf2 K2 simple: the superset walk does not drain");
	Check(r.superset.reach[1] && r.superset.reach[2], "krf2 K2 simple: both successors reached");
	Check(!r.sync.reach[1] && r.sync.reach[2], "krf2 K2 simple: the true value (4 <= 8) takes the false successor");
	Check(Covers(r, &extra), "krf2 K2 simple: the superset covers the synchronous walk");
	Check(extra == 1, "krf2 K2 simple: one extra resource (the true successor's)");
	Check(r.pending.count == 1 && r.pending.entries[0].slot_count == 0, "krf2 K2 simple: one entry, nothing to copy");
	// A clean load is not a fault: the superset changes nothing.
	const auto clean = RunBoth(plan, Memory {{kA, {.stale = 4, .fresh = 4, .dirty = false}}});
	Check(clean.superset.defers == 0 && clean.superset.reach == clean.sync.reach && Covers(clean), "krf2 K2 simple: clean bytes, same walk");
}

// c5ee521b8849cf83: N10 ReadConstBuffer@+0x1c -> N9 Phi -> N8 ULessThanEqual32 -> N7 LogicalNot -> cond3, and dead users of N10 (a Phi and
// a ReadConstBuffer whose address uses it, that nothing reads).
MPlan PlanK2Phi(uint32_t* load) {
	MPlan p;
	p.user           = {0, 8};
	const auto sel   = Add(p, User(0));
	const auto c0    = Add(p, Not(sel)); // user 0 = 0: true -> block 1
	*load            = Add(p, LoadAt(kA));
	const auto other = Add(p, Const(0x100));
	MNode      phi {.op = Op::Phi};
	phi.phi          = {{1, *load}, {2, other}};
	const auto n9    = Add(p, phi);
	const auto n8    = Add(p, Ule(n9, Add(p, User(1))));
	const auto n7    = Add(p, Not(n8));
	MNode      dead_phi {.op = Op::Phi};
	dead_phi.phi     = {{1, *load}};
	(void)Add(p, dead_phi);
	(void)Add(p, LoadAt(kB, *load)); // a dead read whose address is the load
	p.resources      = {Add(p, Const(0x44)), Add(p, Const(0x55))};
	p.blocks         = {{.cond = c0, .succ = {1, 2}}, {.succ = {3}}, {.succ = {3}}, {.cond = n7, .succ = {4, 5}}, {.resources = {0}}, {.resources = {1}}};
	return p;
}

void TestK2Phi() {
	uint32_t   load = 0;
	const auto plan = PlanK2Phi(&load);
	const auto why  = Classify(plan);
	Check(why[load] == SupersetWhy::Superset, "krf2 K2 Phi: the load behind a control Phi is a superset load (dead users ignored)");
	// The S570 graph counts the dead read and Phi as users: not patchable, as S571/S574 saw.
	std::vector<uint32_t> structural, flat;
	for (const auto& b: plan.blocks) {
		if (b.cond != kNone) structural.push_back(b.cond);
	}
	Check(!PatchableLoads(Graph(plan), structural, flat)[load], "krf2 K2 Phi: K-rf v1 rejects it");
	const auto r = RunBoth(plan, Memory {{kA, {.stale = 0x20, .fresh = 4, .dirty = true}}, {kB + 16, {.stale = 1, .fresh = 1}}});
	uint32_t   extra = 0;
	Check(r.superset.drains == 0 && r.superset.defers >= 1, "krf2 K2 Phi: no drain");
	Check(r.superset.reach[4] && r.superset.reach[5] && !r.superset.reach[2], "krf2 K2 Phi: the Phi's condition reaches both successors");
	Check(Covers(r, &extra) && extra == 1, "krf2 K2 Phi: covers, one extra resource");
}

void TestAddressFallback() {
	MPlan p;
	p.user          = {8};
	const auto l1   = Add(p, LoadAt(kA));
	const auto l2   = Add(p, LoadAt(kB, l1)); // the dirty dword is l2's dword index
	const auto cond = Add(p, Not(Add(p, Ule(l2, Add(p, User(0))))));
	p.resources     = {Add(p, Const(0x11)), Add(p, Const(0x22))};
	p.blocks        = {{.cond = cond, .succ = {1, 2}}, {.resources = {0}}, {.resources = {1}}};
	const auto why  = Classify(p);
	Check(why[l1] == SupersetWhy::Address, "krf2 address: a load whose value is another load's address drains");
	Check(why[l2] == SupersetWhy::Superset, "krf2 address: the second load itself is a superset load");
	Memory mem {{kA, {.stale = 2, .fresh = 3, .dirty = true}}};
	for (uint32_t k = 0; k < 8; ++k) mem[kB + k * 4] = {.stale = k * 3, .fresh = k * 3};
	const auto r = RunBoth(p, mem);
	Check(r.superset.defers == 0 && r.superset.drains == 1, "krf2 address: the fault drains");
	const auto f = r.superset.fallback.find(SupersetWhy::Address);
	Check(f != r.superset.fallback.end() && f->second == 1 && r.superset.walkfail == 0, "krf2 address: counted as address, no retry");
	Check(r.superset.reach == r.sync.reach && Covers(r), "krf2 address: the walk is the synchronous one");
}

void TestLoopFallback() {
	// Block 0 -> 1; block 1: i = Phi(0 from block 0, i + load from block 2); cond i <= user -> 2 (loop body, -> 1) or 3.
	MPlan p;
	p.user          = {8};
	const auto zero = Add(p, Const(0));
	const auto load = Add(p, LoadAt(kA));
	MNode      phi {.op = Op::Phi, .cyclic = true};
	const auto i    = Add(p, phi);
	const auto next = Add(p, Sum(load, i)); // the load first: the walker evaluates it before it meets the cycle
	p.nodes[i].phi  = {{0, zero}, {2, next}};
	const auto cond = Add(p, Not(Add(p, Ule(Add(p, User(0)), i))));
	p.resources     = {Add(p, Const(0x11))};
	p.blocks        = {{.succ = {1}}, {.cond = cond, .succ = {2, 3}}, {.succ = {1}}, {.resources = {0}}};
	Check(Classify(p)[load] == SupersetWhy::Loop, "krf2 loop: a loop-carried load drains");
	const auto r = RunBoth(p, Memory {{kA, {.stale = 0x20, .fresh = 4, .dirty = true}}});
	const auto f = r.superset.fallback.find(SupersetWhy::Loop);
	Check(r.superset.defers == 0 && f != r.superset.fallback.end() && f->second >= 1, "krf2 loop: the fault drains, counted as loop");
	Check(r.superset.reach == r.sync.reach && Covers(r), "krf2 loop: the walk is the synchronous one");
}

void TestFlatCopy() {
	uint32_t load = 0;
	auto     p    = PlanK2Simple(&load);
	p.slots       = {load, Add(p, Const(7))};
	p.blocks[0].slots = {0};
	p.blocks[2].slots = {1};
	Check(Classify(p)[load] == SupersetWhy::Superset, "krf2 flat: a load in a condition and its own flat slot is a superset load");
	const auto r = RunBoth(p, Memory {{kA, {.stale = 0x20, .fresh = 4, .dirty = true}}});
	Check(r.superset.drains == 0 && r.pending.count == 1 && r.pending.entries[0].slot_count == 1 && r.pending.entries[0].slots[0] == 0,
	      "krf2 flat: the entry carries the slot");
	Check(r.superset.flat[0] == 0x20 && r.superset.uploaded[0] == 4 && r.sync.flat[0] == 4, "krf2 flat: stale upload, the GPU copy gives 4");
	Check(Covers(r), "krf2 flat: covers after the copy");
}

void TestStaticFallbacks() {
	{
		uint32_t load = 0;
		auto     p    = PlanK2Simple(&load);
		p.slots       = {Add(p, Sum(load, Add(p, Const(1))))};
		p.blocks[0].slots = {0};
		Check(Classify(p)[load] == SupersetWhy::DerivedFlat, "krf2: a value computed from the load in a flat slot drains (derived_flat)");
	}
	{
		uint32_t load = 0;
		auto     p    = PlanK2Simple(&load);
		p.resources.push_back(Add(p, Sum(load, Add(p, Const(1)))));
		Check(Classify(p)[load] == SupersetWhy::Descriptor, "krf2: a load in a descriptor drains (descriptor)");
	}
	{
		uint32_t load = 0;
		auto     p    = PlanK2Simple(&load);
		p.slots.assign(kMaxSlots + 1, load);
		p.blocks[0].slots = {0, 1, 2, 3, 4};
		Check(Classify(p)[load] == SupersetWhy::TooManySlots, "krf2: more flat slots than an entry holds drain (slots)");
	}
	{
		// The address of a load with a flat slot through a control Phi (K1): drains; without the flat slot it may defer.
		MPlan p;
		p.user          = {0, 1, 2, 8};
		const auto c0   = Add(p, Not(Add(p, User(0))));
		MNode      phi {.op = Op::Phi};
		phi.phi         = {{1, Add(p, User(1))}, {2, Add(p, User(2))}};
		const auto ph   = Add(p, phi);
		const auto load = Add(p, LoadAt(kA, ph));
		const auto cond = Add(p, Not(Add(p, Ule(load, Add(p, User(3))))));
		p.resources     = {Add(p, Const(0x11)), Add(p, Const(0x22))};
		p.blocks        = {{.cond = c0, .succ = {1, 2}}, {.succ = {3}}, {.succ = {3}}, {.cond = cond, .succ = {4, 5}, .slots = {0}},
		                   {.resources = {0}}, {.resources = {1}}};
		auto q          = p;
		p.slots         = {load};
		Check(Classify(p)[load] == SupersetWhy::PhiAddress, "krf2 K1: a flat-slot load with a control-Phi address drains (phi_address)");
		q.blocks[3].slots.clear();
		Check(Classify(q)[load] == SupersetWhy::Superset, "krf2 K1: without a flat slot the control-Phi address does not matter");
		const auto r = RunBoth(q, Memory {{kA + 4, {.stale = 0x20, .fresh = 4, .dirty = true}}});
		Check(r.superset.drains == 0 && Covers(r), "krf2 K1: the condition-only load through a Phi address covers");
	}
	{
		// Dead: a load nothing reads; a folded load (no guest load).
		std::vector<GraphNode> g(2);
		g[0] = Load();
		g[1] = Load();
		g[1].emits_load = false;
		const std::vector<Root>     roots;
		const std::vector<uint32_t> flat {1};
		const auto                  why = SupersetLoads(g, roots, flat);
		Check(why[0] == SupersetWhy::Dead && why[1] == SupersetWhy::NoLoad, "krf2: dead and folded loads");
	}
}

void TestWalkFailRetry() {
	// Block 2 (only the superset reaches it) has a flat slot whose value is unknown in that walk: a Phi of a block never reached.
	uint32_t load = 0;
	auto     p    = PlanK2Simple(&load);
	MNode phi {.op = Op::Phi};
	phi.phi       = {{4, Add(p, Const(9))}}; // block 4 has no predecessor
	p.blocks.push_back({});
	p.slots           = {Add(p, phi)};
	p.blocks[1].slots = {0}; // block 1 is the stale value's successor, not the true one
	p.blocks[3].succ.clear();
	const auto r = RunBoth(p, Memory {{kA, {.stale = 0x20, .fresh = 4, .dirty = true}}});
	Check(r.sync.ok && !r.sync.reach[1], "krf2 retry: the synchronous walk does not reach the failing block");
	Check(r.superset.walkfail == 1 && r.superset.retry == 1 && r.superset.drains == 1, "krf2 retry: the failed superset walks again and drains");
	Check(r.superset.reach == r.sync.reach && Covers(r), "krf2 retry: the retried walk is the synchronous one");
}

void TestCover() {
	const std::vector<MDesc> truth {{{1}}, {{2}}};
	std::vector<MDesc>       superset {{{1}}, {{2}}};
	Check(SupersetCoverResources(std::span<const MDesc>(truth), std::span<const MDesc>(superset)).agree, "krf2 cover: equal");
	superset[1] = MDesc {};
	Check(!SupersetCoverResources(std::span<const MDesc>(truth), std::span<const MDesc>(superset)).agree,
	      "krf2 cover: a resource of the true walk the superset lacks (the last one)");
	const std::vector<MDesc> truth2 {{{1}}, MDesc {}};
	const std::vector<MDesc> superset2 {{{1}}, {{5}}};
	const auto               c = SupersetCoverResources(std::span<const MDesc>(truth2), std::span<const MDesc>(superset2));
	Check(c.agree && c.extra == 1, "krf2 cover: an extra resource");
	Pending p;
	const uint32_t s[] {1};
	p.Add(0x100, 0x20, s);
	const std::vector<uint32_t> fresh {4};
	Check(SupersetCoverFlat(std::vector<uint32_t> {7, 4, 0}, std::vector<uint32_t> {7, 0x20, 9}, p, fresh).agree, "krf2 cover: flat");
	Check(!SupersetCoverFlat(std::vector<uint32_t> {7, 5, 0}, std::vector<uint32_t> {7, 0x20, 9}, p, fresh).agree,
	      "krf2 cover: the patched slot must hold the drained dword");
	Check(SupersetCoverFlat(std::vector<uint32_t> {7, 0, 0}, std::vector<uint32_t> {7, 0x20, 9}, p, fresh).agree,
	      "krf2 cover: a patched slot the true walk does not reach");
	Check(!SupersetCoverFlat(std::vector<uint32_t> {7, 4, 3}, std::vector<uint32_t> {7, 0x20, 9}, p, fresh).agree,
	      "krf2 cover: a word the true walk reached differs");
	const std::vector<std::pair<uint64_t, uint64_t>> r1 {{1, 4}}, r2 {{2, 4}, {1, 4}};
	Check(SupersetCoverRanges(std::span(r1), std::span(r2)) && !SupersetCoverRanges(std::span(r2), std::span(r1)), "krf2 cover: ranges");
}

void TestDecideSuperset() {
	SupersetQuery ok;
	ok.armed      = true;
	ok.gpu_thread = true;
	ok.site       = true;
	ok.why        = SupersetWhy::Superset;
	ok.queried    = true;
	ok.dirty      = true;
	ok.room       = true;
	Check(DecideSuperset(ok) == SupersetReject::None, "krf2 decide: defer");
	auto q = ok;
	q.dirty = false;
	Check(DecideSuperset(q) == SupersetReject::NotDirty, "krf2 decide: clean bytes, the ordinary path");
	q     = ok;
	q.why = SupersetWhy::Address;
	Check(DecideSuperset(q) == SupersetReject::Static, "krf2 decide: a static reason");
	q          = ok;
	q.disabled = true;
	Check(DecideSuperset(q) == SupersetReject::Retry, "krf2 decide: a retried scope drains");
	q              = ok;
	q.slots        = 1;
	q.address_base = true;
	Check(DecideSuperset(q) == SupersetReject::AddressBase, "krf2 decide: a flat slot that is an address base");
	q.slots = 0;
	Check(DecideSuperset(q) == SupersetReject::None, "krf2 decide: no slot, the address base does not matter");
	q          = ok;
	q.inflight = true;
	Check(DecideSuperset(q) == SupersetReject::Inflight, "krf2 decide: in flight");
	q      = ok;
	q.site = false;
	Check(DecideSuperset(q) == SupersetReject::NotSite, "krf2 decide: not a resource walker site");
	q       = ok;
	q.armed = false;
	Check(DecideSuperset(q) == SupersetReject::NotArmed, "krf2 decide: outside DISPATCH_INDIRECT");
	q      = ok;
	q.room = false;
	Check(DecideSuperset(q) == SupersetReject::Full, "krf2 decide: full");
	Check(ParseMode("superset") == Mode::Superset && ParseMode("superset_shadow") == Mode::SupersetShadow && IsSuperset(Mode::SupersetShadow) &&
	          !IsSuperset(Mode::On),
	      "krf2: modes parsed");
}

// Random acyclic plans: forward edges, conditions over loads / user data, loads addressed by other values, resources and flat slots per block.
void TestRandom() {
	std::mt19937 rng(579);
	uint32_t     covered = 0, deferred = 0, retried = 0;
	for (int iter = 0; iter < 3000; ++iter) {
		MPlan      p;
		const auto pick = [&](uint32_t n) { return static_cast<uint32_t>(rng() % n); };
		p.user          = {pick(6), pick(6), pick(6)};
		Memory     mem;
		const auto node = [&](int depth, auto&& self) -> uint32_t {
			const auto kind = depth > 2 ? pick(3) : pick(7);
			switch (kind) {
				case 0: return Add(p, Const(pick(6)));
				case 1: return Add(p, User(pick(3)));
				case 2:
				case 3: {
					const uint64_t base = 0x1000 + uint64_t {pick(4)} * 0x100;
					const uint32_t addr = depth < 2 && pick(3) == 0 ? self(depth + 1, self) : kNone;
					for (uint32_t k = 0; k < 8; ++k) {
						const uint32_t fresh = pick(6), stale = pick(3) == 0 ? pick(6) : fresh;
						mem.emplace(base + k * 4, Cell {.stale = stale, .fresh = fresh, .dirty = pick(2) == 0});
					}
					return Add(p, LoadAt(base, addr));
				}
				case 4: return Add(p, Ule(self(depth + 1, self), self(depth + 1, self)));
				case 5: return Add(p, Not(self(depth + 1, self)));
				default: return Add(p, Sum(self(depth + 1, self), self(depth + 1, self)));
			}
		};
		const uint32_t blocks = 2 + pick(5);
		for (uint32_t r = 0; r < blocks + 1; ++r) p.resources.push_back(pick(4) == 0 ? node(1, node) : Add(p, Const(0x10 + r)));
		for (uint32_t b = 0; b < blocks; ++b) {
			MBlock block;
			if (b + 1 < blocks) {
				const uint32_t t = b + 1 + pick(blocks - b - 1), f = b + 1 + pick(blocks - b - 1);
				block.succ       = {t, f};
				if (pick(4) != 0) block.cond = node(0, node);
			}
			if (pick(2) == 0) block.resources.push_back(b);
			if (pick(3) == 0) {
				p.slots.push_back(pick(2) == 0 ? node(1, node) : Add(p, Const(pick(6) + 1)));
				block.slots.push_back(static_cast<uint32_t>(p.slots.size() - 1));
			}
			p.blocks.push_back(block);
		}
		const auto r = RunBoth(p, mem);
		if (!r.sync.ok) continue; // a plan the walker cannot walk at all: not a superset question
		deferred += r.superset.defers != 0 ? 1u : 0u;
		retried += r.superset.walkfail;
		if (Covers(r)) {
			++covered;
		} else {
			Check(false, "krf2 random: the superset covers the synchronous walk");
			std::fprintf(stderr, "krf2 random iter=%d sync.ok=%d sup.ok=%d defers=%u drains=%u walkfail=%u\n", iter, r.sync.ok, r.superset.ok,
			             r.superset.defers, r.superset.drains, r.superset.walkfail);
			for (size_t b = 0; b < p.blocks.size(); ++b)
				std::fprintf(stderr, "  block %zu reach sync=%d sup=%d cond=%u\n", b, int(r.sync.reach[b]), int(r.superset.reach[b]), p.blocks[b].cond);
			for (size_t i = 0; i < r.sync.resources.size(); ++i)
				std::fprintf(stderr, "  res %zu sync=%x sup=%x\n", i, r.sync.resources[i].dwords[0], r.superset.resources[i].dwords[0]);
			for (size_t i = 0; i < r.sync.flat.size(); ++i)
				std::fprintf(stderr, "  flat %zu sync=%x sup=%x up=%x\n", i, r.sync.flat[i], r.superset.flat[i], r.superset.uploaded[i]);
			break;
		}
	}
	Check(covered > 1000 && deferred > 200, "krf2 random: enough plans walked and deferred");
	std::printf("RfGpuPatchTests: krf2 random covered=%u deferred=%u retried=%u\n", covered, deferred, retried);
}

} // namespace krf2

namespace krf3 {

void TestParse() {
	const auto off = ParseKrf3(nullptr);
	Check(!off.Any() && !ParseKrf3("0").Any() && !ParseKrf3("x").Any(), "krf3 parse: unset / 0 / unknown = off");
	const auto all = ParseKrf3("1");
	Check(all.inflight && all.page, "krf3 parse: 1 = inflight + page");
	Check(ParseKrf3("inflight").inflight && !ParseKrf3("inflight").page, "krf3 parse: inflight");
	Check(ParseKrf3("page").page && !ParseKrf3("page").inflight, "krf3 parse: page");
}

void TestDecide() {
	SupersetQuery q;
	q.armed      = true;
	q.gpu_thread = true;
	q.site       = true;
	q.why        = SupersetWhy::Superset;
	q.queried    = true;
	q.dirty      = true;
	q.inflight   = true;
	q.room       = true;
	Check(DecideSuperset(q) == SupersetReject::Inflight, "krf3 decide: v2 (inflight_ok off) drains re-dirtied bytes in flight");
	q.inflight_ok = true;
	Check(DecideSuperset(q) == SupersetReject::None, "krf3 decide: inflight_ok defers dirty bytes with a download in flight");
	q.dirty = false;
	Check(DecideSuperset(q) == SupersetReject::NotDirty, "krf3 decide: clean bytes in flight keep the ordinary path");
	q.dirty = true;
	q.why   = SupersetWhy::Descriptor;
	Check(DecideSuperset(q) == SupersetReject::Static, "krf3 decide: inflight_ok does not defer a static fallback");
	q.why      = SupersetWhy::Superset;
	q.disabled = true;
	Check(DecideSuperset(q) == SupersetReject::Retry, "krf3 decide: inflight_ok does not defer in a retried scope");
}

void TestPageKey() {
	GuestPageGuard g;
	(void)g.OnDefer(0x571b2effcull, 1, true);
	Check(g.NoteGuestRead(0x571b2e080ull), "krf3 key: a guest read of another byte of the deferred dword's page is seen");
	Check(g.NoteGuestRead(0x571b2effcull), "krf3 key: the dword itself");
	Check(!g.NoteGuestRead(0x571b2f03cull), "krf3 key: the next page is not tracked");
	Check(!g.NoteGuestRead(0x555f41dd0ull), "krf3 key: an unrelated page is not tracked");
}

// S580 desert: deferred dwords on two pages (e000 once, f000 twice per flip); the guest reads page e000 three times per flip whenever the walker
// left it dirty (deferred) in that flip; a drain cleans it (no guest fault).
struct DesertStats {
	uint32_t first_ban_flip = UINT32_MAX;
	uint32_t e_defer_flips  = 0; // flips of [500, 2500) in which page e000 was deferred (trials)
	uint32_t f_drains       = 0;
	uint32_t trials         = 0;
	uint32_t guest_faults   = 0; // over [500, 2500)
};
DesertStats RunDesert(GuestPageGuard& g, uint32_t flips, uint32_t stop_reading_at = UINT32_MAX, uint32_t* resumed_at = nullptr) {
	DesertStats st;
	for (uint32_t f = 0; f < flips; ++f) {
		const auto e = g.OnDefer(0x571b2effcull, f, true);
		const auto f1 = g.OnDefer(0x571b2f03cull, f, true);
		const auto f2 = g.OnDefer(0x571b2f07cull, f, true);
		if (e.banned_now && st.first_ban_flip == UINT32_MAX) st.first_ban_flip = f;
		st.trials += (e.trial ? 1u : 0u) + (f1.trial ? 1u : 0u) + (f2.trial ? 1u : 0u);
		st.f_drains += (f1.drain ? 1u : 0u) + (f2.drain ? 1u : 0u);
		if (!e.drain && f >= 500 && f < 2500) ++st.e_defer_flips;
		if (!e.drain && f >= stop_reading_at && resumed_at != nullptr && *resumed_at == UINT32_MAX && f > stop_reading_at + 64) *resumed_at = f;
		if (!e.drain && f < stop_reading_at) {
			for (int k = 0; k < 3; ++k) {
				(void)g.NoteGuestRead(0x571b2e080ull + 0x40u * k);
				if (f >= 500 && f < 2500) ++st.guest_faults;
			}
		}
	}
	return st;
}

void TestDesert() {
	GuestPageGuard g;
	const auto     st = RunDesert(g, 2600);
	Check(st.first_ban_flip < 40, "krf3 desert: the page the guest reads is banned within the first window");
	Check(st.e_defer_flips * 20 <= 2000, "krf3 desert: the read page defers in <= 5 % of the flips (trials only)");
	Check(st.f_drains == 0, "krf3 desert: the page the guest does not read keeps deferring");
	Check(st.trials >= 1 && st.trials <= 6, "krf3 desert: a few trials, the ban doubles");
	Check(st.guest_faults <= 3u * 100u, "krf3 desert: guest faults on the page stay at trial level");
	std::printf("krf3 desert: first ban at flip %u, read page deferred in %u of 2000 flips, %u trials, %u guest faults\n", st.first_ban_flip,
	            st.e_defer_flips, st.trials, st.guest_faults);
}

// S580 garden: 6 defers per flip on two pages; a burst of 30 guest reads on the first page at the level entry (flips 10..12), then a guest read on
// it in ~2 % of the flips. The measurement window starts at F10+330.
void TestGarden() {
	GuestPageGuard g;
	std::mt19937   rng(581);
	uint32_t       window_drains = 0, window_flips = 0, bans = 0;
	for (uint32_t f = 0; f < 1400; ++f) {
		bool any_drain = false;
		for (int k = 0; k < 3; ++k) {
			const auto a = g.OnDefer(0x57954027cull, f, true);
			const auto b = g.OnDefer(0x5794bf8fcull, f, true);
			any_drain    = any_drain || a.drain || b.drain;
			bans += (a.banned_now ? 1u : 0u) + (b.banned_now ? 1u : 0u);
		}
		if (f >= 10 && f <= 12) {
			for (int k = 0; k < 10; ++k) (void)g.NoteGuestRead(0x579540264ull);
		} else if (rng() % 50 == 0) {
			(void)g.NoteGuestRead(0x579540264ull);
		}
		if (f >= 330) {
			++window_flips;
			if (any_drain) ++window_drains;
		}
	}
	Check(bans <= 1, "krf3 garden: at most the entry burst bans");
	Check(window_drains == 0, "krf3 garden: no drain in F10+330..+1330 (the gain stays)");
	std::printf("krf3 garden: %u ban(s), %u of %u window flips drained\n", bans, window_drains, window_flips);
}

// The guest stops reading the page: the ban runs out and the page defers again for good.
void TestSceneChange() {
	GuestPageGuard g;
	uint32_t       resumed = UINT32_MAX;
	(void)RunDesert(g, 9000, 600, &resumed);
	Check(resumed != UINT32_MAX && resumed < 600 + 4096 + 200, "krf3 scene change: the page defers again after the last ban");
	bool drained_late = false;
	for (uint32_t f = 9000; f < 9300; ++f) drained_late = drained_late || g.OnDefer(0x571b2effcull, f, true).drain;
	Check(!drained_late, "krf3 scene change: and stays deferring");
}

// KYTY_KRF3=inflight: the guard counts what it would ban, never drains.
void TestWouldBan() {
	GuestPageGuard g;
	uint32_t       would = 0, drains = 0;
	for (uint32_t f = 0; f < 200; ++f) {
		const auto r = g.OnDefer(0x571b2effcull, f, false);
		would += r.would_ban ? 1u : 0u;
		drains += r.drain ? 1u : 0u;
		(void)g.NoteGuestRead(0x571b2e080ull);
	}
	Check(would >= 5 && drains == 0, "krf3 inflight only: would-bans counted, no drain");
}

// 17 pages: the least recently deferred page that is not banned makes room; a banned page stays banned.
void TestEviction() {
	GuestPageGuard::Config c;
	c.window = 4;
	GuestPageGuard g(c);
	for (uint32_t k = 0; k < 4; ++k) {
		(void)g.OnDefer(0x100000ull, 1, true);
		(void)g.NoteGuestRead(0x100010ull);
	}
	Check(g.Banned(0x100000ull, 2), "krf3 eviction: the read page is banned");
	for (uint64_t p = 1; p <= 16; ++p) (void)g.OnDefer(0x100000ull + p * 0x1000ull, 2 + p, true);
	Check(g.Banned(0x100000ull, 20), "krf3 eviction: the banned page survives 16 newer pages");
	Check(g.NoteGuestRead(0x110000ull), "krf3 eviction: the newest page is tracked");
}

// Guest threads count hits while the CP thread defers (no lost page, no crash; hits only on tracked pages).
void TestThreads() {
	GuestPageGuard        g;
	std::atomic<bool>     stop {false};
	std::atomic<uint64_t> seen {0};
	std::atomic<uint64_t> calls {0};
	// The pages are tracked before the guest threads start (the CP loop below may otherwise finish before a thread runs, S581 PC run).
	for (uint64_t p = 0; p < 16; ++p) (void)g.OnDefer(0x200000ull + p * 0x1000ull, 0, true);
	std::vector<std::thread> threads;
	for (int t = 0; t < 4; ++t) {
		threads.emplace_back([&, t] {
			uint64_t n = 0, i = 0;
			while (!stop.load(std::memory_order_relaxed)) {
				n += g.NoteGuestRead(0x200000ull + static_cast<uint64_t>((i++ + t) % 24) * 0x1000ull + 0x80) ? 1u : 0u;
				calls.fetch_add(1, std::memory_order_relaxed);
			}
			seen.fetch_add(n);
		});
	}
	// The CP defers until the guest threads made calls meanwhile (at least 20000 defers).
	for (uint32_t f = 0; f < 20000 || calls.load(std::memory_order_relaxed) < 40000; ++f) {
		(void)g.OnDefer(0x200000ull + static_cast<uint64_t>(f % 20) * 0x1000ull, f / 20, true);
	}
	stop.store(true);
	for (auto& t: threads) t.join();
	Check(seen.load() > 0, "krf3 threads: guest hits seen while the CP defers");
}

} // namespace krf3

} // namespace

int main() {
	TestPatchable();
	TestDecide();
	TestPending();
	TestAgree();
	TestModel();
	krf2::TestK2Simple();
	krf2::TestK2Phi();
	krf2::TestAddressFallback();
	krf2::TestLoopFallback();
	krf2::TestFlatCopy();
	krf2::TestStaticFallbacks();
	krf2::TestWalkFailRetry();
	krf2::TestCover();
	krf2::TestDecideSuperset();
	krf2::TestRandom();
	krf3::TestParse();
	krf3::TestDecide();
	krf3::TestPageKey();
	krf3::TestDesert();
	krf3::TestGarden();
	krf3::TestSceneChange();
	krf3::TestWouldBan();
	krf3::TestEviction();
	krf3::TestThreads();
	if (g_failures != 0) {
		std::fprintf(stderr, "RfGpuPatchTests: %d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("RfGpuPatchTests: OK");
	return 0;
}
