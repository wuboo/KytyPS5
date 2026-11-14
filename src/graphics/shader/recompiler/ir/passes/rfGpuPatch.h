#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RFGPUPATCH_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RFGPUPATCH_H_

// S570 K-rf: KYTY_RF_GPU_PATCH=0|1|shadow (unset = 0, the old path bit for bit).
//
// Why the CP waits (S569 DIAG): 4 sync_rf_gpu per garden flip, all inside DISPATCH_INDIRECT, all read faults of the compiled SRT walker
// (SrtWalker.cpp, `mov eax, [r10]`) on dwords that the compute dispatch right before it wrote (dirty=1, cmds_since_writer=1). The walker
// flattens the shader's uniform scalar loads (flattened_srt): the CPU must know every such value before it records the dispatch, so the
// fault handler drains the whole queue (~3 ms for the first of the four). The dispatch arguments themselves are not read on the CPU
// (vkCmdDispatchIndirect reads them on the GPU; the zero-args skip reads the backing only when it is clean).
//
// The patch: a walker load whose value is used by nothing but its flattened SRT slots (no descriptor, condition, uniform fill, indirect
// count or other node depends on it, its slot is not a global address base the CPU backs) does not need the CPU to know the value.
//   - the walker compiler marks such loads (PatchableLoads below, a pure function of the walker's node graph) and records the code
//     offset of each one's guest load with its flat slots;
//   - the exception handler, on the CP thread, inside an armed Scope (RenderExecutor::DispatchIndirect), for a fault of such a load on
//     GPU-dirty bytes (not in flight), completes the load with the stale backing value instead of draining, and records (address, slots);
//   - after RebindBuffers uploaded the flattened SRT, the dispatch records, before itself: barrier (any write -> transfer read),
//     vkCmdCopyBuffer of the 4 guest bytes from the buffer cache's GPU buffer into each slot of the uploaded SRT, barrier (transfer write ->
//     compute shader read). The GPU then sees what the drain would have shown the CPU: the guest bytes after every command recorded before.
// shadow: everything as 1, and then the dispatch drains each deferred address anyway, walks the program again with the fresh bytes and
// compares the two snapshots (everything but the patched slots must be equal, rf_patch_ne_n = 0); it records the copies too (they write
// the value the CPU already put there). Counters: rf_patch_*; the old rf_gpu faults split by reader: rf_gpu_walker_n / rf_gpu_other_n.
//
// Pure part (no Vulkan, no IR types): the graph analysis, the fault decision and the pending list. tests/RfGpuPatchTests.cpp.
// Mutants (tests, WILL_FAIL): KYTY_RFPATCH_MUTANT_DIRECT_USES_ONLY (a use through a folded node result is missed), _IGNORE_DIRTY (defers a
// clean or in-flight read), _IGNORE_ADDRESS_BASE (patches a slot the CPU uses as a global address base), _NO_DEDUPE (the same address and
// slot twice).

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR::RfGpuPatch {

// S573 Oracle (KYTY_RF_GPU_PATCH=oracle, DIAG ONLY, never install: may draw with a wrong value): every walker fault on GPU-dirty bytes inside
// DISPATCH_INDIRECT completes with the backing's (stale) dword, patchable or not, with no drain and no GPU copy. It measures the ceiling of any
// lever on these waits (and where the wait goes next: guest fault lambdas, the next sync), not a fix. rf_patch_defer_n counts the completions,
// rf_patch_stale_ne_n the ones whose stale dword differs from the previous completion at the same address (the value is not constant).
// S579 K-rf v2 (KYTY_RF_GPU_PATCH=superset | superset_shadow): see the block "K-rf v2" below.
enum class Mode : uint8_t { Off, On, Shadow, Oracle, Superset, SupersetShadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	if (std::strcmp(text, "oracle") == 0) return Mode::Oracle;
	if (std::strcmp(text, "superset") == 0) return Mode::Superset;
	if (std::strcmp(text, "superset_shadow") == 0) return Mode::SupersetShadow;
	return Mode::Off;
}

inline bool IsSuperset(Mode mode) {
	return mode == Mode::Superset || mode == Mode::SupersetShadow;
}

inline int& OverrideRef() {
	static int value = -1;
	return value;
}

inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_RF_GPU_PATCH"));
	return OverrideRef() >= 0 ? static_cast<Mode>(OverrideRef()) : mode;
}

// ---- graph analysis (walker compile time) -------------------------------------------------------------------------------------------------

// One node of the walker graph. `deps` are the nodes its value is computed from (its args and, for a node that folded into another node,
// that node). `read` marks a guest load; `emits_load` that it really reads guest memory (a load that folded to a constant or to another
// node does not).
struct GraphNode {
	std::vector<uint32_t> deps;
	bool                  read       = false;
	bool                  emits_load = false;
	bool                  cyclic     = false; // referenced while it was being analysed: its value feeds itself
	bool                  control_phi = false; // S579: an evaluated control Phi (its value depends on the blocks the walk reached)
};

// `structural`: the nodes the CPU reads after the walk (conditions, descriptor dwords, walked outputs). `flat`: per flat read slot, the
// node it stores (UINT32_MAX = not a node). Result: per node, true when it is a guest load whose value only reaches flat slots, directly.
inline std::vector<bool> PatchableLoads(std::span<const GraphNode> nodes, std::span<const uint32_t> structural,
                                        std::span<const uint32_t> flat) {
	const size_t      n = nodes.size();
	std::vector<bool> used(n, false);
	// A node is used when a structural root or any other node depends on it. A flat slot storing a node is not a use (that is the slot
	// the copy replaces); everything else is.
	for (const auto root: structural) {
		if (root < n) used[root] = true;
	}
	for (size_t i = 0; i < n; ++i) {
#if defined(KYTY_RFPATCH_MUTANT_DIRECT_USES_ONLY)
		// mutant: only the first dependency counts (a fold or a second argument is missed)
		if (!nodes[i].deps.empty() && nodes[i].deps[0] < n && nodes[i].deps[0] != i) used[nodes[i].deps[0]] = true;
#else
		for (const auto dep: nodes[i].deps) {
			if (dep < n && dep != i) used[dep] = true;
		}
#endif
	}
	std::vector<bool> stored(n, false);
	for (const auto node: flat) {
		if (node < n) stored[node] = true;
	}
	std::vector<bool> out(n, false);
	for (size_t i = 0; i < n; ++i) {
		const auto& node = nodes[i];
		out[i]           = node.read && node.emits_load && !node.cyclic && !used[i] && stored[i];
	}
	return out;
}

constexpr uint32_t kMaxEntries = 8;
constexpr uint32_t kMaxSlots   = 4;

// ---- S573 DIAG (KYTY_DIAG_RF_WHY=1, no change of behaviour): why a walker load is not patchable ------------------------------------------
// S571: all 4 sync_rf_gpu per garden flip were rejected as not patchable. Explain gives, per load node, the reason PatchableLoads has (the plan's
// control-Phi exclusion is reported apart, by the caller), the structural roots (indices into `structural`) and the other nodes that use it,
// and its flat slots. SrtWalker.cpp formats it per load site; the exception handler prints it for the CP's walker faults ([rf-why]).

inline bool WhyEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_DIAG_RF_WHY");
		return v != nullptr && v[0] == '1';
	}();
	return on;
}

enum class Why : uint8_t { Patchable, NotLoad, NoLoad, Cyclic, Root, Node, NotStored, TooManySlots };

inline const char* WhyName(Why why) {
	switch (why) {
		case Why::Patchable: return "patchable";
		case Why::NotLoad: return "not_a_load";
		case Why::NoLoad: return "no_guest_load";
		case Why::Cyclic: return "cyclic";
		case Why::Root: return "structural_root";
		case Why::Node: return "used_by_node";
		case Why::NotStored: return "not_in_flat";
		case Why::TooManySlots: return "too_many_slots";
	}
	return "?";
}

struct Explained {
	Why                   why        = Why::NotLoad;
	uint32_t              flat_slots = 0;
	std::vector<uint32_t> roots; // indices into `structural`
	std::vector<uint32_t> users; // nodes that depend on it
};

// Same rules and the same order of checks as PatchableLoads (without the plan-level control-Phi exclusion).
inline std::vector<Explained> Explain(std::span<const GraphNode> nodes, std::span<const uint32_t> structural, std::span<const uint32_t> flat) {
	const size_t           n = nodes.size();
	std::vector<Explained> out(n);
	for (size_t r = 0; r < structural.size(); ++r) {
		if (structural[r] < n) out[structural[r]].roots.push_back(static_cast<uint32_t>(r));
	}
	for (size_t i = 0; i < n; ++i) {
		for (const auto dep: nodes[i].deps) {
			if (dep < n && dep != i && (out[dep].users.empty() || out[dep].users.back() != i)) out[dep].users.push_back(static_cast<uint32_t>(i));
		}
	}
	for (const auto node: flat) {
		if (node < n) ++out[node].flat_slots;
	}
	for (size_t i = 0; i < n; ++i) {
		auto&       e    = out[i];
		const auto& node = nodes[i];
		if (!node.read) e.why = Why::NotLoad;
		else if (!node.emits_load) e.why = Why::NoLoad;
		else if (node.cyclic) e.why = Why::Cyclic;
		else if (!e.roots.empty()) e.why = Why::Root;
		else if (!e.users.empty()) e.why = Why::Node;
		else if (e.flat_slots == 0) e.why = Why::NotStored;
		else if (e.flat_slots > kMaxSlots) e.why = Why::TooManySlots;
		else e.why = Why::Patchable;
	}
	return out;
}

// The DISPATCH_INDIRECT whose walks run on this thread (its argument address; 0 outside one), for the [rf-why] line.
inline uint64_t& DiagDispatchArgs() {
	static thread_local uint64_t args = 0;
	return args;
}

// ---- fault decision (exception handler, CP thread) ----------------------------------------------------------------------------------------

struct FaultQuery {
	Mode     mode       = Mode::Off;
	bool     armed      = false; // inside a Scope on this thread
	bool     gpu_thread = false;
	uint32_t slots      = 0;     // patchable flat slots of the faulting load (0: not a patchable walker load)
	bool     address_base = false; // one of the slots is a global address base
	bool     queried    = false; // the dirty / in-flight state could be read without waiting
	bool     dirty      = false; // the 4 bytes are GPU-dirty
	bool     inflight   = false; // a GPU->host copy of them is recorded and not published
	bool     room       = false; // the pending list has space
};

enum class Action : uint8_t { Drain, Defer };

enum class Reject : uint8_t { None, NotArmed, NotPatchable, AddressBase, NotDirty, Inflight, Full, Count };

inline Reject Decide(const FaultQuery& q) {
	if (q.mode == Mode::Off || !q.armed || !q.gpu_thread) return Reject::NotArmed;
	const bool oracle = q.mode == Mode::Oracle; // S573 DIAG: the value's use does not matter, only the bytes' state
	if (q.slots == 0 && !oracle) return Reject::NotPatchable;
#if !defined(KYTY_RFPATCH_MUTANT_IGNORE_ADDRESS_BASE)
	if (q.address_base && !oracle) return Reject::AddressBase;
#endif
#if !defined(KYTY_RFPATCH_MUTANT_IGNORE_DIRTY)
	// Clean bytes need no wait (the ordinary fault path reads them); bytes in flight are about to change in the backing.
	if (!q.queried || !q.dirty) return Reject::NotDirty;
	if (q.inflight) return Reject::Inflight;
#endif
	if (!q.room && !oracle) return Reject::Full;
	return Reject::None;
}

// Set by the exception handler before the ordinary fault path: the faulting instruction is a load of the SRT walker (DIAG attribution of
// the CP's read faults, counted in RenderContext::HandleFault).
inline bool& FaultFromWalker() {
	static thread_local bool walker = false;
	return walker;
}

// ---- pending list (per thread, per armed scope) -------------------------------------------------------------------------------------------

struct Entry {
	uint64_t                       address = 0; // guest address of the dword
	uint32_t                       stale   = 0; // what the walker was given
	uint32_t                       slot_count = 0;
	std::array<uint32_t, kMaxSlots> slots {};   // flattened SRT dword indices
};

struct Pending {
	bool                            armed = false;
	uint32_t                        count = 0;
	std::array<Entry, kMaxEntries>  entries {};
	// S579 K-rf v2: loads deferred as unknown in this scope; set after a failed superset walk (SupersetRetry): the scope drains from then on.
	uint32_t                        superset_defers = 0;
	bool                            superset_off    = false;

	[[nodiscard]] bool Room() const { return count < kMaxEntries; }
	// Adds the read; the same address again (a walk restarted by a control phi) only adds new slots. False when full.
	bool Add(uint64_t address, uint32_t stale, std::span<const uint32_t> slots) {
		for (uint32_t i = 0; i < count; ++i) {
			auto& e = entries[i];
#if defined(KYTY_RFPATCH_MUTANT_NO_DEDUPE)
			break;
#endif
			if (e.address != address) continue;
			e.stale = stale;
			for (const auto s: slots) {
				const auto end = e.slots.begin() + e.slot_count;
				if (std::find(e.slots.begin(), end, s) == end && e.slot_count < kMaxSlots) e.slots[e.slot_count++] = s;
			}
			return true;
		}
		if (count >= kMaxEntries) return false;
		auto& e     = entries[count++];
		e           = {};
		e.address   = address;
		e.stale     = stale;
		for (const auto s: slots) {
			if (e.slot_count < kMaxSlots) e.slots[e.slot_count++] = s;
		}
		return true;
	}
	void Clear() { count = 0; }
};

inline Pending& ThreadPending() {
	static thread_local Pending pending;
	return pending;
}

// Arms the fault path for the walks of one dispatch on this thread. The destructor disarms; the entries stay until the next Scope.
class Scope {
public:
	explicit Scope(bool on): m_on(on) {
		if (!m_on) return;
		auto& p = ThreadPending();
		p.Clear();
		p.superset_defers = 0;
		p.superset_off    = false;
		p.armed           = true;
	}
	~Scope() {
		if (m_on) ThreadPending().armed = false;
	}
	void Disarm() {
		if (m_on) ThreadPending().armed = false;
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	bool m_on;
};

// Shadow: the second walk's snapshot against the first. Every word of `a` and `b` must be equal except the patched slots, which must hold
// `fresh` in `b`. Returns false on any other difference (or a size change).
inline bool SnapshotsAgree(std::span<const uint32_t> a, std::span<const uint32_t> b, const Pending& pending) {
	if (a.size() != b.size()) return false;
	std::vector<bool> patched(a.size(), false);
	for (uint32_t i = 0; i < pending.count; ++i) {
		const auto& e = pending.entries[i];
		for (uint32_t k = 0; k < e.slot_count; ++k) {
			if (e.slots[k] < a.size()) patched[e.slots[k]] = true;
		}
	}
	for (size_t i = 0; i < a.size(); ++i) {
		if (!patched[i] && a[i] != b[i]) return false;
	}
	return true;
}


// ==== S579 K-rf v2 "superset" (KYTY_RF_GPU_PATCH=superset | superset_shadow) ======================================================================
// S574: all 4 sync_rf_gpu per garden flip are class K2: a walker load (ReadConstBuffer of CB0, GPU-dirty) whose value only reaches a block
// condition (ULessThanEqual32 > LogicalNot > cond, in c5ee... through a control Phi first). The shader itself evaluates that condition on the GPU
// from the real bytes; the CPU only needs it to know which blocks may run, i.e. which resources and flat slots to prepare.
// The superset: such a load does not wait. The exception handler completes it with the stale backing dword and marks its node "deferred"
// (state byte kDeferredState instead of 2 = known). For the walker a deferred node is unknown, exactly as a value it cannot evaluate: a condition
// that depends on it is unknown and both successors are reached (SuccessorsWhenUnknown), a Phi over it is unknown, and so on; reachability only
// grows. So the walk prepares every resource and flat slot that any value of the dword could need: a superset of the true walk's, with the same
// descriptors (no descriptor depends on the load). The GPU runs the true branch with the true value. A flat slot that stores the deferred dword
// itself (the only flat use allowed) holds the stale dword after the upload and gets the GPU's bytes by a 4-byte copy (the S570 path).
// When the plan does not allow it, the load drains as before and the fault counts its reason (krf2_fallback_<why>_n):
//   loop        the load, a value it reaches or its address is cyclic (loop-carried);
//   address     its value reaches the address, offset or guard of another load (K4);
//   descriptor  its value reaches a descriptor dword (K3);
//   output      its value reaches a walked output (indirect key count, selector, uniform fill);
//   derived_flat a value computed from it is stored in a flat slot (a copy cannot produce it);
//   slots       more flat slots than an entry holds;
//   phi_address it has flat slots and its address depends on an evaluated control Phi (a restarted walk can load it from elsewhere: K1);
//   other       not a registered load site of a resource walker (call targets), dead or no guest load.
// A superset walk that fails where the true one may not (a flat read or descriptor of a block only the superset reaches is unknown, an invalid
// descriptor there) walks again with the drain (SupersetRetry, krf2_fallback_walkfail_n; pipelineCache.cpp) and its plan drains from then on
// (SrtWalkerSupersetBan; those faults: krf2_fallback_retry_n). Only the uses the walker actually evaluates count: the
// graph's deps are runtime deps (a node that folded to a constant or to unknown does not read its args), and nodes no root and no flat slot
// reaches (dead) are ignored.
// superset_shadow: the superset walk, then each deferred dword drained and the program walked again with the fresh bytes; SupersetCover* checks
// that every resource / flat word of the true walk is in the superset unchanged (extra ones counted: krf2_extra_res_n / _wr_n / _flat_n), and
// a GPU readback checks the copied slots against the drained dwords. Mismatches: rf_patch_ne_n (must stay 0).
// Mutants (tests, WILL_FAIL): KYTY_RFPATCH_MUTANT_SKIP_SUCCESSOR (an unknown condition reaches one successor), _COND_AS_TAKEN (the deferred
// dword is taken as known), _NO_FLAT_COPY (no GPU copy into the slot), _DEFER_ADDRESS (a load that reaches an address defers), _DEFER_LOOP
// (a loop-carried load defers), _COVER_MISSES_ONE (the shadow check ignores the last resource), _CPHI_ADDRESS (a flat-slot load with a control-Phi
// dependent address defers).

constexpr uint8_t kKnownState    = 2;
constexpr uint8_t kDeferredState = 4;

// The state byte the handler gives a deferred load's node.
inline uint8_t DeferredStateByte() {
#if defined(KYTY_RFPATCH_MUTANT_COND_AS_TAKEN)
	return kKnownState; // mutant: the stale dword reads as the known value (one successor, the stale one's)
#else
	return kDeferredState;
#endif
}

// The successors a block with an unknown (or empty) condition reaches. Used by the walker compiler for every such block.
template <class T>
inline std::span<const T> SuccessorsWhenUnknown(std::span<const T> successors) {
#if defined(KYTY_RFPATCH_MUTANT_SKIP_SUCCESSOR)
	return successors.size() > 1 ? successors.first(1) : successors;
#else
	return successors;
#endif
}

enum class RootKind : uint8_t { Condition, Descriptor, Output };
struct Root {
	uint32_t node = UINT32_MAX;
	RootKind kind = RootKind::Condition;
};

enum class SupersetWhy : uint8_t { Superset, NotLoad, NoLoad, Dead, Loop, Address, Descriptor, Output, DerivedFlat, TooManySlots, PhiAddress };

inline const char* SupersetWhyName(SupersetWhy why) {
	switch (why) {
		case SupersetWhy::Superset: return "superset";
		case SupersetWhy::NotLoad: return "not_a_load";
		case SupersetWhy::NoLoad: return "no_guest_load";
		case SupersetWhy::Dead: return "dead";
		case SupersetWhy::Loop: return "loop";
		case SupersetWhy::Address: return "address";
		case SupersetWhy::Descriptor: return "descriptor";
		case SupersetWhy::Output: return "output";
		case SupersetWhy::DerivedFlat: return "derived_flat";
		case SupersetWhy::TooManySlots: return "slots";
		case SupersetWhy::PhiAddress: return "phi_address";
	}
	return "?";
}

// `nodes`: the walker graph with RUNTIME deps (a node's operands when it evaluates itself, the node it folded into, nothing when it folded to a
// constant or unknown). `roots`: what the CPU reads after the walk, with its kind. `flat`: per flat read slot the node it stores (UINT32_MAX none).
// Result: per node, whether a fault of its guest load may defer as unknown, or why not.
inline std::vector<SupersetWhy> SupersetLoads(std::span<const GraphNode> nodes, std::span<const Root> roots, std::span<const uint32_t> flat) {
	const size_t n = nodes.size();
	// Live: needed by a root or a flat slot (the walker only runs thunks something it reads calls).
	std::vector<bool>     live(n, false);
	std::vector<uint32_t> stack;
	for (const auto& r: roots) {
		if (r.node < n) stack.push_back(r.node);
	}
	for (const auto f: flat) {
		if (f < n) stack.push_back(f);
	}
	while (!stack.empty()) {
		const auto i = stack.back();
		stack.pop_back();
		if (live[i]) continue;
		live[i] = true;
		for (const auto d: nodes[i].deps) {
			if (d < n && !live[d]) stack.push_back(d);
		}
	}
	std::vector<std::vector<uint32_t>> users(n);
	for (uint32_t i = 0; i < n; ++i) {
		if (!live[i]) continue;
		for (const auto d: nodes[i].deps) {
			if (d < n && d != i && (users[d].empty() || users[d].back() != i)) users[d].push_back(i);
		}
	}
	std::vector<bool>     cond(n, false), desc(n, false), out(n, false);
	std::vector<uint32_t> slots(n, 0);
	for (const auto& r: roots) {
		if (r.node >= n) continue;
		(r.kind == RootKind::Condition ? cond : r.kind == RootKind::Descriptor ? desc : out)[r.node] = true;
	}
	for (const auto f: flat) {
		if (f < n) ++slots[f];
	}
	std::vector<SupersetWhy> result(n, SupersetWhy::NotLoad);
	std::vector<uint8_t>     mark(n, 0);
	for (uint32_t l = 0; l < n; ++l) {
		const auto& node = nodes[l];
		auto&       why  = result[l];
		if (!node.read) continue;
		if (!node.emits_load) {
			why = SupersetWhy::NoLoad;
			continue;
		}
		if (!live[l]) {
			why = SupersetWhy::Dead;
			continue;
		}
		// T: every live node computed from the load (forward); A: what its address is computed from (backward).
		std::fill(mark.begin(), mark.end(), uint8_t {0});
		std::vector<uint32_t> forward, backward;
		bool                  self_loop = false; // the load's value reaches its own address (or itself)
		for (const auto u: users[l]) stack.push_back(u);
		while (!stack.empty()) {
			const auto i = stack.back();
			stack.pop_back();
			if (i == l) self_loop = true;
			if ((mark[i] & 1u) != 0 || i == l) continue;
			mark[i] |= 1u;
			forward.push_back(i);
			for (const auto u: users[i]) stack.push_back(u);
		}
		for (const auto d: node.deps) {
			if (d < n) stack.push_back(d);
		}
		while (!stack.empty()) {
			const auto i = stack.back();
			stack.pop_back();
			if ((mark[i] & 2u) != 0) continue;
			mark[i] |= 2u;
			backward.push_back(i);
			for (const auto d: nodes[i].deps) {
				if (d < n) stack.push_back(d);
			}
		}
		const auto any = [](const std::vector<uint32_t>& set, const auto& pred) { return std::any_of(set.begin(), set.end(), pred); };
		bool loop = node.cyclic || self_loop || any(forward, [&](uint32_t i) { return nodes[i].cyclic; }) ||
		            any(backward, [&](uint32_t i) { return nodes[i].cyclic; });
#if defined(KYTY_RFPATCH_MUTANT_DEFER_LOOP)
		loop = false; // mutant: a loop-carried load defers
#endif
		bool address = any(forward, [&](uint32_t i) { return nodes[i].read; });
#if defined(KYTY_RFPATCH_MUTANT_DEFER_ADDRESS)
		address = false; // mutant: a load that reaches another load's address defers
#endif
		bool phi_address = slots[l] != 0 && any(backward, [&](uint32_t i) { return nodes[i].control_phi; });
#if defined(KYTY_RFPATCH_MUTANT_CPHI_ADDRESS)
		phi_address = false; // mutant: K1 ignored for a load with flat slots
#endif
		if (loop) why = SupersetWhy::Loop;
		else if (address) why = SupersetWhy::Address;
		else if (desc[l] || any(forward, [&](uint32_t i) { return desc[i]; })) why = SupersetWhy::Descriptor;
		else if (out[l] || any(forward, [&](uint32_t i) { return out[i]; })) why = SupersetWhy::Output;
		else if (any(forward, [&](uint32_t i) { return slots[i] != 0; })) why = SupersetWhy::DerivedFlat;
		else if (slots[l] > kMaxSlots) why = SupersetWhy::TooManySlots;
		else if (phi_address) why = SupersetWhy::PhiAddress;
		else why = SupersetWhy::Superset;
	}
	return result;
}

// The flat slots a deferred load's pending entry carries (the GPU copies the bytes into them).
inline std::span<const uint32_t> SupersetEntrySlots(std::span<const uint32_t> slots) {
#if defined(KYTY_RFPATCH_MUTANT_NO_FLAT_COPY)
	return slots.first(0); // mutant: no copy, the slot keeps the stale dword
#else
	return slots;
#endif
}

// ---- fault decision in superset mode (exception handler, CP thread) ---------------------------------------------------------------------------

struct SupersetQuery {
	bool        armed        = false; // inside a Scope on this thread
	bool        gpu_thread   = false;
	bool        site         = false; // a registered load site of the running resource walker
	SupersetWhy why          = SupersetWhy::NotLoad;
	uint32_t    slots        = 0;
	bool        address_base = false;
	bool        queried      = false;
	bool        dirty        = false;
	bool        inflight     = false;
	bool        room         = false;
	bool        disabled     = false; // the scope retries a failed superset walk with the drain
	bool        inflight_ok  = false; // S581 K-rf v3 (KYTY_KRF3=1|inflight): GPU-dirty bytes with an older download in flight defer too
};

// Why a fault does not defer. NotDirty / NotArmed: no wait anyway (or not ours), not a fallback. Static: the site's SupersetWhy.
enum class SupersetReject : uint8_t { None, NotArmed, NotDirty, Inflight, NotSite, Static, Retry, AddressBase, Full };

inline SupersetReject DecideSuperset(const SupersetQuery& q) {
	if (!q.armed || !q.gpu_thread) return SupersetReject::NotArmed;
	// Clean bytes need no wait (the ordinary fault path reads them); bytes in flight are about to change in the backing.
	if (!q.queried || !q.dirty) return SupersetReject::NotDirty;
#if defined(KYTY_RFPATCH_MUTANT_KRF3_INFLIGHT_DRAINS)
	if (q.inflight) return SupersetReject::Inflight; // mutant: v2's drain of re-dirtied bytes whose older download is in flight
#else
	// S581: the deferred value is never used (unknown for the walk; a flat slot gets the GPU buffer's bytes by the copy), so an older download
	// of the same bytes in flight does not matter. v2 drained here, and in the desert that drain waited ~22 ms (S580: 482 of 493 drains).
	if (q.inflight && !q.inflight_ok) return SupersetReject::Inflight;
#endif
	if (!q.site) return SupersetReject::NotSite;
	if (q.why != SupersetWhy::Superset) return SupersetReject::Static;
	if (q.disabled) return SupersetReject::Retry;
	if (q.slots != 0 && q.address_base) return SupersetReject::AddressBase;
	if (!q.room) return SupersetReject::Full;
	return SupersetReject::None;
}

// A superset walk (or its materialization) failed: when this scope deferred a load, the caller walks again with the drain (true).
inline bool SupersetRetry() {
	auto& p = ThreadPending();
	if (!p.armed || p.superset_defers == 0) return false;
	p.superset_off    = true;
	p.superset_defers = 0;
	p.Clear();
	return true;
}

// ---- superset_shadow: the true walk (fresh bytes) against the superset walk -------------------------------------------------------------------

struct Cover {
	bool     agree = true;
	uint32_t extra = 0; // resources / words the superset prepares that the true walk does not
};

// R: a descriptor (dwords, dword_count). The true walk's resource is all zero when it is not active (unreached block) or null; the superset's
// must be equal wherever the true one is not zero.
template <class R>
inline Cover SupersetCoverResources(std::span<const R> truth, std::span<const R> superset, std::vector<uint32_t>* extra_indices = nullptr) {
	Cover c;
	if (truth.size() != superset.size()) return {.agree = false};
	size_t end = truth.size();
#if defined(KYTY_RFPATCH_MUTANT_COVER_MISSES_ONE)
	if (end != 0) --end; // mutant: the last resource is not checked
#endif
	const auto zero = [](const R& r) { return std::all_of(r.dwords.begin(), r.dwords.end(), [](uint32_t w) { return w == 0; }); };
	for (size_t i = 0; i < end; ++i) {
		if (truth[i] == superset[i]) continue;
		if (zero(truth[i]) && truth[i].dword_count == superset[i].dword_count) {
			++c.extra;
			if (extra_indices != nullptr) extra_indices->push_back(static_cast<uint32_t>(i));
			continue;
		}
		c.agree = false;
	}
	return c;
}

// Flat words: equal, or zero in the true walk (a slot of a block it does not reach). A patched slot must hold the drained dword in the true walk,
// or zero there (a slot of the load in a block the true walk does not reach; the copy writes the real dword into it, which nothing reads); the
// superset's has the stale one until the GPU copy (checked on the GPU).
inline Cover SupersetCoverFlat(std::span<const uint32_t> truth, std::span<const uint32_t> superset, const Pending& pending,
                               std::span<const uint32_t> fresh) {
	Cover c;
	if (truth.size() != superset.size()) return {.agree = false};
	std::vector<int32_t> patched(truth.size(), -1);
	for (uint32_t i = 0; i < pending.count; ++i) {
		const auto& e = pending.entries[i];
		for (uint32_t k = 0; k < e.slot_count; ++k) {
			if (e.slots[k] < truth.size()) patched[e.slots[k]] = static_cast<int32_t>(i);
		}
	}
	for (size_t i = 0; i < truth.size(); ++i) {
		if (patched[i] >= 0) {
			if (static_cast<size_t>(patched[i]) >= fresh.size() || (truth[i] != 0 && truth[i] != fresh[static_cast<size_t>(patched[i])]))
				c.agree = false;
			continue;
		}
		if (truth[i] == superset[i]) continue;
		if (truth[i] == 0) {
			++c.extra;
			continue;
		}
		c.agree = false;
	}
	return c;
}

// Every range the true walk read is read by the superset walk too.
template <class P>
inline bool SupersetCoverRanges(std::span<const P> truth, std::span<const P> superset) {
	return std::all_of(truth.begin(), truth.end(),
	                   [&](const P& r) { return std::find(superset.begin(), superset.end(), r) != superset.end(); });
}

// ==== S581 K-rf v3 (KYTY_KRF3=0|1|inflight|page, only with KYTY_RF_GPU_PATCH=superset | superset_shadow) =====================================
// S580: the superset keeps its garden gain (-3.3 / -3.0 ms cp_frame) but the desert is vetoed (hitches > 33 ms 5 -> 67 and 4 -> 40). Per flip of
// the desert CSVs (S581 notes): the drain of the deferred dword did two jobs. (1) It made the CPU wait for the value (the cost K-rf removes).
// (2) It downloaded the dword's 4 KiB page, so the guest's later reads of that page did not fault. Without it the page stays GPU-dirty: guest
// read faults rise 1.3 -> 3.0 per flip (+1.7 ~ the 1.8 drains removed; the rise follows the defers flip bucket by bucket and is gone in the
// buckets where the guest stops reading, while the defers go on) and each waits 8-10 ms instead of 2 (readback_wait of guest threads 2.6 ->
// 30 ms/flip). The page is hot, so every GPU write to it queues an eager download; a walker fault on bytes re-dirtied after such a download
// (dirty + in flight) was rejected by v2 and drained, ~22 ms each (0.2 per flip), and 58 of 67 / 38 of 40 hitch rows have one within 2 rows.
// superset_shadow (which drains the deferred dwords) and the base have neither.
//   inflight: such a fault defers as well (DecideSuperset, inflight_ok). Safe: the stale dword is unknown for the walk, a flat slot gets the GPU
//             buffer's bytes by the copy, and the download in flight only writes the backing.
//   page:     the guest read faults on the page of a deferred dword are counted (GuestPageGuard::NoteGuestRead, guest threads); per window of
//             `window` defers of one page, at least one guest fault per `hit_div` defers bans deferring on that page for `ban` flips (the walker
//             drains there as in the base, which also cleans the page for the guest), doubling up to `ban_max` while trials (`trial_window`
//             defers) keep failing and back to `ban_min` after a clean trial window. The garden's pages see ~0.1 guest faults per 6 defers per flip; the desert's ~3 per 2.6.
//   1 = inflight + page. inflight alone still tracks the pages and counts what page would do (krf3_page_would_ban_n) without banning.
// S589 sticky (KYTY_KRF3=sticky = inflight + page with GuestPageGuard::StickyConfig()): S588 found the v3 guard bistable on the desert (a ban in
// 4 of 5 runs, 163 hitches without it; kr3 alone 21 vs 111). Per S588 CSV: the desert page sees 0.3-0.5 guest faults per flip at ~3 defers
// per flip (0.11-0.16 per defer; 1.1 in the landing burst), the garden 0.002-0.005 per defer (an entry burst of at most 8 faults in 64 flips,
// then ~0.01 per flip). v3 asks for 1 fault per 4 defers in a window of 32 (8 faults in ~11 flips): above the desert's steady rate, so a ban
// came only from bursts, lasted 128 flips, and a trial of 8 defers (~3 flips, before the guest faults again) usually passed. sticky: 1 fault
// per 16 defers in a window of 256 (16 faults), decided as soon as the hits reach it (~50 flips on the desert, the garden stays at <= 8);
// a ban lasts 4096 flips and doubles up to 65536, a trial is a full window, and a clean trial halves the next ban (min 4096).
// S599 v4 (KYTY_KRF3=v4 = inflight + page with GuestPageGuard::V4Config()): a ban only when the CP's lead hurts. S597/S599 on the S590-S596 CSVs:
// sticky banned the garden page on storms of guest reads (16 hits in 8-15 flips, sv1 S595 at 9832 and 18445: 45 % of the garden's flips in a ban,
// +5.6 ms each) while the CP did not starve (cp_wait_guest ~0); on the desert the ban's delay sets the landing tail (ban after 847-1634 flips:
// 30-38 frames > 100 ms), and the landing's harm is the CP starving (cp_wait_guest >= 20 % of a 16-flip window from ~77 flips after the first
// defer). v4 bans a page (A, why=burst) when its guest hits since its first defer reach 16 while it is younger than 128 flips (sticky's early
// rule inside that window: the desert's 7/25 early bans, the garden entry has <= 9), or (B, why=starve) when, at least 64 flips after the first
// defer, the CP waited on the guest for >= 20 % of the last 16 flips, those flips average >= 25 ms and the page deferred in them. Later bursts
// without starvation never ban. The signals are the guard's own: hits (NoteGuestRead), defers (OnDefer) and per flip the CP's wait on the
// guest (Krf4FlipLog: an accumulator beside FrameTiming's CpWaitGuestNs, closed per guest flip with the flip's duration), not frames.csv.
// Offline simulator (artifacts/S599/sim.py): garden 0 bans in 30 runs (max window share 0.061 vs 0.2), desert ban <= min(sticky, +120) in 22/25,
// never later than sticky in any ON run; the same with cw shifted by +-2 flips.
struct Krf3Mode {
	bool inflight = false;
	bool page     = false;
	bool sticky   = false; // S589: the page guard uses GuestPageGuard::StickyConfig()
	bool v4       = false; // S599: the page guard uses GuestPageGuard::V4Config()
	[[nodiscard]] bool Any() const { return inflight || page; }
};

inline Krf3Mode ParseKrf3(const char* text) {
	Krf3Mode m;
	if (text == nullptr) return m;
	if (std::strcmp(text, "1") == 0) {
		m.inflight = true;
		m.page     = true;
	} else if (std::strcmp(text, "inflight") == 0) {
		m.inflight = true;
	} else if (std::strcmp(text, "page") == 0) {
		m.page = true;
	} else if (std::strcmp(text, "sticky") == 0) {
		m.inflight = true;
		m.page     = true;
		m.sticky   = true;
	} else if (std::strcmp(text, "v4") == 0) {
		m.inflight = true;
		m.page     = true;
		m.v4       = true;
	}
	return m;
}

inline Krf3Mode Krf3EnvMode() {
	static const Krf3Mode mode = ParseKrf3(std::getenv("KYTY_KRF3"));
	return mode;
}

// S599 v4: the CP's wait on the guest per guest flip, for the page guard (not FrameTiming: frames.csv may be off in the game, the CP front
// thread's FrameTiming sums are buffered, a muted thread adds nothing). AddCpWait on the CP thread (the CpWaitGuestNs scopes, Krf4WaitScope),
// Close on the presenter thread once per guest flip (FrameTiming::OnGuestFlip, after the flip count moved), Drain on the CP thread (one reader).
// A ring of the last kRing closed flips; an entry is valid while its flip field equals the flip read (written last, a seqlock on the flip).
class Krf4FlipLog {
public:
	static constexpr uint32_t kRing = 64;

	void AddCpWait(uint64_t ns) { m_cw.fetch_add(ns, std::memory_order_relaxed); }

	// Flip `flip` (the count after this flip) ended at `now_ns`. The first close only starts the clock (a flip without a start has no duration).
	void Close(uint64_t flip, int64_t now_ns) {
		const int64_t  last = m_last_close.exchange(now_ns, std::memory_order_relaxed);
		const uint64_t cw   = m_cw.exchange(0, std::memory_order_relaxed);
		if (last == 0 || now_ns <= last || flip == 0) return;
		auto& e = m_ring[flip % kRing];
		e.flip.store(0, std::memory_order_relaxed);
		std::atomic_thread_fence(std::memory_order_release);
		e.host_ns.store(static_cast<uint64_t>(now_ns - last), std::memory_order_relaxed);
		e.cw_ns.store(cw, std::memory_order_relaxed);
		e.flip.store(flip, std::memory_order_release);
		m_head.store(flip, std::memory_order_release);
	}

	// CP thread: f(flip, host_ns, cw_ns) for every flip closed since the last drain, oldest first (at most kRing; overwritten entries skipped).
	template <class F>
	uint32_t Drain(F&& f) {
		const uint64_t head = m_head.load(std::memory_order_acquire);
		if (head <= m_read) return 0;
		uint64_t from = m_read + 1;
		if (head - m_read > kRing) from = head - kRing + 1;
		uint32_t n = 0;
		for (uint64_t flip = from; flip <= head; ++flip) {
			auto&          e    = m_ring[flip % kRing];
			const uint64_t seen = e.flip.load(std::memory_order_acquire);
			const uint64_t host = e.host_ns.load(std::memory_order_relaxed);
			const uint64_t cw   = e.cw_ns.load(std::memory_order_relaxed);
			std::atomic_thread_fence(std::memory_order_acquire);
			if (seen != flip || e.flip.load(std::memory_order_relaxed) != flip) continue; // not written (first close) or overwritten meanwhile
			f(flip, host, cw);
			++n;
		}
		m_read = head;
		return n;
	}

	static int64_t NowNs() {
		return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	}

private:
	struct Entry {
		std::atomic<uint64_t> flip {0};
		std::atomic<uint64_t> host_ns {0};
		std::atomic<uint64_t> cw_ns {0};
	};
	std::atomic<uint64_t>        m_cw {0};
	std::atomic<int64_t>         m_last_close {0};
	std::atomic<uint64_t>        m_head {0};
	uint64_t                     m_read = 0; // the CP's (Drain's) own
	std::array<Entry, kRing>     m_ring {};
};

// Pages of deferred dwords and the guest's read faults on them. OnDefer runs on the CP thread (the walker's fault handler), NoteGuestRead on any
// guest thread (its read fault handler): the page key and the hit count are atomics, the rest belongs to the CP thread (a second CP thread that
// finds the guard busy defers without accounting). S599 v4: OnFlipClosed (CP thread, once per closed guest flip) feeds the starvation window.
class GuestPageGuard {
public:
	static constexpr uint32_t kEntries  = 16;
	static constexpr uint64_t kPageSize = 4096;
	static constexpr uint32_t kStarveWindow = 16; // S599 v4: flips of the starvation window

	struct Config {
		uint32_t window  = 32;   // defers of one page per decision
		uint32_t trial_window = 8; // ... after a ban ran out (a trial decides fast: a desert page costs ~8 defers, not 32)
		uint32_t hit_div = 4;    // ban when hits * hit_div >= defers of the window
		uint64_t ban_min = 128;  // flips
		uint64_t ban_max = 4096; // flips
		bool     early   = false; // S589: ban as soon as hits * hit_div reach the window (not only when the window closes)
		bool     halve   = false; // S589: a clean trial halves the ban length (not below ban_min) instead of resetting it to ban_min
		// S599 v4 (the fields above but ban_min / ban_max / halve are unused then).
		bool     v4             = false;
		uint32_t burst_hits     = 16;         // A: hits since the page's first defer ...
		uint64_t burst_window   = 128;        // ... while it is younger than this (flips)
		uint64_t grace          = 64;         // B: not before this age (flips)
		uint32_t starve_pct     = 20;         // B: CP wait on the guest >= this % of the window's time ...
		uint64_t starve_mean_ns = 25000000;   // ... and the window's flips average >= this
	};

	// S589 KYTY_KRF3=sticky (see Krf3Mode).
	static Config StickyConfig() {
		Config c;
		c.window       = 256;
		c.trial_window = 256;
#if defined(KYTY_RFPATCH_MUTANT_KRF3_STICKY_OLD_RATIO)
		c.hit_div = 4; // mutant: the v3 ratio, above the desert's steady 0.11-0.16 faults per defer
#elif defined(KYTY_RFPATCH_MUTANT_KRF3_STICKY_LOW_RATIO)
		c.hit_div = 64; // mutant: 4 faults per window, the garden's entry burst bans
#else
		c.hit_div = 16;
#endif
#if defined(KYTY_RFPATCH_MUTANT_KRF3_STICKY_SHORT_BAN)
		c.ban_min = 128; // mutant: the v3 ban length, trials inside the desert window
#else
		c.ban_min = 4096;
#endif
		c.ban_max = 65536;
#if defined(KYTY_RFPATCH_MUTANT_KRF3_STICKY_NO_EARLY)
		c.early = false; // mutant: a ban only when the window closes (~85 flips at 3 defers per flip)
#else
		c.early = true;
#endif
		c.halve = true;
		return c;
	}

	// S599 KYTY_KRF3=v4 (see Krf3Mode): sticky's ban lengths, the v4 rules.
	static Config V4Config() {
		Config c;
		c.window       = 256;
		c.trial_window = 256;
		c.hit_div      = 16;
		c.ban_min      = 4096;
		c.ban_max      = 65536;
		c.early        = true;
		c.halve        = true;
		c.v4           = true;
#if defined(KYTY_RFPATCH_MUTANT_KRF4_BURST_ANYTIME)
		c.burst_window = UINT64_MAX; // mutant: A at any age, a garden storm (16 hits in 8-15 flips) bans
#endif
#if defined(KYTY_RFPATCH_MUTANT_KRF4_NO_GRACE)
		c.grace = 0; // mutant: B from the first defer, a level entry that loads (the CP waits on the guest) bans
#endif
		return c;
	}

	enum class Why : uint8_t { None, Burst, Starve };

	struct Result {
		bool     drain      = false; // the walker drains this fault (the page is banned, or was banned just now)
		bool     banned_now = false; // this defer closed a window that bans the page
		bool     would_ban  = false; // the same, with banning off (inflight only): counted, the defer goes on
		bool     trial      = false; // a ban ran out: deferring on the page is tried again
		uint32_t hits       = 0;     // banned_now / would_ban: the guest faults and defers of the deciding window, the ban length (S589 log)
		uint32_t defers     = 0;
		uint64_t ban_len    = 0;
		Why      why        = Why::None; // S599 v4: banned_now / would_ban by which rule, the page's age, the starvation window (B)
		uint64_t age        = 0;
		uint64_t window_cw_ns   = 0;
		uint64_t window_host_ns = 0;
	};

	// S599 v4: one closed guest flip.
	struct FlipResult {
		bool     starved = false; // the window of the last kStarveWindow flips is starved (B's condition without the page's part)
		uint32_t storms  = 0;     // pages (not banned) the guest faulted on >= 2 times since the previous closed flip
		uint32_t pending = 0;     // pages that will be banned at their next defer
	};

	GuestPageGuard() = default;
	explicit GuestPageGuard(Config c): m_config(c) {}
	[[nodiscard]] const Config& GetConfig() const { return m_config; }

	static uint64_t PageOf(uint64_t address) {
#if defined(KYTY_RFPATCH_MUTANT_KRF3_ADDRESS_KEY)
		return address; // mutant: keyed by the dword, a guest read of another byte of the page is not seen
#else
		return address & ~(kPageSize - 1);
#endif
	}

	// CP thread: a superset defer of `address` at guest flip `flip`. may_ban = false only counts (would_ban).
	Result OnDefer(uint64_t address, uint64_t flip, bool may_ban) {
		Result r;
		if (m_busy.test_and_set(std::memory_order_acquire)) return r;
		const auto page = PageOf(address);
		auto*      s    = Find(page);
		const bool fresh = s == nullptr;
		if (fresh) s = Allocate(page);
		if (m_config.v4) {
			OnDeferV4(*s, fresh, flip, may_ban, r);
			m_busy.clear(std::memory_order_release);
			return r;
		}
		s->last_flip = flip;
		if (s->banned) {
#if defined(KYTY_RFPATCH_MUTANT_KRF3_NO_EXPIRY)
			const bool expired = false; // mutant: a ban never runs out (a page the guest stopped reading is lost for good)
#else
			const bool expired = flip >= s->ban_until;
#endif
			if (!expired) {
				r.drain = true;
				m_busy.clear(std::memory_order_release);
				return r;
			}
			s->banned = false;
			s->trial  = true;
			s->defers = 0;
			s->hits.store(0, std::memory_order_relaxed);
			r.trial = true;
		}
		const uint32_t window = s->trial ? m_config.trial_window : m_config.window;
		++s->defers;
		// S589 (early): the window's hits so far already reach the ratio of a whole window.
		const bool early = m_config.early && uint64_t {s->hits.load(std::memory_order_relaxed)} * m_config.hit_div >= window;
		if (early || s->defers >= window) {
			const uint32_t hits   = s->hits.exchange(0, std::memory_order_relaxed);
			const uint32_t defers = s->defers;
			s->defers             = 0;
			r.hits                = hits;
			r.defers              = defers;
			if (early || uint64_t {hits} * m_config.hit_div >= defers) {
				if (may_ban) {
					s->banned    = true;
					s->ban_until = flip + s->ban_len;
					r.ban_len    = s->ban_len;
					s->ban_len   = std::min(s->ban_len * 2, m_config.ban_max);
					s->trial     = false;
					r.drain      = true;
					r.banned_now = true;
				} else {
					r.would_ban = true;
				}
			} else if (s->trial) {
				s->trial   = false;
				s->ban_len = m_config.halve ? std::max(s->ban_len / 2, m_config.ban_min) : m_config.ban_min;
			}
		}
		m_busy.clear(std::memory_order_release);
		return r;
	}

	// S599 v4, CP thread: guest flip `flip` closed, it lasted host_ns and the CP waited on the guest for cw_ns of it. Moves the starvation window,
	// counts storms, and marks the pages that A (an edge: hits that reached burst_hits just before the window ended) or B ban at their next
	// defer. Without v4 it does nothing (sticky and v3 never call it in the emulator; the tests check it changes nothing for them).
	FlipResult OnFlipClosed(uint64_t flip, uint64_t host_ns, uint64_t cw_ns) {
		FlipResult fr;
		if (!m_config.v4) return fr;
		if (m_busy.test_and_set(std::memory_order_acquire)) return fr;
		auto& w = m_window[m_window_next];
		m_window_host += host_ns;
		m_window_cw += cw_ns;
		if (m_window_n == kStarveWindow) {
			m_window_host -= w.host_ns;
			m_window_cw -= w.cw_ns;
		} else {
			++m_window_n;
		}
		w             = {host_ns, cw_ns};
		m_window_next = (m_window_next + 1) % kStarveWindow;
		fr.starved    = WindowStarved();
		for (auto& s: m_slots) {
			if (s.page.load(std::memory_order_relaxed) == 0) continue;
			const uint32_t hits = s.hits.load(std::memory_order_relaxed);
			const uint32_t dh   = hits >= s.hits_seen ? hits - s.hits_seen : hits;
			s.hits_seen         = hits;
			if (s.banned) continue;
			if (dh >= 2) ++fr.storms;
			if (flip < s.first_defer) continue;
			const uint64_t age = flip - s.first_defer;
			if (s.trial && age >= m_config.burst_window) { // S599: a trial that got past the burst window without a ban is clean
				s.trial   = false;
				s.ban_len = std::max(s.ban_len / 2, m_config.ban_min);
			}
			if (s.pending != Why::None) {
				++fr.pending;
				continue;
			}
			Why why = Why::None;
			if (BurstAge(age) && hits >= m_config.burst_hits) {
				why = Why::Burst;
			} else if (fr.starved && age >= m_config.grace && s.last_flip + kStarveWindow >= flip) {
#if !defined(KYTY_RFPATCH_MUTANT_KRF4_STARVE_IGNORED)
				why = Why::Starve; // (mutant: B never bans, the desert waits for its burst)
#endif
			}
			if (why != Why::None) {
				s.pending          = why;
				s.pending_flip     = flip;
				s.pending_cw_ns    = m_window_cw;
				s.pending_host_ns  = m_window_host;
				++fr.pending;
			}
		}
		m_busy.clear(std::memory_order_release);
		return fr;
	}

	// Any guest thread, its read fault on `address`: true (and one hit) when the page holds a deferred dword.
	bool NoteGuestRead(uint64_t address) {
		const auto page = PageOf(address);
		for (auto& s: m_slots) {
			if (s.page.load(std::memory_order_acquire) != page) continue;
#if !defined(KYTY_RFPATCH_MUTANT_KRF3_NO_GUEST_HITS)
			s.hits.fetch_add(1, std::memory_order_relaxed);
#endif
			return true;
		}
		return false;
	}

	[[nodiscard]] bool Banned(uint64_t address, uint64_t flip) {
		const auto* s = Find(PageOf(address));
		return s != nullptr && s->banned && flip < s->ban_until;
	}

private:
	struct Slot {
		std::atomic<uint64_t> page {0}; // 0 = free (a deferred dword is never at guest address 0)
		std::atomic<uint32_t> hits {0};
		uint32_t              defers    = 0;
		uint64_t              ban_until = 0;
		uint64_t              ban_len   = 0;
		uint64_t              last_flip = 0;
		bool                  banned    = false;
		bool                  trial     = false;
		// S599 v4
		uint64_t first_defer     = 0; // flip of the page's first defer (of its trial after a ban)
		uint32_t hits_seen       = 0; // hits at the previous closed flip
		Why      pending         = Why::None;
		uint64_t pending_flip    = 0;
		uint64_t pending_cw_ns   = 0;
		uint64_t pending_host_ns = 0;
	};
	struct WindowFlip {
		uint64_t host_ns = 0;
		uint64_t cw_ns   = 0;
	};

	[[nodiscard]] bool BurstAge(uint64_t age) const {
#if defined(KYTY_RFPATCH_MUTANT_KRF4_BURST_ANYTIME)
		(void)age;
		return true;
#else
		return age < m_config.burst_window;
#endif
	}

	[[nodiscard]] bool WindowStarved() const {
		if (m_window_n < kStarveWindow) return false;
#if defined(KYTY_RFPATCH_MUTANT_KRF4_STARVE_ANY_FLIP)
		// mutant: any flip of the window with a CP wait on the guest starves it (a garden storm's 3-8 ms waits ban)
		return std::any_of(m_window.begin(), m_window.end(), [](const WindowFlip& f) { return f.cw_ns > 0; }) &&
		       m_window_host >= m_config.starve_mean_ns * kStarveWindow;
#else
		return m_window_cw * 100 >= m_window_host * m_config.starve_pct && m_window_host >= m_config.starve_mean_ns * kStarveWindow;
#endif
	}

	// Under m_busy.
	void OnDeferV4(Slot& s, bool fresh, uint64_t flip, bool may_ban, Result& r) {
		if (fresh) s.first_defer = flip;
		s.last_flip = flip;
		if (s.banned) {
#if defined(KYTY_RFPATCH_MUTANT_KRF3_NO_EXPIRY)
			const bool expired = false;
#else
			const bool expired = flip >= s.ban_until;
#endif
			if (!expired) {
				r.drain = true;
				return;
			}
			// The trial: the page starts again as if first deferred now (A's window and B's grace).
			s.banned      = false;
			s.trial       = true;
			s.defers      = 0;
			s.first_defer = flip;
			s.pending     = Why::None;
			s.hits.store(0, std::memory_order_relaxed);
			s.hits_seen = 0;
			r.trial     = true;
		}
		++s.defers;
		const uint64_t age  = flip >= s.first_defer ? flip - s.first_defer : 0;
		const uint32_t hits = s.hits.load(std::memory_order_relaxed);
		Why            why  = Why::None;
		if (BurstAge(age) && hits >= m_config.burst_hits) {
			why = Why::Burst;
		} else if (s.pending != Why::None && flip <= s.pending_flip + kStarveWindow) {
			why = s.pending; // a pending mark older than the window (the page stopped deferring) is dropped
		}
		if (s.pending != Why::None && why == Why::None) s.pending = Why::None;
		if (why == Why::None) return;
		r.why            = why;
		r.hits           = hits;
		r.defers         = s.defers;
		r.age            = age;
		r.window_cw_ns   = why == Why::Starve ? s.pending_cw_ns : 0;
		r.window_host_ns = why == Why::Starve ? s.pending_host_ns : 0;
		if (!may_ban) {
			r.would_ban = true;
			s.pending   = Why::None;
			return;
		}
		s.banned     = true;
		s.ban_until  = flip + s.ban_len;
		r.ban_len    = s.ban_len;
		s.ban_len    = std::min(s.ban_len * 2, m_config.ban_max);
		s.trial      = false;
		s.pending    = Why::None;
		r.drain      = true;
		r.banned_now = true;
	}

	Slot* Find(uint64_t page) {
		for (auto& s: m_slots) {
			if (s.page.load(std::memory_order_relaxed) == page) return &s;
		}
		return nullptr;
	}
	// A free slot, else the least recently deferred one that is not banned, else the least recently deferred one.
	Slot* Allocate(uint64_t page) {
		Slot* best = nullptr;
		for (auto& s: m_slots) {
			if (s.page.load(std::memory_order_relaxed) == 0) {
				best = &s;
				break;
			}
			if (best == nullptr || (best->banned && !s.banned) || (best->banned == s.banned && s.last_flip < best->last_flip)) best = &s;
		}
		best->page.store(0, std::memory_order_relaxed);
		best->hits.store(0, std::memory_order_relaxed);
		best->defers      = 0;
		best->ban_until   = 0;
		best->ban_len     = m_config.ban_min;
		best->banned      = false;
		best->trial       = false;
		best->first_defer = 0;
		best->hits_seen   = 0;
		best->pending     = Why::None;
		best->page.store(page, std::memory_order_release);
		return best;
	}

	Config                       m_config {};
	std::array<Slot, kEntries>   m_slots {};
	std::atomic_flag             m_busy = ATOMIC_FLAG_INIT;
	// S599 v4: the starvation window (CP thread, under m_busy)
	std::array<WindowFlip, kStarveWindow> m_window {};
	uint32_t                              m_window_n    = 0;
	uint32_t                              m_window_next = 0;
	uint64_t                              m_window_host = 0;
	uint64_t                              m_window_cw   = 0;
};

inline GuestPageGuard& GuestGuard() {
	static GuestPageGuard guard(Krf3EnvMode().v4       ? GuestPageGuard::V4Config()
	                            : Krf3EnvMode().sticky ? GuestPageGuard::StickyConfig()
	                                                   : GuestPageGuard::Config {}); // S589, S599
	return guard;
}

// S599 v4: the process's flip log (CP wait on the guest per guest flip).
inline Krf4FlipLog& Krf4Flips() {
	static Krf4FlipLog log;
	return log;
}

// S599 v4: the CP thread feeds the guard every flip closed since its last call; f(FlipResult) per flip (the caller counts).
template <class F>
inline uint32_t Krf4DrainFlips(GuestPageGuard& guard, Krf4FlipLog& log, F&& f) {
	return log.Drain([&](uint64_t flip, uint64_t host_ns, uint64_t cw_ns) { f(guard.OnFlipClosed(flip, host_ns, cw_ns)); });
}

// S599 v4: beside a FrameTiming::ScopedCounterNs(CpWaitGuestNs): the same wait into the flip log, with KYTY_KRF3=v4 only.
class Krf4WaitScope {
public:
	Krf4WaitScope(): Krf4WaitScope(Krf4Flips(), Krf3EnvMode().v4) {}
	Krf4WaitScope(Krf4FlipLog& log, bool on): m_log(log), m_begin(on ? Krf4FlipLog::NowNs() : 0) {}
	~Krf4WaitScope() {
		if (m_begin != 0) {
			const int64_t end = Krf4FlipLog::NowNs();
			if (end > m_begin) m_log.AddCpWait(static_cast<uint64_t>(end - m_begin));
		}
	}
	Krf4WaitScope(const Krf4WaitScope&)            = delete;
	Krf4WaitScope& operator=(const Krf4WaitScope&) = delete;

private:
	Krf4FlipLog& m_log;
	int64_t      m_begin;
};

inline const char* Krf3ModeName(Krf3Mode m) {
	if (m.v4) return "v4";
	if (m.sticky) return "sticky";
	if (m.inflight && m.page) return "1";
	if (m.inflight) return "inflight";
	if (m.page) return "page";
	return "0";
}

inline const char* Krf4WhyName(GuestPageGuard::Why why) {
	switch (why) {
		case GuestPageGuard::Why::Burst: return "burst";
		case GuestPageGuard::Why::Starve: return "starve";
		default: return "none";
	}
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::RfGpuPatch

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RFGPUPATCH_H_ */
