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
enum class Mode : uint8_t { Off, On, Shadow, Oracle };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	if (std::strcmp(text, "oracle") == 0) return Mode::Oracle;
	return Mode::Off;
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
		p.armed = true;
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

} // namespace Libs::Graphics::ShaderRecompiler::IR::RfGpuPatch

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_RFGPUPATCH_H_ */
