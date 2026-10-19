#ifndef KYTY_GRAPHICS_PRESENTATION_CPPCSAMPLER_H_
#define KYTY_GRAPHICS_PRESENTATION_CPPCSAMPLER_H_

// S439 DIAG (KYTY_DIAG_CP_PC=1, with KYTY_DIAG_CP_SAMPLE=1): program counter of the command-processor thread while it is in the sh.walk site.
// The sampler thread (cpSampler.h) suspends the CP thread, reads RIP with thread_get_state(x86_THREAD_STATE64), resumes it and only then
// counts the PC; nothing is allocated, locked or symbolized while the CP thread is stopped. Symbols are resolved when a segment is dumped:
// the JIT registry (Xbyak buffers of the SRT walker, registered by CompileSrtWalker) first, then dladdr.
// Also here: CPU time of a thread (thread_info THREAD_BASIC_INFO) for the CPU-vs-wall columns.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <dlfcn.h>
#include <mach/mach.h>
#endif

namespace Libs::Graphics::CpSampler::Pc {

inline bool Enabled() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_DIAG_CP_PC");
		return text != nullptr && text[0] == '1';
	}();
	return on;
}

// ---- JIT registry: append-only, lock-free for the reader (the sampler thread). Newest registration wins on overlap (a freed buffer's address may be reused).
enum class JitKind : uint8_t { Tree = 1, Slots = 2 };

struct JitEntry {
	std::atomic<uint64_t> lo {0};
	std::atomic<uint64_t> hash {0};
	std::atomic<uint32_t> kind {0};
	std::atomic<uint64_t> size {0}; // written last, release: 0 = not published
};
constexpr size_t kJitCapacity = 4096;
inline std::array<JitEntry, kJitCapacity>& JitEntries() {
	static std::array<JitEntry, kJitCapacity> entries;
	return entries;
}
inline std::atomic<uint32_t>& JitCount() {
	static std::atomic<uint32_t> count {0};
	return count;
}
inline std::atomic<uint32_t>& JitDropped() {
	static std::atomic<uint32_t> dropped {0};
	return dropped;
}

inline void RegisterJit(const void* code, size_t size, JitKind kind, uint64_t hash) {
#ifndef KYTY_CPPC_MUTANT_NO_REGISTER
	if (code == nullptr || size == 0) return;
	const uint32_t index = JitCount().fetch_add(1, std::memory_order_relaxed);
	if (index >= kJitCapacity) {
		JitDropped().fetch_add(1, std::memory_order_relaxed);
		return;
	}
	auto& e = JitEntries()[index];
	e.lo.store(reinterpret_cast<uint64_t>(code), std::memory_order_relaxed);
	e.hash.store(hash, std::memory_order_relaxed);
	e.kind.store(static_cast<uint32_t>(kind), std::memory_order_relaxed);
	e.size.store(size, std::memory_order_release);
#else
	(void)code; (void)size; (void)kind; (void)hash;
#endif
}

// Finds the newest registration containing pc; false when none.
inline bool LookupJit(uint64_t pc, JitKind& kind, uint64_t& hash, uint64_t& offset) {
	const uint32_t count = std::min<uint32_t>(JitCount().load(std::memory_order_relaxed), static_cast<uint32_t>(kJitCapacity));
	for (uint32_t i = count; i-- > 0;) {
		auto&          e    = JitEntries()[i];
		const uint64_t size = e.size.load(std::memory_order_acquire);
		if (size == 0) continue;
		const uint64_t lo = e.lo.load(std::memory_order_relaxed);
		if (pc >= lo && pc - lo < size) {
			kind   = static_cast<JitKind>(e.kind.load(std::memory_order_relaxed));
			hash   = e.hash.load(std::memory_order_relaxed);
			offset = pc - lo;
			return true;
		}
	}
	return false;
}

// "jit:tree:<hash>+0x<off>", "jit:slots:<hash>+0x<off>", "<dli_sname>+0x<off>" / "<image>+0x<off>", or "?".
inline std::string Symbolize(uint64_t pc) {
	char          text[256];
	JitKind       kind;
	uint64_t      hash, offset;
	if (LookupJit(pc, kind, hash, offset)) {
		std::snprintf(text, sizeof(text), "jit:%s:%016llx+0x%llx", kind == JitKind::Tree ? "tree" : "slots", static_cast<unsigned long long>(hash),
		              static_cast<unsigned long long>(offset));
		return text;
	}
#if defined(__APPLE__)
	Dl_info info {};
	if (dladdr(reinterpret_cast<void*>(pc), &info) != 0) {
		if (info.dli_sname != nullptr) {
			std::snprintf(text, sizeof(text), "%s+0x%llx", info.dli_sname, static_cast<unsigned long long>(pc - reinterpret_cast<uint64_t>(info.dli_saddr)));
			return text;
		}
		if (info.dli_fname != nullptr) {
			const char* base = std::strrchr(info.dli_fname, '/');
			std::snprintf(text, sizeof(text), "%s+0x%llx", base != nullptr ? base + 1 : info.dli_fname,
			              static_cast<unsigned long long>(pc - reinterpret_cast<uint64_t>(info.dli_fbase)));
			return text;
		}
	}
#endif
	return "?";
}

// Symbol without the offset, for the per-symbol roll-up ("jit:slots:<hash>" keeps the plan, drops the offset).
inline std::string SymbolOnly(const std::string& symbol) {
	const auto plus = symbol.rfind("+0x");
	return plus == std::string::npos ? symbol : symbol.substr(0, plus);
}

// ---- Histogram of PCs. Single writer (the sampler thread).
class PcHistogram {
public:
	void Add(uint64_t pc) {
		m_cells[pc]++;
		m_total++;
	}
	void AddFail() { m_fail++; }
	uint64_t Total() const { return m_total; }
	uint64_t Fail() const { return m_fail; }
	size_t   Distinct() const { return m_cells.size(); }
	void     Clear() {
        m_cells.clear();
        m_total = 0;
        m_fail  = 0;
	}
	// Most frequent PCs, descending (ties: lower pc first).
	std::vector<std::pair<uint64_t, uint64_t>> Top(size_t n) const {
		std::vector<std::pair<uint64_t, uint64_t>> v(m_cells.begin(), m_cells.end());
		std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
		if (v.size() > n) v.resize(n);
		return v;
	}
	// Per symbol (offset dropped), descending.
	std::vector<std::pair<std::string, uint64_t>> TopSymbols(size_t n) const {
		std::unordered_map<std::string, uint64_t> by;
		for (const auto& [pc, count]: m_cells) by[SymbolOnly(Symbolize(pc))] += count;
		std::vector<std::pair<std::string, uint64_t>> v(by.begin(), by.end());
		std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second != b.second ? a.second > b.second : a.first < b.first; });
		if (v.size() > n) v.resize(n);
		return v;
	}

private:
	std::unordered_map<uint64_t, uint64_t> m_cells;
	uint64_t                               m_total = 0;
	uint64_t                               m_fail  = 0;
};

#if defined(__APPLE__)
// Suspends `thread`, reads RIP (x86_THREAD_STATE64), resumes. Nothing else runs between suspend and resume.
inline bool ReadPc(mach_port_t thread, uint64_t& pc) {
	if (thread == MACH_PORT_NULL) return false;
	if (thread_suspend(thread) != KERN_SUCCESS) return false;
	x86_thread_state64_t state {};
	mach_msg_type_number_t count = x86_THREAD_STATE64_COUNT;
	const kern_return_t    kr    = thread_get_state(thread, x86_THREAD_STATE64, reinterpret_cast<thread_state_t>(&state), &count);
	thread_resume(thread);
	if (kr != KERN_SUCCESS) return false;
	pc = state.__rip;
	return true;
}

// User + system CPU time of `thread` in ns.
inline bool ThreadCpuNs(mach_port_t thread, uint64_t& ns) {
	if (thread == MACH_PORT_NULL) return false;
	thread_basic_info_data_t info {};
	mach_msg_type_number_t   count = THREAD_BASIC_INFO_COUNT;
	if (thread_info(thread, THREAD_BASIC_INFO, reinterpret_cast<thread_info_t>(&info), &count) != KERN_SUCCESS) return false;
	const auto us = [](const time_value_t& t) { return static_cast<uint64_t>(t.seconds) * 1000000ull + static_cast<uint64_t>(t.microseconds); };
	ns            = (us(info.user_time) + us(info.system_time)) * 1000ull;
	return true;
}

inline std::atomic<mach_port_t>& CpThreadPort() {
	static std::atomic<mach_port_t> port {MACH_PORT_NULL};
	return port;
}
// Called on the CP thread.
inline void RegisterCurrentThread() {
	CpThreadPort().store(mach_thread_self(), std::memory_order_relaxed);
}
#endif

// CPU-vs-wall bookkeeping of one thread: Tick(cpu_ns, wall_ns) returns the deltas since the previous tick (the first tick returns 0, 0).
class CpuWall {
public:
	std::pair<uint64_t, uint64_t> Tick(uint64_t cpu_ns, uint64_t wall_ns) {
		std::pair<uint64_t, uint64_t> d {0, 0};
		if (m_have) {
			d.first  = cpu_ns >= m_cpu ? cpu_ns - m_cpu : 0;
			d.second = wall_ns >= m_wall ? wall_ns - m_wall : 0;
		}
		m_cpu  = cpu_ns;
		m_wall = wall_ns;
		m_have = true;
		return d;
	}

private:
	uint64_t m_cpu = 0, m_wall = 0;
	bool     m_have = false;
};

// ---- Guest-read tally of the sh.walk site: walks, reads, bytes, distinct 4 KiB pages per walk (a TLB estimate), calls of the interpreter /
// per-slot JIT / tree JIT. Fed by SrtWalker::RefreshFlatBuffer while WalkActive(); drained once per Get() into FrameTiming counters.
struct WalkTally {
	uint64_t              walks = 0, reads = 0, bytes = 0, pages = 0, interp_calls = 0, jit_slot_calls = 0, jit_tree_calls = 0;
	std::vector<uint64_t> walk_pages; // page numbers read by the walk in progress
};
inline std::atomic<bool>& WalkActive() {
	static std::atomic<bool> on {false};
	return on;
}
inline WalkTally& Tally() {
	thread_local WalkTally tally; // the CP thread walks in the measured scenes; other threads keep their own and are never drained
	return tally;
}
// `n` reads of `size` bytes at `addresses`.
inline void NoteReads(const uint64_t* addresses, size_t n, uint64_t size) {
#ifndef KYTY_CPPC_MUTANT_NO_READS
	auto& t = Tally();
	t.reads += n;
	t.bytes += n * size;
	for (size_t i = 0; i < n; ++i) {
		t.walk_pages.push_back(addresses[i] >> 12);
		if (size != 0 && ((addresses[i] + size - 1) >> 12) != (addresses[i] >> 12)) t.walk_pages.push_back((addresses[i] + size - 1) >> 12);
	}
#else
	(void)addresses; (void)n; (void)size;
#endif
}
// Closes the walk: counts it and folds its distinct pages into the tally.
inline void EndWalk() {
	auto& t = Tally();
	t.walks++;
	std::sort(t.walk_pages.begin(), t.walk_pages.end());
	t.pages += static_cast<uint64_t>(std::unique(t.walk_pages.begin(), t.walk_pages.end()) - t.walk_pages.begin());
	t.walk_pages.clear();
}
// Takes the counts accumulated by this thread since the last Drain.
inline WalkTally Drain() {
	auto&     t = Tally();
	WalkTally out;
	out.walks = std::exchange(t.walks, 0);
	out.reads = std::exchange(t.reads, 0);
	out.bytes = std::exchange(t.bytes, 0);
	out.pages = std::exchange(t.pages, 0);
	out.interp_calls   = std::exchange(t.interp_calls, 0);
	out.jit_slot_calls = std::exchange(t.jit_slot_calls, 0);
	out.jit_tree_calls = std::exchange(t.jit_tree_calls, 0);
	return out;
}

} // namespace Libs::Graphics::CpSampler::Pc

#endif
