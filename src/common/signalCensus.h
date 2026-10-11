#ifndef KYTY_COMMON_SIGNAL_CENSUS_H_
#define KYTY_COMMON_SIGNAL_CENSUS_H_

// S276: census of the host signals the guest process takes (SIGSEGV/SIGBUS/SIGILL/SIGUSR1) and of the path that handled each
// one. Counters are relaxed atomics, always on (one fetch_add per signal; the signal itself costs ~60 us in the Rosetta
// exception server). The per-RIP histogram of SIGILL / emulated instructions is only filled with KYTY_DIAG_SIGCENSUS=1 (default
// OFF). Everything is lock-free and allocation-free: it runs inside signal handlers. Header-only so tests link nothing.

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#if !defined(_WIN32)
#include <pthread.h>
#endif

namespace Common::SignalCensus {

enum class Sig : uint8_t { Segv, Bus, Ill, Usr1, Other, Count };

// Which handler branch consumed (or rejected) a signal.
enum class Path : uint8_t {
	GpuFaultHandled,   // access violation fixed by HandleGpuFault
	GpuFaultUnhandled, // access violation HandleGpuFault refused (or unknown access): the process ends
	WideStoreTrap,     // ud2 planted by SplitWideStores, redirected to the trampoline
	Rsqrt,             // TryEmulateReciprocalSquareRoot
	MonitorxMwaitx,    // TryEmulateMonitorxMwaitx
	Sse4a,             // TryEmulateSse4a (EXTRQ / INSERTQ)
	ShaNi,             // TryEmulateShaNi (not built on macOS)
	CpuExtensions,     // TryEmulateCpuExtensions (RDPID, CLFLUSH..., Zydis based)
	IllUnhandled,      // SIGILL no emulator accepted: the process ends
	Count,
};

inline constexpr const char* kSigNames[] = {"sig_segv", "sig_bus", "sig_ill", "sig_usr1", "sig_other"};
inline constexpr const char* kPathNames[] = {"path_gpu_fault_ok", "path_gpu_fault_fail", "path_wide_store_trap", "path_rsqrt",
                                             "path_monitorx_mwaitx", "path_sse4a", "path_shani", "path_cpuext", "path_ill_unhandled"};
static_assert(sizeof(kSigNames) / sizeof(kSigNames[0]) == static_cast<size_t>(Sig::Count));
static_assert(sizeof(kPathNames) / sizeof(kPathNames[0]) == static_cast<size_t>(Path::Count));

inline constexpr size_t kSigCount  = static_cast<size_t>(Sig::Count);
inline constexpr size_t kPathCount = static_cast<size_t>(Path::Count);
// frames.csv columns: signals, then paths, then signals taken on the registered main thread / on any other thread.
inline constexpr size_t kColumnCount = kSigCount + kPathCount + 2;

inline std::atomic<uint64_t> g_sig[kSigCount] {};
inline std::atomic<uint64_t> g_path[kPathCount] {};
inline std::atomic<uint64_t> g_main {0};
inline std::atomic<uint64_t> g_other {0};
#if !defined(_WIN32)
inline std::atomic<uintptr_t> g_main_thread {0};
#endif

inline Sig SigFromHostSignal(int host_signal) {
	switch (host_signal) {
		case SIGSEGV: return Sig::Segv;
#ifdef KYTY_SIGCENSUS_MUTANT_SIG_SWAP
		case SIGBUS: return Sig::Segv;
#else
		case SIGBUS: return Sig::Bus;
#endif
		case SIGILL: return Sig::Ill;
		case SIGUSR1: return Sig::Usr1;
		default: return Sig::Other;
	}
}

// The thread that runs the guest entry point (the game's main thread); signals on it are counted in the "main" column.
inline void SetMainThread() {
#if !defined(_WIN32)
	g_main_thread.store(reinterpret_cast<uintptr_t>(pthread_self()), std::memory_order_relaxed);
#endif
}

inline bool OnMainThread() {
#if !defined(_WIN32)
	const auto main_thread = g_main_thread.load(std::memory_order_relaxed);
#ifdef KYTY_SIGCENSUS_MUTANT_MAIN_ALWAYS
	return true;
#endif
	return main_thread != 0 && main_thread == reinterpret_cast<uintptr_t>(pthread_self());
#else
	return false;
#endif
}

inline void OnSignal(Sig sig) {
	g_sig[static_cast<size_t>(sig)].fetch_add(1, std::memory_order_relaxed);
	(OnMainThread() ? g_main : g_other).fetch_add(1, std::memory_order_relaxed);
}

inline void OnPath(Path path) {
	g_path[static_cast<size_t>(path)].fetch_add(1, std::memory_order_relaxed);
}

inline void Reset() {
	for (auto& value: g_sig) value.store(0, std::memory_order_relaxed);
	for (auto& value: g_path) value.store(0, std::memory_order_relaxed);
	g_main.store(0, std::memory_order_relaxed);
	g_other.store(0, std::memory_order_relaxed);
}

// Snapshot in column order (see ColumnName).
inline void Snapshot(uint64_t (&out)[kColumnCount]) {
	size_t index = 0;
	for (const auto& value: g_sig) out[index++] = value.load(std::memory_order_relaxed);
	for (const auto& value: g_path) out[index++] = value.load(std::memory_order_relaxed);
	out[index++] = g_main.load(std::memory_order_relaxed);
	out[index++] = g_other.load(std::memory_order_relaxed);
}

inline const char* ColumnName(size_t column) {
	if (column < kSigCount) return kSigNames[column];
	if (column < kSigCount + kPathCount) return kPathNames[column - kSigCount];
	return column == kSigCount + kPathCount ? "sig_main_thread" : "sig_other_threads";
}

// frames.csv: ",sig_segv,...," names (header) and current values (row), appended after the frame counters.
inline void WriteCsvHeader(FILE* file) {
	for (size_t column = 0; column < kColumnCount; column++) std::fprintf(file, ",%s", ColumnName(column));
}

inline void WriteCsvRow(FILE* file) {
	uint64_t values[kColumnCount];
	Snapshot(values);
	for (size_t column = 0; column < kColumnCount; column++) {
#ifdef KYTY_SIGCENSUS_MUTANT_CSV_SHIFTED
		std::fprintf(file, ",%llu", static_cast<unsigned long long>(values[(column + 1) % kColumnCount]));
#else
		std::fprintf(file, ",%llu", static_cast<unsigned long long>(values[column]));
#endif
	}
}

// ---- RIP histogram (KYTY_DIAG_SIGCENSUS=1) ----

inline bool HistogramEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_DIAG_SIGCENSUS");
		return value != nullptr && value[0] == '1';
	}();
	return enabled;
}

inline constexpr size_t kHistogramSlots = 4096; // power of two
inline constexpr size_t kHistogramProbe = 64;
inline constexpr size_t kOpcodeBytes    = 8;    // the only guest code kept per entry

struct HistogramEntry {
	std::atomic<uint64_t> rip {0};
	std::atomic<uint64_t> count {0};
	std::atomic<uint64_t> opcode {0};
	std::atomic<uint32_t> kind {0};
};

inline HistogramEntry        g_histogram[kHistogramSlots] {};
inline std::atomic<uint64_t> g_histogram_overflow {0};

// Counts one hit of `kind` at `rip`. The first hit stores up to kOpcodeBytes of code at rip, never reading past the page the
// CPU just fetched it from.
inline void RecordRip(uint64_t rip, Path kind) {
	if (rip == 0) return;
	auto slot = static_cast<size_t>((rip * 0x9e3779b97f4a7c15ull) >> 40) & (kHistogramSlots - 1);
	for (size_t probe = 0; probe < kHistogramProbe; probe++, slot = (slot + 1) & (kHistogramSlots - 1)) {
		auto&    entry = g_histogram[slot];
		uint64_t key   = entry.rip.load(std::memory_order_relaxed);
		if (key == 0 && entry.rip.compare_exchange_strong(key, rip, std::memory_order_relaxed)) {
			uint8_t       bytes[kOpcodeBytes] {};
#ifdef KYTY_SIGCENSUS_MUTANT_NO_PAGE_CLAMP
			const size_t  to_page_end = kOpcodeBytes;
#else
			const size_t  to_page_end = 4096 - static_cast<size_t>(rip & 4095);
#endif
			std::memcpy(bytes, reinterpret_cast<const void*>(rip), to_page_end < kOpcodeBytes ? to_page_end : kOpcodeBytes);
			uint64_t packed = 0;
			std::memcpy(&packed, bytes, sizeof(packed));
			entry.opcode.store(packed, std::memory_order_relaxed);
			entry.kind.store(static_cast<uint32_t>(kind), std::memory_order_relaxed);
			key = rip;
		}
		if (key == rip) {
#ifdef KYTY_SIGCENSUS_MUTANT_HIST_COUNT_ONCE
			entry.count.store(1, std::memory_order_relaxed);
#else
			entry.count.fetch_add(1, std::memory_order_relaxed);
#endif
			return;
		}
	}
	g_histogram_overflow.fetch_add(1, std::memory_order_relaxed);
}

struct TopEntry {
	uint64_t rip    = 0;
	uint64_t count  = 0;
	uint64_t opcode = 0;
	Path     kind   = Path::IllUnhandled;
};

// Top `n` (<= 16) entries by count, most frequent first; returns how many were found.
inline size_t TopRips(TopEntry (&out)[16], size_t n) {
	if (n > 16) n = 16;
	size_t found = 0;
	for (const auto& entry: g_histogram) {
		const auto count = entry.count.load(std::memory_order_relaxed);
		if (count == 0) continue;
		TopEntry candidate {entry.rip.load(std::memory_order_relaxed), count, entry.opcode.load(std::memory_order_relaxed),
		                    static_cast<Path>(entry.kind.load(std::memory_order_relaxed))};
		if (found == n && count <= out[n - 1].count) continue;
		size_t at = found < n ? found : n - 1;
		if (found < n) found++;
#ifdef KYTY_SIGCENSUS_MUTANT_TOP_UNSORTED
		while (at > 0 && out[at - 1].count > count) {
#else
		while (at > 0 && out[at - 1].count < count) {
#endif
			out[at] = out[at - 1];
			at--;
		}
		out[at] = candidate;
	}
	return found;
}

// Resolves a RIP to "module" + offset; returns false when it is outside every module.
using Resolver = bool (*)(uint64_t rip, char* module, size_t module_size, uint64_t* offset);
using Emit     = void (*)(const char* line);

// Writes the top-16 table through `emit`: count, kind, module+offset and the first opcode bytes.
inline void DumpHistogram(const char* why, Resolver resolver, Emit emit) {
	TopEntry top[16] {};
	const auto found = TopRips(top, 16);
	char line[256];
	std::snprintf(line, sizeof(line), "[sigcensus] %s: %zu rips in top list, histogram overflow=%llu\n", why, found,
	              static_cast<unsigned long long>(g_histogram_overflow.load(std::memory_order_relaxed)));
	emit(line);
	for (size_t i = 0; i < found; i++) {
		char     module[64] = "?";
		uint64_t offset     = top[i].rip;
		if (resolver != nullptr && !resolver(top[i].rip, module, sizeof(module), &offset)) {
			std::snprintf(module, sizeof(module), "unmapped");
		}
		uint8_t bytes[kOpcodeBytes];
		std::memcpy(bytes, &top[i].opcode, sizeof(bytes));
		std::snprintf(line, sizeof(line), "[sigcensus] #%zu count=%llu kind=%s at=%s+0x%llx op=%02x %02x %02x %02x %02x %02x %02x %02x\n",
		              i + 1, static_cast<unsigned long long>(top[i].count), kPathNames[static_cast<size_t>(top[i].kind)], module,
		              static_cast<unsigned long long>(offset), bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
		              bytes[7]);
		emit(line);
	}
}

// The loader registers its module lookup here; the periodic and exit dumps use it.
inline std::atomic<Resolver> g_resolver {nullptr};

inline void SetResolver(Resolver resolver) {
	g_resolver.store(resolver, std::memory_order_relaxed);
}

inline void DumpHistogram(const char* why, Emit emit) {
	DumpHistogram(why, g_resolver.load(std::memory_order_relaxed), emit);
}

inline void ResetHistogram() {
	for (auto& entry: g_histogram) {
		entry.count.store(0, std::memory_order_relaxed);
		entry.opcode.store(0, std::memory_order_relaxed);
		entry.kind.store(0, std::memory_order_relaxed);
		entry.rip.store(0, std::memory_order_relaxed);
	}
	g_histogram_overflow.store(0, std::memory_order_relaxed);
}

} // namespace Common::SignalCensus

#endif /* KYTY_COMMON_SIGNAL_CENSUS_H_ */
