#ifndef KYTY_GRAPHICS_SHADER_NULLDESCDIAG_H_
#define KYTY_GRAPHICS_SHADER_NULLDESCDIAG_H_

// S379: KYTY_DIAG_NULLDESC=1 counts stand-in image binds by why the T# was null, and
// KYTY_DIAG_NULLDESC_FLIPS=lo-hi (default 6590-6600) is the frames.csv `frame` window
// (the same index as frame_flip_<n>), not the guest `flips` column.
// Walker counters live inline so shader test binaries do not link the presenter.

#include "common/logging/log.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <span>
#include <vector>

namespace Libs::Graphics::NullDesc {

// EmptyT stays for the null-table mutant and for a hint that was never classified.
// inactive / filtered_* / zero_mem split the old "empty addr=0" bucket.
enum class Reason : uint8_t {
	None = 0, EmptyT = 1, NullTable = 2, Unresolved = 3, Other = 4,
	Inactive = 5, FilteredValid = 6, FilteredDim = 7, ZeroMem = 8,
};

inline constexpr uint32_t kReasonCount = 9;
inline constexpr uint64_t kDriverWindowLo = 0xfe0000000ull;
inline constexpr uint64_t kDriverWindowHi = 0xfe0200000ull;

struct FlipWindow {
	uint32_t lo = 6590;
	uint32_t hi = 6600;
};

// Empty or unparsable text keeps the default. "N" is the single frame N. "A-B" requires B >= A.
inline FlipWindow ParseFlipWindow(const char* text) {
	FlipWindow window;
	if (text == nullptr || text[0] == '\0') {
		return window;
	}
	char* end = nullptr;
	const auto first = std::strtoull(text, &end, 10);
	if (end == text) {
		return window;
	}
	if (*end == '\0') {
		window.lo = static_cast<uint32_t>(first);
		window.hi = window.lo;
		return window;
	}
	if (*end != '-') {
		return window;
	}
	char* end_hi = nullptr;
	const auto second = std::strtoull(end + 1, &end_hi, 10);
	if (end_hi == end + 1 || *end_hi != '\0' || second < first) {
		return FlipWindow {};
	}
	window.lo = static_cast<uint32_t>(first);
	window.hi = static_cast<uint32_t>(second);
	return window;
}

inline bool InWindow(FlipWindow window, uint32_t frame) {
	return frame >= window.lo && frame <= window.hi;
}

inline FlipWindow Window() {
	static const FlipWindow window = ParseFlipWindow(std::getenv("KYTY_DIAG_NULLDESC_FLIPS"));
	return window;
}

// 40-bit T# base, shifted to a byte address (0 when the descriptor is a null T#).
inline uint64_t TextureBase40(const uint32_t* dwords) {
	if (dwords == nullptr) {
		return 0;
	}
	return ((static_cast<uint64_t>(dwords[0]) | (static_cast<uint64_t>(dwords[1]) << 32u)) &
	        0xffffffffffull)
	       << 8u;
}

// `was_null` / `valid` describe the descriptor before it is forced to zeros.
// A null-table read wins over a failed validity check: a zeroed table looks invalid too.
inline Reason Classify(bool was_null, bool valid, uint32_t null_table_delta, uint32_t unresolved_delta,
                       bool inactive = false, bool filtered_dim = false) {
	if (inactive) {
		return Reason::Inactive;
	}
	const bool bad = was_null || !valid || filtered_dim;
	if (null_table_delta != 0 && bad) {
#if defined(KYTY_NULLDESC_MUTANT_EMPTY_FOR_NULLTABLE)
		return Reason::EmptyT;
#else
		return Reason::NullTable;
#endif
	}
	if (unresolved_delta != 0 && bad) {
		return Reason::Unresolved;
	}
	if (filtered_dim && !was_null) {
		return Reason::FilteredDim;
	}
	if (!valid && !was_null) {
		return Reason::FilteredValid;
	}
	if (bad) {
		return Reason::ZeroMem;
	}
	return Reason::None;
}

inline const char* ReasonName(Reason reason) {
	switch (reason) {
		case Reason::EmptyT: return "empty";
		case Reason::NullTable: return "nulltable";
		case Reason::Unresolved: return "unresolved";
		case Reason::Other: return "other";
		case Reason::Inactive: return "inactive";
		case Reason::FilteredValid: return "filtered_valid";
		case Reason::FilteredDim: return "filtered_dim";
		case Reason::ZeroMem: return "zero_mem";
		case Reason::None: return "none";
	}
	return "none";
}

struct TopEntry {
	uint64_t shader = 0;
	uint32_t slot   = 0;
	uint64_t addr   = 0;
	Reason   reason = Reason::None;
	uint32_t n      = 0;
};

// Bounded histogram. A key that does not fit replaces the current minimum count.
class TopList {
public:
	static constexpr uint32_t kCap = 256;

	void Add(uint64_t shader, uint32_t slot, uint64_t addr, Reason reason) {
		for (auto& entry: entries) {
			if (entry.shader == shader && entry.slot == slot && entry.addr == addr && entry.reason == reason) {
				entry.n++;
				return;
			}
		}
		if (entries.size() < kCap) {
			entries.push_back(TopEntry {shader, slot, addr, reason, 1});
			return;
		}
		auto lowest = std::min_element(entries.begin(), entries.end(),
		                               [](const TopEntry& a, const TopEntry& b) { return a.n < b.n; });
		*lowest = TopEntry {shader, slot, addr, reason, 1};
	}

	std::vector<TopEntry> Top(uint32_t limit) const {
		auto copy = entries;
		std::sort(copy.begin(), copy.end(), [](const TopEntry& a, const TopEntry& b) {
			if (a.n != b.n) return a.n > b.n;
			if (a.shader != b.shader) return a.shader < b.shader;
			return a.slot < b.slot;
		});
		if (copy.size() > limit) {
			copy.resize(limit);
		}
		return copy;
	}

	void Clear() { entries.clear(); }

private:
	std::vector<TopEntry> entries;
};

struct WalkMark {
	uint32_t null_tables = 0;
	uint32_t unresolved  = 0;
};

struct WalkDelta {
	uint32_t null_tables     = 0;
	uint32_t unresolved      = 0;
	uint64_t table           = 0;
	uint64_t unresolved_addr = 0;
};

struct Hint {
	uint8_t  reason = 0;
	uint64_t addr   = 0;
	uint64_t shader = 0;
};

inline bool Enabled() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_NULLDESC");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return on;
}

// Present index of the frame currently being drawn (frames.csv `frame`). The presenter stores the next index.
inline std::atomic<uint32_t>& DrawingFrame() {
	static std::atomic<uint32_t> frame {1};
	return frame;
}

inline std::atomic<uint64_t>& WalkerNullTotal() {
	static std::atomic<uint64_t> total {0};
	return total;
}

inline std::atomic<uint64_t>& WalkerUnresolvedTotal() {
	static std::atomic<uint64_t> total {0};
	return total;
}

inline TopList& WalkerTop() {
	static TopList top;
	return top;
}

inline std::mutex& WalkerTopLock() {
	static std::mutex lock;
	return lock;
}

struct Tls {
	uint32_t null_tables     = 0;
	uint32_t unresolved      = 0;
	uint64_t last_table      = 0;
	uint64_t last_unresolved = 0;
};

inline Tls& WalkTls() {
	thread_local Tls tls;
	return tls;
}

inline WalkMark MarkWalk() {
	const auto& tls = WalkTls();
	return WalkMark {tls.null_tables, tls.unresolved};
}

inline WalkDelta Since(WalkMark mark) {
	const auto& tls = WalkTls();
	WalkDelta delta;
	delta.null_tables     = tls.null_tables - mark.null_tables;
	delta.unresolved      = tls.unresolved - mark.unresolved;
	delta.table           = tls.last_table;
	delta.unresolved_addr = tls.last_unresolved;
	return delta;
}

inline void NoteNullTable(uint64_t table) {
	if (!Enabled()) {
		return;
	}
	auto& tls = WalkTls();
	tls.null_tables++;
	tls.last_table = table;
	WalkerNullTotal().fetch_add(1, std::memory_order_relaxed);
	if (!InWindow(Window(), DrawingFrame().load(std::memory_order_relaxed))) {
		return;
	}
	std::lock_guard lock {WalkerTopLock()};
	WalkerTop().Add(0, 0, table, Reason::NullTable);
}

inline void NoteUnresolved(uint64_t address) {
	if (!Enabled()) {
		return;
	}
	auto& tls = WalkTls();
	tls.unresolved++;
	tls.last_unresolved = address;
	WalkerUnresolvedTotal().fetch_add(1, std::memory_order_relaxed);
}

struct StandinState {
	std::atomic<uint64_t> n[kReasonCount] {};
	std::mutex            lock;
	TopList               top;
	bool                  dumped              = false;
	uint64_t              last[kReasonCount]  {};
	uint64_t              last_walker_null    = 0;
	uint64_t              last_walker_unres   = 0;
	uint64_t              last_minlod_remap   = 0;
	uint64_t              last_minlod_dropped = 0;
	uint64_t              last_driver_reads   = 0;
	uint64_t              last_driver_nonzero = 0;
	uint64_t              last_gpu_dirty      = 0;
	bool                  announced           = false;
};

inline std::atomic<uint64_t>& SrtDriverWindowReads() {
	static std::atomic<uint64_t> total {0};
	return total;
}

inline std::atomic<uint64_t>& SrtDriverWindowNonzero() {
	static std::atomic<uint64_t> total {0};
	return total;
}

inline std::atomic<uint64_t>& SrtGpuDirtyReads() {
	static std::atomic<uint64_t> total {0};
	return total;
}

// Ordinary SRT reads. The compiled walker (main, read_memory empty) does not pass through here.
inline void NoteSrtRead(uint64_t address, std::span<const uint32_t> words, bool gpu_dirty) {
	if (!Enabled()) {
		return;
	}
	if (gpu_dirty) {
		SrtGpuDirtyReads().fetch_add(1, std::memory_order_relaxed);
	}
	if (address < kDriverWindowLo || address >= kDriverWindowHi) {
		return;
	}
	SrtDriverWindowReads().fetch_add(1, std::memory_order_relaxed);
	for (const auto word: words) {
		if (word != 0u) {
			SrtDriverWindowNonzero().fetch_add(1, std::memory_order_relaxed);
			break;
		}
	}
}

inline StandinState& Standins() {
	static StandinState state;
	return state;
}

inline void NoteStandin(uint64_t shader, uint32_t slot, uint64_t addr, Reason reason) {
	if (!Enabled()) {
		return;
	}
	auto& state = Standins();
	const auto index = static_cast<uint32_t>(reason);
	if (index < kReasonCount) {
		state.n[index].fetch_add(1, std::memory_order_relaxed);
	}
	if (!InWindow(Window(), DrawingFrame().load(std::memory_order_relaxed))) {
		return;
	}
	std::lock_guard lock {state.lock};
	state.top.Add(shader, slot, addr, reason);
}

inline void OnPresented(uint32_t frame, uint32_t guest_flips, uint64_t minlod_remap_total,
                        uint64_t minlod_dropped_total) {
	if (!Enabled()) {
		return;
	}
	auto& state = Standins();
	if (!state.announced) {
		state.announced = true;
		const auto window = Window();
		LOGF("[nulldesc] on frame_window=%u-%u (frames.csv frame / frame_flip_N, not guest flips)\n", window.lo,
		     window.hi);
	}
	uint64_t standin[kReasonCount];
	for (uint32_t i = 0; i < kReasonCount; i++) {
		standin[i] = state.n[i].load(std::memory_order_relaxed);
	}
	const auto walker_null = WalkerNullTotal().load(std::memory_order_relaxed);
	const auto walker_bad  = WalkerUnresolvedTotal().load(std::memory_order_relaxed);
	const auto driver_reads = SrtDriverWindowReads().load(std::memory_order_relaxed);
	const auto driver_nonzero = SrtDriverWindowNonzero().load(std::memory_order_relaxed);
	const auto gpu_dirty = SrtGpuDirtyReads().load(std::memory_order_relaxed);
	const auto empty       = static_cast<uint32_t>(Reason::EmptyT);
	const auto nulltable   = static_cast<uint32_t>(Reason::NullTable);
	const auto unresolved  = static_cast<uint32_t>(Reason::Unresolved);
	const auto other       = static_cast<uint32_t>(Reason::Other);
	const auto inactive    = static_cast<uint32_t>(Reason::Inactive);
	const auto filtered_valid = static_cast<uint32_t>(Reason::FilteredValid);
	const auto filtered_dim   = static_cast<uint32_t>(Reason::FilteredDim);
	const auto zero_mem       = static_cast<uint32_t>(Reason::ZeroMem);
	LOGF("[nulldesc] frame=%u guest_flips=%u standin_empty=%llu standin_nulltable=%llu "
	     "standin_unresolved=%llu standin_other=%llu standin_inactive=%llu standin_filtered_valid=%llu "
	     "standin_filtered_dim=%llu standin_zero_mem=%llu walker_nulltable=%llu walker_unresolved=%llu "
	     "srt_read_driver_window=%llu srt_read_driver_nonzero=%llu srt_read_gpu_dirty=%llu "
	     "minlod_remap=%llu minlod_dropped=%llu\n",
	     frame, guest_flips, static_cast<unsigned long long>(standin[empty] - state.last[empty]),
	     static_cast<unsigned long long>(standin[nulltable] - state.last[nulltable]),
	     static_cast<unsigned long long>(standin[unresolved] - state.last[unresolved]),
	     static_cast<unsigned long long>(standin[other] - state.last[other]),
	     static_cast<unsigned long long>(standin[inactive] - state.last[inactive]),
	     static_cast<unsigned long long>(standin[filtered_valid] - state.last[filtered_valid]),
	     static_cast<unsigned long long>(standin[filtered_dim] - state.last[filtered_dim]),
	     static_cast<unsigned long long>(standin[zero_mem] - state.last[zero_mem]),
	     static_cast<unsigned long long>(walker_null - state.last_walker_null),
	     static_cast<unsigned long long>(walker_bad - state.last_walker_unres),
	     static_cast<unsigned long long>(driver_reads - state.last_driver_reads),
	     static_cast<unsigned long long>(driver_nonzero - state.last_driver_nonzero),
	     static_cast<unsigned long long>(gpu_dirty - state.last_gpu_dirty),
	     static_cast<unsigned long long>(minlod_remap_total - state.last_minlod_remap),
	     static_cast<unsigned long long>(minlod_dropped_total - state.last_minlod_dropped));
	for (uint32_t i = 0; i < kReasonCount; i++) {
		state.last[i] = standin[i];
	}
	state.last_walker_null    = walker_null;
	state.last_walker_unres   = walker_bad;
	state.last_driver_reads   = driver_reads;
	state.last_driver_nonzero = driver_nonzero;
	state.last_gpu_dirty      = gpu_dirty;
	state.last_minlod_remap   = minlod_remap_total;
	state.last_minlod_dropped = minlod_dropped_total;
	DrawingFrame().store(frame + 1, std::memory_order_relaxed);

	const auto window = Window();
	if (state.dumped || frame != window.hi) {
		return;
	}
	state.dumped = true;
	std::vector<TopEntry> standin_top;
	std::vector<TopEntry> walker_top;
	{
		std::lock_guard lock {state.lock};
		standin_top = state.top.Top(20);
	}
	{
		std::lock_guard lock {WalkerTopLock()};
		walker_top = WalkerTop().Top(20);
	}
	LOGF("[nulldesc] top frame=%u-%u standin=%zu walker=%zu\n", window.lo, window.hi, standin_top.size(),
	     walker_top.size());
	for (uint32_t rank = 0; rank < standin_top.size(); rank++) {
		const auto& hit = standin_top[rank];
		LOGF("[nulldesc] top rank=%u n=%u reason=%s shader=0x%016llx slot=%u addr=0x%llx\n", rank + 1, hit.n,
		     ReasonName(hit.reason), static_cast<unsigned long long>(hit.shader), hit.slot,
		     static_cast<unsigned long long>(hit.addr));
	}
	for (uint32_t rank = 0; rank < walker_top.size(); rank++) {
		const auto& hit = walker_top[rank];
		LOGF("[nulldesc] walker-top rank=%u n=%u table=0x%llx\n", rank + 1, hit.n,
		     static_cast<unsigned long long>(hit.addr));
	}
}

} // namespace Libs::Graphics::NullDesc

#endif
