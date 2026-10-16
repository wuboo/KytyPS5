#ifndef KYTY_GRAPHICS_SHADER_NULLDESCDIAG_H_
#define KYTY_GRAPHICS_SHADER_NULLDESCDIAG_H_

// S379: KYTY_DIAG_NULLDESC=1 counts stand-in image binds by why the T# was null, and
// KYTY_DIAG_NULLDESC_FLIPS=lo-hi (default 6590-6600) is the frames.csv `frame` window
// (the same index as frame_flip_<n>), not the guest `flips` column.
// Walker counters live inline so shader test binaries do not link the presenter.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

namespace Libs::Graphics::NullDesc {

enum class Reason : uint8_t { None = 0, EmptyT = 1, NullTable = 2, Unresolved = 3, Other = 4 };

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
inline Reason Classify(bool was_null, bool valid, uint32_t null_table_delta, uint32_t unresolved_delta) {
	const bool bad = was_null || !valid;
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
	if (!valid && !was_null) {
		return Reason::Other;
	}
	if (bad) {
		return Reason::EmptyT;
	}
	return Reason::None;
}

inline const char* ReasonName(Reason reason) {
	switch (reason) {
		case Reason::EmptyT: return "empty";
		case Reason::NullTable: return "nulltable";
		case Reason::Unresolved: return "unresolved";
		case Reason::Other: return "other";
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

void NoteStandin(uint64_t shader, uint32_t slot, uint64_t addr, Reason reason);
void OnPresented(uint32_t frame, uint32_t guest_flips, uint64_t minlod_remap_total, uint64_t minlod_dropped_total);

} // namespace Libs::Graphics::NullDesc

#endif
