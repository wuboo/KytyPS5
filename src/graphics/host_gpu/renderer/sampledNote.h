#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_SAMPLEDNOTE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_SAMPLEDNOTE_H_

// S445 (R1): KYTY_TEX_SAMPLED_NOTE=all|watch (default watch). DiagDump::NoteSampledTex records every T# resolved in a flip so FlipReport can list the ones
// inside a watch_tex range. The list is read only for ranges armed by WatchTexTrigger, i.e. only when the control file lists watch_tex. With the control
// file set (bench harness) the old code scanned the whole list under a global mutex for every resolved T#: O(N^2) per flip, ~1 ms/flip on the bench and
// nothing in production (no control file). `watch` notes only while the control file lists watch_tex (or diag_tex); `all` is the old behaviour.
// This header is the pure part (tests/SampledNoteTests.cpp pins it). It feeds no shader or pipeline cache key.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::SampledNote {

enum class Mode { Watch, All };

inline Mode ParseMode(const char* value) {
	return value != nullptr && std::strcmp(value, "all") == 0 ? Mode::All : Mode::Watch;
}

inline Mode CurrentMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_TEX_SAMPLED_NOTE"));
	return mode;
}

// True when a resolved T# has to be noted: `all` always, `watch` only while the control file lists watch_tex or diag_tex.
inline bool ShouldNote(Mode mode, bool watch_listed) {
#ifdef KYTY_SAMPLEDNOTE_MUTANT_GATE_INVERTED
	return mode == Mode::Watch ? !watch_listed : true;
#else
	return mode == Mode::All || watch_listed;
#endif
}

struct Entry {
	uint64_t ps, addr;
	uint32_t slot, fmt, width, height;
};

// The T#s of one flip, deduplicated by (ps, slot, addr), capped.
class List {
public:
	static constexpr size_t kCap = 8192;

	// Returns true when the T# was added.
	bool Note(const Entry& entry) {
		if (entries_.size() >= kCap) return false;
#ifdef KYTY_SAMPLEDNOTE_MUTANT_NO_DEDUP
		entries_.push_back(entry);
		return true;
#else
		if (!keys_.insert(Key {entry.ps, entry.addr, entry.slot}).second) return false;
		entries_.push_back(entry);
		return true;
#endif
	}
	void Clear() {
		entries_.clear();
		keys_.clear();
	}
	const std::vector<Entry>& Entries() const { return entries_; }

private:
	struct Key {
		uint64_t ps, addr;
		uint32_t slot;
		bool     operator==(const Key& o) const { return ps == o.ps && addr == o.addr && slot == o.slot; }
	};
	struct KeyHash {
		size_t operator()(const Key& k) const {
			uint64_t h = k.ps * 0x9e3779b97f4a7c15ull;
			h ^= k.addr + 0x7f4a7c15ull + (h << 6) + (h >> 2);
			h ^= k.slot + 0x9e3779b9ull + (h << 6) + (h >> 2);
			return static_cast<size_t>(h);
		}
	};
	std::vector<Entry>                      entries_;
	std::unordered_set<Key, KeyHash> keys_;
};

} // namespace Libs::Graphics::SampledNote

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_SAMPLEDNOTE_H_
