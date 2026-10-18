#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PVRANGEGEN_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PVRANGEGEN_H_

// S406: KYTY_PV_RANGE_GEN=0|1|shadow. Memo of the VS-store record range of an indexed draw keyed by the GUEST ADDRESS of the index
// range and the draw parameters, validated by the page generation of the memory tracker instead of an XXH3 of the index bytes
// (renderDraw.cpp TryPvVsStoreDraw hashed 31 MB per flip). A result is reused only while
//   - the indices are guest memory (not a host copy),
//   - the whole range is registered in the tracker and clean (neither CPU- nor GPU-modified, i.e. write-protected), and
//   - the largest page stamp of the range is the one seen when the result was made (any write to a protected page faults and takes a
//     newer stamp; see PageStampClock in regionManager.h).
//   0       the old path only (bit for bit as before); S408: unset = 1
//   1       a generation hit is served from the memo (no hash, no scan); everything else takes the old path and refreshes the memo
//   shadow  the old path always; a generation hit is compared with what the old path computed (hash and range): mismatch counted
// Pure part; tests/PvRangeGenTests.cpp pins it.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

namespace Libs::Graphics::PvRangeGen {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) {
#ifdef KYTY_PVRANGEGEN_MUTANT_DEFAULT_OFF
		return Mode::Off;
#else
		return Mode::On; // S408: on by default; KYTY_PV_RANGE_GEN=0 restores the XXH3 of the index bytes
#endif
	}
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_PV_RANGE_GEN"));
	return mode;
}

struct Key {
	uint64_t address       = 0;
	uint64_t size          = 0;
	uint32_t element_size  = 0;
	uint32_t index_count   = 0;
	int32_t  vertex_offset = 0;
	uint32_t drop_limit    = 0;
	bool     operator==(const Key&) const = default;
};

struct Range {
	uint64_t first   = 0;
	uint64_t end     = 0;
	bool     dropped = false;
	bool     operator==(const Range&) const = default;
};

struct Entry {
	uint64_t max_stamp  = 0;
	uint64_t index_hash = 0; // XXH3 of the index bytes when the entry was made (shadow compares it)
	bool     derivable  = false;
	Range    range;
};

// What the tracker says about the index range of the draw now.
struct Probe {
	bool     host_data  = false; // the indices are a host copy: no page to watch
	bool     registered = false;
	bool     clean      = false;
	uint64_t max_stamp  = 0;
};

enum class Outcome : uint8_t { IneligibleHost, IneligibleUnregistered, IneligibleDirty, Miss, Hit };

constexpr Outcome Decide(const Probe& probe, const Entry* entry) {
	if (probe.host_data) return Outcome::IneligibleHost;
	if (!probe.registered) return Outcome::IneligibleUnregistered;
	if (!probe.clean) return Outcome::IneligibleDirty;
	if (entry == nullptr) return Outcome::Miss;
#if defined(KYTY_PVRANGEGEN_MUTANT_IGNORE_STAMP)
	return Outcome::Hit;
#else
	return entry->max_stamp == probe.max_stamp ? Outcome::Hit : Outcome::Miss;
#endif
}

// Shadow: the generation said "unchanged"; was the old path's answer the same?
struct Comparison {
	bool hash_differs  = false;
	bool range_differs = false;
	[[nodiscard]] bool Mismatch() const { return hash_differs || range_differs; }
};
constexpr Comparison Compare(const Entry& entry, uint64_t hash, bool derivable, const Range& range) {
	Comparison result;
	result.hash_differs  = entry.index_hash != hash;
	result.range_differs = entry.derivable != derivable || (derivable && !(entry.range == range));
	return result;
}

class Memo {
public:
	static constexpr size_t kMaxEntries = 4096;
	[[nodiscard]] const Entry* Find(const Key& key) const {
		const auto found = m_entries.find(Bucket(key));
		return found != m_entries.end() && found->second.key == key ? &found->second.entry : nullptr;
	}
	void Insert(const Key& key, const Entry& entry) {
		if (m_entries.size() >= kMaxEntries && m_entries.find(Bucket(key)) == m_entries.end()) m_entries.clear();
		m_entries[Bucket(key)] = {key, entry};
	}
	[[nodiscard]] size_t Size() const { return m_entries.size(); }

private:
	struct Slot {
		Key   key;
		Entry entry;
	};
	static uint64_t Bucket(const Key& key) {
		uint64_t hash = key.address * 0x9e3779b97f4a7c15ull;
		hash ^= key.size + 0x7f4a7c15ull + (hash << 6) + (hash >> 2);
		hash ^= (static_cast<uint64_t>(key.element_size) << 32 | key.index_count) * 0xff51afd7ed558ccdull;
		hash ^= static_cast<uint64_t>(static_cast<uint32_t>(key.vertex_offset)) * 0xc4ceb9fe1a85ec53ull;
		hash ^= key.drop_limit;
		return hash;
	}
	std::unordered_map<uint64_t, Slot> m_entries;
};

} // namespace Libs::Graphics::PvRangeGen

#endif
