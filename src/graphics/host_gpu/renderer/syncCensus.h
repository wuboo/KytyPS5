#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_SYNCCENSUS_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_SYNCCENSUS_H_

// S398 (DIAG, default off): census of the points where the GPU thread drains the queue for a readback.
//   KYTY_DIAG_SYNC=1      per call site of BufferCache::ReadMemory: count, time, eager hits; writer lag and "writer already done".
//   KYTY_DIAG_INDIRECT=1  DrawIndirectMulti: state of the args before the first CPU read, plus a startup feature line.
// This header is the pure part (tests/SyncCensusTests.cpp pins it). Nothing here feeds a shader or pipeline cache key.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <unordered_map>

namespace Libs::Graphics::SyncCensus {

// Where a readback was requested from. Order = the order of the sync_* counters in FrameTiming::Counter.
enum class Site : uint8_t { Dcc, RfGpu, RfGuest, WriteFault, Spec, Other, Count };
inline constexpr uint32_t kSiteCount = static_cast<uint32_t>(Site::Count);

constexpr const char* SiteName(Site site) {
	switch (site) {
		case Site::Dcc: return "dcc";
		case Site::RfGpu: return "rf_gpu";
		case Site::RfGuest: return "rf_guest";
		case Site::WriteFault: return "write_fault";
		case Site::Spec: return "spec";
		default: return "other";
	}
}

// KYTY_DIAG_xxx is on for the value "1" only.
inline bool FlagOn(const char* name) {
	const char* value = std::getenv(name);
	return value != nullptr && value[0] == '1' && value[1] == '\0';
}
inline bool Enabled() {
	static const bool on = FlagOn("KYTY_DIAG_SYNC");
	return on;
}
inline bool IndirectEnabled() {
	static const bool on = FlagOn("KYTY_DIAG_INDIRECT");
	return on;
}

// Tag of the readback site, set by the callers of ReadMemory. Thread-local: the guest thread that faults and the GPU thread that
// serves the readback each have their own, ReadMemory reads it on the requesting thread. Innermost scope wins; leaving a scope
// restores the outer tag, and the tag outside every scope is Other.
inline Site& CurrentSiteRef() {
	static thread_local Site site = Site::Other;
	return site;
}
inline Site CurrentSite() {
	return CurrentSiteRef();
}

class SiteScope {
public:
	explicit SiteScope(Site site): m_previous(CurrentSiteRef()) {
#if defined(KYTY_SYNCCENSUS_MUTANT_TAG_IGNORED)
		(void)site;
#else
		CurrentSiteRef() = site;
#endif
	}
	~SiteScope() {
#if !defined(KYTY_SYNCCENSUS_MUTANT_TAG_NOT_RESTORED)
		CurrentSiteRef() = m_previous;
#endif
	}
	SiteScope(const SiteScope&)            = delete;
	SiteScope& operator=(const SiteScope&) = delete;

private:
	Site m_previous;
};

// S402: the shader program being specialized on this thread (hash + stage name), set by PipelineCache around Get(); the
// readbacks tagged Site::Spec log it, so the reader of the spec drain is named.
struct Reader {
	uint64_t    hash  = 0;
	const char* stage = "?";
};
inline Reader& CurrentReaderRef() {
	static thread_local Reader reader;
	return reader;
}
class ReaderScope {
public:
	ReaderScope(uint64_t hash, const char* stage): m_previous(CurrentReaderRef()) { CurrentReaderRef() = Reader {hash, stage}; }
	~ReaderScope() { CurrentReaderRef() = m_previous; }
	ReaderScope(const ReaderScope&)            = delete;
	ReaderScope& operator=(const ReaderScope&) = delete;

private:
	Reader m_previous;
};

// Last tick (the command buffer being recorded) at which each 4 KiB page was marked GPU-written.
class TickPages {
public:
	static constexpr uint64_t kPageSize = 4096;

	void Note(uint64_t address, uint64_t size, uint64_t tick) {
		if (size == 0) {
			return;
		}
		for (auto page = address / kPageSize; page <= (address + size - 1) / kPageSize; page++) {
			auto& slot = m_pages[page];
			slot       = slot > tick ? slot : tick;
		}
	}
	// Newest write tick over the pages the range touches; 0 = no GPU write known.
	[[nodiscard]] uint64_t MaxTick(uint64_t address, uint64_t size) const {
		if (size == 0) {
			return 0;
		}
		uint64_t result = 0;
		for (auto page = address / kPageSize; page <= (address + size - 1) / kPageSize; page++) {
			const auto found = m_pages.find(page);
#if defined(KYTY_SYNCCENSUS_MUTANT_TICK_IGNORED)
			(void)found;
#else
			if (found != m_pages.end() && found->second > result) {
				result = found->second;
			}
#endif
		}
		return result;
	}
	// A download cleans the pages the range covers completely; the partial pages at its edges keep their tick.
	void Forget(uint64_t address, uint64_t size) {
		if (size == 0) {
			return;
		}
		for (auto page = (address + kPageSize - 1) / kPageSize; (page + 1) * kPageSize <= address + size; page++) {
			m_pages.erase(page);
		}
	}
	[[nodiscard]] size_t Pages() const { return m_pages.size(); }

private:
	std::unordered_map<uint64_t, uint64_t> m_pages;
};

// Ticks between the buffer being recorded and the newest writer of a range (0: the writer sits in the same buffer, or none is known).
constexpr uint64_t Lag(uint64_t current_tick, uint64_t writer_tick) {
	return writer_tick == 0 || writer_tick > current_tick ? 0 : current_tick - writer_tick;
}
// The writer is complete on the GPU (a readback would not have to wait for it).
constexpr bool WriterDone(uint64_t writer_tick, uint64_t known_gpu_tick) {
	return writer_tick == 0 || known_gpu_tick >= writer_tick;
}

// What one served readback saw (filled on the GPU thread).
struct Probe {
	bool     downloaded = false;
	uint64_t writer     = 0; // newest GPU write tick over the range, 0 = unknown
	uint64_t current    = 0; // tick of the buffer being recorded
	bool     done       = false;
	bool     repeat     = false;
	bool     same       = false;
	uint64_t write_seq     = 0;     // S402 (spec): newest write sequence number over the range (0 = none seen)
	bool     written_since = false; // S402 (spec): the range was written since the previous spec read of it
};

// draws + dispatches recorded when the last command buffer was submitted (for the [sync] log).
inline std::atomic<uint64_t>& CommandsAtSubmit() {
	static std::atomic<uint64_t> value {0};
	return value;
}

// Metadata read twice in one flip: key = range, remembers the flip and a hash of what the CPU saw afterwards.
class RepeatTracker {
public:
	struct Result {
		bool repeat;
		bool same;
	};
	Result Note(uint64_t address, uint64_t size, uint64_t flip, uint64_t hash) {
		const auto key   = address ^ (size * 0x9e3779b97f4a7c15ull);
		auto&      entry = m_entries[key];
		Result     result {entry.valid && entry.flip == flip && entry.address == address && entry.size == size, false};
		result.same = result.repeat && entry.hash == hash;
		entry       = {true, address, size, flip, hash};
		if (m_entries.size() > kMaxEntries) {
			m_entries.clear();
		}
		return result;
	}

private:
	static constexpr size_t kMaxEntries = 8192;
	struct Entry {
		bool     valid   = false;
		uint64_t address = 0;
		uint64_t size    = 0;
		uint64_t flip    = 0;
		uint64_t hash    = 0;
	};
	std::unordered_map<uint64_t, Entry> m_entries;
};

inline uint64_t HashBytes(const void* data, uint64_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	uint64_t    hash  = 1469598103934665603ull;
	for (uint64_t i = 0; i < size; i++) {
		hash = (hash ^ bytes[i]) * 1099511628211ull;
	}
	return hash;
}

} // namespace Libs::Graphics::SyncCensus

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_SYNCCENSUS_H_ */
