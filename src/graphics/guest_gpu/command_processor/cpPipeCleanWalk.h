#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPECLEANWALK_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPPIPECLEANWALK_H

// S607 (KYTY_WORKER_CLEAN_WALK=0|1|shadow, unset = 0; KYTY_WORKER_CLEAN_WALK_SPIN = tries of the lock spin, default 256).
// A pipeline worker's walker load on a hot page (walkNoFault.h: a page whose walker load the CP served from the backing before) used to take
// the ordinary load, fault and complete as zero; the job then failed (fail7) and the recorder computed the draw's programs itself
// (S606 data: ~375 draws/flip in the garden, 3.1-3.6 us of GetGraphicsPrograms on the recorder each, plus the lost predecode).
// 1: the worker asks the same question the CP's clean walk asks (cleanWalk.h: bytes not GPU-dirty, not in flight, the tracker answered, with
// a bounded spin on its locks instead of one try) and takes the backing's dword. The answer is only as old as the worker's walk, so the
// recorder checks it again when it joins the job, at the time the CP itself would have walked: the served bytes (and every read of the walk
// on such a page) must not be GPU-dirty or in flight now, and no write into the backing may have touched their pages since the job started.
// Then the dword is the one the CP's clean walk would serve, and the job's result is the CP's. A page of a served load is checked byte by
// byte (it is protected because the GPU wrote *other* bytes of it, so the page check of S490 would always refuse it); a served page the
// worker also read through the backing (TryReadBackingWorker) keeps the page check.
// shadow: the workers serve as in 1; the recorder computes the programs of every job with a served load itself, compares them with the
// worker's (wcw_shadow_ne_n must stay 0) and counts the time that mode 1 would not spend (wcw_shadow_cp_ns).
// Active only with KYTY_CP_PIPE=1 and workers, KYTY_RB_CLEAN_WALK=1 (the CP's own decision) and KYTY_WALK_NOFAULT=1 (the hot-page check
// that routes the worker's load to the reader). Not codegen: the served dword is the CP's, SPIR-V does not change.
//
// Mutants (tests/CpPipeCleanWalkTests.cpp, WILL_FAIL): KYTY_WCW_MUTANT_BUSY_SERVES (a load whose query gave up is served),
// KYTY_WCW_MUTANT_JOIN_NO_BYTES (the join does not check the served bytes), KYTY_WCW_MUTANT_JOIN_NO_INFLIGHT (the join's byte check forgets
// copies in flight), KYTY_WCW_MUTANT_JOIN_NO_WRITTEN (the pages of served loads are not asked for backing writes), KYTY_WCW_MUTANT_SPLIT_DROPS_READ
// (a non-served read on a served page is checked neither by page nor by bytes), KYTY_WCW_MUTANT_PAGE_CHECKS_SERVED (the served pages keep
// the page check: never wrong, but nothing is gained), KYTY_WCW_MUTANT_SPLIT_IGNORES_BACKING (a served page also read through the backing
// is checked by bytes only).

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics::CpPipe::WorkerCleanWalk {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode Parse(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline Mode EnvMode() {
	static const Mode mode = Parse(std::getenv("KYTY_WORKER_CLEAN_WALK"));
	return mode;
}

inline const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::On: return "1";
		case Mode::Shadow: return "shadow";
		default: return "0";
	}
}

constexpr int kDefaultSpin = 256;

inline int ParseSpin(const char* text) {
	if (text == nullptr || *text == '\0') return kDefaultSpin;
	char*      end   = nullptr;
	const long value = std::strtol(text, &end, 10);
	if (end == nullptr || *end != '\0' || value < 1 || value > 65536) return kDefaultSpin;
	return static_cast<int>(value);
}

inline int EnvSpin() {
	static const int spin = ParseSpin(std::getenv("KYTY_WORKER_CLEAN_WALK_SPIN"));
	return spin;
}

// The switch is in effect (set once by the CP when the pipeline starts, before the workers).
constexpr bool ActiveFor(Mode mode, bool pipe_on_with_workers, bool clean_walk_on, bool nofault_on) {
	return mode != Mode::Off && pipe_on_with_workers && clean_walk_on && nofault_on;
}

inline std::atomic<bool>& ActiveFlag() {
	static std::atomic<bool> active {false};
	return active;
}
inline bool Active() {
	return ActiveFlag().load(std::memory_order_relaxed);
}

// --- the worker's decision for one load (the CP's CleanWalk::Decide, without its shadow) ---
enum class Answer : uint8_t { Serve, Dirty, Busy };

// clean_walk_on: KYTY_RB_CLEAN_WALK=1 (the CP serves such a load itself; anything else drains it). queried: the tracker answered (false: the
// lock spin gave up or the range is not the GPU's).
constexpr Answer Decide(bool clean_walk_on, bool queried, bool dirty, bool inflight) {
	if (!clean_walk_on) return Answer::Dirty;
#if defined(KYTY_WCW_MUTANT_BUSY_SERVES)
	queried = true;
#endif
	if (!queried) return Answer::Busy;
	if (dirty || inflight) return Answer::Dirty;
	return Answer::Serve;
}

// --- what one worker served during the walk of its current job ---
struct ThreadLog {
	bool                  active = false;
	std::vector<uint64_t> served; // addresses of the served dwords, in load order
	uint32_t              busy  = 0;
	uint32_t              dirty = 0;

	void Begin() {
		served.clear();
		busy   = 0;
		dirty  = 0;
		active = true;
	}
	void End() { active = false; }
};

inline ThreadLog& WorkerLog() {
	static thread_local ThreadLog log;
	return log;
}

// --- after the walk: which reads the join checks by page, which by bytes ---
constexpr uint64_t kPage = 4096;

constexpr uint64_t PageOf(uint64_t address) {
	return address & ~(kPage - 1);
}

inline void SortUnique(std::vector<uint64_t>& values) {
	std::sort(values.begin(), values.end());
	values.erase(std::unique(values.begin(), values.end()), values.end());
}

// `pages`: in, the pages the worker read through the backing (TryReadBackingWorker's log, any order); out, every page the join checks as a
// page (sorted, unique). `served`: the served dword addresses. `reads`: every read range of the walk (the snapshots' specialization_reads; the
// served loads are among them, the function adds them anyway). `bytes`: out, the ranges the join checks byte by byte (sorted by address, merged).
// `cw_pages`: out, their pages, which the join asks for backing writes too (sorted, unique; a page may be in both lists when a plain read and a
// read across a byte page share it: then both checks run). False: a range spans more
// than `max_pages` pages or wraps (the job is left to the CP, as AppendPages decides).
inline bool SplitReads(std::vector<uint64_t>& pages, std::span<const uint64_t> served, std::span<const std::pair<uint64_t, uint64_t>> reads,
                       std::vector<std::pair<uint64_t, uint64_t>>& bytes, std::vector<uint64_t>& cw_pages, size_t max_pages) {
	bytes.clear();
	cw_pages.clear();
	SortUnique(pages);
	std::vector<uint64_t> byte_pages;
	byte_pages.reserve(served.size());
	for (const uint64_t address: served) byte_pages.push_back(PageOf(address));
	SortUnique(byte_pages);
#if !defined(KYTY_WCW_MUTANT_SPLIT_IGNORES_BACKING)
	// A served page the worker also read through the backing (a specialization or fetch read, logged as a page) keeps the page check.
	byte_pages.erase(std::remove_if(byte_pages.begin(), byte_pages.end(),
	                                [&](uint64_t page) { return std::binary_search(pages.begin(), pages.end(), page); }),
	                 byte_pages.end());
#endif
#if defined(KYTY_WCW_MUTANT_PAGE_CHECKS_SERVED)
	for (const uint64_t page: byte_pages) pages.push_back(page);
	byte_pages.clear();
#endif
	const auto is_byte_page = [&](uint64_t page) { return std::binary_search(byte_pages.begin(), byte_pages.end(), page); };
	// No-op by construction (a byte page was not read through the backing); with the mutant it drops that page check.
	pages.erase(std::remove_if(pages.begin(), pages.end(), is_byte_page), pages.end());
	const auto add          = [&](uint64_t address, uint64_t size) {
        if (size == 0) return true;
        if (UINT64_MAX - address < size) return false;
        const uint64_t first = PageOf(address);
        const uint64_t last  = PageOf(address + size - 1);
        if ((last - first) / kPage >= max_pages) return false;
        bool touches = false;
        for (uint64_t page = first; page <= last; page += kPage) touches = touches || is_byte_page(page);
        if (!touches) {
            for (uint64_t page = first; page <= last; page += kPage) pages.push_back(page);
            return true;
        }
#if defined(KYTY_WCW_MUTANT_SPLIT_DROPS_READ)
        if (size != 4 || std::find(served.begin(), served.end(), address) == served.end()) return true;
#endif
        bytes.emplace_back(address, size);
        for (uint64_t page = first; page <= last; page += kPage) cw_pages.push_back(page);
        return true;
	};
	for (const auto& [address, size]: reads) {
		if (!add(address, size)) return false;
	}
	for (const uint64_t address: served) {
		if (!add(address, 4)) return false;
	}
	SortUnique(pages);
	SortUnique(cw_pages);
	// Overlapping and touching ranges are merged (a descriptor is 4 or 8 consecutive dwords): the same bytes, fewer questions in the join.
	std::sort(bytes.begin(), bytes.end());
	size_t out = 0;
	for (size_t i = 0; i < bytes.size(); i++) {
		if (out != 0 && bytes[i].first <= bytes[out - 1].first + bytes[out - 1].second) {
			const uint64_t end      = std::max(bytes[out - 1].first + bytes[out - 1].second, bytes[i].first + bytes[i].second);
			bytes[out - 1].second = end - bytes[out - 1].first;
		} else {
			bytes[out++] = bytes[i];
		}
	}
	bytes.resize(out);
	return pages.size() + cw_pages.size() <= max_pages;
}

// --- the recorder's join ---
enum class Verdict : uint8_t { Ok, PageDirty, BytesDirty, Written };

// page_dirty(page): the page holds GPU-written bytes the backing does not have (S490 JoinJob). bytes_busy(address, size): the bytes are
// GPU-dirty or a GPU->host copy of them is in flight (the CP's clean-walk question, asked on the CP). written(pages): a write into the backing
// that ended after the job started (or has not ended) touched one of the pages. With `bytes` empty this is JoinJob of S490 unchanged.
template <class PageDirty, class BytesBusy, class Written>
Verdict Join(const std::vector<uint64_t>& pages, const std::vector<std::pair<uint64_t, uint64_t>>& bytes, const std::vector<uint64_t>& cw_pages,
             PageDirty&& page_dirty, BytesBusy&& bytes_busy, Written&& written) {
	for (const uint64_t page: pages) {
		if (page_dirty(page)) return Verdict::PageDirty;
	}
#if !defined(KYTY_WCW_MUTANT_JOIN_NO_BYTES)
	for (const auto& [address, size]: bytes) {
		if (bytes_busy(address, size)) return Verdict::BytesDirty;
	}
#endif
	if (!pages.empty() && written(pages)) return Verdict::Written;
#if !defined(KYTY_WCW_MUTANT_JOIN_NO_WRITTEN)
	if (!cw_pages.empty() && written(cw_pages)) return Verdict::Written;
#endif
	return Verdict::Ok;
}

// The byte question of the join from the two answers of the tracker (the mutant forgets the copies in flight).
constexpr bool BytesBusy(bool dirty, bool inflight) {
#if defined(KYTY_WCW_MUTANT_JOIN_NO_INFLIGHT)
	(void)inflight;
	return dirty;
#else
	return dirty || inflight;
#endif
}

} // namespace Libs::Graphics::CpPipe::WorkerCleanWalk

#endif
