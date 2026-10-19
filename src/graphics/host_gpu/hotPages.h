#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_HOTPAGES_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_HOTPAGES_H_

// S440 (KYTY_WF_HOT_UNPROTECT=0|1|shadow, default 0): a guest page the CPU writes in many flips in a row stops being write-protected by the buffer
// tracker. The write fault (~20 us, SIGBUS under Rosetta) is replaced by a hash of the page taken at every upload: a different hash is treated as
// "CPU-modified", the same hash as "nothing to upload".
//   promote:   a page whose write fault repeats in kHotStreak consecutive guest flips becomes hot (its CPU-dirty bit then stays set: unprotected).
//   consume:   ForEachUploadRange hashes each hot dirty page; changed -> uploaded, unchanged -> skipped. A page is not hot while the GPU writes it.
//   demote:    a hot page whose hash did not change for kColdAfter flips is protected again (and hashed once more after the protection took effect).
//   shadow:    protection and faults stay as they are; the hashes are only compared (hot_same / hot_changed), and a page that is hot, clean and
//              whose hash differs from the one of its last upload is counted in hot_ne (a write the tracker did not see; must be 0).
// This header is the pure part (tests/MemoryTrackerTests.cpp pins it). The page-protection of other watchers (images) is untouched: a page an image
// watches keeps faulting, which marks the image dirty and the page's hash stale.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::HotPages {

enum class Mode : int { Off = 0, On = 1, Shadow = 2 };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) {
		return Mode::Off;
	}
	if (std::strcmp(text, "1") == 0) {
		return Mode::On;
	}
	if (std::strcmp(text, "shadow") == 0) {
		return Mode::Shadow;
	}
	return Mode::Off;
}

// Tests set this to a Mode value; -1 = use the environment.
inline std::atomic<int>& ModeOverride() {
	static std::atomic<int> value {-1};
	return value;
}

inline Mode CurrentMode() {
	const int forced = ModeOverride().load(std::memory_order_relaxed);
	if (forced >= 0) {
		return static_cast<Mode>(forced);
	}
	static const Mode env = ParseMode(std::getenv("KYTY_WF_HOT_UNPROTECT"));
	return env;
}

constexpr uint32_t kHotStreak = 8;  // consecutive flips with a write fault on the page
constexpr uint32_t kColdAfter = 64; // flips without a changed hash before the protection returns

// Guest flip number; the renderer installs FrameTiming::GuestFlips, tests set their own.
using EpochFn = uint64_t (*)();
inline std::atomic<EpochFn>& EpochSource() {
	static std::atomic<EpochFn> source {nullptr};
	return source;
}
inline uint32_t Epoch() {
	const auto fn = EpochSource().load(std::memory_order_relaxed);
	return fn != nullptr ? static_cast<uint32_t>(fn()) : 0u;
}

// Hash of the 4 KiB page at `page`; false when it cannot be read (then the page counts as changed).
using HashFn = bool (*)(uint64_t page, uint64_t* hash);
inline std::atomic<HashFn>& HasherSource() {
	static std::atomic<HashFn> source {nullptr};
	return source;
}

// DIAG (KYTY_DIAG_WRITE_FAULT): called for every range ForEachUploadRange hands to the upload.
using UploadObserverFn = void (*)(uint64_t address, uint64_t size);
inline std::atomic<UploadObserverFn>& UploadObserver() {
	static std::atomic<UploadObserverFn> observer {nullptr};
	return observer;
}

struct Stats {
	std::atomic<uint64_t> promote {0};   // pages that became hot
	std::atomic<uint64_t> demote {0};    // hot pages that left the hot state
	std::atomic<uint64_t> hash {0};      // page hashes taken
	std::atomic<uint64_t> hash_ns {0};   // time of them
	std::atomic<uint64_t> skip {0};      // dirty hot pages whose hash was unchanged (no upload)
	std::atomic<uint64_t> changed {0};   // dirty hot pages whose hash changed (uploaded)
	std::atomic<uint64_t> ne {0};        // shadow: clean hot page whose hash differs from its last upload (must be 0)
	std::atomic<uint64_t> race {0};      // a page changed between its last hash and the return of the protection
	std::atomic<int64_t>  hot_pages {0}; // gauge: pages hot now
};

inline Stats& GetStats() {
	static Stats stats;
	return stats;
}

} // namespace Libs::Graphics::HotPages

// S443 (KYTY_WF_FLIP_UNPROTECT=0|1|shadow, default 0): a page that takes its SECOND write fault in one guest flip stays unprotected (CPU-dirty) until the
// flip ends: the third and later faults of the flip are gone. Every upload copies a dirty page (no hash decides anything), so a kept page is correct by
// construction; the first upload after the flip changed consumes the dirty bit and protects the page again (the copy of that upload is taken after the
// protection, as for any page).
//   shadow: protection and faults stay as they are; the pages that WOULD be kept ("candidates") are hashed right after the upload protected them, and a
//           later upload check of the same flip rehashes every candidate that is still clean. A different hash means a write the tracker never saw;
//           it is hashed once more at once: same value again = ne (a real unseen write; the observer logs the address), another value = ne_unstable
//           (the page is being written by a host thread that does not fault; reported, not counted in ne).
namespace Libs::Graphics::FlipUnprotect {

using Mode = HotPages::Mode;

inline Mode ParseMode(const char* text) { return HotPages::ParseMode(text); }

inline std::atomic<int>& ModeOverride() {
	static std::atomic<int> value {-1};
	return value;
}

inline Mode CurrentMode() {
	const int forced = ModeOverride().load(std::memory_order_relaxed);
	if (forced >= 0) {
		return static_cast<Mode>(forced);
	}
	static const Mode env = ParseMode(std::getenv("KYTY_WF_FLIP_UNPROTECT"));
	return env;
}

// DIAG/log: a shadow mismatch (address of the page, whether the second hash agreed with the first).
using NeObserverFn = void (*)(uint64_t page, bool stable);
inline std::atomic<NeObserverFn>& NeObserver() {
	static std::atomic<NeObserverFn> observer {nullptr};
	return observer;
}

struct Stats {
	std::atomic<uint64_t> refault {0};     // write faults on a page that already faulted in this flip (the second and later ones)
	std::atomic<uint64_t> promote {0};     // pages kept unprotected until the end of their flip
	std::atomic<uint64_t> demote {0};      // kept pages protected again at the first upload after their flip
	std::atomic<uint64_t> keep {0};        // uploads of a kept page that left it unprotected
	std::atomic<uint64_t> cand {0};        // shadow: pages that would have been kept
	std::atomic<uint64_t> hash {0};        // shadow: page hashes taken
	std::atomic<uint64_t> ne {0};          // shadow: candidate page whose bytes changed without a fault, twice the same value (must be 0)
	std::atomic<uint64_t> ne_unstable {0}; // shadow: ... and the second hash differed from the first
	std::atomic<int64_t>  pages {0};       // gauge: pages kept now
};

inline Stats& GetStats() {
	static Stats stats;
	return stats;
}

} // namespace Libs::Graphics::FlipUnprotect

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_HOTPAGES_H_
