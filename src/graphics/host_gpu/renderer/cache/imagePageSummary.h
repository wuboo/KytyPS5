#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_IMAGEPAGESUMMARY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_IMAGEPAGESUMMARY_H_

// S443 (KYTY_WF_FAST_HANDLER=0|1|shadow, default 0): lock-free "does any registered image cover this 4 KiB page?" for the write-fault handler.
// A counting filter keyed by the 4 KiB page number: registration adds one to the bucket of every page of the image's data range, unregistration
// takes it away. Buckets collide, so a non-zero bucket only means "maybe" (the handler then takes TextureCache::m_lock as ever); a zero bucket
// means no image covers the page, and the handler skips the spinlock and FindImagesInRegion. Add/Remove run with TextureCache::m_lock held
// (they are paired with the page owner index), MayCover runs without it.
// This header is the pure part (tests/ImagePageSummaryTests.cpp pins it).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace Libs::Graphics::FastHandler {

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
	static const Mode env = ParseMode(std::getenv("KYTY_WF_FAST_HANDLER"));
	return env;
}

struct Stats {
	std::atomic<uint64_t> fast {0};   // write faults answered without the texture-cache lock (no image on the page)
	std::atomic<uint64_t> slow {0};   // write faults that took the lock because a bucket was non-zero
	std::atomic<uint64_t> ne {0};     // shadow: the filter said "no image" but the page owner index found one (must be 0)
};

inline Stats& GetStats() {
	static Stats stats;
	return stats;
}

class ImagePageSummary {
public:
	static constexpr uint64_t kPageBits   = 12;
	static constexpr uint64_t kBucketBits = 20;

	ImagePageSummary() : m_buckets(static_cast<std::atomic<uint32_t>*>(std::calloc(size_t {1} << kBucketBits, sizeof(std::atomic<uint32_t>)))) {}

	void Add(uint64_t address, uint64_t size) { Walk(address, size, +1); }
	void Remove(uint64_t address, uint64_t size) { Walk(address, size, -1); }

	// false: no registered image covers any page of [address, address + size).
	[[nodiscard]] bool MayCover(uint64_t address, uint64_t size) const {
		if (size == 0) {
			return false;
		}
#if defined(KYTY_IMAGEPAGESUMMARY_MUTANT_NEVER_COVERS)
		return false;
#endif
		const uint64_t last = (address + size - 1) >> kPageBits;
		for (uint64_t page = address >> kPageBits; page <= last; ++page) {
			if (m_buckets[Bucket(page)].load(std::memory_order_acquire) != 0) {
				return true;
			}
		}
		return false;
	}

private:
	static uint64_t Bucket(uint64_t page) {
		page *= 0x9e3779b97f4a7c15ull;
		return page >> (64 - kBucketBits);
	}

	void Walk(uint64_t address, uint64_t size, int delta) {
		if (size == 0 || m_buckets == nullptr) {
			return;
		}
		const uint64_t last = (address + size - 1) >> kPageBits;
		for (uint64_t page = address >> kPageBits; page <= last; ++page) {
#if defined(KYTY_IMAGEPAGESUMMARY_MUTANT_NO_ADD)
			if (delta > 0) {
				continue;
			}
#endif
			auto& bucket = m_buckets[Bucket(page)];
			if (delta > 0) {
				bucket.fetch_add(1, std::memory_order_release);
			} else {
				bucket.fetch_sub(1, std::memory_order_release);
			}
		}
	}

	struct FreeDeleter {
		void operator()(std::atomic<uint32_t>* ptr) const { std::free(ptr); }
	};
	std::unique_ptr<std::atomic<uint32_t>[], FreeDeleter> m_buckets;
};

} // namespace Libs::Graphics::FastHandler

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_CACHE_IMAGEPAGESUMMARY_H_
