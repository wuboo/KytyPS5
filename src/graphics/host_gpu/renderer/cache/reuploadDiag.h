#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_REUPLOADDIAG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_REUPLOADDIAG_H_

// S532 DIAG. KYTY_DIAG_S532=1 counts and logs, inside guest flips 4000–4700,
// CPU-alias marks and guest uploads of image 0x514080000, and the same events
// for color targets of blend-enabled draws in that window. A fast clear and
// the copy pixel shader 3bf81ef1 are marked so a log line can say whether an
// upload landed between them. Unset: one getenv, then nothing.

#include "graphics/host_gpu/renderer/cache/cpuAliasPages.h"
#include "graphics/presentation/frameTiming.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ReuploadDiag {

inline constexpr uint64_t kCard    = 0x514080000ull;
inline constexpr uint64_t kCopyPs  = 0x3bf81ef1b8147b45ull;
inline constexpr uint64_t kLo      = 4000;
inline constexpr uint64_t kHi      = 4700;
inline constexpr uint32_t kBlendCap = 64;

inline bool On() {
	static const bool on = [] {
		const char* value = std::getenv("KYTY_DIAG_S532");
		const bool  enabled = value != nullptr && std::strcmp(value, "1") == 0;
		if (enabled) {
			std::fprintf(stderr, "[s532] cfg diag=1 fix=%d window=%llu-%llu card=0x%llx\n",
			             CpuAliasPages::FixOn() ? 1 : 0,
			             static_cast<unsigned long long>(kLo),
			             static_cast<unsigned long long>(kHi),
			             static_cast<unsigned long long>(kCard));
			std::fflush(stderr);
		}
		return enabled;
	}();
	return on;
}

inline uint64_t Guest() { return FrameTiming::GuestFlips() + 1; }

inline bool InWindow() {
	const uint64_t guest = Guest();
	return guest >= kLo && guest <= kHi;
}

struct BlendSet {
	std::atomic<uint32_t> n {0};
	uint64_t              addr[kBlendCap] {};
};

inline BlendSet& Blends() {
	static BlendSet set;
	return set;
}

inline void NoteBlend(uint64_t address) {
	if (address == 0 || address == kCard) {
		return;
	}
	auto& set = Blends();
	const uint32_t n = set.n.load(std::memory_order_relaxed);
	for (uint32_t i = 0; i < n && i < kBlendCap; i++) {
		if (set.addr[i] == address) {
			return;
		}
	}
	const uint32_t slot = set.n.fetch_add(1, std::memory_order_relaxed);
	if (slot < kBlendCap) {
		set.addr[slot] = address;
	}
}

inline bool IsBlend(uint64_t address) {
	auto& set = Blends();
	const uint32_t n = set.n.load(std::memory_order_relaxed);
	const uint32_t lim = n < kBlendCap ? n : kBlendCap;
	for (uint32_t i = 0; i < lim; i++) {
		if (set.addr[i] == address) {
			return true;
		}
	}
	return false;
}

struct Counts {
	std::atomic<uint64_t> dirty {0};
	std::atomic<uint64_t> maybe {0};
	std::atomic<uint64_t> head {0};
	std::atomic<uint64_t> tail {0};
	std::atomic<uint64_t> upload_cpu {0};
	std::atomic<uint64_t> upload_maybe {0};
	std::atomic<uint64_t> upload_buffer {0};
	std::atomic<uint64_t> upload_first {0};
	std::atomic<uint64_t> clear {0};
	std::atomic<uint64_t> copy {0};
	std::atomic<uint64_t> since_clear {0};
	std::atomic<uint64_t> last_upload_guest {0};
	std::atomic<uint32_t> last_src {0};
	std::atomic<uint64_t> blend_alias {0};
	std::atomic<uint64_t> blend_upload {0};
};

inline Counts& Card() {
	static Counts counts;
	return counts;
}

enum Src : uint32_t { SrcNone = 0, SrcCpu = 1, SrcMaybe = 2, SrcBuffer = 3, SrcFirst = 4 };

inline const char* SrcName(uint32_t src) {
	switch (src) {
		case SrcCpu: return "cpu";
		case SrcMaybe: return "maybe";
		case SrcBuffer: return "buffer";
		case SrcFirst: return "first";
		default: return "none";
	}
}

inline uint32_t SrcOf(const char* name) {
	if (name == nullptr) return SrcNone;
	if (std::strcmp(name, "cpu") == 0) return SrcCpu;
	if (std::strcmp(name, "maybe") == 0) return SrcMaybe;
	if (std::strcmp(name, "buffer") == 0) return SrcBuffer;
	if (std::strcmp(name, "first") == 0) return SrcFirst;
	return SrcNone;
}

inline bool Watched(uint64_t address) {
	if (!On() || !InWindow()) {
		return false;
	}
	return address == kCard || IsBlend(address);
}

inline void CountKind(Counts& counts, const char* kind) {
	if (std::strcmp(kind, "dirty") == 0) counts.dirty.fetch_add(1, std::memory_order_relaxed);
	else if (std::strcmp(kind, "maybe") == 0) counts.maybe.fetch_add(1, std::memory_order_relaxed);
	else if (std::strcmp(kind, "head") == 0) counts.head.fetch_add(1, std::memory_order_relaxed);
	else if (std::strcmp(kind, "tail") == 0) counts.tail.fetch_add(1, std::memory_order_relaxed);
}

// First four alias lines per guest flip for the card. Blend targets are counted only.
inline void OnAlias(uint64_t image, uint64_t image_size, uint64_t write, uint64_t write_size,
                    const char* kind) {
	if (!Watched(image)) {
		return;
	}
	if (image != kCard) {
		Card().blend_alias.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	CountKind(Card(), kind);
	static std::atomic<uint64_t> line_flip {0};
	static std::atomic<uint32_t> line_n {0};
	const uint64_t guest = Guest();
	uint64_t       seen  = line_flip.load(std::memory_order_relaxed);
	if (seen != guest) {
		line_flip.store(guest, std::memory_order_relaxed);
		line_n.store(0, std::memory_order_relaxed);
	}
	if (line_n.fetch_add(1, std::memory_order_relaxed) >= 4) {
		return;
	}
	std::fprintf(stderr,
	             "[s532] alias guest=%llu kind=%s img=0x%llx+%llx write=0x%llx+%llx\n",
	             static_cast<unsigned long long>(guest), kind,
	             static_cast<unsigned long long>(image),
	             static_cast<unsigned long long>(image_size),
	             static_cast<unsigned long long>(write),
	             static_cast<unsigned long long>(write_size));
}

inline void OnUpload(uint64_t image, const char* src, bool first) {
	if (!Watched(image)) {
		return;
	}
	if (image != kCard) {
		Card().blend_upload.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	auto& counts = Card();
	const uint32_t code = SrcOf(src);
	if (code == SrcCpu) counts.upload_cpu.fetch_add(1, std::memory_order_relaxed);
	else if (code == SrcMaybe) counts.upload_maybe.fetch_add(1, std::memory_order_relaxed);
	else if (code == SrcBuffer) counts.upload_buffer.fetch_add(1, std::memory_order_relaxed);
	if (first) counts.upload_first.fetch_add(1, std::memory_order_relaxed);
	counts.since_clear.fetch_add(1, std::memory_order_relaxed);
	const uint64_t guest = Guest();
	counts.last_upload_guest.store(guest, std::memory_order_relaxed);
	counts.last_src.store(first && code == SrcNone ? SrcFirst : code, std::memory_order_relaxed);
	std::fprintf(stderr, "[s532] upload guest=%llu img=0x%llx src=%s first=%d since_clear=%llu\n",
	             static_cast<unsigned long long>(guest), static_cast<unsigned long long>(image),
	             src != nullptr ? src : "none", first ? 1 : 0,
	             static_cast<unsigned long long>(counts.since_clear.load(std::memory_order_relaxed)));
	std::fflush(stderr);
}

inline void OnClear(uint64_t image) {
	if (!Watched(image) || image != kCard) {
		return;
	}
	auto& counts = Card();
	counts.clear.fetch_add(1, std::memory_order_relaxed);
	counts.since_clear.store(0, std::memory_order_relaxed);
	const uint64_t guest = Guest();
	std::fprintf(stderr, "[s532] clear guest=%llu img=0x%llx n=%llu\n",
	             static_cast<unsigned long long>(guest), static_cast<unsigned long long>(image),
	             static_cast<unsigned long long>(counts.clear.load(std::memory_order_relaxed)));
	std::fflush(stderr);
}

inline void OnCopy() {
	if (!On() || !InWindow()) {
		return;
	}
	auto& counts = Card();
	const uint64_t n = counts.copy.fetch_add(1, std::memory_order_relaxed);
	const uint64_t guest = Guest();
	static std::atomic<uint64_t> logged_flip {0};
	static std::atomic<uint32_t> logged_n {0};
	uint64_t seen = logged_flip.load(std::memory_order_relaxed);
	if (seen != guest) {
		logged_flip.store(guest, std::memory_order_relaxed);
		logged_n.store(0, std::memory_order_relaxed);
	}
	const uint64_t since = counts.since_clear.load(std::memory_order_relaxed);
	if (logged_n.fetch_add(1, std::memory_order_relaxed) >= 4 && since == 0) {
		return;
	}
	std::fprintf(stderr,
	             "[s532] copy guest=%llu n=%llu since_clear=%llu last_src=%s last_guest=%llu "
	             "dirty=%llu maybe=%llu head=%llu tail=%llu up_cpu=%llu up_maybe=%llu "
	             "up_buf=%llu up_first=%llu clears=%llu blend_alias=%llu blend_up=%llu\n",
	             static_cast<unsigned long long>(guest), static_cast<unsigned long long>(n + 1),
	             static_cast<unsigned long long>(since),
	             SrcName(counts.last_src.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.last_upload_guest.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.dirty.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.maybe.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.head.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.tail.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.upload_cpu.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.upload_maybe.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.upload_buffer.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.upload_first.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.clear.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.blend_alias.load(std::memory_order_relaxed)),
	             static_cast<unsigned long long>(counts.blend_upload.load(std::memory_order_relaxed)));
	std::fflush(stderr);
}

// Logs the copy shader after the draw's binds, so an upload during setup is already counted.
class DrawScope {
public:
	DrawScope() = default;
	void Arm() { m_copy = true; }
	~DrawScope() {
		if (m_copy) OnCopy();
	}
	DrawScope(const DrawScope&) = delete;
	DrawScope& operator=(const DrawScope&) = delete;

private:
	bool m_copy = false;
};

inline thread_local const char* t_source = nullptr;

inline void BeginSource(const char* src) {
	if (!On()) return;
	t_source = src;
}

inline const char* SourceOr(const char* fallback) {
	return t_source != nullptr ? t_source : fallback;
}

inline void EndSource() { t_source = nullptr; }

} // namespace Libs::Graphics::ReuploadDiag

#endif
