#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_GALAXYBARRIER_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_GALAXYBARRIER_H_

// S527 DIAG. KYTY_DIAG_FULL_BARRIER=1 inserts a memory dependency in the guest-flip window
// KYTY_DIAG_BARRIER_FLIP (default 4100-4500). Unset: one getenv, then a bool, no barrier.
// KYTY_DIAG_BARRIER_KIND selects which dependency (default full):
//   full      ALL_COMMANDS/MEMORY_WRITE -> ALL_COMMANDS/MEMORY_READ|WRITE after every draw and dispatch
//   draw      the same barrier after draws only
//   dispatch  the same barrier after dispatches only
//   xfer      TRANSFER|HOST write -> ALL_COMMANDS memory read|write before every draw
//   ssbo      do not defer the existing buffer-write barrier (L1 off inside the window)
//   5e        the full barrier only before pixel shader 5e7524f281e7d196
//   submit    wait for the GPU after every submit inside the window
//   drain     end the pass and wait for the GPU after every draw inside the window, no texture read
// Not a fix. The recorded draws stay the same; only the dependency changes, and only while the
// variable is 1. Not codegen.

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/presentation/frameTiming.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::GalaxyBarrier {

inline constexpr uint64_t kShader5e = 0x5e7524f281e7d196ull;

enum class Kind { Off, Full, Draw, Dispatch, Xfer, Ssbo, Shader5e, Submit, Drain };

struct Config {
	Kind kind = Kind::Off;
	long lo   = 4100;
	long hi   = 4500;
};

inline const char* KindName(Kind kind) {
	switch (kind) {
		case Kind::Off: return "off";
		case Kind::Full: return "full";
		case Kind::Draw: return "draw";
		case Kind::Dispatch: return "dispatch";
		case Kind::Xfer: return "xfer";
		case Kind::Ssbo: return "ssbo";
		case Kind::Shader5e: return "5e";
		case Kind::Submit: return "submit";
		case Kind::Drain: return "drain";
	}
	return "off";
}

inline Kind ParseKind(const char* text) {
	if (text == nullptr || text[0] == '\0' || std::strcmp(text, "full") == 0) return Kind::Full;
	if (std::strcmp(text, "draw") == 0) return Kind::Draw;
	if (std::strcmp(text, "dispatch") == 0) return Kind::Dispatch;
	if (std::strcmp(text, "xfer") == 0) return Kind::Xfer;
	if (std::strcmp(text, "ssbo") == 0) return Kind::Ssbo;
	if (std::strcmp(text, "5e") == 0) return Kind::Shader5e;
	if (std::strcmp(text, "submit") == 0) return Kind::Submit;
	if (std::strcmp(text, "drain") == 0) return Kind::Drain;
	return Kind::Full;
}

inline void ParseFlip(const char* text, long& lo, long& hi) {
	lo = 4100;
	hi = 4500;
	if (text == nullptr || text[0] == '\0') return;
	if (text[0] == 'g') text++;
	char* end = nullptr;
	const long a = std::strtol(text, &end, 10);
	if (end == text || *end != '-') return;
	char* end2 = nullptr;
	const long b = std::strtol(end + 1, &end2, 10);
	if (end2 == end + 1 || b < a || b - a > 800) return;
	lo = a;
	hi = b;
}

inline const Config& Get() {
	static const Config cfg = [] {
		Config c;
		const char* on = std::getenv("KYTY_DIAG_FULL_BARRIER");
		if (on != nullptr && std::strcmp(on, "1") == 0) {
			c.kind = ParseKind(std::getenv("KYTY_DIAG_BARRIER_KIND"));
			ParseFlip(std::getenv("KYTY_DIAG_BARRIER_FLIP"), c.lo, c.hi);
		}
		std::fprintf(stderr, "[s527-bar] cfg on=%d kind=%s lo=%ld hi=%ld\n", c.kind == Kind::Off ? 0 : 1, KindName(c.kind), c.lo,
		             c.hi);
		return c;
	}();
	return cfg;
}

inline long GuestNow() { return static_cast<long>(FrameTiming::GuestFlips() + 1); }

inline bool InWindow() {
	const Config& cfg = Get();
	if (cfg.kind == Kind::Off) return false;
	const long guest = GuestNow();
	return guest >= cfg.lo && guest <= cfg.hi;
}

inline bool WantAfterDraw() {
	const Kind kind = Get().kind;
	return InWindow() && (kind == Kind::Full || kind == Kind::Draw);
}

inline bool WantAfterDispatch() {
	const Kind kind = Get().kind;
	return InWindow() && (kind == Kind::Full || kind == Kind::Dispatch);
}

inline bool WantBeforeDraw(uint64_t ps) {
	if (!InWindow()) return false;
	const Kind kind = Get().kind;
	if (kind == Kind::Xfer) return true;
	return kind == Kind::Shader5e && ps == kShader5e;
}

inline bool BeforeIsXfer() { return Get().kind == Kind::Xfer; }

inline bool ForceImmediate() { return InWindow() && Get().kind == Kind::Ssbo; }

inline bool WantWaitSubmit() { return InWindow() && Get().kind == Kind::Submit; }

inline bool WantDrainDraw() { return InWindow() && Get().kind == Kind::Drain; }

inline void Note(const char* slot) {
	static std::atomic<uint64_t> n {0};
	static std::atomic<long>     last {-1};
	const uint64_t now   = n.fetch_add(1, std::memory_order_relaxed) + 1;
	const long     guest = GuestNow();
	long           prev  = last.load(std::memory_order_relaxed);
	if (prev != guest && last.compare_exchange_strong(prev, guest, std::memory_order_relaxed)) {
		std::fprintf(stderr, "[s527-bar] guest=%ld n=%llu slot=%s\n", guest, static_cast<unsigned long long>(now), slot);
	}
}

// Caller has already ended the render pass. xfer: transfer and host writes become visible to
// every later access. Otherwise the full memory dependency from the task.
inline void Record(vk::CommandBuffer cmd, bool xfer) {
	vk::MemoryBarrier barrier {};
	if (xfer) {
		barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eHostWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eHost,
		                    vk::PipelineStageFlagBits::eAllCommands, {}, 1, &barrier, 0, nullptr, 0, nullptr);
		Note("xfer");
		return;
	}
	barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands, {}, 1, &barrier, 0,
	                    nullptr, 0, nullptr);
	Note("full");
}

} // namespace Libs::Graphics::GalaxyBarrier

#endif
