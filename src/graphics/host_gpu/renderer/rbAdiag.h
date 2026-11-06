#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_RBADIAG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_RBADIAG_H_

// S502 DIAG (KYTY_RB_ADIAG=1, default off). On the old line the desert readback of the 24 B and 32 B
// compute writes is ~0 faults/flip. This does not change that read. It counts, during a compute
// dispatch, whether the SRT walker loads a dword of those ranges, whether that dword (and its page)
// is GPU-dirty, which reader supplied the bytes, and whether those bytes match a GPU copy taken
// after the writing frame has been submitted.
// The three compute hashes are the writers S494 named. A load is one of: the compiled slot mov,
// the compiled tree mov, SrtProtAvoid, the interpreter memcpy, or a PROGS_MEMO / SRT memo re-read.
// Mutant (tests/RbAdiagTests.cpp): KYTY_ADIAG_MUTANT_IGNORE_NE makes a differing shadow compare look equal.

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::RbAdiag {

inline constexpr uint64_t kHash20a = 0x8b5e4eced20058a4ull;
inline constexpr uint64_t kHash18  = 0xe14b4c0b0d70a885ull;
inline constexpr uint64_t kHash20b = 0x305afd0aa0f66b9aull;

enum class Src : int { Mov = 0, Tree = 1, Prot = 2, Memo = 3, Interp = 4 };

inline bool Enabled() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_RB_ADIAG");
		return text != nullptr && text[0] == '1' && text[1] == '\0';
	}();
	return on;
}

constexpr bool IsWatchedHash(uint64_t hash) {
	return hash == kHash20a || hash == kHash18 || hash == kHash20b;
}

constexpr bool IsWatchedSize(uint64_t size) {
	return size == 0x18 || size == 0x20;
}

// A 4-byte load at `addr` lies inside [base, end).
constexpr bool DwordInside(uint64_t base, uint64_t end, uint64_t addr) {
	if (end < base || end - base < 4 || addr > ~uint64_t {0} - 3) return false;
	return addr >= base && addr + 4 <= end;
}

constexpr bool ShadowNe(bool have_cpu, uint32_t cpu, bool have_gpu, uint32_t gpu) {
#if defined(KYTY_ADIAG_MUTANT_IGNORE_NE)
	(void)have_cpu;
	(void)cpu;
	(void)have_gpu;
	(void)gpu;
	return false;
#else
	return have_cpu && have_gpu && cpu != gpu;
#endif
}

// JIT filter table. `last` is the highest address at which a 4-byte load still fits (end - 4).
// Empty slots are {0, 0}. Written on the command-processor thread, read by the walker it compiles.
struct WatchSlot {
	uint64_t base;
	uint64_t last;
};
extern "C" WatchSlot kyty_adiag_watch[4];
extern "C" void      KytyAdiagPublish(int index, uint64_t base, uint64_t last);

using JitFn = void (*)(uint64_t addr, int src);
inline JitFn& JitHook() {
	static JitFn fn = nullptr;
	return fn;
}

using SpanFn = void (*)(uint64_t addr, uint64_t nbytes, int src);
inline SpanFn& SpanHook() {
	static SpanFn fn = nullptr;
	return fn;
}

// Replaces the snapshotted dword with the bytes the reader returned, without counting the load again.
using UsedFn = void (*)(uint64_t addr, uint64_t nbytes, const uint32_t* used);
inline UsedFn& UsedHook() {
	static UsedFn fn = nullptr;
	return fn;
}

// Query dirty and, when the dword is clean, snapshot the guest bytes a raw mov will see.
inline void BeginSpan(uint64_t addr, uint64_t nbytes, Src src) {
	if (auto* fn = SpanHook()) fn(addr, nbytes, static_cast<int>(src));
}
inline void CommitUsed(uint64_t addr, uint64_t nbytes, const uint32_t* used) {
	if (used == nullptr) return;
	if (auto* fn = UsedHook()) fn(addr, nbytes, used);
}

void EnterDispatch();
void LeaveDispatch();
void EnterShader(uint64_t hash);
void LeaveShader();
void NoteTracked(uint64_t addr, uint64_t size);
void NoteFill(uint64_t addr, uint64_t size);

using PullDword = bool (*)(void* cache, uint64_t addr, uint32_t* out);
void PullDue(PullDword fn, void* cache);

struct DispatchGuard {
	DispatchGuard() {
		if (Enabled()) EnterDispatch();
	}
	~DispatchGuard() {
		if (Enabled()) LeaveDispatch();
	}
	DispatchGuard(const DispatchGuard&)            = delete;
	DispatchGuard& operator=(const DispatchGuard&) = delete;
};

struct ShaderGuard {
	// A user-declared copy constructor suppresses the implicit default constructor.
	ShaderGuard() = default;
	void Arm(uint64_t hash) {
		if (!Enabled() || armed) return;
		EnterShader(hash);
		armed = true;
	}
	~ShaderGuard() {
		if (armed) LeaveShader();
	}
	ShaderGuard(const ShaderGuard&)            = delete;
	ShaderGuard& operator=(const ShaderGuard&) = delete;
	bool armed = false;
};

} // namespace Libs::Graphics::RbAdiag

#endif
