// S585 (KYTY_GUEST_WAIT=clean|1) on Windows: real access violations of `vmovss xmm0, [rcx]` (VEX.128), caught by a vectored handler as the
// emulator's is, and prepared from the real CONTEXT without Options::vex_init_zero (1606ec74: S495/S584 saw VexNoAvx) and with it:
//  A. after vzeroupper (the AVX component in its init state);
//  B. with the upper halves of ymm0 and ymm2 all ones (the AVX component in use);
//  C. as B, with CONTEXT_XSTATE stripped from the context before Prepare (YmmhState::Hardware: the destination's upper half is zeroed in the
//     hardware register and NtContinue must keep it).
// Every case must give the backing value in xmm0 and zero bits 128..255 of ymm0 after the resume (VEX.128 semantics), whether the load was
// emulated or ran natively after the handler unprotected the page. The lines "guest_wait_win: ..." show what Windows hands to the handler
// (CONTEXT_XSTATE, the XSTATE mask, the AVX area) and which path served the load; ymm2's upper half after C is informational (status quo of
// every handled fault).

#include "loader/guestReadEmulator.h"

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

using namespace Loader::GuestReadEmulator;

int         g_fails      = 0;
uint8_t*    g_page       = nullptr;
bool        g_strip      = false;
Why         g_why_off    = Why::None;
Why         g_why_on     = Why::None;
bool        g_vex_init   = false;
bool        g_vex_hw     = false;
int         g_emulated   = 0;
int         g_faults     = 0;
char        g_facts[256] = {};
const float kValue       = 3.25f;

void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "guest_wait_win: FAIL %s\n", what);
		g_fails++;
	}
}

BackingResult ReadBacking(uint64_t /*address*/, uint64_t size, void* dst) {
	if (size != sizeof(kValue)) {
		return BackingResult::Unavailable;
	}
	std::memcpy(dst, &kValue, sizeof(kValue));
	return BackingResult::Ok;
}

LONG CALLBACK Handler(PEXCEPTION_POINTERS info) {
	if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || info->ExceptionRecord->NumberParameters < 2) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	const auto fault = static_cast<uint64_t>(info->ExceptionRecord->ExceptionInformation[1]);
	const auto page  = reinterpret_cast<uint64_t>(g_page);
	if (fault < page || fault >= page + 4096) {
		return EXCEPTION_CONTINUE_SEARCH;
	}
	g_faults++;
	if (g_strip) {
		info->ContextRecord->ContextFlags &= ~static_cast<DWORD>(CONTEXT_XSTATE & ~CONTEXT_AMD64); // the extended area is no longer part of it
	}
	(void)DescribeContext(info->ContextRecord, g_facts, sizeof(g_facts));
	Attempt off;
	(void)Prepare(info->ContextRecord, fault, ReadBacking, &off);
	g_why_off = off.plan.ok ? Why::None : off.plan.why;
	Options options;
	options.vex_init_zero = true;
	Attempt on;
	if (Prepare(info->ContextRecord, fault, ReadBacking, &on, options)) {
		g_vex_init = on.plan.vex_init;
		g_vex_hw   = on.plan.vex_hw;
		Commit(info->ContextRecord, on.plan, on.bytes);
		g_emulated++;
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	g_why_on = on.plan.why;
	// Not emulated: the load runs natively on a readable page (the ordinary path's outcome).
	DWORD old = 0;
	VirtualProtect(g_page, 4096, PAGE_READWRITE, &old);
	std::memcpy(g_page, &kValue, sizeof(kValue));
	return EXCEPTION_CONTINUE_EXECUTION;
}

// Win64: float fn(const float* src /* rcx */, uint8_t* upper /* rdx, 32 bytes */), result in xmm0.
const uint8_t kInit[] = {
    0xc5, 0xf8, 0x77,                   // vzeroupper
    0xc5, 0xfa, 0x10, 0x01,             // vmovss xmm0, [rcx]          (faults)
    0xc4, 0xe3, 0x7d, 0x19, 0xc1, 0x01, // vextractf128 xmm1, ymm0, 1
    0xc5, 0xf8, 0x11, 0x0a,             // vmovups [rdx], xmm1
    0xc5, 0xf8, 0x77,                   // vzeroupper
    0xc3,                               // ret
};
const uint8_t kDirty[] = {
    0xc5, 0xfc, 0xc2, 0xc0, 0x0f,       // vcmpps ymm0, ymm0, ymm0, TRUE (all ones)
    0xc5, 0xec, 0xc2, 0xd2, 0x0f,       // vcmpps ymm2, ymm2, ymm2, TRUE
    0xc5, 0xfa, 0x10, 0x01,             // vmovss xmm0, [rcx]          (faults)
    0xc4, 0xe3, 0x7d, 0x19, 0xc1, 0x01, // vextractf128 xmm1, ymm0, 1
    0xc5, 0xf8, 0x11, 0x0a,             // vmovups [rdx], xmm1
    0xc4, 0xe3, 0x7d, 0x19, 0xd3, 0x01, // vextractf128 xmm3, ymm2, 1
    0xc5, 0xf8, 0x11, 0x5a, 0x10,       // vmovups [rdx + 16], xmm3
    0xc5, 0xf8, 0x77,                   // vzeroupper
    0xc3,                               // ret
};

const char* WhyName(Why why) {
	return why == Why::None ? "ok" : (why == Why::VexNoAvx ? "VexNoAvx" : "other");
}

void RunCase(const char* name, uint8_t* code, const uint8_t* bytes, size_t size, bool strip, bool dirty_ymm2) {
	std::memcpy(code, bytes, size);
	FlushInstructionCache(GetCurrentProcess(), code, size);
	DWORD old = 0;
	VirtualProtect(g_page, 4096, PAGE_NOACCESS, &old);
	g_strip    = strip;
	g_why_off  = Why::None;
	g_why_on   = Why::None;
	g_vex_init = false;
	g_vex_hw   = false;
	g_emulated = 0;
	g_faults   = 0;
	std::memset(g_facts, 0, sizeof(g_facts));
	alignas(16) uint8_t upper[32];
	std::memset(upper, 0xcc, sizeof(upper));
	using Fn          = float (*)(const float*, uint8_t*);
	const float value = reinterpret_cast<Fn>(code)(reinterpret_cast<const float*>(g_page), upper);
	bool        zero  = true;
	for (int i = 0; i < 16; i++) zero &= upper[i] == 0;
	bool ones = true;
	for (int i = 16; i < 32; i++) ones &= upper[i] == 0xff;
	std::printf("guest_wait_win: %s: %s\n", name, g_facts);
	std::printf("guest_wait_win: %s: faults=%d off=%s on=%s emulated=%d vex_init=%d vex_hw=%d value=%g ymm0_upper_zero=%d", name, g_faults,
	            WhyName(g_why_off), WhyName(g_why_on), g_emulated, g_vex_init ? 1 : 0, g_vex_hw ? 1 : 0, static_cast<double>(value), zero ? 1 : 0);
	if (dirty_ymm2) std::printf(" ymm2_upper_kept=%d", ones ? 1 : 0);
	std::printf("\n");
	char what[128];
	std::snprintf(what, sizeof(what), "%s: one fault", name);
	Check(g_faults == 1, what);
	std::snprintf(what, sizeof(what), "%s: xmm0 is the backing value", name);
	Check(value == kValue, what);
	std::snprintf(what, sizeof(what), "%s: bits 128..255 of ymm0 are zero after the load (VEX.128)", name);
	Check(zero, what);
	if (g_why_off == Why::VexNoAvx && g_emulated == 0) {
		std::printf("guest_wait_win: %s: INFO not emulated (VexNoAvx stays for this context)\n", name);
	}
}

} // namespace

int main() {
	if ((GetEnabledXStateFeatures() & XSTATE_MASK_AVX) == 0) {
		std::printf("guest_wait_win: AVX not enabled by the OS, skipped\n");
		return 0;
	}
	auto* code = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	g_page     = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS));
	if (code == nullptr || g_page == nullptr) {
		std::fprintf(stderr, "guest_wait_win: VirtualAlloc failed\n");
		return 1;
	}
	void* handler = AddVectoredExceptionHandler(1, Handler);
	RunCase("A-init", code, kInit, sizeof(kInit), false, false);
	RunCase("B-dirty", code, kDirty, sizeof(kDirty), false, true);
	RunCase("C-noxstate", code, kDirty, sizeof(kDirty), true, true);
	RemoveVectoredExceptionHandler(handler);
	return g_fails == 0 ? 0 : 1;
}
