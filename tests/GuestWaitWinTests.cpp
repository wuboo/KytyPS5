// S585 (KYTY_GUEST_WAIT=clean|1) on Windows: a real access violation of `vmovss xmm0, [rcx]` (VEX.128) right after `vzeroupper`, caught by a
// vectored handler as the emulator's is, prepared twice from the real CONTEXT: without Options::vex_init_zero (1606ec74: S495 saw VexNoAvx
// 105-228 per landing) and with it. The line "guest_wait_win: ..." shows the extended-state facts Windows hands to the handler (CONTEXT_XSTATE,
// the XSTATE mask, the AVX area) and whether the AVX init state is recognised; the load must give the backing value either way.
// Not a mutant target: an unrecognised init state only keeps the old fallback (printed as INFO), it is not a failure.

#include "loader/guestReadEmulator.h"

#include <windows.h>

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

using namespace Loader::GuestReadEmulator;

int          g_fails      = 0;
uint8_t*     g_page       = nullptr;
Why          g_why_off    = Why::None;
Why          g_why_on     = Why::None;
bool         g_vex_init   = false;
int          g_emulated   = 0;
int          g_faults     = 0;
char         g_facts[256] = {};
const float  kValue       = 3.25f;

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
	(void)DescribeContext(info->ContextRecord, g_facts, sizeof(g_facts));
	Attempt off;
	(void)Prepare(info->ContextRecord, fault, ReadBacking, &off);
	g_why_off = off.plan.ok ? Why::None : off.plan.why;
	Options options;
	options.vex_init_zero = true;
	Attempt on;
	if (Prepare(info->ContextRecord, fault, ReadBacking, &on, options)) {
		g_vex_init = on.plan.vex_init;
		Commit(info->ContextRecord, on.plan, on.bytes);
		g_emulated++;
		return EXCEPTION_CONTINUE_EXECUTION;
	}
	g_why_on = on.plan.why;
	// Not emulated: let the load run natively on a readable page (the ordinary path's outcome).
	DWORD old = 0;
	VirtualProtect(g_page, 4096, PAGE_READWRITE, &old);
	std::memcpy(g_page, &kValue, sizeof(kValue));
	return EXCEPTION_CONTINUE_EXECUTION;
}

} // namespace

int main() {
	if ((GetEnabledXStateFeatures() & XSTATE_MASK_AVX) == 0) {
		std::printf("guest_wait_win: AVX not enabled by the OS, skipped\n");
		return 0;
	}
	// vzeroupper; vmovss xmm0, dword ptr [rcx]; ret   (Win64: the argument in rcx, the float result in xmm0)
	static const uint8_t kCode[] = {0xc5, 0xf8, 0x77, 0xc5, 0xfa, 0x10, 0x01, 0xc3};
	auto* code = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_EXECUTE_READWRITE));
	g_page     = static_cast<uint8_t*>(VirtualAlloc(nullptr, 4096, MEM_COMMIT | MEM_RESERVE, PAGE_NOACCESS));
	if (code == nullptr || g_page == nullptr) {
		std::fprintf(stderr, "guest_wait_win: VirtualAlloc failed\n");
		return 1;
	}
	std::memcpy(code, kCode, sizeof(kCode));
	FlushInstructionCache(GetCurrentProcess(), code, sizeof(kCode));
	void* handler = AddVectoredExceptionHandler(1, Handler);
	using Fn      = float (*)(const float*);
	const float value = reinterpret_cast<Fn>(code)(reinterpret_cast<const float*>(g_page));
	RemoveVectoredExceptionHandler(handler);

	std::printf("guest_wait_win: %s\n", g_facts);
	std::printf("guest_wait_win: faults=%d off_why=%u on_why=%u emulated=%d vex_init=%d value=%g\n", g_faults, static_cast<unsigned>(g_why_off),
	            static_cast<unsigned>(g_why_on), g_emulated, g_vex_init ? 1 : 0, static_cast<double>(value));
	Check(g_faults == 1, "one fault");
	Check(value == kValue, "the load gives the backing value");
	Check(g_why_off == Why::None || g_why_off == Why::VexNoAvx, "without the option: emulated (AVX state captured) or the old VexNoAvx");
	if (g_why_off == Why::VexNoAvx) {
		if (g_emulated == 1 && g_vex_init) {
			std::printf("guest_wait_win: INFO the AVX init state is recognised: KYTY_GUEST_WAIT serves these loads\n");
		} else {
			std::printf("guest_wait_win: INFO the AVX init state is NOT recognised here (on_why=%u): the VexNoAvx fallback stays\n",
			            static_cast<unsigned>(g_why_on));
		}
	}
	return g_fails == 0 ? 0 : 1;
}
