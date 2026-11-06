#include "graphics/host_gpu/renderer/rbAdiag.h"

#include <cstdio>

namespace {

int g_failures = 0;

void Expect(bool cond, const char* what) {
	if (cond) return;
	std::fprintf(stderr, "FAIL: %s\n", what);
	++g_failures;
}

} // namespace

int main() {
	using namespace Libs::Graphics::RbAdiag;
	Expect(IsWatchedHash(kHash20a) && IsWatchedHash(kHash18) && IsWatchedHash(kHash20b), "three hashes");
	Expect(!IsWatchedHash(0), "other hash");
	Expect(IsWatchedSize(0x18) && IsWatchedSize(0x20) && !IsWatchedSize(0x1c) && !IsWatchedSize(0x1000), "sizes");
	constexpr uint64_t base = 0x1000;
	Expect(DwordInside(base, base + 0x20, base + 0x1c), "dword at +0x1c of a 32 B write");
	Expect(!DwordInside(base, base + 0x20, base + 0x20), "dword past the end");
	Expect(!DwordInside(base, base + 0x18, base + 0x1c), "0x1c is outside a 24 B write");
	Expect(DwordInside(base, base + 0x18, base + 0x14), "last dword of a 24 B write");
	Expect(ShadowNe(true, 0x80000000u, true, 0x00010000u), "stale dword differs");
	Expect(!ShadowNe(true, 0x11u, true, 0x11u), "equal dwords");
	Expect(!ShadowNe(true, 0x11u, false, 0), "no gpu sample");
	Expect(!ShadowNe(false, 0, true, 1u), "no cpu sample");
	if (g_failures != 0) return 1;
	std::puts("RbAdiagTests: OK");
	return 0;
}
