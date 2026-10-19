// S437: with KYTY_SPEC_READ_FAST unset (the default) nothing reads the page cache of the strict shader reader, so GpuDirtyRanges::Add and
// InflightRanges::Add must not maintain it (they are on the hot path of every GPU write / copy). Mutant: KYTY_SPECFAST_MUTANT_BUSY_NOT_GATED.

#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"
#include "graphics/host_gpu/renderer/cache/inflightRanges.h"
#include "kernel/specPageCache.h"

#include <cstdint>
#include <cstdio>

using namespace Libs::LibKernel::Memory;

namespace {
int g_failures = 0;
void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}
} // namespace

int main() {
	Check(SpecFastEnvMode() == SpecFastMode::Off, "the test must run with KYTY_SPEC_READ_FAST unset");
	auto&         cache = SpecPageCache::Global();
	const uint8_t host[4096] = {};
	constexpr uint64_t Page   = 0x20000;
	cache.Clear();
	cache.Fill(Page, host, 2);
	Check(cache.Lookup(Page, 2) == host, "a filled page is found");
	Libs::Graphics::GpuDirtyRanges dirty;
	dirty.Add(Page + 16, 32);
	Check(cache.Lookup(Page, 2) == host, "mode off: GpuDirtyRanges::Add does not touch the cache");
	Libs::Graphics::InflightRanges inflight;
	inflight.Add(Page + 64, 32);
	Check(cache.Lookup(Page, 2) == host, "mode off: InflightRanges::Add does not touch the cache");
	// the cache itself still invalidates when asked directly (the maintenance is only skipped by the gate)
	cache.InvalidateRange(Page, 32);
	Check(cache.Lookup(Page, 2) == nullptr, "InvalidateRange still works");
	if (g_failures != 0) return 1;
	std::puts("spec page cache off tests passed");
	return 0;
}
