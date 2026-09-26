#include "graphics/host_gpu/renderer/benchTrace.h"

#include "common/logging/log.h"

#include <array>
#include <atomic>
#include <cinttypes>

namespace Libs::Graphics::BenchTrace {

namespace {

struct Entry {
	uint64_t    tick  = 0;
	const char* kind  = nullptr;
	uint64_t    hash0 = 0;
	uint64_t    hash1 = 0;
};

constexpr uint32_t          Capacity = 8192;
std::array<Entry, Capacity> g_entries {};
std::atomic<uint32_t>       g_next {0};

} // namespace

void Record(uint64_t tick, const char* kind, uint64_t hash0, uint64_t hash1) {
	g_entries[g_next.fetch_add(1, std::memory_order_relaxed) % Capacity] = {tick, kind, hash0,
	                                                                          hash1};
}

void Dump(uint64_t tick) {
	uint32_t count = 0;
	for (const auto& entry: g_entries) {
		if (entry.kind != nullptr && entry.tick == tick) {
			LOGF("[bench-trace] tick %" PRIu64 ": %s 0x%016" PRIx64 " 0x%016" PRIx64 "\n", tick,
			     entry.kind, entry.hash0, entry.hash1);
			count++;
		}
	}
	LOGF("[bench-trace] tick %" PRIu64 ": %u recorded operations\n", tick, count);
}

} // namespace Libs::Graphics::BenchTrace
