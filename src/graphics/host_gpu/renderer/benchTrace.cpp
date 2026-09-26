#include "graphics/host_gpu/renderer/benchTrace.h"

#include "common/logging/log.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <string>
#include <vector>

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

bool SkipCompute(uint64_t shader_hash) {
	static const std::vector<uint64_t> skip = [] {
		std::vector<uint64_t> hashes;
		if (const char* value = std::getenv("KYTY_BENCH_SKIP_CS"); value != nullptr) {
			std::string list = value;
			size_t      pos  = 0;
			while (pos < list.size()) {
				const auto end = list.find(',', pos);
				hashes.push_back(std::strtoull(list.substr(pos, end - pos).c_str(), nullptr, 16));
				pos = end == std::string::npos ? list.size() : end + 1;
			}
		}
		return hashes;
	}();
	for (const auto hash: skip) {
		if (hash == shader_hash) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
				LOGF("[bench-skip] skipping compute dispatch of 0x%016" PRIx64 "\n", shader_hash);
			}
			return true;
		}
	}
	return false;
}

} // namespace Libs::Graphics::BenchTrace
