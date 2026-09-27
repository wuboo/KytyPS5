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
			LOGF("[bench-trace] tick %" PRIu64 ": %s %" PRIu64 " 0x%016" PRIx64 " 0x%016" PRIx64
			     "\n",
			     tick, entry.kind, entry.hash0, entry.hash0, entry.hash1);
			count++;
		}
	}
	LOGF("[bench-trace] tick %" PRIu64 ": %u recorded operations\n", tick, count);
}

namespace {
std::atomic<uint64_t> g_tick {0};
} // namespace

void SetTick(uint64_t tick) {
	g_tick.store(tick, std::memory_order_relaxed);
}

uint64_t Tick() {
	return g_tick.load(std::memory_order_relaxed);
}

namespace {
struct Current {
	const char* kind  = "none";
	uint64_t    hash0 = 0;
	uint64_t    hash1 = 0;
};
thread_local Current g_current;

struct GpuWrite {
	uint64_t vaddr = 0;
	uint64_t size  = 0;
	uint64_t tick  = 0;
	Current  op;
};
constexpr uint32_t                  WriteCapacity = 1024;
std::array<GpuWrite, WriteCapacity> g_writes {};
std::atomic<uint32_t>               g_next_write {0};
} // namespace

void SetCurrent(const char* kind, uint64_t hash0, uint64_t hash1) {
	g_current = {kind, hash0, hash1};
}

void RecordGpuWrite(uint64_t vaddr, uint64_t size) {
	g_writes[g_next_write.fetch_add(1, std::memory_order_relaxed) % WriteCapacity] = {
	    vaddr, size, Tick(), g_current};
}

void LogGpuWriter(const char* reason, uint64_t vaddr, uint64_t size) {
	const auto next = g_next_write.load(std::memory_order_relaxed);
	for (uint32_t i = 1; i <= WriteCapacity && i <= next; i++) {
		const auto& write = g_writes[(next - i) % WriteCapacity];
		if (write.vaddr < vaddr + size && vaddr < write.vaddr + write.size) {
			// Log each distinct writer a few times.
			static std::array<std::pair<uint64_t, uint32_t>, 64> seen {};
			const auto key = write.op.hash0 ^ (reinterpret_cast<uintptr_t>(write.op.kind) << 1u);
			for (auto& [hash, count]: seen) {
				if (hash == key || count == 0) {
					hash = key;
					if (++count <= 4) {
						LOGF("[bench-writer] %s range=0x%" PRIx64 "+0x%" PRIx64
						     " written by %s 0x%016" PRIx64 " 0x%016" PRIx64 " write=0x%" PRIx64
						     "+0x%" PRIx64 " tick=%" PRIu64 " now=%" PRIu64 "\n",
						     reason, vaddr, size, write.op.kind, write.op.hash0, write.op.hash1,
						     write.vaddr, write.size, write.tick, Tick());
					}
					return;
				}
			}
			return;
		}
	}
	static std::atomic<uint32_t> unknown {0};
	if (unknown.fetch_add(1, std::memory_order_relaxed) < 8) {
		LOGF("[bench-writer] %s range=0x%" PRIx64 "+0x%" PRIx64 " writer not in ring\n", reason,
		     vaddr, size);
	}
}

bool SkipDraw(uint64_t vs_hash, uint64_t ps_hash) {
	static const std::vector<uint64_t> skip = [] {
		std::vector<uint64_t> hashes;
		if (const char* value = std::getenv("KYTY_BENCH_SKIP_DRAW"); value != nullptr) {
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
		if (hash != 0 && (hash == vs_hash || hash == ps_hash)) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
				LOGF("[bench-skip] skipping draw vs=0x%016" PRIx64 " ps=0x%016" PRIx64 "\n",
				     vs_hash, ps_hash);
			}
			return true;
		}
	}
	return false;
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
