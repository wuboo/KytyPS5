// S430: KYTY_SPEC_READ_FAST. The page cache of the strict shader reader (src/kernel/specPageCache.h) against a model of the guest backing, with the
// real GpuDirtyRanges / InflightRanges (their Add() is what invalidates entries).
// Mutants: KYTY_SPECFAST_MUTANT_{FAST_IGNORES_DIRTY, FAST_NO_INVALIDATE_ON_WRITE, NO_INFLIGHT_INVALIDATE, NO_MAP_SEQ, CACHE_BYTES, CROSS_PAGE}.

#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"
#include "graphics/host_gpu/renderer/cache/inflightRanges.h"
#include "kernel/specPageCache.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

using namespace Libs::LibKernel::Memory;
using Libs::Graphics::GpuDirtyRanges;
using Libs::Graphics::InflightRanges;

static int g_failed = 0;

#define CHECK(cond)                                                                    \
	do {                                                                               \
		if (!(cond)) {                                                                 \
			std::fprintf(stderr, "SpecPageCacheTests: failed: %s (line %d)\n", #cond, __LINE__); \
			g_failed++;                                                                \
		}                                                                              \
	} while (0)

constexpr uint64_t Base = 0x10000;
constexpr uint64_t Page = 4096;

// The guest backing (3 pages), the GPU sync state, and the ordinary reader (Full) that drains dirty bytes / waits for in-flight copies the
// way TryReadBufferBacking does: the GPU's bytes (0xEE) reach the backing at that moment.
struct Model {
	std::vector<uint8_t>                       backing = std::vector<uint8_t>(3 * Page, 0x11);
	GpuDirtyRanges                             dirty;
	InflightRanges                             inflight;
	std::vector<std::pair<uint64_t, uint64_t>> inflight_added;
	uint64_t                                   seq    = 0;
	bool                                       mapped = true;
	int full_calls = 0, drained = 0, waited = 0, dirty_queries = 0, inflight_queries = 0, resolve_calls = 0;

	uint64_t MapSeq() const { return seq; }
	bool     Dirty(uint64_t page) {
        dirty_queries++;
        return dirty.Intersects(page, Page);
	}
	bool Inflight(uint64_t page) {
		inflight_queries++;
		return inflight.Intersects(page, Page);
	}
	bool ResolvePage(uint64_t page, const uint8_t** host, uint64_t* s) {
		if (!mapped || (seq & 1u) != 0) return false; // a change under way holds the unique lock
		resolve_calls++;
		*host = backing.data() + (page - Base);
		*s    = seq;
		return true;
	}
	bool Full(uint64_t vaddr, void* data, uint64_t size) {
		full_calls++;
		if (size == 0 || !mapped || vaddr < Base || vaddr - Base > backing.size() || size > backing.size() - (vaddr - Base)) return false;
		if (dirty.Intersects(vaddr, size)) {
			drained++;
			dirty.Subtract(vaddr & ~(Page - 1), 2 * Page); // the drain publishes the GPU bytes
			std::memset(backing.data() + (vaddr - Base), 0xEE, size);
		}
		if (inflight.Intersects(vaddr, size)) {
			waited++;
			for (const auto& [a, n]: inflight_added) inflight.Remove(a, n);
			inflight_added.clear();
		}
		std::memcpy(data, backing.data() + (vaddr - Base), size);
		return true;
	}
	void AddInflight(uint64_t a, uint64_t n) {
		inflight.Add(a, n);
		inflight_added.emplace_back(a, n);
	}
	bool Read(SpecPageCache& cache, uint64_t vaddr, void* data, uint64_t size, bool* hit) {
		return ReadViaPageCache(cache, *this, vaddr, data, size, hit);
	}
};

int main() {
	auto& cache = SpecPageCache::Global();

	// Mode parsing.
	CHECK(ParseSpecFastMode(nullptr) == SpecFastMode::Off);
	CHECK(ParseSpecFastMode("") == SpecFastMode::Off);
	CHECK(ParseSpecFastMode("0") == SpecFastMode::Off);
	CHECK(ParseSpecFastMode("1") == SpecFastMode::On);
	CHECK(ParseSpecFastMode("shadow") == SpecFastMode::Shadow);
	CHECK(ParseSpecFastMode("yes") == SpecFastMode::Off);

	// T1 QuietPageHit: the second read of a quiet page asks neither the dirty set, nor the in-flight set, nor the ordinary reader.
	{
		cache.Clear();
		Model    m;
		uint32_t value = 0;
		bool     hit   = true;
		CHECK(m.Read(cache, Base + 8, &value, 4, &hit));
		CHECK(!hit);
		CHECK(m.resolve_calls == 1 && m.full_calls == 0);
		CHECK(value == 0x11111111u);
		const int dq = m.dirty_queries, iq = m.inflight_queries;
		CHECK(m.Read(cache, Base + 16, &value, 4, &hit));
		CHECK(hit);
		CHECK(m.dirty_queries == dq && m.inflight_queries == iq && m.full_calls == 0 && m.resolve_calls == 1);
	}
	// T9 DirtyAtFill: a page that is dirty when first read is not remembered; the ordinary reader drains it.
	{
		cache.Clear();
		Model m;
		m.dirty.Add(Base + 8, 16);
		uint32_t value = 0;
		bool     hit   = true;
		CHECK(m.Read(cache, Base + 8, &value, 4, &hit));
		CHECK(!hit && m.full_calls == 1 && m.drained == 1 && m.resolve_calls == 0);
		CHECK(value == 0xEEEEEEEEu);
	}
	// T2 DirtyAddInvalidates: the GPU writes into a remembered page; the next read drains and sees the GPU's bytes.
	{
		cache.Clear();
		Model    m;
		uint32_t value = 0;
		bool     hit   = false;
		CHECK(m.Read(cache, Base + Page, &value, 4, &hit) && !hit);
		CHECK(m.Read(cache, Base + Page, &value, 4, &hit) && hit);
		m.dirty.Add(Base + Page + 8, 16);
		CHECK(m.Read(cache, Base + Page + 8, &value, 4, &hit));
		CHECK(!hit);
		CHECK(m.drained == 1);
		CHECK(value == 0xEEEEEEEEu);
	}
	// T3 InflightAddInvalidates: a GPU->host copy of a remembered page is recorded; the next read waits for it.
	{
		cache.Clear();
		Model    m;
		uint32_t value = 0;
		bool     hit   = false;
		CHECK(m.Read(cache, Base, &value, 4, &hit) && !hit);
		CHECK(m.Read(cache, Base, &value, 4, &hit) && hit);
		m.AddInflight(Base + 4, 8);
		CHECK(m.Read(cache, Base + 4, &value, 4, &hit));
		CHECK(!hit);
		CHECK(m.waited == 1);
	}
	// T4 MapSeqInvalidates: a change of the mappings (odd while under way, even after) ends the entry; an unmap makes the read fail.
	{
		cache.Clear();
		Model    m;
		uint32_t value = 0;
		bool     hit   = false;
		CHECK(m.Read(cache, Base, &value, 4, &hit) && !hit);
		CHECK(m.Read(cache, Base, &value, 4, &hit) && hit);
		m.seq = 3; // under way
		CHECK(m.Read(cache, Base, &value, 4, &hit));
		CHECK(!hit && m.full_calls == 1);
		m.seq = 4; // done
		CHECK(m.Read(cache, Base, &value, 4, &hit));
		CHECK(!hit);
		CHECK(m.Read(cache, Base, &value, 4, &hit) && hit);
		m.mapped = false;
		m.seq    = 6;
		CHECK(!m.Read(cache, Base, &value, 4, &hit));
		CHECK(!hit);
	}
	// T5 BytesNeverCached: a CPU write of the guest is seen by the next read of a remembered page.
	{
		cache.Clear();
		Model    m;
		uint32_t value = 0;
		bool     hit   = false;
		CHECK(m.Read(cache, Base + 32, &value, 4, &hit) && !hit);
		m.backing[32] = 0x77;
		CHECK(m.Read(cache, Base + 32, &value, 4, &hit) && hit);
		CHECK((value & 0xFFu) == 0x77u);
	}
	// T6 CrossPageBypass: a read over a page boundary never uses an entry (the next page is dirty here).
	{
		cache.Clear();
		Model    m;
		uint32_t value = 0;
		bool     hit   = false;
		CHECK(m.Read(cache, Base, &value, 4, &hit) && !hit);
		m.dirty.Add(Base + Page, 8);
		uint8_t span[8] = {};
		CHECK(m.Read(cache, Base + Page - 4, span, 8, &hit));
		CHECK(!hit && m.full_calls == 1 && m.drained == 1);
		CHECK(span[4] == 0xEE);
	}
	// T7 LargeAddClears: a range of more than 256 pages clears every entry, a smaller one only its own pages.
	{
		cache.Clear();
		uint8_t byte = 0;
		for (uint64_t i = 0; i < 3; ++i) cache.Fill(Base + i * Page, &byte, 0);
		cache.InvalidateRange(Base + Page + 1, 1);
		CHECK(cache.Lookup(Base, 0) == &byte);
		CHECK(cache.Lookup(Base + Page, 0) == nullptr);
		CHECK(cache.Lookup(Base + 2 * Page, 0) == &byte);
		cache.InvalidateRange(Base, 300 * Page);
		CHECK(cache.Lookup(Base, 0) == nullptr && cache.Lookup(Base + 2 * Page, 0) == nullptr);
	}
	// T8 EdgeSizes: an empty read and an overflowing one go to the ordinary reader.
	{
		cache.Clear();
		Model    m;
		uint32_t value = 0;
		bool     hit   = true;
		CHECK(!m.Read(cache, Base, &value, 0, &hit));
		CHECK(!hit && m.full_calls == 1);
		m.full_calls = 0;
		(void)m.Read(cache, UINT64_MAX - 2, &value, 4, &hit);
		CHECK(!hit && m.full_calls == 1);
	}
	cache.Clear();
	if (g_failed != 0) {
		std::fprintf(stderr, "SpecPageCacheTests: %d failed\n", g_failed);
		return 1;
	}
	std::printf("SpecPageCacheTests: ok\n");
	return 0;
}
