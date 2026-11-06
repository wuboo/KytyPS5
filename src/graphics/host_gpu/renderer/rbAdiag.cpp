#include "graphics/host_gpu/renderer/rbAdiag.h"

#include "common/logging/log.h"
#include "graphics/presentation/frameTiming.h"
#include "kernel/memory.h"

#include <atomic>
#include <cstring>

namespace Libs::Graphics::RbAdiag {

namespace {

thread_local int      t_dispatch = 0;
thread_local uint64_t t_hash     = 0;
thread_local int      t_span     = 0;

enum class Kind : uint8_t { None, Tracked, Known };

struct Range {
	uint64_t base     = 0;
	uint64_t end      = 0;
	uint64_t hash     = 0;
	Kind     kind     = Kind::None;
	bool     counted  = false;
	bool     pending  = false;
	uint32_t cpu      = 0;
	uint64_t cpu_addr = 0;
	uint64_t cpu_flip = 0;
	bool     prefer_1c = false;
};

Range            g_ranges[4] {};
uint64_t         g_pulled_flip = ~uint64_t {0};
uint64_t         g_other_sizes[8] {};
int              g_other_n = 0;
std::atomic<int> g_mismatch_logs {0};

void PublishAll() {
	for (int i = 0; i < 4; ++i) {
		const auto& range = g_ranges[i];
		const uint64_t last = range.end >= range.base + 4 ? range.end - 4 : 0;
		KytyAdiagPublish(i, range.base, range.base == 0 ? 0 : last);
	}
}

Range* FindExact(uint64_t addr, uint64_t size) {
	if (size > ~uint64_t {0} - addr) return nullptr;
	for (auto& range: g_ranges) {
		if (range.base == addr && range.end == addr + size) return &range;
	}
	return nullptr;
}

Range* Upsert(uint64_t addr, uint64_t size, uint64_t hash) {
	if (auto* found = FindExact(addr, size)) {
		found->hash = hash;
		return found;
	}
	Range* slot = nullptr;
	for (auto& range: g_ranges) {
		if (range.base == 0) {
			slot = &range;
			break;
		}
	}
	if (slot == nullptr) slot = &g_ranges[0];
	*slot          = {};
	slot->base     = addr;
	slot->end      = addr + size;
	slot->hash     = hash;
	PublishAll();
	return slot;
}

Range* FindDword(uint64_t addr) {
	for (auto& range: g_ranges) {
		if (range.base != 0 && DwordInside(range.base, range.end, addr)) return &range;
	}
	return nullptr;
}

void RememberCpu(Range& range, uint64_t addr, uint32_t word) {
	const bool at_1c = addr == range.base + 0x1c;
	if (range.pending && range.prefer_1c && !at_1c) return;
	range.pending   = true;
	range.cpu       = word;
	range.cpu_addr  = addr;
	range.cpu_flip  = FrameTiming::GuestFlips();
	range.prefer_1c = at_1c;
}

void ObserveDword(uint64_t addr, Src src) {
	Range* range = FindDword(addr);
	if (range == nullptr) return;
	FrameTiming::Add(FrameTiming::Counter::AdLdN);
	if (addr == range->base + 0x1c) FrameTiming::Add(FrameTiming::Counter::AdLd1cN);
	switch (src) {
		case Src::Mov: FrameTiming::Add(FrameTiming::Counter::AdLdMovN); break;
		case Src::Tree: FrameTiming::Add(FrameTiming::Counter::AdLdTreeN); break;
		case Src::Prot: FrameTiming::Add(FrameTiming::Counter::AdLdProtN); break;
		case Src::Memo: FrameTiming::Add(FrameTiming::Counter::AdLdMemoN); break;
		case Src::Interp: FrameTiming::Add(FrameTiming::Counter::AdLdInterpN); break;
	}
	bool dirty = false, inflight = false, busy = false;
	const bool known = LibKernel::Memory::QueryGuestFaultBytes(addr, sizeof(uint32_t), &dirty, &inflight, &busy);
	if (!known) {
		FrameTiming::Add(FrameTiming::Counter::AdLdUnknownN);
	} else {
		if (dirty) FrameTiming::Add(FrameTiming::Counter::AdLdDirtyN);
		if (inflight) FrameTiming::Add(FrameTiming::Counter::AdLdInflightN);
		if (!dirty) {
			bool page_dirty = false, page_inflight = false, page_busy = false;
			const uint64_t page = addr & ~uint64_t {0xfff};
			if (LibKernel::Memory::QueryGuestFaultBytes(page, 0x1000, &page_dirty, &page_inflight, &page_busy) && page_dirty) {
				FrameTiming::Add(FrameTiming::Counter::AdLdPageN);
			}
		}
	}
	if (known && !dirty && !inflight && !busy) {
		uint32_t word = 0;
		std::memcpy(&word, reinterpret_cast<const void*>(addr), sizeof(word));
		RememberCpu(*range, addr, word);
	}
}

void OnSpan(uint64_t addr, uint64_t nbytes, int src) {
	if (!Enabled() || t_dispatch <= 0 || t_span > 0 || nbytes == 0 || addr > ~uint64_t {0} - (nbytes - 1)) return;
	++t_span;
	const auto which = static_cast<Src>(src);
	uint64_t   cursor = addr;
	const uint64_t end = addr + nbytes;
	if ((cursor & 3u) != 0) cursor = (cursor + 3u) & ~uint64_t {3};
	for (; cursor + 4 <= end; cursor += 4) {
		if (FindDword(cursor) == nullptr) continue;
		ObserveDword(cursor, which);
	}
	--t_span;
}

void OnUsed(uint64_t addr, uint64_t nbytes, const uint32_t* used) {
	if (!Enabled() || t_dispatch <= 0 || used == nullptr || nbytes < 4 || addr > ~uint64_t {0} - (nbytes - 1)) return;
	uint64_t cursor = addr;
	const uint64_t end = addr + nbytes;
	if ((cursor & 3u) != 0) cursor = (cursor + 3u) & ~uint64_t {3};
	for (; cursor + 4 <= end; cursor += 4) {
		Range* range = FindDword(cursor);
		if (range == nullptr) continue;
		const uint32_t word = used[(cursor - addr) / 4];
		RememberCpu(*range, cursor, word);
	}
}

void OnJit(uint64_t addr, int src) {
	if (!Enabled() || t_dispatch <= 0) return;
	if ((addr & 3u) != 0) return;
	ObserveDword(addr, static_cast<Src>(src));
}

struct Install {
	Install() {
		JitHook()  = OnJit;
		SpanHook() = OnSpan;
		UsedHook() = OnUsed;
	}
};
Install g_install;

} // namespace

void EnterDispatch() { ++t_dispatch; }
void LeaveDispatch() {
	if (t_dispatch > 0) --t_dispatch;
}
void EnterShader(uint64_t hash) {
	t_hash = hash;
	if (IsWatchedHash(hash)) FrameTiming::Add(FrameTiming::Counter::AdCsN);
}
void LeaveShader() { t_hash = 0; }

void NoteTracked(uint64_t addr, uint64_t size) {
	if (!Enabled() || !IsWatchedHash(t_hash)) return;
	if (!IsWatchedSize(size)) {
		FrameTiming::Add(FrameTiming::Counter::AdWrOtherN);
		for (int i = 0; i < g_other_n; ++i) {
			if (g_other_sizes[i] == size) return;
		}
		if (g_other_n < 8) {
			g_other_sizes[g_other_n++] = size;
			LOGF("[rb-adiag] write hash=0x%016llx addr=0x%016llx size=0x%llx kind=other\n",
			     static_cast<unsigned long long>(t_hash), static_cast<unsigned long long>(addr),
			     static_cast<unsigned long long>(size));
		}
		return;
	}
	Range* range = Upsert(addr, size, t_hash);
	range->kind = Kind::Tracked;
	FrameTiming::Add(FrameTiming::Counter::AdWrN);
	FrameTiming::Add(FrameTiming::Counter::AdWrDirtyN);
	if (!range->counted) {
		range->counted = true;
		LOGF("[rb-adiag] write hash=0x%016llx addr=0x%016llx size=0x%llx kind=tracked\n",
		     static_cast<unsigned long long>(t_hash), static_cast<unsigned long long>(addr),
		     static_cast<unsigned long long>(size));
	}
}

void NoteFill(uint64_t addr, uint64_t size) {
	if (!Enabled() || !IsWatchedHash(t_hash) || !IsWatchedSize(size)) return;
	Range* range = Upsert(addr, size, t_hash);
	const bool from_obtain = range->kind == Kind::Tracked;
	range->kind = Kind::Known;
	FrameTiming::Add(FrameTiming::Counter::AdWrKnownN);
	if (!from_obtain) FrameTiming::Add(FrameTiming::Counter::AdWrN);
	if (!range->counted) {
		range->counted = true;
		LOGF("[rb-adiag] write hash=0x%016llx addr=0x%016llx size=0x%llx kind=known-fill\n",
		     static_cast<unsigned long long>(t_hash), static_cast<unsigned long long>(addr),
		     static_cast<unsigned long long>(size));
	}
}

void PullDue(PullDword fn, void* cache) {
	if (!Enabled() || fn == nullptr || cache == nullptr) return;
	const uint64_t flip = FrameTiming::GuestFlips();
	if (flip == g_pulled_flip) return;
	g_pulled_flip = flip;
	for (auto& range: g_ranges) {
		if (!range.pending || range.cpu_flip >= flip) continue;
		uint32_t gpu = 0;
		const bool got = fn(cache, range.cpu_addr, &gpu);
		if (!got) {
			if (g_mismatch_logs.fetch_add(1, std::memory_order_relaxed) < 4) {
				LOGF("[rb-adiag] shadow-miss addr=0x%016llx cpu=0x%08x\n",
				     static_cast<unsigned long long>(range.cpu_addr), range.cpu);
			}
			range.pending   = false;
			range.prefer_1c = false;
			continue;
		}
		FrameTiming::Add(FrameTiming::Counter::AdShN);
		if (ShadowNe(true, range.cpu, true, gpu)) {
			FrameTiming::Add(FrameTiming::Counter::AdShNeN);
			if (g_mismatch_logs.fetch_add(1, std::memory_order_relaxed) < 4) {
				LOGF("[rb-adiag] shadow-ne addr=0x%016llx cpu=0x%08x gpu=0x%08x\n",
				     static_cast<unsigned long long>(range.cpu_addr), range.cpu, gpu);
			}
		}
		range.pending   = false;
		range.prefer_1c = false;
	}
}

} // namespace Libs::Graphics::RbAdiag
