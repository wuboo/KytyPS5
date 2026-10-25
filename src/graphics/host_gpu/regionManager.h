#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_

#include "common/assert.h"
#include "graphics/host_gpu/hotPages.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/protectRing.h"
#include "graphics/host_gpu/regionDefinitions.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <utility>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#undef MemoryBarrier
#elif defined(__APPLE__)
#include <pthread.h>
#elif defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Libs::Graphics {

class TrackingSpinLock final {
public:
	void lock() noexcept {
		const auto thread = CurrentThread();
		while (m_lock.test_and_set(std::memory_order_acquire)) {
			EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) == thread);
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
		}
		m_owner.store(thread, std::memory_order_relaxed);
	}
	void unlock() noexcept {
		EXIT_NOT_IMPLEMENTED(m_owner.load(std::memory_order_relaxed) != CurrentThread());
		m_owner.store(0, std::memory_order_relaxed);
		m_lock.clear(std::memory_order_release);
	}

private:
	static uint32_t CurrentThread() noexcept {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		return GetCurrentThreadId();
#elif defined(__APPLE__)
		// mach thread port is a nonzero per-thread id (0 is the "no owner" sentinel).
		static thread_local const uint32_t tid =
		    static_cast<uint32_t>(pthread_mach_thread_np(pthread_self()));
		return tid;
#elif defined(__linux__)
		static thread_local const uint32_t tid = static_cast<uint32_t>(::syscall(SYS_gettid));
		return tid;
#else
		EXIT("region tracking thread identity is unsupported on this platform\n");
#endif
	}

	std::atomic_flag     m_lock = ATOMIC_FLAG_INIT;
	std::atomic_uint32_t m_owner {0};
};

static_assert(std::atomic_uint32_t::is_always_lock_free);

// S406 (KYTY_PV_RANGE_GEN): one stamp per page, taken from a process-wide clock every time the page BECOMES CPU- or GPU-modified.
// A stamp is greater than every stamp given before it, so the maximum over a range grows whenever any page of it changes; a
// cached result is valid while the range is clean (neither CPU- nor GPU-modified: its pages are write-protected, so a write would
// fault and take a new stamp) and the maximum is the one seen when the result was made. Clearing a dirty bit does not take a stamp.
inline std::atomic<uint64_t>& PageStampClock() {
	static std::atomic<uint64_t> clock {0};
	return clock;
}

class RegionManager final {
public:
	RegionManager(PageManager& page_manager, uint64_t cpu_addr)
	    : m_page_manager(page_manager), m_cpu_addr(cpu_addr) {
		if (m_cpu_addr % TRACKER_REGION_SIZE != 0) {
			EXIT("invalid region tracking manager construction\n");
		}
		m_cpu_dirty.Fill();
		m_writable.Fill();
		m_readable.Fill();
	}

	KYTY_CLASS_NO_COPY(RegionManager);

	[[nodiscard]] uint64_t GetCpuAddr() const { return m_cpu_addr; }
	// Lock-free: grows with every change of the dirty bits, made under the lock. Equal values
	// mean no change in between (see MemoryTracker::ForEachUploadRange).
	[[nodiscard]] uint64_t Generation() const noexcept {
		return m_generation.load(std::memory_order_acquire);
	}
	template <DirtySource source>
	[[nodiscard]] bool IsModified(uint64_t offset, uint64_t size) const {
		const auto [start, end] = GetPageRange(m_cpu_addr + offset, size);
		return GetBits<source>().FirstRangeFrom(start).first < end;
	}

	template <DirtySource source, bool enable>
	void ChangeState(uint64_t vaddr, uint64_t size) {
		const auto [start, end] = GetPageRange(vaddr, size);
		if constexpr (source == DirtySource::Cpu && enable) {
			if (RegionBits(m_gpu_dirty, start, end).Any()) {
				EXIT("CPU dirty state conflicts with GPU dirty state\n");
			}
		}
		if constexpr (source == DirtySource::Gpu && enable) {
			if (RegionBits(m_cpu_dirty, start, end).Any()) {
				EXIT("GPU dirty state conflicts with CPU dirty state\n");
			}
		}
		Bump();
		auto& bits = GetBits<source>();
		if constexpr (enable) {
#if defined(KYTY_MEMTRACKER_MUTANT_NO_BUMP_CPU)
			constexpr bool stamp = source != DirtySource::Cpu;
#elif defined(KYTY_MEMTRACKER_MUTANT_NO_BUMP_GPU)
			constexpr bool stamp = source != DirtySource::Gpu;
#else
			constexpr bool stamp = true;
#endif
			if constexpr (stamp) {
				const auto value = PageStampClock().fetch_add(1, std::memory_order_relaxed) + 1;
				for (size_t page = start; page < end; ++page) m_stamps[page] = value;
			}
			bits.SetRange(start, end);
		} else {
			bits.UnsetRange(start, end);
		}
		if constexpr (source == DirtySource::Cpu) {
			UpdateProtection<!enable, false>();
		} else {
			UpdateProtection<enable, true>();
		}
	}

	// Largest page stamp over the pages of [vaddr, vaddr + size) (0: no page of it ever became modified).
	[[nodiscard]] uint64_t MaxStamp(uint64_t vaddr, uint64_t size) const {
		const auto [start, end] = GetPageRange(vaddr, size);
		uint64_t result = 0;
		for (size_t page = start; page < end; ++page) result = std::max(result, m_stamps[page]);
		return result;
	}

	template <DirtySource source, bool clear, typename Func>
	void ForEachModifiedRange(uint64_t vaddr, uint64_t size, Func&& func) {
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      bits         = GetBits<source>();
		if (bits.FirstRangeFrom(start).first >= end) {
			return;
		}
		RegionBits mask(bits, start, end);
		if constexpr (clear) {
			Bump();
			bits.UnsetRange(start, end);
			if constexpr (source == DirtySource::Cpu) {
				UpdateProtection<true, false>();
			} else {
				UpdateProtection<false, true>();
			}
		}
		for (const auto [first, last]: mask) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// S440 (KYTY_WF_HOT_UNPROTECT): everything below runs with `lock` held.
	[[nodiscard]] bool HasHot() const { return m_hot_state != nullptr && (m_hot_state->count != 0 || m_hot_state->cand_n != 0); }
	[[nodiscard]] uint32_t HotCountRelaxed() const { return m_hot_count.load(std::memory_order_relaxed); }
	[[nodiscard]] bool IsHot(size_t page) const { return m_hot_state != nullptr && m_hot_state->hot.Get(page); }

	// A write fault released [vaddr, vaddr + size): count one more flip for each page (hot after kHotStreak flips in a row).
	void NoteWriteFault(uint64_t vaddr, uint64_t size) {
		const auto flip_mode = FlipUnprotect::CurrentMode();
		if (HotPages::CurrentMode() == HotPages::Mode::Off && flip_mode == HotPages::Mode::Off) {
			return;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      hs           = EnsureHot();
		const auto epoch        = HotPages::Epoch();
		if (flip_mode != HotPages::Mode::Off) {
			for (size_t page = start; page < end; ++page) {
				NoteFlipFault(hs, page, epoch, flip_mode);
			}
		}
		if (HotPages::CurrentMode() == HotPages::Mode::Off) {
			return;
		}
		for (size_t page = start; page < end; ++page) {
			if (hs.streak[page] != 0 && hs.last_fault[page] == epoch) {
				continue;
			}
			hs.streak[page] = (hs.streak[page] != 0 && hs.last_fault[page] + 1 == epoch) ? static_cast<uint8_t>(std::min<uint32_t>(255u, hs.streak[page] + 1u)) : 1;
			hs.last_fault[page] = epoch;
			if (hs.streak[page] >= HotPages::kHotStreak && !hs.hot.Get(page)) {
				hs.hot.Set(page);
				hs.hash_valid.Unset(page); // the buffer holds whatever the last (unhashed) upload gave it
				hs.last_change[page] = epoch;
				SetHotCount(hs, hs.count + 1);
				HotPages::GetStats().promote.fetch_add(1, std::memory_order_relaxed);
				HotPages::GetStats().hot_pages.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}

	// An explicit CPU-modification (MarkRegionAsCpuModified, UntrackMemory): whatever the buffer holds for these pages is not what the hash describes.
	void InvalidateHashes(uint64_t vaddr, uint64_t size) {
		if (m_hot_state == nullptr) {
			return;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		m_hot_state->hash_valid.UnsetRange(start, end);
	}

	// The GPU is about to own these pages: hot pages in the range become ordinary dirty pages (the caller consumed them already).
	void ReleaseHot(uint64_t vaddr, uint64_t size) {
		if (!HasHot()) {
			return;
		}
		const ProtectRing::TagScope tag("buf-release-hot");
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      hs           = *m_hot_state;
		bool       any          = false;
		for (size_t page = start; page < end; ++page) {
			DropCandidate(hs, page);
			if (hs.hot.Get(page)) {
				DropHot(hs, page);
				m_cpu_dirty.Unset(page);
				any = true;
			}
		}
		if (any) {
			UpdateProtection<true, false>();
		}
	}

	// Shadow: hot pages that are clean (protected, nothing dirty) must still hash like their last upload.
	void ShadowCheckClean(uint64_t vaddr, uint64_t size) {
		if (!HasHot() || HotPages::CurrentMode() != HotPages::Mode::Shadow) {
			return;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      hs           = *m_hot_state;
		for (size_t page = start; page < end; ++page) {
			if (!hs.hot.Get(page) || m_cpu_dirty.Get(page) || m_gpu_dirty.Get(page) || !hs.hash_valid.Get(page)) {
				continue;
			}
			uint64_t hash = 0;
			if (HashPage(page, &hash) && hash != hs.hash[page]) {
				HotPages::GetStats().ne.fetch_add(1, std::memory_order_relaxed);
				hs.hash[page] = hash;
			}
		}
	}

	// S443 shadow (KYTY_WF_FLIP_UNPROTECT=shadow): a candidate page that is clean (protected) must hash like it did right after its protection; a candidate of an
	// earlier flip leaves the set.
	void ShadowCheckFlip(uint64_t vaddr, uint64_t size) {
		if (m_hot_state == nullptr || m_hot_state->cand_n == 0 || FlipUnprotect::CurrentMode() != HotPages::Mode::Shadow) {
			return;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		auto&      hs           = *m_hot_state;
		const auto epoch        = HotPages::Epoch();
		for (size_t page = start; page < end; ++page) {
			if (!hs.cand.Get(page)) {
				continue;
			}
			if (hs.sticky_epoch[page] != epoch) {
				DropCandidate(hs, page);
				continue;
			}
			if (m_cpu_dirty.Get(page) || m_gpu_dirty.Get(page) || !hs.cand_hash_valid.Get(page)) {
				continue;
			}
#if defined(KYTY_MEMTRACKER_MUTANT_FLIP_SHADOW_BLIND)
			continue;
#endif
			uint64_t first = 0;
			if (!HashPage(page, &first)) {
				continue;
			}
			FlipUnprotect::GetStats().hash.fetch_add(1, std::memory_order_relaxed);
			if (first == hs.fhash[page]) {
				continue;
			}
			uint64_t second = 0;
			const bool stable = HashPage(page, &second) && second == first;
			(stable ? FlipUnprotect::GetStats().ne : FlipUnprotect::GetStats().ne_unstable).fetch_add(1, std::memory_order_relaxed);
			if (const auto observer = FlipUnprotect::NeObserver().load(std::memory_order_relaxed)) {
				observer(m_cpu_addr + page * TRACKER_PAGE_SIZE, stable);
			}
			hs.fhash[page] = second;
		}
	}

	// ForEachModifiedRange<Cpu, true> that knows hot pages. keep_hot: the caller will not hand the pages to the GPU for writing.
	template <typename Func>
	void ConsumeCpuDirty(uint64_t vaddr, uint64_t size, bool keep_hot, Func&& func) {
		const ProtectRing::TagScope tag(keep_hot ? "buf-consume" : "buf-consume-gpu-write");
		if (!HasHot()) {
			ForEachModifiedRange<DirtySource::Cpu, true>(vaddr, size, func);
			return;
		}
		const auto [start, end] = GetPageRange(vaddr, size);
		if (m_cpu_dirty.FirstRangeFrom(start).first >= end) {
			return;
		}
		const RegionBits mask(m_cpu_dirty, start, end);
		RegionBits       report = mask;
		RegionBits       clear  = mask;
		RegionBits       recheck;
		recheck.Clear();
		auto&      hs    = *m_hot_state;
		const auto mode  = HotPages::CurrentMode();
		const auto epoch = HotPages::Epoch();
		const auto t0    = std::chrono::steady_clock::now();
		uint64_t   hashed = 0;
		const bool flip_on = FlipUnprotect::CurrentMode() == HotPages::Mode::On;
		for (size_t page = start; page < end; ++page) {
			if (!mask.Get(page) || !hs.hot.Get(page)) {
				continue;
			}
			if (hs.sticky.Get(page)) {
				// S443: kept until the end of its flip; the page is reported (copied) by every upload, whether or not it stays unprotected.
#if defined(KYTY_MEMTRACKER_MUTANT_FLIP_NEVER_DEMOTE)
				const bool same_flip = true;
#else
				const bool same_flip = hs.sticky_epoch[page] == epoch;
#endif
				if (keep_hot && same_flip && flip_on) {
					clear.Unset(page);
					FlipUnprotect::GetStats().keep.fetch_add(1, std::memory_order_relaxed);
				} else {
					DropHot(hs, page);
				}
#if defined(KYTY_MEMTRACKER_MUTANT_FLIP_SKIPS_UPLOAD)
				report.Unset(page);
#endif
				continue;
			}
#if defined(KYTY_MEMTRACKER_MUTANT_HOT_SKIPS_HASH)
			const bool changed = !hs.hash_valid.Get(page);
#else
			uint64_t hash = 0;
			const bool ok = HashPage(page, &hash);
			hashed++;
			const bool changed = !ok || !hs.hash_valid.Get(page) || hs.hash[page] != hash;
#endif
			if (changed) {
#if defined(KYTY_MEMTRACKER_MUTANT_HOT_SKIPS_HASH)
				hs.hash_valid.Set(page); // the first upload is the last: no later change is ever noticed
#else
				hs.hash[page] = ok ? hash : 0;
				hs.hash_valid.Unset(page);
				if (ok) {
					hs.hash_valid.Set(page);
				}
#endif
				hs.last_change[page] = epoch;
				HotPages::GetStats().changed.fetch_add(1, std::memory_order_relaxed);
			} else {
				if (mode == HotPages::Mode::On) {
					report.Unset(page); // shadow reports the page as ever
				}
				HotPages::GetStats().skip.fetch_add(1, std::memory_order_relaxed);
			}
			if (mode != HotPages::Mode::On) {
				continue; // shadow: the page is consumed and protected as ever; the hash is only remembered
			}
#if defined(KYTY_MEMTRACKER_MUTANT_HOT_NEVER_REPROTECT)
			const bool cold = false;
#else
			const bool cold = !changed && epoch - hs.last_change[page] >= HotPages::kColdAfter;
#endif
			if (keep_hot && !cold) {
				clear.Unset(page); // stays dirty, so unprotected
			} else {
				DropHot(hs, page);
				if (!changed) {
					recheck.Set(page);
				}
			}
		}
		if (hashed != 0) {
			HotPages::GetStats().hash.fetch_add(hashed, std::memory_order_relaxed);
			HotPages::GetStats().hash_ns.fetch_add(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count()), std::memory_order_relaxed);
		}
		for (size_t page = start; page < end; ++page) {
			if (clear.Get(page)) {
				m_cpu_dirty.Unset(page);
			}
		}
		UpdateProtection<true, false>();
		// The protection is in force: a write before it went unseen by the hash, so look again and upload the page if it differs.
		for (size_t page = start; page < end; ++page) {
			if (!recheck.Get(page)) {
				continue;
			}
			uint64_t hash = 0;
			if (!HashPage(page, &hash) || hash != hs.hash[page]) {
				HotPages::GetStats().race.fetch_add(1, std::memory_order_relaxed);
				report.Set(page);
			}
		}
		if (hs.cand_n != 0 && FlipUnprotect::CurrentMode() == HotPages::Mode::Shadow) {
			// S443 shadow: hash the candidates once the protection is in force (a hash taken earlier would miss a write between it and the protection).
			for (size_t page = start; page < end; ++page) {
				if (!mask.Get(page) || !hs.cand.Get(page) || m_cpu_dirty.Get(page)) {
					continue;
				}
				if (hs.sticky_epoch[page] != epoch) {
					DropCandidate(hs, page);
					continue;
				}
				uint64_t hash = 0;
				if (HashPage(page, &hash)) {
					hs.fhash[page] = hash;
					hs.cand_hash_valid.Set(page);
					FlipUnprotect::GetStats().hash.fetch_add(1, std::memory_order_relaxed);
				} else {
					hs.cand_hash_valid.Unset(page);
				}
			}
		}
		for (const auto [first, last]: report) {
			func(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	// Appends the hot pages to `add(address, bytes)` (their writes raise no fault, so nothing else tells the BDA buffers about them).
	template <typename AddFn>
	void ForEachHotPage(AddFn&& add) {
		if (!HasHot()) {
			return;
		}
		for (const auto [first, last]: m_hot_state->hot) {
			add(m_cpu_addr + first * TRACKER_PAGE_SIZE, (last - first) * TRACKER_PAGE_SIZE);
		}
	}

	TrackingSpinLock lock;

private:

	struct HotState {
		RegionBits                                 hot;
		RegionBits                                 hash_valid; // `hash` is the hash of the page as the buffer last received it
		std::array<uint8_t, TRACKER_REGION_PAGES>  streak {};
		std::array<uint32_t, TRACKER_REGION_PAGES> last_fault {};
		std::array<uint32_t, TRACKER_REGION_PAGES> last_change {};
		std::array<uint64_t, TRACKER_REGION_PAGES> hash {};
		uint32_t                                   count = 0;
		// S443 (KYTY_WF_FLIP_UNPROTECT): `sticky` marks the hot pages kept for their flip (On); `cand` the pages that would be (shadow).
		RegionBits                                 sticky;
		RegionBits                                 fault_seen; // fault_epoch[page] is valid
		RegionBits                                 cand;
		RegionBits                                 cand_hash_valid;
		std::array<uint32_t, TRACKER_REGION_PAGES> fault_epoch {}; // flip of the last write fault
		std::array<uint32_t, TRACKER_REGION_PAGES> sticky_epoch {}; // flip in which the page became sticky / candidate
		std::array<uint64_t, TRACKER_REGION_PAGES> fhash {};
		uint32_t                                   cand_n = 0;
	};

	HotState& EnsureHot() {
		if (m_hot_state == nullptr) {
			m_hot_state = std::make_unique<HotState>();
			m_hot_state->hot.Clear();
			m_hot_state->hash_valid.Clear();
			m_hot_state->sticky.Clear();
			m_hot_state->fault_seen.Clear();
			m_hot_state->cand.Clear();
			m_hot_state->cand_hash_valid.Clear();
		}
		return *m_hot_state;
	}

	void SetHotCount(HotState& hs, uint32_t count) {
		hs.count = count;
		m_hot_count.store(count, std::memory_order_relaxed);
	}

	// The second write fault of `page` in guest flip `epoch` (S443): keep the page unprotected until the flip ends (On) / remember it as a candidate (shadow).
	void NoteFlipFault(HotState& hs, size_t page, uint32_t epoch, HotPages::Mode mode) {
#if !defined(KYTY_MEMTRACKER_MUTANT_STICKY_REFAULT_BLIND)
		if (hs.sticky.Get(page)) {
			// S444: a write fault on a page the tracker keeps unprotected: somebody put the host protection back.
			FlipUnprotect::GetStats().sticky_refault.fetch_add(1, std::memory_order_relaxed);
			if (const auto observer = FlipUnprotect::StickyRefaultObserver().load(std::memory_order_relaxed)) {
				observer(m_cpu_addr + page * TRACKER_PAGE_SIZE, epoch, hs.sticky_epoch[page]);
			}
		}
#endif
		if (!hs.fault_seen.Get(page) || hs.fault_epoch[page] != epoch) {
			hs.fault_seen.Set(page);
			hs.fault_epoch[page] = epoch;
			return;
		}
		FlipUnprotect::GetStats().refault.fetch_add(1, std::memory_order_relaxed);
		if (mode == HotPages::Mode::On) {
			if (hs.hot.Get(page)) {
				return; // already hot (S440) or kept
			}
			hs.hot.Set(page);
			hs.sticky.Set(page);
			hs.sticky_epoch[page] = epoch;
			hs.hash_valid.Unset(page);
			SetHotCount(hs, hs.count + 1);
			FlipUnprotect::GetStats().promote.fetch_add(1, std::memory_order_relaxed);
			FlipUnprotect::GetStats().pages.fetch_add(1, std::memory_order_relaxed);
		} else if (!hs.cand.Get(page)) {
			hs.cand.Set(page);
			hs.cand_hash_valid.Unset(page);
			hs.sticky_epoch[page] = epoch;
			hs.cand_n++;
			FlipUnprotect::GetStats().cand.fetch_add(1, std::memory_order_relaxed);
		}
	}

	void DropCandidate(HotState& hs, size_t page) {
		if (hs.cand.Get(page)) {
			hs.cand.Unset(page);
			hs.cand_hash_valid.Unset(page);
			hs.cand_n--;
		}
	}

	void DropHot(HotState& hs, size_t page) {
		if (hs.sticky.Get(page)) {
			hs.sticky.Unset(page);
			hs.hot.Unset(page);
			hs.hash_valid.Unset(page);
			hs.streak[page] = 0;
			SetHotCount(hs, hs.count - 1);
			FlipUnprotect::GetStats().demote.fetch_add(1, std::memory_order_relaxed);
			FlipUnprotect::GetStats().pages.fetch_sub(1, std::memory_order_relaxed);
			return;
		}
		hs.hot.Unset(page);
		hs.hash_valid.Unset(page); // later uploads of the page are not hashed: the stored hash describes an older buffer
		hs.streak[page] = 0;
		SetHotCount(hs, hs.count - 1);
		HotPages::GetStats().demote.fetch_add(1, std::memory_order_relaxed);
		HotPages::GetStats().hot_pages.fetch_sub(1, std::memory_order_relaxed);
	}

	bool HashPage(size_t page, uint64_t* hash) const {
		const auto fn = HotPages::HasherSource().load(std::memory_order_relaxed);
		return fn != nullptr && fn(m_cpu_addr + page * TRACKER_PAGE_SIZE, hash);
	}
	void Bump() noexcept { m_generation.fetch_add(1, std::memory_order_release); }

	template <bool track, bool is_read>
	void UpdateProtection() {
		const ProtectRing::DefaultTagScope tag(is_read ? (track ? "buf-gpu-own" : "buf-gpu-release") : (track ? "buf-protect" : "buf-unprotect"));
		const auto protection = is_read ? ~m_gpu_dirty : m_cpu_dirty;
		auto&      previous   = is_read ? m_readable : m_writable;
		auto       mask       = protection ^ previous;
		if (mask.None()) {
			return;
		}
		previous = protection;
		m_page_manager.UpdatePageWatchersForRegion<track, is_read>(m_cpu_addr, mask);
	}

	template <DirtySource source>
	RegionBits& GetBits() {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	template <DirtySource source>
	const RegionBits& GetBits() const {
		if constexpr (source == DirtySource::Cpu) {
			return m_cpu_dirty;
		} else {
			return m_gpu_dirty;
		}
	}

	[[nodiscard]] std::pair<size_t, size_t> GetPageRange(uint64_t vaddr, uint64_t size) const {
		if (size == 0 || vaddr < m_cpu_addr || vaddr >= m_cpu_addr + TRACKER_REGION_SIZE ||
		    size > m_cpu_addr + TRACKER_REGION_SIZE - vaddr) {
			EXIT("range lies outside its tracking region\n");
		}
		const auto offset = vaddr - m_cpu_addr;
		return {static_cast<size_t>(offset / TRACKER_PAGE_SIZE),
		        static_cast<size_t>((offset + size + TRACKER_PAGE_SIZE - 1) / TRACKER_PAGE_SIZE)};
	}

	PageManager& m_page_manager;
	uint64_t     m_cpu_addr = 0;
	RegionBits   m_cpu_dirty;
	RegionBits   m_gpu_dirty;
	RegionBits   m_writable;
	RegionBits   m_readable;
	std::array<uint64_t, TRACKER_REGION_PAGES> m_stamps {};
	std::unique_ptr<HotState>                  m_hot_state;
	std::atomic<uint32_t>                      m_hot_count {0};
	// Starts at 1: a sum of existing regions' generations is never 0.
	std::atomic<uint64_t> m_generation {1};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_REGIONMANAGER_H_
