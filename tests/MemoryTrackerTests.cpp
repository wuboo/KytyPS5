#include "common/hostException.h"
#include "common/virtualMemory.h"
#include "graphics/host_gpu/protectRing.h"
#include "graphics/host_gpu/hotPages.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <semaphore>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef min
#undef max
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <mach-o/dyld.h>
#include <csignal>
#include <limits.h>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#else
#include <csignal>
#include <map>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

using Libs::Graphics::GuestRange;
using Libs::Graphics::MemoryTracker;
using Libs::Graphics::PageManager;
using Libs::Graphics::RangeSet;
using Libs::Graphics::TRACKER_ADDRESS_SIZE;

void Check(bool value, const char *text) {
  if (!value) {
    std::fprintf(stderr, "MemoryTrackerTests: failed: %s\n", text);
#if defined(KYTY_MEMTRACKER_MUTANT_NO_BUMP_CPU) || defined(KYTY_MEMTRACKER_MUTANT_NO_BUMP_GPU) || \
    defined(KYTY_MEMTRACKER_MUTANT_HOT_SKIPS_HASH) || defined(KYTY_MEMTRACKER_MUTANT_HOT_NEVER_REPROTECT) || \
    defined(KYTY_MEMTRACKER_MUTANT_FLIP_NEVER_DEMOTE) || defined(KYTY_MEMTRACKER_MUTANT_FLIP_SKIPS_UPLOAD) || defined(KYTY_MEMTRACKER_MUTANT_FLIP_SHADOW_BLIND) || \
    defined(KYTY_MEMTRACKER_MUTANT_STICKY_REFAULT_BLIND) || defined(KYTY_MEMTRACKER_MUTANT_RING_NO_TAG) || defined(KYTY_MEMTRACKER_MUTANT_RING_NO_MATCH)
    std::_Exit(1); // a mutant must fail with an exit code (an abort is not inverted by WILL_FAIL)
#else
    std::abort();
#endif
  }
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
using DWORD = uint32_t;
constexpr uint32_t PAGE_NOACCESS = 1;
constexpr uint32_t PAGE_READONLY = 2;
constexpr uint32_t PAGE_READWRITE = 3;
constexpr uint32_t MEM_RESERVE = 0;
constexpr uint32_t MEM_COMMIT = 0;
constexpr uint32_t MEM_RELEASE = 0;

int ToHostProt(uint32_t protection) {
  switch (protection) {
  case PAGE_NOACCESS:
    return PROT_NONE;
  case PAGE_READONLY:
    return PROT_READ;
  default:
    return PROT_READ | PROT_WRITE;
  }
}

uint32_t Protection(const void *address) {
#if defined(__APPLE__)
  mach_vm_address_t region_address =
      reinterpret_cast<mach_vm_address_t>(address);
  mach_vm_size_t region_size = 0;
  vm_region_basic_info_data_64_t info{};
  mach_msg_type_number_t info_count = VM_REGION_BASIC_INFO_COUNT_64;
  mach_port_t object_name = MACH_PORT_NULL;
  Check(mach_vm_region(mach_task_self(), &region_address, &region_size,
                       VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &info_count,
                       &object_name) == KERN_SUCCESS,
        "mach_vm_region failed");
  if (object_name != MACH_PORT_NULL) {
    mach_port_deallocate(mach_task_self(), object_name);
  }
  return (info.protection & VM_PROT_WRITE) != 0
             ? PAGE_READWRITE
             : (info.protection & VM_PROT_READ) != 0 ? PAGE_READONLY
                                                     : PAGE_NOACCESS;
#else
  const auto addr = reinterpret_cast<uintptr_t>(address);
  std::FILE *maps = std::fopen("/proc/self/maps", "r");
  Check(maps != nullptr, "open /proc/self/maps failed");
  char line[512];
  uint32_t result = 0;
  while (std::fgets(line, sizeof(line), maps) != nullptr) {
    unsigned long start = 0;
    unsigned long end = 0;
    char perms[8]{};
    if (std::sscanf(line, "%lx-%lx %7s", &start, &end, perms) != 3) {
      continue;
    }
    if (addr >= start && addr < end) {
      result = perms[1] == 'w'   ? PAGE_READWRITE
               : perms[0] == 'r' ? PAGE_READONLY
                                 : PAGE_NOACCESS;
      break;
    }
  }
  std::fclose(maps);
  return result;
#endif
}

std::map<void *, size_t> &AllocationSizes() {
  static std::map<void *, size_t> sizes;
  return sizes;
}

void *VirtualAlloc(void *address, size_t size, DWORD, uint32_t protection) {
#if defined(__APPLE__)
  mach_vm_address_t raw_address = reinterpret_cast<mach_vm_address_t>(address);
  const auto flags = address != nullptr ? VM_FLAGS_FIXED : VM_FLAGS_ANYWHERE;
  if (mach_vm_allocate(mach_task_self(), &raw_address, size, flags) !=
      KERN_SUCCESS) {
    return nullptr;
  }
  if (mach_vm_protect(mach_task_self(), raw_address, size, false,
                      static_cast<vm_prot_t>(ToHostProt(protection))) !=
      KERN_SUCCESS) {
    mach_vm_deallocate(mach_task_self(), raw_address, size);
    return nullptr;
  }
  void *raw = reinterpret_cast<void *>(raw_address);
  AllocationSizes()[raw] = size;
  return raw;
#else
  const int extra = address != nullptr ? MAP_FIXED_NOREPLACE : 0;
  void *raw = ::mmap(address, size, ToHostProt(protection),
                     MAP_PRIVATE | MAP_ANONYMOUS | extra, -1, 0);
  if (raw == MAP_FAILED) {
    return nullptr;
  }
  AllocationSizes()[raw] = size;
  return raw;
#endif
}

int VirtualFree(void *address, size_t, DWORD) {
  auto &sizes = AllocationSizes();
  auto it = sizes.find(address);
  if (it == sizes.end()) {
    return 0;
  }
#if defined(__APPLE__)
  const int ok = mach_vm_deallocate(mach_task_self(),
                                    reinterpret_cast<mach_vm_address_t>(address),
                                    it->second) == KERN_SUCCESS
                     ? 1
                     : 0;
#else
  const int ok = ::munmap(address, it->second) == 0 ? 1 : 0;
#endif
  sizes.erase(it);
  return ok;
}

int VirtualProtect(void *address, size_t size, uint32_t protection,
                   DWORD *old_protection) {
  if (old_protection != nullptr) {
    *old_protection = Protection(address);
  }
#if defined(__APPLE__)
  return mach_vm_protect(mach_task_self(),
                         reinterpret_cast<mach_vm_address_t>(address), size,
                         false, static_cast<vm_prot_t>(ToHostProt(protection))) ==
                 KERN_SUCCESS
             ? 1
             : 0;
#else
  return ::mprotect(address, size, ToHostProt(protection)) == 0 ? 1 : 0;
#endif
}
#else
uint32_t Protection(const void *address) {
  MEMORY_BASIC_INFORMATION info{};
  Check(VirtualQuery(address, &info, sizeof(info)) != 0, "VirtualQuery failed");
  return info.Protect;
}
#endif

bool IsWritable(const void *address) {
  return Protection(address) == PAGE_READWRITE;
}

uint64_t g_protection_calls = 0;

struct ProtectionCall {
  uint64_t address;
  uint64_t size;
  Common::VirtualMemory::Mode mode;
};

std::vector<ProtectionCall> g_protection_log;
std::mutex g_protection_log_mutex;

void ResetProtectionLog() {
  g_protection_calls = 0;
  g_protection_log.clear();
}

bool ProtectAddressSpace(uint64_t vaddr, uint64_t size,
                         Common::VirtualMemory::Mode mode) {
  uint32_t protection = PAGE_NOACCESS;
  if (mode == Common::VirtualMemory::Mode::Read) {
    protection = PAGE_READONLY;
  } else if (mode == Common::VirtualMemory::Mode::ReadWrite) {
    protection = PAGE_READWRITE;
  }
  DWORD old_protection = 0;
  {
    std::lock_guard lock(g_protection_log_mutex);
    g_protection_calls++;
    g_protection_log.push_back({vaddr, size, mode});
  }
  return VirtualProtect(reinterpret_cast<void *>(vaddr), size, protection,
                        &old_protection) != 0;
}

struct TrackerHarness {
  TrackerHarness() : tracker(page_manager) {}

  PageManager page_manager;
  MemoryTracker tracker;
};

uint8_t *AllocateFixedGuestRange(uint64_t size, uintptr_t offset) {
  // Keep the mapping inside the tracker's guest address space and preserve the
  // requested region alignment. A fixed address can be occupied by the host
  // process (notably by the macOS runner's ASLR layout).
  constexpr uintptr_t first_base = 0x0000000200000000ull;
  constexpr uintptr_t stride = 0x0000000100000000ull;
  for (uintptr_t attempt = 0; attempt < 256; attempt++) {
    auto *wanted = reinterpret_cast<void *>(first_base + offset + attempt * stride);
    auto *memory = static_cast<uint8_t *>(
        VirtualAlloc(wanted, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
    if (memory == wanted) {
      return memory;
    }
    if (memory != nullptr) {
      Check(VirtualFree(memory, 0, MEM_RELEASE) != 0,
            "release unexpected fixed allocation failed");
    }
  }
  Check(false, "no free fixed guest address found");
  return nullptr;
}

uint8_t *Allocate(PageManager &manager, uint64_t pages) {
  const auto size = manager.GetPageSize() * pages;
  return AllocateFixedGuestRange(size, 0x10000);
}

void Release(uint8_t *memory) {
  Check(VirtualFree(memory, 0, MEM_RELEASE) != 0, "VirtualFree failed");
}

void TestRangeSet() {
  RangeSet ranges;
  ranges.Add(0x1000, 0x80);
  ranges.Add(0x1080, 0x80);
  ranges.Add(0x1200, 0x40);
  Check(ranges.Contains(0x1010, 0xe0) && !ranges.Contains(0x1010, 0x200),
        "range set containment did not require full coverage");
  Check(ranges.Intersects(0x0fff, 2) && ranges.Intersects(0x11ff, 2) &&
            !ranges.Intersects(0x1100, 0x100) &&
            !ranges.Intersects(0x1240, 1),
        "range set intersection did not preserve half-open boundaries");
  std::vector<std::pair<uint64_t, uint64_t>> intersections;
  ranges.ForEachInRange(0x1070, 0x1b0, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1070 &&
            intersections[0].second == 0x1100 &&
            intersections[1].first == 0x1200 && intersections[1].second == 0x1220,
        "range set did not merge and intersect exact byte ranges");
  ranges.Subtract(0x1040, 0x1e0);
  intersections.clear();
  ranges.ForEachInRange(0x1000, 0x300, [&](uint64_t start, uint64_t end) {
    intersections.emplace_back(start, end);
  });
  Check(intersections.size() == 2 && intersections[0].first == 0x1000 &&
            intersections[0].second == 0x1040 &&
            intersections[1].first == 0x1220 && intersections[1].second == 0x1240,
        "range set subtraction did not preserve both exact tails");
}

void TestGuestRange() {
  constexpr GuestRange empty{};
  constexpr GuestRange first_byte{1, 1};
  constexpr uint64_t extended_end = Libs::LibKernel::Memory::kExtendedMemoryBase +
                                    Libs::LibKernel::Memory::kExtendedMemorySize;
  constexpr GuestRange last_byte{extended_end - 1, 1};

  static_assert(empty.Empty() && !empty.Valid() && empty.ValidOrEmpty());
  static_assert(!first_byte.Empty() && first_byte.Valid() &&
                first_byte.ValidOrEmpty() && first_byte.End() == 2);
  static_assert(last_byte.Valid() && last_byte.End() == extended_end);

  Check(!GuestRange{0, 1}.Empty() && !GuestRange{0, 1}.ValidOrEmpty(),
        "zero-address nonempty guest range is rejected");
  Check(!GuestRange{1, 0}.Empty() && !GuestRange{1, 0}.ValidOrEmpty(),
        "nonzero-address empty guest range is rejected");
  Check(GuestRange{Libs::LibKernel::Memory::kExtendedMemoryBase, 1}.Valid() &&
            !GuestRange{Libs::Graphics::LOWER_ADDRESS_SIZE, 1}.Valid() &&
            !GuestRange{Libs::LibKernel::Memory::kExtendedMemoryBase - 1, 2}.Valid() &&
            !GuestRange{extended_end - 1, 2}.Valid(),
        "extended range and gap boundaries are enforced");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE, 1}.Valid(),
        "first address beyond the guest range is rejected");
  Check(!GuestRange{TRACKER_ADDRESS_SIZE - 1, 2}.Valid(),
        "guest range crossing the address-space end is rejected");
  Check(!GuestRange{UINT64_MAX, 2}.Valid(), "wrapping guest range is rejected");
}

void TestQueriesDoNotRequireMappedOwnership() {
  constexpr uint64_t address = 0x0000000203000000ull;
  TrackerHarness harness;
  const auto page_size = harness.page_manager.GetPageSize();
  Check(harness.tracker.IsRegionCpuModified(address, page_size) &&
            !harness.tracker.IsRegionGpuModified(address, page_size),
        "unowned tracker range did not expose its initial CPU-dirty state");
}

void TestConcurrentRegionPublication() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  std::binary_semaphore start_first{0};
  std::binary_semaphore start_second{0};
  std::atomic_uint32_t cpu_dirty_results{0};
  std::jthread first([&] {
    start_first.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::jthread second([&] {
    start_second.acquire();
    if (tracker.IsRegionCpuModified(address, page_size)) {
      cpu_dirty_results.fetch_add(1, std::memory_order_relaxed);
    }
  });
  start_first.release();
  start_second.release();
  first.join();
  second.join();

  tracker.UntrackMemory(address, page_size);
  Release(memory);
  Check(cpu_dirty_results.load(std::memory_order_relaxed) == 2,
        "concurrent region publication lost initial CPU ownership");
}

void TestCpuDirtyUpload() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);
  Check(tracker.IsRegionCpuModified(address + 16, 32),
        "new region was not CPU dirty");

  uint32_t ranges = 0;
  bool uploaded = false;
  tracker.ForEachUploadRange(
      address + 16, 32, false,
      [&](uint64_t upload_address, uint64_t upload_size) noexcept {
        Check(upload_address == address && upload_size == page_size,
              "upload range was not page aligned");
        ranges++;
      },
      [&]() noexcept { uploaded = true; });
  Check(ranges == 1 && uploaded &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "upload did not clear CPU dirty state and arm protection");

  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(tracker.IsRegionCpuModified(address, page_size) && IsWritable(memory),
        "explicit CPU dirtiness did not release write protection");
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

// S406 (KYTY_PV_RANGE_GEN): page stamps. A stamp is taken when a page BECOMES CPU- or GPU-modified (never when it is cleaned), so the
// maximum over a range moves whenever any page of it was written, and a range that is clean again after a write still shows it.
// Mutants NO_BUMP_CPU / NO_BUMP_GPU (compile definitions of this target) must fail here.
void TestPageGeneration() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto whole = page_size * 4;

  auto q = tracker.QueryPageGeneration(address, whole);
  Check(!q.registered && !q.clean, "a range of a region the tracker never saw is not registered");
  Check(tracker.IsRegionCpuModified(address, whole), "new region is CPU dirty");
  q = tracker.QueryPageGeneration(address, whole);
  Check(q.registered && !q.clean, "registered but CPU dirty is not clean");

  const auto make_clean = [&] {
    tracker.ForEachUploadRange(
        address, whole, false, [&](uint64_t, uint64_t) noexcept {}, [&]() noexcept {});
  };
  make_clean();
  q = tracker.QueryPageGeneration(address, whole);
  const auto s0 = q.max_stamp;
  Check(q.registered && q.clean, "an uploaded range is clean");

  tracker.MarkRegionAsCpuModified(address + page_size + 16, 32); // page 1
  q = tracker.QueryPageGeneration(address, whole);
  const auto s1 = q.max_stamp;
  Check(!q.clean && s1 > s0, "a CPU write took a newer stamp");
  const auto page0 = tracker.QueryPageGeneration(address, page_size);
  Check(page0.clean && page0.max_stamp == s0, "a write to page 1 does not move page 0");
  make_clean();
  q = tracker.QueryPageGeneration(address, whole);
  Check(q.clean && q.max_stamp == s1, "cleaning takes no stamp: the range is clean again and still shows the write");

  tracker.MarkRegionAsGpuModified(address, page_size);
  q = tracker.QueryPageGeneration(address, whole);
  const auto s2 = q.max_stamp;
  Check(!q.clean && s2 > s1, "a GPU write took a newer stamp");
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  q = tracker.QueryPageGeneration(address, whole);
  Check(q.clean && q.max_stamp == s2, "unmarking takes no stamp");

  bool flushed = false;
  tracker.InvalidateRegion(address + 2 * page_size, page_size, [&] { flushed = true; });
  q = tracker.QueryPageGeneration(address, whole);
  Check(!flushed && !q.clean && q.max_stamp > s2, "invalidation (the fault route) took a newer stamp");

  // A range that runs into a region the tracker has not created is not registered.
  q = tracker.QueryPageGeneration(address, whole + Libs::Graphics::TRACKER_REGION_SIZE);
  Check(!q.registered && !q.clean, "a range leaving the registered regions is not registered");

  tracker.UntrackMemory(address, whole);
  Release(memory);
}
// A repeated ForEachUploadRange is skipped only while its regions are
// unchanged: a CPU write, a GPU unmark or a download in between makes the next
// identical call run again, and a new tracker does not take an old one's skip.
void TestRepeatedUploadSkip() {
  uint8_t *memory = nullptr;
  uint64_t address = 0;
  uint64_t page_size = 0;
  uint32_t ranges = 0;
  uint32_t uploads = 0;
  const auto upload = [&](MemoryTracker &tracker, bool written) {
    ranges = 0;
    tracker.ForEachUploadRange(
        address, page_size, written,
        [&](uint64_t, uint64_t) noexcept { ranges++; },
        [&]() noexcept { uploads++; });
  };
  {
    TrackerHarness harness;
    auto &tracker = harness.tracker;
    page_size = harness.page_manager.GetPageSize();
    memory = Allocate(harness.page_manager, 2);
    address = reinterpret_cast<uint64_t>(memory);

    upload(tracker, false);
    Check(ranges == 1 && uploads == 1, "first upload did not copy the page");
    upload(tracker, false);
    Check(ranges == 0 && uploads == 2 && Protection(memory) == PAGE_READONLY,
          "repeated upload copied or skipped its upload callback");
    tracker.InvalidateRegion(address + 8, 8, [] {});
    upload(tracker, false);
    Check(ranges == 1, "upload after a CPU write was skipped");
    upload(tracker, false);
    Check(ranges == 0 && !tracker.IsRegionCpuModified(address, page_size),
          "upload after a CPU write left the page dirty");

    upload(tracker, true);
    Check(tracker.IsRegionGpuModified(address, page_size) &&
              Protection(memory) == PAGE_NOACCESS,
          "written upload did not mark the page GPU-dirty");
    tracker.UnmarkRegionAsGpuModified(address, page_size);
    upload(tracker, true);
    Check(tracker.IsRegionGpuModified(address, page_size) &&
              Protection(memory) == PAGE_NOACCESS,
          "written upload after an unmark was skipped");
    tracker.ForEachDownloadRange<true>(address, page_size,
                                       [](uint64_t, uint64_t) noexcept {});
    Check(!tracker.IsRegionGpuModified(address, page_size),
          "download did not clear the GPU-dirty page");
    upload(tracker, true);
    Check(tracker.IsRegionGpuModified(address, page_size) &&
              Protection(memory) == PAGE_NOACCESS,
          "written upload after a download was skipped");
    upload(tracker, true);
    Check(ranges == 0 && tracker.IsRegionGpuModified(address, page_size) &&
              uploads == 8,
          "repeated written upload changed the page's state");

    tracker.UnmarkRegionAsGpuModified(address, page_size * 2);
    tracker.UntrackMemory(address, page_size * 2);
  }
  {
    TrackerHarness harness;
    upload(harness.tracker, false);
    Check(ranges == 1, "a new tracker skipped its first upload");
    harness.tracker.UntrackMemory(address, page_size * 2);
  }
  Release(memory);
}

void TestCleanUploadPreservesOwnership() {
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(region_size * 2, region_size);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, region_size * 2, false,
                             [](uint64_t, uint64_t) noexcept {},
                             []() noexcept {});

  for (const auto start : {address + 63 * page_size,
                           address + region_size - page_size}) {
    const auto before = start - page_size;
    const auto after = start + 2 * page_size;
    tracker.MarkRegionAsCpuModified(before, page_size);
    tracker.MarkRegionAsCpuModified(after, page_size);
    uint32_t ranges = 0;
    uint32_t completions = 0;
    ResetProtectionLog();
    tracker.ForEachUploadRange(
        start, 2 * page_size, false,
        [&](uint64_t, uint64_t) noexcept { ranges++; },
        [&]() noexcept { completions++; });
    Check(ranges == 0 && completions == 1 && g_protection_calls == 0 &&
              !tracker.IsRegionCpuModified(start, 2 * page_size),
          "clean read-only upload changed dirty state or protection");

    tracker.ForEachUploadRange(
        start, 2 * page_size, true,
        [&](uint64_t, uint64_t) noexcept { ranges++; },
        [&]() noexcept { completions++; });
    Check(ranges == 0 && completions == 2 &&
              tracker.IsRegionGpuModified(start, page_size) &&
              tracker.IsRegionGpuModified(start + page_size, page_size) &&
              Protection(reinterpret_cast<void *>(start)) == PAGE_NOACCESS &&
              Protection(reinterpret_cast<void *>(start + page_size)) ==
                  PAGE_NOACCESS &&
              tracker.IsRegionCpuModified(before, page_size) &&
              tracker.IsRegionCpuModified(after, page_size) &&
              !tracker.IsRegionGpuModified(before, page_size) &&
              !tracker.IsRegionGpuModified(after, page_size) &&
              IsWritable(reinterpret_cast<void *>(before)) &&
              IsWritable(reinterpret_cast<void *>(after)),
          "clean written upload lost GPU ownership or changed dirty neighbors");
    tracker.UnmarkRegionAsGpuModified(start, 2 * page_size);
  }
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void BenchmarkCleanUploads() {
  constexpr uint64_t size = 256ull * 1024 * 1024;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(size, 0);
  const auto address = reinterpret_cast<uint64_t>(memory);
  tracker.ForEachUploadRange(address, size, false,
                             [](uint64_t, uint64_t) noexcept {},
                             []() noexcept {});
  std::puts("chunk_bytes,sweeps,calls,elapsed_ns,ns_per_sweep,ns_per_call");
  for (const uint64_t chunk : {size, Libs::Graphics::TRACKER_REGION_SIZE,
                               uint64_t{64 * 1024}}) {
    uint64_t ranges = 0;
    uint64_t completions = 0;
    uint64_t sweeps = 0;
    ResetProtectionLog();
    const auto start = std::chrono::steady_clock::now();
    std::chrono::nanoseconds elapsed{};
    do {
      for (uint64_t offset = 0; offset < size; offset += chunk) {
        tracker.ForEachUploadRange(
            address + offset, chunk, false,
            [&](uint64_t, uint64_t) noexcept { ranges++; },
            [&]() noexcept { completions++; });
      }
      sweeps++;
      elapsed = std::chrono::steady_clock::now() - start;
    } while (elapsed < std::chrono::milliseconds(250));
    Check(ranges == 0 && completions == sweeps * (size / chunk) &&
              g_protection_calls == 0,
          "clean upload benchmark changed ranges, completion, or protection");
    std::printf("%llu,%llu,%llu,%lld,%.2f,%.2f\n",
                static_cast<unsigned long long>(chunk),
                static_cast<unsigned long long>(sweeps),
                static_cast<unsigned long long>(completions),
                static_cast<long long>(elapsed.count()),
                static_cast<double>(elapsed.count()) / sweeps,
                static_cast<double>(elapsed.count()) / completions);
  }
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestRangeInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  constexpr uint64_t size = Libs::Graphics::TRACKER_REGION_SIZE * 2;
  auto *memory = AllocateFixedGuestRange(size, 0x1000000);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, size) && !IsWritable(memory),
        "range invalidation setup did not establish GPU ownership");
  uint32_t flushes = 0;
  tracker.InvalidateRegion(address + 16, size - 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, size - 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, size - 32);
  });
  Check(flushes == 1 && !tracker.IsRegionGpuModified(address, size) &&
            tracker.IsRegionCpuModified(address, size) && IsWritable(memory) &&
            IsWritable(memory + size - 1),
        "range invalidation did not batch ownership transfer across regions");
  tracker.InvalidateRegion(address + 16, size - 32, [&] { flushes++; });
  Check(flushes == 1,
        "clean range invalidation unnecessarily requested a GPU flush");
  tracker.UntrackMemory(address, size);
  Release(memory);
}

void TestGpuReacquisitionAfterInvalidation() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size),
        "reacquisition setup did not establish GPU ownership");

  uint32_t flushes = 0;
  uint32_t uploads = 0;
  std::binary_semaphore reacquire{0};
  std::binary_semaphore reacquired{0};
  std::jthread publisher([&] {
    reacquire.acquire();
    tracker.ForEachUploadRange(
        address + 16, 32, true,
        [&](uint64_t, uint64_t) noexcept { uploads++; }, []() noexcept {});
    reacquired.release();
  });
  tracker.InvalidateRegion(address + 16, 32, [&] {
    flushes++;
    tracker.ForEachDownloadRange<true>(address + 16, 32,
                                       [](uint64_t, uint64_t) noexcept {});
    tracker.MarkRegionAsCpuModified(address + 16, 32);
    reacquire.release();
    reacquired.acquire();
  });
  publisher.join();
  Check(flushes == 1 && uploads == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionCpuModified(address, page_size) &&
            !IsWritable(memory),
        "invalidation rejected a new generation of GPU ownership");

  tracker.UnmarkRegionAsGpuModified(address, page_size);
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDirtyBits() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 2);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  Check(tracker.IsRegionGpuModified(address, page_size) &&
            !tracker.IsRegionGpuModified(address + page_size, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "GPU dirty state escaped the requested range");
  tracker.UnmarkRegionAsGpuModified(address, page_size);
  Check(!tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "GPU dirty state did not restore write-only tracking");
  tracker.MarkRegionAsCpuModified(address, page_size);
  tracker.UntrackMemory(address, page_size * 2);
  Release(memory);
}

void TestExactDirtyIntervalsSharingTrackerPage() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  RangeSet exact_dirty;
  exact_dirty.Add(address + 64, 16);
  exact_dirty.Add(address + 192, 32);

  ResetProtectionLog();
  tracker.MarkRegionAsGpuModified(address + 64, 16);
  tracker.MarkRegionAsGpuModified(address + 192, 32);
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "disjoint byte dirtiness duplicated the page watcher");

  exact_dirty.Subtract(address + 64, 16);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 1 &&
            tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_NOACCESS,
        "draining one exact interval prematurely released its shared page");

  exact_dirty.Subtract(address + 192, 32);
  if (!exact_dirty.Intersects(address, page_size)) {
    tracker.UnmarkRegionAsGpuModified(address, page_size);
  }
  Check(g_protection_calls == 2 &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            Protection(memory) == PAGE_READONLY,
        "draining the final exact interval did not release its tracker page");

  tracker.UntrackMemory(address, page_size);
  Release(memory);
}

void TestGpuDownloadProtectionMirrors() {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);

  tracker.ForEachUploadRange(
      address, page_size * 4, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(address + 16, 32);
  tracker.MarkRegionAsGpuModified(address + page_size * 2 + 16, 32);

  std::vector<std::pair<uint64_t, uint64_t>> visited;
  ResetProtectionLog();
  tracker.ForEachDownloadRange<false>(
      address, page_size * 3,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.size() == 2 && visited[0].first == address &&
            visited[0].second == page_size &&
            visited[1].first == address + page_size * 2 &&
            visited[1].second == page_size && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address, page_size * 3),
        "non-clearing download changed protection or lost sparse ranges");

  visited.clear();
  bool protected_during_download = false;
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        protected_during_download = Protection(memory) == PAGE_NOACCESS;
        visited.push_back({range_address, range_size});
      });
  Check(protected_during_download && visited.size() == 1 &&
            visited[0].first == address &&
            visited[0].second == page_size && g_protection_log.size() == 1 &&
            g_protection_log[0].address == address &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(address, page_size) &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size) &&
            Protection(memory) == PAGE_READONLY &&
            Protection(memory + page_size * 2) == PAGE_NOACCESS,
        "partial download did not preserve the CPU/GPU protection mirrors");

  visited.clear();
  ResetProtectionLog();
  tracker.ForEachDownloadRange<true>(
      address + 16, 32,
      [&](uint64_t range_address, uint64_t range_size) noexcept {
        visited.push_back({range_address, range_size});
      });
  Check(visited.empty() && g_protection_calls == 0 &&
            tracker.IsRegionGpuModified(address + page_size * 2, page_size),
        "idempotent partial download disturbed another GPU-owned page");

  tracker.UnmarkRegionAsGpuModified(address, page_size * 3);
  Check(!tracker.IsRegionGpuModified(address, page_size * 3) &&
            Protection(memory + page_size * 2) == PAGE_READONLY,
        "broad final unmark did not restore write-only tracking");
  ResetProtectionLog();
  tracker.MarkRegionAsCpuModified(address + 16, 32);
  Check(
      g_protection_log.size() == 1 && g_protection_log[0].address == address &&
          g_protection_log[0].size == page_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::ReadWrite &&
          IsWritable(memory) && !IsWritable(memory + page_size),
      "CPU-dirty transition did not release only its write watcher");

  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

void TestCrossRegionUpload() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  auto *memory = AllocateFixedGuestRange(region_size * 2, 0x10000);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto boundary = (address + region_size - 1) & ~(region_size - 1);
  for (const bool is_written : {false, true}) {
    TrackerHarness harness;
    auto &tracker = harness.tracker;
    uint32_t ranges = 0;
    tracker.ForEachUploadRange(
        boundary - page_size, page_size * 2, is_written,
        [&](uint64_t start, uint64_t size) noexcept {
          Check(start == boundary - page_size + ranges * page_size &&
                    size == page_size,
                "cold upload did not visit both region boundaries exactly");
          ranges++;
        },
        [&]() noexcept {
          Check(ranges == 2 &&
                    Protection(reinterpret_cast<void *>(boundary - page_size)) ==
                        PAGE_READONLY &&
                    Protection(reinterpret_cast<void *>(boundary)) == PAGE_READONLY,
                "cold upload completed before clearing and protecting both regions");
        });
    for (const auto page : {boundary - page_size, boundary}) {
      Check(!tracker.IsRegionCpuModified(page, page_size) &&
                tracker.IsRegionGpuModified(page, page_size) == is_written &&
                Protection(reinterpret_cast<void *>(page)) ==
                    (is_written ? PAGE_NOACCESS : PAGE_READONLY),
            "cross-region upload did not preserve final ownership and protection");
    }
    Check(IsWritable(reinterpret_cast<void *>(boundary - 2 * page_size)) &&
              IsWritable(reinterpret_cast<void *>(boundary + page_size)),
          "cross-region upload changed neighboring pages");
    tracker.UnmarkRegionAsGpuModified(boundary - page_size, 2 * page_size);
    tracker.UntrackMemory(address, region_size * 2);
  }
  Release(memory);
}

void TestConcurrentColdUploads() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto *memory = AllocateFixedGuestRange(region_size * 2, region_size);
  const auto address = reinterpret_cast<uint64_t>(memory);
  std::counting_semaphore<2> start{0};
  std::counting_semaphore<2> upload_entered{0};
  std::counting_semaphore<2> finish_upload{0};
  std::vector<std::jthread> workers;
  for (const auto page : {address, address + region_size}) {
    workers.emplace_back([&, page] {
      start.acquire();
      uint32_t ranges = 0;
      tracker.ForEachUploadRange(
          page, page_size, true,
          [&](uint64_t upload_address, uint64_t upload_size) noexcept {
            Check(upload_address == page && upload_size == page_size,
                  "concurrent cold upload lost its dirty page");
            ranges++;
          },
          [&]() noexcept {
            Check(ranges == 1, "concurrent cold upload skipped its dirty page");
            upload_entered.release();
            finish_upload.acquire();
          });
    });
  }
  start.release(2);
  const bool first_entered = upload_entered.try_acquire_for(std::chrono::seconds(5));
  const bool second_entered = upload_entered.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release(2);
  for (auto &worker : workers) {
    worker.join();
  }
  Check(first_entered && second_entered,
        "cold writable uploads serialized disjoint regions");
  for (const auto page : {address, address + region_size}) {
    Check(!tracker.IsRegionCpuModified(page, page_size) &&
              tracker.IsRegionGpuModified(page, page_size) &&
              Protection(reinterpret_cast<void *>(page)) == PAGE_NOACCESS,
          "concurrent cold upload did not retain GPU ownership");
    tracker.UnmarkRegionAsGpuModified(page, page_size);
  }
  tracker.UntrackMemory(address, region_size * 2);
  Release(memory);
}

void TestUploadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  // Publish both managers before the concurrent section so this test measures
  // tracker access serialization rather than manager allocation.
  Check(tracker.IsRegionCpuModified(allocation_base, page_size) &&
            tracker.IsRegionCpuModified(second_region, page_size),
        "disjoint upload setup did not initialize both regions");

  std::binary_semaphore upload_entered{0};
  std::binary_semaphore finish_upload{0};
  std::binary_semaphore query_finished{0};
  std::atomic_bool query_result{false};
  std::jthread uploader([&] {
    tracker.ForEachUploadRange(
        allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          upload_entered.release();
          finish_upload.acquire();
        });
  });
  upload_entered.acquire();
  std::jthread query([&] {
    query_result.store(tracker.IsRegionCpuModified(second_region, page_size),
                       std::memory_order_relaxed);
    query_finished.release();
  });

  const bool completed_while_upload_blocked =
      query_finished.try_acquire_for(std::chrono::seconds(5));
  finish_upload.release();
  uploader.join();
  query.join();

  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_upload_blocked &&
            query_result.load(std::memory_order_relaxed),
        "upload callback serialized an unrelated tracker region");
}

void TestDownloadDoesNotSerializeDisjointRegion() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto second_region =
      (allocation_base & ~(region_size - 1)) + region_size;
  Check(second_region + page_size <= allocation_base + region_size * 2,
        "test allocation does not span two tracker regions");

  tracker.ForEachUploadRange(
      allocation_base, page_size, true, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.ForEachUploadRange(
      second_region, page_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});

  std::binary_semaphore download_entered{0};
  std::binary_semaphore finish_download{0};
  std::binary_semaphore mutation_finished{0};
  std::jthread downloader([&] {
    tracker.ForEachDownloadRange<false>(
        allocation_base, second_region + page_size - allocation_base,
        [&](uint64_t address, uint64_t) noexcept {
          if (address == allocation_base) {
            download_entered.release();
            finish_download.acquire();
          }
        });
  });
  download_entered.acquire();
  std::jthread mutation([&] {
    tracker.MarkRegionAsGpuModified(second_region, page_size);
    mutation_finished.release();
  });

  const bool completed_while_download_blocked =
      mutation_finished.try_acquire_for(std::chrono::seconds(5));
  finish_download.release();
  downloader.join();
  mutation.join();

  const bool both_gpu_owned =
      tracker.IsRegionGpuModified(allocation_base, page_size) &&
      tracker.IsRegionGpuModified(second_region, page_size);
  tracker.UnmarkRegionAsGpuModified(allocation_base, page_size);
  tracker.UnmarkRegionAsGpuModified(second_region, page_size);
  tracker.MarkRegionAsCpuModified(allocation_base, page_size);
  tracker.MarkRegionAsCpuModified(second_region, page_size);
  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
  Check(completed_while_download_blocked && both_gpu_owned,
        "download callback serialized an unrelated tracker region");
}

void TestGpuUnmarkUsesRegionMask() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size + page_size <=
            allocation_base + region_size * 2,
        "test allocation does not span two complete tracker regions");

  const auto sparse_begin = region_base + page_size;
  tracker.ForEachUploadRange(
      sparse_begin, page_size * 3, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(sparse_begin, page_size);
  tracker.MarkRegionAsGpuModified(sparse_begin + page_size * 2, page_size);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(
      g_protection_calls == 1 && g_protection_log.size() == 1 &&
          g_protection_log[0].address == sparse_begin &&
          g_protection_log[0].size == page_size * 3 &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(sparse_begin, page_size * 3) &&
          Protection(reinterpret_cast<void *>(sparse_begin)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size)) ==
              PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(sparse_begin + page_size * 2)) ==
              PAGE_READONLY,
      "GPU unmark did not coalesce a sparse 4 MiB region mask");
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(sparse_begin, page_size * 3);
  Check(g_protection_calls == 0,
        "idempotent GPU unmark performed a protection call");

  const auto boundary = region_base + region_size;
  const auto cross_begin = boundary - page_size;
  tracker.ForEachUploadRange(
      cross_begin, page_size * 2, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(cross_begin, page_size * 2);
  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(cross_begin, page_size * 2);
  Check(g_protection_calls == 2 && g_protection_log.size() == 2 &&
            g_protection_log[0].address == cross_begin &&
            g_protection_log[0].size == page_size &&
            g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
            g_protection_log[1].address == boundary &&
            g_protection_log[1].size == page_size &&
            g_protection_log[1].mode == Common::VirtualMemory::Mode::Read &&
            !tracker.IsRegionGpuModified(cross_begin, page_size * 2),
        "cross-region GPU unmark did not use one update per 4 MiB region");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

void TestFullRegionGpuUnmarkBatching() {
  constexpr auto region_size = Libs::Graphics::TRACKER_REGION_SIZE;
  constexpr auto page_size = Libs::Graphics::TRACKER_PAGE_SIZE;
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  auto *memory = Allocate(page_manager, region_size * 2 / page_size);
  const auto allocation_base = reinterpret_cast<uint64_t>(memory);
  const auto region_base =
      (allocation_base + region_size - 1) & ~(region_size - 1);
  Check(region_base + region_size <= allocation_base + region_size * 2,
        "test allocation does not contain a complete tracker region");

  tracker.ForEachUploadRange(
      region_base, region_size, false, [](uint64_t, uint64_t) noexcept {},
      []() noexcept {});
  tracker.MarkRegionAsGpuModified(region_base, region_size);
  Check(tracker.IsRegionGpuModified(region_base, region_size) &&
            Protection(reinterpret_cast<void *>(region_base)) ==
                PAGE_NOACCESS &&
            Protection(reinterpret_cast<void *>(region_base + region_size -
                                                page_size)) == PAGE_NOACCESS,
        "full-region setup did not establish GPU read protection");

  ResetProtectionLog();
  tracker.UnmarkRegionAsGpuModified(region_base, region_size);
  Check(
      g_protection_log.size() == 1 &&
          g_protection_log[0].address == region_base &&
          g_protection_log[0].size == region_size &&
          g_protection_log[0].mode == Common::VirtualMemory::Mode::Read &&
          !tracker.IsRegionGpuModified(region_base, region_size) &&
          Protection(reinterpret_cast<void *>(region_base)) == PAGE_READONLY &&
          Protection(reinterpret_cast<void *>(region_base + region_size -
                                              page_size)) == PAGE_READONLY,
      "full-region GPU unmark did not use one exact 4 MiB protection request");

  tracker.UntrackMemory(allocation_base, region_size * 2);
  Release(memory);
}

[[noreturn]] void RunDeathCase(const char *name) {
  TrackerHarness harness;
  auto &tracker = harness.tracker;
  auto &page_manager = harness.page_manager;
  const auto page_size = page_manager.GetPageSize();
  auto *memory = Allocate(page_manager, 1);
  const auto address = reinterpret_cast<uint64_t>(memory);
  if (std::strcmp(name, "gpu-dirty-explicit-cpu") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        []() noexcept {});
    tracker.MarkRegionAsCpuModified(address, page_size);
  } else if (std::strcmp(name, "reentrant-upload") == 0) {
    tracker.ForEachUploadRange(
        address, page_size, true, [](uint64_t, uint64_t) noexcept {},
        [&]() noexcept {
          (void)tracker.IsRegionCpuModified(address, page_size);
        });
  } else if (std::strcmp(name, "recursive-tracking-lock") == 0) {
    Libs::Graphics::TrackingSpinLock lock;
    lock.lock();
    lock.lock();
  } else if (std::strcmp(name, "non-owner-tracking-unlock") == 0) {
    Libs::Graphics::TrackingSpinLock lock;
    lock.lock();
    std::thread worker([&] { lock.unlock(); });
    worker.join();
  }
  std::_Exit(0x7f);
}

void CheckDeathCase(const char *name) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
  char path[MAX_PATH]{};
  Check(GetModuleFileNameA(nullptr, path, MAX_PATH) != 0,
        "GetModuleFileName failed");
  std::string command = std::string("\"") + path + "\" --death " + name;
  std::vector<char> mutable_command(command.begin(), command.end());
  mutable_command.push_back('\0');
  STARTUPINFOA startup{sizeof(startup)};
  PROCESS_INFORMATION process{};
  Check(CreateProcessA(nullptr, mutable_command.data(), nullptr, nullptr, FALSE,
                       CREATE_NO_WINDOW, nullptr, nullptr, &startup,
                       &process) != 0,
        "CreateProcess failed");
  Check(WaitForSingleObject(process.hProcess, 10000) == WAIT_OBJECT_0,
        "MemoryTracker death test timed out");
  DWORD exit_code = 0;
  Check(
      GetExitCodeProcess(process.hProcess, &exit_code) != 0 &&
          (exit_code == 321 || exit_code == EXCEPTION_NONCONTINUABLE_EXCEPTION),
      "MemoryTracker death path used the wrong exit");
  CloseHandle(process.hThread);
  CloseHandle(process.hProcess);
#else
#if defined(__APPLE__)
  std::vector<char> path(PATH_MAX);
  uint32_t path_size = static_cast<uint32_t>(path.size());
  if (_NSGetExecutablePath(path.data(), &path_size) != 0) {
    path.resize(path_size);
    Check(_NSGetExecutablePath(path.data(), &path_size) == 0,
          "_NSGetExecutablePath failed");
  }
#endif
  const pid_t pid = ::fork();
  Check(pid >= 0, "fork failed");
  if (pid == 0) {
#if defined(__APPLE__)
    ::execl(path.data(), "MemoryTrackerTests", "--death", name, nullptr);
#else
    ::execl("/proc/self/exe", "MemoryTrackerTests", "--death", name, nullptr);
#endif
    std::_Exit(0x7e);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid, "waitpid failed");
  const bool fatal_exit =
      WIFEXITED(status) && WEXITSTATUS(status) == (321 & 0xff);
  Check(fatal_exit || WIFSIGNALED(status),
        "MemoryTracker death path used the wrong exit");
#endif
}

void TestFatalPaths() {
  for (const char *name : {"gpu-dirty-explicit-cpu", "reentrant-upload",
                           "recursive-tracking-lock", "non-owner-tracking-unlock"}) {
    CheckDeathCase(name);
  }
}

#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
void *g_fault_stack = nullptr;
constexpr size_t FAULT_STACK_SIZE = 64 * 1024;
volatile sig_atomic_t g_stack_faults = 0;

bool HandleStackFault(const Common::HostException::ExceptionInfo &info) {
  using namespace Common::HostException;
  stack_t active_stack{};
  const auto fault_address = reinterpret_cast<uintptr_t>(g_fault_stack) +
                             FAULT_STACK_SIZE - sizeof(uintptr_t);
  if (info.type != ExceptionType::AccessViolation ||
      info.access_violation_type != AccessViolationType::Write ||
      info.access_violation_vaddr != fault_address ||
      ::sigaltstack(nullptr, &active_stack) != 0 ||
      (active_stack.ss_flags & SS_ONSTACK) == 0) {
    std::_Exit(1);
  }
  g_stack_faults = 1;
  return ::mprotect(g_fault_stack, FAULT_STACK_SIZE,
                    PROT_READ | PROT_WRITE) == 0;
}

// A stack write must fault before any signal frame can use the protected stack.
__attribute__((naked)) void WriteProtectedStack(void *) {
  asm volatile("mov %rsp, %rax\n"
               "mov %rdi, %rsp\n"
               "push %rax\n"
               "pop %rsp\n"
               "ret\n");
}

void TestFaultOnProtectedStack() {
  const pid_t pid = ::fork();
  Check(pid >= 0, "stack fault fork failed");
  if (pid == 0) {
    std::thread worker([] {
      Check(Common::HostException::InitializeThreadSignalStack(),
            "initialize thread signal stack failed");
      Check(Common::HostException::InstallHandler(HandleStackFault),
            "install stack fault handler failed");
      g_fault_stack = ::mmap(nullptr, FAULT_STACK_SIZE, PROT_READ,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      Check(g_fault_stack != MAP_FAILED, "allocate protected stack failed");
      WriteProtectedStack(static_cast<char *>(g_fault_stack) + FAULT_STACK_SIZE);
      Check(g_stack_faults == 1, "protected stack write did not resume");
      struct sigaction action{};
      Check(::sigaction(SIGSEGV, nullptr, &action) == 0 &&
                action.sa_handler != SIG_DFL,
            "stack fault reset the process handler");
      Check(::munmap(g_fault_stack, FAULT_STACK_SIZE) == 0,
            "release protected stack failed");
    });
    worker.join();
    std::_Exit(0);
  }
  int status = 0;
  Check(::waitpid(pid, &status, 0) == pid && WIFEXITED(status) &&
            WEXITSTATUS(status) == 0,
        "fault on a protected stack did not recover");
}
#endif

} // namespace

namespace Libs::LibKernel::Memory {

bool ProtectGuestHostMemory(uint64_t vaddr, uint64_t size,
                            Common::VirtualMemory::Mode mode) {
  return ProtectAddressSpace(vaddr, size, mode);
}

} // namespace Libs::LibKernel::Memory

// S440 (KYTY_WF_HOT_UNPROTECT): a page written in kHotStreak flips in a row stops being protected; the hash of the page at every upload decides
// whether it changed; kColdAfter flips without a change protect it again. Mutants HOT_SKIPS_HASH / HOT_NEVER_REPROTECT must fail here.
uint64_t g_hot_epoch = 0;

uint64_t HotEpoch() { return g_hot_epoch; }

bool HotHash(uint64_t page, uint64_t *hash) {
  const auto *bytes = reinterpret_cast<const uint8_t *>(page);
  uint64_t value = 1469598103934665603ull;
  for (uint64_t i = 0; i < 4096; i++) {
    value = (value ^ bytes[i]) * 1099511628211ull;
  }
  *hash = value;
  return true;
}

uint32_t HotUpload(MemoryTracker &tracker, uint64_t address, uint64_t size,
                   bool is_written = false) {
  uint32_t ranges = 0;
  tracker.ForEachUploadRange(
      address, size, is_written,
      [&](uint64_t, uint64_t) noexcept { ranges++; }, []() noexcept {});
  return ranges;
}

void HotFault(MemoryTracker &tracker, uint64_t address, uint64_t size) {
  tracker.InvalidateRegion(address, size, [] {});
}

void TestHotPages() {
  using Libs::Graphics::HotPages::GetStats;
  using Libs::Graphics::HotPages::Mode;
  Libs::Graphics::HotPages::EpochSource().store(&HotEpoch);
  Libs::Graphics::HotPages::HasherSource().store(&HotHash);
  Libs::Graphics::HotPages::ModeOverride().store(static_cast<int>(Mode::On));
  auto &stats = GetStats();
  const auto promote0 = stats.promote.load();
  const auto demote0 = stats.demote.load();

  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto page = address + page_size; // page 1; page 0 and 2 stay ordinary
  g_hot_epoch = 0;
  Check(HotUpload(tracker, address, page_size * 4) == 1,
        "first upload did not report the new region");

  // Written and uploaded in kHotStreak flips in a row: still protected before the last one, unprotected after it.
  for (uint32_t flip = 1; flip <= Libs::Graphics::HotPages::kHotStreak; flip++) {
    g_hot_epoch = flip;
    HotFault(tracker, page, 1);
    memory[page_size] = static_cast<uint8_t>(flip);
    Check(HotUpload(tracker, page, 1) == 1, "write before the page was hot was not uploaded");
    if (flip < Libs::Graphics::HotPages::kHotStreak) {
      Check(!IsWritable(memory + page_size), "page unprotected before it was hot");
    }
  }
  Check(stats.promote.load() == promote0 + 1 && stats.hot_pages.load() == 1,
        "page did not become hot after kHotStreak flips");
  Check(IsWritable(memory + page_size) && tracker.IsRegionCpuModified(page, 1),
        "hot page was protected again by its upload");
  Check(HotUpload(tracker, address, page_size) == 0 && IsWritable(memory) == false,
        "ordinary neighbour page changed");

  // The hash decides: a write that raised no fault is uploaded once, an unchanged page is not uploaded.
  uint64_t last_change = Libs::Graphics::HotPages::kHotStreak;
  g_hot_epoch = last_change + 1;
  memory[page_size + 7] = 0x5a;
  Check(HotUpload(tracker, page, 1) == 1, "hash did not notice a write without a fault");
  last_change = g_hot_epoch;
  Check(HotUpload(tracker, page, 1) == 0 && IsWritable(memory + page_size),
        "unchanged hot page was uploaded or protected");
  const auto skip0 = stats.skip.load();
  Check(HotUpload(tracker, page, 1) == 0 && stats.skip.load() == skip0 + 1,
        "unchanged hot page was not counted as skipped");

  // Hot pages are reported for BDA synchronisation although they raise no fault.
  {
    Libs::Graphics::RangeSet bda;
    tracker.AddHotPages(bda); // BufferCache::TakeBdaSync adds them (S487: the BDA queue moved there)
    Check(bda.Intersects(page, 1), "hot page missing from BDA sync");
  }

  // An explicit CPU modification makes the next upload happen although the bytes did not change.
  tracker.MarkRegionAsCpuModified(page, 1);
  Check(HotUpload(tracker, page, 1) == 1 && HotUpload(tracker, page, 1) == 0,
        "explicit CPU modification did not force an upload of a hot page");

  // kColdAfter flips without a change: the last unprotected upload is one flip earlier.
  g_hot_epoch = last_change + Libs::Graphics::HotPages::kColdAfter - 1;
  Check(HotUpload(tracker, page, 1) == 0 && IsWritable(memory + page_size) && stats.demote.load() == demote0,
        "hot page was protected too early");
  g_hot_epoch = last_change + Libs::Graphics::HotPages::kColdAfter;
  Check(HotUpload(tracker, page, 1) == 0 && !IsWritable(memory + page_size) &&
            !tracker.IsRegionCpuModified(page, 1) && stats.demote.load() == demote0 + 1 &&
            stats.hot_pages.load() == 0,
        "unchanged hot page was not protected again");
  // ...and an ordinary write is seen by the fault again.
  g_hot_epoch++;
  HotFault(tracker, page, 1);
  memory[page_size + 1] = 0x33;
  Check(HotUpload(tracker, page, 1) == 1, "write after the page went cold was not uploaded");

  // The GPU takes a hot page for writing: hot state ends, ownership moves without a conflict.
  for (uint32_t flip = 0; flip < Libs::Graphics::HotPages::kHotStreak; flip++) {
    g_hot_epoch++;
    HotFault(tracker, page, 1);
    memory[page_size + 2] = static_cast<uint8_t>(g_hot_epoch);
    Check(HotUpload(tracker, page, 1) == 1, "second streak was not uploaded");
  }
  Check(stats.hot_pages.load() == 1 && IsWritable(memory + page_size), "page did not turn hot a second time");
  (void)HotUpload(tracker, page, 1, true);
  Check(stats.hot_pages.load() == 0 && tracker.IsRegionGpuModified(page, 1) &&
            !IsWritable(memory + page_size) && !tracker.IsRegionCpuModified(page, 1),
        "GPU write ownership of a hot page failed");
  tracker.UnmarkRegionAsGpuModified(page, 1);

  // A hash from an earlier hot period must not decide a later one: A (hashed) -> cold -> B uploaded unhashed -> hot again with A written back.
  {
    const auto aba = address + 3 * page_size;
    const auto first_epoch = g_hot_epoch + 10;
    for (uint32_t flip = 0; flip < Libs::Graphics::HotPages::kHotStreak; flip++) {
      g_hot_epoch = first_epoch + flip;
      HotFault(tracker, aba, 1);
      memory[3 * page_size] = static_cast<uint8_t>(0xa1);
      (void)HotUpload(tracker, aba, 1);
    }
    g_hot_epoch += Libs::Graphics::HotPages::kColdAfter;
    Check(HotUpload(tracker, aba, 1) == 0 && !IsWritable(memory + 3 * page_size), "ABA page did not go cold");
    for (uint32_t flip = 0; flip < Libs::Graphics::HotPages::kHotStreak; flip++) {
      g_hot_epoch++;
      HotFault(tracker, aba, 1);
      memory[3 * page_size] = flip + 1 < Libs::Graphics::HotPages::kHotStreak ? 0xb2 : 0xa1;
      Check(HotUpload(tracker, aba, 1) == 1, "page written back to its old value was not uploaded after a cold period");
    }
    tracker.UntrackMemory(aba, 1);
  }

  // Shadow: protection and uploads are as ever; unchanged hot pages are counted; a write the tracker never saw is counted in ne.
  Libs::Graphics::HotPages::ModeOverride().store(static_cast<int>(Mode::Shadow));
  tracker.UntrackMemory(address, page_size * 4);
  (void)HotUpload(tracker, address, page_size * 4);
  const auto skip1 = stats.skip.load();
  const auto ne0 = stats.ne.load();
  for (uint32_t flip = 0; flip < Libs::Graphics::HotPages::kHotStreak + 2; flip++) {
    g_hot_epoch++;
    HotFault(tracker, page, 1);
    Check(HotUpload(tracker, page, 1) == 1 && !IsWritable(memory + page_size),
          "shadow changed the uploads or the protection");
  }
  Check(stats.skip.load() > skip1 && stats.ne.load() == ne0, "shadow did not count unchanged hot pages");
  Check(HotUpload(tracker, page, 1) == 0 && stats.ne.load() == ne0, "shadow counted a clean unchanged page");
  DWORD old = 0;
  Check(VirtualProtect(memory + page_size, page_size, PAGE_READWRITE, &old) != 0, "VirtualProtect failed");
  memory[page_size + 3] ^= 0xff;
  (void)HotUpload(tracker, page, 1);
  Check(stats.ne.load() == ne0 + 1, "shadow missed a write that raised no fault");

  Libs::Graphics::HotPages::ModeOverride().store(-1);
  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

// S443 (KYTY_WF_FLIP_UNPROTECT): a page with a second write fault in one flip stays unprotected until the flip ends; every upload copies it; the first upload
// after the flip protects it again. Mutants FLIP_NEVER_DEMOTE / FLIP_SKIPS_UPLOAD must fail the On part, FLIP_SHADOW_BLIND the shadow part.
uint64_t g_noise = 0;

bool HotHashNoisy(uint64_t page, uint64_t *hash) {
  const bool ok = HotHash(page, hash);
  *hash ^= g_noise;
  if (g_noise != 0) {
    g_noise++; // every call sees another value
  }
  return ok;
}

void TestFlipUnprotect() {
  using Libs::Graphics::FlipUnprotect::GetStats;
  using Libs::Graphics::HotPages::Mode;
  Libs::Graphics::HotPages::EpochSource().store(&HotEpoch);
  Libs::Graphics::HotPages::HasherSource().store(&HotHash);
  Libs::Graphics::FlipUnprotect::ModeOverride().store(static_cast<int>(Mode::On));
  auto &stats = GetStats();
  const auto promote0 = stats.promote.load();
  const auto demote0 = stats.demote.load();
  const auto refault0 = stats.refault.load();

  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto page = address + page_size;
  g_hot_epoch = 100;
  Check(HotUpload(tracker, address, page_size * 4) == 1, "first upload did not report the new region");

  // One fault in a flip: protected again by the upload, nothing kept.
  HotFault(tracker, page, 1);
  memory[page_size] = 1;
  Check(HotUpload(tracker, page, 1) == 1 && !IsWritable(memory + page_size), "single fault page was not protected again");
  Check(stats.promote.load() == promote0, "a single fault promoted the page");

  // The second fault of the same flip: the page stays writable and is copied by every upload, with no further fault.
  HotFault(tracker, page, 1);
  memory[page_size] = 2;
  Check(stats.refault.load() == refault0 + 1 && stats.promote.load() == promote0 + 1 && stats.pages.load() == 1,
        "second fault of a flip did not promote the page");
  Check(HotUpload(tracker, page, 1) == 1 && IsWritable(memory + page_size) && tracker.IsRegionCpuModified(page, 1),
        "kept page was protected or not uploaded");
  memory[page_size] = 3; // no fault: the page is writable
  Check(HotUpload(tracker, page, 1) == 1 && IsWritable(memory + page_size), "write to a kept page was not uploaded");
  Check(HotUpload(tracker, page, 1) == 1, "kept page was not uploaded when nothing changed (it is dirty: always copied)");
  Check(!IsWritable(memory) && !IsWritable(memory + 2 * page_size), "neighbour pages lost their protection");
  {
    Libs::Graphics::RangeSet bda;
    tracker.AddHotPages(bda);
    Check(bda.Intersects(page, 1), "kept page missing from BDA sync");
  }

  // The flip ends: the next upload copies the page and protects it; a later fault starts a fresh count.
  g_hot_epoch = 101;
  memory[page_size] = 4;
  Check(HotUpload(tracker, page, 1) == 1 && !IsWritable(memory + page_size) && !tracker.IsRegionCpuModified(page, 1) &&
            stats.demote.load() == demote0 + 1 && stats.pages.load() == 0,
        "kept page was not protected again after its flip");
  HotFault(tracker, page, 1);
  memory[page_size] = 5;
  Check(stats.promote.load() == promote0 + 1, "the first fault of a new flip promoted the page");
  Check(HotUpload(tracker, page, 1) == 1 && !IsWritable(memory + page_size), "first fault of the new flip was not uploaded");

  // The GPU takes a kept page for writing: the kept state ends, ownership moves without a conflict.
  g_hot_epoch = 102;
  HotFault(tracker, page, 1);
  (void)HotUpload(tracker, page, 1);
  HotFault(tracker, page, 1);
  Check(stats.pages.load() == 1, "page did not become kept again");
  (void)HotUpload(tracker, page, 1, true);
  Check(stats.pages.load() == 0 && tracker.IsRegionGpuModified(page, 1) && !IsWritable(memory + page_size) &&
            !tracker.IsRegionCpuModified(page, 1),
        "GPU write ownership of a kept page failed");
  tracker.UnmarkRegionAsGpuModified(page, 1);

  // Shadow: protection and uploads as ever; candidates are hashed after the protection; an unseen write is counted once, twice the same hash = ne.
  Libs::Graphics::FlipUnprotect::ModeOverride().store(static_cast<int>(Mode::Shadow));
  Libs::Graphics::HotPages::HasherSource().store(&HotHashNoisy);
  tracker.UntrackMemory(address, page_size * 4);
  (void)HotUpload(tracker, address, page_size * 4);
  const auto cand0 = stats.cand.load();
  const auto ne0 = stats.ne.load();
  const auto unstable0 = stats.ne_unstable.load();
  g_hot_epoch = 200;
  HotFault(tracker, page, 1);
  Check(HotUpload(tracker, page, 1) == 1 && !IsWritable(memory + page_size), "shadow changed the first upload");
  HotFault(tracker, page, 1);
  Check(stats.cand.load() == cand0 + 1 && IsWritable(memory + page_size), "shadow did not find a candidate");
  Check(HotUpload(tracker, page, 1) == 1 && !IsWritable(memory + page_size), "shadow kept the page unprotected");
  const auto hash0 = stats.hash.load();
  Check(HotUpload(tracker, page, 1) == 0 && stats.hash.load() > hash0 && stats.ne.load() == ne0, "shadow counted a clean candidate or never hashed it");
  DWORD old = 0;
  Check(VirtualProtect(memory + page_size, page_size, PAGE_READWRITE, &old) != 0, "VirtualProtect failed");
  memory[page_size + 9] ^= 0xff; // a write the tracker never saw
  (void)HotUpload(tracker, page, 1);
  Check(stats.ne.load() == ne0 + 1 && stats.ne_unstable.load() == unstable0, "shadow missed an unseen write");
  g_noise = 1; // the hash keeps changing: not a stable page
  (void)HotUpload(tracker, page, 1);
  Check(stats.ne_unstable.load() == unstable0 + 1 && stats.ne.load() == ne0 + 1, "an unstable page was counted as ne");
  g_noise = 0;
  g_hot_epoch = 201; // the candidate of the earlier flip is forgotten: no more hashing of it
  const auto hash1 = stats.hash.load();
  (void)HotUpload(tracker, page, 1);
  Check(stats.hash.load() == hash1, "a candidate outlived its flip");

  Libs::Graphics::FlipUnprotect::ModeOverride().store(-1);
  Libs::Graphics::HotPages::HasherSource().store(&HotHash);
  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

// S444: a page kept unprotected for its flip stays writable through the uploads of that flip (no protection change is recorded for it after the promotion);
// a protection put back by another watcher (an image) is the cause of a refault and the ring names it.
uint64_t g_sticky_n = 0;
uint64_t g_sticky_page = 0;
uint32_t g_sticky_flip = 0;
size_t g_sticky_why_n = 0;
const char *g_sticky_why_tag = nullptr;

void OnStickyRefault(uint64_t page, uint32_t flip, uint32_t sticky_flip) {
  using namespace Libs::Graphics::ProtectRing;
  g_sticky_n++;
  g_sticky_page = page;
  g_sticky_flip = sticky_flip;
  (void)flip;
  Entry why[kMaxWhy];
  g_sticky_why_n = Find(page, why, kMaxWhy);
  g_sticky_why_tag = g_sticky_why_n != 0 ? why[0].tag : nullptr;
}

void TestStickyRefault() {
  using namespace Libs::Graphics::ProtectRing;
  using Libs::Graphics::HotPages::Mode;
  Libs::Graphics::HotPages::EpochSource().store(&HotEpoch);
  Libs::Graphics::HotPages::HasherSource().store(&HotHash);
  Libs::Graphics::FlipUnprotect::ModeOverride().store(static_cast<int>(Mode::On));
  Libs::Graphics::FlipUnprotect::StickyRefaultObserver().store(&OnStickyRefault);
  EnabledFlag().store(true);

  TrackerHarness harness;
  auto &tracker = harness.tracker;
  const auto page_size = harness.page_manager.GetPageSize();
  auto *memory = Allocate(harness.page_manager, 4);
  const auto address = reinterpret_cast<uint64_t>(memory);
  const auto page = address + page_size;
  g_hot_epoch = 300;
  Check(HotUpload(tracker, address, page_size * 4) == 1, "first upload did not report the new region");

  // Ring: tag of the caller, newest first, only entries that cover the page.
  tracker.UntrackMemory(address + 3 * page_size, page_size); // no watcher left on page 3: the next watcher changes the host protection
  {
    const TagScope tag("test-tag");
    harness.page_manager.UpdatePageWatchers<true>(address + 3 * page_size, page_size);
  }
  Entry found[kMaxWhy];
  Check(Find(address + 3 * page_size, found, kMaxWhy) >= 1 && found[0].tag != nullptr && std::strcmp(found[0].tag, "test-tag") == 0,
        "ring entry lost the tag of its caller");
  Check(found[0].flip == 300 && found[0].size == page_size, "ring entry lost flip or size");
  harness.page_manager.UpdatePageWatchers<false>(address + 3 * page_size, page_size);
  Check(Find(address + 3 * page_size, found, 1) == 1 && found[0].mode == static_cast<int>(Common::VirtualMemory::Mode::ReadWrite),
        "newest ring entry is not the last change");
  Check(Find(address - page_size, found, kMaxWhy) == 0, "ring matched a page outside every range");

  // fault -> sticky -> upload in the same flip -> write: nothing restores the protection.
  HotFault(tracker, page, 1);
  (void)HotUpload(tracker, page, 1);
  HotFault(tracker, page, 1);
  Check(IsWritable(memory + page_size) && g_sticky_n == 0, "second fault of a flip did not leave the page writable");
  const auto cursor = Cursor().load();
  Check(HotUpload(tracker, page, 1) == 1, "kept page was not uploaded");
  memory[page_size] = 7;
  Check(HotUpload(tracker, page, 1) == 1 && IsWritable(memory + page_size), "kept page was protected by a later upload");
  for (uint64_t seq = cursor + 1; seq <= Cursor().load(); ++seq) {
    const auto &entry = Slots()[seq % kRingSize];
    Check(!(entry.seq == seq && page >= entry.addr && page - entry.addr < entry.size), "a protection change covered the kept page after its promotion");
  }

  // An image watcher re-protects the kept page (TrackImage after an upload of the image): the next write faults and the tracker names the caller.
  {
    const TagScope tag("img-track");
    harness.page_manager.UpdatePageWatchers<true>(page, page_size);
  }
  Check(!IsWritable(memory + page_size), "image watcher did not protect the page");
  HotFault(tracker, page, 1);
  Check(g_sticky_n == 1 && g_sticky_page == page && g_sticky_flip == 300, "refault of a kept page was not reported");
  Check(g_sticky_why_n >= 1 && g_sticky_why_tag != nullptr && std::strcmp(g_sticky_why_tag, "img-track") == 0, "refault did not name the caller that put the protection back");
  const auto text = FormatRefault(page, 300, 300, nullptr, 0);
  Check(text.find("host_page16k=") != std::string::npos && text.find("none-in-ring") != std::string::npos, "refault line malformed");

  harness.page_manager.UpdatePageWatchers<false>(page, page_size);
  EnabledFlag().store(false);
  Libs::Graphics::FlipUnprotect::StickyRefaultObserver().store(nullptr);
  Libs::Graphics::FlipUnprotect::ModeOverride().store(-1);
  tracker.UntrackMemory(address, page_size * 4);
  Release(memory);
}

int main(int argc, char **argv) {
  if (argc == 3 && std::strcmp(argv[1], "--death") == 0) {
    RunDeathCase(argv[2]);
  }
  if (argc == 2 && std::strcmp(argv[1], "--benchmark-clean-upload") == 0) {
    BenchmarkCleanUploads();
    return 0;
  }
  TestGuestRange();
  TestRangeSet();
  TestQueriesDoNotRequireMappedOwnership();
  TestConcurrentRegionPublication();
  TestCpuDirtyUpload();
  TestPageGeneration();
  TestRepeatedUploadSkip();
  TestCleanUploadPreservesOwnership();
  TestRangeInvalidation();
  TestGpuReacquisitionAfterInvalidation();
  TestGpuDirtyBits();
  TestExactDirtyIntervalsSharingTrackerPage();
  TestGpuDownloadProtectionMirrors();
  TestCrossRegionUpload();
  TestConcurrentColdUploads();
  TestUploadDoesNotSerializeDisjointRegion();
  TestDownloadDoesNotSerializeDisjointRegion();
  TestGpuUnmarkUsesRegionMask();
  TestFullRegionGpuUnmarkBatching();
  TestHotPages();
  TestFlipUnprotect();
  TestStickyRefault();
  TestFatalPaths();
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
  TestFaultOnProtectedStack();
#endif
  std::puts("MemoryTrackerTests: all cases passed");
  return 0;
}
