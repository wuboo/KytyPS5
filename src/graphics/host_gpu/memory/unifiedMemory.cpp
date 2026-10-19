#include "common/debugCounters.h"
#include "graphics/host_gpu/memory/unifiedMemory.h"
#include "graphics/host_gpu/memory/umaReadFence.h"

#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameTiming.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <string>

#if !defined(_WIN32)
#include <csignal>
#include <signal.h>
#endif

namespace Libs::Graphics {

namespace {

thread_local Uma3::WaitSource t_wait_source = Uma3::WaitSource::Other;

constexpr uint64_t MiB = 1024 * 1024;

#if !defined(_WIN32)
// SIGTERM / SIGINT: the handler only stores the signal number (async-signal-safe); the [uma]
// thread prints the last counter line and then lets the signal take its original course. A run
// that the harness kills therefore still leaves the totals in the log.
volatile std::sig_atomic_t g_uma_signal = 0;
struct sigaction           g_uma_prev_term {};
struct sigaction           g_uma_prev_int {};
bool                       g_uma_handlers_installed = false;

void UmaSignalHandler(int sig) {
	g_uma_signal = sig;
}

void InstallUmaSignalHandlers() {
	struct sigaction action {};
	action.sa_handler = UmaSignalHandler;
	sigemptyset(&action.sa_mask);
	action.sa_flags = 0;
	g_uma_handlers_installed = sigaction(SIGTERM, &action, &g_uma_prev_term) == 0 &&
	                           sigaction(SIGINT, &action, &g_uma_prev_int) == 0;
}

void RestoreUmaSignalHandlers() {
	if (g_uma_handlers_installed) {
		sigaction(SIGTERM, &g_uma_prev_term, nullptr);
		sigaction(SIGINT, &g_uma_prev_int, nullptr);
		g_uma_handlers_installed = false;
	}
}

// After the last line: hand the signal back to what was installed before us (the default action
// ends the process). Called on the [uma] thread, never from the handler.
void ForwardUmaSignal(int sig) {
	const auto& previous = sig == SIGTERM ? g_uma_prev_term : g_uma_prev_int;
	RestoreUmaSignalHandlers();
	g_uma_signal = 0;
	if (previous.sa_handler == SIG_IGN) {
		return;
	}
	if (previous.sa_handler != SIG_DFL && (previous.sa_flags & SA_SIGINFO) == 0) {
		previous.sa_handler(sig);
		return;
	}
	raise(sig);
}
#endif

uint64_t Checksum(const uint8_t* data, size_t size) {
	uint64_t hash = 1469598103934665603ull;
	for (size_t i = 0; i < size; ++i) {
		hash = (hash ^ data[i]) * 1099511628211ull;
	}
	return hash;
}

} // namespace

bool UnifiedMemory::ShadowCompareRequested() {
	const char* value = std::getenv("KYTY_UMA_SHADOW");
	return LibKernel::Memory::UmaShadow() || (value != nullptr && value[0] == '1');
}

UnifiedMemory::UnifiedMemory(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {
	if (!LibKernel::Memory::UmaEnabled()) {
		return;
	}
	LogStage("import start");
	m_stage.store(static_cast<int>(Stage::Importing), std::memory_order_relaxed);
	if (!InitImport()) {
		Shutdown();
		return;
	}
	m_stage.store(static_cast<int>(Stage::Imported), std::memory_order_relaxed);
	LogStage("import done");
	m_table = std::make_unique<Table>(m_sentinel_gpu);
	m_shadow = ShadowCompareRequested();
	m_dshadow = LibKernel::Memory::UmaShadow();
	if (m_dshadow) {
		if (const char* every = std::getenv("KYTY_UMA_DSHADOW_N"); every != nullptr) {
			m_dshadow_every = std::max<uint64_t>(1, std::strtoull(every, nullptr, 10));
		}
		LOGF("UMA: data shadow-compare of CPU reads enabled, every %" PRIu64 " reads\n", m_dshadow_every);
	}
	if (m_shadow) {
		if (const char* every = std::getenv("KYTY_UMA_SHADOW_N"); every != nullptr) {
			m_shadow_interval = std::max<uint32_t>(1, static_cast<uint32_t>(std::strtoul(every, nullptr, 10)));
		}
		LOGF("UMA: shadow-compare enabled, every %u prepares\n", m_shadow_interval);
	}
	m_active = true;
	m_diag_barriers = UmaBarrierDiag::ModeFromEnv();
	if (m_diag_barriers != UmaBarrierDiag::Mode::Off) {
		LOGF("UMA: KYTY_DIAG_UMA_BARRIERS=%d (%s)\n", static_cast<int>(m_diag_barriers),
		     m_diag_barriers == UmaBarrierDiag::Mode::Insert ? "count and insert a full barrier"
		                                                     : "count only");
	}
	m_stage3 = LibKernel::Memory::UmaStage3();
	m_bda_narrow = FrameTiming::OptEnabled("bda_wait_narrow");
	if (m_stage3) {
		LOGF("UMA: stage 3 (all buffers are arena windows; buffer tracker watches images only)\n");
	}
	// Replays the existing mappings into the event queue; applied by the first Prepare.
	LogStage("replaying existing mappings into the listener");
	LibKernel::Memory::SetGuestMappingListener(this);
	m_stage.store(static_cast<int>(Stage::ListenerSet), std::memory_order_relaxed);
	LOGF("UMA: t=%" PRIu64 " ms listener set, %" PRIu64 " mapping events queued by the replay\n",
	     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - m_t0)
	                               .count()),
	     m_events_queued.load());
	LogDeviceBudget("start");
	StartProgressThread();
}

void UnifiedMemory::LogStage(const char* what) {
	LOGF("UMA: t=%" PRIu64 " ms %s\n",
	     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - m_t0)
	                               .count()),
	     what);
}

void UnifiedMemory::StartProgressThread() {
#if !defined(_WIN32)
	InstallUmaSignalHandlers();
#endif
	m_progress_thread = std::jthread([this](std::stop_token stop) {
		while (!stop.stop_requested()) {
			bool due = false;
			for (int i = 0; i < 20 && !stop.stop_requested(); ++i) {
				std::this_thread::sleep_for(std::chrono::milliseconds(100));
#if !defined(_WIN32)
				if (g_uma_signal != 0) {
					const int sig = g_uma_signal;
					char      tag[32];
					std::snprintf(tag, sizeof(tag), "FINAL signal=%d", sig);
					LogProgress(tag);
					ForwardUmaSignal(sig);
					break;
				}
#endif
				due = i == 19;
			}
			if (due && !stop.stop_requested()) {
				LogProgress();
			}
		}
	});
}

void UnifiedMemory::LogProgress(const char* tag) {
	namespace FT = FrameTiming;
	const auto now_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                              std::chrono::steady_clock::now() - m_t0)
	                                              .count());
	const auto last    = m_last_prepare_ms.load();
	const auto writes  = uint64_t {0};
	const auto flips   = FT::GuestFlips();
	const auto submits = Common::DebugCounters::Get(Common::DebugCounters::Counter::QueueSubmits);
	// First line: the counters that were there before; second part of the same line: the
	// S76 totals (mismatch, chunks, EOP writer, barriers) so one line is enough after a kill.
	const std::string prefix = tag[0] != '\0' ? std::string(tag) + " " : std::string();
	char              shadow_part[128];
	UmaReadFence::FormatShadowCounters(shadow_part, sizeof(shadow_part), m_dshadow, m_dshadow_calls.load(), m_dshadow_eq.load(),
	                                   m_dshadow_ne.load());
	LOGF("[uma] %st=%" PRIu64 " stage=%d events_queued=%" PRIu64 " events=%" PRIu64
	     " pages_written=%" PRIu64 " prepares=%" PRIu64 " since_prepare_ms=%" PRId64
	     " pending_unmaps=%" PRIu64 " uma_bda_waits=%" PRIu64 " bda_wait_ms=%" PRIu64
	     " uma_unmapped_bda=%" PRIu64 " uma_rb_skipped=%" PRIu64 " uma_legacy_hits=%" PRIu64
	     " uma_cp_waits=%" PRIu64 " views=%" PRIu64 " fallbacks=%" PRIu64
	     " shadow_mismatch=%" PRIu64 " shadow_view_mismatch=%" PRIu64
	     " chunks_imported=%u/%u chunks_failed=%u imported_mb=%" PRIu64
	     " chunk_map_imports=%" PRIu64 " chunk_fault_imports=%" PRIu64 " chunk_straddles=%" PRIu64
	     " chunk_overlap_mb=%" PRIu64 " visible_ranges=%" PRIu64 " visible_unmapped=%" PRIu64
	     " eop_writes=%" PRIu64 " eop_latency_avg_us=%" PRIu64 " eop_latency_max_us=%" PRIu64
	     " eop_pending=%" PRIu64 " eop_early=%" PRIu64 " eop_flushes=%" PRIu64
	     " wait_submits=%" PRIu64 " cp_writes_deferred=%" PRIu64 " gds_eop_sync=%" PRIu64
	     " submits=%" PRIu64 " flips=%" PRIu64 " submits_per_flip=%.1f"
	     " diag_barriers=%d diag_raw_missing=%" PRIu64 " diag_waw_missing=%" PRIu64
	     " diag_covered=%" PRIu64 " diag_inserted=%" PRIu64
	     "%s\n",
	     prefix.c_str(), now_ms, m_stage.load(), m_events_queued.load(),
	     m_events_applied.load(), m_pages_written.load(), m_prepares.load(),
	     m_prepares.load() == 0 ? int64_t {-1} : static_cast<int64_t>(now_ms - last),
	     m_pending_unmaps.load(), m_waits.load(), m_wait_ns.load() / 1000000, m_unmapped_bda.load(),
	     m_rb_skipped.load(), m_legacy_hits.load(), m_cp_waits.load(), m_views.load(),
	     m_view_fallbacks.load(), m_shadow_mismatch.load(), m_shadow_view_mismatch.load(),
	     m_chunks_imported.load(), m_layout.Count(), m_chunks_failed.load(),
	     m_imported_bytes.load() / MiB, m_chunk_map_imports.load(), m_chunk_fault_imports.load(),
	     m_chunk_straddles.load(), m_layout.Overlap() / MiB, m_visible_ranges.load(),
	     m_visible_unmapped.load(), writes,
	     writes != 0 ? uint64_t {0} / writes : uint64_t {0},
	     uint64_t {0}, uint64_t {0},
	     uint64_t {0}, uint64_t {0},
	     uint64_t {0}, uint64_t {0},
	     uint64_t {0}, submits, flips,
	     flips != 0 ? static_cast<double>(submits) / static_cast<double>(flips) : 0.0,
	     static_cast<int>(m_diag_barriers), m_diag_raw_missing.load(), m_diag_waw_missing.load(),
	     m_diag_covered.load(), m_diag_inserted.load(), shadow_part);
	// S81: who waits and why (count/ms per source x reason; global = only the pointer-writer tick).
	{
		std::string waits;
		for (uint32_t src = 0; src < WaitSources; ++src) {
			for (uint32_t why = 0; why < WaitReasons; ++why) {
				char part[96];
				const auto slot = src * WaitReasons + why;
				std::snprintf(part, sizeof(part), " uma_wait_%s_%s_ms=%" PRIu64 "/%" PRIu64,
				              Uma3::WaitSourceName(static_cast<Uma3::WaitSource>(src)),
				              Uma3::WaitReasonName(static_cast<Uma3::WaitReason>(why)),
				              m_wait_by_ns[slot].load() / 1000000, m_wait_by_n[slot].load());
				waits += part;
			}
		}
		LOGF("%s uma waits (ms/count):%s bda_narrow=%d narrow_skipped=%" PRIu64 "\n", prefix.c_str(),
		     waits.c_str(), m_bda_narrow ? 1 : 0, m_narrow_skipped.load());
	}
}

void UnifiedMemory::LogDeviceBudget(const char* when) {
	if (!m_graphics.memory_budget_ext_enabled) {
		LOGF("[uma] device budget (%s): VK_EXT_memory_budget is not enabled\n", when);
		return;
	}
	// MoltenVK reports heap usage = MTLDevice.currentAllocatedSize and heap budget =
	// MTLDevice.recommendedMaxWorkingSetSize (vkGetMTLDeviceMVK is not exported by the dylib).
	const auto properties =
	    m_graphics.physical_device.getMemoryProperties2<vk::PhysicalDeviceMemoryProperties2,
	                                                    vk::PhysicalDeviceMemoryBudgetPropertiesEXT>();
	const auto& memory = properties.get<vk::PhysicalDeviceMemoryProperties2>().memoryProperties;
	const auto& budget = properties.get<vk::PhysicalDeviceMemoryBudgetPropertiesEXT>();
	for (uint32_t heap = 0; heap < memory.memoryHeapCount; ++heap) {
		LOGF("[uma] device budget (%s) heap %u: currentAllocatedSize=%" PRIu64
		     " MB recommendedMaxWorkingSetSize=%" PRIu64 " MB heap_size=%" PRIu64 " MB\n",
		     when, heap, static_cast<uint64_t>(budget.heapUsage[heap]) / MiB,
		     static_cast<uint64_t>(budget.heapBudget[heap]) / MiB,
		     static_cast<uint64_t>(memory.memoryHeaps[heap].size) / MiB);
	}
}

UnifiedMemory::~UnifiedMemory() {
	Shutdown();
}

bool UnifiedMemory::InitImport() {
	namespace Memory = LibKernel::Memory;
	const auto fail  = [](const char* what, vk::Result result = vk::Result::eSuccess) {
		LOGF("UMA: disabled, %s%s%s\n", what, result == vk::Result::eSuccess ? "" : ": ",
		     result == vk::Result::eSuccess ? "" : vk::to_string(result).c_str());
		return false;
	};
	if (!m_graphics.external_memory_host_enabled) {
		return fail("VK_EXT_external_memory_host is not enabled");
	}
	m_arena         = Memory::GetGuestArena(Memory::GUEST_ARENA_DIRECT);
	const auto base = m_arena.base;
	if (base == 0 || m_arena.size == 0) {
		return fail("no guest backing");
	}
	const auto properties =
	    m_graphics.physical_device
	        .getProperties2<vk::PhysicalDeviceProperties2,
	                        vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>();
	const auto reported = properties.get<vk::PhysicalDeviceExternalMemoryHostPropertiesEXT>()
	                          .minImportedHostPointerAlignment;
	// 16 KB is enforced whatever the driver reports (Rosetta: 4 KB; an unaligned import is accepted
	// silently and then misbehaves).
	m_import_alignment = UmaChunks::ImportAlignment(reported);
	if (base % m_import_alignment != 0) {
		return fail("backing is not aligned to the import alignment");
	}
	// A tail that is not a multiple of the alignment cannot be imported: it takes the copy path.
	const auto usable = m_arena.size & ~(m_import_alignment - 1);
	if (usable == 0) {
		return fail("backing is smaller than the import alignment");
	}
	uint64_t chunk_mb = 0;
	if (const char* value = std::getenv("KYTY_UMA_CHUNK_MB"); value != nullptr) {
		chunk_mb = std::strtoull(value, nullptr, 10);
	}
	const auto  chunk_bytes = UmaChunks::ChunkBytes(chunk_mb);
	uint64_t    overlap_mb  = 0;
	bool        overlap_set = false;
	if (const char* value = std::getenv("KYTY_UMA_OVERLAP_MB"); value != nullptr) {
		overlap_mb  = std::strtoull(value, nullptr, 10);
		overlap_set = true;
	}
	m_layout = UmaChunks::Layout(usable, chunk_bytes,
	                             UmaChunks::OverlapBytes(overlap_mb, overlap_set, chunk_bytes));
	m_slots  = std::make_unique<ChunkSlot[]>(m_layout.Count());
	{
		std::lock_guard lock(m_chunk_mutex);
		m_chunk_objects.resize(m_layout.Count());
	}

	LogStage("import: sentinel page");
	m_sentinel = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0,
	                                      AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress,
	                                      PageSize);
	std::memset(m_sentinel->Mapped().data(), 0, static_cast<size_t>(m_sentinel->Size()));
	m_sentinel->Flush(0, m_sentinel->Size());
	m_sentinel_gpu = m_sentinel->BufferDeviceAddress();
	SetVulkanObjectNameF(m_graphics.device, m_sentinel->Handle(), "Kyty.UmaSentinelPage");
	LOGF("UMA: lazy chunked import host=0x%016" PRIx64 " size=%" PRIu64 " MB chunk=%" PRIu64
	     " MB overlap=%" PRIu64 " MB chunks=%u import_alignment=%" PRIu64
	     " (driver reports %" PRIu64 ") nothing imported yet\n",
	     base, usable / MiB, m_layout.ChunkSize() / MiB, m_layout.Overlap() / MiB, m_layout.Count(),
	     m_import_alignment,
	     static_cast<uint64_t>(reported));
	return true;
}

bool UnifiedMemory::EnsureChunk(uint32_t index) {
	if (!m_active || index >= m_layout.Count()) {
		return false;
	}
	auto& slot = m_slots[index];
	switch (UmaChunks::Decide(static_cast<UmaChunks::State>(slot.state.load(std::memory_order_acquire)))) {
		case UmaChunks::Action::Use: return true;
		case UmaChunks::Action::Legacy: return false;
		case UmaChunks::Action::Import: break;
	}
	std::lock_guard lock(m_chunk_mutex);
	if (static_cast<UmaChunks::State>(slot.state.load(std::memory_order_acquire)) !=
	    UmaChunks::State::NotImported) {
		return slot.state.load(std::memory_order_acquire) ==
		       static_cast<uint8_t>(UmaChunks::State::Imported);
	}
	const auto started = std::chrono::steady_clock::now();
	const auto begin   = m_layout.Begin(index);
	const auto size    = m_layout.End(index) - begin;
	const auto host    = m_arena.base + begin;
	auto&      objects = m_chunk_objects[index];
	const auto fail    = [&](const char* what, vk::Result result = vk::Result::eSuccess) {
		LOGF("UMA: chunk %u (arena +%" PRIu64 " MB, %" PRIu64 " MB) import failed, %s%s%s; its ranges "
		     "take the copy path\n",
		     index, begin / MiB, size / MiB, what, result == vk::Result::eSuccess ? "" : ": ",
		     result == vk::Result::eSuccess ? "" : vk::to_string(result).c_str());
		if (objects.buffer) {
			m_graphics.device.destroyBuffer(objects.buffer);
			objects.buffer = nullptr;
		}
		slot.state.store(static_cast<uint8_t>(UmaChunks::State::Failed), std::memory_order_release);
		m_chunks_failed.fetch_add(1, std::memory_order_relaxed);
		return false;
	};
	if (!UmaChunks::IsImportable(host, size, m_import_alignment)) {
		return fail("pointer or size is not aligned to the import alignment");
	}
	vk::MemoryHostPointerPropertiesEXT host_properties {};
	auto result = m_graphics.device.getMemoryHostPointerPropertiesEXT(
	    vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT, reinterpret_cast<void*>(host),
	    &host_properties);
	if (result != vk::Result::eSuccess) {
		return fail("getMemoryHostPointerPropertiesEXT failed", result);
	}
	vk::BufferCreateInfo buffer_info {};
	buffer_info.size  = size;
	buffer_info.usage = AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	result            = m_graphics.device.createBuffer(&buffer_info, nullptr, &objects.buffer);
	if (result != vk::Result::eSuccess) {
		return fail("createBuffer failed", result);
	}
	const auto requirements = m_graphics.device.getBufferMemoryRequirements(objects.buffer);
	const auto type_bits    = host_properties.memoryTypeBits & requirements.memoryTypeBits;
	const auto& memory      = m_graphics.GetPhysicalDeviceMemoryProperties();
	uint32_t    type        = UINT32_MAX;
	for (uint32_t i = 0; i < memory.memoryTypeCount && type == UINT32_MAX; ++i) {
		constexpr auto wanted = vk::MemoryPropertyFlagBits::eHostVisible |
		                        vk::MemoryPropertyFlagBits::eHostCoherent;
		if ((type_bits & (1u << i)) != 0 &&
		    (memory.memoryTypes[i].propertyFlags & wanted) == wanted) {
			type = i;
		}
	}
	if (type == UINT32_MAX) {
		return fail("no host-visible memory type accepts the import");
	}
	vk::ImportMemoryHostPointerInfoEXT import_info {};
	import_info.handleType   = vk::ExternalMemoryHandleTypeFlagBits::eHostAllocationEXT;
	import_info.pHostPointer = reinterpret_cast<void*>(host);
	vk::MemoryAllocateFlagsInfo flags_info {};
	flags_info.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	flags_info.pNext = &import_info;
	vk::MemoryAllocateInfo allocate_info {};
	allocate_info.pNext           = &flags_info;
	allocate_info.allocationSize  = size;
	allocate_info.memoryTypeIndex = type;
	result = m_graphics.device.allocateMemory(&allocate_info, nullptr, &objects.memory);
	if (result == vk::Result::eSuccess) {
		result = m_graphics.device.bindBufferMemory(objects.buffer, objects.memory, 0);
	}
	if (result != vk::Result::eSuccess) {
		if (objects.memory) {
			m_graphics.device.freeMemory(objects.memory);
			objects.memory = nullptr;
		}
		return fail("importing the chunk failed", result);
	}
	vk::BufferDeviceAddressInfo address_info {};
	address_info.buffer = objects.buffer;
	const auto gpu      = m_graphics.device.getBufferAddress(address_info);
	if (gpu == 0) {
		m_graphics.device.freeMemory(objects.memory);
		objects.memory = nullptr;
		return fail("imported chunk has no device address");
	}
	SetVulkanObjectNameF(m_graphics.device, objects.buffer, "Kyty.UmaGuestBacking.Chunk");
	objects.wrapper = std::make_unique<Buffer>(m_graphics, m_scheduler, objects.buffer, gpu, size);
	slot.gpu.store(gpu, std::memory_order_release);
	slot.state.store(static_cast<uint8_t>(UmaChunks::State::Imported), std::memory_order_release);
	const auto imported = m_chunks_imported.fetch_add(1, std::memory_order_relaxed) + 1;
	m_imported_bytes.fetch_add(size, std::memory_order_relaxed);
	m_chunk_import_ns.fetch_add(
	    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                              std::chrono::steady_clock::now() - started)
	                              .count()),
	    std::memory_order_relaxed);
	LOGF("UMA: t=%" PRIu64 " ms chunk %u imported (arena +%" PRIu64 " MB, %" PRIu64
	     " MB) gpu=0x%016" PRIx64 " [chunks_imported=%u/%u imported_mb=%" PRIu64 "]\n",
	     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                               std::chrono::steady_clock::now() - m_t0)
	                               .count()),
	     index, begin / MiB, size / MiB, static_cast<uint64_t>(gpu), imported, m_layout.Count(),
	     m_imported_bytes.load() / MiB);
	if (imported == 1 || imported % 8 == 0) {
		LogDeviceBudget("after chunk import");
	}
	QueueRepublish(index);
	return true;
}

void UnifiedMemory::QueueRepublish(uint32_t index) {
	std::lock_guard lock(m_event_mutex);
	m_republish.push_back(index);
	m_events_pending.store(true, std::memory_order_release);
}

bool UnifiedMemory::ResolveChunk(uint64_t offset, uint64_t size, UmaChunks::Place* place) const {
	const auto own = m_layout.Find(offset, size);
	if (!own.ok) {
		return false;
	}
	const auto imported = [this](uint32_t index) {
		return static_cast<UmaChunks::State>(m_slots[index].state.load(std::memory_order_acquire)) ==
		       UmaChunks::State::Imported;
	};
	*place = own;
	if (imported(own.index)) {
		return true;
	}
	const auto previous = m_layout.FindPrevious(offset, size);
	if (previous.ok && imported(previous.index)) {
		*place = previous;
	}
	return true;
}

void UnifiedMemory::ImportVisible(const std::vector<VisibleRange>& visible) {
	for (const auto& range: visible) {
		m_visible_ranges.fetch_add(1, std::memory_order_relaxed);
		bool mapped_all = false;
		uint64_t covered = 0;
		m_sync.Map().ForEachRunIn(range.vaddr, range.size, [&](uint64_t, uint64_t size, uint64_t offset) {
			covered += size;
			const auto first = m_layout.Cell(offset);
			const auto last  = m_layout.LastCell(offset, size);
			for (auto index = first; index <= last; ++index) {
				const bool fresh = static_cast<UmaChunks::State>(m_slots[index].state.load(
				                       std::memory_order_acquire)) == UmaChunks::State::NotImported;
				if (EnsureChunk(index) && fresh) {
					m_chunk_map_imports.fetch_add(1, std::memory_order_relaxed);
				}
			}
		});
		mapped_all = covered == range.size;
		if (!mapped_all) {
			m_visible_unmapped.fetch_add(1, std::memory_order_relaxed);
		}
	}
}

vk::DeviceAddress UnifiedMemory::DeviceAddressOf(uint64_t offset, uint64_t size) const {
	if (!m_slots) {
		return 0;
	}
	return UmaChunks::DeviceAddress(m_layout, offset, size, [this](uint32_t index) -> uint64_t {
		return m_slots[index].gpu.load(std::memory_order_acquire);
	});
}

void UnifiedMemory::Shutdown() {
	if (m_progress_thread.joinable()) {
		m_progress_thread.request_stop();
		m_progress_thread.join();
	}
#if !defined(_WIN32)
	RestoreUmaSignalHandlers();
#endif
	if (m_active) {
		LibKernel::Memory::SetGuestMappingListener(nullptr);
		LogProgress("FINAL shutdown");
		// Same counter names as before the refactor (uma_global_waits mirrors bda_waits).
		LOGF("UMA: events=%" PRIu64 " pages_written=%" PRIu64 " bda_waits=%" PRIu64
		     " bda_wait_ms=%" PRIu64 " uma_unmapped_bda=%" PRIu64 " uma_sentinel_pages=%" PRIu64
		     " uma_views=%" PRIu64 " uma_view_fallbacks=%" PRIu64 " uma_deferred_unmaps=%" PRIu64
		     " uma_global_waits=%" PRIu64 "\n",
		     m_events_applied.load(), m_pages_written.load(), m_waits.load(), m_wait_ns.load() / 1000000,
		     m_unmapped_bda.load(), m_sentinel_pages.load(), m_views.load(), m_view_fallbacks.load(),
		     m_sync.DeferredTotal(), m_waits.load());
		LOGF("UMA: chunks_imported=%u/%u chunks_failed=%u imported_mb=%" PRIu64 " chunk_mb=%" PRIu64
		     " chunk_fault_imports=%" PRIu64 " chunk_straddles=%" PRIu64 " chunk_import_ms=%" PRIu64
		     "\n",
		     m_chunks_imported.load(), m_layout.Count(), m_chunks_failed.load(),
		     m_imported_bytes.load() / MiB, m_layout.ChunkSize() / MiB, m_chunk_fault_imports.load(),
		     m_chunk_straddles.load(), m_chunk_import_ns.load() / 1000000);
		if (m_diag_barriers != UmaBarrierDiag::Mode::Off) {
			LOGF("UMA DIAG barriers: raw_missing=%" PRIu64 " waw_missing=%" PRIu64 " covered=%" PRIu64
			     " inserted=%" PRIu64 " tracker_dropped=%" PRIu64 "\n",
			     m_diag_raw_missing.load(), m_diag_waw_missing.load(), m_diag_covered.load(),
			     m_diag_inserted.load(), m_diag_tracker.Dropped());
		}
		LogDeviceBudget("exit");
		if (m_stage3) {
			LOGF("UMA3: uma_views_written=%" PRIu64 " uma_legacy_hits=%" PRIu64
			     " uma_legacy_texel=%" PRIu64 " uma_legacy_image=%" PRIu64
			     " uma_legacy_uncovered=%" PRIu64 " uma_rb_skipped=%" PRIu64
			     " uma_cp_waits=%" PRIu64 " uma_watchers_dropped=%" PRIu64 "\n",
			     m_views_written, m_legacy_hits.load(), m_legacy_texel.load(), m_legacy_image.load(),
			     m_legacy_uncovered.load(), m_rb_skipped.load(), m_cp_waits.load(),
			     m_watchers_dropped);
		}
		if (m_dshadow) {
			LOGF("UMA: uma_dshadow_calls=%" PRIu64 " uma_dshadow_eq=%" PRIu64 " uma_dshadow_ne=%" PRIu64 "\n",
			     m_dshadow_calls.load(), m_dshadow_eq.load(), m_dshadow_ne.load());
		}
		if (m_shadow) {
			LOGF("UMA: shadow-compare runs=%" PRIu64 " uma_shadow_mismatch=%" PRIu64
			     " (view=%" PRIu64 ") skipped=%" PRIu64
			     " table_pages=%zu table_leaves=%zu table_kb=%zu\n",
			     m_shadow_runs, m_shadow_mismatch.load(), m_shadow_view_mismatch.load(), m_shadow_skipped,
			     m_table->MappedPages(), m_table->LeafCount(), m_table->MemoryBytes() / 1024);
		}
	}
	m_active = false;
	bool any = false;
	{
		std::lock_guard lock(m_chunk_mutex);
		for (const auto& objects: m_chunk_objects) {
			any = any || objects.buffer || objects.memory;
		}
	}
	if (any) {
		(void)m_graphics.device.waitIdle();
	}
	m_sentinel.reset();
	m_sentinel_gpu = 0;
	{
		std::lock_guard lock(m_chunk_mutex);
		for (auto& objects: m_chunk_objects) {
			objects.wrapper.reset();
			if (objects.buffer) {
				m_graphics.device.destroyBuffer(objects.buffer);
				objects.buffer = nullptr;
			}
			if (objects.memory) {
				m_graphics.device.freeMemory(objects.memory);
				objects.memory = nullptr;
			}
		}
	}
}

void UnifiedMemory::OnMapped(uint64_t vaddr, uint64_t size, LibKernel::Memory::GuestArenaId arena,
                             uint64_t offset) {
	if (arena != m_arena.id) {
		return;
	}
	std::lock_guard lock(m_event_mutex);
	m_events.push_back({vaddr, size, offset, true});
	m_events_queued.fetch_add(1, std::memory_order_relaxed);
	m_events_pending.store(true, std::memory_order_release);
}

void UnifiedMemory::OnGpuVisible(uint64_t vaddr, uint64_t size) {
	if (!m_active || size == 0) {
		return;
	}
	std::lock_guard lock(m_event_mutex);
	m_visible.push_back({vaddr, size});
	m_events_pending.store(true, std::memory_order_release);
}

void UnifiedMemory::OnUnmapped(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_event_mutex);
	m_events.push_back({vaddr, size, 0, false});
	m_events_queued.fetch_add(1, std::memory_order_relaxed);
	m_events_pending.store(true, std::memory_order_release);
}

void UnifiedMemory::Mirror(uint64_t first, uint64_t last) {
	for (auto page = first; page < last; page += Table::PageSize) {
		uint64_t offset = 0;
		vk::DeviceAddress address = 0;
		// A page whose chunk is not imported yet is cleared: it takes the shadow / fault path until
		// the chunk is imported and republished.
		if (m_sync.Map().RangeOffset(page, Table::PageSize, &offset) &&
		    (address = DeviceAddressOf(offset, Table::PageSize)) != 0) {
			m_table->Set(page, address);
		} else {
			m_table->Clear(page);
		}
	}
}

void UnifiedMemory::ApplyEvents(const PublishFn& publish) {
	std::vector<Sync::Event>   events;
	std::vector<VisibleRange>  visible;
	std::vector<uint32_t>      republish;
	{
		std::lock_guard lock(m_event_mutex);
		events.swap(m_events);
		visible.swap(m_visible);
		m_events_pending.store(false, std::memory_order_relaxed);
	}
	const auto emit = [&](uint64_t first, uint64_t last) {
		Mirror(first, last);
		publish(first, last);
	};
	// Mappings are published at once; an unmap waits for the submits recorded so far (the tick
	// being recorded included) before its entries change.
	m_sync.Apply(events, m_scheduler.CurrentTick(), emit);
	m_sync.Retire([this](uint64_t tick) { return m_scheduler.GetMasterSemaphore().IsFree(tick); },
	              emit);
	// GPU-visible ranges import their chunks here, before the submit that may use them; the
	// imports queue their own republication, taken right after.
	ImportVisible(visible);
	{
		std::lock_guard lock(m_event_mutex);
		republish.swap(m_republish);
	}
	// A chunk imported since the last call: its mapped pages get their entries now. Pages with a
	// deferred unmap are not in the map, so their old entries are left to Retire.
	for (const auto index: republish) {
		const auto begin = m_layout.Begin(index);
		const auto end   = m_layout.End(index);
		m_sync.Map().ForEachRun([&](uint64_t vaddr, uint64_t size, uint64_t backing_offset) {
			const auto lo = std::max(backing_offset, begin);
			const auto hi = std::min(backing_offset + size, end);
			if (lo >= hi) {
				return;
			}
			const auto first = (vaddr + (lo - backing_offset) + PageSize - 1) & ~(PageSize - 1);
			const auto last  = (vaddr + (hi - backing_offset)) & ~(PageSize - 1);
			if (first < last) {
				emit(first, last);
			}
		});
	}
	m_pending_unmaps.store(m_sync.PendingCount(), std::memory_order_relaxed);
}

void UnifiedMemory::SyncMappings(const PublishFn& publish) {
	if (m_active && m_events_pending.load(std::memory_order_acquire)) {
		ApplyEvents(publish);
	}
}

void UnifiedMemory::Prepare(const PublishFn& publish, bool writes_through_pointers) {
	if (!m_active) {
		return;
	}
	if (m_prepares.fetch_add(1, std::memory_order_relaxed) == 0) {
		m_stage.store(static_cast<int>(Stage::FirstPrepare), std::memory_order_relaxed);
		LOGF("UMA: t=%" PRIu64 " ms first Prepare (GPU thread), %" PRIu64 " mapping events queued\n",
		     static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		                               std::chrono::steady_clock::now() - m_t0)
		                               .count()),
		     m_events_queued.load());
	}
	m_last_prepare_ms.store(static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
	                                                  std::chrono::steady_clock::now() - m_t0)
	                                                  .count()),
	                        std::memory_order_relaxed);
	ApplyEvents(publish);
	// Conservative: a submit with direct access may write guest memory at addresses nobody knows
	// before it runs. Stage 3 narrows that to the submits that hold a shader storing through
	// pointers; windows and fills record their exact ranges instead.
	m_unknown_writer.Submit(m_scheduler.CurrentTick(), writes_through_pointers, m_stage3);
	m_bda_tick.store(m_unknown_writer.Value(), std::memory_order_release);
	if (m_shadow && ++m_shadow_counter >= m_shadow_interval) {
		m_shadow_counter = 0;
		ShadowCompare();
	}
}

vk::DeviceAddress UnifiedMemory::PageEntry(uint64_t page_vaddr) const {
	uint64_t offset = 0;
	if (m_active && m_sync.Map().PageOffset(page_vaddr, &offset)) {
		return DeviceAddressOf(offset, PageSize); // 0: chunk not imported, or the page straddles two
	}
	return 0;
}

bool UnifiedMemory::ImportForPage(uint64_t page_vaddr, const PublishFn& publish) {
	if (!m_active) {
		return false;
	}
	SyncMappings(publish); // the mirror must know every mapping made so far
	uint64_t offset = 0;
	if (!m_sync.Map().PageOffset(page_vaddr, &offset)) {
		return false;
	}
	UmaChunks::Place place;
	if (!ResolveChunk(offset, PageSize, &place)) {
		return false; // straddles two chunks: shadow buffer
	}
	const bool fresh = static_cast<UmaChunks::State>(m_slots[place.index].state.load(
	                       std::memory_order_acquire)) == UmaChunks::State::NotImported;
	if (!EnsureChunk(place.index)) {
		return false;
	}
	if (fresh) {
		m_chunk_fault_imports.fetch_add(1, std::memory_order_relaxed);
	}
	SyncMappings(publish); // republish the chunk now: the faulting shader must not fault again
	return PageEntry(page_vaddr) != 0;
}

vk::DeviceAddress UnifiedMemory::UnmappedEntry(uint64_t page_vaddr) {
	if (!m_active || m_sentinel_gpu == 0 ||
	    LibKernel::Memory::IsGuestRangeMapped(page_vaddr, PageSize)) {
		return 0;
	}
	return m_sentinel_gpu;
}

uint64_t UnifiedMemory::CountUnmappedBda() {
	return m_unmapped_bda.fetch_add(1, std::memory_order_relaxed) + 1;
}

void UnifiedMemory::CountPublished(uint64_t pages, uint64_t sentinel_pages) {
	m_sentinel_pages += sentinel_pages;
	m_pages_written += pages;
	++m_events_applied;
}

bool UnifiedMemory::View(uint64_t vaddr, uint64_t size, bool is_written, Window* window) {
	uint64_t offset = 0;
	if (!m_active || !m_sync.Map().RangeOffset(vaddr, size, &offset)) {
		return false;
	}
	// A window cannot span two imports: a range over a chunk boundary takes the copy path.
	UmaChunks::Place place;
	if (!ResolveChunk(offset, size, &place)) {
		m_chunk_straddles.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	if (!EnsureChunk(place.index)) {
		return false;
	}
	offset = offset - place.begin;
	++m_views;
	if (m_diag_barriers != UmaBarrierDiag::Mode::Off) {
		DiagBarrierCheck(vaddr, size, is_written);
	}
	if (is_written) {
		++m_views_written;
		m_written.Mark(vaddr, size);
		std::lock_guard lock(m_fence_mutex);
		m_fences.Record(vaddr, size, 0, m_scheduler.CurrentTick());
	}
	const auto& objects = m_chunk_objects[place.index];
	window->buffer      = objects.buffer;
	window->wrapper     = objects.wrapper.get();
	window->offset      = offset;
	if (m_shadow && m_stage3) {
		m_recent_views[m_recent_view_next++ % m_recent_views.size()] = {vaddr, size,
		                                                                  place.begin + offset};
		if (++m_shadow_view_counter >= 4096) {
			m_shadow_view_counter = 0;
			ShadowCompare();
		}
	}
	return true;
}

void UnifiedMemory::DiagBarrierCheck(uint64_t vaddr, uint64_t size, bool is_written) {
	const auto tick  = m_scheduler.CurrentTick();
	const auto epoch = UmaBarrierDiag::Epoch();
	switch (m_diag_tracker.Check(tick, vaddr, size, epoch)) {
		case UmaBarrierDiag::Verdict::None: break;
		case UmaBarrierDiag::Verdict::Covered: m_diag_covered.fetch_add(1, std::memory_order_relaxed); break;
		case UmaBarrierDiag::Verdict::Missing: {
			const auto n = (is_written ? m_diag_waw_missing : m_diag_raw_missing)
			                   .fetch_add(1, std::memory_order_relaxed) + 1;
			if ((n & (n - 1)) == 0) {
				LOGF("UMA DIAG: %s of 0x%016" PRIx64 "+0x%" PRIx64
				     " follows a write in the same submit (tick %" PRIu64
				     ") with no memory barrier recorded since (count=%" PRIu64 ")\n",
				     is_written ? "write" : "read", vaddr, size, tick, n);
			}
			if (m_diag_barriers == UmaBarrierDiag::Mode::Insert && m_scheduler.Active() &&
			    !m_scheduler.Current().IsInvalid()) {
				// Same recipe as the GDS branch of CommitBindings: leave the render pass, then a
				// full memory barrier.
				auto& command = m_scheduler.Current();
				command.EndRendering();
				vk::MemoryBarrier barrier {};
				barrier.srcAccessMask = vk::AccessFlagBits::eMemoryWrite;
				barrier.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
				command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
				                                 vk::PipelineStageFlagBits::eAllCommands, {}, 1,
				                                 &barrier, 0, nullptr, 0, nullptr);
				UmaBarrierDiag::NoteBarrier();
				m_diag_inserted.fetch_add(1, std::memory_order_relaxed);
			}
			break;
		}
	}
	if (is_written) {
		m_diag_tracker.NoteWrite(tick, vaddr, size, UmaBarrierDiag::Epoch());
	}
}

bool UnifiedMemory::Covers(uint64_t vaddr, uint64_t size) const {
	uint64_t offset = 0;
	if (!m_active || size == 0 || !m_sync.Map().RangeOffset(vaddr, size, &offset)) {
		return false;
	}
	UmaChunks::Place place;
	return ResolveChunk(offset, size, &place) &&
	       static_cast<UmaChunks::State>(m_slots[place.index].state.load(std::memory_order_acquire)) !=
	           UmaChunks::State::Failed;
}

bool UnifiedMemory::RangeImported(uint64_t vaddr, uint64_t size) const {
	uint64_t offset = 0;
	if (!m_active || size == 0 || !m_sync.Map().RangeOffset(vaddr, size, &offset)) {
		return false;
	}
	UmaChunks::Place place;
	return ResolveChunk(offset, size, &place) &&
	       static_cast<UmaChunks::State>(m_slots[place.index].state.load(std::memory_order_acquire)) ==
	           UmaChunks::State::Imported;
}

void UnifiedMemory::RecordWriter(uint64_t vaddr, uint64_t size) {
	if (m_active) {
		RecordWriterAt(vaddr, size, m_scheduler.CurrentTick());
		if (m_diag_barriers != UmaBarrierDiag::Mode::Off && GuestGpu::IsGpuThread()) {
			m_diag_tracker.NoteWrite(m_scheduler.CurrentTick(), vaddr, size, UmaBarrierDiag::Epoch());
		}
	}
}

void UnifiedMemory::RecordWriterAt(uint64_t vaddr, uint64_t size, uint64_t tick) {
	if (!m_active) {
		return;
	}
	m_written.Mark(vaddr, size);
	std::lock_guard lock(m_fence_mutex);
	m_fences.Record(vaddr, size, 0, tick);
}

void UnifiedMemory::CountLegacy(Legacy reason, uint64_t vaddr, uint64_t size) {
	std::atomic<uint64_t>* counter = &m_legacy_hits;
	switch (reason) {
		case Legacy::Texel: counter = &m_legacy_texel; break;
		case Legacy::ImageSync: counter = &m_legacy_image; break;
		case Legacy::Uncovered: counter = &m_legacy_uncovered; break;
		case Legacy::Other: break;
	}
	const auto n = counter->fetch_add(1, std::memory_order_relaxed) + 1;
	if (reason == Legacy::Other && (n & (n - 1)) == 0) {
		LOGF("UMA3: a range the arena can back took the copy path (uma_legacy_hits=%" PRIu64
		     ") addr=0x%016" PRIx64 " size=0x%" PRIx64 "\n",
		     n, vaddr, size);
	}
}

bool UnifiedMemory::TimelineRetired(uint32_t timeline, uint64_t tick) {
	// Only the graphics timeline exists today; async-compute timelines register here.
	(void)timeline;
	return m_scheduler.GetMasterSemaphore().IsFree(tick);
}

void UnifiedMemory::TimelineWait(uint32_t timeline, uint64_t tick) {
	(void)timeline;
	// The tick may still be the one being recorded. The GPU thread submits it itself (waiting
	// for a command buffer it has not sent would never end); any other thread waits for the GPU
	// thread to submit it.
	if (GuestGpu::IsGpuThread()) {
		if (tick == m_scheduler.CurrentTick()) {
			m_scheduler.Wait(tick);
		} else {
			m_scheduler.GetMasterSemaphore().Wait(tick);
		}
	} else if (tick >= m_scheduler.CurrentTick()) {
		// Another thread cannot submit the tick, and an idle GPU thread would never do it: ask
		// the GPU thread to submit and wait (the way a readback fault always did).
		auto& gpu = m_scheduler.Context().GetGpu();
		if (!gpu.IsStopping()) {
			gpu.SendCommandSync([this, tick] {
				if (tick == m_scheduler.CurrentTick()) {
					m_scheduler.Wait(tick);
				} else if (tick < m_scheduler.CurrentTick()) {
					m_scheduler.GetMasterSemaphore().Wait(tick);
				}
			});
		}
	} else {
		m_scheduler.GetMasterSemaphore().Wait(tick);
	}
	m_scheduler.WaitPriorityOperations(tick);
}

uint32_t UnifiedMemory::PendingTimelines(uint64_t vaddr, uint64_t size, WriteFenceTable::Ticks* ticks_out,
                                          Uma3::WaitReason* reason, bool* skipped_by_narrow) {
	WriteFenceTable::Ticks ticks {};
	{
		std::lock_guard lock(m_fence_mutex);
		m_fences.Prune([this](uint32_t timeline, uint64_t t) { return TimelineRetired(timeline, t); });
		ticks = size == 0 ? m_fences.MaxTicks(0, UINT64_MAX - 1) : m_fences.MaxTicks(vaddr, size);
	}
	const auto retired = [this](uint32_t timeline, uint64_t tick) {
		return TimelineRetired(timeline, tick);
	};
	const auto global  = m_bda_tick.load(std::memory_order_acquire);
	const auto unknown = Uma3::EffectiveUnknownTick(global, m_bda_narrow, m_written, vaddr, size);
	const auto mask    = Uma3::PendingTimelines(ticks, unknown, retired, ticks_out);
	if (reason != nullptr) {
		*reason = Uma3::ClassifyWait(ticks, retired);
	}
	if (skipped_by_narrow != nullptr) {
		*skipped_by_narrow = mask == 0 && unknown != global &&
		                     Uma3::PendingTimelines(ticks, global, retired) != 0;
	}
	return mask;
}

bool UnifiedMemory::HasPendingWriters(uint64_t vaddr, uint64_t size) {
	if (!m_active) {
		return false;
	}
	WriteFenceTable::Ticks ticks {};
	return PendingTimelines(vaddr, size, &ticks) != 0;
}

void UnifiedMemory::WaitWriters(uint64_t vaddr, uint64_t size) {
	if (!m_active || CommandScheduler::InDeferredOperation()) {
		return;
	}
	WriteFenceTable::Ticks ticks {};
	Uma3::WaitReason       reason  = Uma3::WaitReason::Range;
	bool                   skipped = false;
	const auto             pending = PendingTimelines(vaddr, size, &ticks, &reason, &skipped);
	if (pending == 0) {
		if (skipped) {
			m_narrow_skipped.fetch_add(1, std::memory_order_relaxed);
		}
		return;
	}
	const auto start = std::chrono::steady_clock::now();
	for (uint32_t timeline = 0; timeline < WriteFenceTable::MaxTimelines; ++timeline) {
		if ((pending & (1u << timeline)) != 0) {
			TimelineWait(timeline, ticks[timeline]);
		}
	}
	const auto elapsed_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                                  std::chrono::steady_clock::now() - start)
	                                                  .count());
	m_waits.fetch_add(1, std::memory_order_relaxed);
	m_wait_ns.fetch_add(elapsed_ns, std::memory_order_relaxed);
	const auto slot = static_cast<uint32_t>(t_wait_source) * WaitReasons + static_cast<uint32_t>(reason);
	m_wait_by_n[slot].fetch_add(1, std::memory_order_relaxed);
	m_wait_by_ns[slot].fetch_add(elapsed_ns, std::memory_order_relaxed);
}

UnifiedMemory::WaitSourceScope::WaitSourceScope(Uma3::WaitSource source): m_previous(t_wait_source) {
	t_wait_source = source;
}

UnifiedMemory::WaitSourceScope::~WaitSourceScope() {
	t_wait_source = m_previous;
}

void UnifiedMemory::WaitWritersForCp(uint64_t vaddr, uint64_t size) {
	if (!m_active || !HasPendingWriters(vaddr, size)) {
		return;
	}
	m_cp_waits.fetch_add(1, std::memory_order_relaxed);
	const WaitSourceScope scope(Uma3::WaitSource::Cp);
	WaitWriters(vaddr, size);
}

bool UnifiedMemory::ReadForCpu(uint64_t vaddr, void* data, uint64_t size) {
	{
		// A caller that named itself keeps its name (ReadMemory ...); an anonymous one is the CPU read.
		const WaitSourceScope scope(t_wait_source == Uma3::WaitSource::Other ? Uma3::WaitSource::Cpu
		                                                                      : t_wait_source);
		WaitWriters(vaddr, size);
	}
	ShadowCheckRead(vaddr, size);
	return LibKernel::Memory::TryReadBacking(vaddr, data, size);
}

// KYTY_UMA=shadow: a CPU read about to consume bytes the write fences call final is repeated after
// a full drain (every writer of every range retired). A difference means a GPU writer the fences
// did not know about (uma_dshadow_ne). A guest thread writing the same bytes at that moment would
// count too; the first few differences are logged with address and first differing offset.
void UnifiedMemory::ShadowCheckRead(uint64_t vaddr, uint64_t size) {
	if (!m_dshadow) {
		return;
	}
	const auto n = m_dshadow_calls.fetch_add(1, std::memory_order_relaxed) + 1;
	if (!UmaReadFence::ShouldCheck(n, m_dshadow_every, size, 65536)) {
		return;
	}
	std::vector<uint8_t> before(size);
	std::vector<uint8_t> after(size);
	if (!LibKernel::Memory::TryReadBacking(vaddr, before.data(), size)) {
		return;
	}
	WaitWriters(0, 0);
	if (!LibKernel::Memory::TryReadBacking(vaddr, after.data(), size)) {
		return;
	}
	size_t first = 0;
	if (!UmaReadFence::Differs(before.data(), after.data(), size, &first)) {
		m_dshadow_eq.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const auto bad = m_dshadow_ne.fetch_add(1, std::memory_order_relaxed) + 1;
	if (bad <= 20) {
		LOGF("UMA: data shadow NE x%" PRIu64 " addr=0x%016" PRIx64 " size=%" PRIu64
		     " first_diff=+%zu before=0x%02x after=0x%02x\n",
		     bad, vaddr, size, first, before[first], after[first]);
	}
}

// DIAG (KYTY_UMA_SHADOW=1): the refactored layers must agree with the mirror they replaced.
//  1. Page tables: every page the published GpuPageTable maps must be what the mapping mirror
//     says (pages with a deferred unmap pending are skipped), and the other way round.
//  2. Bytes: a sample of the mapped ranges read through Translate + the arena alias must equal
//     the legacy read (TryReadBacking by guest address); a sample that changes while being read
//     (GPU or CPU writer) is skipped.
void UnifiedMemory::ShadowCompare() {
	++m_shadow_runs;
	uint64_t bad = 0;
	m_sync.Map().ForEachRun([&](uint64_t vaddr, uint64_t size, uint64_t backing_offset) {
		for (auto page = (vaddr + PageSize - 1) & ~(PageSize - 1);
		     page + PageSize <= vaddr + size; page += PageSize) {
			if (m_sync.IsPending(page)) {
				continue;
			}
			uint64_t entry = 0;
			const auto expected = DeviceAddressOf(backing_offset + (page - vaddr), PageSize);
			const bool found    = m_table->LookupWide(page, PageBits, &entry);
			// 0: the chunk is not imported, so nothing may be published for the page.
			if (expected == 0 ? (found && entry != 0) : (!found || entry != expected)) {
				++bad;
			}
		}
	});
	m_table->ForEachMapped([&](uint64_t page, uint64_t device) {
		if (m_sync.IsPending(page & ~(PageSize - 1))) {
			return;
		}
		uint64_t offset = 0;
		if (!m_sync.Map().RangeOffset(page, Table::PageSize, &offset) ||
		    device != DeviceAddressOf(offset, Table::PageSize)) {
			++bad;
		}
	});

	// Bytes: one 64 KB sample per run, rotating through the runs.
	constexpr uint64_t kSample = 64 * 1024;
	uint64_t           index   = 0;
	std::vector<uint8_t> a(kSample), a2(kSample), b(kSample);
	m_sync.Map().ForEachRun([&](uint64_t vaddr, uint64_t size, uint64_t) {
		if ((index++ + m_shadow_cursor) % 8 != 0) {
			return;
		}
		const auto bytes = std::min<uint64_t>(size, kSample);
		std::vector<LibKernel::Memory::GuestArenaSpan> spans;
		if (!LibKernel::Memory::TranslateGuestRange(vaddr, bytes, &spans)) {
			++bad;
			return;
		}
		const auto read_arena = [&](std::vector<uint8_t>& out) {
			uint64_t at = 0;
			for (const auto& span: spans) {
				std::memcpy(out.data() + at, reinterpret_cast<const void*>(m_arena.base + span.offset),
				            static_cast<size_t>(span.size));
				at += span.size;
			}
		};
		read_arena(a);
		const bool legacy = LibKernel::Memory::TryReadBacking(vaddr, b.data(), bytes);
		read_arena(a2);
		if (std::memcmp(a.data(), a2.data(), bytes) != 0) {
			++m_shadow_skipped; // a writer is active
			return;
		}
		if (!legacy || Checksum(a.data(), bytes) != Checksum(b.data(), bytes)) {
			++bad;
		}
	});
	++m_shadow_cursor;
	// Stage 3: every recent window must be what the kernel's translation says (the mirror the
	// views are cut from may not drift from the mappings).
	if (m_stage3) {
		for (const auto& view: m_recent_views) {
			if (view.size == 0 || !Covers(view.vaddr, view.size)) {
				continue; // empty slot, or unmapped since (a deferred unmap keeps GPU access only)
			}
			std::vector<LibKernel::Memory::GuestArenaSpan> spans;
			if (!LibKernel::Memory::TranslateGuestRange(view.vaddr, view.size, &spans) ||
			    spans.size() != 1 || spans[0].offset != view.offset) {
				++bad;
				++m_shadow_view_mismatch;
			}
		}
	}
	if (bad != 0) {
		m_shadow_mismatch += bad;
		LOGF("UMA: shadow-compare MISMATCH x%" PRIu64 " (uma_shadow_mismatch=%" PRIu64 ")\n", bad,
		     m_shadow_mismatch.load());
	}
}

} // namespace Libs::Graphics
