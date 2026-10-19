#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hotPages.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpWaitDiag.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/recompiler/ir/passes/specShadow.h"
#include "kernel/memory.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <memory>
#include <unordered_set>
#include <xxhash.h>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto&                buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		const auto [it, inserted] = m_buffers.emplace(buffer.CpuAddress(), id);
		(void)it;
		EXIT_IF(!inserted);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		m_memory_tracker.QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		const auto found = m_buffers.find(buffer.CpuAddress());
		EXIT_IF(found == m_buffers.end() || found->second != id);
		m_buffers.erase(found);
		EXIT_IF(buffer.Size() > m_total_used_memory);
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	if (!buffer.is_deleted) {
		m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
	}
}

void BufferCache::DeleteBuffer(BufferId id) {
	if (IsBufferInvalid(id)) {
		return;
	}
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] {
			if (m_graphics.device_address_destruction_waits_for_queue) {
				const CpWait::CauseScope cause(CpWait::Cause::DeleteDrain);
				// Every command buffer submitted so far may reference this device-address buffer.
				// ponytail: drains the queue per deletion batch; a per-buffer retire tick would
				// need the driver to stop referencing buffers the work does not use.
				m_scheduler.GetMasterSemaphore().Wait(m_scheduler.CurrentTick() - 1);
			}
			m_slot_buffers.erase(id);
		});
	} else {
		m_slot_buffers.erase(id);
	}
}

BufferCache::KnownFillPlan BufferCache::PlanKnownFill(uint64_t vaddr, uint64_t size) {
	KnownFillPlan plan;
	// The same pieces DownloadBufferMemory would copy: GPU-modified bytes of the tracker pages of the range.
	m_memory_tracker.ForEachDownloadRange<false>(vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			plan.pieces.push_back({start, end - start, 0});
		});
	});
	if (plan.pieces.empty()) {
		return plan;
	}
	for (auto& piece: plan.pieces) {
		const auto decision = DccFillPublish::Decide(m_write_ledger.QueryFill(piece.address, piece.size),
		                                             HasPendingDownload(piece.address, piece.size));
		if (!decision.publish) {
			FrameTiming::Add(FrameTiming::Counter::DccPublishFallbackN);
			const auto query = m_write_ledger.QueryFill(piece.address, piece.size);
			const auto why   = DccFillPublish::Explain(query, HasPendingDownload(piece.address, piece.size));
			FrameTiming::Add(static_cast<FrameTiming::Counter>(static_cast<uint32_t>(FrameTiming::Counter::DccFallbackNoFillN) +
			                                                   static_cast<uint32_t>(why)));
			if (SyncCensus::Enabled()) {
				LogDccFallback(piece.address, piece.size, why);
			}
			plan.pieces.clear();
			return plan;
		}
		piece.value = decision.value;
	}
	plan.active = true;
	return plan;
}

void BufferCache::PublishKnownFill(const KnownFillPlan& plan) {
	for (const auto& piece: plan.pieces) {
		DccFillPublish::Apply(
		    piece.address, piece.size, piece.value,
		    [](uint64_t address, const uint8_t* bytes, uint64_t bytes_size) {
			    Libs::LibKernel::Memory::WriteBacking(address, bytes, bytes_size);
		    },
		    [this](uint64_t address, uint64_t bytes_size) {
			    m_gpu_modified_ranges.Subtract(address, bytes_size);
			    m_write_ledger.Trim(address, bytes_size);
			    m_memory_tracker.UnmarkRegionAsGpuModified(address, bytes_size);
		    });
		FrameTiming::Add(FrameTiming::Counter::DccPublishN);
	}
}

void BufferCache::ShadowCompareKnownFill(const KnownFillPlan& plan) {
	static std::atomic<uint32_t> logged {0};
	for (const auto& piece: plan.pieces) {
		std::vector<uint8_t> actual(static_cast<size_t>(piece.size));
		if (!Libs::LibKernel::Memory::TryReadBacking(piece.address, actual.data(), actual.size())) {
			continue;
		}
		FrameTiming::Add(FrameTiming::Counter::DccKnownWouldHitN);
		const auto diff = DccFillPublish::FirstDifference(piece.value, piece.address, actual.data(), piece.size);
		if (diff == piece.size) {
			continue;
		}
		FrameTiming::Add(FrameTiming::Counter::DccKnownMismatchN);
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("[dcc-known-mismatch] addr=0x%016" PRIx64 " size=0x%" PRIx64 " value=0x%08" PRIx32 " first_diff=0x%" PRIx64 "\n",
			     piece.address, piece.size, piece.value, diff);
		}
	}
}

bool BufferCache::HasPendingDownload(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_eager_mutex);
	for (const auto& [page, entry]: m_eager_pages) {
		if (entry.tick != 0 && page < vaddr + size && vaddr < page + TRACKER_PAGE_SIZE) {
			return true;
		}
	}
	return false;
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	CpPhases::Scope phase(CpPhases::Phase::ReadbackCpu);
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size     = 0;
	const auto                  buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges.Set(), address, bytes,
		                                           "buffer download");
		    // S428 (B2): every piece becomes "in flight" before the range stops being "dirty" (before: dirty -> neither -> in flight, a window in
		    // which a guest thread's fault handler saw clean bytes and a stale backing); the whole step is under the dirty-set lock.
		    m_gpu_modified_ranges.BeginDownloadCopy(m_inflight_download_ranges, address, bytes, [&](uint64_t start, uint64_t end) {
			    copies.emplace_back(start - buffer_address, total_size, end - start);
			    // Keep packed ranges on separate cache lines, as in shadPS4.
			    total_size += Common::AlignUp(end - start, 64);
		    });
		    if (DccFillPublish::EnvMode() != DccFillPublish::Mode::Off) {
			    m_write_ledger.Forget(address, bytes);
		    }
	    });
	if (copies.empty()) {
		return false;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);

	FrameTiming::Add(FrameTiming::Counter::RbDownloadBytes, total_size);
	auto [mapped, offset] = m_download_buffer.Map(total_size, 64);
	std::unique_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	const auto& download = temporary ? *temporary : m_download_buffer;
	for (auto& copy: copies) {
		copy.dstOffset += offset;
	}

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(copies.size()), copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = offset;
	after.size          = total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
	// S418: from the BeginDownloadCopy above until the backing is written the range is in flight (no longer GPU-modified, not yet current in the backing).
	auto publish = [this, mapped, offset, total_size, buffer_address,
	                copies = std::move(copies), owner = std::move(temporary)] {
		(owner ? *owner : m_download_buffer).Invalidate(offset, total_size);
		for (const auto& copy: copies) {
			Libs::LibKernel::Memory::WriteBacking(buffer_address + copy.srcOffset,
			                                      mapped + (copy.dstOffset - offset), copy.size);
			m_inflight_download_ranges.Remove(buffer_address + copy.srcOffset, copy.size);
		}
	};
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(std::move(publish));
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		publish();
	}
	return true;
}

namespace {
// S440 (KYTY_WF_HOT_UNPROTECT): the tracker hashes a page the way the upload would read it.
bool HashGuestPage(uint64_t page, uint64_t* hash) {
	alignas(64) uint8_t data[TRACKER_PAGE_SIZE];
	if (!LibKernel::Memory::CopySparseMemory(page, data, sizeof(data))) {
		return false;
	}
	*hash = XXH3_64bits(data, sizeof(data));
	return true;
}

uint64_t GuestFlipsForTracker() {
	return FrameTiming::GuestFlips();
}

} // namespace

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache) {
	HotPages::EpochSource().store(&GuestFlipsForTracker, std::memory_order_relaxed);
	HotPages::HasherSource().store(&HashGuestPage, std::memory_order_relaxed);
	m_scheduler.SetPreSubmitHook([this] { FlushEagerReadbacks(); });
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
	m_buffers.clear();
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid memory-invalidation range\n");
	}
	// With KYTY_DCC_KNOWN the fill must still be known when the readback decides, so the CPU write forgets it afterwards.
	const bool known = DccFillPublish::EnvMode() != DccFillPublish::Mode::Off;
	m_memory_tracker.InvalidateRegion(vaddr, size, [this, vaddr, size] {
		FrameTiming::Add(FrameTiming::Counter::ReadbackWriteFault);
		const SyncCensus::SiteScope site_tag(SyncCensus::Site::WriteFault);
		ReadMemory(vaddr, size, true);
	});
	if (known) {
		m_write_ledger.Forget(vaddr, size); // a CPU write: remembered fills of the range are stale
	}
}

namespace {
// S398: sync_<kind>_<site> counters sit in SyncCensus::Site order after their first member.
FrameTiming::Counter SyncCounter(FrameTiming::Counter first, SyncCensus::Site site) {
	return static_cast<FrameTiming::Counter>(static_cast<uint32_t>(first) + static_cast<uint32_t>(site));
}

// S406: sync_spec_purpose_<x>_n, in the order cond, slot, desc, fill, wg, call, other.
FrameTiming::Counter SpecPurposeCounter(ShaderRecompiler::IR::SpecShadow::Purpose purpose) {
	using P = ShaderRecompiler::IR::SpecShadow::Purpose;
	const auto index = purpose == P::Other ? 6u : static_cast<uint32_t>(purpose) - 1u;
	return static_cast<FrameTiming::Counter>(static_cast<uint32_t>(FrameTiming::Counter::SyncSpecPurposeCondN) + index);
}

} // namespace

bool BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	FrameTiming::Add(FrameTiming::Counter::Readbacks);
	const CpWait::CauseScope wait_cause(CpWait::CauseOfSite(SyncCensus::CurrentSite())); // S413: explicit readback by its SyncCensus tag
	const bool census = SyncCensus::Enabled();
	const auto site   = census ? SyncCensus::CurrentSite() : SyncCensus::Site::Other;
	// KYTY_DCC_EAGER: the requesting thread's tag says whether this is a DCC metadata read (the GPU-thread lambda below has its own).
	const bool eager_dcc = EagerPages::EagerEnv() && SyncCensus::CurrentSite() == SyncCensus::Site::Dcc;
	const auto wait_begin = std::chrono::steady_clock::now();
	const auto add_wait   = [&] {
		const auto ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                          std::chrono::steady_clock::now() - wait_begin)
		                                          .count());
		FrameTiming::Add(FrameTiming::Counter::ReadbackWaitNs, ns);
		if (census) {
			FrameTiming::Add(SyncCounter(FrameTiming::Counter::SyncDccN, site));
			FrameTiming::Add(SyncCounter(FrameTiming::Counter::SyncDccNs, site), ns);
			if (site == SyncCensus::Site::Spec) FrameTiming::Add(SpecPurposeCounter(ShaderRecompiler::IR::SpecShadow::CurrentPurpose()));
		}
		return ns;
	};
	SyncCensus::Probe probe;
	if (census) {
		probe.thread_id  = SyncCensus::ThreadId();
		probe.gpu_thread = GuestGpu::IsGpuThread();
		// Snapshots BEFORE any wait or drain (a drain submits and resets them).
		probe.cmds_since_submit = SyncCensus::Diff(FrameTiming::Get(FrameTiming::Counter::Draws) +
		                                               FrameTiming::Get(FrameTiming::Counter::Dispatches),
		                                           SyncCensus::CommandsAtSubmit().load(std::memory_order_relaxed));
		probe.has_snapshot = true;
	}
	if (!is_write && WaitEagerReadback(vaddr, size, eager_dcc, census ? &probe : nullptr)) {
		FrameTiming::Add(FrameTiming::Counter::ReadbackEagerHits);
		if (eager_dcc) {
			FrameTiming::Add(FrameTiming::Counter::DccEagerHitN);
		}
		if (census) {
			FrameTiming::Add(SyncCounter(FrameTiming::Counter::SyncEagerHitDccN, site));
		}
		const auto ns = add_wait();
		if (census) {
			NoteEagerProbe(probe);
			LogSyncReadback(site, vaddr, size, ns, probe, true);
		}
		return true;
	}
	bool downloaded = true;
	const auto send_time = std::chrono::steady_clock::now();
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write, &downloaded, &probe, census, site, eager_dcc, send_time] {
		const auto lambda_begin = std::chrono::steady_clock::now();
		if (census) {
			probe.queue_ns = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(lambda_begin - send_time).count());
		}
		// A download already queued for this page must land before this one decides anything.
		(void)WaitEagerReadback(vaddr, size, eager_dcc);
		if (!is_write) {
			NoteReadbackFault(vaddr, size, eager_dcc);
		}
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		auto& buffer = m_slot_buffers[FindBuffer(vaddr, size)];

		if (census) {
			if (site == SyncCensus::Site::Spec) {
				probe.write_seq = m_sync_seq.MaxTick(vaddr, size);
				auto& last      = m_spec_last_seq[vaddr * 1000003ull ^ size];
				probe.written_since = last != probe.write_seq;
				last                = probe.write_seq;
			}
			probe.writer  = m_sync_ticks.MaxTick(vaddr, size);
			probe.current = m_scheduler.CurrentTick();
			m_scheduler.GetMasterSemaphore().Refresh();
			probe.done = SyncCensus::WriterDone(probe.writer, m_scheduler.GetMasterSemaphore().KnownGpuTick());
			// S404: snapshots before the drain (the Submit inside the download resets cmds_since_submit).
			const auto commands = FrameTiming::Get(FrameTiming::Counter::Draws) + FrameTiming::Get(FrameTiming::Counter::Dispatches);
			probe.cmds_since_submit = SyncCensus::Diff(commands, SyncCensus::CommandsAtSubmit().load(std::memory_order_relaxed));
			probe.backlog           = SyncCensus::Backlog(probe.current, m_scheduler.GetMasterSemaphore().KnownGpuTick());
			probe.cmds_since_writer = probe.writer == 0 ? 0 : SyncCensus::Diff(commands, m_sync_cmds.MaxTick(vaddr, size));
			probe.has_snapshot      = true;
			if (SyncCensus::StaleFreshSite(site, size) && !is_write) {
				probe.stale_fresh.size = size;
				probe.stale_fresh.ok   = LibKernel::Memory::TryReadBacking(vaddr, probe.stale_fresh.stale, size);
			}
		}
		const auto    known_mode = DccFillPublish::EnvMode();
		KnownFillPlan fill_plan;
		if (known_mode != DccFillPublish::Mode::Off) {
			fill_plan = PlanKnownFill(vaddr, size);
		}
		if (known_mode == DccFillPublish::Mode::On && fill_plan.active) {
			// A range that is one known fill: no copy, no wait for the queue.
			PublishKnownFill(fill_plan);
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
			if (census) {
				m_sync_ticks.Forget(vaddr, size);
			}
		} else if (DownloadBufferMemory<false>(buffer, vaddr, size)) {
			FrameTiming::Add(FrameTiming::Counter::ReadbackDownloads);
			m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
			if (known_mode == DccFillPublish::Mode::Shadow && fill_plan.active) {
				ShadowCompareKnownFill(fill_plan);
			}
			if (census) {
				probe.downloaded = true;
				m_sync_ticks.Forget(vaddr, size);
				if (site == SyncCensus::Site::Dcc && size <= 0x10000) {
					std::vector<uint8_t> bytes(size);
					uint64_t             hash = 0;
					if (LibKernel::Memory::TryReadBacking(vaddr, bytes.data(), size)) {
						hash = SyncCensus::HashBytes(bytes.data(), size);
					}
					const auto repeat = m_sync_repeat.Note(vaddr, size, FrameTiming::GuestFlips(), hash);
					probe.repeat      = repeat.repeat;
					probe.same        = repeat.same;
				}
			}
		} else if (!is_write) {
			// A read fault that downloads nothing leaves the page protected and faults again;
			// the caller restores access.
			downloaded = false;
			static std::atomic<uint64_t> noops {0};
			const auto                   n = noops.fetch_add(1, std::memory_order_relaxed) + 1;
			if ((n & (n - 1)) == 0) {
				LOGF("[read-fault-noop] n=%" PRIu64 " addr=0x%016" PRIx64
				     " tracker_gpu_modified=%d dirty_ranges=%d\n",
				     n, vaddr, IsRegionGpuModified(vaddr, 1) ? 1 : 0,
				     HasGpuDirtyBytes(vaddr, 1) ? 1 : 0);
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
		}
		if (census) {
			if (probe.stale_fresh.size != 0 && probe.stale_fresh.ok) {
				probe.stale_fresh.ok = LibKernel::Memory::TryReadBacking(vaddr, probe.stale_fresh.fresh, size);
			}
			probe.lambda_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
			                                            std::chrono::steady_clock::now() - lambda_begin)
			                                            .count());
		}
	});
	const auto ns = add_wait();
	if (eager_dcc) {
		FrameTiming::Add(FrameTiming::Counter::DccEagerMissN);
	}
	if (census) {
		NoteEagerProbe(probe);
		FrameTiming::Add(FrameTiming::Counter::SyncQueueNs, probe.queue_ns);
		FrameTiming::Add(FrameTiming::Counter::SyncLambdaNs, probe.lambda_ns);
		if (probe.downloaded && probe.stale_fresh.size != 0 && probe.stale_fresh.ok) {
			const bool equal = probe.stale_fresh.Equal();
			if (site == SyncCensus::Site::Spec) {
				FrameTiming::Add(equal ? FrameTiming::Counter::SyncStaleEqSpecN : FrameTiming::Counter::SyncStaleNeSpecN);
			} else {
				FrameTiming::Add(equal ? FrameTiming::Counter::SyncStaleEqRfGpuN : FrameTiming::Counter::SyncStaleNeRfGpuN);
			}
		}
		if (probe.downloaded) {
			const auto lag = SyncCensus::Lag(probe.current, probe.writer);
			FrameTiming::Add(FrameTiming::Counter::SyncLagSum, lag);
			if (lag == 0) {
				FrameTiming::Add(FrameTiming::Counter::SyncLag0N);
			}
			if (probe.done) {
				FrameTiming::Add(FrameTiming::Counter::SyncWriterDoneN);
				FrameTiming::Add(FrameTiming::Counter::SyncWriterDoneNs, ns);
			}
			if (probe.repeat) {
				FrameTiming::Add(FrameTiming::Counter::SyncDccRepeatN);
				if (probe.same) {
					FrameTiming::Add(FrameTiming::Counter::SyncDccRepeatSameN);
				}
			}
		}
		LogSyncReadback(site, vaddr, size, ns, probe, false);
	}
	return downloaded;
}

// S404: why a DCC piece took the old path, once per (addr, flip) inside the same log window as [sync].
void BufferCache::LogDccFallback(uint64_t address, uint64_t size, DccFillPublish::Why why) {
	static std::atomic<uint64_t> logged {0};
	static uint64_t              last_flip = ~0ull;
	static std::unordered_set<uint64_t> seen; // GPU thread only
	const auto                   flip = FrameTiming::GuestFlips();
	if (flip != last_flip) {
		seen.clear();
		last_flip = flip;
	}
	if (!seen.insert(address).second) {
		return;
	}
	const auto n = logged.fetch_add(1, std::memory_order_relaxed);
	if (n >= 300 && !(flip >= 6300 && flip <= 6310)) {
		return;
	}
	LOGF("[dcc-fallback] flip=%" PRIu64 " addr=0x%016" PRIx64 " size=0x%" PRIx64 " why=%s\n", flip, address, size,
	     DccFillPublish::WhyName(why));
}

void BufferCache::LogSyncReadback(SyncCensus::Site site, uint64_t vaddr, uint64_t size, uint64_t wait_ns,
                                  const SyncCensus::Probe& probe, bool eager) {
	static std::atomic<uint64_t> logged {0};
	const auto                   flip = FrameTiming::GuestFlips();
	const auto                   n    = logged.fetch_add(1, std::memory_order_relaxed);
	if (n >= 300 && !(flip >= 6300 && flip <= 6310)) {
		return;
	}
	// S404: cmds_since_submit is the snapshot taken before the drain; the live difference would count from the drain's own Submit.
	const auto commands = probe.has_snapshot
	                          ? probe.cmds_since_submit
	                          : SyncCensus::Diff(FrameTiming::Get(FrameTiming::Counter::Draws) +
	                                                 FrameTiming::Get(FrameTiming::Counter::Dispatches),
	                                             SyncCensus::CommandsAtSubmit().load(std::memory_order_relaxed));
	char reader[160] = "";
	if (site == SyncCensus::Site::Spec) {
		const auto& tag = SyncCensus::CurrentReaderRef();
		std::snprintf(reader, sizeof(reader), " reader=%016" PRIx64 "/%s path=%s purpose=%s wseq=%" PRIu64 " wsince=%d", tag.hash,
		              tag.stage, tag.path, ShaderRecompiler::IR::SpecShadow::PurposeName(ShaderRecompiler::IR::SpecShadow::CurrentPurpose()), probe.write_seq,
		              probe.written_since ? 1 : 0);
	}
	char stale[2 * SyncCensus::StaleFresh::kMaxSize * 2 + 48] = "";
	if (probe.stale_fresh.size != 0 && probe.stale_fresh.ok) {
		std::snprintf(stale, sizeof(stale), " stale=%s fresh=%s eq=%d",
		              SyncCensus::ToHex(probe.stale_fresh.stale, probe.stale_fresh.size).c_str(),
		              SyncCensus::ToHex(probe.stale_fresh.fresh, probe.stale_fresh.size).c_str(), probe.stale_fresh.Equal() ? 1 : 0);
	}
	char eager_text[160] = "";
	if (probe.eager_seen) {
		std::snprintf(eager_text, sizeof(eager_text), " etick=%" PRIu64 " elag=%" PRIu64 " edone=%d ewait_us=%" PRIu64,
		              probe.eager_tick, probe.eager_lag, probe.eager_already_done ? 1 : 0, probe.eager_wait_ns / 1000);
	}
	LOGF("[sync] flip=%" PRIu64 " site=%s addr=0x%016" PRIx64 " size=0x%" PRIx64 " wait_us=%" PRIu64
	     " dl=%d lag=%" PRIu64 " done=%d eager=%d cmds_since_submit=%" PRIu64 " backlog=%" PRIu64 " cmds_since_writer=%" PRIu64
	     " tid=%u gpu_thread=%d queue_us=%" PRIu64 " lambda_us=%" PRIu64 "%s%s%s\n",
	     flip, SyncCensus::SiteName(site), vaddr, size, wait_ns / 1000, probe.downloaded ? 1 : 0,
	     SyncCensus::Lag(probe.current, probe.writer), probe.done ? 1 : 0, eager ? 1 : 0, commands, probe.backlog,
	     probe.cmds_since_writer, probe.thread_id, probe.gpu_thread ? 1 : 0, probe.queue_ns / 1000, probe.lambda_ns / 1000, reader,
	     stale, eager_text);
}

namespace {
bool EagerReadbackEnabled() {
	static const bool enabled = FrameTiming::OptEnabled("eager_readback");
	return enabled;
}
constexpr uint32_t EagerHotFaults = 4;   // faults on one tracker page before it counts as hot
constexpr size_t   EagerMaxPages  = 256; // hot pages tracked at most
constexpr size_t   EagerMaxPagesDcc = 1024; // ... with KYTY_DCC_EAGER=1 (DCC metadata ranges are 2..72 pages)
} // namespace

void BufferCache::NoteReadbackFault(uint64_t vaddr, uint64_t size, bool dcc) {
	if (!EagerReadbackEnabled()) {
		return;
	}
	// A DCC metadata read counts every page of its range; any other fault only the first (a range read is one fault).
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = dcc && size != 0 ? Common::AlignDown(vaddr + size - 1, TRACKER_PAGE_SIZE) : first;
	const auto limit = EagerPages::EagerEnv() ? EagerMaxPagesDcc : EagerMaxPages;
	for (auto page = first; page <= last; page += TRACKER_PAGE_SIZE) {
		if (++m_readback_fault_counts[page] < EagerHotFaults) {
			continue;
		}
		std::lock_guard lock(m_eager_mutex);
		if (m_eager_pages.size() < limit) {
			m_eager_pages.try_emplace(page);
			if (dcc) {
				FrameTiming::Set(FrameTiming::Counter::EagerPagesN, m_eager_pages.size());
			}
		}
	}
}

void BufferCache::NoteGpuWrite(uint64_t vaddr, uint64_t size) {
	if (!EagerReadbackEnabled()) {
		return;
	}
	std::lock_guard lock(m_eager_mutex);
	if (m_eager_pages.empty()) {
		return;
	}
	const auto end = vaddr + size;
	for (auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE); page < end;
	     page += TRACKER_PAGE_SIZE) {
		const auto found = m_eager_pages.find(page);
		if (found == m_eager_pages.end()) {
			continue;
		}
		found->second.seq++;
		if (std::find(m_eager_pending.begin(), m_eager_pending.end(), page) ==
		    m_eager_pending.end()) {
			m_eager_pending.push_back(page);
		}
	}
}

void BufferCache::FlushEagerReadbacks() {
	if (m_eager_pending.empty() || !GuestGpu::IsGpuThread()) {
		return;
	}
	const auto tick = m_scheduler.CurrentTick();
	// A run is one page, or with KYTY_DCC_EAGER=1 the adjacent pending pages (one copy instead of one barrier per page).
	std::vector<EagerPages::Run> runs;
	if (EagerPages::EagerEnv()) {
		runs = EagerPages::MergePages(m_eager_pending, TRACKER_PAGE_SIZE);
	} else {
		for (const auto page: m_eager_pending) {
			runs.push_back({page, page + TRACKER_PAGE_SIZE});
		}
	}
	for (const auto& run: runs) {
		const auto run_size = run.end - run.begin;
		if (!m_gpu_modified_ranges.Intersects(run.begin, run_size)) {
			continue;
		}
		// Pages that carried no GPU write get no download entry (a page the synchronous path cleaned meanwhile).
		std::vector<uint64_t> dirty_pages;
		for (auto page = run.begin; page < run.end; page += TRACKER_PAGE_SIZE) {
			if (m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
				dirty_pages.push_back(page);
			}
		}
		// Download every GPU-written range of the run, owner by owner, as ReadMemory does.
		const int guards = std::max<int>(8, static_cast<int>(run_size / TRACKER_PAGE_SIZE));
		for (int guard = 0; guard < guards && m_gpu_modified_ranges.Intersects(run.begin, run_size);
		     guard++) {
			uint64_t first = 0;
			m_gpu_modified_ranges.ForEachInRange(run.begin, run_size,
			                                     [&](uint64_t start, uint64_t) {
				                                     if (first == 0) {
					                                     first = start;
				                                     }
			                                     });
			if (first == 0) {
				break;
			}
			auto&      buffer = m_slot_buffers[FindBuffer(first, 1)];
			const auto begin  = std::max(run.begin, buffer.CpuAddress());
			const auto end = std::min(run.end, buffer.CpuAddress() + buffer.Size());
			if (!DownloadBufferMemory<true>(buffer, begin, end - begin)) {
				break;
			}
		}
		for (const auto page: dirty_pages) {
			if (m_gpu_modified_ranges.Intersects(page, TRACKER_PAGE_SIZE)) {
				continue; // left to the synchronous path
			}
			uint64_t seq = 0;
			{
				std::lock_guard lock(m_eager_mutex);
				auto&           entry = m_eager_pages[page];
				entry.tick            = tick;
				seq                   = entry.seq;
			}
			FrameTiming::Add(FrameTiming::Counter::ReadbackEagerDownloads);
			// Queued after DownloadBufferMemory's own publication, so the bytes are in place.
			m_scheduler.DeferPriorityOperation([this, page, tick, seq] {
				std::lock_guard lock(m_eager_mutex);
				auto&           entry = m_eager_pages[page];
				if (entry.tick != tick) {
					return;
				}
				entry.tick = 0;
				if (entry.seq == seq) {
					m_memory_tracker.UnmarkRegionAsGpuModified(page, TRACKER_PAGE_SIZE);
				}
			});
		}
	}
	m_eager_pending.clear();
}

// S404: counters for an eager wait the probe saw (the requesting thread's side of an rf_guest fault).
void BufferCache::NoteEagerProbe(const SyncCensus::Probe& probe) {
	if (!probe.eager_seen) {
		return;
	}
	if (probe.eager_already_done) {
		FrameTiming::Add(FrameTiming::Counter::SyncEagerAlreadyDoneN);
	}
	FrameTiming::Add(FrameTiming::Counter::SyncEagerLagSum, probe.eager_lag);
	FrameTiming::Add(FrameTiming::Counter::SyncEagerWaitNs, probe.eager_wait_ns);
}

bool BufferCache::WaitEagerReadback(uint64_t vaddr, uint64_t size, bool dcc, SyncCensus::Probe* probe) {
	const CpWait::CauseScope wait_cause(CpWait::Cause::Eager);
	if (!EagerReadbackEnabled() || CommandScheduler::InDeferredOperation()) {
		return false;
	}
	const auto page = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	if (Common::AlignDown(vaddr + size - 1, TRACKER_PAGE_SIZE) != page) {
		return dcc && WaitEagerReadbackPages(vaddr, size, probe);
	}
	uint64_t tick = 0;
	{
		std::lock_guard lock(m_eager_mutex);
		const auto      found = m_eager_pages.find(page);
		if (found == m_eager_pages.end() || found->second.tick == 0) {
			return false;
		}
		tick = found->second.tick;
	}
	// The tick was taken while its command buffer was being submitted, so it is (or is about
	// to be) in flight; a timeline wait may precede the signal operation.
	const auto wait_begin = probe != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	if (probe != nullptr) {
		NoteEagerWait(*probe, tick);
	}
	m_scheduler.GetMasterSemaphore().Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	if (probe != nullptr) {
		probe->eager_wait_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                 std::chrono::steady_clock::now() - wait_begin)
		                                                 .count());
	}
	return !m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE);
}

void BufferCache::NoteEagerWait(SyncCensus::Probe& probe, uint64_t tick) {
	probe.eager_seen         = true;
	probe.eager_tick         = tick;
	probe.current            = m_scheduler.CurrentTick();
	probe.eager_lag          = SyncCensus::Diff(probe.current, tick);
	probe.eager_already_done = m_scheduler.GetMasterSemaphore().KnownGpuTick() >= tick; // before the Wait
	probe.backlog            = SyncCensus::Backlog(probe.current, m_scheduler.GetMasterSemaphore().KnownGpuTick());
}

// KYTY_DCC_EAGER=1: a DCC metadata range of several pages is served when every page is clean or has an eager download in
// flight, and no page is GPU-modified again once the newest of those downloads has landed (a GPU write after the download
// keeps its page marked; that range takes the synchronous path).
bool BufferCache::WaitEagerReadbackPages(uint64_t vaddr, uint64_t size, SyncCensus::Probe* probe) {
	const CpWait::CauseScope wait_cause(CpWait::Cause::Eager);
	const auto first = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto last  = Common::AlignDown(vaddr + size - 1, TRACKER_PAGE_SIZE);
	std::vector<EagerPages::PageState> pages;
	{
		std::lock_guard lock(m_eager_mutex);
		for (auto page = first; page <= last; page += TRACKER_PAGE_SIZE) {
			const auto found = m_eager_pages.find(page);
			pages.push_back({false, found == m_eager_pages.end() ? 0 : found->second.tick});
		}
	}
	size_t index = 0;
	for (auto page = first; page <= last; page += TRACKER_PAGE_SIZE, index++) {
		pages[index].gpu_modified = m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE);
	}
	const auto plan = EagerPages::MakePlan(pages);
	if (!plan.wait) {
		return false;
	}
	const auto wait_begin = probe != nullptr ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
	if (probe != nullptr) {
		NoteEagerWait(*probe, plan.max_tick);
	}
	m_scheduler.GetMasterSemaphore().Wait(plan.max_tick);
	m_scheduler.WaitPriorityOperations(plan.max_tick);
	if (probe != nullptr) {
		probe->eager_wait_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
		                                                 std::chrono::steady_clock::now() - wait_begin)
		                                                 .count());
	}
	index = 0;
	for (auto page = first; page <= last; page += TRACKER_PAGE_SIZE, index++) {
		pages[index].gpu_modified = m_memory_tracker.IsRegionGpuModified(page, TRACKER_PAGE_SIZE);
	}
	return EagerPages::Served(pages);
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice
			// versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	EXIT_IF(m_scheduler.Current().IsInvalid());
	const auto end     = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr              = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	if (is_written) {
		NoteGpuWrite(vaddr, size);
	}
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto              native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(
		    vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer,
		    vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(
		    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
		    vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	auto& command = m_scheduler.Current();
	if (command.IsInvalid() || !GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: buffer request requires a recording command buffer\n");
	}

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		if (DccFillPublish::EnvMode() != DccFillPublish::Mode::Off) {
			// Any other GPU write hides a remembered fill of these bytes.
			m_write_ledger.RecordWrite(vaddr, size);
			if (m_shadow_fill.size != 0 && m_shadow_fill.vaddr == vaddr && m_shadow_fill.size == size) {
				// shadow: the dispatch that replaces nothing writes the fill value; remember it as a fill (the prediction).
				m_write_ledger.RecordFill(vaddr, size, m_shadow_fill.value);
				m_shadow_fill = {};
			}
		}
		if (RfOp::Enabled()) {
			m_rfop_writers.Note(vaddr, size, RfOp::CurrentPacket().opcode);
		}
		if (SyncCensus::Enabled()) {
			m_sync_ticks.Note(vaddr, size, m_scheduler.CurrentTick());
			m_sync_seq.Note(vaddr, size, ++m_sync_seq_next);
			m_sync_cmds.Note(vaddr, size, FrameTiming::Get(FrameTiming::Counter::Draws) + FrameTiming::Get(FrameTiming::Counter::Dispatches));
		}
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid image source\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds, bool force_gpu) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!force_gpu && !IsRegionGpuModified(vaddr, size)) {
		// Access the guest mapping so write faults invalidate cached buffers and images.
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
	if (DccFillPublish::EnvMode() == DccFillPublish::Mode::On) {
		// Newest entry: the buffer cache now knows these bytes (DCC clear codes need no readback).
		m_write_ledger.RecordFill(vaddr, size, value);
	}
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) &&
	    !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		FrameTiming::Add(FrameTiming::Counter::DmaMemcpyBytes, size);
		// The destination is usually write-protected for tracking, and the copy would take one
		// fault per 4 KiB page (a signal round trip, slow under Rosetta), each doing what these
		// two calls do for the whole range at once. The range is not GPU-modified (checked
		// above), so no readback is involved.
		static const bool preinvalidate = FrameTiming::OptEnabled("dma_preinvalidate");
		if (preinvalidate) {
			InvalidateMemory(dst_vaddr, size);
			m_texture_cache.InvalidateMemory(dst_vaddr, size);
		}
		std::memcpy(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		            size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

MemoryTracker::PageGeneration BufferCache::QueryPageGeneration(uint64_t vaddr, uint64_t size) {
	auto result = m_memory_tracker.QueryPageGeneration(vaddr, size);
	if (result.clean && m_gpu_modified_ranges.Intersects(vaddr, size)) result.clean = false;
	return result;
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::TryQueryGuestFault(uint64_t vaddr, uint64_t size, bool* dirty, bool* inflight) const {
	// Dirty first, in flight second: BeginDownloadCopy adds to in-flight before it removes from dirty, so "not dirty" followed by "not in flight"
	// means the copy has been published (or never existed).
	if (!m_gpu_modified_ranges.TryIntersects(vaddr, size, dirty)) {
		return false;
	}
	return m_inflight_download_ranges.TryIntersects(vaddr, size, inflight);
}

bool BufferCache::HasInflightDownloadBytes(uint64_t vaddr, uint64_t size) const {
	return m_inflight_download_ranges.Intersects(vaddr, size);
}

void BufferCache::WaitInflightDownloads() {
	// S419: the copies were recorded into the current command buffer; submit it and let their publishes run.
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
}

bool BufferCache::CanReadBacking(uint64_t vaddr, uint64_t size, int* reason) {
	const bool dirty    = HasGpuDirtyBytes(vaddr, size);
	const bool inflight = HasInflightDownloadBytes(vaddr, size);
	if (reason != nullptr) {
		*reason = dirty ? 1 : (inflight ? 2 : 0);
	}
	return CanReadBackingDecision(dirty, inflight);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		if (buffer.CpuAddress() == 0) {
			return false;
		}
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges.Set(), buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) ||
		    m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: garbage collection retained GPU ownership\n");
		}
		m_memory_tracker.UntrackMemory(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	// Take before walking: writes queued during synchronization remain pending.
	const auto pending = m_memory_tracker.TakeBdaSync();
	pending.ForEach([&](uint64_t begin, uint64_t end) {
		mapped.ForEachInRange(begin, end - begin, [this](uint64_t start, uint64_t finish) {
			SynchronizeBuffersInRange(start, finish - start);
		});
	});
}

void BufferCache::SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size) {
	const auto end = vaddr + size;
	auto       it  = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			(void)SynchronizeBuffer(buffer, start, finish - start, false, false);
		}
	}
}

} // namespace Libs::Graphics
