#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/dccFillPublish.h"
#include "graphics/host_gpu/renderer/cache/inflightRanges.h"
#include "graphics/host_gpu/renderer/cache/eagerPages.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"
#include "graphics/host_gpu/renderer/cache/gpuWriteLedger.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/rfOpDiag.h"
#include "graphics/host_gpu/renderer/syncCensus.h"

#include <array>
#include <map>
#include <mutex>
#include <span>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Returns false when a read fault found nothing to download.
	bool                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	// force_gpu: write on the GPU even when no GPU-modified range is tracked there (a CPU fill of a tracked range takes a
	// write fault per page); used by KYTY_DCC_KNOWN=1 to replace a uniform-fill dispatch.
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds, bool force_gpu = false);
	// KYTY_DCC_KNOWN=1: true when the newest GPU write to every byte of the range was one FillBuffer (any other GPU write,
	// download or CPU write to the range hides it), so the bytes equal *value without a readback.
	[[nodiscard]] bool LatestGpuWriteIsFill(uint64_t vaddr, uint64_t size, uint32_t* value) const {
		return m_write_ledger.LatestIsFill(vaddr, size, value);
	}
	// KYTY_DCC_KNOWN=shadow: the dispatch that follows writes `value` over the range; the ledger records the fill when the
	// dispatch binds exactly that range for writing (the dispatch itself still runs).
	void NoteShadowFill(uint64_t vaddr, uint64_t size, uint32_t value) { m_shadow_fill = {vaddr, size, value}; }
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	// S418: a recorded GPU->host copy of the range has not reached the backing yet (the tracker no longer lists it as GPU-modified).
	[[nodiscard]] bool HasInflightDownloadBytes(uint64_t vaddr, uint64_t size) const;
	// S428: HasGpuDirtyBytes + HasInflightDownloadBytes for a signal handler of a guest thread; never waits, false when a lock is held by the
	// command processor (the caller takes the ordinary drain).
	[[nodiscard]] bool TryQueryGuestFault(uint64_t vaddr, uint64_t size, bool* dirty, bool* inflight) const;
	// S419: blocks until every recorded GPU->host copy is published to the backing (GPU thread only).
	void WaitInflightDownloads();
	// S418: the physical backing holds the current bytes: no GPU write and no copy in flight. `reason` (optional): 1 = GPU-dirty, 2 = in flight.
	[[nodiscard]] bool CanReadBacking(uint64_t vaddr, uint64_t size, int* reason = nullptr);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// S406 (KYTY_PV_RANGE_GEN): see MemoryTracker::QueryPageGeneration; `clean` also needs no GPU-dirty byte range.
	[[nodiscard]] MemoryTracker::PageGeneration QueryPageGeneration(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               ProcessFaultBuffer();
	void               SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	// Queue after the range's state change becomes visible, or a drain could consume it first.
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size);
	[[nodiscard]] RangeSet TakeBdaSync();
	void               RunGarbageCollector();
	// S412 (KYTY_DIAG_RFOP)
	[[nodiscard]] uint32_t RfOpWriter(uint64_t vaddr, uint64_t size) const { return m_rfop_writers.Find(vaddr, size); }

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);

	// Eager readback (KYTY_OPT_OFF=eager_readback disables). Guest threads that poll a small
	// GPU-written value read-fault on every GPU write, and each fault used to queue behind the
	// GPU thread and drain the GPU. Once a tracker page has faulted often, a GPU write to it
	// queues its download right before the submit that carries the write; the page stays
	// protected until that download has landed, and then its completion unprotects it. A fault
	// in between waits for that one submission only.
	void NoteReadbackFault(uint64_t vaddr, uint64_t size, bool dcc);
	void NoteGpuWrite(uint64_t vaddr, uint64_t size);
	void FlushEagerReadbacks();
	[[nodiscard]] bool WaitEagerReadback(uint64_t vaddr, uint64_t size, bool dcc = false, SyncCensus::Probe* probe = nullptr);
	[[nodiscard]] bool WaitEagerReadbackPages(uint64_t vaddr, uint64_t size, SyncCensus::Probe* probe = nullptr);
	void               NoteEagerWait(SyncCensus::Probe& probe, uint64_t tick);
	static void        NoteEagerProbe(const SyncCensus::Probe& probe);
	static void        LogDccFallback(uint64_t address, uint64_t size, DccFillPublish::Why why);
	// KYTY_DCC_KNOWN: the GPU-modified pieces of a range, each with the fill value that describes all of it.
	struct KnownFillPiece {
		uint64_t address;
		uint64_t size;
		uint32_t value;
	};
	struct KnownFillPlan {
		bool                        active = false; // every piece is a known fill: the range can be published without a copy
		std::vector<KnownFillPiece> pieces;
	};
	[[nodiscard]] KnownFillPlan PlanKnownFill(uint64_t vaddr, uint64_t size);
	void                        PublishKnownFill(const KnownFillPlan& plan);
	void                        ShadowCompareKnownFill(const KnownFillPlan& plan);
	[[nodiscard]] bool          HasPendingDownload(uint64_t vaddr, uint64_t size);
	void LogSyncReadback(SyncCensus::Site site, uint64_t vaddr, uint64_t size, uint64_t wait_ns,
	                     const SyncCensus::Probe& probe, bool eager);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	GpuDirtyRanges                                    m_gpu_modified_ranges;
	InflightRanges                                    m_inflight_download_ranges;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	MemoryTracker                                     m_memory_tracker;
	std::mutex                                        m_bda_sync_mutex;
	RangeSet                                          m_bda_sync;
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;

	struct EagerPage {
		uint64_t tick = 0; // submission carrying the download; 0 when none is in flight
		uint64_t seq  = 0; // GPU writes seen for the page; a newer write keeps it protected
	};
	std::unordered_map<uint64_t, uint32_t> m_readback_fault_counts; // GPU thread
	std::vector<uint64_t>                  m_eager_pending;          // GPU thread
	std::mutex                             m_eager_mutex;
	std::unordered_map<uint64_t, EagerPage> m_eager_pages; // hot tracker pages, m_eager_mutex
	GpuWriteLedger m_write_ledger; // KYTY_DCC_KNOWN != 0 only
	struct ShadowFill {
		uint64_t vaddr = 0;
		uint64_t size  = 0;
		uint32_t value = 0;
	};
	ShadowFill m_shadow_fill; // GPU thread
	// S398 (KYTY_DIAG_SYNC=1), GPU thread only.
	SyncCensus::TickPages     m_sync_ticks;
	RfOp::WriterOps           m_rfop_writers;   // S412: opcode of the newest GPU writer per page (KYTY_DIAG_RFOP)
	SyncCensus::TickPages     m_sync_cmds;      // S404: draws + dispatches recorded when each page was last marked GPU-written
	SyncCensus::TickPages     m_sync_seq;       // S402: write sequence number per page (spec log)
	uint64_t                  m_sync_seq_next = 0;
	std::unordered_map<uint64_t, uint64_t> m_spec_last_seq; // S402: per spec-read range, the write sequence seen by the previous read
	SyncCensus::RepeatTracker m_sync_repeat;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
