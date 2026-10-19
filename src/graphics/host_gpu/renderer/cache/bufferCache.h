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
#include "graphics/host_gpu/memory/uma3Policy.h"
#include "graphics/host_gpu/memory/unifiedMemory.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/gpuDirtyRanges.h"
#include "graphics/host_gpu/renderer/cache/gpuWriteLedger.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/rfOpDiag.h"
#include "graphics/host_gpu/renderer/syncCensus.h"

#include <array>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
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
	            TextureCache& texture_cache, UnifiedMemory& unified);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Returns false when a read fault found nothing to download.
	bool                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	// FindBuffer for a binding that ObtainBuffer will serve: in stage 3 a range the arena backs
	// needs no buffer object at all (NULL_BUFFER_ID), so none is created, registered or tracked.
	[[nodiscard]] BufferId FindBindingBuffer(uint64_t vaddr, uint64_t size);
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
	// UMA (KYTY_UMA=1): the guest direct-memory backing is imported in chunks, on first use, and the BDA
	// page table points shader accesses straight at it.
	[[nodiscard]] bool UmaActive() const noexcept { return m_unified.Active(); }
	// GPU thread, before recording a submit whose shaders use BDA: applies queued mapping changes
	// to the page table and records the tick that must retire before the CPU reads guest memory.
	void PrepareUma(bool writes_through_pointers = true);
	// Stage 3 (KYTY_UMA=3): every buffer is a window of the imported arena. True when the buffer
	// cache keeps no copies, no download state and no page watchers for arena-backed memory.
	[[nodiscard]] bool UmaStage3() const noexcept { return m_unified.Stage3(); }
	// Descriptor view of the imported backing for a range the guest backing holds in full (UMA).
	// The returned offset is the byte offset in the arena buffer; the caller aligns it down to the
	// storage-buffer alignment and passes the remainder to the shader (same as ObtainBuffer).
	// Returns false (and counts a fallback) when the range must keep the shadow-copy path.
	[[nodiscard]] bool TryObtainUmaView(uint64_t vaddr, uint64_t size, bool is_written,
	                                    bool is_texel_buffer, vk::Buffer* buffer, uint64_t* offset);
	// Fault path: caches a faulting range; with UMA, pages the guest has not mapped at all get
	// the sentinel page instead (and count in uma_unmapped_bda).
	void               FindBufferForFault(uint64_t start, uint64_t size);
	void               ProcessFaultBuffer();
	void               SynchronizeBda(const RangeSet& mapped);
	void               QueueBdaSync(uint64_t vaddr, uint64_t size) {
		m_memory_tracker.QueueBdaSync(vaddr, size);
	}
	void               RunGarbageCollector();
	// S412 (KYTY_DIAG_RFOP)
	[[nodiscard]] uint32_t RfOpWriter(uint64_t vaddr, uint64_t size) const { return m_rfop_writers.Find(vaddr, size); }

private:
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const {
		const auto* buffer = m_slot_buffers.try_get(id);
		return buffer == nullptr || buffer->is_deleted;
	}

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
	// Page-table entry for one cache page: the imported backing when the page is directly usable,
	// else the covering shadow buffer, else 0 (fault path).
	[[nodiscard]] vk::DeviceAddress PageTableEntry(uint64_t page_vaddr);
	void                            PublishUmaPages(uint64_t first, uint64_t last);

	// --- stage 3 ---
	// Window of the arena for the range, or nullopt when the copy path must serve it (counted by
	// reason). Applies queued mapping changes first.
	[[nodiscard]] std::optional<std::pair<Buffer*, uint64_t>>
	     TryObtainArenaWindow(uint64_t vaddr, uint64_t size, bool is_written, bool is_texel_buffer);
	// First stage-3 use: buffer watchers that exist from before are dropped (counted).
	void EnterUmaStage3();
	// True when a copy-path buffer was ever registered over the range. Only those pages can carry
	// buffer tracker state; thread safe (write faults arrive on guest threads).
	[[nodiscard]] bool UmaTrackerCanSkip(uint64_t vaddr, uint64_t size);

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
	void SynchronizeBuffersInRange(uint64_t vaddr, uint64_t size);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);

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
	UnifiedMemory&                                    m_unified;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	GpuDirtyRanges                                    m_gpu_modified_ranges;
	InflightRanges                                    m_inflight_download_ranges;
	MemoryTracker                                     m_memory_tracker;
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
	bool                                   m_stage3_entered = false; // GPU thread
	std::mutex                             m_uma_legacy_mutex;
	Uma3::LegacyRanges                     m_uma_legacy_ranges; // m_uma_legacy_mutex
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
