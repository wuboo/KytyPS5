#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UNIFIEDMEMORY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UNIFIEDMEMORY_H_

#include "common/common.h"
#include "graphics/host_gpu/memory/gpuPageTable.h"
#include "graphics/host_gpu/memory/uma3Policy.h"
#include "graphics/host_gpu/memory/umaBarrierDiag.h"
#include "graphics/host_gpu/memory/umaChunks.h"
#include "graphics/host_gpu/memory/umaSync.h"
#include "graphics/host_gpu/memory/writeFenceTable.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "kernel/guestMemoryArena.h"

#include <vulkan/vulkan.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;

// UMA (KYTY_UMA=1): the guest memory arena is imported into the GPU and shaders touch it in place.
// This subsystem owns everything about that import:
//  - the imported VkDeviceMemory / VkBuffer and its device address,
//  - the guest-page -> device-address tables (the CPU mirror of the mappings, the published
//    two-level GpuPageTable, deferred unmaps),
//  - the WriteFenceTable that says which submit may have written which guest range,
//  - the CPU read API (WaitWriters / ReadForCpu) and the buffer window API (View).
// It subscribes to the kernel's IGuestMappingListener; the listener only queues events, which are
// applied on the GPU thread (Prepare). Clients (BufferCache, shader specialization reads, EOP)
// call the methods below and never see the kernel arenas directly.
class UnifiedMemory final: public LibKernel::Memory::IGuestMappingListener {
public:
	// The page granule the shader-visible flat table uses (BufferCache::CACHING_PAGEBITS).
	static constexpr uint32_t PageBits = 14;
	static constexpr uint64_t PageSize = uint64_t {1} << PageBits;

	// Called with the guest range [first, last) (page aligned) whose shader-table entries must be
	// rewritten; the callee reads the new entries through PageEntry / SentinelAddress.
	using PublishFn = std::function<void(uint64_t first, uint64_t last)>;

	// The window of the imported arena that backs a guest range: a range inside one chunk.
	struct Window {
		vk::Buffer buffer;
		Buffer*    wrapper = nullptr; // non-owning Buffer around `buffer` (the chunk)
		uint64_t   offset  = 0;       // byte offset in `buffer`
	};

	UnifiedMemory(GraphicContext& graphics, CommandScheduler& scheduler);
	~UnifiedMemory() override;
	KYTY_CLASS_NO_COPY(UnifiedMemory);

	[[nodiscard]] bool Active() const noexcept { return m_active; }
	// Stage 3 (KYTY_UMA=3): every buffer is a window of the arena and the buffer tracker only
	// watches the pages of images. Clients route by this flag once, never by per-site checks.
	[[nodiscard]] bool Stage3() const noexcept { return m_stage3; }
	// Bytes of guest memory imported into the GPU so far (chunks are imported on first use).
	[[nodiscard]] uint64_t ImportedBytes() const noexcept {
		return m_active ? m_imported_bytes.load(std::memory_order_relaxed) : 0;
	}

	// --- IGuestMappingListener: kernel threads, under the address-space lock; queue only ---
	void OnMapped(uint64_t vaddr, uint64_t size, LibKernel::Memory::GuestArenaId arena,
	              uint64_t offset) override;
	void OnUnmapped(uint64_t vaddr, uint64_t size) override;
	// Any thread: the guest range became GPU-visible (RenderContext::MapMemory). The chunks that
	// back it are imported on the GPU thread (Prepare / SyncMappings), before any submit can use
	// the range; a fault-time import would lose the stores of the first dispatch that hits it.
	void OnGpuVisible(uint64_t vaddr, uint64_t size);

	// --- GPU thread ---
	// Before recording a submit that may touch guest memory: applies the queued mapping changes
	// (publish is called for every range whose shader-table entries changed) and records the tick
	// the CPU must wait for before it reads guest memory.
	// writes_through_pointers: the submit holds a shader that stores through guest pointers (BDA).
	// Only those submits become the conservative "unknown range" writer (stage 3); stages 1-2 keep
	// treating every submit with direct access that way.
	void Prepare(const PublishFn& publish, bool writes_through_pointers = true);
	// Applies the queued mapping changes now (a window request must see mappings made since the
	// last Prepare). Cheap when nothing is queued.
	void SyncMappings(const PublishFn& publish);

	// Shader-table entry for a flat-table page that lies entirely in one imported chunk, 0 when it
	// does not (the caller then falls back to a shadow buffer or the fault path). A page of a chunk
	// that is not imported yet is 0 too: the first shader access faults and ImportForPage imports it.
	[[nodiscard]] vk::DeviceAddress PageEntry(uint64_t page_vaddr) const;
	// GPU thread, fault path: imports the chunk behind the page (when the arena can back it) and
	// republishes the entries of that chunk. True when the page now has an entry.
	[[nodiscard]] bool ImportForPage(uint64_t page_vaddr, const PublishFn& publish);
	// Page with nothing mapped at all: shaders reach the zeroed sentinel page (no guest faults on
	// Apple GPUs). 0 when the page is partly mapped or UMA is off.
	[[nodiscard]] vk::DeviceAddress UnmappedEntry(uint64_t page_vaddr);
	[[nodiscard]] vk::DeviceAddress SentinelAddress() const noexcept { return m_sentinel_gpu; }
	// Counts a published unmapped (sentinel) page / a batch of rewritten entries.
	uint64_t CountUnmappedBda();
	void     CountPublished(uint64_t pages, uint64_t sentinel_pages);

	// Window of the arena for [vaddr, vaddr + size) when one contiguous run inside one chunk backs
	// all of it; imports the chunk on first use (GPU thread, before the submit that uses it). A
	// range over a chunk boundary, or in a chunk whose import failed, is not served: the copy path
	// takes it. With is_written the range is recorded as written by the submit being recorded.
	[[nodiscard]] bool View(uint64_t vaddr, uint64_t size, bool is_written, Window* window);
	void               CountViewFallback() { ++m_view_fallbacks; }
	// True when one contiguous run inside one non-failed chunk backs the whole range (View without
	// side effects: nothing is imported).
	[[nodiscard]] bool Covers(uint64_t vaddr, uint64_t size) const;
	// True when the chunk that backs the whole range is imported into the GPU (no side effects).
	[[nodiscard]] bool RangeImported(uint64_t vaddr, uint64_t size) const;
	// Records a GPU writer of the range for the submit being recorded (fills, copies, labels).
	void RecordWriter(uint64_t vaddr, uint64_t size);
	// Records a writer that completes at a given tick (an EOP label written when `tick` retires).
	void RecordWriterAt(uint64_t vaddr, uint64_t size, uint64_t tick);

	// Why a range that the arena could back still took the copy path (stage 3). Anything but
	// Texel / ImageSync / Uncovered is a bug; the counters are in the log summary.
	enum class Legacy { Texel, ImageSync, Uncovered, Other };
	// Chunk statistics for the log (any thread).
	[[nodiscard]] uint32_t ChunksImported() const noexcept {
		return m_chunks_imported.load(std::memory_order_relaxed);
	}
	[[nodiscard]] uint32_t ChunkCount() const noexcept { return m_layout.Count(); }
	void CountLegacy(Legacy reason, uint64_t vaddr, uint64_t size);
	void CountReadbackSkipped() { m_rb_skipped.fetch_add(1, std::memory_order_relaxed); }
	void CountWatchersDropped(uint64_t buffers) { m_watchers_dropped += buffers; }

	// --- any thread ---
	// Waits for every writer of the range (all ranges when size is 0) and of the conservative
	// "any submit with direct access" tick. No copy is involved: afterwards the arena holds the
	// bytes. A no-op inside a deferred operation (the tick would never retire).
	void WaitWriters(uint64_t vaddr = 0, uint64_t size = 0);
	// True while a writer of the range (or an unknown-range BDA writer) has not retired: the CPU
	// backing may be older than what the GPU is about to produce. Replaces "GPU-modified" tracking.
	[[nodiscard]] bool HasPendingWriters(uint64_t vaddr, uint64_t size);
	// WaitWriters for a command-processor read (IB, WAIT_REG_MEM, COPY_DATA source, indirect
	// arguments): same wait, counted separately (uma_cp_waits).
	void WaitWritersForCp(uint64_t vaddr, uint64_t size);
	// WaitWriters + read from the arena alias. False when the range is not backed by the arena.
	[[nodiscard]] bool ReadForCpu(uint64_t vaddr, void* data, uint64_t size);
	// KYTY_UMA=shadow: repeats a CPU read of `vaddr` after a full drain and counts differences.
	void ShadowCheckRead(uint64_t vaddr, uint64_t size);

	[[nodiscard]] static bool ShadowCompareRequested();

	// Names the caller of the WaitWriters calls made in its scope (this thread), for the
	// uma_wait_<source>_<reason> counters. Nests: the innermost scope wins.
	class WaitSourceScope {
	public:
		explicit WaitSourceScope(Uma3::WaitSource source);
		~WaitSourceScope();
		KYTY_CLASS_NO_COPY(WaitSourceScope);

	private:
		Uma3::WaitSource m_previous;
	};

private:
	struct VisibleRange {
		uint64_t vaddr;
		uint64_t size;
	};
	using Sync = UmaPageSync<PageBits>;
	using Table = GpuPageTable<44, 14>;

	bool InitImport();
	// Imports chunk `index` (GPU thread; mutex inside). True when the chunk is usable afterwards.
	// A failed import marks the chunk Failed: its ranges take the copy path, never retried.
	bool EnsureChunk(uint32_t index);
	// The chunk a window [offset, offset + size) is cut from: an imported chunk that holds it (the
	// cell's own, else the overlap alias before it), otherwise the cell's own (to be imported).
	// False when no single chunk holds the whole range.
	[[nodiscard]] bool ResolveChunk(uint64_t offset, uint64_t size, UmaChunks::Place* place) const;
	// GPU thread: imports the chunks of the queued GPU-visible ranges.
	void ImportVisible(const std::vector<VisibleRange>& visible);
	// Device address of [offset, offset + size) of the arena, 0 when not in one imported chunk.
	[[nodiscard]] vk::DeviceAddress DeviceAddressOf(uint64_t offset, uint64_t size) const;
	// Queues the chunk's mapped pages for republication (applied by the next ApplyEvents).
	void QueueRepublish(uint32_t index);
	// Memory budget of the device heaps (MoltenVK: currentAllocatedSize / recommendedMaxWorkingSetSize).
	void LogDeviceBudget(const char* when);
	void Shutdown();
	void ApplyEvents(const PublishFn& publish);
	// Rewrites the 4 KB entries of [first, last) from the current mapping mirror (GPU thread).
	void Mirror(uint64_t first, uint64_t last);
	// Timelines (bit mask) a reader of the range must still wait on.
	// reason (optional) receives why a pending wait is needed; skipped_by_narrow (optional) is set
	// when KYTY_OPT_ON=bda_wait_narrow removed the only reason to wait.
	[[nodiscard]] uint32_t PendingTimelines(uint64_t vaddr, uint64_t size, WriteFenceTable::Ticks* ticks,
	                                        Uma3::WaitReason* reason = nullptr,
	                                        bool* skipped_by_narrow = nullptr);
	[[nodiscard]] bool TimelineRetired(uint32_t timeline, uint64_t tick);
	void               TimelineWait(uint32_t timeline, uint64_t tick);
	void               ShadowCompare();
	// Start-up stages and the periodic [uma] counter line (own thread: it must keep printing when
	// the GPU thread or the guest is stuck).
	void LogStage(const char* what);
	void StartProgressThread();
	// One complete [uma] line; `tag` marks the last one ("FINAL signal=15").
	void LogProgress(const char* tag = "");
	// KYTY_DIAG_UMA_BARRIERS: a use of window bytes that follow a write in the same submit.
	void DiagBarrierCheck(uint64_t vaddr, uint64_t size, bool is_written);

	GraphicContext&   m_graphics;
	CommandScheduler& m_scheduler;

	bool                         m_active = false;
	bool                         m_stage3 = false;
	LibKernel::Memory::GuestArenaInfo m_arena;

	// Lazy chunked import. The slots are read lock-free (PageEntry, Covers); a chunk's objects are
	// written under m_chunk_mutex before its `gpu` / `state` are published with release.
	struct ChunkSlot {
		std::atomic<uint64_t> gpu {0}; // device address of the chunk start, 0 = not imported
		std::atomic<uint8_t>  state {static_cast<uint8_t>(UmaChunks::State::NotImported)};
	};
	struct ChunkObjects {
		vk::DeviceMemory        memory;
		vk::Buffer              buffer;
		std::unique_ptr<Buffer> wrapper; // wraps `buffer`, non-owning
	};
	UmaChunks::Layout              m_layout;
	uint64_t                       m_import_alignment = UmaChunks::MinImportAlignment;
	std::unique_ptr<ChunkSlot[]>   m_slots;
	std::mutex                     m_chunk_mutex;
	std::vector<ChunkObjects>      m_chunk_objects; // m_chunk_mutex; one per chunk
	std::vector<uint32_t>          m_republish;     // m_event_mutex: imported chunks to publish
	std::vector<VisibleRange>      m_visible;       // m_event_mutex: ranges to import eagerly
	std::atomic<uint64_t>          m_imported_bytes {0};
	std::atomic<uint32_t>          m_chunks_imported {0};
	std::atomic<uint32_t>          m_chunks_failed {0};
	std::atomic<uint64_t>          m_chunk_import_ns {0};
	std::atomic<uint64_t>          m_chunk_fault_imports {0}; // imports started by a BDA fault
	std::atomic<uint64_t>          m_chunk_straddles {0};     // windows refused: over a boundary
	std::atomic<uint64_t>          m_chunk_map_imports {0};   // imports started by a GPU-visible mapping
	std::atomic<uint64_t>          m_visible_ranges {0};      // GPU-visible ranges seen
	std::atomic<uint64_t>          m_visible_unmapped {0};    // ...of which a part had no backing yet
	// KYTY_DIAG_UMA_BARRIERS (GPU thread writes the tracker; the counters are atomics)
	UmaBarrierDiag::Mode           m_diag_barriers = UmaBarrierDiag::Mode::Off;
	UmaBarrierDiag::SameSubmitTracker m_diag_tracker;
	std::atomic<uint64_t>          m_diag_raw_missing {0}; // read after a write, no barrier between
	std::atomic<uint64_t>          m_diag_waw_missing {0}; // write after a write, no barrier between
	std::atomic<uint64_t>          m_diag_covered {0};     // same-submit reuse with a barrier between
	std::atomic<uint64_t>          m_diag_inserted {0};    // KYTY_DIAG_UMA_BARRIERS=2: barriers added
	std::unique_ptr<Buffer>      m_sentinel; // one zeroed page for unmapped BDA pages
	vk::DeviceAddress            m_sentinel_gpu = 0;

	std::mutex                   m_event_mutex;
	std::vector<Sync::Event>     m_events; // m_event_mutex
	std::atomic<bool>            m_events_pending {false};
	Sync                         m_sync;   // GPU thread
	std::unique_ptr<Table>       m_table;  // GPU thread: what has been published

	WriteFenceTable              m_fences; // m_fence_mutex
	std::mutex                   m_fence_mutex;
	std::atomic<uint64_t>        m_bda_tick {0};
	Uma3::UnknownWriterTick      m_unknown_writer; // GPU thread (Prepare)
	// KYTY_OPT_ON=bda_wait_narrow: ranges outside every cell ever written by a window / fill /
	// copy / label do not wait for the global pointer-writer tick.
	Uma3::WrittenCells           m_written;
	bool                         m_bda_narrow = false;
	std::atomic<uint64_t>        m_narrow_skipped {0};
	static constexpr uint32_t    WaitSources = static_cast<uint32_t>(Uma3::WaitSource::Count);
	static constexpr uint32_t    WaitReasons = static_cast<uint32_t>(Uma3::WaitReason::Count);
	std::array<std::atomic<uint64_t>, WaitSources * WaitReasons> m_wait_by_ns {};
	std::array<std::atomic<uint64_t>, WaitSources * WaitReasons> m_wait_by_n {};

	// counters (see Log summary)
	std::atomic<uint64_t> m_views {0};
	std::atomic<uint64_t> m_view_fallbacks {0};
	std::atomic<uint64_t> m_sentinel_pages {0};
	std::atomic<uint64_t> m_events_applied {0};
	std::atomic<uint64_t> m_pages_written {0};
	std::atomic<uint64_t> m_unmapped_bda {0};
	std::atomic<uint64_t> m_waits {0};
	std::atomic<uint64_t> m_cp_waits {0};
	std::atomic<uint64_t> m_rb_skipped {0};
	std::atomic<uint64_t> m_legacy_hits {0};   // arena-backable range on a path that must not copy
	std::atomic<uint64_t> m_legacy_texel {0};
	std::atomic<uint64_t> m_legacy_image {0};
	std::atomic<uint64_t> m_legacy_uncovered {0};
	uint64_t              m_watchers_dropped = 0;
	uint64_t              m_views_written    = 0;
	std::atomic<uint64_t> m_wait_ns {0};

	// Progress diagnostics (start-up hang hunt): relaxed atomics read by the [uma] thread.
	enum class Stage : int { Created, Importing, Imported, ListenerSet, FirstPrepare };
	std::atomic<int>      m_stage {static_cast<int>(Stage::Created)};
	std::atomic<uint64_t> m_events_queued {0};  // OnMapped/OnUnmapped calls (kernel threads)
	std::atomic<uint64_t> m_prepares {0};
	std::atomic<uint64_t> m_pending_unmaps {0}; // deferred unmaps not yet published
	std::atomic<uint64_t> m_last_prepare_ms {0}; // ms since construction at the last Prepare
	std::chrono::steady_clock::time_point m_t0 = std::chrono::steady_clock::now();
	std::jthread          m_progress_thread; // last member: joined first

	// Recent windows handed out (stage 3 shadow-compare re-derives each through the kernel).
	struct RecentView {
		uint64_t vaddr  = 0;
		uint64_t size   = 0;
		uint64_t offset = 0;
	};
	std::array<RecentView, 64> m_recent_views {};
	uint32_t                   m_recent_view_next = 0;
	uint32_t                   m_shadow_view_counter = 0;

	// KYTY_UMA_SHADOW=1
	std::atomic<uint64_t> m_shadow_view_mismatch {0};
	bool     m_dshadow         = false; // KYTY_UMA=shadow
	uint64_t m_dshadow_every   = 4;     // KYTY_UMA_DSHADOW_N
	std::atomic<uint64_t> m_dshadow_calls {0};
	std::atomic<uint64_t> m_dshadow_eq {0};
	std::atomic<uint64_t> m_dshadow_ne {0};
	bool     m_shadow          = false;
	uint32_t m_shadow_interval = 600;
	uint32_t m_shadow_counter  = 0;
	uint64_t m_shadow_runs     = 0;
	std::atomic<uint64_t> m_shadow_mismatch {0};
	uint64_t m_shadow_skipped  = 0;
	uint64_t m_shadow_cursor   = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UNIFIEDMEMORY_H_
