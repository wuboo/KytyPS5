#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/wfDiag.h"
#include "graphics/host_gpu/protectRing.h"
#include "graphics/host_gpu/renderer/rfOpDiag.h"
#include "graphics/host_gpu/renderer/gfDiag.h"
#include "graphics/host_gpu/renderer/syncCensus.h"
#include "kernel/memory.h"
#include "graphics/presentation/cpPhaseScope.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#include <mutex>
#include <unordered_set>

#include <pthread.h>

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_command_scheduler.EnableAsyncSubmit();
	m_command_scheduler.EnableStarveDiag(); // S410
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

namespace {
// S443 (KYTY_DIAG_WRITE_FAULT): id and name of the faulting thread, read once per thread.
uint64_t WfThreadId() {
#if defined(__APPLE__)
	static thread_local const uint64_t tid = static_cast<uint64_t>(pthread_mach_thread_np(pthread_self()));
#else
	static thread_local const uint64_t tid = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(pthread_self()));
#endif
	return tid;
}

const char* WfThreadName() {
	static thread_local char name[64] = {};
	static thread_local bool done     = false;
	if (!done) {
		(void)pthread_getname_np(pthread_self(), name, sizeof(name));
		done = true;
	}
	return name;
}

// S444 (KYTY_DIAG_WRITE_FAULT): a write fault on a page kept unprotected for its flip: print who put the host protection back (ring of protection changes).
void StickyRefaultLine(uint64_t page, uint32_t flip, uint32_t sticky_flip) {
	if (ProtectRing::LinesPrinted().fetch_add(1, std::memory_order_relaxed) >= ProtectRing::kMaxLines) {
		return;
	}
	ProtectRing::Entry why[ProtectRing::kMaxWhy];
	const auto         n = ProtectRing::Find(page, why, ProtectRing::kMaxWhy);
	LOGF("%s\n", ProtectRing::FormatRefault(page, flip, sticky_flip, why, n).c_str());
}
} // namespace

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	const bool on_gpu_thread = GuestGpu::IsGpuThread();
	if (access == PageFaultAccess::Write) {
		FrameTiming::Add(on_gpu_thread ? FrameTiming::Counter::WriteFaultsGpu
		                               : FrameTiming::Counter::WriteFaults);
		// S440 (KYTY_DIAG_WRITE_FAULT): owner of the page before the invalidation untracks it, then the handler's time.
		const bool wf_diag  = WriteFault::Enabled();
			static const bool ring_on = [wf_diag] {
				if (wf_diag) {
					ProtectRing::EnabledFlag().store(true, std::memory_order_relaxed);
					FlipUnprotect::StickyRefaultObserver().store(&StickyRefaultLine, std::memory_order_relaxed);
				}
				return wf_diag;
			}();
			(void)ring_on;
		const auto wf_begin = wf_diag ? FrameTiming::SteadyNowNs() : int64_t {0};
		if (wf_diag) {
			const auto page      = fault_vaddr & ~(m_page_manager.GetPageSize() - 1);
			uint64_t   image_size = 0;
			const bool image     = m_texture_cache.DiagRegionOwner(page, &image_size);
			const bool buffer    = !m_buffer_cache.IsRegionCpuModified(page, m_page_manager.GetPageSize());
			const auto flip      = FrameTiming::GuestFlips();
			WriteFault::Finished finished;
			if (WriteFault::Pages().Note(page, flip, WriteFault::MakeOwner(buffer, image), image_size, &finished)) {
				FrameTiming::Set(FrameTiming::Counter::WfPagesDistinct, finished.distinct);
				FrameTiming::Set(FrameTiming::Counter::WfPagesEveryFlip, finished.every);
				static std::atomic<bool> wf_first {true};
				if (wf_first.exchange(false, std::memory_order_relaxed) || finished.flip % WriteFault::kTopEveryFlips == 0) {
					for (const auto& entry: WriteFault::Pages().Top(WriteFault::kTopPages)) {
						LOGF("%s\n", WriteFault::FormatLine(entry, finished.flip).c_str());
					}
					LOGF("%s\n", WriteFault::FormatRuns(finished).c_str());
					for (const auto& entry: WriteFault::Threads().Snapshot()) {
						LOGF("%s\n", WriteFault::FormatThread(entry, finished.flip).c_str());
					}
				}
			}
		}
		uint64_t invalidate_size = fault_size;
#if defined(__APPLE__)
		// Rosetta aborts the process ("unexpectedly need to EmulateForward on a synchronous
		// exception") when a 256-bit AVX store that crosses from a writable page into a
		// write-protected one faults mid-block. Release the whole run of watched pages that
		// follows the fault, so a copy that has started writing does not cross into a
		// protected page further on.
		static const uint64_t release_ahead = [] {
			const char* value = std::getenv("KYTY_ROSETTA_RELEASE_AHEAD_MB");
			return (value != nullptr ? std::strtoull(value, nullptr, 10) : 64ull) << 20u;
		}();
		if (release_ahead != 0) {
			const auto run_end = m_page_manager.WatchedRunEnd(fault_vaddr, release_ahead);
			if (run_end > fault_vaddr && IsMapped(fault_vaddr, run_end - fault_vaddr)) {
				invalidate_size = run_end - fault_vaddr;
			}
		}
		static std::atomic<uint64_t> faults {0};
		static std::atomic<uint64_t> released_pages {0};
		const auto                   n = faults.fetch_add(1, std::memory_order_relaxed) + 1;
		const auto released = released_pages.fetch_add(invalidate_size / m_page_manager.GetPageSize(),
		                                         std::memory_order_relaxed) +
		                   invalidate_size / m_page_manager.GetPageSize();
		if ((n & (n - 1)) == 0) {
			LOGF("[rosetta-release-ahead] write faults=%" PRIu64 " pages invalidated=%" PRIu64
			     "\n",
			     n, released);
		}
#endif
		m_buffer_cache.InvalidateMemory(fault_vaddr, invalidate_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, invalidate_size);
		if (wf_diag) {
			FrameTiming::Add(FrameTiming::Counter::WfN);
			const auto wf_ns = FrameTiming::ElapsedNs(wf_begin, FrameTiming::SteadyNowNs());
			FrameTiming::Add(FrameTiming::Counter::WfHandlerNs, wf_ns);
			WriteFault::Threads().NoteWrite(WfThreadId(), WfThreadName(), wf_ns);
		}
	} else {
		FrameTiming::Add(on_gpu_thread ? FrameTiming::Counter::ReadbackReadFaultGpu
		                               : FrameTiming::Counter::ReadbackReadFault);
		const SyncCensus::SiteScope site_tag(on_gpu_thread ? SyncCensus::Site::RfGpu : SyncCensus::Site::RfGuest);
		// S412 (KYTY_DIAG_RFOP): who on the command processor thread reads the page, and what it would have seen before the drain.
		const bool rf_op = on_gpu_thread && RfOp::Enabled();
		RfOp::Event rf_event;
		uint64_t    eager_before = 0;
		int64_t     read_begin   = 0;
		if (rf_op) {
			rf_event.stale.ok    = LibKernel::Memory::TryReadBacking(fault_vaddr, rf_event.stale.data, sizeof(rf_event.stale.data));
			// S413: state of the bytes and of the page before the drain, and the breadcrumb of the code that is reading.
			rf_event.dirty      = m_buffer_cache.HasGpuDirtyBytes(fault_vaddr, sizeof(rf_event.stale.data));
			rf_event.page_dirty = m_buffer_cache.IsRegionGpuModified(fault_vaddr & ~uint64_t {4095}, 4096);
			rf_event.site       = RfOp::SiteLabel(CpSampler::CpSite().load(std::memory_order_relaxed));
			eager_before        = FrameTiming::Get(FrameTiming::Counter::ReadbackEagerHits);
			read_begin          = FrameTiming::SteadyNowNs();
		}
		// S421 (KYTY_DIAG_GUEST_FAULT): a read fault of a guest thread; was the page read from the backing by the CP in this flip?
		const bool gf = !on_gpu_thread && access == PageFaultAccess::Read && GuestFault::Enabled();
		GuestFault::Event gf_event;
		int64_t           gf_begin = 0;
		uint64_t          gf_eager_before = 0;
		if (gf) {
			gf_event.flip       = FrameTiming::GuestFlips();
			gf_event.addr       = fault_vaddr;
			gf_event.dirty      = m_buffer_cache.HasGpuDirtyBytes(fault_vaddr, 16);
			gf_event.page_dirty = m_buffer_cache.IsRegionGpuModified(fault_vaddr & ~uint64_t {4095}, 4096);
			const auto hit      = GuestFault::Reads().Find(fault_vaddr);
			gf_event.after_backing = GuestFault::AfterBacking(hit, gf_event.flip);
			gf_event.backing_site  = hit.found ? hit.site : GuestFault::Site::None;
			gf_eager_before        = FrameTiming::Get(FrameTiming::Counter::ReadbackEagerHits);
			gf_begin               = FrameTiming::SteadyNowNs();
		}
		const bool wf_threads = !on_gpu_thread && access == PageFaultAccess::Read && WriteFault::Enabled();
		const auto wf_read_begin = wf_threads ? FrameTiming::SteadyNowNs() : int64_t {0};
		const bool downloaded = m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
		if (wf_threads) {
			WriteFault::Threads().NoteRead(WfThreadId(), WfThreadName(), FrameTiming::ElapsedNs(wf_read_begin, FrameTiming::SteadyNowNs()));
		}
		if (gf) {
			const auto wait_ns = FrameTiming::ElapsedNs(gf_begin, FrameTiming::SteadyNowNs());
			gf_event.wait_us   = wait_ns / 1000;
			gf_event.eager     = FrameTiming::Get(FrameTiming::Counter::ReadbackEagerHits) != gf_eager_before;
			FrameTiming::Add(FrameTiming::Counter::GfN);
			FrameTiming::Add(FrameTiming::Counter::GfWaitNs, wait_ns);
			if (gf_event.after_backing) {
				FrameTiming::Add(FrameTiming::Counter::GfAfterBackingN);
			}
			static std::atomic<uint64_t> gf_logged {0};
			if (GuestFault::ShouldLog(gf_logged.fetch_add(1, std::memory_order_relaxed), GuestFault::InWindow(gf_event.flip))) {
				const auto& ctx = RfOp::FaultCtxRef();
				uint64_t stack_lo = 0, stack_hi = 0, frames[RfOp::kMaxFrames];
				if (RfOp::CurrentStackBounds(&stack_lo, &stack_hi)) {
					gf_event.bt_text = RfOp::FormatBacktrace(frames, RfOp::WalkFrames(ctx, stack_lo, stack_hi, frames));
				}
				LOGF("%s\n", GuestFault::FormatLine(gf_event).c_str());
			}
		}
		if (rf_op) {
			const auto wait_ns = FrameTiming::ElapsedNs(read_begin, FrameTiming::SteadyNowNs());
			rf_event.wait_us   = wait_ns / 1000;
			rf_event.eager     = FrameTiming::Get(FrameTiming::Counter::ReadbackEagerHits) != eager_before;
			rf_event.fresh.ok  = LibKernel::Memory::TryReadBacking(fault_vaddr, rf_event.fresh.data, sizeof(rf_event.fresh.data));
			rf_event.flip      = FrameTiming::GuestFlips();
			rf_event.packet    = RfOp::CurrentPacket();
			rf_event.addr      = fault_vaddr;
			rf_event.writer_op = m_buffer_cache.RfOpWriter(fault_vaddr, fault_size);
			FrameTiming::Add(static_cast<FrameTiming::Counter>(static_cast<uint32_t>(FrameTiming::Counter::RfOpWaitRegMemN) +
			                                                   RfOp::SlotOf(rf_event.packet.opcode)));
			FrameTiming::Add(FrameTiming::Counter::RfOpWaitNs, wait_ns);
			if (rf_event.packet.opcode == RfOp::kOpDrawIndexIndirect) {
				FrameTiming::Add(FrameTiming::Counter::RfOpDrawIndexIndirectN);
				FrameTiming::Add(FrameTiming::Counter::RfOpDrawIndexIndirectWaitNs, wait_ns);
			}
			static std::atomic<uint64_t> logged {0};
			// S486 DIAG: slow (>= 2 ms) faults of the landing phase always logged, to stdout (the bench runs with --printf-direction Silent).
			static std::atomic<uint64_t> slow_logged {0};
			const bool slow_landing = rf_event.wait_us >= 2000 && rf_event.flip >= 4400 && rf_event.flip <= 5300 && slow_logged.fetch_add(1, std::memory_order_relaxed) < 400;
			if (slow_landing || RfOp::ShouldLog(logged.fetch_add(1, std::memory_order_relaxed), rf_event.flip)) {
				const auto& ctx = RfOp::FaultCtxRef();
				rf_event.pc_text = RfOp::ResolvePc(ctx.pc);
				uint64_t stack_lo = 0, stack_hi = 0, frames[RfOp::kMaxFrames];
				if (RfOp::CurrentStackBounds(&stack_lo, &stack_hi)) {
					rf_event.bt_text = RfOp::FormatBacktrace(frames, RfOp::WalkFrames(ctx, stack_lo, stack_hi, frames));
				}
#if defined(_WIN32)
				if (slow_landing) { // S486 DIAG: image-relative return addresses (map file: kyty_emulator_clang_lld_link.map)
					const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
					std::string text;
					char        one[40];
					std::snprintf(one, sizeof(one), "pc=%llx", static_cast<unsigned long long>(ctx.pc - base));
					text += one;
					// no unwinder: the stack words that point into the image's .text (0x1000..0x1410000) are the return addresses and some stale values
					const auto* sp = reinterpret_cast<const uint64_t*>(ctx.rsp);
					uint32_t    found = 0;
					for (uint32_t w = 0; w < 512 && found < 16; w++) {
						const uint64_t v = sp[w];
						if (v >= base + 0x1000 && v < base + 0x1410000) {
							std::snprintf(one, sizeof(one), "|%llx", static_cast<unsigned long long>(v - base));
							text += one;
							found++;
						}
					}
					rf_event.bt_text = text;
				}
#endif
				if (slow_landing) {
					std::printf("%s\n", RfOp::FormatLine(rf_event).c_str());
					std::fflush(stdout);
				} else {
					LOGF("%s\n", RfOp::FormatLine(rf_event).c_str());
				}
			}
		}
		if (!downloaded && access != PageFaultAccess::Execute) {
			// Nothing to download and the tracker wants the page readable, yet it faulted:
			// the protection was lost outside the tracker. Put back what the tracker expects
			// instead of refaulting forever.
			int host_protection = -1;
			if (m_page_manager.RestoreReadAccess(fault_vaddr, &host_protection)) {
				static std::mutex                   mutex;
				static std::unordered_set<uint64_t> logged;
				std::lock_guard                     lock(mutex);
				const auto page = fault_vaddr & ~(m_page_manager.GetPageSize() - 1);
				if (logged.size() < 4096 && logged.insert(page).second) {
					LOGF("[read-fault-restore] addr=0x%016" PRIx64 " host_prot=%d\n", page,
					     host_protection);
				}
			}
		}
		if (access == PageFaultAccess::Execute) {
			// Fetching code from a watched page is neither a read nor a write the caches can
			// satisfy, and a read leaves the protection in place, so the fetch would fault
			// again forever. The guest owns code it runs: stop watching the page.
			m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
			m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		}
	}
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	if (!m_bda_logged) {
		Log::WriteToConsoleAndLog("GPU: using buffer device address (BDA) shader memory access.\n");
		m_bda_logged = true;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.SynchronizeBda(m_mapped_ranges);
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	CpPhases::Scope phase(CpPhases::Phase::Gc);
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
