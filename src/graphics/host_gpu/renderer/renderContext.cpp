#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/benchTrace.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdlib>
#if defined(__APPLE__)
#include <execinfo.h>
#endif

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
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

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
#if defined(__APPLE__)
	// A guard page is unwatched memory kept read-only in front of a watched page (see
	// PageManager). Any access to it releases the watched run behind it, which drops the guard.
	if (m_page_manager.IsGuardPage(fault_vaddr)) {
		const auto page_size = m_page_manager.GetPageSize();
		const auto next      = (fault_vaddr & ~(page_size - 1)) + page_size;
		const auto run_end   = m_page_manager.WatchedRunEnd(next, 64ull << 20u);
		if (run_end > next && IsMapped(next, run_end - next)) {
			m_buffer_cache.InvalidateMemory(next, run_end - next);
			m_texture_cache.InvalidateMemory(next, run_end - next);
		}
		static std::atomic<uint64_t> guard_faults {0};
		const auto n = guard_faults.fetch_add(1, std::memory_order_relaxed) + 1;
		if ((n & (n - 1)) == 0) {
			LOGF("[rosetta-guard] guard faults=%" PRIu64 "\n", n);
		}
		// If the caches kept the next page watched, the guard stays and the access would fault
		// forever; report it instead of spinning.
		return !m_page_manager.IsGuardPage(fault_vaddr);
	}
#endif
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	const bool on_gpu_thread = GuestGpu::IsGpuThread();
	if (access == PageFaultAccess::Write) {
		FrameTiming::Add(on_gpu_thread ? FrameTiming::Counter::WriteFaultsGpu
		                               : FrameTiming::Counter::WriteFaults);
		uint64_t invalidate_size = fault_size;
#if defined(__APPLE__)
		// Rosetta aborts the process ("unexpectedly need to EmulateForward on a synchronous
		// exception") when a 256-bit AVX store that crosses from a writable page into a
		// write-protected one faults mid-block. Release the whole run of watched pages that
		// follows the fault, so a copy that has started writing does not cross into a
		// protected page further on.
		static const uint64_t release_ahead = [] {
			const char* value = std::getenv("KYTY_ROSETTA_RELEASE_AHEAD_MB");
			// Off by default (see KYTY_ROSETTA_GUARDS): widening a write fault to the whole
			// watched run turns one fault into a readback per GPU-dirty 4 MiB region.
			return (value != nullptr ? std::strtoull(value, nullptr, 10) : 0ull) << 20u;
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
#if defined(__APPLE__)
		{
			// Diagnostic: a store that faulted near the end of its page will cross into the next
			// page on retry; if that page is still protected, Rosetta aborts.
			const auto page_size = m_page_manager.GetPageSize();
			const auto next      = (fault_vaddr & ~(page_size - 1)) + page_size;
			if (next - fault_vaddr <= 32 && m_page_manager.WatchedRunEnd(next, page_size) > next) {
				static std::atomic<uint32_t> still_watched {0};
				if (still_watched.fetch_add(1, std::memory_order_relaxed) < 32) {
					LOGF("[rosetta-guard] next page still watched after write fault: fault=0x%016" PRIx64
					     " invalidated=0x%" PRIx64 " guard=%u\n",
					     fault_vaddr, invalidate_size, m_page_manager.IsGuardPage(next) ? 1u : 0u);
				}
			}
		}
#endif
	} else {
		FrameTiming::Add(on_gpu_thread ? FrameTiming::Counter::ReadbackReadFaultGpu
		                               : FrameTiming::Counter::ReadbackReadFault);
		// TEMP diagnostic (2026-09-28): DrawIndirect/DrawIndirectMulti/DispatchIndirect's
		// USE_THREAD_DIMENSIONS mode/SetPredication were all checked and ruled out as the source
		// of this game's ~97s/165s of readback_wait_ns in the menu scene -- log what's actually
		// running (BenchTrace's "current operation", set before draw/dispatch/fill/copy) and a
		// raw backtrace (execinfo, not a debugger -- lldb doesn't attach reliably here) to find
		// the real one instead of guessing from function names again.
		if (on_gpu_thread) {
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 40) {
				const char* kind  = "none";
				uint64_t    hash0 = 0;
				uint64_t    hash1 = 0;
				BenchTrace::GetCurrent(kind, hash0, hash1);
				LOGF("[bench-fault] addr=0x%016" PRIx64 " current=%s hash0=0x%016" PRIx64
				     " hash1=0x%016" PRIx64 "\n",
				     fault_vaddr, kind, hash0, hash1);
#if defined(__APPLE__)
				void* frames[16];
				const int n = backtrace(frames, 16);
				char** syms = backtrace_symbols(frames, n);
				if (syms != nullptr) {
					for (int i = 0; i < n; i++) {
						LOGF("[bench-fault]   %s\n", syms[i]);
					}
					free(syms);
				}
#endif
			}
		}
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
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
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		if (m_command_scheduler.Active()) {
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
	const auto epoch = g_cpu_dirty_epoch.load(std::memory_order_acquire);
	if (epoch != m_bda_synced_epoch) {
		std::shared_lock lock(m_mapped_ranges_mutex);
		m_mapped_ranges.ForEach([this](uint64_t start, uint64_t end) {
			m_buffer_cache.SynchronizeBuffersInRange(start, end - start);
		});
		m_bda_synced_epoch = epoch;
	}
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
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
