#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/cpWaitDiag.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/frameTiming.h"

#include "common/logging/log.h"

#include <atomic>
#include <optional>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
}

MasterSemaphore::~MasterSemaphore() {
	if (m_semaphore != nullptr) {
		m_graphics.device.destroySemaphore(m_semaphore, nullptr);
	}
}

void MasterSemaphore::Refresh() {
	uint64_t   counter = 0;
	const auto result  = m_graphics.device.getSemaphoreCounterValue(m_semaphore, &counter);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	auto known = m_gpu_tick.load(std::memory_order_acquire);
	while (known < counter &&
	       !m_gpu_tick.compare_exchange_weak(known, counter, std::memory_order_release,
	                                         std::memory_order_relaxed)) {
	}
}

void MasterSemaphore::Wait(uint64_t tick) {
	if (IsFree(tick)) {
		return;
	}
	Refresh();
	if (IsFree(tick)) {
		return;
	}

	vk::SemaphoreWaitInfo wait_info {};
	wait_info.semaphoreCount = 1;
	wait_info.pSemaphores    = &m_semaphore;
	wait_info.pValues        = &tick;

	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::GpuWaits,
	                                         Common::DebugCounters::Counter::GpuWaitNs);
	// S132: only the command-processor thread's blocking time is cp_wait_gpu_ns.
	std::optional<FrameTiming::ScopedCounterNs> cp_wait;
	if (GuestGpu::IsGpuThread()) cp_wait.emplace(FrameTiming::Counter::CpWaitGpuNs);
	// S413 (KYTY_DIAG_CP_WAIT=1): the site that asked (read before the Wait phase replaces it), the ticks the GPU is behind, the time.
	const bool     diag = GuestGpu::IsGpuThread() && CpWait::Enabled();
	uint32_t       diag_site = 0;
	uint64_t       diag_lag = 0;
	int64_t        diag_begin = 0;
	if (diag) {
		diag_site  = CpSampler::CpSite().load(std::memory_order_relaxed);
		diag_lag   = tick > m_gpu_tick.load(std::memory_order_relaxed) ? tick - m_gpu_tick.load(std::memory_order_relaxed) : 0;
		diag_begin = FrameTiming::SteadyNowNs();
	}
	{
		CpPhases::Scope phase(CpPhases::Phase::Wait);
		const auto result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
		EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
		Refresh();
	}
	if (diag) {
		const auto     ns    = FrameTiming::ElapsedNs(diag_begin, FrameTiming::SteadyNowNs());
		const auto     cause = CpWait::CurrentCause();
		FrameTiming::Add(static_cast<FrameTiming::Counter>(static_cast<uint32_t>(FrameTiming::Counter::CpWaitCauseRfGpuNs) +
		                                                   static_cast<uint32_t>(cause)),
		                 ns);
		if (ns / 1000 >= CpWait::kMinWaitUs) {
			static std::atomic<uint64_t> logged {0};
			CpWait::Event                event;
			event.flip = FrameTiming::GuestFlips();
			if (CpWait::ShouldLog(logged.fetch_add(1, std::memory_order_relaxed), event.flip, ns / 1000)) {
				event.opcode  = RfOp::CurrentPacket().opcode != RfOp::kNoOpcode ? RfOp::CurrentPacket().opcode
				                                                                : CpSampler::CpOpcode().load(std::memory_order_relaxed);
				event.site    = RfOp::SiteLabel(diag_site);
				event.cause   = cause;
				event.wait_us = ns / 1000;
				event.lag     = diag_lag;
				LOGF("%s\n", CpWait::FormatLine(event).c_str());
			}
		}
	}
}

} // namespace Libs::Graphics
