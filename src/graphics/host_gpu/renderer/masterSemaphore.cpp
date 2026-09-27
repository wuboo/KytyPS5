#include "graphics/host_gpu/renderer/masterSemaphore.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/benchTrace.h"

#include <chrono>
#include <cinttypes>
#include <cstdlib>

namespace Libs::Graphics {

MasterSemaphore::MasterSemaphore(GraphicContext& graphics): m_graphics(graphics) {
	vk::SemaphoreTypeCreateInfo type_info {};
	type_info.semaphoreType = vk::SemaphoreType::eTimeline;
	type_info.initialValue  = 0;

	vk::SemaphoreCreateInfo create_info {};
	create_info.pNext = &type_info;

	const auto result = m_graphics.device.createSemaphore(&create_info, nullptr, &m_semaphore);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_semaphore == nullptr);
	if (const char* value = std::getenv("KYTY_BENCH_STALL_WATCH");
	    value == nullptr || value[0] != '0') {
		m_stall_watch = std::jthread([this](std::stop_token stop) { StallWatch(stop); });
	}
}

void MasterSemaphore::StallWatch(std::stop_token stop) {
	uint64_t last_value = UINT64_MAX;
	uint64_t reported   = UINT64_MAX;
	uint32_t idle_polls = 0;
	while (!stop.stop_requested()) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		uint64_t value = 0;
		if (m_graphics.device.getSemaphoreCounterValue(m_semaphore, &value) !=
		    vk::Result::eSuccess) {
			continue;
		}
		const bool pending = CurrentTick() > value + 1;
		idle_polls         = (pending && value == last_value) ? idle_polls + 1 : 0;
		last_value         = value;
		// Only the renderer's scheduler has large ticks; the swapchain's stay small.
		if (idle_polls == 10 && value > 100000 && reported != value) {
			reported = value;
			LOGF("[bench-stall] GPU has not finished tick %" PRIu64
			     " for 5 s (recorded up to %" PRIu64 ")\n",
			     value + 1, CurrentTick() - 1);
			BenchTrace::Dump(value + 1);
		}
	}
}

MasterSemaphore::~MasterSemaphore() {
	if (m_stall_watch.joinable()) {
		m_stall_watch.request_stop();
		m_stall_watch.join();
	}
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

	// Bench diagnostic: report waits that take unusually long, then keep waiting.
	constexpr uint64_t report_ns = 5'000'000'000ull;
	auto               result    = m_graphics.device.waitSemaphores(&wait_info, report_ns);
	if (result == vk::Result::eTimeout) {
		uint64_t gpu_value = 0;
		(void)m_graphics.device.getSemaphoreCounterValue(m_semaphore, &gpu_value);
		LOGF("[bench-wait] timeline wait >5 s: waiting for tick %" PRIu64 ", GPU reached %" PRIu64
		     "\n",
		     tick, gpu_value);
		BenchTrace::Dump(gpu_value + 1);
		result = m_graphics.device.waitSemaphores(&wait_info, UINT64_MAX);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	Refresh();
}

} // namespace Libs::Graphics
