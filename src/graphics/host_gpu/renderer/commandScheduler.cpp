#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/syncCensus.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/host_gpu/renderer/eagerSubmit.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/presentation/frameTiming.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <optional>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

void ReportVulkanFatal(const char* what, vk::Result result, uint64_t tick, uint32_t debug_op,
                       uint64_t debug_submit, uint32_t arg0, uint32_t arg1, uint32_t arg2,
                       uint32_t arg3, uint64_t arg4) {
	LOGF("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	     " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	     what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	     debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::printf("%s failed: %s (%d), tick=%" PRIu64 " debug_op=%u debug_submit=%" PRIu64
	            " args=%u,%u,%u,%u,0x%016" PRIx64 "\n",
	            what, vk::to_string(result).c_str(), static_cast<int>(result), tick, debug_op,
	            debug_submit, arg0, arg1, arg2, arg3, arg4);
	std::fflush(stdout);
}

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }) {
	if (FrameTiming::DiagFlagOn(std::getenv("KYTY_DIAG_GPU_DONE"))) {
		m_gpu_done = std::make_unique<GpuDone::Tracker>(
		    [this](uint64_t value) {
			    // 50 ms slices so Shutdown can stop the waiter.
			    while (!m_gpu_done_stop.load(std::memory_order_relaxed) && !m_master.IsFree(value)) {
				    vk::SemaphoreWaitInfo wait_info {};
				    const auto            handle = m_master.Handle();
				    wait_info.semaphoreCount     = 1;
				    wait_info.pSemaphores        = &handle;
				    wait_info.pValues            = &value;
				    (void)m_graphics.device.waitSemaphores(&wait_info, 50'000'000ull);
				    m_master.Refresh();
			    }
		    },
		    [] { return FrameTiming::SteadyNowNs(); },
		    [](uint64_t ns) { FrameTiming::Add(FrameTiming::Counter::GpuActiveEstNs, ns); });
	}
}

CommandScheduler::~CommandScheduler() {
	Shutdown();
	m_gpu_done_stop.store(true, std::memory_order_relaxed); // S406
	m_gpu_done.reset();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		Submit();
	}
	if (m_async_submit) {
		DrainAsyncSubmits();
		m_submit_thread.request_stop();
		m_submit_cv.notify_all();
		if (m_submit_thread.joinable()) {
			m_submit_thread.join();
		}
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::BeginRendering(const RenderState& state) {
	Current().BeginRendering(state);
}

void CommandScheduler::EndRendering() {
	if (Active() && !m_command.IsInvalid()) {
		Current().EndRendering();
	}
}

bool CommandScheduler::RenderingOpen() const {
	return Active() && !m_command.IsInvalid() && m_command.m_rendering;
}

bool CommandScheduler::RenderingAttaches(vk::ImageView view) const {
	if (!RenderingOpen() || !view) return false;
	const auto& state = m_command.m_render_state;
	for (uint32_t i = 0; i < state.num_color_attachments; i++) {
		if (state.color_attachments[i].image_view == view) return true;
	}
	return state.depth_stencil_attachment.image_view == view;
}

void CommandScheduler::Flush() {
	// Submitting a command buffer nothing was recorded into still costs a Metal command buffer
	// under MoltenVK. It only matters when something waits for the tick it would signal: a
	// deferred operation (EOP write, interrupt) queued for the current tick.
	static const bool skip_empty = FrameTiming::OptEnabled("empty_submit");
	if (skip_empty && !m_command.IsInvalid() && !m_command.m_recorded &&
	    !HasOperationsAtCurrentTick()) {
		FrameTiming::Add(FrameTiming::Counter::SkippedSubmits);
		return;
	}
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::RequestFlush() {
	static const bool     lazy = FrameTiming::OptEnabled("lazy_flush");
	static const uint32_t max  = [] {
		const char* value = std::getenv("KYTY_LAZY_FLUSH_MAX");
		return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 8u;
	}();
	if (!lazy) {
		Flush();
		return;
	}
	const auto now = std::chrono::steady_clock::now();
	if (m_flush_requests++ == 0) {
		m_first_flush_request = now;
	}
	if (m_flush_requests >= max || now - m_first_flush_request >= std::chrono::milliseconds(1)) {
		FlushPending();
	} else {
		FrameTiming::Add(FrameTiming::Counter::FlushesDeferred);
	}
}

void CommandScheduler::FlushPending() {
	if (m_flush_requests != 0) {
		m_flush_requests = 0;
		Flush();
	}
}

bool CommandScheduler::HasOperationsAtCurrentTick() {
	const auto      tick = CurrentTick();
	std::lock_guard lock(m_operation_mutex);
	return (!m_pending_operations.empty() && m_pending_operations.back().tick >= tick) ||
	       (!m_priority_operations.empty() && m_priority_operations.back().tick >= tick);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::FlushAndWait() {
	NoteDrainBegin();
	const auto tick = Submit();
	m_master.Wait(tick);
	BeginNext();
	NoteDrainEnd();
}

void CommandScheduler::Finish() {
	CheckActive();
	NoteDrainBegin();
	FrameTiming::Add(FrameTiming::Counter::Finishes);
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	NoteDrainEnd();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		NoteDrainBegin();
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
		NoteDrainEnd();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	bool refreshed = false;
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty()) {
				return;
			}
			const auto tick = m_pending_operations.front().tick;
			if (!m_master.IsFree(tick)) {
				// The timeline is polled once, and not for the open tick: it is not submitted yet.
				if (refreshed || tick >= CurrentTick()) {
					return;
				}
			} else {
				operation = std::move(m_pending_operations.front());
				m_pending_operations.pop();
			}
		}
		if (!operation.callback) {
			m_master.Refresh();
			refreshed = true;
			continue;
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active      = true;
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active      = false;
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && !m_priority_active; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at = m_priority_active && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	CheckActive();
	return m_command;
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	m_command.m_buffer = m_command_pool.Commit();
	m_open_since_ns    = FrameTiming::SteadyNowNs();
	m_command.Begin();
	m_dynamic_state.Invalidate();
	return m_command;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	CpPhases::Scope phase(CpPhases::Phase::SubmitCpu); // encode runs on the submit thread, which is not bound
	EXIT_IF(m_command.IsInvalid());
	EXIT_IF(submit.num_wait_semaphores > SubmitInfo::MaxSemaphores ||
	        submit.num_signal_semaphores >= SubmitInfo::MaxSemaphores);

	if (m_pre_submit && !m_in_pre_submit) {
		m_in_pre_submit = true;
		m_pre_submit();
		m_in_pre_submit = false;
	}
	bool first_after_drain = false; // S328
	if (!m_command.m_recorded) {
		FrameTiming::Add(FrameTiming::Counter::EmptySubmits);
	} else if (m_seg_first_pending && m_drain_depth == 0) {
		first_after_drain = true;
		FrameTiming::Add(FrameTiming::Counter::SegDrainToFirstSubmitNs,
		                 FrameTiming::ElapsedNs(m_seg_start_ns, FrameTiming::SteadyNowNs()));
		m_seg_first_pending = false;
	}
	m_flush_requests = 0; // this submit carries everything a pending flush would
	m_command.End();
	ConsumeSegPassEnd(); // End() closed the last pass; the first pass end of the segment may be that one
	const auto buffer   = m_command.m_buffer;
	auto&      graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	SubmitJob job;
	job.buffer          = buffer;
	job.debug_op        = m_command.m_debug_op;
	job.debug_submit_id = m_command.m_debug_submit_id;
	job.debug_args[0]   = m_command.m_debug_arg0;
	job.debug_args[1]   = m_command.m_debug_arg1;
	job.debug_args[2]   = m_command.m_debug_arg2;
	job.debug_args[3]   = m_command.m_debug_arg3;
	job.debug_arg4      = m_command.m_debug_arg4;
	job.first_seg       = first_after_drain;
	job.queued_ns       = first_after_drain ? FrameTiming::SteadyNowNs() : 0; // S328 (the async path overwrites it with the same clock)
	// Only timeline-only submissions may be deferred: a binary semaphore wait/signal must reach
	// the queue before whatever the caller does next (e.g. vkQueuePresentKHR).
	const bool deferrable =
	    m_async_submit && submit.num_wait_semaphores == 0 && submit.num_signal_semaphores == 0;
	uint64_t tick = 0;
	if (deferrable) {
		std::lock_guard lock(m_submit_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);
		job.submit = submit;
		job.tick   = tick;
		job.queued_ns = FrameTiming::SteadyNowNs();
		m_submit_jobs.push_back(job);
		m_submit_cv.notify_one();
	} else {
		DrainAsyncSubmits();
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);
		job.submit = submit;
		job.tick   = tick;
		QueueSubmitNow(job);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);
	FrameTiming::OnQueueSubmit();
	if (SyncCensus::Enabled()) {
		SyncCensus::CommandsAtSubmit().store(FrameTiming::Get(FrameTiming::Counter::Draws) +
		                                         FrameTiming::Get(FrameTiming::Counter::Dispatches),
		                                     std::memory_order_relaxed);
	}

	m_command.m_buffer = nullptr;
	return tick;
}

void CommandScheduler::QueueSubmitNow(SubmitJob& job) {
	vk::TimelineSemaphoreSubmitInfo timeline_info {};
	timeline_info.waitSemaphoreValueCount   = job.submit.num_wait_semaphores;
	timeline_info.pWaitSemaphoreValues      = job.submit.wait_ticks.data();
	timeline_info.signalSemaphoreValueCount = job.submit.num_signal_semaphores;
	timeline_info.pSignalSemaphoreValues    = job.submit.signal_ticks.data();

	vk::SubmitInfo submit_info {};
	submit_info.pNext                = &timeline_info;
	submit_info.waitSemaphoreCount   = job.submit.num_wait_semaphores;
	submit_info.pWaitSemaphores      = job.submit.wait_semaphores.data();
	submit_info.pWaitDstStageMask    = job.submit.wait_stages.data();
	submit_info.commandBufferCount   = 1;
	submit_info.pCommandBuffers      = &job.buffer;
	submit_info.signalSemaphoreCount = job.submit.num_signal_semaphores;
	submit_info.pSignalSemaphores    = job.submit.signal_semaphores.data();

	vk::Result result;
	const int64_t vk_begin_ns = job.first_seg ? FrameTiming::SteadyNowNs() : 0;
	{
		FrameTiming::ScopedCounterNs encode(FrameTiming::Counter::SubmitEncodeNs);
		result = m_graphics.queue.submit(1, &submit_info, nullptr);
	}
	if (job.first_seg) { // S328: the gap after a drain, in pieces
		const int64_t vk_end_ns = FrameTiming::SteadyNowNs();
		FrameTiming::Add(FrameTiming::Counter::FirstSubmitN);
		FrameTiming::Add(FrameTiming::Counter::FirstSubmitEnqToVkNs, FrameTiming::ElapsedNs(job.queued_ns, vk_begin_ns));
		FrameTiming::Add(FrameTiming::Counter::FirstSubmitEncodeNs, FrameTiming::ElapsedNs(vk_begin_ns, vk_end_ns));
	}
	if (result != vk::Result::eSuccess) {
		ReportVulkanFatal("vkQueueSubmit", result, job.tick, job.debug_op, job.debug_submit_id,
		                  job.debug_args[0], job.debug_args[1], job.debug_args[2],
		                  job.debug_args[3], job.debug_arg4);
	}
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
	if (m_gpu_done) m_gpu_done->NoteSubmit(job.tick); // S406: the moment vkQueueSubmit returned
}

void CommandScheduler::EnableAsyncSubmit() {
	if (!FrameTiming::OptEnabled("async_submit") || m_async_submit) {
		return;
	}
	m_async_submit  = true;
	m_submit_thread = std::jthread([this](std::stop_token stop) { SubmitThread(stop); });
}

void CommandScheduler::SubmitThread(std::stop_token stop) {
	CpPhases::NameCurrentThread("Thread_Submit");
	for (;;) {
		SubmitJob job;
		{
			std::unique_lock lock(m_submit_mutex);
			m_submit_cv.wait(lock, [&] { return stop.stop_requested() || !m_submit_jobs.empty(); });
			if (m_submit_jobs.empty()) {
				return;
			}
			job = m_submit_jobs.front();
			m_submit_jobs.pop_front();
			FrameTiming::Add(FrameTiming::Counter::SubmitQueueWaitNs,
			                 FrameTiming::ElapsedNs(job.queued_ns, FrameTiming::SteadyNowNs()));
			m_submit_busy = true;
		}
		{
			Common::LockGuard lock(m_graphics.queue_mutex);
			QueueSubmitNow(job);
		}
		{
			std::lock_guard lock(m_submit_mutex);
			m_submit_busy = false;
		}
		m_submit_cv.notify_all();
	}
}

void CommandScheduler::DrainAsyncSubmits() {
	if (!m_async_submit) {
		return;
	}
	std::unique_lock lock(m_submit_mutex);
	m_submit_cv.wait(lock, [&] { return m_submit_jobs.empty() && !m_submit_busy; });
}

void CommandScheduler::NoteDrainBegin() {
	if (m_drain_depth++ != 0) return;
	ConsumeSegPassEnd();
	if (m_seg_start_ns == 0) return;
	const uint64_t length = FrameTiming::ElapsedNs(m_seg_start_ns, FrameTiming::SteadyNowNs());
	const uint64_t waited = FrameTiming::Get(FrameTiming::Counter::CpWaitGuestNs) +
	                        FrameTiming::Get(FrameTiming::Counter::CpWaitGpuNs) - m_seg_wait_base;
	FrameTiming::Add(FrameTiming::Counter::SegN);
	FrameTiming::Add(FrameTiming::Counter::SegRecordNs, length > waited ? length - waited : 0);
}

void CommandScheduler::NoteDrainEnd() {
	if (m_drain_depth == 0 || --m_drain_depth != 0) return;
	m_seg_start_ns      = FrameTiming::SteadyNowNs();
	m_seg_first_pending = true;
	m_seg_mid_splits    = 0;
	m_seg_wait_base     = FrameTiming::Get(FrameTiming::Counter::CpWaitGuestNs) + FrameTiming::Get(FrameTiming::Counter::CpWaitGpuNs);
	m_command.m_pass_end_ns   = 0;
	m_command.m_note_pass_end = true;
}

namespace {
bool EagerEnabled() {
	static const bool enabled = [] { const char* v = std::getenv("KYTY_EAGER_SUBMIT"); return v != nullptr && (v[0] == '1' || v[0] == '2'); }();
	return enabled;
}
// S328: KYTY_EAGER_SUBMIT=2 also splits ONE open pass in the middle, in the first command buffer after a drain.
bool EagerMidEnabled() {
	static const bool enabled = [] { const char* v = std::getenv("KYTY_EAGER_SUBMIT"); return v != nullptr && v[0] == '2'; }();
	return enabled;
}
uint64_t EagerMinUs() {
	static const uint64_t min_us = [] {
		const char* v = std::getenv("KYTY_EAGER_SUBMIT_US");
		return v != nullptr ? static_cast<uint64_t>(std::strtoull(v, nullptr, 10)) : 300ull;
	}();
	return min_us;
}
} // namespace

// Asks the policy; true = the buffer was split (when pass_closing, the open pass was closed first and *closed_pass is set).
bool CommandScheduler::EagerSplitNow(bool pass_closing, bool* closed_pass, bool mid_pass) {
	Sync::EagerQuery query;
	query.enabled      = true;
	query.pass_open    = RenderingOpen();
	query.pass_closing = pass_closing;
	query.has_work     = m_command.m_recorded;
	query.gpu_idle     = true; // refreshed below only when everything cheaper already says yes
	query.age_us       = FrameTiming::ElapsedNs(m_open_since_ns, FrameTiming::SteadyNowNs()) / 1000u;
	query.min_age_us   = EagerMinUs();
	query.mid_mode     = mid_pass && EagerMidEnabled();
	query.first_cb_after_drain = m_seg_first_pending && m_drain_depth == 0;
	query.mid_splits   = m_seg_mid_splits;
	if (!Sync::ShouldEagerSplit(query)) return false;
	m_master.Refresh();
	query.gpu_idle = m_master.KnownGpuTick() + 1 >= CurrentTick();
	if (!Sync::ShouldEagerSplit(query)) return false;
	if (query.pass_open) {
		m_command.EndRendering();
		*closed_pass = true;
	}
	Flush();
	return true;
}

void CommandScheduler::MaybeEagerSplit() {
	if (!EagerEnabled() || !Active() || m_command.IsInvalid() || InDeferredOperation() || m_in_pre_submit) return;
	bool closed = false;
	if (EagerSplitNow(false, &closed)) FrameTiming::Add(FrameTiming::Counter::EagerSubmitN);
}

void CommandScheduler::MaybeEagerSplitAtPass(const RenderState& next) {
	if (!EagerEnabled() || !Active() || m_command.IsInvalid() || InDeferredOperation() || m_in_pre_submit) return;
	if (!m_command.m_rendering) return;
	bool closed = false;
	if (m_command.SamePass(next)) {
		// S328: the draw continues the open pass; the policy may still split it in the middle (first buffer after a drain only).
		if (!EagerMidEnabled() || !EagerSplitNow(false, &closed, true)) return;
		++m_seg_mid_splits;
		FrameTiming::Add(FrameTiming::Counter::EagerMidN);
		m_command.ArmReopenLoad(); // the new buffer reopens this pass with loadOp=LOAD
		return;
	}
	if (EagerSplitNow(true, &closed)) FrameTiming::Add(FrameTiming::Counter::EagerSubmitPassN);
}

void CommandScheduler::ConsumeSegPassEnd() {
	if (m_command.m_pass_end_ns == 0) return;
	if (m_seg_start_ns != 0) {
		FrameTiming::Add(FrameTiming::Counter::SegFirstPassEndNs, FrameTiming::ElapsedNs(m_seg_start_ns, m_command.m_pass_end_ns));
	}
	m_command.m_pass_end_ns = 0;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
