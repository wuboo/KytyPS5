#include "graphics/guest_gpu/graphicsRun.h"
#include "common/cpTuning.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/frameTiming.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/diagOut.h"
#include "common/memEvents.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/guest_gpu/pm4Hist.h"
#include "graphics/host_gpu/renderer/cache/bdaDefer.h"
#include "graphics/host_gpu/renderer/pipeline/rdHazard.h"
#include "graphics/host_gpu/renderer/gpuKeepWarm.h"
#include "graphics/host_gpu/renderer/indirectGpu.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/rfOpDiag.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/sync.h"
#include "graphics/host_gpu/renderer/syncCensus.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/recompiler/ir/passes/rfGpuPatch.h"
#include "kernel/memory.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics {

static thread_local Pm4Execution* g_current_execution = nullptr;
static thread_local GuestGpu*     g_gpu_state         = nullptr;

GuestGpu::GuestGpu(RenderContext& renderer): m_renderer(renderer) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	m_gfx_cp = std::make_unique<CommandProcessor>(renderer, 0);
	m_thread = std::jthread(ThreadRun, this);
}

GuestGpu::~GuestGpu() {
	Shutdown();
}

void GuestGpu::Shutdown() {
	std::lock_guard shutdown_lock(m_shutdown_mutex);
	if (m_shutdown_complete) {
		return;
	}
	{
		Common::LockGuard lock(m_queue_mutex);
		m_accepting = false;
		m_stopping  = true;
		m_work_available.SignalAll();
	}
	if (m_thread.joinable()) {
		m_thread.join();
	}
	m_shutdown_complete = true;
}

bool GuestGpu::IsStopping() {
	Common::LockGuard lock(m_queue_mutex);
	return m_stopping;
}

void GuestGpu::SendCommand(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_commands.push_back(std::move(command));
	m_pending_commands.fetch_add(1, std::memory_order_release);
	m_work_available.Signal();
}

void GuestGpu::ProcessCommands() {
	while (m_pending_commands.load(std::memory_order_acquire) != 0) {
		Common::UniqueFunction<void> command;
		{
			Common::LockGuard lock(m_queue_mutex);
			EXIT_IF(m_commands.empty());
			command = std::move(m_commands.front());
			m_commands.pop_front();
			EXIT_IF(m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
		}
		command();
	}
}

void GuestGpu::SendCommandSync(Common::UniqueFunction<void>&& command) {
	EXIT_IF(!command);
	if (IsGpuThread()) {
		command();
		return;
	}
	std::binary_semaphore done {0};
	SendCommand([operation = std::move(command), &done]() mutable {
		operation();
		done.release();
	});
	done.acquire();
}

void GuestGpu::Submit(std::span<const uint32_t> commands) {
	if (commands.empty()) {
		return;
	}
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 0;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitCompute(uint32_t queue, std::span<const uint32_t> commands) {
	EXIT_IF(commands.empty());

	EXIT_NOT_IMPLEMENTED(queue < ComputeQueueBase || queue >= ComputeQueueBase + ComputeQueueCount);

	const auto compute_queue = queue - ComputeQueueBase;
	Submission submission;
	submission.type     = SubmissionType::Commands;
	submission.queue_id = 1 + compute_queue;
	submission.commands = commands;
	Enqueue(std::move(submission));
}

void GuestGpu::SubmitFlipPreparation(uint64_t request_id) {
	Submission submission;
	submission.type            = SubmissionType::FlipPreparation;
	submission.queue_id        = 0;
	submission.flip_request_id = request_id;
	Enqueue(std::move(submission));
}

void GuestGpu::SuspendPoint() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	// Do not hold a queue lock while waiting: asynchronous work may be needed to
	// finish the preceding graphics frame. The first point returns immediately.
	m_suspend_point_ready->acquire();
	Submission submission;
	submission.type = SubmissionType::SuspendPoint;
	Enqueue(std::move(submission));
	m_done_num++;
}

int GuestGpu::GetFrameNum() const {
	return m_done_num;
}

CommandProcessor& GuestGpu::GetProcessor(uint32_t queue_id) {
	if (queue_id == 0) {
		return *m_gfx_cp;
	}
	auto& processor = m_compute_cp[queue_id - 1];
	if (processor == nullptr) {
		processor = std::make_unique<CommandProcessor>(m_renderer, ComputeQueueBase + queue_id - 1);
	}
	return *processor;
}

void CommandProcessor::Reset() {
	m_sh_ctx.Reset();
	m_ucfg.Reset();
	m_ctx.Reset();
	m_saved_ctx.reset();
	m_ds                 = {};
	m_num_instances_addr = 0;
	m_ni_front.pending_addr = 0; // S577: the epoch keeps counting (the recorder's memo is keyed by it)
	m_predicate_skip     = false;
}

void CommandProcessor::ApplyContextStateOperation(ContextStateOperation operation) {
	switch (operation) {
		case ContextStateOperation::Clear: m_ctx.Reset(); break;
		case ContextStateOperation::Push:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_ctx);
			break;
		case ContextStateOperation::Pop:
			EXIT_IF(!m_saved_ctx.has_value());
			m_ctx = *m_saved_ctx;
			m_saved_ctx.reset();
			break;
		case ContextStateOperation::PushClear:
			EXIT_IF(m_saved_ctx.has_value());
			m_saved_ctx.emplace(m_ctx);
			m_ctx.Reset();
			break;
	}
}

void CommandProcessor::BufferInit() {
	m_draws = 0;
	GetScheduler().Begin(m_ctx, m_ucfg, m_sh_ctx);
}

void CommandProcessor::BufferFlush() {
	FrameTiming::Add(FrameTiming::Counter::CpFlushes);
	GetScheduler().RequestFlush();
}

void CommandProcessor::BufferFlushNow() {
	FrameTiming::Add(FrameTiming::Counter::CpFlushes);
	GetScheduler().Flush();
	// The occlusion counters reach a guest that polls them without waiting for a later fence.
	m_renderer.GetOcclusionQueries().Publish();
}

bool CommandProcessor::FlushRequested() const {
	return GetScheduler().FlushRequested();
}

void CommandProcessor::FlushPending() {
	if (GetScheduler().FlushRequested()) {
		GetScheduler().FlushPending();
	}
}

void CommandProcessor::BufferFlushAndWait() {
	GetScheduler().FlushAndWait();
}

void CommandProcessor::BufferWait() {
	BufferInit();
	GetScheduler().Finish();
}

void CommandProcessor::WaitForRewind(bool valid) {
	if (!valid) {
		SuspendPm4();
		return;
	}
	PromoteGuestWrites(); // S601: the game thread may have set the valid bit after writing the commands' data
}

// S601 (KYTY_BDA_DEFER, port of KytyPS5#1933, bdaDefer.h).
void CommandProcessor::PromoteGuestWrites() {
	if (!BdaDefer::EnvOn()) {
		return;
	}
	// Keeps the caller's read of the guest value ahead of the check for held-back events, or the check could miss an event
	// queued before the value was written.
	std::atomic_thread_fence(std::memory_order_acquire);
	m_renderer.GetBufferCache().PromoteBdaSync();
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func) {
	switch (func) {
		case 0: return true;
		case 1: return (value & mask) < ref;
		case 2: return (value & mask) <= ref;
		case 3: return (value & mask) == ref;
		case 4: return (value & mask) != ref;
		case 5: return (value & mask) >= ref;
		case 6: return (value & mask) > ref;
		default: EXIT("unknown wait compare function: %" PRIu32 "\n", func);
	}

	return false;
}

template <typename T>
void CommandProcessor::WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t wait_op) {
	EXIT_IF(addr == nullptr);
	if ((wait_op & ~1u) != 0) {
		EXIT("unsupported wait_reg_mem operation: 0x%08" PRIx32 "\n", wait_op);
	}

	if (m_pipe != nullptr) { // S479: the writes still in the FIFO are what the memory will hold when the wait is reached
		bool virtual_satisfied = false;
		if (PipeWaitVirtual(reinterpret_cast<uint64_t>(addr), sizeof(T), func, ref, mask, &virtual_satisfied) && virtual_satisfied) {
			PromoteGuestWrites(); // S601: satisfied by a CP write still in the FIFO; promoting is only conservative here
			return;
		}
	}
	const bool wait_satisfied = TestWaitRegMemValue(*addr, ref, mask, func);
	if (Pm4Hist::Enabled()) Pm4Hist::Wait(reinterpret_cast<uintptr_t>(addr), wait_satisfied); // S477 D1 DIAG
	if (!wait_satisfied) {
		SuspendPm4();
		return;
	}
	// S601: the value may come from a game thread that wrote the data it guards first; those writes must reach the BDA draws that
	// follow, even though the submission never suspended.
	PromoteGuestWrites();
}

template void CommandProcessor::WaitRegMem<uint32_t>(uint32_t, const uint32_t*, uint32_t, uint32_t,
                                                     uint32_t);
template void CommandProcessor::WaitRegMem<uint64_t>(uint32_t, const uint64_t*, uint64_t, uint64_t,
                                                     uint32_t);

void CommandProcessor::MemSemaphore(uint64_t* counter, bool signal, bool write) {
	if (signal) {
		*counter = write ? 1u : *counter + 1u;
	} else if (*counter == 0) {
		SuspendPm4();
	} else {
		(*counter)--;
		PromoteGuestWrites(); // S601: a game thread may have signalled the count after writing the data it guards
	}
}

void CommandProcessor::WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num,
                                 uint32_t write_control) {
	const uint32_t dst_sel = ((write_control >> 30u) & 0x1u) | ((write_control >> 7u) & 0x1eu);
	const bool     write_one_address = ((write_control >> 16u) & 0x1u) != 0;

	switch (dst_sel) {
		case 0:
		case 2:
		case 4:
		case 5:
		case 6: break;
		default: EXIT("unsupported writeData destination selector 0x%02" PRIx32 "\n", dst_sel);
	}
	if (dw_num == 0) {
		return;
	}
	NumInstLazy::NoteGpuWrite(reinterpret_cast<uint64_t>(dst), write_one_address ? 4u : static_cast<uint64_t>(dw_num) * 4u); // S577
	Common::MemEvents::Record(Common::MemEvents::Kind::GpuWriteData, reinterpret_cast<uint64_t>(dst), write_one_address ? 4u : static_cast<uint64_t>(dw_num) * 4u,
	                          dw_num, dst_sel);

	// Completion labels can be reused immediately; memcpy may write a destination twice.
	auto* output = static_cast<volatile uint32_t*>(dst);
	for (uint32_t i = 0; i < dw_num; i++) {
		*output = src[i];
		output += !write_one_address;
	}
}

void CommandProcessor::WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes) {
	if (dst_address == 0 || (num_bytes != sizeof(uint32_t) && num_bytes != sizeof(uint64_t)) ||
	    (dst_address & (num_bytes - 1u)) != 0) {
		EXIT("invalid reference-clock copy, dst=0x%016" PRIx64 " size=%u\n", dst_address,
		     num_bytes);
	}
	const auto value = Sync::ReadReferenceClock();
	std::memcpy(reinterpret_cast<void*>(dst_address), &value, num_bytes);
	static std::atomic<uint32_t> clock_log_count {0};
	if (clock_log_count.fetch_add(1) < 64) {
		LOGF("\t copy_data reference clock: dst=0x%016" PRIx64 " value=0x%016" PRIx64
		     " size=%u\n",
		     dst_address, value, num_bytes);
	}
}

void CommandProcessor::GetLodStats(volatile uint32_t* dst, uint32_t size, bool reset) {
	auto& counters = m_renderer.GetRenderExecutor().m_mip_stats;
	// Each of up to 256 2-dword slots after the 16-dword header holds the clamped sample count
	// (dword 0 bits 0-23), the slot id (bits 24-31), the unclamped sample count (dword 1 bits 0-23)
	// and the finest mip sampled (bits 24-27, 15 for none). A non-zero header dword 0 then marks
	// the report complete.
	if (dst != nullptr && size >= 64) {
		const auto slots = std::min((size - 64) / 8, 256u);
		for (uint32_t slot = 0; slot < slots; slot++) {
			const auto& counter = counters[slot];
			dst[16 + slot * 2]  = (slot << 24u) | uint32_t {counter.clamped};
			dst[16 + slot * 2 + 1] =
			    (uint32_t {counter.finest_level} << 24u) | uint32_t {counter.unclamped};
		}
		dst[0] = 1;
	}
	if (reset) {
		counters = {};
	}
}

void CommandProcessor::DmaData(uint8_t dst_sel, uint64_t dst_address_or_offset, uint8_t src_sel,
                               uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes) {
	if (num_bytes == 0 || static_cast<uint32_t>(dst_address_or_offset) == 0x3022cu) {
		return;
	}
	auto decode_gds = [](uint8_t selector, bool& is_gds) {
		switch (selector) {
			case 0:
			case 3: is_gds = false; return true;
			case 1: is_gds = true; return true;
			default: return false;
		}
	};
	if (dst_sel == 2) {
		// kNowhere discards the GL2 prefetch destination without a guest-visible write.
		if (src_sel != 3) {
			EXIT("unsupported dmaData nowhere source selector 0x%02" PRIx8 "\n", src_sel);
		}
		return;
	}
	bool dst_gds = false;
	if (!decode_gds(dst_sel, dst_gds)) {
		EXIT("unsupported dmaData destination selector 0x%02" PRIx8 "\n", dst_sel);
	}
	if (!dst_gds) {
		NumInstLazy::NoteGpuWrite(dst_address_or_offset, num_bytes); // S577 (ObtainBuffer notes it too)
		Common::MemEvents::Record(Common::MemEvents::Kind::GpuDma, dst_address_or_offset, num_bytes, src_address_or_offset_or_immediate, src_sel,
		                          src_sel == 2 ? "fill" : "copy");
	}
	auto& buffer_cache = m_renderer.GetBufferCache();
	if (src_sel == 2) {
		buffer_cache.FillBuffer(
		    dst_address_or_offset, num_bytes,
		    static_cast<uint32_t>(src_address_or_offset_or_immediate & 0xffffffffu), dst_gds);
		return;
	}
	bool src_gds = false;
	if (!decode_gds(src_sel, src_gds)) {
		EXIT("unsupported dmaData source selector 0x%02" PRIx8 "\n", src_sel);
	}
	if (src_gds && dst_gds) {
		EXIT("unsupported dmaData GDS-to-GDS copy\n");
	}
	buffer_cache.CopyBuffer(dst_address_or_offset, src_address_or_offset_or_immediate, num_bytes,
	                        dst_gds, src_gds);
}

void GuestGpu::Enqueue(Submission submission) {
	EXIT_IF(submission.queue_id >= QueueCount);
	submission.enqueue_ns = FrameTiming::SteadyNowNs();
	Common::LockGuard lock(m_queue_mutex);
	EXIT_IF(!m_accepting);
	m_queues[submission.queue_id].push_back(std::move(submission));
	m_submission_count++;
	m_work_available.Signal();
	GpuKeepWarm::NoteActivity(); // KYTY_GPU_KEEPWARM: resumes a paused spin; a relaxed load when off
}

void GuestGpu::WaitForIdle() {
	EXIT_IF(IsGpuThread() || CommandScheduler::InDeferredOperation());
	Common::LockGuard lock(m_queue_mutex);
	while (m_processing || !m_commands.empty() || m_submission_count != 0) {
		m_idle.Wait(&m_queue_mutex);
	}
}

void GuestGpu::ThreadRun(void* data) {
	auto* gpu = static_cast<GuestGpu*>(data);
	EXIT_IF(gpu == nullptr);
	KYTY_PROFILER_THREAD("Thread_Gpu");
	g_gpu_state = gpu;
	CpPhases::BindCpThread();
	CpPhases::NameCurrentThread("Thread_Gpu");
	FrameTiming::RegisterCpThread(); // S570 T1b (KYTY_THREAD_CPU=lite)
	Common::CpTuning::ApplyToCpThread();

	for (;;) {
		Krf4PollFlips(); // S599 KYTY_KRF3=v4: the closed guest flips into the guest page guard (a relaxed load when nothing closed)
		// Lazy flush: before the thread waits (no command, no runnable submission), submit what
		// the batched flushes held back, so nothing the guest waits for stays unsubmitted.
		if (gpu->m_gfx_cp->FlushRequested()) {
			bool idle = false;
			{
				Common::LockGuard lock(gpu->m_queue_mutex);
				idle = gpu->m_commands.empty();
				for (uint32_t id = 0; idle && id < QueueCount; id++) {
					idle = gpu->m_queues[id].empty() || gpu->m_queues[id].front().blocked;
				}
			}
			if (idle) {
				gpu->m_gfx_cp->FlushPending();
			}
		}
		Submission                   submission;
		Common::UniqueFunction<void> command;
		bool                         should_stop = false;
		{
			Common::LockGuard lock(gpu->m_queue_mutex);
			while (gpu->m_commands.empty() && gpu->m_submission_count == 0 && !gpu->m_stopping) {
				gpu->m_processing = false;
				gpu->m_idle.Signal();
				FrameTiming::ScopedCounterNs wait(FrameTiming::Counter::CpWaitGuestNs);
				ShaderRecompiler::IR::RfGpuPatch::Krf4WaitScope krf4_wait; // S599 KYTY_KRF3=v4: the same wait for the guest page guard
				gpu->m_work_available.Wait(&gpu->m_queue_mutex);
			}
			if (gpu->m_stopping && gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_processing = false;
				gpu->m_idle.SignalAll();
				should_stop = true;
			} else if (!gpu->m_commands.empty()) {
				command = std::move(gpu->m_commands.front());
				gpu->m_commands.pop_front();
				EXIT_IF(gpu->m_pending_commands.fetch_sub(1, std::memory_order_acq_rel) == 0);
				gpu->m_processing = true;
			} else {
				int selected_queue = -1;
				for (uint32_t offset = 0; offset < QueueCount; offset++) {
					const auto id = (gpu->m_next_queue + offset) % QueueCount;
					if (!gpu->m_queues[id].empty() && !gpu->m_queues[id].front().blocked) {
						selected_queue = static_cast<int>(id);
						break;
					}
				}
				if (selected_queue < 0) {
					gpu->m_processing = false;
					{
						FrameTiming::ScopedCounterNs wait(FrameTiming::Counter::CpWaitGuestNs);
						ShaderRecompiler::IR::RfGpuPatch::Krf4WaitScope krf4_wait; // S599
						gpu->m_work_available.WaitFor(&gpu->m_queue_mutex, 100);
					}
					for (auto& queue: gpu->m_queues) {
						if (!queue.empty()) {
							queue.front().blocked = false;
						}
					}
					continue;
				}
				auto& queue = gpu->m_queues[static_cast<uint32_t>(selected_queue)];
				submission  = std::move(queue.front());
				queue.pop_front();
				gpu->m_submission_count--;
				gpu->m_next_queue = (static_cast<uint32_t>(selected_queue) + 1) % QueueCount;
				gpu->m_processing = true;
				FrameTiming::NoteGuestLead(FrameTiming::ElapsedNs(submission.enqueue_ns, FrameTiming::SteadyNowNs())); // S441
			}
		}
		if (should_stop) {
			gpu->m_gfx_cp->BufferWait();
			g_gpu_state = nullptr;
			return;
		}

		if (command) {
			{
				Common::DebugCounters::ScopedTimer busy(
				    Common::DebugCounters::Counter::GpuThreadBusyNs);
				const auto busy_begin = std::chrono::steady_clock::now();
				{
					CpPhases::Scope phase(CpPhases::Phase::Cmds);
					command();
				}
				const auto busy_ns = static_cast<uint64_t>(
				    std::chrono::duration_cast<std::chrono::nanoseconds>(
				        std::chrono::steady_clock::now() - busy_begin)
				        .count());
				FrameTiming::AddGpuBusy(busy_ns);
				FrameTiming::Add(FrameTiming::Counter::CpBusyNs, busy_ns);
			}

			Common::LockGuard lock(gpu->m_queue_mutex);
			gpu->m_processing = false;
			if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
				gpu->m_idle.SignalAll();
			}
			continue;
		}

		const bool complete = [&] {
			Common::DebugCounters::ScopedTimer busy(
			    Common::DebugCounters::Counter::GpuThreadBusyNs);
			const auto busy_begin = std::chrono::steady_clock::now();
			const bool done       = gpu->Process(submission);
			const auto busy_ns    = static_cast<uint64_t>(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
			                                                         busy_begin)
			        .count());
			FrameTiming::AddGpuBusy(busy_ns);
			FrameTiming::Add(FrameTiming::Counter::CpBusyNs, busy_ns);
			return done;
		}();

		Common::LockGuard lock(gpu->m_queue_mutex);
		if (!complete) {
			submission.blocked = true;
			gpu->m_queues[submission.queue_id].push_front(std::move(submission));
			gpu->m_submission_count++;
		} else {
			for (auto& queue: gpu->m_queues) {
				if (!queue.empty()) {
					queue.front().blocked = false;
				}
			}
		}
		gpu->m_processing = false;
		if (gpu->m_commands.empty() && gpu->m_submission_count == 0) {
			gpu->m_idle.SignalAll();
		}
	}
}

bool GuestGpu::Process(Submission& submission) {
	// S601 (KYTY_BDA_DEFER): game-thread writes made before this submission, or before the wait it resumes from, must reach the BDA
	// draws it records.
	if (BdaDefer::EnvOn()) {
		m_renderer.GetBufferCache().PromoteBdaSync();
	}
	auto& cp = GetProcessor(submission.queue_id);

	cp.BufferInit();
	bool complete = true;

	switch (submission.type) {
		case SubmissionType::Commands:
			complete =
			    cp.Process(submission.execution, submission.commands) == Pm4ProcessResult::Complete;
			if (submission.execution.MadeProgress()) {
				// A suspended submission waits for something; submit what it recorded now.
				if (complete) {
					cp.BufferFlush();
				} else {
					cp.BufferFlushNow();
				}
			}
			break;
		case SubmissionType::FlipPreparation:
			cp.PrepareCpuFlip(submission.flip_request_id);
			break;
		case SubmissionType::SuspendPoint:
			// Age caches at the frame boundary, not once per command list.
			m_renderer.RunGarbageCollector();
			m_renderer.GetCommandScheduler().DeferPriorityOperation(
			    [ready = m_suspend_point_ready] { ready->release(); });
			cp.BufferFlush();
			cp.Reset();
			break;
	}

	return complete;
}

Pm4ProcessResult CommandProcessor::Process(Pm4Execution&             execution,
                                           std::span<const uint32_t> commands) {
	KYTY_PROFILER_BLOCK("CommandProcessor::Process");
	EXIT_IF(g_current_execution != nullptr);
	EXIT_IF(commands.size() > UINT32_MAX);
	if (execution.m_buffer_stack.empty() && !commands.empty()) {
		execution.m_buffer_stack.push_back({commands});
	}
	execution.m_suspended     = false;
	execution.m_made_progress = false;

	struct ExecutionScope {
		explicit ExecutionScope(Pm4Execution& execution) { g_current_execution = &execution; }
		~ExecutionScope() { g_current_execution = nullptr; }
	} execution_scope(execution);

	if (m_pipe == nullptr) PipeInit(); // S479 (KYTY_CP_PIPE)
	if (PipeFrontOn()) {
		PipeFrontRun(execution, g_gpu_state); // S549 (KYTY_CP_FRONT_THREAD): the front thread parses, this thread records
	} else {
		ProcessPm4(execution);
	}
	return execution.m_buffer_stack.empty() ? Pm4ProcessResult::Complete
	                                        : Pm4ProcessResult::Blocked;
}

// S549: the front thread's entry for one stream (Process runs PipeFrontRun on Thread_Gpu meanwhile). The execution is shared: the front owns it until its End item.
void CommandProcessor::FrontParse(Pm4Execution& execution) {
	struct ExecutionScope {
		explicit ExecutionScope(Pm4Execution& execution) { g_current_execution = &execution; }
		~ExecutionScope() { g_current_execution = nullptr; }
	} execution_scope(execution);
	ProcessPm4(execution);
}

void CommandProcessor::ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain) {
	EXIT_IF(g_current_execution == nullptr);
	EXIT_IF(!g_current_execution->m_next_buffer.empty());
	g_current_execution->m_next_buffer = commands;
	g_current_execution->m_chain       = chain;
}

void CommandProcessor::SuspendPm4() {
	EXIT_IF(g_current_execution == nullptr);
	g_current_execution->m_suspended = true;
}

void CommandProcessor::ProcessPm4(Pm4Execution& execution) {
	CpPhases::Scope parse_phase(CpPhases::Phase::Parse); // S323: handler bodies are Phase::Handler, children subtract
	if (Pm4Hist::Enabled()) { // S477 D1 DIAG
		Pm4Hist::BeginCall();
		Pm4Hist::MaybeDump(FrameTiming::GuestFlips(), [](const char* l) { Common::DiagOut::Write(l); });
	}
	if (RdHazard::Enabled()) RdHazard::MaybeDump(FrameTiming::GuestFlips()); // S477 D2 DIAG
	const bool debug_dump = graphics_debug_dump_enabled();
	const bool front      = OnFrontThread(); // S549: this is the front thread of a CP with KYTY_CP_FRONT_THREAD (g_gpu_state is null here)
	while (!execution.m_buffer_stack.empty()) {
		if (front) {
			// S479 rule on two threads: a guest command runs on Thread_Gpu after every item parsed before it (a Sync item).
			if (PipeFrontCommandsPending()) PipeFrontSync(2, 0, nullptr, 0, 0);
		} else if (g_gpu_state != nullptr) {
			// S479: a command from a guest thread (a fault readback, an unmap) must see every draw parsed before it, as it did when draws ran where they were parsed.
			if (m_pipe != nullptr && g_gpu_state->m_pending_commands.load(std::memory_order_acquire) != 0) PipeDrain(0, 0x1f0);
			CpPhases::Scope phase(CpPhases::Phase::Cmds);
			g_gpu_state->ProcessCommands();
		}
		auto& cursor = execution.m_buffer_stack.back();
		if (cursor.offset_dw == cursor.commands.size()) {
			execution.m_buffer_stack.pop_back();
			continue;
		}

		const auto* const packet        = cursor.commands.data() + cursor.offset_dw;
		const auto        total_dw      = static_cast<uint32_t>(cursor.commands.size());
		const auto        remaining_dw  = total_dw - cursor.offset_dw;
		const auto        packet_header = packet[0];
		const auto        opcode        = (packet_header >> 8u) & 0xffu;

		if (packet_header == 0x80000000u || Pm4::PacketSizeDw(packet_header) == 1u) {
			cursor.offset_dw++;
			execution.m_made_progress = true;
			continue;
		}

		EXIT_NOT_IMPLEMENTED(remaining_dw < 2);

		if (debug_dump) {
			LOGF("CP packet: offset=0x%05" PRIx32 " cmd_id=0x%08" PRIx32 " op=0x%02" PRIx32
			     " len=%" PRIu32 "\n",
			     cursor.offset_dw, packet_header, opcode, KYTY_PM4_LEN(packet_header));
		}

		if ((packet_header & 1u) != 0 && ShouldSkipPredicatedPackets()) {
			auto packet_dw = KYTY_PM4_LEN(packet_header);
			EXIT_NOT_IMPLEMENTED(packet_dw > remaining_dw);
			cursor.offset_dw += packet_dw;
			execution.m_made_progress = true;
			continue;
		}

		auto handler = g_cp_op_func[opcode];

		if (handler == nullptr) {
			const auto offset = cursor.offset_dw;
			LOGF("unknown PM4 packet: data=0x%016" PRIx64 ", num_dw=%" PRIu32
			     ", offset=0x%05" PRIx32 ", current=0x%016" PRIx64 "\n",
			     reinterpret_cast<uint64_t>(packet - offset), total_dw, offset,
			     reinterpret_cast<uint64_t>(packet));
			const auto  dump_begin = (offset > 8 ? offset - 8 : 0);
			const auto  dump_end   = std::min<uint32_t>(total_dw, offset + 16);
			auto* const base       = packet - offset;
			for (uint32_t i = dump_begin; i < dump_end; i++) {
				LOGF("\t%05" PRIx32 "%s %08" PRIx32 "\n", i, (i == offset ? ":" : " "), base[i]);
			}
			EXIT("unknown op\n\t%05" PRIx32 ":\n\tcmd_id = %08" PRIx32 "\n", offset, packet_header);
		}

		// S479 (KYTY_CP_PIPE): draws, dispatches, signals and CPU writes wait in the FIFO; the rest runs here as before, after a drain when it is not known to be harmless.
		if (m_pipe != nullptr) {
			const auto pipe_dw  = KYTY_PM4_LEN(packet_header);
			const auto pipe_cls = CpPipe::Classify(opcode, opcode == Pm4::IT_NOP ? KYTY_PM4_R(packet_header) : 0u, pipe_dw >= 2u ? packet[1] : 0u, pipe_dw);
			if (pipe_cls == CpPipe::OpClass::Draw || pipe_cls == CpPipe::OpClass::Dispatch || pipe_cls == CpPipe::OpClass::Ordered) {
				EXIT_NOT_IMPLEMENTED(pipe_dw == 0 || pipe_dw > remaining_dw);
				if (CpPhases::Enabled()) FrameTiming::Add(FrameTiming::Counter::Pm4Packets);
				PipeEnqueue(pipe_cls, opcode, packet, pipe_dw);
				cursor.offset_dw += pipe_dw;
				execution.m_made_progress = true;
				continue;
			}
			if (pipe_cls == CpPipe::OpClass::Inline && CpPipe::ReadsIndirectTable(opcode) && pipe_dw >= 5u) {
				uint64_t table = 0, table_size = 0;
				if (CpPipe::IndirectTableRange(packet + 1, &table, &table_size) && PipeOverlaps(table, table_size)) PipeDrain(0, opcode);
			}
			if (pipe_cls == CpPipe::OpClass::Drain && !front) {
				PipeDrain(0, opcode == Pm4::IT_NOP ? 256u + KYTY_PM4_R(packet_header) : opcode);
			}
			if (front && pipe_cls == CpPipe::OpClass::Drain) {
				// S549: a drain-class handler draws, signals or reads the predication on Thread_Gpu, on the front's live state, while the front waits.
				const uint32_t packet_dw = PipeFrontSync(1, opcode, packet, remaining_dw, total_dw);
				EXIT_IF(packet_dw > remaining_dw);
				if (execution.m_suspended) {
					if (RdHazard::Enabled()) RdHazard::AtCallEnd(); // S477 D2 DIAG
					return;
				}
				cursor.offset_dw += packet_dw;
				execution.m_made_progress = true;
				if (!execution.m_next_buffer.empty()) {
					if (execution.m_chain) {
						cursor = {execution.m_next_buffer};
					} else {
						execution.m_buffer_stack.push_back({execution.m_next_buffer});
					}
					execution.m_next_buffer = {};
				}
				continue;
			}
		}

		if (CpPhases::Enabled()) FrameTiming::Add(FrameTiming::Counter::Pm4Packets);
		if (RdHazard::Enabled() && Pm4Hist::Classify(opcode, opcode == Pm4::IT_NOP ? KYTY_PM4_R(packet_header) : 0u) == Pm4Hist::Cls::Draw) RdHazard::OnDraw(); // S477 D2 DIAG
		if (Pm4Hist::Enabled()) Pm4Hist::Packet(opcode, opcode == Pm4::IT_NOP ? KYTY_PM4_R(packet_header) : 0u, KYTY_PM4_LEN(packet_header) + 1u); // S477 D1 DIAG
		uint32_t packet_dw;
		{
			CpPhases::Scope handler_phase(CpPhases::Phase::Handler);
			if (CpPhases::Enabled()) {
				using C = FrameTiming::Counter;
				switch (opcode) { // S463 DIAG: handler self time by opcode class
				case Pm4::IT_DRAW_INDEX_2: case Pm4::IT_DRAW_INDEX_AUTO: case Pm4::IT_DRAW_INDIRECT: case Pm4::IT_DRAW_INDEX_INDIRECT:
				case Pm4::IT_DRAW_INDEX_OFFSET_2: case Pm4::IT_DRAW_INDIRECT_MULTI: case Pm4::IT_DRAW_INDEX_INDIRECT_MULTI: case Pm4::IT_DISPATCH_DRAW:
					handler_phase.SetSub(C::HdlDrawNs, C::HdlDrawN); break;
				case Pm4::IT_SET_SH_REG: case Pm4::IT_SET_CONTEXT_REG: case Pm4::IT_SET_CONTEXT_REG_INDIRECT: case Pm4::IT_SET_UCONFIG_REG:
				case Pm4::IT_NUM_INSTANCES: case Pm4::IT_SET_CONFIG_REG: case Pm4::IT_SET_SH_REG_INDIRECT:
					handler_phase.SetSub(C::HdlRegNs, C::HdlRegN); break;
				case Pm4::IT_NOP: handler_phase.SetSub(C::HdlNopNs, C::HdlNopN); break;
				case Pm4::IT_WAIT_REG_MEM: case Pm4::IT_WAIT_REG_MEM_64: handler_phase.SetSub(C::HdlWaitNs, C::HdlWaitN); break;
				case Pm4::IT_DISPATCH_DIRECT: case Pm4::IT_DISPATCH_INDIRECT: handler_phase.SetSub(C::HdlDispNs, C::HdlDispN); break;
				default: handler_phase.SetSub(C::HdlOtherNs, C::HdlOtherN); break;
				}
			}
			if (CpSampler::Active().load(std::memory_order_relaxed)) CpSampler::CpOpcode().store(opcode, std::memory_order_relaxed);
			const bool rf_op = RfOp::Enabled(); // S412 (KYTY_DIAG_RFOP): the opcode and packet a page read fault is attributed to
			if (rf_op) RfOp::BeginPacket(opcode, packet, remaining_dw);
			packet_dw = handler(*this, packet_header & ~1u, packet + 1, remaining_dw, total_dw) + 1;
			if (rf_op) RfOp::EndPacket();
			if (CpSampler::Active().load(std::memory_order_relaxed)) CpSampler::CpOpcode().store(CpSampler::kNoOpcode, std::memory_order_relaxed);
		}
		EXIT_IF(packet_dw > remaining_dw);
		if (execution.m_suspended) {
			// Defense in depth: the op above should already have been preceded by a flush (it is
			// not a batchable draw opcode, or it *is* one and decided for itself whether to defer);
			// flush again before giving up the command buffer across a suspend/resume boundary
			// that may span an arbitrary amount of real time.
			PipeDrain(2, 0x1f2); // S479
			if (RdHazard::Enabled()) RdHazard::AtCallEnd(); // S477 D2 DIAG
			return;
		}
		cursor.offset_dw += packet_dw;
		execution.m_made_progress = true;
		if (!execution.m_next_buffer.empty()) {
			// Chains and taken branches reuse the fetcher; only calls retain a return cursor.
			if (execution.m_chain) {
				cursor = {execution.m_next_buffer};
			} else {
				execution.m_buffer_stack.push_back({execution.m_next_buffer});
			}
			execution.m_next_buffer = {};
		}
	}
	PipeDrain(2, 0x1f2); // S479
	if (RdHazard::Enabled()) RdHazard::AtCallEnd(); // S477 D2 DIAG
}

// S479: the state of the CP belongs to the CP thread. A pipeline worker works on a snapshot and never gets here; the gate says so loudly if it ever does (S478: a
// scout that wrote m_index_type_and_size made the GPU run twice as long).
template <class T, class V>
static void SetCpMember(T& member, V value) {
	EXIT_IF(!CpPipe::WriteCpMember(member, value));
}

void CommandProcessor::SetIndexType(uint32_t index_type_and_size) {
	SetCpMember(m_ds.index_type_and_size, index_type_and_size & 0x3u);
}

void CommandProcessor::SetIndexBaseAddress(uint64_t index_base_addr) {
	SetCpMember(m_ds.index_base_addr, index_base_addr);
}

void CommandProcessor::SetIndexBufferSize(uint32_t index_buffer_size) {
	SetCpMember(m_ds.index_buffer_size, index_buffer_size);
}

void CommandProcessor::SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr) {
	SetCpMember(m_ds.draw_indirect_base, draw_indirect_args_base_addr);
}

void CommandProcessor::SetDispatchIndirectArgsBaseAddress(
    uint64_t dispatch_indirect_args_base_addr) {
	SetCpMember(m_ds.dispatch_indirect_base, dispatch_indirect_args_base_addr);
}

void CommandProcessor::SetNumInstances(uint32_t num_instances) {
	if (num_instances == 0) {
		num_instances = 1;
	}

	SetCpMember(m_ds.num_instances, num_instances);
	SetCpMember(m_num_instances_addr, uint64_t {0});
	m_ni_front.OnPacket(); // S577 (the gate let SetCpMember through: the front's state): the lazy view forgets the indirect draw's record too
}

uint32_t CommandProcessor::NumInstances() {
	if (m_num_instances_addr != 0) {
		FrameTiming::Add(FrameTiming::Counter::NumInstEagerN);
		SetCpMember(m_ds.num_instances, *reinterpret_cast<const uint32_t*>(m_num_instances_addr));
		SetCpMember(m_num_instances_addr, uint64_t {0});
		// S577: with KYTY_NUMINST_LAZY=1 the front never gets here; a draw outside the front's items (none known) resolved the register, so the lazy view
		// must not read the record a second time. shadow keeps it: its front reads here on purpose.
		if (NumInstLazy::EnvMode() == NumInstLazy::Mode::On) m_ni_front.pending_addr = 0;
	}
	return m_ds.num_instances;
}

// S577 (KYTY_NUMINST_LAZY=1|shadow, numInstLazy.h): the recorder executes the first item of an epoch that inherits the count. The read is the one the
// unpiped CP does at the draw (it may fault and drain the GPU queue, on this thread); every later item of the epoch takes the memo.
uint32_t CommandProcessor::ResolveLazyNumInstances(const CpPipe::CpDrawState& ds) {
	using C        = FrameTiming::Counter;
	auto&      watch   = NumInstLazy::GlobalWatch();
	const bool tainted = watch.TaintedFor(ds.ni_epoch);
	bool       did_read = false;
	const auto t0       = FrameTiming::SteadyNowNs();
	const uint32_t lazy = m_ni_resolver.Resolve(
	    ds.ni_addr, ds.ni_epoch, [](uint64_t addr) { return *reinterpret_cast<const volatile uint32_t*>(addr); }, &did_read);
	if (did_read) {
		FrameTiming::Add(C::NumInstReadN);
		FrameTiming::Add(C::NumInstReadNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
		if (tainted) FrameTiming::Add(C::NumInstTaintUseN);
		if (watch.epoch.load(std::memory_order_relaxed) == ds.ni_epoch) watch.Disarm();
	} else {
		FrameTiming::Add(C::NumInstHitN);
	}
	if (NumInstLazy::EnvMode() == NumInstLazy::Mode::Shadow) {
		FrameTiming::Add(C::NumInstShadowN);
		if (lazy != ds.num_instances) {
			FrameTiming::Add(C::NumInstNeN);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 16) {
				LOGF("[numinst-lazy] shadow: addr=0x%" PRIx64 " epoch=%" PRIu64 " eager=%u lazy=%u read=%d tainted=%d\n", ds.ni_addr, ds.ni_epoch,
				     ds.num_instances, lazy, did_read ? 1 : 0, tainted ? 1 : 0);
				std::fprintf(stderr, "[numinst-lazy] shadow: addr=0x%" PRIx64 " epoch=%" PRIu64 " eager=%u lazy=%u read=%d tainted=%d\n", ds.ni_addr,
				             ds.ni_epoch, ds.num_instances, lazy, did_read ? 1 : 0, tainted ? 1 : 0);
			}
		}
		return ds.num_instances; // shadow draws with the old value
	}
	return lazy;
}

void CommandProcessor::SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
                                      const volatile void* address) {
	uint64_t value = 0;

	switch (op) {
		case 0x00:
			m_predicate_skip = false;
			return;
		case 0x01: {
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			m_renderer.GetOcclusionQueries().Publish();
			// One begin/end pair per DB; bit 63 marks each counter ready.
			constexpr uint64_t ready_bit = 1ull << 63u;
			const auto* results = reinterpret_cast<const volatile uint64_t*>(address);
			for (uint32_t db = 0; db < 16u; db++) {
				const auto begin = results[db * 2u];
				const auto end   = results[db * 2u + 1u];
				if ((begin & end & ready_bit) == 0) {
					if (wait_op == 0) {
						SuspendPm4();
					} else {
						m_predicate_skip = false;
					}
					return;
				}
				value += end - begin;
			}
		} break;
		case 0x03:
			// The wait selector applies only to Z-pass query readiness.
			EXIT_NOT_IMPLEMENTED(address == nullptr);
			value = *reinterpret_cast<const volatile uint64_t*>(address);
			PromoteGuestWrites(); // S601
			break;
		default: EXIT("unknown predication op: 0x%08" PRIx32 "\n", op);
	}
	switch (condition) {
		case 0x00: m_predicate_skip = (value != 0); break;
		case 0x01: m_predicate_skip = (value == 0); break;
		default: EXIT("unknown predication condition: 0x%08" PRIx32 "\n", condition);
	}
}

// S540 (KYTY_FLUSH_DRAWS=N, port of upstream 7d4e0d9b): submits the recorded work every N draws so the host GPU starts before a long buffer ends.
void CommandProcessor::CountDraw() {
	if (m_flush_draws == 0) {
		return;
	}
	if (m_draws == m_flush_draws) {
		GetScheduler().FlushEveryDraws();
		m_draws = 0;
	}
	m_draws++;
}

void CommandProcessor::DrawIndex(DrawIndexArgs args) {
	CountDraw();
	args.index_type_and_size = Ds().index_type_and_size;
	if (args.instance_count == 0) {
		args.instance_count = ExecNumInstances();
	}
	m_renderer.GetRenderExecutor().DrawIndex(CurrentBuffer(), args);
}

const void* CommandProcessor::IndexAddress(uint32_t first_index) const {
	const auto& ds         = Ds(); // S549: the executed item's copy when the front runs on its own thread
	uint64_t    index_size = 0;
	switch (ds.index_type_and_size) {
		case 0: index_size = 2; break;
		case 1: index_size = 4; break;
		case 2: index_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", ds.index_type_and_size);
	}
	return reinterpret_cast<const void*>(ds.index_base_addr + first_index * index_size);
}

void CommandProcessor::DrawIndexOffset(uint32_t index_offset, uint32_t index_count) {
	DrawIndex({.index_count = index_count, .index_addr = IndexAddress(index_offset)});
}

static void PatchIndirectDrawOffsets(CommandProcessor& cp, IndirectDrawRegisters registers,
                                     uint32_t vertex_offset, uint32_t instance_offset,
                                     uint32_t first_index, uint32_t draw_index) {
	const auto write = [&](uint32_t location, uint32_t value) {
		if (location == Pm4::SH_NOP) {
			return;
		}
		EXIT_NOT_IMPLEMENTED(location >= Pm4::SH_NUM ||
		                     g_hw_sh_indirect_func[location] == nullptr);
		g_hw_sh_indirect_func[location](cp, location, value);
	};
	write(registers.vertex_offset, vertex_offset);
	write(registers.instance_offset, instance_offset);
	write(registers.index_offset, first_index);
	write(registers.draw_index, draw_index);
}

// Writes the GS user-data slots the packet patches; false when the shaders cannot read the
// offsets from GPU records: a first-index or subdraw-index destination, or another stage's
// user data.
static bool IndirectDrawOffsets(IndirectDrawRegisters registers, ShaderIndirectOffsets& offsets) {
	if (registers.index_offset != Pm4::SH_NOP || registers.draw_index != Pm4::SH_NOP) {
		return false;
	}
	for (const auto [location, slot]: {std::pair {registers.vertex_offset, &offsets.vertex},
	                                   std::pair {registers.instance_offset, &offsets.instance}}) {
		if (location == Pm4::SH_NOP) {
			continue;
		}
		const uint32_t index = location - Pm4::SPI_SHADER_USER_DATA_GS_0;
		if (index >= HW::UserSgprInfo::SGPRS_MAX) {
			return false;
		}
		*slot = static_cast<uint8_t>(index);
	}
	return true;
}


void CommandProcessor::DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
                                    uint32_t draw_initiator, bool indexed) {
	const auto args_size =
	    indexed ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand);
	DrawIndirectMulti(data_offset, 1, nullptr, args_size, registers, draw_initiator, indexed);
}

void CommandProcessor::DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
                                         const volatile uint32_t* count_addr,
                                         uint32_t stride_in_bytes, IndirectDrawRegisters registers,
                                         uint32_t draw_initiator, bool indexed) {
	CountDraw();
	EXIT_NOT_IMPLEMENTED((draw_initiator & ~0x20u) != (indexed ? 0u : 2u));
	EXIT_NOT_IMPLEMENTED(m_ds.draw_indirect_base == 0);
	// The guest records match Vulkan's indirect commands byte for byte.
	EXIT_NOT_IMPLEMENTED((stride_in_bytes & 3u) != 0 ||
	                     stride_in_bytes < (indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                                                : sizeof(vk::DrawIndirectCommand)));

	const auto draw_count = count_addr == nullptr
	                            ? max_count_or_count
	                            : std::min(uint32_t {*count_addr}, max_count_or_count);
	if (draw_count == 0) {
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::IndirectDraws, draw_count);

	auto&      executor  = m_renderer.GetRenderExecutor();
	const auto args_addr = m_ds.draw_indirect_base + data_offset;
	// NUM_INSTANCES keeps the last draw's instance count.
	m_num_instances_addr = args_addr + static_cast<uint64_t>(draw_count - 1) * stride_in_bytes +
	                       offsetof(vk::DrawIndirectCommand, instanceCount);
	// S577 (KYTY_NUMINST_LAZY): the lazy view gets a new epoch; the watch over the record is armed once this draw is recorded (GPU writes after it
	// are the ones the old path's read did not see).
	m_ni_front.OnIndirectDraw(m_num_instances_addr);
	struct ArmWatch {
		uint64_t addr, epoch;
		~ArmWatch() {
			if (NumInstLazy::EnvMode() != NumInstLazy::Mode::Off) NumInstLazy::GlobalWatch().Arm(addr, epoch);
		}
	} const arm_watch {m_num_instances_addr, m_ni_front.epoch};
	// S472 (KYTY_INDIRECT_GPU_MESH=1): an indexed one-draw indirect whose host shader is a mesh shader is drawn from the guest args on the GPU
	// (no CPU read of the args). The executor declines every other draw (ig_request.declined); those take the records path below.
	MeshIndirectRequest mesh_request;
	mesh_request.mode              = MeshIndirect::EnvMode();
	mesh_request.index_buffer_size = m_ds.index_buffer_size;
	if (mesh_request.mode == MeshIndirect::Mode::On && indexed && draw_count == 1 && count_addr == nullptr &&
	    registers.vertex_offset == Pm4::SH_NOP && registers.instance_offset == Pm4::SH_NOP &&
	    registers.index_offset == Pm4::SH_NOP && m_ds.index_buffer_size != 0 && m_ds.index_type_and_size < 2 &&
	    m_renderer.GetGraphics().draw_indirect_first_instance) {
		IndirectGpuRequest ig_request;
		ig_request.mode        = IndirectGpu::Mode::On;
		ig_request.args_addr   = args_addr;
		mesh_request.args_addr = args_addr;
		FrameTiming::Add(FrameTiming::Counter::IndirectGpuCandidateN);
		DrawIndex({.index_count    = m_ds.index_buffer_size,
		           .index_addr     = reinterpret_cast<const void*>(m_ds.index_base_addr),
		           .instance_count = 1,
		           .offset_source  = DrawOffsetSource::IndirectArgs,
		           .indirect_gpu   = &ig_request,
		           .mesh_indirect  = &mesh_request});
		if (!ig_request.declined) {
			FrameTiming::Add(FrameTiming::Counter::IndirectGpuN);
			return;
		}
	}
	// The GPU reads the records if the shaders allow; the VS prolog removes Vulkan's bases
	// from the index VGPRs and copies them only to the user SGPRs the packet patches.
	DrawIndirectArgs records {args_addr, draw_count, stride_in_bytes};
	if (IndirectDrawOffsets(registers, records.offsets)) {
		if (!indexed && executor.DrawAuto(CurrentBuffer(), {.indirect = records})) {
			return;
		}
		if (indexed && m_ds.index_buffer_size != 0 &&
		    executor.DrawIndex(CurrentBuffer(), {.index_count         = m_ds.index_buffer_size,
		                                         .index_addr          = IndexAddress(0),
		                                         .index_type_and_size = m_ds.index_type_and_size,
		                                         .indirect            = records})) {
			return;
		}
	}

	const uint32_t max_indices = m_ds.index_buffer_size != 0 ? m_ds.index_buffer_size : UINT32_MAX;
	for (uint32_t i = 0; i < draw_count; i++) {
		const auto addr = args_addr + static_cast<uint64_t>(i) * stride_in_bytes;
		if (!indexed) {
			const auto& args = *reinterpret_cast<const vk::DrawIndirectCommand*>(addr);
			PatchIndirectDrawOffsets(*this, registers, args.firstVertex, args.firstInstance, 0, i);
			executor.DrawAuto(CurrentBuffer(), {.vertex_count   = args.vertexCount,
			                                    .instance_count = args.instanceCount,
			                                    .first_vertex   = args.firstVertex,
			                                    .first_instance = args.firstInstance,
			                                    .offset_source  = DrawOffsetSource::IndirectArgs});
			continue;
		}

		const auto& args = *reinterpret_cast<const vk::DrawIndexedIndirectCommand*>(addr);
		PatchIndirectDrawOffsets(*this, registers, static_cast<uint32_t>(args.vertexOffset),
		                         args.firstInstance, args.firstIndex, i);
		executor.DrawIndex(CurrentBuffer(), {.index_count = std::min(args.indexCount, max_indices),
		                                     .index_addr  = IndexAddress(args.firstIndex),
		                                     .instance_count      = args.instanceCount,
		                                     .index_type_and_size = m_ds.index_type_and_size,
		                                     .base_vertex         = args.vertexOffset,
		                                     .first_instance      = args.firstInstance,
		                                     .offset_source = DrawOffsetSource::IndirectArgs});
	}
}

void CommandProcessor::DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y,
                                      uint32_t thread_group_z, uint32_t mode) {
	ExecSh().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchDirect(CurrentBuffer(), thread_group_x, thread_group_y,
	                                              thread_group_z, mode, IsAsyncComputeQueue());
}

void CommandProcessor::ProcessPssr(const Pssr::Request& request) {
	EXIT_NOT_IMPLEMENTED(IsAsyncComputeQueue());
	m_renderer.GetRenderExecutor().ProcessPssr(CurrentBuffer(), request);
}

void CommandProcessor::DispatchIndirect(uint64_t args_addr, uint32_t mode) {
	EXIT_NOT_IMPLEMENTED(args_addr == 0 || (args_addr & 3u) != 0);
	if ((mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0) {
		const auto* args = reinterpret_cast<const vk::DispatchIndirectCommand*>(args_addr);
		DispatchDirect(args->x, args->y, args->z, mode);
		return;
	}
	ExecSh().SetCsWaveSize(Pm4::ComputeWaveSize(mode));
	m_renderer.GetRenderExecutor().DispatchIndirect(CurrentBuffer(), args_addr, mode,
	                                                IsAsyncComputeQueue());
}

void CommandProcessor::DrawIndexAuto(DrawAutoArgs args) {
	CountDraw();
	if (args.instance_count == 0) {
		args.instance_count = ExecNumInstances();
	}
	m_renderer.GetRenderExecutor().DrawAuto(CurrentBuffer(), args);
}

void CommandProcessor::WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index) {
	BufferFlush();

	m_renderer.GetVideoOut().WaitFlipDone(static_cast<int>(video_out_handle),
	                                      static_cast<int>(display_buffer_index));
}

void CommandProcessor::ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address,
                                  uint64_t data, uint32_t interrupt_context_id) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::ReleaseMem()\n"
		     "\t data_sel           = %" PRIu32 "\n"
		     "\t interrupt_selector = %" PRIu32 "\n"
		     "\t address            = 0x%016" PRIx64 "\n"
		     "\t data               = 0x%016" PRIx64 "\n"
		     "\t interrupt_context  = 0x%08" PRIx32 "\n",
		     data_sel, interrupt_selector, reinterpret_cast<uint64_t>(address), data,
		     interrupt_context_id);
	}

	// QueuedInterrupt: kNone, kOnly, kAfterWriteConfirm, kDataAfterWriteConfirm, kSendIntCtx.
	EXIT_NOT_IMPLEMENTED(interrupt_selector > 4);
	const bool interrupt = (interrupt_selector != 0 && interrupt_selector != 3);
	// kSendIntCtx has no label, and kOnly writes none on graphics queues.
	if (address == nullptr || interrupt_selector == 4 ||
	    (interrupt_selector == 1 && !IsAsyncComputeQueue())) {
		data_sel = 0;
	}
	if (data_sel == 0 && !interrupt) {
		return;
	}

	// Dumped occlusion counters are written before any end-of-pipe signal reaches the guest.
	m_renderer.GetOcclusionQueries().Publish();

	// A label beside GPU-written bytes goes to the backing store, so they are not downloaded.
	// Other labels go through the guest mapping: its fault keeps a clean cached copy current and
	// downloads GPU-written bytes under the label before the store.
	const auto write_label = [this, address](const void* value, uint64_t size) {
		const auto vaddr   = reinterpret_cast<uint64_t>(address);
		auto&      buffers = m_renderer.GetBufferCache();
		if (!GuestRange {vaddr, size}.Valid() || !buffers.IsRegionGpuModified(vaddr, size) ||
		    buffers.HasGpuDirtyBytes(vaddr, size) ||
		    !LibKernel::Memory::TryWriteBacking(vaddr, value, size)) {
			std::memcpy(address, value, size);
		}
	};
	// EndOfPipeData: kNone, kImmediate32b, kImmediate64b, kRefClk; 5 reads GDS.
	switch (data_sel) {
		case 0: break;
		case 1: {
			const auto value = static_cast<uint32_t>(data);
			write_label(&value, sizeof(value));
			break;
		}
		case 3: data = Sync::ReadReferenceClock(); [[fallthrough]];
		case 2: write_label(&data, sizeof(data)); break;
		case 5: {
			GetScheduler().Finish();
			const auto gds = static_cast<uint32_t>(data);
			Sync::ReadGds(*m_renderer.GetBufferCache().GetGdsBuffer(),
			              static_cast<uint32_t*>(address), gds & 0xffffu, gds >> 16u);
			break;
		}
		default: EXIT("unknown release_mem data selector: %" PRIu32 "\n", data_sel);
	}

	// GuestGpu::Process's end-of-slice flush submits the work this interrupt waits on.
	if (interrupt) {
		GetScheduler().DeferPriorityOperation(
		    [&renderer = m_renderer, event_id = m_interrupt_event_id, interrupt_context_id] {
			    renderer.TriggerInterrupt(event_id, interrupt_context_id);
		    });
	}
}

void CommandProcessor::TriggerEvent(uint32_t event_type, uint32_t event_index,
                                    uint64_t event_address) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::TriggerEvent()\n"
		     "\t event_type  = 0x%08" PRIx32 "\n"
		     "\t event_index = 0x%08" PRIx32 "\n"
		     "\t address     = 0x%016" PRIx64 "\n",
		     event_type, event_index, event_address);
	}

	const auto valid_cache_event_index = event_index == 0x00000000 || event_index == 0x00000007;
	// Partial flushes and cache writebacks only order GPU work, which the renderer's resource
	// barriers already order.
	switch (event_type) {
		// CsPartialFlush, GsPartialFlush, PsPartialFlush.
		case 0x00000007:
		case 0x0000000f:
		case 0x00000010: break;
		// CbDbDataWritebackInvalidate, CbDataWritebackInvalidate, DbDataWritebackInvalidate,
		// DbMetadataWritebackInvalidate, CbMetadataWritebackInvalidate.
		case 0x00000016:
		case 0x00000031:
		case 0x0000002a:
		case 0x0000002c:
		case 0x0000002e:
			if (!valid_cache_event_index) {
				EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type,
				     event_index);
			}
			break;
		case 0x0000000d:
		case 0x0000000e:
		case 0x00000012:
		case 0x00000017:
		case 0x00000018:
		case 0x00000019:
		case 0x0000001a:
		case 0x0000001b:
		case 0x00000038:
		case 0x0000003a:
			LOGF("\t temporary: ignoring unsupported event_write type 0x%08" PRIx32
			     ", index 0x%08" PRIx32 "\n",
			     event_type, event_index);
			break;
		case 0x00000039: {
			if (event_index != 0x00000001 || event_address == 0 || (event_address & 0x7u) != 0) {
				EXIT("invalid occlusion-counter dump: index=0x%08" PRIx32 ", address=0x%016" PRIx64
				     "\n",
				     event_index, event_address);
			}
			if (auto& queries = m_renderer.GetOcclusionQueries(); queries.Enabled()) {
				queries.Dump(event_address);
				break;
			}
			static std::once_flag warning_once;
			std::call_once(warning_once, [] {
				std::printf("Warning: game uses occlusion queries, which are treated as always "
				            "visible without --occlusion-queries; GPU usage may be higher and FPS "
				            "may be lower.\n");
			});

			// Without host occlusion queries, publish an always-visible result. The PS5 layout
			// contains one interleaved begin/end pair per DB, and bit 63 marks a result ready.
			constexpr uint64_t ready_bit    = 1ull << 63u;
			constexpr uint64_t counter_mask = ready_bit - 1u;
			auto*              results      = reinterpret_cast<volatile uint64_t*>(event_address);
			const auto         value        = ready_bit | m_synthetic_occlusion_counter;
			for (uint32_t db = 0; db < 16u; db++) {
				results[db * 2u] = value;
			}
			m_synthetic_occlusion_counter = (m_synthetic_occlusion_counter + 1u) & counter_mask;
			break;
		}
		default:
			EXIT("unknown event type: 0x%08" PRIx32 ", 0x%08" PRIx32 "\n", event_type, event_index);
	}
}

void CommandProcessor::Flip(int handle, int index, int flip_mode, int64_t flip_arg) {
	if (graphics_debug_dump_enabled()) {
		LOGF("CommandProcessor::Flip()\n");
	}
	m_renderer.GetOcclusionQueries().Publish();

	auto& command = CurrentBuffer();
	auto  request = Sync::PrepareVideoOutFlip(command, handle, index, flip_mode, flip_arg);
	GetScheduler().DeferPriorityOperation(
	    [&renderer = m_renderer, request] { renderer.GetVideoOut().CompleteFlip(request); });
	GetScheduler().SetSubmitReason(GpuDone::SubmitReason::Flip); // S410
	GetScheduler().Flush();
}

void CommandProcessor::PrepareCpuFlip(uint64_t request_id) {
	auto& command = CurrentBuffer();
	m_renderer.GetVideoOut().PrepareFlip(request_id, command);
	GetScheduler().DeferPriorityOperation(
	    [this, request_id] { m_renderer.GetVideoOut().CompleteFlip(request_id); });
	GetScheduler().SetSubmitReason(GpuDone::SubmitReason::Flip); // S410
	GetScheduler().Flush();
}

bool GuestGpu::IsGpuThread() noexcept {
	return g_gpu_state != nullptr;
}

void GuestGpu::Krf4PollFlips() {
	namespace RfP = ShaderRecompiler::IR::RfGpuPatch;
	if (!RfP::Krf3EnvMode().v4 || !IsGpuThread()) return; // one reader: the CP thread
	RfP::Krf4DrainFlips(RfP::GuestGuard(), RfP::Krf4Flips(), [](const RfP::GuestPageGuard::FlipResult& fr) {
		if (fr.starved) FrameTiming::Add(FrameTiming::Counter::Krf4StarveFlipN);
		if (fr.storms != 0) FrameTiming::Add(FrameTiming::Counter::Krf4StormN, fr.storms);
	});
}

} // namespace Libs::Graphics
