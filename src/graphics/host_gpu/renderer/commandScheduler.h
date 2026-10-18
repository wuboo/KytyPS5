#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/gpuDone.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	CommandScheduler(RenderContext& context, GraphicContext& graphics);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           BeginRendering(const RenderState& state);
	void           EndRendering();
	void           Flush();
	void           Flush(SubmitInfo& submit);
	// Lazy flush (KYTY_OPT_OFF=lazy_flush disables): the command processor asks for a flush at
	// the end of every PM4 submission and ReleaseMem; batch those into one submit, made once
	// enough are pending, after 1 ms, or when the GPU thread is about to wait.
	void               RequestFlush();
	void               FlushPending();
	[[nodiscard]] bool FlushRequested() const noexcept { return m_flush_requests != 0; }
	void           FlushAndWait();
	// S326 (KYTY_EAGER_SUBMIT=1): at a safe point (before a draw / dispatch) submits the open buffer when the GPU queue is empty (eagerSubmit.h).
	void MaybeEagerSplit();
	// S327: a draw needs `next` as the pass state. When that differs from the open pass (which BeginRendering would close anyway) and the
	// eager policy agrees, closes it and splits the buffer here, before anything of the draw is recorded.
	void MaybeEagerSplitAtPass(const RenderState& next);
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Runs on the recording thread just before a command buffer is closed for submission, so
	// callers can append work that must follow everything recorded so far (not re-entrant).
	void SetPreSubmitHook(std::function<void()> hook) { m_pre_submit = std::move(hook); }
	// Hand finished command buffers to a submit thread (KYTY_OPT_OFF=async_submit disables).
	// MoltenVK encodes the whole Metal command buffer inside vkQueueSubmit, which was a quarter of
	// the GPU thread in menus. Submissions keep their tick order; ones that wait on or signal a
	// binary semaphore drain the thread and are submitted inline. Only for the renderer's
	// scheduler (the presenter's waits on swapchain binary semaphores).
	void EnableAsyncSubmit();
	// Returns once every submission handed to the submit thread is on the Vulkan queue.
	void WaitSubmitted() { DrainAsyncSubmits(); }
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	[[nodiscard]] bool RenderingOpen() const;
	[[nodiscard]] bool RenderingAttaches(vk::ImageView view) const;
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	std::function<void()> m_pre_submit;
	bool                  m_in_pre_submit = false;

	struct SubmitJob {
		vk::CommandBuffer buffer;
		SubmitInfo        submit;
		uint64_t          tick     = 0;
		uint32_t          debug_op = 0;
		uint64_t          debug_submit_id = 0;
		uint32_t          debug_args[4]   = {};
		uint64_t          debug_arg4      = 0;
		bool              first_seg       = false; // S328: first submit with work after a drain
		int64_t           queued_ns       = 0; // S132: steady clock at push (async_submit queue wait)
	};
	void SubmitThread(std::stop_token stop);
	void DrainAsyncSubmits();
	void QueueSubmitNow(SubmitJob& job);
	bool                    m_async_submit = false;
	std::atomic<bool>       m_gpu_done_stop {false};            // S406 (KYTY_DIAG_GPU_DONE=1)
	std::unique_ptr<GpuDone::Tracker> m_gpu_done;
	std::mutex              m_submit_mutex;
	std::condition_variable m_submit_cv;
	std::deque<SubmitJob>   m_submit_jobs;
	bool                    m_submit_busy = false;
	std::jthread            m_submit_thread;

	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	void BeginNext();
	// S326 segments: drain end -> next drain (seg_* counters) and the DIAG gap after a drain.
	void NoteDrainBegin();
	void NoteDrainEnd();
	int64_t  m_seg_start_ns = 0;      // 0 = no segment open
	bool     m_seg_first_pending = false;
	uint32_t m_drain_depth = 0;
	uint64_t m_seg_wait_base = 0;     // cp_wait_guest_ns + cp_wait_gpu_ns at the segment start
	void ConsumeSegPassEnd();
	bool     EagerSplitNow(bool pass_closing, bool* closed_pass, bool mid_pass = false);
	uint32_t m_seg_mid_splits = 0;    // S328: mid-pass eager splits in this segment
	int64_t  m_open_since_ns = 0;     // when the open buffer began (eager policy age)
	bool HasOperationsAtCurrentTick();
	uint32_t                              m_flush_requests = 0;
	std::chrono::steady_clock::time_point m_first_flush_request {};
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	std::mutex                   m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	bool                         m_priority_active      = false;
	uint64_t                     m_priority_active_tick = 0;
	OperationState               m_operation_state      = OperationState::Open;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
