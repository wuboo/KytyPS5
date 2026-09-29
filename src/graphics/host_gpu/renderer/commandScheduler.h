#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/render.h"

#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
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
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
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
	};
	void SubmitThread(std::stop_token stop);
	void DrainAsyncSubmits();
	void QueueSubmitNow(SubmitJob& job);
	bool                    m_async_submit = false;
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
	bool HasOperationsAtCurrentTick();
	uint32_t                              m_flush_requests = 0;
	std::chrono::steady_clock::time_point m_first_flush_request {};
	void PriorityOperationsThread(std::stop_token stop);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
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
