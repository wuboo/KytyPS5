#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H

#include "common/assert.h"
#include "graphics/guest_gpu/command_processor/cpPipe.h"
#include "graphics/guest_gpu/command_processor/numInstLazy.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pssr.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics {

class GuestGpu;

// S540 (KYTY_FLUSH_DRAWS=N, default 0 = off): read once per command processor.
inline uint32_t FlushDrawsFromEnv() {
	const char* value = std::getenv("KYTY_FLUSH_DRAWS");
	return value != nullptr ? static_cast<uint32_t>(std::strtoul(value, nullptr, 10)) : 0u;
}

bool TestWaitRegMemValue(uint64_t value, uint64_t ref, uint64_t mask, uint32_t func);

enum class Pm4ProcessResult { Complete, Blocked };

enum class ContextStateOperation : uint32_t {
	Clear     = 0,
	Push      = 1,
	Pop       = 2,
	PushClear = 3,
};

struct IndirectDrawRegisters {
	uint32_t vertex_offset;
	uint32_t instance_offset;
	uint32_t index_offset;
	uint32_t draw_index;
};

class Pm4Execution {
public:
	[[nodiscard]] bool MadeProgress() const noexcept { return m_made_progress; }

private:
	friend class CommandProcessor;

	struct BufferCursor {
		std::span<const uint32_t> commands;
		uint32_t                  offset_dw = 0;
	};

	std::vector<BufferCursor> m_buffer_stack;
	std::span<const uint32_t> m_next_buffer;
	bool                      m_chain         = false;
	bool                      m_suspended     = false;
	bool                      m_made_progress = false;
};

class CommandProcessor {
public:
	CommandProcessor(RenderContext& renderer, int interrupt_event_id); // cpPipe.cpp (m_pipe holds a type that is complete there)
	~CommandProcessor();

	KYTY_CLASS_NO_COPY(CommandProcessor);

	void Reset();
	void ApplyContextStateOperation(ContextStateOperation operation);

	void            BufferInit();
	void            BufferFlush();
	void               BufferFlushNow();
	void               FlushPending();
	[[nodiscard]] bool FlushRequested() const;
	void            BufferFlushAndWait();
	void            BufferWait();
	HW::Context&    GetCtx() { EXIT_IF(!CpPipe::MayWriteCpState()); return m_ctx; }
	HW::UserConfig& GetUcfg() { EXIT_IF(!CpPipe::MayWriteCpState()); return m_ucfg; }
	HW::Shader&     GetShCtx() { EXIT_IF(!CpPipe::MayWriteCpState()); return m_sh_ctx; }

	void SetIndexType(uint32_t index_type_and_size);
	void SetIndexBaseAddress(uint64_t index_base_addr);
	void SetIndexBufferSize(uint32_t index_buffer_size);
	void SetDrawIndirectArgsBaseAddress(uint64_t draw_indirect_args_base_addr);
	void SetDispatchIndirectArgsBaseAddress(uint64_t dispatch_indirect_args_base_addr);
	[[nodiscard]] uint64_t GetDispatchIndirectArgsBaseAddress() const {
		return Ds().dispatch_indirect_base;
	}
	void SetNumInstances(uint32_t num_instances);
	void DrawIndex(DrawIndexArgs args);
	void DrawIndexOffset(uint32_t index_offset, uint32_t index_count);
	void DrawIndexAuto(DrawAutoArgs args);
	void DrawIndirect(uint32_t data_offset, IndirectDrawRegisters registers,
	                  uint32_t draw_initiator, bool indexed);
	void DrawIndirectMulti(uint32_t data_offset, uint32_t max_count_or_count,
	                       const volatile uint32_t* count_addr, uint32_t stride_in_bytes,
	                       IndirectDrawRegisters registers, uint32_t draw_initiator, bool indexed);
	void ReleaseMem(uint32_t data_sel, uint32_t interrupt_selector, void* address, uint64_t data,
	                uint32_t interrupt_context_id);
	void Flip(int handle, int index, int flip_mode, int64_t flip_arg);
	void PrepareCpuFlip(uint64_t request_id);
	void DispatchDirect(uint32_t thread_group_x, uint32_t thread_group_y, uint32_t thread_group_z,
	                    uint32_t mode);
	void DispatchIndirect(uint64_t args_addr, uint32_t mode);
	void WaitFlipDone(uint32_t video_out_handle, uint32_t display_buffer_index);
	void ProcessPssr(const Pssr::Request& request);
	void TriggerEvent(uint32_t event_type, uint32_t event_index, uint64_t event_address = 0);

	void WaitForRewind(bool valid);

	template <typename T>
	void WaitRegMem(uint32_t func, const T* addr, T ref, T mask, uint32_t wait_op);
	void MemSemaphore(uint64_t* counter, bool signal, bool write);
	void WriteData(uint32_t* dst, const uint32_t* src, uint32_t dw_num, uint32_t write_control);
	void WriteReferenceClock(uint64_t dst_address, uint32_t num_bytes);
	void GetLodStats(volatile uint32_t* dst, uint32_t size, bool reset);
	void DmaData(uint8_t dst_sel, uint64_t dst_address_or_offset, uint8_t src_sel,
	             uint64_t src_address_or_offset_or_immediate, uint32_t num_bytes);
	void SetPredication(uint32_t condition, uint32_t op, uint32_t wait_op,
	                    const volatile void* address);
	[[nodiscard]] bool ShouldSkipPredicatedPackets() const { return m_predicate_skip; }

	Pm4ProcessResult Process(Pm4Execution& execution, std::span<const uint32_t> commands);

	// S479 (KYTY_CP_PIPE, cpPipe.cpp): the FIFO between the front (the PM4 parser) and the recorder (the handlers that draw and signal). Both run on the CP thread,
	// or (S549, KYTY_CP_FRONT_THREAD) the front runs on its own thread and Thread_Gpu only records.
	[[nodiscard]] bool PipeOn() const { return m_pipe != nullptr; }
	[[nodiscard]] bool PipeFrontOn() const;
	void               PipeDrain(unsigned reason, uint32_t key); // runs every item in the FIFO; reason: 0 an opcode that must see the draws done, 1 wait, 2 end of stream; key: what for (diag histogram)
	void             ProcessIndirectBuffer(std::span<const uint32_t> commands, bool chain);

	[[nodiscard]] bool IsAsyncComputeQueue() const { return m_interrupt_event_id >= 0x20; }

private:
	void ProcessPm4(Pm4Execution& execution);
	void PipeInit();
	void FrontParse(Pm4Execution& execution); // S549: graphicsRun.cpp
	// S549 (KYTY_CP_FRONT_THREAD, cpPipe.cpp): Thread_Gpu hands the stream to the front thread and executes the FIFO until the front's End item.
	void     PipeFrontRun(Pm4Execution& execution, GuestGpu* gpu);
	void     PipeFrontThread();
	void     PipeFrontExecute(uint64_t seq);
	void     PipeFrontRelease(uint64_t done);
	// Front side: true when this thread is the front of this CP. A Sync item: the recorder runs `handler` on the live state while the front waits; returns its length.
	[[nodiscard]] bool OnFrontThread() const;
	uint32_t PipeFrontSync(uint32_t kind, uint32_t opcode, const uint32_t* packet, uint32_t remaining_dw, uint32_t total_dw);
	void     PipeFrontDrain(unsigned reason, uint32_t key);
	void     PipeFrontEnd();
	void     PipeFrontEnqueue(CpPipe::OpClass cls, uint32_t opcode, const uint32_t* packet, uint32_t packet_dw);
	[[nodiscard]] bool PipeFrontCommandsPending() const;
	void PipeEnqueue(CpPipe::OpClass cls, uint32_t opcode, const uint32_t* packet, uint32_t packet_dw);
	void PipeExecuteFront();
	bool PipeOverlaps(uint64_t address, uint64_t size) const;
	bool PipeWaitVirtual(uint64_t address, uint64_t size, uint32_t func, uint64_t ref, uint64_t mask, bool* satisfied);
	[[nodiscard]] HW::Shader&     ExecSh() { return m_exec_sh != nullptr ? *m_exec_sh : m_sh_ctx; }
	[[nodiscard]] HW::UserConfig& ExecUcfg() { return m_exec_ucfg != nullptr ? *m_exec_ucfg : m_ucfg; }
	// S549: the draw state of the item being executed (the recorder), else the CP's own.
	[[nodiscard]] const CpPipe::CpDrawState& Ds() const { return CpPipe::SelectDrawState(m_exec_ds, m_ds); }
	void SuspendPm4();
	void                CountDraw();
	uint32_t            NumInstances();
	// S549: the executed item's instance count (the front resolved it at the snapshot), else NumInstances(). S577 (KYTY_NUMINST_LAZY): an item whose
	// copy carries a record address resolves it here, on the recorder (graphicsRun.cpp).
	uint32_t ExecNumInstances() {
		if (m_exec_ds == nullptr) return NumInstances();
		return m_exec_ds->ni_addr != 0 ? ResolveLazyNumInstances(*m_exec_ds) : m_exec_ds->num_instances;
	}
	uint32_t ResolveLazyNumInstances(const CpPipe::CpDrawState& ds);
	// S577: the front's NUM_INSTANCES snapshot for a Draw/Dispatch item of the front thread (cpPipe.cpp).
	void SnapshotNumInstances(CpPipe::CpDrawState& ds);
	const void*         IndexAddress(uint32_t first_index) const;
	CommandScheduler&   GetScheduler() const { return m_renderer.GetCommandScheduler(); }
	CommandBuffer&      CurrentBuffer() { return GetScheduler().Current(); }

	RenderContext& m_renderer;
	HW::Context    m_ctx;
	std::optional<HW::Context> m_saved_ctx;
	HW::UserConfig m_ucfg;
	HW::Shader     m_sh_ctx;
	// Index type, sizes, bases, instance count (S549: one struct, a FIFO item carries a copy).
	CpPipe::CpDrawState m_ds;
	// Persistent draw state. Indirect draws leave the address of their last instance count,
	// read only when a later draw inherits it.
	uint64_t m_num_instances_addr = 0;
	// S577 (KYTY_NUMINST_LAZY): the front's lazy view of the register (set with m_num_instances_addr by an indirect draw, cleared by NUM_INSTANCES and
	// Reset, never by a read) and the recorder's memo of the resolved epoch.
	NumInstLazy::FrontState m_ni_front;
	NumInstLazy::Resolver   m_ni_resolver;

	const int m_interrupt_event_id;
	const uint32_t m_flush_draws = FlushDrawsFromEnv(); // S540 (KYTY_FLUSH_DRAWS), 0 is off
	uint32_t  m_draws            = 0;                   // since the buffer start or CountDraw's flush
	uint64_t  m_synthetic_occlusion_counter = 0;
	bool      m_predicate_skip              = false;

	struct PipeState;
	std::unique_ptr<PipeState> m_pipe;
	HW::Shader*                m_exec_sh   = nullptr; // S479: the snapshot of the deferred draw / dispatch being executed
	HW::UserConfig*            m_exec_ucfg = nullptr;
	const CpPipe::CpDrawState* m_exec_ds   = nullptr; // S549: the draw state of the item the recorder executes (front thread on)
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_COMMAND_PROCESSOR_H
