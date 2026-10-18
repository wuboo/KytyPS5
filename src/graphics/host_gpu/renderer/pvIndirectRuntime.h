#pragma once

// S408 (KYTY_PV_INDIRECT): GPU side of the prepared args of a VS-store indirect draw (pvIndirect.h has the arithmetic and the entry
// life cycle). Owns the compute pipeline of pv_indirect_prepare.comp, the host ring the prepared args are written to, and the status
// rows (one per flip modulo kStatusRows, one entry per args address) the CPU reads two flips later without waiting.

#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/pvIndirect.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>

namespace Libs::Graphics {
class CommandScheduler;
struct GraphicContext;

class PvIndirectRuntime {
public:
	static constexpr uint32_t kRingSlots     = 4096;
	static constexpr uint32_t kSlotBytes     = 32; // 5 dwords, padded
	static constexpr uint32_t kStatusRows    = 8;
	static constexpr uint32_t kStatusDwords  = 16; // [0] mismatch [1] overflow [2] max instance [3] draws [4..8] in [9..13] out
	static constexpr uint32_t kPushDwords    = 8;

	PvIndirectRuntime(GraphicContext& graphics, CommandScheduler& scheduler);
	~PvIndirectRuntime();
	PvIndirectRuntime(const PvIndirectRuntime&)            = delete;
	PvIndirectRuntime& operator=(const PvIndirectRuntime&) = delete;

	// What a draw hands over: where the guest args sit in the buffer cache and what the entry expects.
	struct Job {
		vk::Buffer           args_buffer = nullptr;
		uint64_t             args_offset = 0;
		PvIndirect::Fields   expected;
		uint32_t             cap      = 0;
		uint32_t             entry_id = 0;
		uint64_t             flip     = 0;
	};
	struct Prepared {
		bool                                  ok = false;
		vk::Buffer                            out_buffer = nullptr; // for vkCmdDrawIndexedIndirect
		uint64_t                              out_offset = 0;
		std::array<uint32_t, kPushDwords>     push {};
		std::array<vk::DescriptorBufferInfo, 3> infos {};
	};
	// The status of one entry in one flip, as the shader wrote it.
	struct Report {
		uint32_t                entry_id = 0;
		uint64_t                flip     = 0; // the flip the draws belonged to
		PvIndirect::Status      status;
		PvIndirect::Args        last_in;
		std::array<uint32_t, 5> last_out {};
	};
	using Reporter = std::function<void(const Report&)>;

	// Moves the status rows to `flip`: the entries used in flip-2 are reported (no wait), the row of `flip` is cleared.
	void AdvanceFlip(uint64_t flip, const Reporter& report);
	// Takes a ring slot for the command buffer being recorded. ok = false: the slot is still in use by the GPU (ring too small).
	[[nodiscard]] Prepared Reserve(const Job& job);
	// Commands (the recorder of PvIndirect::RecordPrepare calls these in order).
	void Bind(vk::CommandBuffer command);
	void Push(vk::CommandBuffer command, const Prepared& prepared);
	void Dispatch(vk::CommandBuffer command);
	void BarrierToIndirect(vk::CommandBuffer command);

private:
	GraphicContext&                          m_graphics;
	CommandScheduler&                        m_scheduler;
	std::unique_ptr<Buffer>                  m_ring;
	std::unique_ptr<Buffer>                  m_status;
	vk::DescriptorSetLayout                  m_set_layout      = nullptr;
	vk::PipelineLayout                       m_pipeline_layout = nullptr;
	vk::Pipeline                             m_pipeline        = nullptr;
	std::array<uint64_t, kRingSlots>         m_slot_tick {};
	uint64_t                                 m_next_slot = 0;
	uint64_t                                 m_flip      = 0;
	std::array<uint32_t, kStatusRows>        m_row_used {}; // bit per entry id
	bool                                     m_started = false;
};

} // namespace Libs::Graphics
