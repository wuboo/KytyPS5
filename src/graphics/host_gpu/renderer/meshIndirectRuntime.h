#pragma once

// S472 (KYTY_INDIRECT_GPU_MESH): GPU side of the mesh indirect draw (meshIndirect.h has the arithmetic). Owns the compute pipeline of
// mesh_indirect_prepare.comp and the rings its output goes to: a device-local one (mode On: the draw reads it as indirect args) and a
// host-visible one (mode Shadow: what the compute wrote is compared with what the CPU computed, once the GPU has finished with the slot).

#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/meshIndirect.h"

#include <array>
#include <cstdint>
#include <memory>

namespace Libs::Graphics {
class CommandScheduler;
struct GraphicContext;

class MeshIndirectRuntime {
public:
	static constexpr uint32_t kSlots     = 1024;
	static constexpr uint32_t kSlotBytes = 32; // 8 dwords
	static constexpr uint32_t kPushDwords = 10;

	MeshIndirectRuntime(GraphicContext& graphics, CommandScheduler& scheduler);
	~MeshIndirectRuntime();
	MeshIndirectRuntime(const MeshIndirectRuntime&)            = delete;
	MeshIndirectRuntime& operator=(const MeshIndirectRuntime&) = delete;

	struct Job {
		vk::Buffer                args_buffer = nullptr;
		uint64_t                  args_offset = 0;
		MeshIndirect::MeshInfo    mesh;
		MeshIndirect::Limits      limits;
		uint32_t                  index_buffer_size = 0;
		bool                      native_fetch      = false;
		bool                      shadow            = false;
	};
	struct Prepared {
		bool                                    ok          = false;
		uint32_t                                slot        = 0;
		vk::Buffer                              out_buffer  = nullptr;
		uint64_t                                out_offset  = 0;
		std::array<uint32_t, kPushDwords>       push {};
		std::array<vk::DescriptorBufferInfo, 2> infos {};
	};

	// Takes a ring slot for the command buffer being recorded. ok = false: the slot is still in use by the GPU.
	[[nodiscard]] Prepared Reserve(const Job& job);
	// Shadow: the plan the CPU computed for `slot` (compared in Reap once the slot is finished).
	void ExpectShadow(const Prepared& prepared, const MeshIndirect::Plan& cpu_plan);
	// Compares every finished shadow slot (no wait).
	void Reap();
	void Bind(vk::CommandBuffer command);
	void Push(vk::CommandBuffer command, const Prepared& prepared);
	void Dispatch(vk::CommandBuffer command);
	void BarrierToIndirect(vk::CommandBuffer command);

private:
	GraphicContext&                              m_graphics;
	CommandScheduler&                            m_scheduler;
	std::unique_ptr<Buffer>                      m_ring;        // device local (On)
	std::unique_ptr<Buffer>                      m_shadow_ring; // host visible (Shadow)
	vk::DescriptorSetLayout                      m_set_layout      = nullptr;
	vk::PipelineLayout                           m_pipeline_layout = nullptr;
	vk::Pipeline                                 m_pipeline        = nullptr;
	std::array<uint64_t, kSlots>                 m_slot_tick {};
	std::array<uint64_t, kSlots>                 m_shadow_tick {};  // non-zero: a shadow compare is pending
	std::array<MeshIndirect::Plan, kSlots>       m_shadow_expect {};
	uint64_t                                     m_next_slot        = 0;
	uint64_t                                     m_next_shadow_slot = 0;
};

} // namespace Libs::Graphics
