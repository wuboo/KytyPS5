#include "graphics/host_gpu/renderer/meshIndirectRuntime.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/gpuLabelsVk.h"
#include "gpu_tiler_shaders/mesh_indirect_prepare_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameTiming.h"

#include <atomic>
#include <cinttypes>
#include <cstring>

namespace Libs::Graphics {

MeshIndirectRuntime::MeshIndirectRuntime(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {
	m_ring = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, uint64_t {kSlots} * kSlotBytes);
	m_shadow_ring = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, AllFlags, uint64_t {kSlots} * kSlotBytes);
	EXIT_IF(m_shadow_ring->Mapped().empty());
	std::memset(m_shadow_ring->Mapped().data(), 0, m_shadow_ring->Mapped().size());
	m_shadow_ring->Flush(0, m_shadow_ring->Size());
	SetVulkanObjectNameF(m_graphics.device, m_ring->Handle(), "Mesh Indirect Ring");
	SetVulkanObjectNameF(m_graphics.device, m_shadow_ring->Handle(), "Mesh Indirect Shadow Ring");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_set_layout),
	                     "create mesh-indirect descriptor layout");

	const auto module = CompileSPV(MESH_INDIRECT_PREPARE_SPV, m_graphics.device);
	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0, kPushDwords * sizeof(uint32_t)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	                     "create mesh-indirect pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create mesh-indirect pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Mesh Indirect Prepare");
}

MeshIndirectRuntime::~MeshIndirectRuntime() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

MeshIndirectRuntime::Prepared MeshIndirectRuntime::Reserve(const Job& job) {
	Prepared prepared;
	EXIT_IF(job.args_buffer == nullptr);
	uint64_t&      next = job.shadow ? m_next_shadow_slot : m_next_slot;
	const uint64_t slot = next % kSlots;
	if (job.shadow) {
		Reap(); // a finished shadow slot is compared before it is reused
		if (m_shadow_tick[slot] != 0) return prepared; // the previous use of the slot has not finished yet
		m_shadow_tick[slot] = m_scheduler.CurrentTick();
	} else {
		if (m_slot_tick[slot] != 0 && !m_scheduler.IsFree(m_slot_tick[slot])) return prepared; // the ring is full of unfinished draws
		m_slot_tick[slot] = m_scheduler.CurrentTick();
	}
	next++;

	const uint64_t alignment = m_graphics.StorageMinAlignment();
	const uint64_t aligned   = job.args_offset / alignment * alignment;
	const uint64_t in_bytes  = job.args_offset - aligned;
	EXIT_IF((in_bytes & 3u) != 0);
	prepared.ok         = true;
	prepared.slot       = static_cast<uint32_t>(slot);
	prepared.out_buffer = job.shadow ? m_shadow_ring->Handle() : m_ring->Handle();
	prepared.out_offset = slot * kSlotBytes;
	prepared.push       = {static_cast<uint32_t>(in_bytes / 4), static_cast<uint32_t>(slot * kSlotBytes / 4), job.index_buffer_size,
	                       job.mesh.primitives_per_group, job.mesh.primitive_size, job.mesh.primitive_step, job.native_fetch ? 1u : 0u,
	                       job.limits.max_x, job.limits.max_y, job.limits.max_total};
	prepared.infos[0]   = {job.args_buffer, aligned, in_bytes + sizeof(vk::DrawIndexedIndirectCommand)};
	prepared.infos[1]   = {prepared.out_buffer, 0, uint64_t {kSlots} * kSlotBytes};
	return prepared;
}

void MeshIndirectRuntime::ExpectShadow(const Prepared& prepared, const MeshIndirect::Plan& cpu_plan) {
	EXIT_IF(!prepared.ok);
	m_shadow_expect[prepared.slot] = cpu_plan;
}

void MeshIndirectRuntime::Reap() {
	using C = FrameTiming::Counter;
	static std::atomic<uint32_t> logged {0};
	for (uint32_t slot = 0; slot < kSlots; ++slot) {
		if (m_shadow_tick[slot] == 0 || !m_scheduler.IsFree(m_shadow_tick[slot])) continue;
		m_shadow_ring->Invalidate(uint64_t {slot} * kSlotBytes, kSlotBytes);
		MeshIndirect::Plan gpu {};
		std::memcpy(&gpu, m_shadow_ring->Mapped().data() + uint64_t {slot} * kSlotBytes, sizeof(gpu));
		const bool equal = gpu == m_shadow_expect[slot];
		FrameTiming::Add(C::MeshIndShadowN);
		FrameTiming::Add(equal ? C::MeshIndShadowEqN : C::MeshIndShadowNeN);
		if (gpu.violation != 0) FrameTiming::Add(C::MeshIndViolationN);
		if (!equal && logged.fetch_add(1, std::memory_order_relaxed) < 20) {
			const auto& e = m_shadow_expect[slot];
			LOGF("[mesh-ind-ne] cpu groups=%u,%u,%u idx=%u vo=%u fi=%u start=%u viol=%u gpu groups=%u,%u,%u idx=%u vo=%u fi=%u start=%u viol=%u\n",
			     e.groups_x, e.groups_y, e.groups_z, e.index_count, e.vertex_offset, e.first_instance, e.start_index, e.violation, gpu.groups_x,
			     gpu.groups_y, gpu.groups_z, gpu.index_count, gpu.vertex_offset, gpu.first_instance, gpu.start_index, gpu.violation);
		}
		m_shadow_tick[slot] = 0;
	}
}

void MeshIndirectRuntime::Bind(vk::CommandBuffer command) {
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
}

void MeshIndirectRuntime::Push(vk::CommandBuffer command, const Prepared& prepared) {
	std::array<vk::WriteDescriptorSet, 2> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &prepared.infos[index];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, kPushDwords * sizeof(uint32_t), prepared.push.data());
}

void MeshIndirectRuntime::Dispatch(vk::CommandBuffer command) {
	GpuLabel::With(command, GpuLabel::Class::PvIndirect, [&] { command.dispatch(1, 1, 1); }, "I.meshind");
	FrameTiming::NoteDispatch(FrameTiming::DispatchClass::Other);
}

void MeshIndirectRuntime::BarrierToIndirect(vk::CommandBuffer command) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr,
	                        0, nullptr);
}

} // namespace Libs::Graphics
