#include "graphics/host_gpu/renderer/pvIndirectRuntime.h"

#include "common/assert.h"
#include "gpu_tiler_shaders/pv_indirect_prepare_spv.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameTiming.h"

#include <cstring>

namespace Libs::Graphics {

PvIndirectRuntime::PvIndirectRuntime(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler) {
	m_ring = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
	                                  uint64_t {kRingSlots} * kSlotBytes);
	// The status is read on the CPU (Invalidate): host memory, like the VS-store guard.
	m_status = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Download, 0, AllFlags,
	                                    uint64_t {kStatusRows} * PvIndirect::kMaxEntries * kStatusDwords * sizeof(uint32_t));
	EXIT_IF(m_status->Mapped().empty());
	std::memset(m_status->Mapped().data(), 0, m_status->Mapped().size());
	m_status->Flush(0, m_status->Size());
	SetVulkanObjectNameF(m_graphics.device, m_ring->Handle(), "Pv Indirect Args Ring");
	SetVulkanObjectNameF(m_graphics.device, m_status->Handle(), "Pv Indirect Status");

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr},
	};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	layout_info.bindingCount = std::size(bindings);
	layout_info.pBindings    = bindings;
	RequireVulkanSuccess(m_graphics.device.createDescriptorSetLayout(&layout_info, nullptr, &m_set_layout),
	                     "create pv-indirect descriptor layout");

	const auto module = CompileSPV(PV_INDIRECT_PREPARE_SPV, m_graphics.device);
	const vk::PushConstantRange push_range {vk::ShaderStageFlagBits::eCompute, 0, kPushDwords * sizeof(uint32_t)};
	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = 1;
	pipeline_layout_info.pSetLayouts            = &m_set_layout;
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_range;
	RequireVulkanSuccess(m_graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr, &m_pipeline_layout),
	                     "create pv-indirect pipeline layout");

	vk::PipelineShaderStageCreateInfo stage {};
	stage.stage  = vk::ShaderStageFlagBits::eCompute;
	stage.module = module;
	stage.pName  = "main";
	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage  = stage;
	pipeline_info.layout = m_pipeline_layout;
	const auto result = m_graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr, &m_pipeline);
	m_graphics.device.destroyShaderModule(module, nullptr);
	RequireVulkanSuccess(result, "create pv-indirect pipeline");
	SetVulkanObjectNameF(m_graphics.device, m_pipeline, "Pv Indirect Prepare");
}

PvIndirectRuntime::~PvIndirectRuntime() {
	m_graphics.device.destroyPipeline(m_pipeline, nullptr);
	m_graphics.device.destroyPipelineLayout(m_pipeline_layout, nullptr);
	m_graphics.device.destroyDescriptorSetLayout(m_set_layout, nullptr);
}

void PvIndirectRuntime::AdvanceFlip(uint64_t flip, const Reporter& report) {
	auto* mapped = m_status->Mapped().data();
	const auto row_bytes = uint64_t {PvIndirect::kMaxEntries} * kStatusDwords * sizeof(uint32_t);
	const auto zero_row  = [&](uint64_t row) {
		std::memset(mapped + row * row_bytes, 0, row_bytes);
		m_status->Flush(row * row_bytes, row_bytes);
		m_row_used[row] = 0;
	};
	if (!m_started) {
		m_started = true;
		m_flip    = flip;
		return;
	}
	if (flip > m_flip + kStatusRows) {
		// A long gap: the draws of the missed flips are long finished and not reported.
		for (uint32_t row = 0; row < kStatusRows; ++row) zero_row(row);
		m_flip = flip;
		return;
	}
	// Flip g starts: report what the draws of flip g-2 wrote (the GPU has had two flips; nothing is waited for), then clear the row g uses.
	while (m_flip < flip) {
		++m_flip;
		if (m_flip >= 2) {
			const uint64_t reported_flip = m_flip - 2;
			const uint64_t row           = reported_flip % kStatusRows;
			if (m_row_used[row] != 0) {
				m_status->Invalidate(row * row_bytes, row_bytes);
				for (uint32_t entry = 0; entry < PvIndirect::kMaxEntries; ++entry) {
					if ((m_row_used[row] & (1u << entry)) == 0) continue;
					uint32_t words[kStatusDwords];
					std::memcpy(words, mapped + row * row_bytes + uint64_t {entry} * kStatusDwords * sizeof(uint32_t), sizeof(words));
					Report r;
					r.entry_id = entry;
					r.flip     = reported_flip;
					r.status   = {words[0], words[1], words[2], words[3]};
					r.last_in  = {words[4], words[5], words[6], words[7], words[8]};
					for (uint32_t k = 0; k < 5; ++k) r.last_out[k] = words[9 + k];
					if (report) report(r);
				}
			}
		}
		zero_row(m_flip % kStatusRows);
	}
}

PvIndirectRuntime::Prepared PvIndirectRuntime::Reserve(const Job& job) {
	Prepared prepared;
	EXIT_IF(job.entry_id >= PvIndirect::kMaxEntries || job.args_buffer == nullptr);
	const uint64_t slot = m_next_slot % kRingSlots;
	if (m_slot_tick[slot] != 0 && !m_scheduler.IsFree(m_slot_tick[slot])) return prepared; // the ring is full of unfinished draws
	m_slot_tick[slot] = m_scheduler.CurrentTick();
	m_next_slot++;

	const uint64_t alignment = m_graphics.StorageMinAlignment();
	const uint64_t aligned   = job.args_offset / alignment * alignment;
	const uint64_t in_bytes  = job.args_offset - aligned;
	EXIT_IF((in_bytes & 3u) != 0);
	const uint64_t row = job.flip % kStatusRows;
	m_row_used[row] |= 1u << job.entry_id;

	prepared.ok         = true;
	prepared.out_buffer = m_ring->Handle();
	prepared.out_offset = slot * kSlotBytes;
	prepared.push       = {job.expected.index_count, job.expected.first_index, job.expected.vertex_offset, job.expected.first_instance,
	                       job.cap, static_cast<uint32_t>(in_bytes / 4), static_cast<uint32_t>(slot * kSlotBytes / 4),
	                       static_cast<uint32_t>((row * PvIndirect::kMaxEntries + job.entry_id) * kStatusDwords)};
	prepared.infos[0]   = {job.args_buffer, aligned, in_bytes + sizeof(vk::DrawIndexedIndirectCommand)};
	prepared.infos[1]   = {m_ring->Handle(), 0, m_ring->Size()};
	prepared.infos[2]   = {m_status->Handle(), 0, m_status->Size()};
	return prepared;
}

void PvIndirectRuntime::Bind(vk::CommandBuffer command) {
	command.bindPipeline(vk::PipelineBindPoint::eCompute, m_pipeline);
}

void PvIndirectRuntime::Push(vk::CommandBuffer command, const Prepared& prepared) {
	std::array<vk::WriteDescriptorSet, 3> writes {};
	for (uint32_t index = 0; index < writes.size(); ++index) {
		writes[index].dstBinding      = index;
		writes[index].descriptorCount = 1;
		writes[index].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[index].pBufferInfo     = &prepared.infos[index];
	}
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, m_pipeline_layout, 0, writes);
	command.pushConstants(m_pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0, kPushDwords * sizeof(uint32_t), prepared.push.data());
}

void PvIndirectRuntime::Dispatch(vk::CommandBuffer command) {
	command.dispatch(1, 1, 1);
	FrameTiming::NoteDispatch(FrameTiming::DispatchClass::Other);
}

void PvIndirectRuntime::BarrierToIndirect(vk::CommandBuffer command) {
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
	command.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eDrawIndirect, {}, 1, &barrier, 0, nullptr,
	                        0, nullptr);
}

} // namespace Libs::Graphics
