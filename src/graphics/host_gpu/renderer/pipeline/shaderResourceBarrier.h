#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cpEmit.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

namespace Libs::Graphics {

struct ShaderStageRuntime;

vk::ShaderStageFlagBits NativeShaderStage(ShaderType stage);
vk::PipelineStageFlags  ShaderPipelineStages(vk::ShaderStageFlags stages);
vk::MemoryBarrier       MakeShaderWriteDependency();
vk::BufferMemoryBarrier MakeGdsDependency(vk::Buffer buffer);
bool HasShaderBufferWrites(const ShaderStageRuntime& runtime);
void ShaderAccessBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);
void ShaderWriteHazardBarrier(vk::CommandBuffer      vk_buffer,
                              vk::PipelineStageFlags destination_stages);
void ShaderWriteBarrier(vk::CommandBuffer vk_buffer, vk::PipelineStageFlags source_stages);

// S565: the same barriers on any command handle (vk::CommandBuffer or the KYTY_CP_EMIT IR handle CpEmit::Cmd, which records them in the IR
// instead of converting to the raw handle, which drains).
template <class Command>
void ShaderAccessBarrierTo(const Command& command, vk::PipelineStageFlags source_stages) {
	const vk::MemoryBarrier barrier {.srcAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
	                                 .dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite};
	command.pipelineBarrier(source_stages, vk::PipelineStageFlagBits::eAllCommands, vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
}
template <class Command>
void ShaderWriteHazardBarrierTo(const Command& command, vk::PipelineStageFlags destination_stages) {
	const vk::MemoryBarrier barrier {.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite,
	                                 .dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite};
	command.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, destination_stages, vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
}
template <class Command>
void ShaderWriteBarrierTo(const Command& command, vk::PipelineStageFlags source_stages) {
	const auto barrier = MakeShaderWriteDependency();
	command.pipelineBarrier(source_stages,
	                        vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eAllGraphics | vk::PipelineStageFlagBits::eTransfer,
	                        vk::DependencyFlags {}, 1, &barrier, 0, nullptr, 0, nullptr);
}
inline void ShaderAccessBarrier(const CpEmit::Cmd& command, vk::PipelineStageFlags source_stages) {
	ShaderAccessBarrierTo(command, source_stages);
}
inline void ShaderWriteHazardBarrier(const CpEmit::Cmd& command, vk::PipelineStageFlags destination_stages) {
	ShaderWriteHazardBarrierTo(command, destination_stages);
}
inline void ShaderWriteBarrier(const CpEmit::Cmd& command, vk::PipelineStageFlags source_stages) {
	ShaderWriteBarrierTo(command, source_stages);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_SHADERRESOURCEBARRIER_H_
