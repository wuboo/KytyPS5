#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_NARROWBARRIERVK_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_NARROWBARRIERVK_H_

// S556 (KYTY_NARROW_BARRIER, see narrowBarrier.h): synchronization2 masks of a narrowed barrier.

#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/host_gpu/renderer/narrowBarrier.h"

namespace Libs::Graphics::NarrowBarrier {

// Every stage at which a buffer the renderer binds to guest work can be read or written. Cache
// buffers have transfer src/dst, uniform, index, vertex, indirect, storage and device-address usage
// (streamBuffer.h AllFlags); the LDS/GDS utility buffers are storage buffers. That allows:
// index/vertex fetch (eVertexInput), indirect arguments (eDrawIndirect, also used by dispatch and
// mesh-task indirect), every shader stage (ePreRasterizationShaders = vertex/tess/geometry/task/mesh
// as supported, fragment, compute) and copy/fill/update (eCopy, eClear: vkCmdFillBuffer and
// vkCmdUpdateBuffer are clear commands). Not reachable for a buffer: colour/depth attachment
// stages, conditional rendering (only the tiler's own predicate buffer has that usage, and guest
// work never binds it), blit/resolve (image only), host (no device-local cache buffer is written or
// read through a mapping; readbacks copy into the download buffer, which keeps its own host barrier).
inline constexpr vk::PipelineStageFlags2 kBufferStages =
    vk::PipelineStageFlagBits2::eVertexInput | vk::PipelineStageFlagBits2::eDrawIndirect |
    vk::PipelineStageFlagBits2::ePreRasterizationShaders | vk::PipelineStageFlagBits2::eFragmentShader |
    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eCopy |
    vk::PipelineStageFlagBits2::eClear;

// The stages among those that can write a buffer: shader stores/atomics and copy/fill/update.
inline constexpr vk::PipelineStageFlags2 kBufferWriteStages =
    vk::PipelineStageFlagBits2::ePreRasterizationShaders | vk::PipelineStageFlagBits2::eFragmentShader |
    vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eCopy |
    vk::PipelineStageFlagBits2::eClear;

// The writes a buffer can receive at kBufferWriteStages. Read bits are left out on purpose: a read in
// a first access scope has no effect (only writes are made available); a read-after-write hazard is
// carried by the execution dependency on kBufferStages alone.
inline constexpr vk::AccessFlags2 kBufferWrites =
    vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eShaderStorageWrite |
    vk::AccessFlagBits2::eTransferWrite;

// Every access a buffer can see at kBufferStages (the second access scope of an "after" barrier).
inline constexpr vk::AccessFlags2 kBufferAccess =
    vk::AccessFlagBits2::eIndirectCommandRead | vk::AccessFlagBits2::eIndexRead |
    vk::AccessFlagBits2::eVertexAttributeRead | vk::AccessFlagBits2::eUniformRead |
    vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderStorageRead |
    vk::AccessFlagBits2::eShaderSampledRead | vk::AccessFlagBits2::eShaderWrite |
    vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferRead |
    vk::AccessFlagBits2::eTransferWrite;

static_assert((static_cast<VkPipelineStageFlags2>(kBufferStages) & static_cast<VkPipelineStageFlags2>(kBufferWriteStages)) ==
                  static_cast<VkPipelineStageFlags2>(kBufferWriteStages),
              "every buffer write stage is a buffer stage");
static_assert((static_cast<VkAccessFlags2>(kBufferAccess) & static_cast<VkAccessFlags2>(kBufferWrites)) ==
                  static_cast<VkAccessFlags2>(kBufferWrites),
              "every buffer write is a buffer access");

// The copy stage of vkCmdCopyBuffer.
inline constexpr vk::PipelineStageFlags2 kCopyStage = vk::PipelineStageFlagBits2::eCopy;

inline vk::PipelineStageFlags2 ToStages2(vk::PipelineStageFlags stages) {
	// The synchronization2 bits below bit 32 are the synchronization1 bits.
	return vk::PipelineStageFlags2(static_cast<VkPipelineStageFlags2>(static_cast<VkPipelineStageFlags>(stages)));
}

inline vk::BufferMemoryBarrier2 BufferBarrier(vk::Buffer buffer, Span span, vk::PipelineStageFlags2 src_stages,
                                              vk::AccessFlags2 src_access, vk::PipelineStageFlags2 dst_stages,
                                              vk::AccessFlags2 dst_access) {
	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask        = src_stages;
	barrier.srcAccessMask       = src_access;
	barrier.dstStageMask        = dst_stages;
	barrier.dstAccessMask       = dst_access;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer;
	barrier.offset              = span.offset;
	barrier.size                = span.size;
	return barrier;
}

inline void Record(vk::CommandBuffer command, const vk::BufferMemoryBarrier2& barrier,
                   vk::DependencyFlags flags = vk::DependencyFlagBits::eByRegion) {
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = flags;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

inline void Record(vk::CommandBuffer command, const vk::MemoryBarrier2& barrier) {
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

// Narrowed write-hazard barrier before guest work that writes memory (dispatch, or a draw with the
// immediate barrier): every earlier access of a buffer (kBufferStages, execution) and every earlier
// buffer write (kBufferWrites, memory) happen before `dst_stages`' shader reads and writes. Global
// (VkMemoryBarrier2), because address writes have no range.
inline void WriteHazard(vk::CommandBuffer command, vk::PipelineStageFlags2 dst_stages) {
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = kBufferStages;
	barrier.srcAccessMask = kBufferWrites;
	barrier.dstStageMask  = dst_stages;
	barrier.dstAccessMask = vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite;
	Record(command, barrier);
}

} // namespace Libs::Graphics::NarrowBarrier

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_NARROWBARRIERVK_H_ */
