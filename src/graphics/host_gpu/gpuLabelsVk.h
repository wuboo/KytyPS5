#ifndef KYTY_GRAPHICS_HOST_GPU_GPULABELSVK_H_
#define KYTY_GRAPHICS_HOST_GPU_GPULABELSVK_H_

#include "graphics/host_gpu/gpuLabels.h"
#include "graphics/host_gpu/renderer/cpEmit.h"
#include "graphics/host_gpu/vulkanCommon.h"

namespace Libs::Graphics::GpuLabel {

namespace detail {
inline vk::CommandBuffer Cb(void* cb) {
	return vk::CommandBuffer(static_cast<VkCommandBuffer>(cb));
}
inline void VkBegin(void* cb, const char* label) {
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdBeginDebugUtilsLabelEXT == nullptr) return;
	vk::DebugUtilsLabelEXT info {};
	info.pLabelName = label;
	Cb(cb).beginDebugUtilsLabelEXT(info);
}
inline void VkEnd(void* cb) {
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdEndDebugUtilsLabelEXT == nullptr) return;
	Cb(cb).endDebugUtilsLabelEXT();
}
inline void VkInsert(void* cb, const char* label) {
	if (VULKAN_HPP_DEFAULT_DISPATCHER.vkCmdInsertDebugUtilsLabelEXT == nullptr) return;
	vk::DebugUtilsLabelEXT info {};
	info.pLabelName = label;
	Cb(cb).insertDebugUtilsLabelEXT(info);
}
inline const Backend kVulkan {VkBegin, VkEnd, VkInsert};
} // namespace detail

// `record` is the command that opens the encoder (dispatch, beginRendering, copy ...).
template <class F, class... Args>
void With(vk::CommandBuffer cb, Class c, F&& record, const char* format, Args... args) {
	Run(detail::kVulkan, static_cast<VkCommandBuffer>(cb), c, static_cast<F&&>(record), format, args...);
}

// S565: `record` records through the KYTY_CP_EMIT IR handle. Without the label it goes to the IR (or straight on, as the handle says). With
// it the stream is drained before the label begins and again before it ends, so the label encloses the command and nothing on the raw
// handle overlaps the emitter.
template <class F, class... Args>
void With(const CpEmit::Cmd& cb, Class c, F&& record, const char* format, Args... args) {
	if (!Active(c)) {
		record();
		return;
	}
	const vk::CommandBuffer raw = cb; // drains
	Run(
	    detail::kVulkan, static_cast<VkCommandBuffer>(raw), c,
	    [&] {
		    record();
		    (void)static_cast<vk::CommandBuffer>(cb); // drains what record() wrote
	    },
	    format, args...);
}

} // namespace Libs::Graphics::GpuLabel

#endif
