#ifndef KYTY_GRAPHICS_HOST_GPU_GPULABELSVK_H_
#define KYTY_GRAPHICS_HOST_GPU_GPULABELSVK_H_

#include "graphics/host_gpu/gpuLabels.h"
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

} // namespace Libs::Graphics::GpuLabel

#endif
