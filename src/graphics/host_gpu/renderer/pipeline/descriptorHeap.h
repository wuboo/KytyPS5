#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <deque>
#include <unordered_map>

namespace Libs::Graphics {

// Bumped whenever a VkBuffer or VkImageView the renderer bound may be destroyed, so a descriptor
// set cached by content is not reused after a handle it references could have been recycled.
inline std::atomic<uint64_t> g_descriptor_resource_generation {0};

struct GraphicContext;
class MasterSemaphore;

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout layout);
	// Changes when the current pool is replaced; sets from older pools must not be reused.
	[[nodiscard]] uint64_t PoolGeneration() const noexcept { return m_pool_generation; }

private:
	static constexpr uint32_t DescriptorSetBatch = 32;

	struct Batch {
		std::array<vk::DescriptorSet, DescriptorSetBatch> sets {};
		uint32_t                                          size       = 0;
		uint32_t                                          allocation = DescriptorSetBatch;
	};

	[[nodiscard]] bool Allocate(vk::DescriptorSetLayout layout, Batch& batch);
	void               CreateDescriptorPool();

	GraphicContext&                                     m_graphics;
	MasterSemaphore&                                    m_master_semaphore;
	vk::DescriptorPool                                  m_current_pool = nullptr;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>> m_pending_pools;
	std::unordered_map<vk::DescriptorSetLayout, Batch>  m_sets;
	uint64_t                                            m_pool_generation = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
