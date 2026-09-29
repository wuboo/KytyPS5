#include "common/assert.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/presentation/presenter.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/presentation/videoOut.h"
#include "graphics/presentation/window/windowInternal.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <deque>
#include <limits>
#include <memory>
#include <vector>
#include <vulkan/vk_platform.h>

// IWYU pragma: no_include <intrin.h>

namespace Libs::Graphics {

struct Presenter::Frame {
	VulkanImage image;
	uint64_t    present_tick = 0;
	bool        busy         = false;
	bool        reusing_last = false;

	void Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format);
	void Transit(vk::CommandBuffer command, vk::ImageLayout layout, vk::AccessFlags2 access);
	void CopyFrom(CommandBuffer& command, Image& source);
	void Clear(CommandBuffer& command, const vk::ClearColorValue& color);
};

class FramePool final {
public:
	FramePool(WindowContext& window, CommandScheduler& scheduler)
	    : m_window(window), m_scheduler(scheduler) {}
	~FramePool() {
		m_scheduler.Wait(m_scheduler.CurrentTick() - 1);
		for (auto& frame: m_frames) {
			if (frame->image.image != nullptr) {
				m_window.graphic_ctx.DeleteImage(frame->image);
			}
		}
	}
	KYTY_CLASS_NO_COPY(FramePool);

	void Initialize(uint32_t count, vk::Format format) {
		if (count == 0 || format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires at least one frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!m_frames.empty()) {
			EXIT("prepared-frame pool was initialized twice\n");
		}
		m_format = format;
		m_frames.reserve(count);
		for (uint32_t i = 0; i < count; i++) {
			auto frame = std::make_unique<Presenter::Frame>();
			m_free.push_back(frame.get());
			m_frames.push_back(std::move(frame));
		}
	}

	void SetFormat(vk::Format format) {
		if (format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool requires a presentation format\n");
		}
		Common::LockGuard lock(m_mutex);
		m_format = format;
	}

	vk::Format GetFormat() {
		Common::LockGuard lock(m_mutex);
		if (m_format == vk::Format::eUndefined) {
			EXIT("prepared-frame pool has no presentation format\n");
		}
		return m_format;
	}

	Presenter::Frame* Acquire() {
		m_mutex.Lock();
		if (m_frames.empty()) {
			EXIT("prepared-frame pool was used before swapchain initialization\n");
		}
		while (m_free.empty()) {
			m_available.Wait(&m_mutex);
		}
		auto* frame = m_free.front();
		m_free.pop_front();
		if (frame->busy) {
			EXIT("prepared-frame pool returned an invalid frame\n");
		}
		if (m_last_frame == frame) {
			m_last_frame = nullptr;
		}
		frame->busy         = true;
		frame->reusing_last = false;
		m_mutex.Unlock();

		WaitForFrame(*frame);
		return frame;
	}

	Presenter::Frame* AcquireLast() {
		m_mutex.Lock();
		auto* frame = m_last_frame;
		if (frame == nullptr) {
			m_mutex.Unlock();
			return nullptr;
		}
		auto free = std::find(m_free.begin(), m_free.end(), frame);
		if (free == m_free.end() || frame->busy) {
			m_mutex.Unlock();
			EXIT("last submitted frame is not available for reuse\n");
		}
		m_free.erase(free);
		m_last_frame        = nullptr;
		frame->busy         = true;
		frame->reusing_last = true;
		m_mutex.Unlock();

		WaitForFrame(*frame);
		return frame;
	}

	void ValidateForPresent(Presenter::Frame* frame, bool reuse) {
		Common::LockGuard lock(m_mutex);
		if (frame == nullptr || !frame->busy || frame->reusing_last != reuse) {
			EXIT("prepared frame has invalid presentation ownership\n");
		}
	}

	void Release(Presenter::Frame* frame, bool make_last = false) {
		if (frame == nullptr) {
			EXIT("cannot release a null prepared frame\n");
		}
		Common::LockGuard lock(m_mutex);
		if (!frame->busy) {
			EXIT("prepared frame was released twice\n");
		}
		frame->busy         = false;
		frame->reusing_last = false;
		if (make_last) {
			m_last_frame = frame;
		}
		m_free.push_back(frame);
		m_available.Signal();
	}

private:
	void WaitForFrame(Presenter::Frame& frame) { m_scheduler.Wait(frame.present_tick); }

	WindowContext&                                 m_window;
	CommandScheduler&                              m_scheduler;
	Common::Mutex                                  m_mutex;
	Common::CondVar                                m_available;
	std::vector<std::unique_ptr<Presenter::Frame>> m_frames;
	std::deque<Presenter::Frame*>                  m_free;
	Presenter::Frame*                              m_last_frame = nullptr;
	vk::Format                                     m_format     = vk::Format::eUndefined;
};

void Presenter::Frame::Configure(GraphicContext& graphics, vk::Extent2D extent, vk::Format format) {
	if (extent.width == 0 || extent.height == 0 || format == vk::Format::eUndefined) {
		EXIT("unsupported prepared frame, extent=%ux%u format=%d\n", extent.width, extent.height,
		     static_cast<int>(format));
	}
	auto&      dst        = image;
	const bool compatible = dst.image != nullptr && dst.extent.width == extent.width &&
	                        dst.extent.height == extent.height && dst.format == format;
	if (compatible) {
		return;
	}

	const auto features = graphics.GetFormatProperties(format).optimalTilingFeatures;
	const auto required =
	    vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eSampledImageFilterLinear |
	    vk::FormatFeatureFlagBits::eTransferSrc | vk::FormatFeatureFlagBits::eTransferDst;
	if ((features & required) != required) {
		EXIT("prepared presentation format lacks optimal blit support: format=%d features=0x%x\n",
		     static_cast<int>(format), static_cast<vk::FormatFeatureFlags::MaskType>(features));
	}

	if (dst.image != nullptr) {
		graphics.DeleteImage(dst);
	}

	vk::ImageCreateInfo create {};
	create.sType         = vk::StructureType::eImageCreateInfo;
	create.imageType     = vk::ImageType::e2D;
	create.extent        = {extent.width, extent.height, 1};
	create.mipLevels     = 1;
	create.arrayLayers   = 1;
	create.format        = format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	create.sharingMode = vk::SharingMode::eExclusive;
	create.samples     = vk::SampleCountFlagBits::e1;
	if (!graphics.CreateImage(create, dst)) {
		EXIT("failed to allocate prepared presentation image, extent=%ux%u format=%d\n",
		     extent.width, extent.height, static_cast<int>(format));
	}
}

void Presenter::Frame::Transit(vk::CommandBuffer command, vk::ImageLayout layout,
                               vk::AccessFlags2 access) {
	const auto     stage  = access == vk::AccessFlagBits2::eTransferRead ||
	                                access == vk::AccessFlagBits2::eTransferWrite
	                            ? vk::PipelineStageFlagBits2::eTransfer
	                            : vk::PipelineStageFlagBits2::eAllCommands;
	constexpr auto writes = vk::AccessFlagBits2::eTransferWrite |
	                        vk::AccessFlagBits2::eShaderWrite | vk::AccessFlagBits2::eMemoryWrite;
	if (image.state.layout == layout && image.state.access_mask == access &&
	    !static_cast<bool>(image.state.access_mask & writes)) {
		return;
	}
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask                    = image.state.pl_stage;
	barrier.srcAccessMask                   = image.state.access_mask;
	barrier.dstStageMask                    = stage;
	barrier.dstAccessMask                   = access;
	barrier.oldLayout                       = image.state.layout;
	barrier.newLayout                       = layout;
	barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	barrier.image                           = image.image;
	barrier.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	barrier.subresourceRange.baseMipLevel   = 0;
	barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
	barrier.subresourceRange.baseArrayLayer = 0;
	barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
	image.state = {stage, access, layout};
	image.subresource_states.clear();
}

void Presenter::Frame::CopyFrom(CommandBuffer& command_buffer, Image& source) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	vk::ImageCopy copy {};
	copy.srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, source.backing.layers};
	copy.dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, image.layers};
	copy.extent         = {std::min(source.backing.extent.width, image.extent.width),
	                       std::min(source.backing.extent.height, image.extent.height), 1};
	EXIT_IF(copy.srcSubresource.layerCount != copy.dstSubresource.layerCount);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, image.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy);
	Transit(command, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
}

void Presenter::Frame::Clear(CommandBuffer& command_buffer, const vk::ClearColorValue& color) {
	command_buffer.EndRendering();
	auto command = command_buffer.Handle();
	Transit(command, vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite);
	const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	command.clearColorImage(image.image, vk::ImageLayout::eTransferDstOptimal, &color, 1, &range);
	Transit(command, vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead);
}

class Swapchain final {
public:
	enum class Status : uint8_t { Success, Recreate, SurfaceLost };

	explicit Swapchain(WindowContext& window): m_window(window) {}
	~Swapchain();
	KYTY_CLASS_NO_COPY(Swapchain);

	void                 Create();
	void                 Recreate(bool surface_lost = false);
	[[nodiscard]] bool   NeedsResize() const;
	[[nodiscard]] Status AcquireNextImage();
	[[nodiscard]] bool   PrepareSystemOverlay();
	void                 RecordPresentCommands(CommandBuffer& command, VulkanImage& source,
	                                           bool draw_system_overlay);
	uint64_t             Submit(CommandScheduler& scheduler);
	[[nodiscard]] Status Present();

	[[nodiscard]] uint32_t ImageCount() const noexcept {
		return static_cast<uint32_t>(m_images.size());
	}
	[[nodiscard]] vk::Format Format() const noexcept { return m_format; }

private:
	void Destroy();

	WindowContext&   m_window;
	vk::SwapchainKHR m_handle = nullptr;
	vk::Format       m_format = vk::Format::eUndefined;
	vk::Extent2D     m_extent {};
	// Drawable pixel size observed when this swapchain was created.
	vk::Extent2D                   m_window_extent {};
	std::vector<vk::Image>         m_images;
	std::vector<vk::ImageView>     m_image_views;
	std::vector<vk::Semaphore>     m_image_acquired;
	std::vector<vk::Semaphore>     m_render_complete;
	std::unique_ptr<SystemOverlay> m_system_overlay;
	uint32_t                       m_image_index = static_cast<uint32_t>(-1);
	uint32_t                       m_frame_index = 0;
};

struct Presenter::Impl {
	explicit Impl(WindowContext& owner)
	    : renderer(*owner.render_context), window(owner), swapchain(owner),
	      present_scheduler(renderer, owner.graphic_ctx), frames(owner, present_scheduler) {
		EXIT_IF(owner.render_context == nullptr);
		swapchain.Create();
		frames.Initialize(swapchain.ImageCount(), swapchain.Format());
	}

	void RecoverSwapchain(Swapchain::Status status) {
		LOGF("Recovering Vulkan swapchain%s\n",
		     status == Swapchain::Status::SurfaceLost ? " and surface" : "");
		swapchain.Recreate(status == Swapchain::Status::SurfaceLost);
		frames.SetFormat(swapchain.Format());
	}

	Image& ResolveSurface(const ImageInfo& info) {
		TextureCache::ImageDesc desc {};
		desc.info                  = info;
		desc.view_info.format      = info.pixel_format;
		desc.view_info.type        = vk::ImageViewType::e2D;
		desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
		desc.view_info.base_level  = 0;
		desc.view_info.level_count = 1;
		desc.view_info.base_layer  = 0;
		desc.view_info.layer_count = 1;
		desc.view_info.usage       = vk::ImageUsageFlagBits::eTransferSrc;
		desc.type                  = TextureCache::BindingType::VideoOut;

		auto&      cache      = renderer.GetTextureCache();
		const auto image_id   = cache.FindImage(desc);
		auto&      image      = cache.GetImage(image_id);
		image.usage.video_out = true;
		cache.UpdateImage(image_id);
		return image;
	}

	RenderContext&        renderer;
	WindowContext&        window;
	Swapchain             swapchain;
	CommandScheduler      present_scheduler;
	FramePool             frames;
	std::atomic<uint64_t> presented_overlay_revision {0};
};

void Swapchain::Create() {
	auto& graphics = m_window.graphic_ctx;
	EXIT_IF(graphics.device == nullptr);
	EXIT_IF(m_window.surface == nullptr);

	// Capture the size before querying surface capabilities. If the window changes
	// during creation, the next presentation will detect the newer size.
	{
		Common::LockGuard lock(m_window.mutex);
		m_window_extent = {graphics.screen_width, graphics.screen_height};
	}
	EXIT_IF(m_window_extent.width == 0);
	EXIT_IF(m_window_extent.height == 0);
	m_window.RefreshSurfaceCapabilities();
	const auto& surface = m_window.surface_capabilities;
	EXIT_NOT_IMPLEMENTED(surface.formats.empty());

	m_extent = surface.capabilities.currentExtent;
	if (m_extent.width == std::numeric_limits<uint32_t>::max()) {
		m_extent.width =
		    std::clamp(m_window_extent.width, surface.capabilities.minImageExtent.width,
		               surface.capabilities.maxImageExtent.width);
		m_extent.height =
		    std::clamp(m_window_extent.height, surface.capabilities.minImageExtent.height,
		               surface.capabilities.maxImageExtent.height);
	}
	uint32_t image_count = surface.capabilities.minImageCount + 1;
	if (surface.capabilities.maxImageCount != 0) {
		image_count = std::min(image_count, surface.capabilities.maxImageCount);
	}
	const auto transform =
	    surface.capabilities.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity
	        ? vk::SurfaceTransformFlagBitsKHR::eIdentity
	        : surface.capabilities.currentTransform;
	const auto composite =
	    surface.capabilities.supportedCompositeAlpha & vk::CompositeAlphaFlagBitsKHR::eOpaque
	        ? vk::CompositeAlphaFlagBitsKHR::eOpaque
	        : vk::CompositeAlphaFlagBitsKHR::eInherit;

	vk::SurfaceFormatKHR format {vk::Format::eR8G8B8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
	if (surface.formats.size() != 1 || surface.formats.front().format != vk::Format::eUndefined) {
		const auto it = std::find_if(surface.formats.begin(), surface.formats.end(),
		                             [](const vk::SurfaceFormatKHR& candidate) {
			                             return candidate.format == vk::Format::eB8G8R8A8Unorm ||
			                                    candidate.format == vk::Format::eR8G8B8A8Unorm;
		                             });
		if (it == surface.formats.end()) {
			EXIT("no supported UNORM swapchain format\n");
		}
		format = *it;
	}
	m_format                      = format.format;
	const auto swapchain_features = graphics.GetFormatProperties(m_format).optimalTilingFeatures;
	if (!static_cast<bool>(swapchain_features & vk::FormatFeatureFlagBits::eBlitDst)) {
		EXIT("swapchain format cannot be a blit destination: format=%d\n",
		     static_cast<int>(m_format));
	}

	vk::SwapchainCreateInfoKHR create_info {};
	create_info.sType            = vk::StructureType::eSwapchainCreateInfoKHR;
	create_info.surface          = m_window.surface;
	create_info.minImageCount    = image_count;
	create_info.imageFormat      = format.format;
	create_info.imageColorSpace  = format.colorSpace;
	create_info.imageExtent      = m_extent;
	create_info.imageArrayLayers = 1;
	create_info.imageUsage =
	    vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferDst;
	create_info.imageSharingMode = vk::SharingMode::eExclusive;
	create_info.preTransform     = transform;
	create_info.compositeAlpha   = composite;
	switch (Config::GetPresentMode()) {
		case Config::PresentMode::Mailbox:
			create_info.presentMode = vk::PresentModeKHR::eMailbox;
			break;
		case Config::PresentMode::Immediate:
			create_info.presentMode = vk::PresentModeKHR::eImmediate;
			break;
		case Config::PresentMode::Fifo:
		default: create_info.presentMode = vk::PresentModeKHR::eFifo; break;
	}
	if (std::find(surface.present_modes.begin(), surface.present_modes.end(),
	              create_info.presentMode) == surface.present_modes.end()) {
		LOGF("warning: requested present mode is unavailable; falling back to Fifo\n");
		create_info.presentMode = vk::PresentModeKHR::eFifo;
	}
	create_info.clipped = VK_TRUE;
	RequireVulkanSuccess(graphics.device.createSwapchainKHR(&create_info, nullptr, &m_handle),
	                     "vkCreateSwapchainKHR");
	EXIT_IF(m_handle == nullptr);

	m_images = EnumerateVulkan<vk::Image>(
	    "vkGetSwapchainImagesKHR", [&](uint32_t* count, vk::Image* images) {
		    return graphics.device.getSwapchainImagesKHR(m_handle, count, images);
	    });
	EXIT_NOT_IMPLEMENTED(m_images.empty());

	m_image_views.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		vk::ImageViewCreateInfo view {};
		view.sType                           = vk::StructureType::eImageViewCreateInfo;
		view.image                           = m_images[i];
		view.viewType                        = vk::ImageViewType::e2D;
		view.format                          = m_format;
		view.components                      = {};
		view.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
		view.subresourceRange.baseArrayLayer = 0;
		view.subresourceRange.baseMipLevel   = 0;
		view.subresourceRange.layerCount     = 1;
		view.subresourceRange.levelCount     = 1;
		RequireVulkanSuccess(graphics.device.createImageView(&view, nullptr, &m_image_views[i]),
		                     "vkCreateImageView");
		EXIT_IF(m_image_views[i] == nullptr);
	}

	vk::SemaphoreCreateInfo semaphore_info {};
	semaphore_info.sType = vk::StructureType::eSemaphoreCreateInfo;
	m_image_acquired.resize(m_images.size());
	m_render_complete.resize(m_images.size());
	for (size_t i = 0; i < m_images.size(); i++) {
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_image_acquired[i]),
		    "create swapchain image-acquired semaphore");
		RequireVulkanSuccess(
		    graphics.device.createSemaphore(&semaphore_info, nullptr, &m_render_complete[i]),
		    "create swapchain render-complete semaphore");
	}
	m_image_index = static_cast<uint32_t>(-1);
	m_frame_index = 0;
}

Swapchain::~Swapchain() {
	Destroy();
}

void Swapchain::Destroy() {
	if (m_handle == nullptr && m_image_acquired.empty() && m_render_complete.empty() &&
	    m_image_views.empty()) {
		return;
	}
	auto& graphics = m_window.graphic_ctx;

	{
		Common::LockGuard queue_lock(graphics.queue_mutex);
		RequireVulkanSuccess(graphics.queue.waitIdle(), "wait for swapchain queue");
	}
	if (m_system_overlay != nullptr) {
		m_system_overlay->ReleaseVulkan();
	}

	for (const auto semaphore: m_image_acquired) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto semaphore: m_render_complete) {
		if (semaphore != nullptr) {
			graphics.device.destroySemaphore(semaphore, nullptr);
		}
	}
	for (const auto view: m_image_views) {
		if (view != nullptr) {
			graphics.device.destroyImageView(view, nullptr);
		}
	}
	if (m_handle != nullptr) {
		graphics.device.destroySwapchainKHR(m_handle, nullptr);
	}

	m_handle        = nullptr;
	m_format        = vk::Format::eUndefined;
	m_extent        = {};
	m_window_extent = {};
	m_image_index   = static_cast<uint32_t>(-1);
	m_frame_index   = 0;
	m_images.clear();
	m_image_views.clear();
	m_image_acquired.clear();
	m_render_complete.clear();
}

void Swapchain::Recreate(bool surface_lost) {
	Destroy();
	if (surface_lost) {
#if defined(__APPLE__)
		// Surface recreation goes through SDL_Vulkan_CreateSurface, which touches the
		// window's view/layer and must run on the main thread on macOS.
		EXIT_IF(!SDL_RunOnMainThread(
		    [](void* window) { static_cast<WindowContext*>(window)->RecreateSurface(); }, &m_window,
		    true));
#else
		m_window.RecreateSurface();
#endif
	}
	Create();
}

bool Swapchain::NeedsResize() const {
	Common::LockGuard lock(m_window.mutex);
	return m_window_extent.width != m_window.graphic_ctx.screen_width ||
	       m_window_extent.height != m_window.graphic_ctx.screen_height;
}

Swapchain::Status Swapchain::AcquireNextImage() {
	EXIT_IF(m_handle == nullptr || m_frame_index >= m_image_acquired.size());
	m_image_index     = static_cast<uint32_t>(-1);
	const auto result = m_window.graphic_ctx.device.acquireNextImageKHR(
	    m_handle, std::numeric_limits<uint64_t>::max(), m_image_acquired[m_frame_index], nullptr,
	    &m_image_index);
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorUnknown:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorUnknown\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkAcquireNextImageKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkAcquireNextImageKHR failed: %s\n", vk::to_string(result).c_str());
	}
	EXIT_IF(m_image_index >= m_images.size());
	return Status::Success;
}

bool Swapchain::PrepareSystemOverlay() {
	if (m_system_overlay == nullptr) {
		m_system_overlay = std::make_unique<SystemOverlay>(m_window.graphic_ctx);
	}
	return m_system_overlay->PrepareFrame(m_extent, m_format, ImageCount());
}

void Swapchain::RecordPresentCommands(CommandBuffer& command, VulkanImage& source,
                                      bool draw_system_overlay) {
	if (source.state.layout != vk::ImageLayout::eTransferSrcOptimal) {
		EXIT("invalid prepared presentation image, vk_image=%p layout=%d\n",
		     static_cast<void*>(source.image), static_cast<int>(source.state.layout));
	}
	EXIT_IF(m_image_index >= m_images.size());
	auto vk_command = command.Handle();

	vk::ImageMemoryBarrier to_transfer {};
	to_transfer.sType                           = vk::StructureType::eImageMemoryBarrier;
	to_transfer.dstAccessMask                   = vk::AccessFlagBits::eTransferWrite;
	to_transfer.oldLayout                       = vk::ImageLayout::eUndefined;
	to_transfer.newLayout                       = vk::ImageLayout::eTransferDstOptimal;
	to_transfer.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_transfer.image                           = m_images[m_image_index];
	to_transfer.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_transfer.subresourceRange.baseMipLevel   = 0;
	to_transfer.subresourceRange.levelCount     = 1;
	to_transfer.subresourceRange.baseArrayLayer = 0;
	to_transfer.subresourceRange.layerCount     = 1;
	// Match the acquire wait stage so the layout transition cannot precede acquisition.
	vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                           vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlags {}, 0,
	                           nullptr, 0, nullptr, 1, &to_transfer);

	vk::ImageBlit region {};
	region.srcSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.srcSubresource.mipLevel       = 0;
	region.srcSubresource.baseArrayLayer = 0;
	region.srcSubresource.layerCount     = 1;
	region.srcOffsets[1].x               = static_cast<int>(source.extent.width);
	region.srcOffsets[1].y               = static_cast<int>(source.extent.height);
	region.srcOffsets[1].z               = 1;
	region.dstSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
	region.dstSubresource.mipLevel       = 0;
	region.dstSubresource.baseArrayLayer = 0;
	region.dstSubresource.layerCount     = 1;
	region.dstOffsets[1].x               = static_cast<int>(m_extent.width);
	region.dstOffsets[1].y               = static_cast<int>(m_extent.height);
	region.dstOffsets[1].z               = 1;
	vk_command.blitImage(source.image, vk::ImageLayout::eTransferSrcOptimal,
	                     m_images[m_image_index], vk::ImageLayout::eTransferDstOptimal, 1, &region,
	                     vk::Filter::eLinear);

	vk::ImageMemoryBarrier to_present {};
	to_present.sType         = vk::StructureType::eImageMemoryBarrier;
	to_present.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	to_present.dstAccessMask = draw_system_overlay ? vk::AccessFlagBits::eColorAttachmentRead |
	                                                     vk::AccessFlagBits::eColorAttachmentWrite
	                                               : vk::AccessFlagBits::eMemoryRead;
	to_present.oldLayout     = vk::ImageLayout::eTransferDstOptimal;
	to_present.newLayout     = draw_system_overlay ? vk::ImageLayout::eColorAttachmentOptimal
	                                               : vk::ImageLayout::ePresentSrcKHR;
	to_present.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_present.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
	to_present.image                           = m_images[m_image_index];
	to_present.subresourceRange.aspectMask     = vk::ImageAspectFlagBits::eColor;
	to_present.subresourceRange.baseMipLevel   = 0;
	to_present.subresourceRange.levelCount     = 1;
	to_present.subresourceRange.baseArrayLayer = 0;
	to_present.subresourceRange.layerCount     = 1;
	vk_command.pipelineBarrier(
	    vk::PipelineStageFlagBits::eTransfer,
	    draw_system_overlay ? vk::PipelineStageFlagBits::eColorAttachmentOutput
	                        : vk::PipelineStageFlagBits::eAllCommands,
	    vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1, &to_present);
	if (draw_system_overlay) {
		m_system_overlay->Record(vk_command, m_image_views[m_image_index]);
		to_present.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
		to_present.dstAccessMask = vk::AccessFlagBits::eMemoryRead;
		to_present.oldLayout     = vk::ImageLayout::eColorAttachmentOptimal;
		to_present.newLayout     = vk::ImageLayout::ePresentSrcKHR;
		vk_command.pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
		                           vk::PipelineStageFlagBits::eAllCommands,
		                           vk::DependencyFlagBits::eByRegion, 0, nullptr, 0, nullptr, 1,
		                           &to_present);
	}
}

uint64_t Swapchain::Submit(CommandScheduler& scheduler) {
	EXIT_IF(m_frame_index >= m_image_acquired.size() || m_image_index >= m_render_complete.size());
	SubmitInfo submit;
	submit.AddWait(m_image_acquired[m_frame_index], 1, vk::PipelineStageFlagBits::eTransfer);
	submit.AddSignal(m_render_complete[m_image_index]);
	const auto tick = scheduler.Submit(submit);
	// Bench diagnostic: which timeline ticks carry a swapchain-acquire wait.
	LOGF("[bench-present] tick=%" PRIu64 " frame_index=%u image_index=%u\n", tick, m_frame_index,
	     m_image_index);
	return tick;
}

Swapchain::Status Swapchain::Present() {
	EXIT_IF(m_image_index >= m_render_complete.size());
	const auto         ready = m_render_complete[m_image_index];
	vk::PresentInfoKHR present {};
	present.sType              = vk::StructureType::ePresentInfoKHR;
	present.swapchainCount     = 1;
	present.pSwapchains        = &m_handle;
	present.pImageIndices      = &m_image_index;
	present.pWaitSemaphores    = &ready;
	present.waitSemaphoreCount = 1;

	vk::Result result;
	{
		Common::LockGuard lock(m_window.graphic_ctx.queue_mutex);
		result = m_window.graphic_ctx.queue.presentKHR(&present);
	}
	switch (result) {
		case vk::Result::eSuccess: break;
		case vk::Result::eSuboptimalKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eSuboptimalKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorOutOfDateKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorOutOfDateKHR\n");
			return Status::Recreate;
		case vk::Result::eErrorSurfaceLostKHR:
			LOGF("vkQueuePresentKHR returned vk::Result::eErrorSurfaceLostKHR\n");
			return Status::SurfaceLost;
		default: EXIT("vkQueuePresentKHR failed: %s\n", vk::to_string(result).c_str());
	}
	m_frame_index = (m_frame_index + 1u) % static_cast<uint32_t>(m_images.size());
	return Status::Success;
}

Presenter::Presenter(WindowContext& window): m_impl(std::make_unique<Impl>(window)) {}

Presenter::~Presenter() = default;

Presenter::Frame& Presenter::PrepareFrame(CommandBuffer& buffer, const ImageInfo& info) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(buffer.IsInvalid());
	auto*             frame = m_impl->frames.Acquire();
	Common::LockGuard render_lock(m_impl->renderer.GetMutex());
	auto&             image = m_impl->ResolveSurface(info);
	if (image.backing.format == vk::Format::eUndefined) {
		EXIT("unsupported presentation source, image=%p\n", static_cast<const void*>(&image));
	}

	auto frame_format = info.pixel_format;
	switch (frame_format) {
		case vk::Format::eR8G8B8A8Srgb: frame_format = vk::Format::eR8G8B8A8Unorm; break;
		case vk::Format::eB8G8R8A8Srgb: frame_format = vk::Format::eB8G8R8A8Unorm; break;
		default: break;
	}
	frame->Configure(m_impl->window.graphic_ctx,
	                 {image.backing.extent.width, image.backing.extent.height}, frame_format);
	frame->CopyFrom(buffer, image);
	return *frame;
}

Presenter::Frame& Presenter::PrepareBlankFrame(uint32_t width, uint32_t height, bool opaque,
                                               CommandBuffer* producer) {
	KYTY_PROFILER_FUNCTION();
	auto              format = m_impl->frames.GetFormat();
	auto*             frame  = m_impl->frames.Acquire();
	Common::LockGuard render_lock(m_impl->renderer.GetMutex());
	frame->Configure(m_impl->window.graphic_ctx, {width, height}, format);
	vk::ClearColorValue clear {};
	clear.float32[3] = opaque ? 1.0f : 0.0f;
	if (producer != nullptr) {
		EXIT_IF(producer->IsInvalid());
		frame->Clear(*producer, clear);
	} else {
		auto& command = m_impl->present_scheduler.BeginCommand();
		frame->Clear(command, clear);
		frame->present_tick = m_impl->present_scheduler.Submit();
	}
	return *frame;
}

Presenter::Frame* Presenter::PrepareLastFrame() {
	return m_impl->frames.AcquireLast();
}

bool Presenter::IsGuestPaused() const noexcept {
	return m_impl->window.loop.paused.load(std::memory_order_acquire);
}

bool Presenter::NeedsSystemOverlayRefresh() const noexcept {
	const auto visual = GetSystemOverlayVisualState();
	return visual.active ||
	       visual.revision != m_impl->presented_overlay_revision.load(std::memory_order_acquire);
}

RenderContext& Presenter::Renderer() const noexcept {
	return m_impl->renderer;
}

namespace {

// Bench: KYTY_FRAME_DUMP_DIR=<dir> writes the presented guest frame every KYTY_FRAME_DUMP_EVERY_S
// seconds (default 15) as frame_<seconds>.ppm, independent of whether the host window is visible
// (a window capture of an occluded window only returns its thumbnail).
// KYTY_FRAME_VIDEO_PIPE=<fifo> instead streams frames as raw rgb24 1280x720 (for ffmpeg -f rawvideo)
// every KYTY_FRAME_DUMP_EVERY_MS milliseconds; a gap repeats the last frame, so the video keeps
// wall-clock time.
struct FrameDumpConfig {
	std::string dir;
	std::string video;
	uint64_t    every_us = 15000000;
};

const FrameDumpConfig& GetFrameDumpConfig() {
	static const FrameDumpConfig config = [] {
		FrameDumpConfig c;
		if (const char* dir = std::getenv("KYTY_FRAME_DUMP_DIR"); dir != nullptr) {
			c.dir = dir;
		}
		if (const char* every = std::getenv("KYTY_FRAME_DUMP_EVERY_S"); every != nullptr) {
			c.every_us = std::max<uint64_t>(1, std::strtoull(every, nullptr, 10)) * 1000000;
		}
		if (const char* video = std::getenv("KYTY_FRAME_VIDEO_PIPE"); video != nullptr) {
			c.video    = video;
			c.every_us = 100000;
		}
		if (const char* every = std::getenv("KYTY_FRAME_DUMP_EVERY_MS"); every != nullptr) {
			c.every_us = std::max<uint64_t>(1, std::strtoull(every, nullptr, 10)) * 1000;
		}
		return c;
	}();
	return config;
}

bool FrameDumpDue(uint64_t* stamp_s) {
	const auto& config = GetFrameDumpConfig();
	if (config.dir.empty() && config.video.empty()) {
		return false;
	}
	static const auto start = std::chrono::steady_clock::now();
	static uint64_t   next  = config.every_us;
	const auto        now_us = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() -
                                                              start)
            .count());
	if (now_us < next) {
		return false;
	}
	*stamp_s = config.video.empty() ? now_us / 1000000 : now_us;
	next     = now_us - now_us % config.every_us + config.every_us;
	return true;
}

void DecodeDumpPixel(uint32_t v, vk::Format format, uint32_t* rgb) {
	uint32_t r = 0, g = 0, b = 0;
	switch (format) {
		case vk::Format::eB8G8R8A8Unorm:
		case vk::Format::eB8G8R8A8Srgb: b = v & 0xffu, g = (v >> 8u) & 0xffu, r = (v >> 16u) & 0xffu; break;
		case vk::Format::eA2B10G10R10UnormPack32:
			r = (v & 0x3ffu) >> 2u, g = ((v >> 10u) & 0x3ffu) >> 2u, b = ((v >> 20u) & 0x3ffu) >> 2u;
			break;
		case vk::Format::eA2R10G10B10UnormPack32:
			b = (v & 0x3ffu) >> 2u, g = ((v >> 10u) & 0x3ffu) >> 2u, r = ((v >> 20u) & 0x3ffu) >> 2u;
			break;
		default: r = v & 0xffu, g = (v >> 8u) & 0xffu, b = (v >> 16u) & 0xffu; break;
	}
	rgb[0] += r, rgb[1] += g, rgb[2] += b;
}

void WriteFrameVideo(const uint8_t* pixels, uint32_t width, uint32_t height, vk::Format format,
                     uint64_t stamp_us) {
	constexpr uint32_t OUT_W = 1280, OUT_H = 720;
	const auto&        config = GetFrameDumpConfig();
	static FILE*       pipe   = std::fopen(config.video.c_str(), "wb");
	static uint64_t    last_us = 0;
	static std::vector<uint8_t> out(static_cast<size_t>(OUT_W) * OUT_H * 3);
	if (pipe == nullptr) {
		return;
	}
	const auto* src = reinterpret_cast<const uint32_t*>(pixels);
	for (uint32_t oy = 0; oy < OUT_H; oy++) {
		const uint32_t y0 = oy * height / OUT_H, y1 = std::max(y0 + 1, (oy + 1) * height / OUT_H);
		for (uint32_t ox = 0; ox < OUT_W; ox++) {
			const uint32_t x0 = ox * width / OUT_W, x1 = std::max(x0 + 1, (ox + 1) * width / OUT_W);
			uint32_t       rgb[3] {};
			for (uint32_t y = y0; y < y1; y++) {
				for (uint32_t x = x0; x < x1; x++) {
					DecodeDumpPixel(src[static_cast<size_t>(y) * width + x], format, rgb);
				}
			}
			const uint32_t n = (y1 - y0) * (x1 - x0);
			auto*          o = &out[(static_cast<size_t>(oy) * OUT_W + ox) * 3];
			o[0] = static_cast<uint8_t>(rgb[0] / n), o[1] = static_cast<uint8_t>(rgb[1] / n),
			o[2] = static_cast<uint8_t>(rgb[2] / n);
		}
	}
	const uint64_t copies =
	    last_us == 0 ? 1 : std::clamp<uint64_t>((stamp_us - last_us + config.every_us / 2) / config.every_us, 1, 600);
	last_us = stamp_us;
	for (uint64_t i = 0; i < copies; i++) {
		std::fwrite(out.data(), 1, out.size(), pipe);
	}
	std::fflush(pipe);
}

void WriteFrameDump(const uint8_t* pixels, uint32_t width, uint32_t height, vk::Format format,
                    uint64_t stamp_s) {
	if (!GetFrameDumpConfig().video.empty()) {
		WriteFrameVideo(pixels, width, height, format, stamp_s);
		return;
	}
	const auto path = GetFrameDumpConfig().dir + "/frame_" + std::to_string(stamp_s) + ".ppm";
	FILE*      file = std::fopen(path.c_str(), "wb");
	if (file == nullptr) {
		return;
	}
	std::fprintf(file, "P6\n%u %u\n255\n", width, height);
	std::vector<uint8_t> row(static_cast<size_t>(width) * 3);
	for (uint32_t y = 0; y < height; y++) {
		const auto* src = reinterpret_cast<const uint32_t*>(pixels) + static_cast<size_t>(y) * width;
		for (uint32_t x = 0; x < width; x++) {
			const uint32_t v = src[x];
			uint8_t        r = 0, g = 0, b = 0;
			switch (format) {
				case vk::Format::eB8G8R8A8Unorm:
				case vk::Format::eB8G8R8A8Srgb:
					b = v & 0xffu, g = (v >> 8u) & 0xffu, r = (v >> 16u) & 0xffu;
					break;
				case vk::Format::eA2B10G10R10UnormPack32:
					r = static_cast<uint8_t>((v & 0x3ffu) >> 2u);
					g = static_cast<uint8_t>(((v >> 10u) & 0x3ffu) >> 2u);
					b = static_cast<uint8_t>(((v >> 20u) & 0x3ffu) >> 2u);
					break;
				case vk::Format::eA2R10G10B10UnormPack32:
					b = static_cast<uint8_t>((v & 0x3ffu) >> 2u);
					g = static_cast<uint8_t>(((v >> 10u) & 0x3ffu) >> 2u);
					r = static_cast<uint8_t>(((v >> 20u) & 0x3ffu) >> 2u);
					break;
				default: r = v & 0xffu, g = (v >> 8u) & 0xffu, b = (v >> 16u) & 0xffu; break;
			}
			row[x * 3] = r, row[x * 3 + 1] = g, row[x * 3 + 2] = b;
		}
		std::fwrite(row.data(), 1, row.size(), file);
	}
	std::fclose(file);
	LOGF("[frame-dump] %s (%ux%u format=%d)\n", path.c_str(), width, height,
	     static_cast<int>(format));
}

} // namespace

void Presenter::Present(Frame& frame, bool reuse) {
	KYTY_PROFILER_FUNCTION();
	m_impl->frames.ValidateForPresent(&frame, reuse);

	const auto overlay_visual = GetSystemOverlayVisualState();
	auto&      swapchain      = m_impl->swapchain;
	// Some window systems keep presenting an old swapchain after a resize.
	if (swapchain.NeedsResize()) {
		m_impl->RecoverSwapchain(Swapchain::Status::Recreate);
	}
	std::unique_ptr<Buffer> dump;
	vk::Extent3D            dump_extent {};
	vk::Format              dump_format = vk::Format::eUndefined;
	uint64_t                dump_stamp  = 0;
	for (uint32_t attempt = 0; attempt < 2; attempt++) {
		auto status = swapchain.AcquireNextImage();
		if (status != Swapchain::Status::Success) {
			m_impl->RecoverSwapchain(status);
			continue;
		}
		{
			Common::LockGuard render_lock(m_impl->renderer.GetMutex());
			auto&             command = m_impl->present_scheduler.BeginCommand();
			const bool        draw_system_overlay =
			    overlay_visual.active && swapchain.PrepareSystemOverlay();
			swapchain.RecordPresentCommands(command, frame.image, draw_system_overlay);
			uint64_t stamp_s = 0;
			if (dump == nullptr && FrameDumpDue(&stamp_s) &&
			    frame.image.extent.width * frame.image.extent.height != 0) {
				const auto& src  = frame.image;
				const auto  size = static_cast<uint64_t>(src.extent.width) * src.extent.height * 4;
				dump = std::make_unique<Buffer>(m_impl->window.graphic_ctx, m_impl->present_scheduler,
				                                MemoryUsage::Download, 0,
				                                vk::BufferUsageFlagBits::eTransferDst, size);
				dump_extent = src.extent;
				dump_format = src.format;
				dump_stamp  = stamp_s;
				vk::BufferImageCopy region {};
				region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
				region.imageSubresource.layerCount = 1;
				region.imageExtent                 = vk::Extent3D {src.extent.width, src.extent.height, 1};
				command.Handle().copyImageToBuffer(src.image, vk::ImageLayout::eTransferSrcOptimal,
				                                   dump->Handle(), 1, &region);
				vk::BufferMemoryBarrier to_host {};
				to_host.srcAccessMask       = vk::AccessFlagBits::eTransferWrite;
				to_host.dstAccessMask       = vk::AccessFlagBits::eHostRead;
				to_host.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				to_host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
				to_host.buffer              = dump->Handle();
				to_host.size                = size;
				command.Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
				                                 vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
				                                 &to_host, 0, nullptr);
			}
			frame.present_tick = swapchain.Submit(m_impl->present_scheduler);
		}
		status = swapchain.Present();
		if (status != Swapchain::Status::Success) {
			m_impl->RecoverSwapchain(status);
			continue;
		}

		m_impl->presented_overlay_revision.store(overlay_visual.revision,
		                                         std::memory_order_release);
		m_impl->window.UpdateTitle();
		FrameTiming::OnFramePresented();
		if (dump != nullptr) {
			m_impl->present_scheduler.Wait(frame.present_tick);
			dump->Invalidate(0, dump->Size());
			if (!dump->Mapped().empty()) {
				WriteFrameDump(dump->Mapped().data(), dump_extent.width, dump_extent.height,
				               dump_format, dump_stamp);
			}
		}
		m_impl->frames.Release(&frame, true);
		return;
	}
	LOGF("Vulkan presentation retry exhausted; dropping frame\n");
	m_impl->frames.Release(&frame, reuse);
}

void Presenter::Discard(Frame& frame) {
	m_impl->frames.Release(&frame);
}

WindowContext::WindowContext() = default;

} // namespace Libs::Graphics
