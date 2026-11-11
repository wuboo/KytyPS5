#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_HOSTDCC_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_HOSTDCC_H_

// S531 DIAG. KYTY_DIAG_NO_HOST_DCC=rgba16f|all turns off host color compression for color-attachment
// images (RGBA16F only, or every such image). Unset: one getenv, then nothing. Not a fix.
// The clean switch is VK_EXT_image_compression_control DISABLED. The RX 9060 XT driver does not
// advertise it, and color images already carry STORAGE, so the fallback leaves VkImageCreateInfo
// unchanged. The log records that as path=storage-already and counts views whose format differs
// from the image (UNORM/UINT/R32G32 on RGBA16F, and clears through another view).

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::HostDcc {

inline constexpr uint64_t kCardAddress = 0x514080000ull;

enum class Mode : uint8_t { Off, Rgba16f, All };
enum class Path : uint8_t { None, Ext, StorageAdded, Already };

struct Counts {
	uint64_t disabled_ext    = 0;
	uint64_t storage_added   = 0;
	uint64_t storage_already = 0;
	uint64_t view_same       = 0;
	uint64_t view_alias      = 0;
	uint64_t clear_alias     = 0;
};

inline Counts& Stats() {
	static Counts counts;
	return counts;
}

inline const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::Rgba16f: return "rgba16f";
		case Mode::All: return "all";
		default: return "off";
	}
}

inline const char* PathName(Path path) {
	switch (path) {
		case Path::Ext: return "ext";
		case Path::StorageAdded: return "storage-added";
		case Path::Already: return "storage-already";
		default: return "none";
	}
}

inline Mode Current() {
	static const Mode mode = [] {
		const char* value  = std::getenv("KYTY_DIAG_NO_HOST_DCC");
		Mode        parsed = Mode::Off;
		if (value != nullptr) {
			if (std::strcmp(value, "rgba16f") == 0) {
				parsed = Mode::Rgba16f;
			} else if (std::strcmp(value, "all") == 0) {
				parsed = Mode::All;
			}
		}
		std::fprintf(stderr, "[s531-hd] cfg mode=%s\n", ModeName(parsed));
		return parsed;
	}();
	return mode;
}

inline bool WantDisable(vk::Format format, vk::ImageUsageFlags usage) {
	const Mode mode = Current();
	if (mode == Mode::Off) {
		return false;
	}
	if (!static_cast<bool>(usage & vk::ImageUsageFlagBits::eColorAttachment)) {
		return false;
	}
	if (mode == Mode::All) {
		return true;
	}
	return format == vk::Format::eR16G16B16A16Sfloat;
}

inline void LogSummary(const char* why) {
	const Counts& counts = Stats();
	std::fprintf(stderr,
	             "[s531-hd] summary %s ext=%llu added=%llu already=%llu view_same=%llu "
	             "view_alias=%llu clear_alias=%llu\n",
	             why, static_cast<unsigned long long>(counts.disabled_ext),
	             static_cast<unsigned long long>(counts.storage_added),
	             static_cast<unsigned long long>(counts.storage_already),
	             static_cast<unsigned long long>(counts.view_same),
	             static_cast<unsigned long long>(counts.view_alias),
	             static_cast<unsigned long long>(counts.clear_alias));
}

inline Path Apply(vk::ImageCreateInfo& create, vk::ImageCompressionControlEXT& slot,
                  bool extension_on) {
	if (!WantDisable(create.format, create.usage)) {
		return Path::None;
	}
	if (extension_on) {
		slot        = vk::ImageCompressionControlEXT {};
		slot.flags  = vk::ImageCompressionFlagBitsEXT::eDisabled;
		create.pNext = &slot;
		Stats().disabled_ext++;
		return Path::Ext;
	}
	if (!static_cast<bool>(create.usage & vk::ImageUsageFlagBits::eStorage)) {
		create.usage |= vk::ImageUsageFlagBits::eStorage;
		Stats().storage_added++;
		return Path::StorageAdded;
	}
	Stats().storage_already++;
	return Path::Already;
}

inline void LogCreate(uint64_t guest, int format, uint32_t width, uint32_t height, uint32_t usage,
                      uint32_t flags, Path path) {
	if (path == Path::None) {
		return;
	}
	static int samples = 0;
	const bool card    = guest == kCardAddress;
	if (!card && samples >= 6) {
		return;
	}
	if (!card) {
		samples++;
	}
	std::fprintf(stderr,
	             "[s531-hd] image guest=0x%llx fmt=%d %ux%u usage=0x%x flags=0x%x tiling=optimal "
	             "path=%s\n",
	             static_cast<unsigned long long>(guest), format, width, height, usage, flags,
	             PathName(path));
	if (card) {
		LogSummary("card");
	}
}

inline void NoteView(uint64_t guest, vk::Format image_format, vk::Format view_format,
                     vk::ImageUsageFlags usage) {
	if (!WantDisable(image_format, usage)) {
		return;
	}
	if (view_format == image_format) {
		const uint64_t n = ++Stats().view_same;
		if (n == 1 || n % 200000 == 0) {
			LogSummary("view");
		}
		return;
	}
	const uint64_t n = ++Stats().view_alias;
	if (n <= 24) {
		std::fprintf(stderr, "[s531-hd] view-alias guest=0x%llx image_fmt=%d view_fmt=%d n=%llu\n",
		             static_cast<unsigned long long>(guest), static_cast<int>(image_format),
		             static_cast<int>(view_format), static_cast<unsigned long long>(n));
	}
}

inline void NoteClear(uint64_t guest, vk::Format image_format, vk::Format clear_format,
                      vk::ImageUsageFlags usage) {
	if (clear_format == image_format || !WantDisable(image_format, usage)) {
		return;
	}
	const uint64_t n = ++Stats().clear_alias;
	if (n <= 16) {
		std::fprintf(stderr, "[s531-hd] clear-alias guest=0x%llx image_fmt=%d clear_fmt=%d n=%llu\n",
		             static_cast<unsigned long long>(guest), static_cast<int>(image_format),
		             static_cast<int>(clear_format), static_cast<unsigned long long>(n));
	}
}

} // namespace Libs::Graphics::HostDcc

#endif
