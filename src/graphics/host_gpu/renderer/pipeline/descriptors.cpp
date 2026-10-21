#include "graphics/host_gpu/renderer/pipeline/descriptors.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/controlFile.h"
#include "graphics/guest_gpu/lodStats.h"
#include "graphics/presentation/cpSampler.h"
#include "common/common.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/hostMemory.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/diagDump.h"
#include "graphics/shader/nullDescDiag.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/minLodView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/bindPrepMemo.h"
#include "graphics/host_gpu/renderer/pipeline/descAssert.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <optional>
#include <algorithm>
#include <atomic>
#include <bit>
#include <fmt/format.h>
#include <limits>
#include <span>
#include <vector>
#include <xxhash.h>

#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif

namespace Libs::Graphics {

namespace {

using BindingKind = ShaderRecompiler::IR::DescriptorBindingKind;

} // namespace

vk::DescriptorType NativeDescriptorType(BindingKind kind) {
	const auto image_class = ShaderRecompiler::IR::ImageBindingResourceClass(kind);
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Sampled) {
		return vk::DescriptorType::eSampledImage;
	}
	if (image_class == ShaderRecompiler::IR::ImageResourceClass::Storage) {
		return vk::DescriptorType::eStorageImage;
	}
	switch (kind) {
		case BindingKind::Samplers: return vk::DescriptorType::eSampler;
		case BindingKind::Buffers:
		case BindingKind::Gds:
		case BindingKind::BdaPagetable:
		case BindingKind::FaultBuffer:
		case BindingKind::FlattenedSrt:
		case BindingKind::ShaderData:
		case BindingKind::SharedMemory: return vk::DescriptorType::eStorageBuffer;
		case BindingKind::Count: EXIT("invalid native descriptor binding kind");
	}
	EXIT("invalid native descriptor binding kind");
}

vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture, uint32_t element) {
	if (!texture.image_id) {
		EXIT_IF(texture.image_view != nullptr || !texture.mip_views.empty());
		return {nullptr, nullptr, vk::ImageLayout::eGeneral};
	}
	vk::ImageView view = nullptr;
	if (texture.mip_views.empty()) {
		if (element == 0u) {
			view = texture.image_view;
		}
	} else if (element < texture.mip_views.size()) {
		view = texture.mip_views[element];
	}
	EXIT_IF(view == nullptr || texture.layout == vk::ImageLayout::eUndefined);
	return {nullptr, view, texture.layout};
}

static Prospero::ImageType TextureType(const ShaderTextureResource& descriptor) {
	const auto type = descriptor.Type();
	return type == Prospero::ImageType::kCube ? Prospero::ImageType::kColor2DArray : type;
}

static Prospero::ImageType TextureBaseType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor1DArray: return Prospero::ImageType::kColor1D;
		case Prospero::ImageType::kColor2DArray:
		case Prospero::ImageType::kColor2DMsaa:
		case Prospero::ImageType::kColor2DMsaaArray: return Prospero::ImageType::kColor2D;
		default: return type;
	}
}

static bool IsMultisampledTexture(Prospero::ImageType type) {
	return type == Prospero::ImageType::kColor2DMsaa ||
	       type == Prospero::ImageType::kColor2DMsaaArray;
}

static vk::DescriptorBufferInfo
NativeStorageBuffer(RenderContext& context, const PreparedBindings::BufferSource& source,
                    const ShaderRecompiler::IR::BufferResource& resource, uint32_t& buffer_offset) {
	buffer_offset = 0;

	const auto& [address, size, id] = source;
	if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
		return {context.GetBufferCache().GetBuffer(NULL_BUFFER_ID).Handle(), 0, 16};
	}
	const auto& graphics  = context.GetGraphics();
	const auto  alignment = graphics.StorageMinAlignment();
	if (size > graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange) {
		EXIT("storage buffer range is unsupported\n");
	}
	const bool metadata_read = resource.read && Config::SupportZmaskEnabled() &&
	                           (address & 0x7fffu) == 0 &&
	                           context.GetTextureCache().IsMeta(address);
	auto [buffer, offset]    = context.GetBufferCache().ObtainBuffer(
	    address, size, resource.written || metadata_read,
	    resource.formatted || (resource.read && Config::SyncRawImageBuffersEnabled()), id);
	if (metadata_read) buffer->Fill(offset, size, 0xf); // HTILE ZMask::kUncompressed.
	const auto aligned_offset = Common::AlignDown(offset, alignment);
	const auto adjustment     = offset - aligned_offset;
	const auto max_range      = graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange;
	if (adjustment >= 256 || size > max_range - adjustment) {
		EXIT("storage buffer offset adjustment is unsupported\n");
	}
	buffer_offset = static_cast<uint32_t>(adjustment);
	const vk::DescriptorBufferInfo result {buffer->Handle(), aligned_offset, size + adjustment};
	if (resource.written) {
		context.GetTextureCache().InvalidateMemoryFromGPU(address, size);
	}
	return result;
}

bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor, bool r128) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	const uint32_t     field3_expected = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     field4_expected = descriptor.Depth() | (descriptor.BaseArray5() << 16u);
	const uint32_t     field5_expected = (static_cast<uint32_t>(descriptor.PerfMod5()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.MaxMip()) << 4u);
	const bool         common          = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                                     (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                                     descriptor.fields[3] == field3_expected;
	if (r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	const bool full = common && descriptor.fields[4] == field4_expected &&
	                  descriptor.fields[5] == field5_expected;
	if (!full ||
	    (descriptor.MsaaDepth() && !IsMultisampledTexture(descriptor.Type()))) {
		return false;
	}
	const auto metadata_control = descriptor.fields[6] & 0x00ffffffu;
	if (metadata_control == 0) {
		return true;
	}
	constexpr uint32_t htile_control = 0x00280000u;
	const uint32_t expected_control  = htile_control | (descriptor.MsaaDepth() ? (1u << 10u) : 0u);
	const auto     metadata_addr     = descriptor.MetaAddr() << 8u;
	return metadata_control == expected_control && GuestRange {metadata_addr, 1}.Valid() &&
	       (metadata_addr & 0x7fffu) == 0 &&
	       descriptor.TileMode() == Prospero::TileMode::kDepth;
}

static void ValidateSampledDepthBinding(const ShaderRecompiler::IR::ImageResource& resource,
                                        const ShaderTextureResource& descriptor, const Image& image,
                                        vk::Format view_format, uint64_t size) {
	const bool resource_ok = IsSupportedSampledDepthResource(resource);
	// Comparison sampling changes the host format, not the guest metadata encoding.
	const bool encoding_ok = descriptor.TileMode() != Prospero::TileMode::kDepth ||
	                         IsSupportedDepthTextureEncoding(descriptor, resource.r128);
	const bool view_ok =
	    IsSupportedSampledDepthView(image.info.pixel_format, view_format, descriptor.DstSelXYZW());
	if (resource_ok && encoding_ok && view_ok) {
		return;
	}
	const auto descriptor_pitch =
	    TileGetTexturePitch(descriptor.Format(), static_cast<uint32_t>(descriptor.Width5()) + 1u,
	                        descriptor.TileMode());
	EXIT("unsupported sampled depth image: resource=%d encoding=%d view=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d atomic=%d compare=%d "
	     "guest_format=%u swizzle=0x%03x image_format=%d view_format=%d image_layers=%u "
	     "descriptor_type=%u base_array=%u depth=%u descriptor_pitch=%u target_pitch=%u "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, encoding_ok, view_ok, static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.read, resource.written, resource.atomic,
	     resource.depth_compare, static_cast<uint32_t>(descriptor.Format()),
	     descriptor.DstSelXYZW(), static_cast<int>(image.info.pixel_format),
	     static_cast<int>(view_format), image.info.resources.layers,
	     static_cast<uint32_t>(descriptor.Type()), descriptor.BaseArray5(), descriptor.Depth(),
	     descriptor_pitch, image.info.pitch, descriptor.Base40(), size, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

static bool IsSupportedStorageTextureDescriptor(const ShaderRecompiler::IR::ImageResource& resource,
                                                const ShaderTextureResource& descriptor) {
	const auto tile              = descriptor.TileMode();
	const bool is_color_1d       = descriptor.Type() == Prospero::ImageType::kColor1D;
	const bool is_color_1d_array = descriptor.Type() == Prospero::ImageType::kColor1DArray;
	const bool valid_1d_slice =
	    (is_color_1d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_1d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_1d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1D &&
	                   descriptor.Height5() == 0 && valid_1d_slice;
	const bool is_1d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim1DArray &&
	    is_color_1d_array && descriptor.Height5() == 0 &&
	    descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_color_2d       = descriptor.Type() == Prospero::ImageType::kColor2D;
	const bool is_color_2d_array = descriptor.Type() == Prospero::ImageType::kColor2DArray;
	const bool valid_2d_slice =
	    (is_color_2d && descriptor.Depth() == 0 && descriptor.BaseArray5() == 0) ||
	    (is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth());
	const bool is_2d =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D && valid_2d_slice;
	// Storage cube coordinates address individual faces, including partial cube views.
	const bool is_cube = resource.cube && descriptor.Type() == Prospero::ImageType::kCube &&
	                     descriptor.Width5() == descriptor.Height5() &&
	                     descriptor.BaseArray5() <= descriptor.Depth();
	const bool is_2d_array =
	    resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2DArray &&
	    ((!resource.cube && is_color_2d_array && descriptor.BaseArray5() <= descriptor.Depth()) ||
	     is_cube);
	const bool is_3d = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim3D &&
	                   descriptor.Type() == Prospero::ImageType::kColor3D &&
	                   descriptor.BaseArray5() == 0;
	TileTextureBlockLayout tile_layout {};
	bool                   supported_tile = false;
	switch (tile) {
		case Prospero::TileMode::kLinear: supported_tile = true; break;
		case Prospero::TileMode::kDepth:
			supported_tile =
			    !resource.read && !Prospero::IsFmaskTextureFormat(descriptor.Format()) &&
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard256B:
			supported_tile =
			    (is_2d || is_2d_array) &&
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		case Prospero::TileMode::kStandard4KB:
		case Prospero::TileMode::kStandard64KB:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, is_3d, tile_layout);
			break;
		case Prospero::TileMode::kRenderTarget:
			supported_tile =
			    TileGetTextureBlockLayout(descriptor.Format(), tile, false, tile_layout);
			break;
		default: break;
	}
	const auto swizzle = descriptor.DstSelXYZW();
	const bool supported_swizzle =
	    IsValidImageSwizzle(swizzle) &&
	    (swizzle == DstSel(4, 5, 6, 7) || !resource.read || resource.atomic);
	return (is_1d || is_1d_array || is_2d || is_2d_array || is_3d) && supported_tile &&
	       descriptor.BaseLevel() <= descriptor.LastLevel() && descriptor.MinLod() == 0 &&
	       supported_swizzle && descriptor.BCSwizzle() == 0 && !descriptor.MsaaDepth();
}

static bool IsSupportedStorageTextureEncoding(const ShaderRecompiler::IR::ImageResource& resource,
                                              const ShaderTextureResource& descriptor) {
	constexpr uint32_t field1_reserved_mask = 0x200fff00u;
	constexpr uint32_t field2_reserved_mask = 0xf0003000u;
	constexpr uint32_t field5_expected      = 0x00700000u;
	constexpr uint32_t field5_max_mip_mask  = 0x000000f0u;
	const uint32_t     expected_field3 = descriptor.DstSelXYZW() |
	                                     (static_cast<uint32_t>(descriptor.BaseLevel()) << 12u) |
	                                     (static_cast<uint32_t>(descriptor.LastLevel()) << 16u) |
	                                     (static_cast<uint32_t>(descriptor.TileMode()) << 20u) |
	                                     (static_cast<uint32_t>(descriptor.Type()) << 28u);
	const uint32_t     expected_field4 =
	    descriptor.Depth() | (static_cast<uint32_t>(descriptor.BaseArray5()) << 16u);
	const bool common = (descriptor.fields[1] & field1_reserved_mask) == 0 &&
	                    (descriptor.fields[2] & field2_reserved_mask) == 0 &&
	                    descriptor.fields[3] == expected_field3;
	if (resource.r128) {
		return common && descriptor.fields[4] == 0 && descriptor.fields[5] == 0 &&
		       descriptor.fields[6] == 0 && descriptor.fields[7] == 0;
	}
	return common && descriptor.fields[4] == expected_field4 &&
	       (descriptor.fields[5] & ~field5_max_mip_mask) == field5_expected;
}

void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size) {
	const auto format           = descriptor.Format();
	const bool resource_ok      = IsSupportedStorageImageResource(resource);
	const bool descriptor_ok    = IsSupportedStorageTextureDescriptor(resource, descriptor);
	const bool encoding_ok      = IsSupportedStorageTextureEncoding(resource, descriptor);
	const bool uint_resource    = resource.numeric_class == Prospero::TextureNumericClass::Uint;
	const bool raw_sint_storage = uint_resource &&
	    ShaderRecompiler::IR::RawSignedStorageFormat(resource, format) != Prospero::BufferFormat::kInvalid;
	const auto numeric_class =
	    Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	const bool raw_float_atomic = format == Prospero::BufferFormat::k32Float && uint_resource &&
	                              resource.atomic && !resource.atomic64;
	const bool format_ok =
	    raw_sint_storage || raw_float_atomic ||
	    (numeric_class != Prospero::TextureNumericClass::Unsupported &&
	     numeric_class != Prospero::TextureNumericClass::Sint &&
	     uint_resource == (numeric_class == Prospero::TextureNumericClass::Uint) &&
	     (!resource.atomic || format == (resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
	                                                       : Prospero::BufferFormat::k32UInt)));
	if (resource_ok && descriptor_ok && encoding_ok && format_ok && size != 0) {
		return;
	}
	EXIT("unsupported storage texture: resource=%d descriptor=%d encoding=%d format=%d "
	     "class=%u numeric=%u dimension=%u mip_mode=%u atomic=%d compare=%d "
	     "base_level=%u last_level=%u max_mip=%u min_lod=%u base_array=%u bc=%u msaa=%d "
	     "depth_tile_bpe=%u swizzle_ok=%d "
	     "addr=0x%016" PRIx64 " size=0x%016" PRIx64
	     " extent=%ux%ux%u type=%u format=%u tile=%u swizzle=0x%03x read=%d written=%d "
	     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
	     resource_ok, descriptor_ok, encoding_ok, format_ok,
	     static_cast<uint32_t>(resource.resource_class),
	     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(resource.dimension),
	     static_cast<uint32_t>(resource.mip_mode), resource.atomic, resource.depth_compare,
	     descriptor.BaseLevel(), descriptor.LastLevel(), descriptor.MaxMip(), descriptor.MinLod(),
	     descriptor.BaseArray5(), descriptor.BCSwizzle(), descriptor.MsaaDepth(),
	     Prospero::RenderTargetBytesPerElement(format),
	     IsValidImageSwizzle(descriptor.DstSelXYZW()), descriptor.Base40(), size,
	     static_cast<uint32_t>(descriptor.Width5()) + 1u,
	     static_cast<uint32_t>(descriptor.Height5()) + 1u,
	     static_cast<uint32_t>(descriptor.Depth()) + 1u, static_cast<uint32_t>(descriptor.Type()),
	     static_cast<uint32_t>(format), static_cast<uint32_t>(descriptor.TileMode()),
	     descriptor.DstSelXYZW(), resource.read, resource.written, descriptor.fields[0],
	     descriptor.fields[1], descriptor.fields[2], descriptor.fields[3], descriptor.fields[4],
	     descriptor.fields[5], descriptor.fields[6], descriptor.fields[7]);
}

// Stand-in for a null T# on devices without robustness2.nullDescriptor.
static TextureCache::ImageDesc NullTextureDesc(const ShaderRecompiler::IR::ImageResource& resource,
                                               TextureCache::BindingType                  binding) {
	TextureCache::ImageDesc desc {};
	switch (resource.numeric_class) {
		case Prospero::TextureNumericClass::Float:
			desc.info.guest_format = Prospero::BufferFormat::k32Float;
			break;
		case Prospero::TextureNumericClass::Uint:
			desc.info.guest_format = resource.atomic64 ? Prospero::BufferFormat::k32_32UInt
			                                         : Prospero::BufferFormat::k32UInt;
			break;
		case Prospero::TextureNumericClass::Sint:
			desc.info.guest_format = Prospero::BufferFormat::k32SInt;
			break;
		default: EXIT("null image has unsupported numeric class\n");
	}
	desc.info.pixel_format    = VulkanFormat(desc.info.guest_format);
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = Prospero::NumBytesPerElement(desc.info.guest_format);
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format     = resource.atomic64 ? vk::Format::eR64Uint : desc.info.pixel_format;
	desc.view_info.type       = vk::ImageViewType::e2D;
	desc.view_info.aspect     = vk::ImageAspectFlagBits::eColor;
	desc.view_info.usage      = binding == TextureCache::BindingType::Storage
	                                ? vk::ImageUsageFlagBits::eStorage
	                                : vk::ImageUsageFlagBits::eSampled;
	desc.type                 = binding;
	return desc;
}

static ImageViewInfo TextureViewInfo(const ShaderRecompiler::IR::ImageResource& resource,
                                     const ShaderTextureResource& descriptor, vk::Format format,
                                     const SurfaceFormatInfo& surface_format, bool storage,
                                     uint32_t view_levels, uint32_t image_layers) {
	ImageViewInfo view {};
	view.format      = format;
	view.aspect      = vk::ImageAspectFlagBits::eColor;
	view.base_level  = descriptor.BaseLevel();
	view.level_count = view_levels;
	if (descriptor.MinLod() > descriptor.LastLevel() * 256u) {
		EXIT("texture minimum LOD exceeds last mip level: min_lod=%u last_level=%u\n",
		     descriptor.MinLod(), descriptor.LastLevel());
	}
	const auto base_lod = view.base_level * 256u;
	if (descriptor.MinLod() > base_lod) {
		view.min_lod = descriptor.MinLod() - base_lod;
	}
	view.usage = storage ? vk::ImageUsageFlagBits::eStorage : vk::ImageUsageFlagBits::eSampled;
	view.mapping =
	    storage || surface_format.conversion_format != Prospero::BufferFormat::kInvalid
	        ? vk::ComponentMapping {}
	        : TextureGetComponentMapping(descriptor.DstSelXYZW(), surface_format.host_to_storage);
	switch (resource.dimension) {
		case ShaderRecompiler::Decoder::ImageDimension::Dim1D:
			view.type       = vk::ImageViewType::e1D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim1DArray:
			view.type       = vk::ImageViewType::e1DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim3D:
			view.type        = vk::ImageViewType::e3D;
			view.base_layer  = 0;
			view.layer_count = 1;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DArray:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaaArray:
			view.type       = vk::ImageViewType::e2DArray;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture array base layer is out of bounds\n");
			}
			view.layer_count = image_layers - view.base_layer;
			break;
		case ShaderRecompiler::Decoder::ImageDimension::Dim2D:
		case ShaderRecompiler::Decoder::ImageDimension::Dim2DMsaa:
			view.type       = vk::ImageViewType::e2D;
			view.base_layer = descriptor.BaseArray5();
			if (view.base_layer >= image_layers) {
				EXIT("texture base layer is out of bounds\n");
			}
			view.layer_count = 1;
			break;
		default: EXIT("unsupported texture view dimension\n");
	}
	return view;
}

static bool ResolveTextureMipView(ImageInfo& info, uint32_t physical_levels, bool metadata,
                                  uint32_t view_levels, uint32_t& base_level) {
	if (!info.IsTiled()) {
		return false;
	}
	ImageInfo physical        = info;
	physical.resources.levels = physical_levels;
	physical.UpdateSize();
	info.UpdateSize();
	const auto same_mip = [&](uint32_t target, uint32_t source) {
		const auto& left  = physical.mip_layout[target];
		const auto& right = info.mip_layout[source];
		return left.offset == right.offset && left.size == right.size &&
		       left.pitch == right.pitch && left.height == right.height &&
		       left.tail_x == right.tail_x && left.tail_y == right.tail_y &&
		       left.surface_z == right.surface_z &&
		       physical.MipExtent(target) == info.MipExtent(source);
	};
	if (physical.first_tail_level == info.first_tail_level &&
	    physical.tiled_slice_stride == info.tiled_slice_stride &&
	    physical.data.size == info.data.size) {
		bool matches = true;
		for (uint32_t level = 0; level < physical_levels; ++level) {
			matches &= same_mip(level, level);
		}
		if (matches) {
			return true;
		}
	}
	if (metadata ||
	    (info.TransferLayers() > 1 && physical.tiled_slice_stride != info.tiled_slice_stride)) {
		return false;
	}
	// T# addresses the last mip. A view can select the same stored subresources
	// with different mip indices; inaccessible mips need no host representation.
	for (uint32_t base = 0; base + view_levels <= physical_levels; ++base) {
		bool matches = true;
		for (uint32_t i = 0; i < view_levels; ++i) {
			const auto source = base_level + i;
			const auto target = base + i;
			if (!same_mip(target, source) ||
			    (target >= physical.first_tail_level) != (source >= info.first_tail_level)) {
				matches = false;
				break;
			}
		}
		if (matches) {
			info       = std::move(physical);
			base_level = base;
			return true;
		}
	}
	return false;
}

// force_tex (control file): a sampled 2D T# of a listed pixel shader slot reads as a 1x1 image of one colour.
// Without `key` only entries without a filter apply; with the decoded T# an entry with a filter applies when it fits.
static bool ForcedTextureBinding(TextureCache& texture_cache, const ShaderRecompiler::IR::ImageResource& resource,
                                 uint64_t ps_hash, uint32_t slot, const Common::ControlFile::TexKey* key,
                                 TextureBinding& out) {
	uint32_t rgba = 0;
	if (ps_hash == 0 || resource.written ||
	    resource.resource_class != ShaderRecompiler::IR::ImageResourceClass::Sampled ||
	    resource.numeric_class != Prospero::TextureNumericClass::Float ||
	    (resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim2D &&
	     resource.dimension != ShaderRecompiler::Decoder::ImageDimension::Dim2DArray) ||
	    !Common::ControlFile::Current().ForceTexFor(ps_hash, slot, &rgba, key)) {
		return false;
	}
	TextureCache::ImageDesc desc {};
	desc.info.pixel_format    = vk::Format::eR8G8B8A8Unorm;
	desc.info.guest_format    = Prospero::BufferFormat::k8_8_8_8UNorm;
	desc.info.type            = Prospero::ImageType::kColor2D;
	desc.info.extent          = {1, 1, 1};
	desc.info.resources       = {1, 1};
	desc.info.bytes_per_block = 4;
	desc.info.samples         = 1;
	desc.info.mip_layout[0]   = {0, 0, 1, 1};
	desc.view_info.format      = vk::Format::eR8G8B8A8Unorm;
	desc.view_info.type        = resource.dimension == ShaderRecompiler::Decoder::ImageDimension::Dim2D
	                                 ? vk::ImageViewType::e2D
	                                 : vk::ImageViewType::e2DArray;
	desc.view_info.aspect      = vk::ImageAspectFlagBits::eColor;
	desc.view_info.level_count = 1;
	desc.view_info.layer_count = 1;
	desc.view_info.usage       = vk::ImageUsageFlagBits::eSampled;
	desc.type                  = TextureCache::BindingType::Texture;
	const auto id              = texture_cache.GetForcedImage(rgba);
	static std::atomic<uint64_t> binds {0};
	if (const auto n = binds.fetch_add(1, std::memory_order_relaxed) + 1; n <= 16 || (n & (n - 1)) == 0) {
		LOGF("[force-tex] ps=0x%016" PRIx64 " slot=%u rgba=0x%08x binds=%" PRIu64 "\n", ps_hash, slot, rgba, n);
	}
	out = {id, nullptr, std::move(desc)};
	return true;
}

// Everything a texture binding needs before the cache lookup: a function of its inputs only.
static TextureDescription DescribeTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                          const ShaderRecompiler::IR::DescriptorValue& value,
                                          const GraphicContext& graphics) {
	if (resource.atomic64 && !graphics.shader_image_int64_atomics_enabled) {
		EXIT("64-bit image atomics require shaderImageInt64Atomics\n");
	}
	auto descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
	const bool storage = resource.written;
	if (storage) {
		ValidateStorageImageResource(resource);
	}

	if (descriptor.IsNull()) {
		TextureDescription result {};
		result.null = true;
		return result;
	}

	if (!storage && descriptor.MipStatsEnabled() && LodStats::CurrentMode() == LodStats::Mode::Real) {
		LodStats::NoteSampled(descriptor.MipStatsSlot());
	}

	const auto address         = descriptor.Base40();
	const auto width           = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto height          = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto base_level      = descriptor.BaseLevel();
	const auto last_level      = descriptor.LastLevel();
	const auto type            = TextureType(descriptor);
	const bool multisampled    = IsMultisampledTexture(type);
	const auto max_mip         = resource.r128 ? last_level : descriptor.MaxMip();
	const auto physical_levels = multisampled ? 1u : static_cast<uint32_t>(max_mip) + 1u;
	// IMAGE_STORE addresses BASE_LEVEL; only IMAGE_STORE_MIP selects other view mips.
	const bool single_storage_mip =
	    storage && resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic;
	const auto view_levels = multisampled || single_storage_mip
	                             ? 1u
	                             : static_cast<uint32_t>(last_level - base_level) + 1u;
	auto levels =
	    multisampled ? 1u : std::max(physical_levels, base_level + view_levels);
	const auto tile       = descriptor.TileMode();
	const bool depth_tile = tile == Prospero::TileMode::kDepth;
	const bool msaa_tile  = depth_tile || tile == Prospero::TileMode::kRenderTarget;
	const bool msaa_array = type == Prospero::ImageType::kColor2DMsaaArray;
	if ((!multisampled && base_level > last_level) ||
	    (multisampled &&
	     (base_level != 0 || last_level == 0 || last_level > 3 || max_mip != last_level ||
	      !msaa_tile || (descriptor.MsaaDepth() && !depth_tile) ||
	      (!msaa_array && (descriptor.Depth() != 0 || descriptor.BaseArray5() != 0))))) {
		EXIT("unsupported texture mip view: base=%u last=%u levels=%u max=%u type=%u tile=%u "
		     "class=%u numeric=%u dimension=%u mip_mode=%u read=%d written=%d "
		     "dwords=%08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n",
		     base_level, last_level, levels, descriptor.MaxMip(),
		     static_cast<uint32_t>(descriptor.Type()), static_cast<uint32_t>(tile),
		     static_cast<uint32_t>(resource.resource_class),
		     static_cast<uint32_t>(resource.numeric_class),
		     static_cast<uint32_t>(resource.dimension), static_cast<uint32_t>(resource.mip_mode),
		     resource.read, resource.written, descriptor.fields[0], descriptor.fields[1],
		     descriptor.fields[2], descriptor.fields[3], descriptor.fields[4], descriptor.fields[5],
		     descriptor.fields[6], descriptor.fields[7]);
	}
	const auto samples = multisampled ? 1u << last_level : 1u;
	const auto depth          = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto format         = descriptor.Format();
	const auto surface_format = TextureGetSurfaceFormatInfo(format);
	const bool shader_conversion =
	    surface_format.conversion_format != Prospero::BufferFormat::kInvalid;
	const bool sampled_numeric_class =
	    storage || resource.numeric_class ==
	                   Prospero::SampledTextureNumericClass(Prospero::RemapTextureFormat(format));
	if (!storage && resource.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled &&
	    !sampled_numeric_class) {
		EXIT("sampled image numeric class mismatch: numeric=%u format=%u addr=0x%016" PRIx64 "\n",
		     static_cast<uint32_t>(resource.numeric_class), static_cast<uint32_t>(format), address);
	}

	const bool    volume       = type == Prospero::ImageType::kColor3D;
	const bool    layered      = type == Prospero::ImageType::kColor1DArray ||
	                             type == Prospero::ImageType::kColor2DArray ||
	                             type == Prospero::ImageType::kColor2DMsaaArray;
	const auto    image_layers = layered ? depth : 1u;
	auto          view_base    = static_cast<uint32_t>(base_level);
	auto pixel_format = surface_format.vk_format;
	if (resource.depth_compare) {
		if (const auto* depth_format = FindGuestDepthFormatPolicy(format)) {
			pixel_format = depth_format->depth_attachment_format;
		}
	}
	const auto raw_sint_format = ShaderRecompiler::IR::RawSignedStorageFormat(resource, format);
	const auto storage_view_format = resource.atomic64 ? vk::Format::eR64Uint
	                                 : storage && resource.atomic ? vk::Format::eR32Uint
	                                 : raw_sint_format != Prospero::BufferFormat::kInvalid
	                                     ? VulkanFormat(raw_sint_format)
	                                     : SrgbStorageViewFormat(pixel_format);
	const auto view_format         = storage && storage_view_format != vk::Format::eUndefined
	                                     ? storage_view_format
	                                     : pixel_format;
	const auto block_bytes         = Prospero::BlockCompressedBytesPerBlock(format);
	TextureCache::ImageDesc desc {};
	desc.info.data         = {address, 0};
	desc.info.pixel_format = pixel_format;
	desc.info.guest_format = format;
	desc.info.type         = TextureBaseType(type);
	desc.info.extent       = {width, height, volume ? depth : 1u};
	desc.info.resources    = {levels, image_layers};
	desc.info.bytes_per_block =
	    block_bytes != 0 ? block_bytes : Prospero::NumBytesPerElement(format);
	desc.info.samples   = samples;
	desc.info.tile_mode = tile;
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		desc.info.metadata.kind = ImageMetadataKind::Dcc;
	}
	if (levels > physical_levels) {
		if (!ResolveTextureMipView(desc.info, physical_levels,
		                           !resource.r128 && descriptor.MetaCompress(), view_levels,
		                           view_base)) {
			EXIT("unsupported texture mip view changes physical layout: base=%u last=%u max=%u "
			     "extent=%ux%ux%u tile=%u\n",
			     base_level, last_level, max_mip, width, height, depth,
			     static_cast<uint32_t>(tile));
		}
	} else {
		desc.info.UpdateSize();
	}
	const auto alignment = desc.info.DataAlignment();
	EXIT_NOT_IMPLEMENTED(desc.info.data.size == 0 || alignment == 0 ||
	                     (address & (static_cast<uint64_t>(alignment) - 1u)) != 0);
	if (storage) {
		ValidateStorageTexture(resource, descriptor, desc.info.data.size);
	}
	if (!resource.r128 && descriptor.MetaCompress() && tile != Prospero::TileMode::kDepth) {
		TileSizeAlign metadata_size {};
		(void)TileGetDccSize(width, height, volume ? depth : image_layers,
		                     desc.info.bytes_per_block, physical_levels, tile, metadata_size,
		                     std::countr_zero(samples));
		desc.info.metadata.range         = {descriptor.MetaAddr() << 8u, metadata_size.size};
		desc.info.metadata.dcc_alpha_msb = descriptor.DccAlphaPos();
	}
	desc.view_info = TextureViewInfo(resource, descriptor, view_format, surface_format, storage,
	                                 view_levels, desc.info.resources.layers);
	desc.view_info.base_level = view_base;
	// The counters are added by ResolveTexture on every call, not once per memoized description.
	uint32_t minlod_notes = 0;
	// Without VK_EXT_image_view_min_lod (MoltenVK) the T# MinLod clamp cannot reach the view: fold it into a higher
	// view base level (sampled, non-Dynamic only). KYTY_TSHARP_MINLOD_APPLE=0 turns that off for A/B.
	if (desc.view_info.min_lod != 0 && !storage &&
	    resource.mip_mode != ShaderRecompiler::IR::ImageMipMode::Dynamic) {
		const auto original = desc.view_info.min_lod;
		if (MinLodIsFractional(original)) {
			minlod_notes |= TextureDescription::MinLodFrac;
		}
		if (graphics.min_lod_remap) {
			const auto mapped = MapMinLodToBaseLevel(desc.view_info.base_level,
			                                         desc.view_info.level_count, original);
			desc.view_info.base_level  = mapped.base_level;
			desc.view_info.level_count = mapped.level_count;
			desc.view_info.min_lod     = mapped.min_lod;
			minlod_notes |= TextureDescription::MinLodRemapped;
			if (resource.queries_size) {
				minlod_notes |= TextureDescription::MinLodResinfo;
			}
		}
		if (MinLodDropped(graphics.has_view_min_lod, graphics.min_lod_remap, false, false, original)) {
			minlod_notes |= TextureDescription::MinLodDropped;
		}
	}
	desc.type = storage ? TextureCache::BindingType::Storage : TextureCache::BindingType::Texture;
	TextureDescription result {};
	result.desc              = std::move(desc);
	result.descriptor        = descriptor;
	result.pixel_format      = pixel_format;
	result.view_format       = view_format;
	result.shader_conversion = shader_conversion;
	result.minlod_notes      = minlod_notes;
	return result;
}

// The fields DescribeTexture reads; it is given only these, so equal inputs give equal results.
static ShaderRecompiler::IR::ImageResource
TextureDescriptionInputs(const ShaderRecompiler::IR::ImageResource& resource) {
	ShaderRecompiler::IR::ImageResource inputs {};
	inputs.resource_class = resource.resource_class;
	inputs.numeric_class  = resource.numeric_class;
	inputs.dimension      = resource.dimension;
	inputs.mip_mode       = resource.mip_mode;
	inputs.mip_count      = resource.mip_count;
	inputs.read           = resource.read;
	inputs.written        = resource.written;
	inputs.atomic         = resource.atomic;
	inputs.atomic64       = resource.atomic64;
	inputs.depth_compare  = resource.depth_compare;
	inputs.cube           = resource.cube;
	inputs.r128           = resource.r128;
	inputs.queries_size   = resource.queries_size;
	return inputs;
}

static uint64_t TextureDescriptionKey(const ShaderRecompiler::IR::ImageResource&   inputs,
                                      const ShaderRecompiler::IR::DescriptorValue& value) {
	uint64_t   hash = 0xcbf29ce484222325ull;
	const auto mix  = [&hash](uint64_t word) {
		hash = (hash ^ word) * 0x100000001b3ull;
		hash ^= hash >> 32u;
	};
	for (uint32_t i = 0; i < value.dwords.size(); i += 2) {
		mix(value.dwords[i] | (uint64_t {value.dwords[i + 1]} << 32u));
	}
	mix(value.dword_count | (uint64_t {static_cast<uint32_t>(inputs.dimension)} << 32u));
	mix(static_cast<uint32_t>(inputs.resource_class) |
	    (uint64_t {static_cast<uint32_t>(inputs.numeric_class)} << 32u));
	mix(static_cast<uint32_t>(inputs.mip_mode) | (uint64_t {inputs.mip_count} << 32u));
	mix(uint64_t {inputs.read} | (uint64_t {inputs.written} << 8u) |
	    (uint64_t {inputs.atomic} << 16u) | (uint64_t {inputs.atomic64} << 24u) |
	    (uint64_t {inputs.depth_compare} << 32u) | (uint64_t {inputs.cube} << 40u) |
	    (uint64_t {inputs.r128} << 48u) | (uint64_t {inputs.queries_size} << 56u));
	return hash;
}

const TextureDescription&
RenderExecutor::FindTextureDescription(const ShaderRecompiler::IR::ImageResource&   resource,
                                       const ShaderRecompiler::IR::DescriptorValue& value,
                                       TexMemo::Entry<ImageId>**                    memo) {
	constexpr size_t MaxTextureDescriptions = 16384;
	const auto       inputs                 = TextureDescriptionInputs(resource);
	const auto       key                    = TextureDescriptionKey(inputs, value);
	if (const auto found = m_texture_descriptions.find(key);
	    found != m_texture_descriptions.end() && found->second.value == value &&
	    found->second.inputs == inputs) {
		if (memo != nullptr) *memo = &found->second.memo;
		return found->second.description;
	}
	if (m_texture_descriptions.size() >= MaxTextureDescriptions) {
		m_texture_descriptions.clear();
	}
	auto& entry  = m_texture_descriptions[key];
	entry.inputs = inputs;
	entry.value  = value;
	entry.description = DescribeTexture(inputs, value, m_context.GetGraphics());
	entry.memo        = {}; // S438: a new description (or a key collision) never inherits a remembered image
	if (memo != nullptr) *memo = &entry.memo;
	return entry.description;
}

static void AddMinLodNotes(uint32_t notes) {
	if (notes & TextureDescription::MinLodFrac) FrameTiming::Add(FrameTiming::Counter::MinLodFrac);
	if (notes & TextureDescription::MinLodRemapped) FrameTiming::Add(FrameTiming::Counter::MinLodRemapped);
	if (notes & TextureDescription::MinLodResinfo) FrameTiming::Add(FrameTiming::Counter::MinLodResinfo);
	if (notes & TextureDescription::MinLodDropped) FrameTiming::Add(FrameTiming::Counter::MinLodDropped);
}

static thread_local uint64_t g_s457_shader = 0; // S457 DIAG
static thread_local int g_s457_stage = -1;

TextureBinding RenderExecutor::ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
                                              const ShaderRecompiler::IR::DescriptorValue& value,
                                              uint64_t ps_hash, uint32_t slot,
                                              const NullDesc::Hint* null_hint, TexResolveNote* note) {
	auto& texture_cache = m_context.GetTextureCache();
	if (note != nullptr) *note = {};
	// force_tex depends on the shader and slot, so it stays outside of the description memo.
	if (TextureBinding forced; ForcedTextureBinding(texture_cache, resource, ps_hash, slot, nullptr, forced)) {
		return forced;
	}
	if (ps_hash != 0 && Common::ControlFile::Current().ForceTexHasFilter()) {
		const auto key_descriptor = DecodeNativeDescriptor<ShaderTextureResource>(value);
		if (!key_descriptor.IsNull()) {
			const Common::ControlFile::TexKey key {
			    key_descriptor.Base40(), static_cast<uint32_t>(key_descriptor.Format()),
			    static_cast<uint32_t>(key_descriptor.Width5()) + 1u,
			    static_cast<uint32_t>(key_descriptor.Height5()) + 1u};
			if (TextureBinding forced; ForcedTextureBinding(texture_cache, resource, ps_hash, slot, &key, forced)) {
				return forced;
			}
		}
	}
	// S438: KYTY_TEX_RESOLVE_MEMO=0 keeps the copy (the cache lookups below must not depend on the memo's storage); 1 and shadow read the
	// description in place and copy only the ImageDesc that FindImage may change and the binding keeps.
	static const bool                   memo_logged = [] {
		LOGF("[tex-resolve-memo] mode=%s\n", TexMemo::ModeName(TexMemo::EnvMode()));
		return true;
	}();
	(void)memo_logged;
	const auto                          memo_mode = TexMemo::EnvMode();
	TexMemo::Entry<ImageId>*            memo      = nullptr;
	std::optional<TextureDescription>   legacy_copy;
	std::optional<TextureCache::ImageDesc> own_desc;
	const TextureDescription&           described = [&]() -> const TextureDescription& {
		KYTY_CP_SITE(BindResolveDescCopy); // S430
		const auto& found = FindTextureDescription(resource, value, &memo);
		if (memo_mode == TexMemo::Mode::Off) {
			return legacy_copy.emplace(found);
		}
		own_desc.emplace(found.desc);
		return found;
	}();
	auto&       desc                = legacy_copy ? legacy_copy->desc : *own_desc;
	const bool  storage             = resource.written;
	if (described.null) {
		if (m_context.GetGraphics().null_descriptor_enabled) {
			return {.layout = vk::ImageLayout::eGeneral};
		}
		auto       null_desc = NullTextureDesc(resource, storage ? TextureCache::BindingType::Storage
		                                                         : TextureCache::BindingType::Texture);
		const auto null_id   = texture_cache.FindImage(null_desc);
		auto reason = NullDesc::Reason::EmptyT;
		uint64_t addr = 0;
		uint64_t shader = ps_hash;
		if (null_hint != nullptr) {
			if (null_hint->reason != 0) {
				reason = static_cast<NullDesc::Reason>(null_hint->reason);
			}
			addr = null_hint->addr;
			if (null_hint->shader != 0) {
				shader = null_hint->shader;
			}
		}
		NullDesc::NoteStandin(shader, slot, addr, reason);
		return {null_id, nullptr, std::move(null_desc)};
	}
	if (const auto notes = described.minlod_notes; notes != 0) AddMinLodNotes(notes);
	const auto& descriptor          = described.descriptor;
	const auto  address             = descriptor.Base40();
	const auto  format              = descriptor.Format();
	const auto  width               = static_cast<uint32_t>(descriptor.Width5()) + 1u;
	const auto  height              = static_cast<uint32_t>(descriptor.Height5()) + 1u;
	const auto  depth               = static_cast<uint32_t>(descriptor.Depth()) + 1u;
	const auto  base_level          = descriptor.BaseLevel();
	const auto  last_level          = descriptor.LastLevel();
	const auto  type                = TextureType(descriptor);
	const auto  tile                = descriptor.TileMode();
	const auto  max_mip             = resource.r128 ? last_level : descriptor.MaxMip();
	bool        desc_untouched      = false; // S469: FindImage left `desc` as the description made it
	auto        id                  = [&] {
		KYTY_CP_SITE(BindResolveFindImage); // S430
		if (memo_mode == TexMemo::Mode::Off) {
			return texture_cache.FindImage(desc, described.shader_conversion);
		}
		const auto pristine_view = desc.view_info;
		// The remembered image: used in `1` (the registration check, the touch and MaterializeColorClear as in FindImage), only asked in `shadow`.
		const auto metadata_base_layer = desc.view_info.base_layer;
		const auto verdict             = texture_cache.ReuseImage(*memo, desc, metadata_base_layer, memo_mode == TexMemo::Mode::On);
		FrameTiming::Add(verdict == TexMemo::Verdict::Hit        ? FrameTiming::Counter::TexMemoHitN
		                 : verdict == TexMemo::Verdict::Stale    ? FrameTiming::Counter::TexMemoInvalidateN
		                                                         : FrameTiming::Counter::TexMemoMissN);
		if (verdict == TexMemo::Verdict::Hit && memo_mode == TexMemo::Mode::On) {
			desc_untouched = true;
			return memo->id;
		}
		TextureCache::FindImageInfo info;
		const auto                  full = texture_cache.FindImage(desc, described.shader_conversion, &info);
		if (verdict == TexMemo::Verdict::Hit) {
			const bool same = TexMemo::ShadowAgrees(*memo, full, !info.view_adjusted && desc.view_info == pristine_view);
			if (!same) {
				FrameTiming::Add(FrameTiming::Counter::TexMemoNeN);
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1) < 20) {
					LOGF("[tex-memo-ne] addr=0x%llx memo=%u full=%u view_adjusted=%d exact=%d\n",
					     static_cast<unsigned long long>(desc.info.data.address), memo->id.index, full.index, info.view_adjusted ? 1 : 0,
					     info.exact ? 1 : 0);
				}
			}
		}
		desc_untouched = !info.view_adjusted && desc.view_info == pristine_view;
		if (TexMemo::CanStore(info) && full) {
			*memo = {full, info.generation, true};
		} else {
			*memo = {};
			FrameTiming::Add(FrameTiming::Counter::TexMemoUncacheableN);
		}
		return full;
	}();
	auto*       image               = &texture_cache.GetImage(id);
	const bool  stencil_association = static_cast<bool>(image->depth_id);
	{
	KYTY_CP_SITE(BindResolveValidate); // S430
	if (stencil_association) {
		id    = image->depth_id;
		image = &texture_cache.GetImage(id);
	} else if (image->info.IsDepth()) {
		if (storage) {
			EXIT("depth target cannot be bound as a storage image\n");
		}
		ValidateSampledDepthBinding(resource, descriptor, *image, described.pixel_format,
		                            desc.info.data.size);
	} else if (storage) {
		ValidateStorageColorView(image->info.pixel_format, described.view_format,
		                         descriptor.DstSelXYZW());
	} else {
		(void)SelectSampledColorView(image->info.pixel_format, described.pixel_format,
		                             descriptor.DstSelXYZW());
	}
	}
	const bool diag_once = ps_hash != 0 && DiagDump::WantsTex(ps_hash, slot);
	const bool diag_all  = ps_hash != 0 && DiagDump::WantsTexAll(ps_hash, slot);
	const bool diag_dump = ps_hash != 0 &&
	                       DiagDump::WantsDumpTex(ps_hash, slot,
	                                              Common::ControlFile::TexKey {address, static_cast<uint32_t>(format), width, height});
	DiagDump::NoteSampledTex(ps_hash, slot, address, static_cast<uint32_t>(format), width, height);
	{ // S457 DIAG: KYTY_DIAG_S457_FLIP=<n> lists every T# resolved in guest flips n..n+59
		static const long s457_flip = std::getenv("KYTY_DIAG_S457_FLIP") ? std::atol(std::getenv("KYTY_DIAG_S457_FLIP")) : -1;
		if (s457_flip >= 0 && static_cast<long>(FrameTiming::GuestFlips() + 1) >= s457_flip && static_cast<long>(FrameTiming::GuestFlips() + 1) < s457_flip + 60) {
			static FILE* s457_file = std::fopen((std::string(std::getenv("KYTY_DIAG_S457_FILE") ? std::getenv("KYTY_DIAG_S457_FILE") : "s457-tex.txt")).c_str(), "a");
			if (s457_file) {
				std::fprintf(s457_file, "[s457-tex] flip=%ld ps=0x%016" PRIx64 " slot=%u addr=0x%" PRIx64 " fmt=%u %ux%u storage=%d stage=%d sh=0x%016" PRIx64 "\n",
				             static_cast<long>(FrameTiming::GuestFlips() + 1), ps_hash, slot, address, static_cast<uint32_t>(format), width, height, storage ? 1 : 0, g_s457_stage, g_s457_shader);
				std::fflush(s457_file);
			}
		}
	}
	const bool diag_watch = ps_hash != 0 &&
	                        DiagDump::WantsWatchTex(ps_hash, slot,
	                                                Common::ControlFile::TexKey {address, static_cast<uint32_t>(format), width, height});
	if (diag_once || diag_all || diag_dump || diag_watch) {
		DiagDump::TexInfo info;
		info.flip          = FrameTiming::GuestFlips() + 1;
		info.ps            = ps_hash;
		info.slot          = slot;
		info.addr          = address;
		info.guest_format  = static_cast<uint32_t>(format);
		info.numeric_class = static_cast<uint32_t>(resource.numeric_class);
		info.vk_format     = static_cast<uint32_t>(image->info.pixel_format);
		info.width         = width;
		info.height        = height;
		info.depth         = depth;
		info.base_level    = base_level;
		info.last_level    = last_level;
		info.max_mip       = max_mip;
		info.min_lod       = descriptor.MinLod();
		info.tile_mode     = static_cast<uint32_t>(tile);
		info.pitch         = desc.info.pitch;
		info.type          = static_cast<uint32_t>(type);
		info.dimension     = static_cast<uint32_t>(resource.dimension);
		info.dst_sel       = descriptor.DstSelXYZW();
		info.image_levels  = image->info.resources.levels;
		info.image_layers  = image->info.resources.layers;
		info.byte_size     = desc.info.data.size;
		info.from_cache    = image->backing.image != nullptr;
		info.host_image    = reinterpret_cast<uint64_t>(static_cast<VkImage>(image->backing.image));
		info.gpu_modified  = image->IsGpuModified();
		info.cpu_dirty     = image->IsCpuDirty();
		info.uma           = "0"; // UMA is not part of this build
		std::copy(std::begin(descriptor.fields), std::begin(descriptor.fields) + 8, info.dwords);
		if (diag_all) {
			DiagDump::TexDraw draw;
			draw.info = info;
			// BC7 T#: decode the first blocks of mip 0 straight from guest memory (no readback, no host image).
			if (format == Prospero::BufferFormat::kBc7UNorm || format == Prospero::BufferFormat::kBc7Srgb) {
				draw.bc7     = DiagDump::Bc7Summarize(reinterpret_cast<const uint8_t*>(address), desc.info.data.size, 256);
				draw.has_bc7 = true;
			}
			DiagDump::AddTexDraw(std::move(draw));
		}
		if (diag_dump) {
			DiagDump::AddDumpTexRequest({id, info});
		}
		if (diag_watch) {
			DiagDump::WatchTexTrigger(info, desc.info.data.size);
		}
		if (diag_once) {
			LOGF("%s\n", DiagDump::TexDescribe(info).c_str());
			DiagDump::AddTexRequest({id, info});
		}
	}
	if (note != nullptr && memo_mode == TexMemo::Mode::On && desc_untouched && memo != nullptr && memo->valid &&
	    !(diag_once || diag_all || diag_dump || diag_watch)) {
		note->eligible     = true;
		note->memo         = *memo;
		note->address      = address;
		note->format       = static_cast<uint32_t>(format);
		note->width        = width;
		note->height       = height;
		note->minlod_notes = described.minlod_notes;
	}
	return {id, nullptr, std::move(desc)};
}

static vk::Sampler NativeSampler(RenderContext&                                  context,
                                 const ShaderRecompiler::IR::CompiledShaderInfo& program,
                                 uint32_t index,
                                 const ShaderRecompiler::IR::DescriptorValue& value) {
	auto        descriptor = DecodeNativeDescriptor<ShaderSamplerResource>(value);
	const auto& sampler = program.info.samplers[index];
	if (!sampler.depth_compare) {
		descriptor.fields[0] &= ~(0x7u << 12u);
	}
	if (sampler.force_point_filtering) {
		descriptor.SetPointFiltering();
	}
	return context.GetSamplerCache().GetSampler(descriptor, sampler.integer_border);
}

static vk::DescriptorBufferInfo NativeUpload(RenderContext&            context,
                                             std::span<const uint32_t> data) {
	EXIT_IF(data.empty());
	auto& command_buffer = context.GetCommandScheduler().Current();
	EXIT_IF(command_buffer.IsInvalid());
	auto&      buffer = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	const auto offset = buffer.Copy(data.data(), data.size_bytes(), 256);
	return {buffer.Handle(), offset, data.size_bytes()};
}

void RenderExecutor::BindImage(ImageId id, bool storage) {
	if (!id) {
		return;
	}
	auto& image = m_context.GetTextureCache().GetImage(id);
	if (image.info.data.Empty()) {
		return;
	}
	if (image.binding.is_bound) {
		image.binding.force_general |= image.binding.shader_write != storage;
	}
	image.binding.is_bound = true;
	image.binding.shader_write |= storage;
	m_bound_images.push_back(id);
}

void RenderExecutor::BindRenderTarget(ImageId id) {
	auto& image             = m_context.GetTextureCache().GetImage(id);
	image.binding.is_target = true;
	m_bound_images.push_back(id);
}

void RenderExecutor::ResetBindings() {
	for (const auto id: m_bound_images) {
		if (auto* image = m_context.GetTextureCache().m_slot_images.try_get(id); image != nullptr) {
			image->binding = {};
		}
	}
	m_bound_images.clear();
}

namespace {

// S469 (KYTY_BIND_PREP_MEMO): what PrepareBindings left for one program, replayed while its inputs are the same (see bindPrepMemo.h).
struct BindPrepItem {
	TextureBinding          binding; // as PrepareBindings left it (mip_views empty)
	TexMemo::Entry<ImageId> memo;    // the texture cache's remembered image of this slot (the replay asks it)
	bool                    storage = false;
	uint64_t                ps_hash = 0;
	uint32_t                slot = 0, format = 0, width = 0, height = 0, minlod_notes = 0;
	uint64_t                address = 0;
};

struct BindPrepEntry {
	std::vector<BindPrepItem> images;
	std::vector<vk::Sampler>  samplers;
	std::vector<uint32_t>     shader_data;
};

BindPrepMemo::Table<BindPrepEntry>& BindPrepTable() {
	static thread_local BindPrepMemo::Table<BindPrepEntry> table;
	return table;
}

// Everything PrepareBindings reads for one program: the shader's own resource fields the resolve depends on, the T#/S# words, the user-data words.
template <class Program, class Snapshot>
uint64_t BindInputsHash(const Program& program, const Snapshot& snapshot) {
	BindPrepMemo::InputsHash h;
	h.Program(program.shader_hash, static_cast<uint32_t>(program.stage), static_cast<uint32_t>(program.info.images.size()),
	          static_cast<uint32_t>(program.info.samplers.size()), program.bindings.ShaderDataDwords());
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		const auto& r = program.info.images[i];
		const auto& d = snapshot.images[i];
		h.ImageResource(static_cast<uint64_t>(r.resource_class) | static_cast<uint64_t>(r.numeric_class) << 8 | static_cast<uint64_t>(r.dimension) << 16 |
		                    static_cast<uint64_t>(r.mip_mode) << 24 | static_cast<uint64_t>(r.mip_count) << 32,
		                static_cast<uint64_t>(r.read) | static_cast<uint64_t>(r.written) << 1 | static_cast<uint64_t>(r.atomic) << 2 |
		                    static_cast<uint64_t>(r.atomic64) << 3 | static_cast<uint64_t>(r.depth_compare) << 4 | static_cast<uint64_t>(r.cube) << 5 |
		                    static_cast<uint64_t>(r.r128) << 6 | static_cast<uint64_t>(r.queries_size) << 7 | static_cast<uint64_t>(d.dword_count) << 16);
		h.Image(d.dwords.data(), static_cast<uint32_t>(d.dwords.size()));
	}
	for (const auto& s: program.info.samplers) {
		const auto& d = snapshot.samplers[s.snapshot_index];
		h.Sampler(d.dwords.data(), static_cast<uint32_t>(d.dwords.size()),
		          static_cast<uint32_t>(s.depth_compare) | static_cast<uint32_t>(s.force_point_filtering) << 1 | static_cast<uint32_t>(s.integer_border) << 2);
	}
	uint32_t index = 0;
	for (const auto reg: program.info.user_data_registers) h.UserData(index++, snapshot.user_data[reg - program.user_data_base]);
	return h.Digest();
}

struct BindSameState {
	uint64_t inputs     = 0;
	uint64_t generation = 0;
};

} // namespace

void RenderExecutor::PrepareBindings(const ShaderStageRuntime& runtime,
                                     PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(!runtime);
	const auto& program  = *runtime.program;
	const auto& snapshot = *runtime.resources;
	auto&       texture_cache = m_context.GetTextureCache();
	const auto  memo_mode = BindPrepMemo::EnvMode();
	static const bool memo_logged = [] {
		LOGF("[bind-prep-memo] mode=%s\n", BindPrepMemo::ModeName(BindPrepMemo::EnvMode()));
		return true;
	}();
	(void)memo_logged;
	// S469 DIAG (KYTY_DIAG_CP_PHASES=1): shadow counters of the memo - would the inputs be the same as at the previous call of this program?
	const bool bind_diag = CpPhases::Enabled();
	uint64_t   inputs = 0;
	if (bind_diag || memo_mode != BindPrepMemo::Mode::Off) inputs = BindInputsHash(program, snapshot);
	int64_t    bind_diag_begin = 0;
	bool       bind_diag_same = false;
	if (bind_diag) {
		static thread_local std::unordered_map<const void*, BindSameState> last;
		auto&      state = last[&program];
		const auto gen   = texture_cache.RegistrationGeneration();
		FrameTiming::Add(FrameTiming::Counter::BindPrepN);
		if (state.inputs == inputs) {
			FrameTiming::Add(FrameTiming::Counter::BindSameWordsN);
			if (state.generation == gen) {
				FrameTiming::Add(FrameTiming::Counter::BindSameGenN);
				bind_diag_same = true;
			}
		}
		state = {inputs, gen};
		bind_diag_begin = FrameTiming::SteadyNowNs();
	}
	struct BindDiagEnd {
		bool    on;
		bool    same;
		int64_t begin;
		~BindDiagEnd() {
			if (on && same) FrameTiming::Add(FrameTiming::Counter::BindSameGenNs, FrameTiming::ElapsedNs(begin, FrameTiming::SteadyNowNs()));
		}
	} bind_diag_end {bind_diag, bind_diag_same, bind_diag_begin};

	const uint64_t control_gen = memo_mode != BindPrepMemo::Mode::Off ? Common::ControlFile::Current().gen : 0;
	BindPrepEntry* entry       = memo_mode != BindPrepMemo::Mode::Off ? BindPrepTable().Find(&program, inputs, control_gen) : nullptr;
	// Every remembered image must still be the one the texture cache would hand out (commit: also touches it and runs MaterializeColorClear).
	const auto images_hit = [&](const BindPrepEntry& e, bool commit) {
		for (const auto& item: e.images) {
			if (texture_cache.ReuseImage(item.memo, item.binding.desc, item.binding.desc.view_info.base_layer, commit) != TexMemo::Verdict::Hit) return false;
			const auto& image = texture_cache.GetImage(item.memo.id);
			const auto  live  = static_cast<bool>(image.depth_id) ? image.depth_id : item.memo.id;
			if (!(live == item.binding.image_id)) return false;
		}
		return true;
	};

	prepared.runtime = &runtime;
	prepared.gds = {nullptr, 0, VK_WHOLE_SIZE};
	prepared.flattened_srt = {};
	prepared.shader_data_buffer = {};
	prepared.shared_memory = {};
	prepared.images.resize(program.info.images.size());
	if (entry != nullptr && memo_mode == BindPrepMemo::Mode::On && images_hit(*entry, true)) {
		for (uint32_t i = 0; i < entry->images.size(); i++) {
			const auto& item = entry->images[i];
			DiagDump::NoteSampledTex(item.ps_hash, item.slot, item.address, item.format, item.width, item.height);
			AddMinLodNotes(item.minlod_notes);
			FrameTiming::Add(FrameTiming::Counter::TexMemoHitN);
			BindImage(item.binding.image_id, item.storage);
			prepared.images[i] = item.binding;
		}
		prepared.samplers    = entry->samplers;
		prepared.shader_data = entry->shader_data;
		if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
			prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
		}
		FrameTiming::Add(FrameTiming::Counter::BindPrepMemoHitN);
		return;
	}
	const bool predicted = entry != nullptr && memo_mode == BindPrepMemo::Mode::Shadow && images_hit(*entry, false);
	if (memo_mode != BindPrepMemo::Mode::Off) FrameTiming::Add(entry == nullptr ? FrameTiming::Counter::BindPrepMemoMissN : FrameTiming::Counter::BindPrepMemoStaleN);
	prepared.samplers.clear();
	prepared.shader_data.clear();
	static thread_local std::vector<TexResolveNote> notes;
	if (memo_mode != BindPrepMemo::Mode::Off) notes.assign(program.info.images.size(), TexResolveNote {});
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		KYTY_CP_SITE(BindResolveTexture);
		NullDesc::Hint null_hint {};
		if (i < snapshot.image_null_notes.size()) {
			null_hint.reason = snapshot.image_null_notes[i].reason;
			null_hint.addr   = snapshot.image_null_notes[i].addr;
		}
		null_hint.shader = program.shader_hash;
		g_s457_shader = program.shader_hash; g_s457_stage = static_cast<int>(program.stage); // S457 DIAG
		auto binding = ResolveTexture(program.info.images[i], snapshot.images[i],
		                              program.stage == ShaderType::Pixel ? program.shader_hash : 0, i, &null_hint,
		                              memo_mode != BindPrepMemo::Mode::Off ? &notes[i] : nullptr);
		BindImage(binding.image_id, binding.desc.type == TextureCache::BindingType::Storage);
		binding.mip_views.swap(prepared.images[i].mip_views);
		binding.mip_views.clear();
		prepared.images[i] = std::move(binding);
	}
	prepared.samplers.reserve(program.info.samplers.size());
	{
		KYTY_CP_SITE(BindSamplers);
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			prepared.samplers.push_back(NativeSampler(
			    m_context, program, i, snapshot.samplers[program.info.samplers[i].snapshot_index]));
		}
	}
	prepared.shader_data.reserve(program.bindings.ShaderDataDwords());
	for (const auto reg: program.info.user_data_registers) {
		prepared.shader_data.push_back(snapshot.user_data[reg - program.user_data_base]);
	}
	prepared.shader_data.resize(program.bindings.ShaderDataDwords());
	if (program.bindings.descriptor_counts[static_cast<size_t>(BindingKind::Gds)] != 0) {
		prepared.gds.buffer = m_context.GetBufferCache().GetGdsBuffer()->Handle();
	}
	if (memo_mode == BindPrepMemo::Mode::Off) return;
	if (predicted) {
		// Shadow: the replay would have returned the stored entry; it must equal what the full path just made.
		bool same = entry->images.size() == prepared.images.size() && entry->samplers == prepared.samplers && entry->shader_data == prepared.shader_data;
		for (uint32_t i = 0; same && i < entry->images.size(); i++) {
			const auto& a = entry->images[i].binding;
			const auto& b = prepared.images[i];
			same = a.image_id == b.image_id && a.image_view == b.image_view && a.layout == b.layout && a.desc.type == b.desc.type &&
			       a.desc.view_info == b.desc.view_info && a.desc.info.data.address == b.desc.info.data.address &&
			       a.desc.info.data.size == b.desc.info.data.size && (entry->images[i].storage == (b.desc.type == TextureCache::BindingType::Storage));
		}
		FrameTiming::Add(FrameTiming::Counter::BindPrepMemoPredictedN);
		if (!same) {
			FrameTiming::Add(FrameTiming::Counter::BindPrepMemoNeN);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 16) {
				LOGF("[bind-prep-memo-ne] shader=0x%llx images=%zu samplers=%zu\n", static_cast<unsigned long long>(program.shader_hash),
				     prepared.images.size(), prepared.samplers.size());
			}
		}
	}
	bool eligible = true;
	for (const auto& note: notes) eligible &= note.eligible;
	if (!eligible) {
		BindPrepTable().Erase(&program);
		FrameTiming::Add(FrameTiming::Counter::BindPrepMemoUncacheableN);
		return;
	}
	BindPrepEntry stored;
	stored.images.resize(prepared.images.size());
	for (uint32_t i = 0; i < prepared.images.size(); i++) {
		auto& item        = stored.images[i];
		item.binding      = prepared.images[i];
		item.memo         = notes[i].memo;
		item.storage      = prepared.images[i].desc.type == TextureCache::BindingType::Storage;
		item.ps_hash      = program.stage == ShaderType::Pixel ? program.shader_hash : 0;
		item.slot         = i;
		item.address      = notes[i].address;
		item.format       = notes[i].format;
		item.width        = notes[i].width;
		item.height       = notes[i].height;
		item.minlod_notes = notes[i].minlod_notes;
	}
	stored.samplers    = prepared.samplers;
	stored.shader_data = prepared.shader_data;
	BindPrepTable().Store(&program, inputs, control_gen, std::move(stored));
	FrameTiming::Add(FrameTiming::Counter::BindPrepMemoStoreN);
}

void RenderExecutor::FindBuffers(std::span<PreparedBindings* const> stages) {
	KYTY_PROFILER_FUNCTION();
	auto& cache = m_context.GetBufferCache();
	for (auto* stage: stages) {
		auto& prepared = *stage;
		EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
		const auto& program  = *prepared.runtime->program;
		const auto& snapshot = *prepared.runtime->resources;
		const auto& layout = program.bindings;
		prepared.buffer_sources.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
		if (layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)] == 0) {
			continue;
		}
		for (uint32_t resource = 0; resource < program.info.buffers.size(); ++resource) {
			const auto& metadata = program.info.buffers[resource];
			if (metadata.descriptor_index == UINT32_MAX) continue;
			auto& source = prepared.buffer_sources[metadata.descriptor_index];
			const auto descriptor = DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[resource]);
			const auto address = descriptor.Base48();
			auto size = descriptor.GetSize();
			if (address < BufferCache::CACHING_PAGESIZE || size == 0) {
				source = {};
				continue;
			}
			if (descriptor.NumRecords() == UINT32_MAX) {
				if (program.info.buffers[resource].written) {
					// Sentinel write ranges end before tables captured by any bound stage.
					for (const auto* reader: stages) {
						for (const auto [read_address, read_size]: reader->runtime->resources->specialization_reads) {
							if (read_size == 0) continue;
							if (read_address > address) {
								size = std::min(size, read_address - address);
							} else if (address - read_address < read_size) {
								EXIT("scalar resource reads overlap a shader buffer write\n");
							}
						}
					}
				} else {
					const auto& graphics = m_context.GetGraphics();
					const auto limit = uint64_t {graphics.GetPhysicalDeviceProperties().limits.maxStorageBufferRange} -
					                   (graphics.StorageMinAlignment() - 1);
					size = std::min({size, uint64_t {256} * 1024 * 1024, limit});
				}
			}
			size = Libs::LibKernel::Memory::ClampRangeSize(address, size);
			source = {address, size, cache.FindBuffer(address, size)};
			{ // S458 DIAG: KYTY_DIAG_S458_SRT=<flip> dumps the first 64 dwords of every buffer of PS 2c35.. in flips n..n+5
				static const long s458_flip = std::getenv("KYTY_DIAG_S458_SRT") ? std::atol(std::getenv("KYTY_DIAG_S458_SRT")) : -1;
				const long gf = static_cast<long>(FrameTiming::GuestFlips() + 1);
				if (s458_flip >= 0 && gf >= s458_flip && gf < s458_flip + 150 && program.shader_hash == 0x2c3545445213721eull && size >= 4) {
					static int s458_n = 0; if (s458_n++ >= 40) continue;
					if (FILE* fp = std::fopen(std::getenv("KYTY_DIAG_S458_FILE") ? std::getenv("KYTY_DIAG_S458_FILE") : "s458-srt.txt", "a")) {
						std::fprintf(fp, "[s458-srt] flip=%ld res=%u addr=0x%" PRIx64 " size=%" PRIu64 "\n", gf, resource, address, size);
						const auto* d = reinterpret_cast<const uint32_t*>(address);
						for (int i = 0; i < 64 && i * 4 < static_cast<int>(size); i++) {
							float f; std::memcpy(&f, &d[i], 4);
							std::fprintf(fp, "  d%02d=%08x %g\n", i, d[i], static_cast<double>(f));
						}
						std::fclose(fp);
					}
				}
			}
		}
	}
}

void RenderExecutor::RebindBuffers(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	KYTY_CP_SITE(BindRebindBuffers);
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program   = *prepared.runtime->program;
	const auto& snapshot  = *prepared.runtime->resources;
	const auto& layout    = program.bindings;
	EXIT_IF(prepared.buffer_sources.size() != layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);

	prepared.buffers.resize(layout.descriptor_counts[static_cast<size_t>(BindingKind::Buffers)]);
	EXIT_IF(prepared.shader_data.size() != layout.ShaderDataDwords());
	std::fill(prepared.shader_data.begin() + layout.memory_offset_dword, prepared.shader_data.end(),
	          0);
	auto pack_memory_offset = [&](uint32_t index, uint32_t offset) {
		const auto dword = layout.memory_offset_dword + index / 4u;
		const auto shift = (index % 4u) * 8u;
		prepared.shader_data[dword] |= offset << shift;
	};
	for (const auto& resource: program.info.buffers) {
		const auto index = resource.descriptor_index;
		if (index == UINT32_MAX) continue;
		uint32_t buffer_offset = 0;
		prepared.buffers[index] = NativeStorageBuffer(m_context, prepared.buffer_sources[index],
		                                              resource, buffer_offset);
		pack_memory_offset(index, buffer_offset);
	}
	KYTY_CP_SITE(BindNativeUpload);
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::FlattenedSrt)] != 0) {
		prepared.flattened_srt = NativeUpload(m_context, snapshot.flattened_srt);
	}
	if (layout.descriptor_counts[static_cast<size_t>(BindingKind::ShaderData)] != 0) {
		prepared.shader_data_buffer = NativeUpload(m_context, prepared.shader_data);
	}
}

void RenderExecutor::RebindImages(PreparedBindings& prepared) {
	KYTY_PROFILER_FUNCTION();
	KYTY_CP_SITE(BindRebindImages);
	EXIT_IF(prepared.runtime == nullptr || !*prepared.runtime);
	const auto& program  = *prepared.runtime->program;
	const auto& snapshot = *prepared.runtime->resources;
	auto&       images   = prepared.images;
	EXIT_IF(images.size() != program.info.images.size());
	auto& texture_cache = m_context.GetTextureCache();
	// Acquire each view before a later overlapping descriptor can replace its image.
	for (uint32_t i = 0; i < program.info.images.size(); i++) {
		if (!images[i].image_id) continue;
		const auto old_image = texture_cache.m_slot_images.try_get(images[i].image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			NullDesc::Hint null_hint {};
			if (i < snapshot.image_null_notes.size()) {
				null_hint.reason = snapshot.image_null_notes[i].reason;
				null_hint.addr   = snapshot.image_null_notes[i].addr;
			}
			null_hint.shader = program.shader_hash;
			g_s457_shader = program.shader_hash; g_s457_stage = static_cast<int>(program.stage); // S457 DIAG
			KYTY_CP_SITE(BindResolveTexture);
			images[i] = ResolveTexture(program.info.images[i], snapshot.images[i],
			                           program.stage == ShaderType::Pixel ? program.shader_hash : 0, i, &null_hint);
			BindImage(images[i].image_id,
			          images[i].desc.type == TextureCache::BindingType::Storage);
		}
		auto& binding = images[i];
		binding.mip_views.clear();
		const auto& resource = program.info.images[i];
		if (resource.mip_mode == ShaderRecompiler::IR::ImageMipMode::Dynamic) {
			EXIT_IF(resource.mip_count == 0u ||
			        resource.mip_count != binding.desc.view_info.level_count);
			binding.mip_views.reserve(resource.mip_count);
			for (uint32_t mip = 0; mip < resource.mip_count; mip++) {
				auto desc = binding.desc;
				desc.view_info.base_level += mip;
				desc.view_info.level_count = 1;
				// The shader selects the mip after applying the guest minimum LOD.
				desc.view_info.min_lod = 0;
				binding.mip_views.push_back(texture_cache.FindTexture(binding.image_id, desc));
			}
			binding.image_view = binding.mip_views.front();
		} else {
			binding.image_view = texture_cache.FindTexture(binding.image_id, binding.desc);
		}
		auto&      image   = texture_cache.GetImage(binding.image_id);
		const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
		image.usage.storage |= storage;
	}
}

void RenderExecutor::PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
                                             std::span<RenderColorInfo>         colors) {
	bool uses_dma = false;
	{
		KYTY_CP_SITE(BindFindBuffers);
		CpPhases::SubTimer timer(FrameTiming::Counter::BindFindNs); // S463 DIAG
		FindBuffers(stages);
	}
	for (auto* stage: stages) {
		uses_dma |= stage->runtime->program->info.uses_dma;
	}
	if (uses_dma) {
		CpPhases::SubTimer timer(FrameTiming::Counter::BindBdaNs); // S463 DIAG
		m_context.PrepareBda();
	}
	{
		CpPhases::SubTimer timer(FrameTiming::Counter::BindImgNs); // S463 DIAG
		for (auto* stage: stages) {
			RebindImages(*stage);
		}
	}
	auto& cache = m_context.GetTextureCache();
	std::optional<CpPhases::SubTimer> rt_timer(std::in_place, FrameTiming::Counter::BindRtNs); // S463 DIAG
	for (auto& target: colors) {
		EXIT_IF(!target.image_id);
		const auto old_image = cache.m_slot_images.try_get(target.image_id);
		if (old_image == nullptr || (!old_image->registered && !old_image->info.data.Empty()) ||
		    old_image->binding.needs_rebind) {
			if (old_image != nullptr) {
				old_image->binding = {};
			}
			target.desc.view_info.base_level = target.guest_mip_level;
			target.desc.view_info.base_layer = target.guest_array_layer;
			target.image_id                  = cache.FindImage(target.desc);
			BindRenderTarget(target.image_id);
		}
	}
	// Discovery can read back PS5 metadata and submit the scheduler. Reserve draw buffers only
	// after image identities are final; attachment layout transitions follow buffer alias copies.
	rt_timer.reset();
	CpPhases::SubTimer buf_timer(FrameTiming::Counter::BindBufNs); // S463 DIAG
	for (auto* stage: stages) {
		RebindBuffers(*stage);
	}
}

// Resource-state half of CommitBindings(): the GDS dependency and each bound image's layout/access
// transition, both of which can end the current render pass (Image::Transit ends rendering whenever
// it records a barrier). Kept separate so a caller that must bind inside an already-begun render pass
// can run it first, outside of it -- see FlushPendingPrototypeBatch(). Image transitions are no-ops
// once an image is in the requested state, so the second run inside CommitBindings() records
// nothing; the GDS barrier is unconditional, which is why GDS draws are not batched.
void RenderExecutor::TransitionBoundResources(CommandBuffer&                     buffer,
                                              std::span<PreparedBindings* const> prepared_bindings,
                                              bool prototype_vertex_capture) {
	auto vk_buffer = buffer.Handle();
	for (auto* prepared: prepared_bindings) {
		const auto& program     = *prepared->runtime->program;
		auto&       descriptors = *prepared;
		const auto  shader_stage =
		    (prototype_vertex_capture ? vk::ShaderStageFlagBits::eCompute
		                                       : NativeShaderStage(program.stage));
		const auto shader_stages = ShaderPipelineStages(shader_stage);
		if (descriptors.gds.buffer != nullptr) {
			buffer.EndRendering();
			const auto barrier = MakeGdsDependency(descriptors.gds.buffer);
			vk_buffer.pipelineBarrier(
			    vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer |
			        vk::PipelineStageFlagBits::eAllGraphics |
			        vk::PipelineStageFlagBits::eComputeShader,
			    shader_stages, vk::DependencyFlags {}, 0, nullptr, 1, &barrier, 0, nullptr);
		}

		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			auto& binding = descriptors.images[i];
			if (!binding.image_id) continue;
			auto& image = m_context.GetTextureCache().GetImage(binding.image_id);
			const auto&                 view = binding.desc.view_info;
			const ImageSubresourceRange range {view.base_level, view.level_count, view.base_layer,
			                                   view.layer_count};
			const bool storage = binding.desc.type == TextureCache::BindingType::Storage;
			if (image.info.data.Empty()) {
				image.Transit(vk::ImageLayout::eGeneral,
				              storage ? vk::AccessFlagBits2::eShaderRead |
				                            vk::AccessFlagBits2::eShaderWrite
				                      : vk::AccessFlagBits2::eShaderRead,
				              range, vk_buffer);
			} else if (image.binding.is_target) {
				const auto layout = image.binding.attachment_layout;
				EXIT_IF(layout == vk::ImageLayout::eUndefined);
				if (image.info.IsDepth()) {
					const auto host_view =
					    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
					EXIT_IF(storage || host_view == image.views.end());
					const auto aspect = host_view->info.aspect;
					if (aspect & ~DepthReadableAspects(layout)) {
						EXIT("sampling a writable depth/stencil attachment aspect\n");
					}
				}
				image.Transit(layout,
				              image.binding.attachment_access | vk::AccessFlagBits2::eShaderRead |
				                  (image.binding.shader_write ? vk::AccessFlagBits2::eShaderWrite
				                                              : vk::AccessFlags2 {}),
				              {}, vk_buffer);
			} else if (image.binding.force_general && !image.info.IsDepth()) {
				const vk::AccessFlags2 storage_access = image.binding.shader_write
				                                            ? vk::AccessFlagBits2::eShaderWrite
				                                            : vk::AccessFlags2 {};
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | storage_access, {}, vk_buffer);
			} else if (storage) {
				image.Transit(vk::ImageLayout::eGeneral,
				              vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eShaderWrite,
				              range, vk_buffer);
			} else {
				image.Transit(image.info.IsDepth() ? vk::ImageLayout::eDepthStencilReadOnlyOptimal
				                                   : vk::ImageLayout::eShaderReadOnlyOptimal,
				              vk::AccessFlagBits2::eShaderRead, range, vk_buffer);
			}
			binding.layout = image.backing.state.layout;
		}
	}
}

void MergeGuestRanges(std::vector<GuestRange>& ranges) {
	std::sort(ranges.begin(), ranges.end(),
	          [](const GuestRange& a, const GuestRange& b) { return a.address < b.address; });
	size_t merged = 0;
	for (const auto range: ranges) {
		if (range.size == 0 || range.address > UINT64_MAX - range.size) {
			EXIT("invalid image overlap range\n");
		}
		if (merged != 0 && range.address <= ranges[merged - 1].End()) {
			auto& last = ranges[merged - 1];
			last.size  = std::max(last.End(), range.End()) - last.address;
		} else {
			ranges[merged++] = range;
		}
	}
	ranges.resize(merged);
}

bool ScalarReadsMayOverlap(std::span<const std::pair<uint64_t, uint64_t>> reads,
                           std::span<const GuestRange>                    writes) {
	if (writes.empty()) {
		return false;
	}
	for (const auto [address, size]: reads) {
		if (size == 0 || address > UINT64_MAX - size) {
			return true;
		}
		const auto next = std::upper_bound(
		    writes.begin(), writes.end(), address,
		    [](uint64_t value, const GuestRange& range) { return value < range.End(); });
		if (next != writes.end() && next->address < address + size) {
			return true;
		}
	}
	return false;
}

void RenderExecutor::GatherWriteRanges(std::span<PreparedBindings* const> prepared_bindings) {
	m_write_ranges.clear();
	for (const auto id: m_bound_images) {
		const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
		if (image == nullptr || (!image->binding.shader_write && !image->binding.is_target)) {
			continue;
		}
		for (const auto written:
		     {image->info.data, image->info.stencil, image->info.metadata.range}) {
			if (written.size != 0) {
				m_write_ranges.push_back(written);
			}
		}
	}
	for (const auto* writer: prepared_bindings) {
		const auto& program = *writer->runtime->program;
		for (const auto& resource: program.info.buffers) {
			if (!resource.written || resource.descriptor_index == UINT32_MAX) continue;
			const auto& written = writer->buffer_sources[resource.descriptor_index];
			if (written.size != 0) {
				m_write_ranges.push_back({written.address, written.size});
			}
		}
	}
	MergeGuestRanges(m_write_ranges);
}

void RenderExecutor::CommitBindings(CommandBuffer&                     buffer,
                                    vk::PipelineBindPoint              pipeline_bind_point,
                                    const PipelineCache::Pipeline&     pipeline,
                                    std::span<PreparedBindings* const> prepared_bindings) {
	KYTY_PROFILER_FUNCTION();
	auto   vk_buffer        = buffer.Handle();
	size_t image_descriptor_count = 0;
	size_t write_count      = 0;
	ShaderRecompiler::IR::PushData push_data;
	bool                           has_push_data = false;
	constexpr auto                 GraphicsStages =
	    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eMeshEXT |
	    vk::ShaderStageFlagBits::eTessellationControl |
	    vk::ShaderStageFlagBits::eTessellationEvaluation | vk::ShaderStageFlagBits::eFragment;
	vk::ShaderStageFlags push_stages = pipeline_bind_point == vk::PipelineBindPoint::eGraphics
	                                       ? vk::ShaderStageFlagBits::eFragment
	                                       : vk::ShaderStageFlags {};
	for (const auto* prepared: prepared_bindings) {
		EXIT_IF(prepared == nullptr || prepared->runtime == nullptr || !*prepared->runtime);
		const auto& program = *prepared->runtime->program;
		write_count += std::popcount(program.bindings.descriptor_mask);
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto count = program.bindings.descriptor_counts[index];
			if (index >= ShaderRecompiler::IR::FirstImageBinding &&
			    index <= static_cast<size_t>(BindingKind::Samplers)) {
				image_descriptor_count += count;
			}
		}
		const auto shader_stage =
		    (pipeline.prototype_vertex_capture ? vk::ShaderStageFlagBits::eCompute
		                                       : NativeShaderStage(program.stage));
		push_stages |= shader_stage;
		EXIT_IF((pipeline_bind_point == vk::PipelineBindPoint::eGraphics &&
		         (shader_stage & GraphicsStages) == vk::ShaderStageFlags {}) ||
		        (pipeline_bind_point == vk::PipelineBindPoint::eCompute &&
		         shader_stage != vk::ShaderStageFlagBits::eCompute));
	}
	// KYTY_DESC_ASSERT: the overlap checks below are pure EXIT assertions of the recompiler's own invariants; 0 / sample skip them.
	static const DescAssert::Mode assert_mode = DescAssert::GlobalMode();
	bool                          run_assertions = true;
	if (assert_mode != DescAssert::Mode::On) {
		static std::atomic<uint64_t> assert_draws {0};
		run_assertions = DescAssert::ShouldRun(assert_mode, assert_draws.fetch_add(1, std::memory_order_relaxed));
		FrameTiming::Add(run_assertions ? FrameTiming::Counter::DescAssertRunN : FrameTiming::Counter::DescAssertSkipN);
	} else {
		FrameTiming::Add(FrameTiming::Counter::DescAssertRunN);
	}
	bool writes_gathered = false;
	for (const auto* reader: prepared_bindings) {
		const auto& reads = reader->runtime->resources->specialization_reads;
		if (reads.empty() || !run_assertions) continue;
		for (const auto* writer: prepared_bindings) {
			if (writer->runtime->program->has_address_writes) {
				EXIT("scalar resource reads cannot be proven disjoint from shader address writes\n");
			}
		}
		if (!writes_gathered) {
			writes_gathered = true;
			GatherWriteRanges(prepared_bindings);
		}
		if (!ScalarReadsMayOverlap(reads, m_write_ranges)) continue;
		// Name the overlapping write.
		for (const auto [address, size]: reads) {
			for (const auto id: m_bound_images) {
				const auto* image = m_context.GetTextureCache().m_slot_images.try_get(id);
				if (image == nullptr ||
				    (!image->binding.shader_write && !image->binding.is_target)) continue;
				for (const auto written: {image->info.data, image->info.stencil,
				                          image->info.metadata.range}) {
					if (written.size != 0 && ImageRangeOverlaps(address, size,
					                                          written.address, written.size)) {
						EXIT("scalar resource reads overlap an image or attachment write\n");
					}
				}
			}
			for (const auto* writer: prepared_bindings) {
				const auto& program = *writer->runtime->program;
				for (const auto& resource: program.info.buffers) {
					if (!resource.written || resource.descriptor_index == UINT32_MAX) continue;
					const auto& written = writer->buffer_sources[resource.descriptor_index];
					if (written.size != 0 && ImageRangeOverlaps(address, size,
					                                          written.address, written.size)) {
						EXIT("scalar resource reads overlap a shader buffer write\n");
					}
				}
			}
		}
	}
	m_descriptor_images.resize(image_descriptor_count);
	m_descriptor_writes.clear();
	m_descriptor_writes.reserve(write_count);
	size_t image_cursor = 0;
	std::array<vk::DescriptorBufferInfo, 2> address_buffers;

	TransitionBoundResources(buffer, prepared_bindings, pipeline.prototype_vertex_capture);

	for (auto* prepared: prepared_bindings) {
		const auto& program     = *prepared->runtime->program;
		auto&       descriptors = *prepared;

		std::array<size_t, ShaderRecompiler::IR::ImageBindingCount> image_starts {};
		for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
			const auto index = std::countr_zero(mask);
			const auto kind = static_cast<BindingKind>(index);
			vk::WriteDescriptorSet write {};
			write.dstBinding = ShaderRecompiler::IR::NativeBinding(program.stage, kind);
			write.descriptorType = NativeDescriptorType(kind);
			write.descriptorCount = program.bindings.descriptor_counts[index];
			const auto image_start = image_cursor;
			if (ShaderRecompiler::IR::ImageBindingResourceClass(kind) !=
			    ShaderRecompiler::IR::ImageResourceClass::None) {
				image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] = image_start;
				image_cursor += write.descriptorCount;
			} else {
				switch (kind) {
					case BindingKind::Buffers:
						EXIT_IF(descriptors.buffers.size() != write.descriptorCount);
						for (const auto& view: descriptors.buffers) {
							EXIT_IF(view.buffer == nullptr);
						}
						write.pBufferInfo = descriptors.buffers.data();
						break;
					case BindingKind::BdaPagetable:
					case BindingKind::FaultBuffer: {
						auto& cache = m_context.GetBufferCache();
						const auto* bda_buffer = kind == BindingKind::BdaPagetable
						                             ? cache.GetBdaPageTableBuffer()
						                             : cache.GetFaultBuffer();
						auto& view = address_buffers[kind == BindingKind::FaultBuffer];
						view = {bda_buffer->Handle(), 0, bda_buffer->Size()};
						write.pBufferInfo = &view;
						break;
					}
					case BindingKind::FlattenedSrt:
					case BindingKind::ShaderData:
					case BindingKind::SharedMemory:
					case BindingKind::Gds: {
						const vk::DescriptorBufferInfo* view = &descriptors.gds;
						if (kind == BindingKind::FlattenedSrt) {
							view = &descriptors.flattened_srt;
						} else if (kind == BindingKind::ShaderData) {
							view = &descriptors.shader_data_buffer;
						} else if (kind == BindingKind::SharedMemory) {
							view = &descriptors.shared_memory;
						}
						EXIT_IF(view->buffer == nullptr);
						write.pBufferInfo = view;
						break;
					}
					case BindingKind::Samplers:
						for (const auto sampler: descriptors.samplers) {
							EXIT_IF(sampler == nullptr);
							m_descriptor_images[image_cursor++] = {sampler, nullptr, vk::ImageLayout::eUndefined};
						}
						break;
					case BindingKind::Count: EXIT("invalid descriptor binding kind");
				}
			}
			if (image_cursor != image_start) {
				write.pImageInfo = m_descriptor_images.data() + image_start;
			}
			m_descriptor_writes.push_back(write);
		}
		for (uint32_t i = 0; i < program.info.images.size(); ++i) {
			const auto& resource = program.info.images[i];
			const auto& texture = descriptors.images[i];
			EXIT_IF(texture.image_id &&
			        resource.mip_count != (texture.mip_views.empty() ? 1u : texture.mip_views.size()));
			const auto kind = *ShaderRecompiler::IR::DescriptorBindingForImage(resource);
			const auto start = image_starts[ShaderRecompiler::IR::ImageBindingIndex(kind)] +
			                   resource.descriptor_index;
			for (uint32_t mip = 0; mip < resource.mip_count; ++mip) {
				m_descriptor_images[start + mip] = MakeImageInfo(texture, mip);
			}
		}

		const auto shader_data_dwords = program.bindings.ShaderDataDwords();
		EXIT_IF(prepared->shader_data.size() != shader_data_dwords);
		if (program.bindings.UsesPushData()) {
			std::ranges::copy(prepared->shader_data,
			                  push_data.dwords.begin() + program.bindings.push_data_start_dword);
			has_push_data = true;
		}
	}

	if (has_push_data) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, push_stages, 0, sizeof(push_data),
		                        push_data.dwords.data());
	}

	if (!m_descriptor_writes.empty()) {
		EXIT_IF(pipeline.descriptor_set_layout == nullptr);
		if (pipeline.uses_push_descriptors) {
			vk_buffer.pushDescriptorSetKHR(pipeline_bind_point, pipeline.pipeline_layout, 0,
			                               static_cast<uint32_t>(m_descriptor_writes.size()),
			                               m_descriptor_writes.data());
		} else {
			// Consecutive draws often bind exactly the same resources. Descriptor sets are never
			// rewritten after their update, so the previous set of this layout can be bound again
			// while its pool is current and nothing it references can have been destroyed.
			static const bool reuse = FrameTiming::OptEnabled("desc_reuse");
			auto&             heap  = m_context.GetDescriptorHeap();
			auto&             key   = m_descriptor_key;
			key.clear();
			const auto update = [&key](const void* data, size_t size) {
				const auto* bytes = static_cast<const uint8_t*>(data);
				key.insert(key.end(), bytes, bytes + size);
			};
			for (const auto& write: m_descriptor_writes) {
				const std::array<uint32_t, 4> header {write.dstBinding, write.dstArrayElement,
				                                      write.descriptorCount,
				                                      static_cast<uint32_t>(write.descriptorType)};
				update(header.data(), sizeof(header));
				if (write.pBufferInfo != nullptr) {
					update(write.pBufferInfo,
					       sizeof(vk::DescriptorBufferInfo) * write.descriptorCount);
				}
				if (write.pImageInfo != nullptr) {
					update(write.pImageInfo,
					       sizeof(vk::DescriptorImageInfo) * write.descriptorCount);
				}
				if (write.pTexelBufferView != nullptr) {
					update(write.pTexelBufferView, sizeof(vk::BufferView) * write.descriptorCount);
				}
			}
			const auto hash      = XXH3_64bits(key.data(), key.size());
			const auto resources = g_descriptor_resource_generation.load(std::memory_order_relaxed);
			auto&      cached    = m_last_descriptor_sets[pipeline.descriptor_set_layout];
			const bool identical = cached.set != nullptr && cached.hash == hash;
			const bool reusable  = identical && cached.pool_generation == heap.PoolGeneration() &&
			                       cached.resource_generation == resources;
			if (identical) {
				FrameTiming::Add(FrameTiming::Counter::DescriptorSetsIdentical);
			}
			vk::DescriptorSet set = nullptr;
			if (reuse && reusable) {
				FrameTiming::Add(FrameTiming::Counter::DescriptorSetsReused);
				set = cached.set;
			} else {
				set = heap.Commit(pipeline.descriptor_set_layout);
				for (auto& write: m_descriptor_writes) {
					write.dstSet = set;
				}
				m_context.GetGraphics().device.updateDescriptorSets(
				    static_cast<uint32_t>(m_descriptor_writes.size()), m_descriptor_writes.data(),
				    0, nullptr);
				cached = {hash, heap.PoolGeneration(), resources, set};
			}
			vk_buffer.bindDescriptorSets(pipeline_bind_point, pipeline.pipeline_layout, 0, 1, &set,
			                             0, nullptr);
		}
	}
}

} // namespace Libs::Graphics
