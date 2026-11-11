#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/drsNoZero.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/guest_gpu/command_processor/cpPipe.h"
#include "graphics/guest_gpu/command_processor/cpPipeJob.h"
#include "graphics/host_gpu/renderer/drawRepeatDiag.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/host_gpu/renderer/meshIndirectRuntime.h"
#include "graphics/host_gpu/renderer/pvIndirect.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/s537Diag.h"
#include "graphics/shader/recompiler/s538Diag.h"
#include "common/memEvents.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <chrono>
#include <fmt/format.h>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <thread>
#include <xxhash.h>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.has_dynamic_vertex_fetch) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = vs_input_info.stage.module->info;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.module->info.info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.module->info.info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	// Rect lists are drawn without culling.
	vk::CullModeFlags cull_mode {};
	if (!Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		if (mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
	}
	dynamic_state.SetCullMode(cull_mode);
	dynamic_state.SetFrontFace(mode.face ? vk::FrontFace::eClockwise
	                                     : vk::FrontFace::eCounterClockwise);

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps({front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		                            {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}
}

// S537 DIAG (KYTY_DIAG_S537_DIR): the textures of the card pixel shader, one line per new (slot, image, state).
static void S537NoteTextures(RenderContext& context, const PreparedBindings& pixel) {
	static std::mutex                   mutex;
	static std::unordered_set<uint64_t> seen;
	static uint32_t                     lines = 0;
	const std::scoped_lock              lock(mutex);
	if (lines >= 2000) return;
	const auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
	                     std::chrono::steady_clock::now().time_since_epoch()).count();
	for (uint32_t slot = 0; slot < pixel.images.size(); slot++) {
		const auto& binding = pixel.images[slot];
		if (!binding.image_id) continue;
		const auto& image = context.GetTextureCache().GetImage(binding.image_id);
		const uint64_t key = (image.serial << 12) ^ (uint64_t {slot} << 4) ^ (image.IsGpuModified() ? 1u : 0u) ^
		                     (image.IsCpuDirty() ? 2u : 0u) ^ (image.usage.render_target ? 4u : 0u) ^ (image.diag.uploads << 20);
		const auto flip = FrameTiming::GuestFlips() + 1;
		static std::unordered_set<uint64_t> compared;
		static uint32_t                     compares = 0;
		if (slot == 2 && flip >= 4280 && flip <= 4470 && compares < 16 &&
		    compared.insert((image.serial << 16) ^ image.diag.uploads).second) {
			compares++;
			const auto tag = fmt::format("flip={} slot={}", flip, slot);
			context.GetTextureCache().S537CompareGuest(binding.image_id, tag.c_str());
		}
		static uint64_t last_periodic = 0;
		if (slot == 2 && flip >= 3990 && flip <= 4470 && flip % 10 == 0 && flip != last_periodic) {
			// S537 DIAG: every 10 flips, the guest mip 0 of the bound slot-2 image (hash, zero 16-byte blocks).
			last_periodic = flip;
			const auto m0 = image.info.mip_layout[0];
			std::vector<uint8_t> bytes(m0.size);
			uint64_t hash = 0, zero = 0;
			if (m0.size != 0 && m0.size <= (16u << 20) &&
			    LibKernel::Memory::TryReadBacking(image.info.data.address + m0.offset, bytes.data(), bytes.size())) {
				hash = XXH3_64bits(bytes.data(), bytes.size());
				static const uint8_t z[16] {};
				for (size_t b = 0; b + 16 <= bytes.size(); b += 16) zero += std::memcmp(bytes.data() + b, z, 16) == 0 ? 1 : 0;
			}
			Log::WriteToConsoleAndLog(fmt::format("[s537-per] flip={} addr=0x{:x} serial={} mip0_xxh={:016x} zero16={} uploads={} "
			                                      "upload_flip={} cpu_dirty={} writes={} page_writes={}\n",
			                                      flip, image.info.data.address, image.serial, hash, zero, image.diag.uploads,
			                                      image.diag.upload_flip, image.IsCpuDirty() ? 1 : 0, image.diag.writes,
			                                      image.diag.page_writes));
		}
		if (!seen.insert(key).second) continue;
		lines++;
		// Guest mip 0 now, to tell the same asset at another address from other content.
		uint64_t   mip0_hash = 0;
		const auto mip0      = image.info.mip_layout[0];
		if (mip0.size != 0 && mip0.size <= (16u << 20)) {
			std::vector<uint8_t> bytes(mip0.size);
			if (LibKernel::Memory::TryReadBacking(image.info.data.address + mip0.offset, bytes.data(), bytes.size())) {
				mip0_hash = XXH3_64bits(bytes.data(), bytes.size());
			}
		}
		Log::WriteToConsoleAndLog(fmt::format(
		    "[s537-tex] t_ms={} flip={} mip0_xxh={:016x} slot={} addr=0x{:x} serial={} {}x{}x{} gfmt={} vk={} view_addr=0x{:x} rt={} gpu_mod={} "
		    "cpu_dirty={} uploads={} upload_flip={} writes={} write_flip={}\n",
		    now, flip, mip0_hash, slot, image.info.data.address, image.serial, image.info.extent.width, image.info.extent.height,
		    image.info.extent.depth, static_cast<int>(image.info.guest_format), static_cast<int>(image.info.pixel_format),
		    binding.desc.info.data.address, image.usage.render_target ? 1 : 0, image.IsGpuModified() ? 1 : 0,
		    image.IsCpuDirty() ? 1 : 0, image.diag.uploads, image.diag.upload_flip, image.diag.writes, image.diag.write_flip));
	}
}

// S538 DIAG (KYTY_DIAG_S538=1 / KYTY_DIAG_S538_FORCE=1, s538Diag.h): the source of the card shader's slot-2 T#.
namespace S538Diag {

inline uint64_t T40(const std::array<uint32_t, 8>& d) {
	return ((uint64_t {d[0]} | (uint64_t {d[1]} << 32u)) & 0xffffffffffull) << 8u;
}

// Polled guest places: [0] the T#'s place, [1..3] small (pointer) loads of the walk.
struct Watch {
	std::atomic<uint64_t> addr[8] {};
	std::atomic<uint32_t> len[8] {};
	std::atomic<int64_t>  change_ns[8] {};
	std::atomic<uint64_t> change_flip[8] {};
	std::atomic<uint64_t> changes[8] {};
};
inline Watch& GetWatch() {
	static Watch watch;
	return watch;
}

inline void StartWatcher() {
	static std::once_flag once;
	std::call_once(once, [] {
		std::thread([] {
			auto&                                   w = GetWatch();
			std::array<std::array<uint32_t, 8>, 8> prev {};
			std::array<uint64_t, 8>                prev_addr {};
			uint32_t                               lines = 0;
			for (;;) {
				std::this_thread::sleep_for(std::chrono::microseconds(500));
				const auto flip = FrameTiming::GuestFlips() + 1;
				if (flip > 4520) return;
				for (int i = 0; i < 8; i++) {
					const auto a   = w.addr[i].load(std::memory_order_relaxed);
					const auto len = std::min<uint32_t>(w.len[i].load(std::memory_order_relaxed), 32);
					if (a == 0 || len == 0) continue;
					std::array<uint32_t, 8> now {};
					if (!LibKernel::Memory::TryReadBacking(a, now.data(), len)) continue;
					if (a != prev_addr[i]) {
						prev_addr[i] = a;
						prev[i]      = now;
						continue;
					}
					if (std::memcmp(now.data(), prev[i].data(), len) == 0) continue;
					const auto ns = FrameTiming::SteadyNowNs();
					w.change_ns[i].store(ns, std::memory_order_relaxed);
					w.change_flip[i].store(flip, std::memory_order_relaxed);
					w.changes[i].fetch_add(1, std::memory_order_relaxed);
					if (lines < 6000) {
						lines++;
						std::string text = fmt::format("[s538-w] t_ms={} flip={} slot={} addr=0x{:x} new=", ns / 1000000, flip, i, a);
						for (uint32_t k = 0; k < len / 4; k++) text += fmt::format("{:08x}{}", now[k], k + 1 < len / 4 ? "," : "");
						text += " old=";
						for (uint32_t k = 0; k < len / 4; k++) text += fmt::format("{:08x}{}", prev[i][k], k + 1 < len / 4 ? "," : "");
						Log::WriteToConsoleAndLog(text + "\n");
					}
					prev[i] = now;
				}
			}
		}).detach();
	});
}

// The guest mip 0 hash of a bound image (cached by serial and upload count, refreshed every 10 flips).
inline uint64_t BoundMip0(RenderContext& context, const TextureBinding& binding, uint64_t flip, uint64_t* off, uint64_t* size) {
	struct Entry {
		uint64_t hash = 0, flip = 0, uploads = 0, off = 0, size = 0;
	};
	static std::unordered_map<uint64_t, Entry> cache;
	const auto& image = context.GetTextureCache().GetImage(binding.image_id);
	auto&       e     = cache[image.serial];
	const auto  m0    = image.info.mip_layout[0];
	if (e.flip == 0 || e.uploads != image.diag.uploads || flip >= e.flip + 10) {
		e         = {};
		e.flip    = flip;
		e.uploads = image.diag.uploads;
		e.off     = m0.offset;
		e.size    = m0.size;
		if (m0.size != 0 && m0.size <= (16u << 20)) {
			std::vector<uint8_t> bytes(m0.size);
			if (LibKernel::Memory::TryReadBacking(image.info.data.address + m0.offset, bytes.data(), bytes.size())) {
				e.hash = XXH3_64bits(bytes.data(), bytes.size());
			}
		}
	}
	*off  = e.off;
	*size = e.size;
	return e.hash;
}

struct Good {
	bool                    have = false;
	std::array<uint32_t, 8> t {};
	uint64_t                mip0_off = 0, mip0_size = 0, checked_flip = 0;
	std::vector<std::pair<uint64_t, uint64_t>> layouts; // every (mip 0 offset, size) a good T# had
	bool                    valid = false;
	uint64_t                forced = 0, kept = 0, missing = 0;
};
inline Good& GetGood() {
	static Good good;
	return good;
}

inline uint64_t HashAt(uint64_t address, uint64_t size) {
	if (size == 0 || size > (16u << 20)) return 0;
	std::vector<uint8_t> bytes(size);
	if (!LibKernel::Memory::TryReadBacking(address, bytes.data(), bytes.size())) return 0;
	return XXH3_64bits(bytes.data(), bytes.size());
}

// After the bindings: learn the T# of the particle asset.
inline void Learn(RenderContext& context, const ShaderStageRuntime& ps, const PreparedBindings& pixel) {
	const auto& snap = *ps.resources;
	if (snap.images.size() < 3 || pixel.images.size() < 3 || !pixel.images[2].image_id) return;
	const auto flip = FrameTiming::GuestFlips() + 1;
	uint64_t   off = 0, size = 0;
	if (BoundMip0(context, pixel.images[2], flip, &off, &size) != ShaderRecompiler::S538::kGoodMip0) return;
	auto& g = GetGood();
	if (!g.have || g.t != snap.images[2].dwords) {
		Log::WriteToConsoleAndLog(fmt::format("[s538-good] flip={} t40=0x{:x} mip0_off=0x{:x} mip0_size=0x{:x} was_have={}\n", flip,
		                                      T40(snap.images[2].dwords), off, size, g.have ? 1 : 0));
	}
	g.have         = true;
	g.t            = snap.images[2].dwords;
	g.mip0_off     = off;
	g.mip0_size    = size;
	if (std::find(g.layouts.begin(), g.layouts.end(), std::make_pair(off, size)) == g.layouts.end()) g.layouts.emplace_back(off, size);
	g.valid        = true;
	g.checked_flip = flip;
}

// Before the bindings: the causal control (KYTY_DIAG_S538_FORCE=1).
inline void Force(const ShaderStageRuntime& ps) {
	auto& snap = const_cast<ShaderRecompiler::IR::ResourceSnapshot&>(*ps.resources);
	if (snap.images.size() < 3) return;
	auto&      g    = GetGood();
	const auto flip = FrameTiming::GuestFlips() + 1;
	auto&      cur  = snap.images[2].dwords;
	if (!g.have || cur == g.t || T40(cur) == 0) return;
	static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> seen; // base -> (hash, flip)
	auto& s = seen[T40(cur)];
	if (s.second == 0 || flip >= s.second + 10) {
		s = {0, flip};
		for (const auto& [o, n]: g.layouts) {
			s.first = HashAt(T40(cur) + o, n);
			if (s.first == ShaderRecompiler::S538::kGoodMip0) break;
		}
	}
	if (s.first == ShaderRecompiler::S538::kGoodMip0) {
		g.kept++;
		return;
	}
	if (flip >= g.checked_flip + 10) {
		g.valid        = HashAt(T40(g.t) + g.mip0_off, g.mip0_size) == ShaderRecompiler::S538::kGoodMip0;
		g.checked_flip = flip;
	}
	static uint64_t last_log = 0;
	if (!g.valid) {
		g.missing++;
	} else {
		cur = g.t;
		g.forced++;
	}
	if (flip != last_log && flip % 10 == 0) {
		last_log = flip;
		Log::WriteToConsoleAndLog(fmt::format("[s538-force] flip={} cur40=0x{:x} cur_hash={:016x} good40=0x{:x} valid={} forced={} kept={} missing={}\n",
		                                      flip, T40(snap.images[2].dwords), s.first, T40(g.t), g.valid ? 1 : 0, g.forced, g.kept,
		                                      g.missing));
	}
}

// A one-off scan of all committed guest memory for copies of a T# (its first two dwords at any dword offset); matches with
// how many of the 8 dwords agree, the range they lie in, and the first non-command-buffer ones go to watcher slots 4..7.
inline void StartScan(const char* what, std::array<uint32_t, 8> t, uint64_t flip, uint64_t skip_from, uint64_t skip_to, int first_slot) {
	std::thread([=] {
		const auto t0     = FrameTiming::SteadyNowNs();
		const auto ranges = LibKernel::Memory::DiagMappedRanges();
		uint64_t   scanned = 0, matches = 0;
		int        slot    = first_slot;
		const uint64_t key = uint64_t {t[0]} | (uint64_t {t[1]} << 32u);
		std::vector<uint8_t> chunk;
		for (const auto& [start, size]: ranges) {
			if (size > (uint64_t {16} << 30)) continue;
			for (uint64_t off = 0; off < size; off += (4u << 20)) {
				const uint64_t n = std::min<uint64_t>(4u << 20, size - off) + 28; // overlap for a T# across chunks
				const uint64_t want = std::min<uint64_t>(n, size - off);
				chunk.resize(want);
				if (!LibKernel::Memory::TryReadBacking(start + off, chunk.data(), want)) continue;
				scanned += want;
				for (uint64_t k = 0; k + 8 <= want; k += 4) {
					uint64_t v;
					std::memcpy(&v, chunk.data() + k, 8);
					if (v != key) continue;
					const uint64_t a = start + off + k;
					if (k >= (4u << 20)) continue; // counted in the next chunk
					matches++;
					uint32_t same = 0;
					std::array<uint32_t, 8> words {};
					if (LibKernel::Memory::TryReadBacking(a, words.data(), 32)) {
						for (int d = 0; d < 8; d++) same += words[d] == t[d] ? 1 : 0;
					}
					if (matches <= 48) {
						const auto desc = LibKernel::Memory::DiagDescribeRange(a, 4);
						Log::WriteToConsoleAndLog(fmt::format("[s538-scan] {} flip={} match=0x{:x} same={} {}\n", what, flip, a, same,
						                                      desc.empty() ? std::string("?") : desc[0]));
					}
					if (same >= 6 && (a & 0xf) == 0 && (a < skip_from || a >= skip_to) && slot < first_slot + 2 && slot < 8) {
						auto& w = GetWatch();
						w.addr[slot].store(a, std::memory_order_relaxed);
						w.len[slot].store(32, std::memory_order_relaxed);
						slot++;
					}
				}
			}
		}
		Log::WriteToConsoleAndLog(fmt::format("[s538-scan] {} done flip={} ranges={} scanned_mb={} matches={} ms={}\n", what, flip, ranges.size(),
		                                      scanned >> 20, matches, (FrameTiming::SteadyNowNs() - t0) / 1000000));
	}).detach();
}

// After the bindings: the T# the renderer binds against guest memory now (KYTY_DIAG_S538=1).
inline void Note(RenderContext& context, const ShaderStageRuntime& ps, const PreparedBindings& pixel, const HW::PixelShaderInfo& ps_regs) {
	const auto flip = FrameTiming::GuestFlips() + 1;
	if (flip < 3300 || flip > 4480) return;
	const auto& snap = *ps.resources;
	if (snap.images.size() < 3) return;
	StartWatcher();
	const auto& t = snap.images[2].dwords;
	// Merged load ranges of the walk.
	std::vector<std::pair<uint64_t, uint64_t>> ranges(snap.specialization_reads.begin(), snap.specialization_reads.end());
	std::sort(ranges.begin(), ranges.end());
	std::vector<std::pair<uint64_t, uint64_t>> merged;
	for (const auto& [a, n]: ranges) {
		if (n == 0) continue;
		if (!merged.empty() && a <= merged.back().first + merged.back().second) {
			merged.back().second = std::max(merged.back().second, a + n - merged.back().first);
		} else {
			merged.emplace_back(a, n);
		}
	}
	uint64_t   off = 0, size = 0;
	const auto mip0 = pixel.images.size() > 2 && pixel.images[2].image_id ? BoundMip0(context, pixel.images[2], flip, &off, &size) : 0;
	const uint64_t key = XXH3_64bits(t.data(), sizeof(t)) ^ (mip0 * 7);
	// One line per distinct (T#, user data, mip 0) per 10-flip bucket.
	static std::unordered_set<uint64_t> seen;
	static uint64_t                     bucket = 0;
	static uint32_t                     lines  = 0;
	static uint64_t                     draws_in_bucket = 0;
	if (flip / 10 != bucket) {
		if (bucket != 0) Log::WriteToConsoleAndLog(fmt::format("[s538-b] flips={}-{} card_draws={} keys={}\n", bucket * 10, bucket * 10 + 9, draws_in_bucket, seen.size()));
		bucket = flip / 10;
		seen.clear();
		draws_in_bucket = 0;
	}
	draws_in_bucket++;
	if (!seen.insert(key).second || lines >= 20000 || (flip < 3900 && seen.size() > 12)) return;
	lines++;
	const auto now = FrameTiming::SteadyNowNs();
	// The CP's source of PS user data 0..7 (the T# of slot 2 when the shader takes it from user data).
	std::string us;
	uint64_t    ud0_addr = 0;
	int         ud_mem_eq = 1, ud_val_eq = 1;
	for (uint32_t i = 0; i < 8 && i < snap.user_data.size(); i++) {
		const auto a = ps_regs.s538_src[i];
		uint32_t   m = 0;
		const bool r = a != 0 && LibKernel::Memory::TryReadBacking(a, &m, 4);
		if (!r || m != snap.user_data[i]) ud_mem_eq = 0;
		if (ps_regs.ps_user_sgpr.value[i] != snap.user_data[i]) ud_val_eq = 0;
		if (i == 0) ud0_addr = a;
		us += fmt::format("{}{}:0x{:x}:{:08x}", i ? "," : "", ps_regs.s538_kind[i], a, r ? m : 0xdeadbeefu);
	}
	// Copies of the card's T# in guest memory, once early in the card window and once late (and of the particle asset's T#).
	static int scans = 0;
	if (ud0_addr != 0 && ((scans == 0 && flip >= 3920) || (scans == 1 && flip >= 4050) || (scans == 2 && flip >= 4300))) {
		scans++;
		const uint64_t page = ud0_addr & ~uint64_t {0xfff};
		static const char* const names[] = {"", "card-pre", "card-mid", "card-late"};
		StartScan(names[scans], t, flip, page, page + 0x1000, scans == 1 ? 4 : (scans == 2 ? 6 : 8));
		const auto& g = GetGood();
		if (scans == 2 && g.have && g.t != t) StartScan("good-mid", g.t, flip, page, page + 0x1000, 8);
	}
	// A change of the card's T#: the memory events of the new and the old texture's range.
	static uint64_t last_t40 = 0;
	static uint32_t switches = 0;
	if (T40(t) != last_t40) {
		if (last_t40 != 0 && flip >= 3900 && switches < 40 && Common::MemEvents::Enabled()) {
			switches++;
			Log::WriteToConsoleAndLog(fmt::format("[s538-sw] flip={} from=0x{:x} to=0x{:x} mip0={:016x}\n", flip, last_t40, T40(t), mip0));
			for (const auto& line: Common::MemEvents::WatchLines(Common::MemEvents::Global(), T40(t), 0x60000, 12)) {
				Log::WriteToConsoleAndLog(fmt::format("[s538-sw] flip={} new 0x{:x} {}\n", flip, T40(t), line));
			}
			for (const auto& e: Common::MemEvents::Global().Query(last_t40, 0x60000, 6)) {
				Log::WriteToConsoleAndLog(fmt::format("[s538-sw] flip={} old 0x{:x} {}\n", flip, last_t40, Common::MemEvents::Format(e)));
			}
		}
		last_t40 = T40(t);
	}
	if (ud0_addr != 0) {
		auto& w0 = GetWatch();
		const bool direct = ps_regs.s538_kind[0] == 1;
		w0.addr[3].store(ud0_addr, std::memory_order_relaxed);
		w0.len[3].store(direct ? 32u : 4u, std::memory_order_relaxed);
	}
	static uint64_t described = 0;
	if (ud0_addr != 0 && described < 3 && (described == 0 || flip >= 3990)) {
		described = described == 0 ? 1 : described + 1;
		for (const auto& line: LibKernel::Memory::DiagDescribeRange(ud0_addr, 32)) {
			Log::WriteToConsoleAndLog(fmt::format("[s538-range] flip={} ud0=0x{:x} kind={} {}\n", flip, ud0_addr, ps_regs.s538_kind[0], line));
		}
	}
	std::string ud;
	for (size_t i = 0; i < snap.user_data.size() && i < 16; i++) ud += fmt::format("{:08x}{}", snap.user_data[i], i + 1 < snap.user_data.size() && i < 15 ? "," : "");
	std::string ts;
	for (size_t i = 0; i < 8; i++) ts += fmt::format("{:08x}{}", t[i], i < 7 ? "," : "");
	// Find the T# in the loaded ranges as they are now.
	uint64_t loc = 0, loc_range = 0;
	int      eq  = 0;
	std::string smalls;
	uint32_t    small_n = 0;
	for (const auto& [a, n]: merged) {
		if (n > (64u << 10)) continue;
		std::vector<uint32_t> words((n + 3) / 4);
		if (!LibKernel::Memory::TryReadBacking(a, words.data(), words.size() * 4)) continue;
		if (n <= 16 && small_n < 6) {
			smalls += fmt::format(" 0x{:x}:{}=", a, n);
			for (size_t k = 0; k < words.size(); k++) smalls += fmt::format("{:08x}{}", words[k], k + 1 < words.size() ? "," : "");
			auto& w = GetWatch();
			if (small_n < 2) {
				w.addr[1 + small_n].store(a, std::memory_order_relaxed);
				w.len[1 + small_n].store(static_cast<uint32_t>(std::min<uint64_t>(n, 32)), std::memory_order_relaxed);
			}
			small_n++;
		}
		if (loc != 0) continue;
		for (size_t k = 0; k + 8 <= words.size(); k++) {
			if (words[k] == t[0] && words[k + 1] == t[1]) {
				loc       = a + k * 4;
				loc_range = a;
				eq        = std::memcmp(&words[k], t.data(), 32) == 0 ? 1 : 2;
				break;
			}
		}
	}
	auto& w = GetWatch();
	if (loc != 0) {
		w.addr[0].store(loc, std::memory_order_relaxed);
		w.len[0].store(32, std::memory_order_relaxed);
	}
	bool       dirty = false, inflight = false, busy = false;
	const bool queried  = loc != 0 && LibKernel::Memory::QueryGuestFaultBytes(loc, 32, &dirty, &inflight, &busy);
	const bool gpu_dirt = loc != 0 && LibKernel::Memory::QueryShaderGpuDirty(loc, 32);
	std::string rs;
	for (size_t i = 0; i < merged.size() && i < 12; i++) rs += fmt::format("{}0x{:x}:{}", i ? "," : "", merged[i].first, merged[i].second);
	Log::WriteToConsoleAndLog(fmt::format(
	    "[s538-ts] t_ms={} flip={} path={} walk_age_us={} ud{}={} t2={} t40=0x{:x} bound_mip0={:016x} loc=0x{:x} range=0x{:x} eq={} "
	    "q={} dirty={} inflight={} busy={} gpu_dirty={} w_changes={} w_last_flip={} w_last_age_ms={} nranges={} ranges={} small={} "
	    "ud_src={} ud_mem_eq={} ud_val_eq={} udw_changes={} udw_last_flip={} udw_last_age_ms={}\n",
	    now / 1000000, flip, snap.s538_path, snap.s538_walk_ns != 0 ? (now - snap.s538_walk_ns) / 1000 : -1, snap.user_data.size(), ud, ts,
	    T40(t), mip0, loc, loc_range, eq, queried ? 1 : 0, dirty ? 1 : 0, inflight ? 1 : 0, busy ? 1 : 0, gpu_dirt ? 1 : 0,
	    w.changes[0].load(), w.change_flip[0].load(),
	    w.change_ns[0].load() != 0 ? (now - w.change_ns[0].load()) / 1000000 : -1, merged.size(), rs, smalls, us, ud_mem_eq, ud_val_eq,
	    w.changes[3].load(), w.change_flip[3].load(), w.change_ns[3].load() != 0 ? (now - w.change_ns[3].load()) / 1000000 : -1));
	if (Common::MemEvents::Enabled()) {
		const auto probe = [&](const char* what, uint64_t a, uint64_t n) {
			for (const auto& e: Common::MemEvents::Global().Query(a, n, 4)) {
				Log::WriteToConsoleAndLog(fmt::format("[s538-ev] flip={} {} 0x{:x} {}\n", flip, what, a, Common::MemEvents::Format(e)));
			}
		};
		if (loc != 0) probe("tsharp", loc, 32);
		if (ud0_addr != 0) probe("ud0", ud0_addr, 32);
		for (size_t i = 0; i < merged.size() && i < 12; i++) {
			if (merged[i].second <= 16) probe("small", merged[i].first, merged[i].second);
		}
	}
}

} // namespace S538Diag

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

struct DrawCallInfo {
	bool     indexed        = true;
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;

	[[nodiscard]] const char* Name() const { return indexed ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	KYTY_CP_SITE(DrawRenderState);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		const bool meta_clear =
		    metadata.kind == ImageMetadataKind::Htile &&
		    cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		// Apply pending clears to compressed stencil as well as depth.
		if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("depth attachment feedback loop is not supported by the host\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// vertex_offset is the first vertex of non-indexed draws.
struct DrawEmitInfo {
	int32_t          vertex_offset  = 0;
	uint32_t         first_instance = 0;
	DrawIndirectArgs indirect;
};

struct DrawIndexBufferSource {
	uint64_t    address            = 0;
	const void* host_data          = nullptr;
	uint64_t    size               = 0;
	uint32_t    guest_element_size = 0;
};

// The task shader's inputs for an indirect mesh draw: Params in gpu_mesh_indirect.task.
struct MeshTaskParams {
	vk::DeviceAddress records        = 0;
	uint64_t          index_address  = 0;
	uint32_t          stride         = 0;
	uint32_t          index_size     = 0;
	uint32_t          index_limit    = 0;
	uint32_t          primitive_size = 0;
	uint32_t          group_step     = 0;
	uint32_t          max_groups     = 0;
	uint32_t          max_instances  = 0;
	uint32_t          max_total      = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

static void CompareWithJob(const DrawRenderState& state, const CpPipe::GfpJob& job); // S479 shadow, below

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            DrawRenderState& state) {
	m_context.GetImageHeap().BeginDraw();
	// S479 (KYTY_CP_PIPE): the programs of a deferred draw were computed by a worker on the registers the draw was parsed with (S490: kept only when they are all
	// cached, the pages the worker read are not GPU-dirty and nothing was written into them since: CpPipe::JoinJob).
	CpPipe::GfpJob* const job      = CpPipe::CurrentJob();
	const bool            job_used = job != nullptr && CpPipe::JoinJob(*job);
	if (job_used && CpPipe::EnvMode() == CpPipe::Mode::On) {
		state.vertex_info[0] = job->out.vertex_info[0];
		state.ps_input_info  = job->out.pixel_info;
		state.ps_active      = job->out.ps_active;
		state.mrt_mask       = job->out.mrt_mask;
		FrameTiming::Add(FrameTiming::Counter::PipeUsedN);
	} else {
		if (job != nullptr && !job_used) FrameTiming::Add(FrameTiming::Counter::PipeFallbackN);
		CpPhases::Scope shaders(CpPhases::Phase::DrawShaders);
		m_context.GetPipelineCache().GetGraphicsPrograms(buffer, state);
		if (job_used) CompareWithJob(state, *job);
	}
	// Vertex programs beyond the host mesh limits or with unsupported call targets are not
	// compiled.
	if (state.vertex_info[0].stage.module == nullptr) {
		return false;
	}
	{
		CpPhases::Scope targets(CpPhases::Phase::DrawTargets);
		for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
			if ((state.mrt_mask & (1u << slot)) != 0) {
				// S511: a fast state has only slot 0 constructed. Resolve assigns the entry (`r = {}`),
				// which needs a live object. color_count == color_constructed means the next slot is new.
				if (state.color_count == state.color_constructed) {
					std::construct_at(&state.color_info[state.color_constructed++]);
				}
				ResolveRenderColorTarget(buffer, state.color_info[state.color_count], slot);
				if (state.color_info[state.color_count].image_id) {
					state.color_count++;
				}
			}
		}
		ResolveRenderDepthTarget(buffer, state.depth_info);
	}
	// TODO(PPSA09243): Vulkan passes take one sample count. Where PS5 attachments mix counts,
	// keep those matching the raster count instead of resolving multisampled shadows.
	const auto colors       = std::span {state.color_info, state.color_count};
	const auto pass_samples = state.depth_info.image_id ? state.depth_info.desc.info.samples
	                                                    : state.color_info[0].desc.info.samples;
	if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
		    return color.desc.info.samples != pass_samples;
	    })) {
		const auto raster =
		    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
		const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
			return color.desc.info.samples != raster;
		});
		state.color_count -= static_cast<uint32_t>(dropped.size());
		if (state.depth_info.image_id && state.depth_info.desc.info.samples != raster) {
			state.depth_info = {};
		}
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(
			    "Warning: dropping attachments whose sample count differs from the raster.\n");
		}
	}

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type =
	    source.guest_element_size == 4 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.indexed && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.indexed) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count,
				               static_cast<uint32_t>(emit.vertex_offset), emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.indexed) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count,
					               i + static_cast<uint32_t>(emit.vertex_offset),
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(ucfg.GetPrimType()));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto& depth = state.depth_info;
	auto& color = state.color_info[0];
	EXIT_NOT_IMPLEMENTED(
	    !m_context.GetGraphics().maintenance8_enabled ||
	    !Prospero::IsRectList(ucfg.GetPrimType()) || draw.indexed || draw.index_count != 3 ||
	    draw.instance_count != 1 || emit.vertex_offset != 0 || emit.first_instance != 0 ||
	    state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	    color.desc.info.samples != 1 || depth.desc.info.extent.width != color.Extent().width ||
	    depth.desc.info.extent.height != color.Extent().height ||
	    depth.desc.view_info.layer_count != 1 || color.desc.view_info.layer_count != 1 ||
	    hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	    state.ps_input_info.ps_depth_export_enable ||
	    state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		buffer.BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		buffer.EndRendering();
		const auto command = buffer.Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

// S472: the commands of the mesh prepare compute (the shared order is PvIndirect::RecordPrepare).
namespace {
struct MeshIndirectRecorder {
	CommandScheduler&                    scheduler;
	CommandBuffer&                       buffer;
	const RenderState&                   rendering;
	MeshIndirectRuntime&                 runtime;
	const MeshIndirectRuntime::Prepared& prepared;
	vk::CommandBuffer                    command;
	void EndPass() {
		const bool same = buffer.SamePass(rendering);
		scheduler.EndRendering();
		if (same) buffer.ArmReopenLoad();
	}
	void BindCompute() { runtime.Bind(command); }
	void Push() { runtime.Push(command, prepared); }
	void Dispatch() { runtime.Dispatch(command); }
	void BarrierToIndirect() { runtime.BarrierToIndirect(command); }
};
} // namespace

bool RenderExecutor::DrawReadsPending(std::span<PreparedBindings* const> stages, const ShaderVertexInputInfo& vertex_input,
                                    const DrawIndexBufferSource& index_source, const DrawEmitInfo& emit, uint64_t mesh_args_addr) const {
	std::array<DeferBarrier::Touch, 64>                  touches {};
	uint32_t                                             touch_n = 0;
	std::array<std::pair<uint64_t, uint64_t>, 40>        plain {};
	uint32_t                                             plain_n  = 0;
	bool                                                 uses_dma = false;
	bool                                                 overflow = false;
	const auto add_plain = [&](uint64_t begin, uint64_t size) {
		if (begin == 0 || size == 0) {
			return;
		}
		if (plain_n >= plain.size()) {
			overflow = true;
			return;
		}
		plain[plain_n++] = {begin, begin + size};
	};
	for (const auto* stage: stages) {
		const auto& info = stage->runtime->module->info;
		uses_dma |= info.info.uses_dma;
		for (uint32_t i = 0; i < info.info.buffers.size(); i++) {
			const auto& resource = info.info.buffers[i];
			if (resource.descriptor_index == UINT32_MAX || resource.descriptor_index >= stage->buffer_sources.size()) {
				continue;
			}
			const auto& source = stage->buffer_sources[resource.descriptor_index];
			if (source.address == 0 || source.size == 0) {
				continue;
			}
			if (touch_n >= touches.size()) {
				overflow = true;
				break;
			}
			touches[touch_n++] = {source.address, source.address + source.size,
			                      resource.atomic && !resource.stored && !resource.loaded};
		}
	}
	for (int i = 0; i < vertex_input.buffers_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
		add_plain(vertex_input.buffers[i].addr, VertexBufferDescriptorSize(i, vertex_input));
	}
	add_plain(index_source.address, index_source.size);
	if (emit.indirect.address != 0 && emit.indirect.count != 0) {
		const auto size = static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
		                  (index_source.guest_element_size != 0 ? sizeof(vk::DrawIndexedIndirectCommand) : sizeof(vk::DrawIndirectCommand));
		add_plain(emit.indirect.address, size);
	}
	add_plain(mesh_args_addr, mesh_args_addr != 0 ? sizeof(vk::DrawIndexedIndirectCommand) : 0);
	if (overflow) {
		return !m_pending_writes.empty();
	}
	return DeferBarrier::ReadsPending(m_pending_writes, std::span {touches.data(), touch_n}, uses_dma, std::span {plain.data(), plain_n});
}

bool RenderExecutor::NotePendingBufferWrites(std::span<PreparedBindings* const> stages) {
	std::array<DeferBarrier::Range, 64> slots {};
	uint32_t                            count = static_cast<uint32_t>(m_pending_writes.size());
	if (count > slots.size()) {
		return false;
	}
	for (uint32_t i = 0; i < count; i++) {
		slots[i] = m_pending_writes[i];
	}
	for (const auto* stage: stages) {
		const auto& info = stage->runtime->module->info.info;
		for (uint32_t i = 0; i < info.buffers.size(); i++) {
			const auto& resource = info.buffers[i];
			if (!resource.written || resource.descriptor_index == UINT32_MAX ||
			    resource.descriptor_index >= stage->buffer_sources.size()) {
				continue;
			}
			const auto& source = stage->buffer_sources[resource.descriptor_index];
			if (!DeferBarrier::NoteWrite(slots, count, source.address, source.address + source.size,
			                             resource.atomic && !resource.stored)) {
				return false;
			}
		}
	}
	m_pending_writes.assign(slots.begin(), slots.begin() + count);
	return true;
}

MeshIndirectRuntime& RenderExecutor::MeshIndirectRt() {
	if (!m_mesh_ind_rt) m_mesh_ind_rt = std::make_shared<MeshIndirectRuntime>(m_context.GetGraphics(), m_context.GetCommandScheduler());
	return *m_mesh_ind_rt;
}

void RenderExecutor::ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
                                         DrawRenderState& state, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source,
                                         MeshIndirectRequest*         mesh_gpu_request) {
	auto&      ucfg                     = buffer.GetUserConfig();
	const bool primitive_restart_enable = state.primitive_restart_enable;
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto  vertex_stages = state.VertexStages();
	const bool  mesh_active   = state.vertex_info[0].stage.module->info.stage == ShaderType::Mesh;
	const bool  mesh_task     = state.vertex_info[0].stage.module->mesh_task;
	const auto& mesh          = state.vertex_info[0].mesh;
	const auto& limits        = m_context.GetGraphics().mesh_shader_properties;
	uint32_t    mesh_groups   = 0;
	// S472 (KYTY_INDIRECT_GPU_MESH=1): the draw is recorded from the guest args; the groups come from a prepare compute.
	const bool mesh_gpu = mesh_active && mesh_gpu_request != nullptr;
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.indexed || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed);
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed, primitive_restart_enable);
		}
	}
	if (mesh_gpu) {
		EXIT_IF(mesh.fast_launch);
	} else if (mesh_task) {
		EXIT_NOT_IMPLEMENTED(emit.indirect.count > std::min(limits.maxTaskWorkGroupCount[0],
		                                                    limits.maxTaskWorkGroupTotalCount));
	} else if (mesh_active) {
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups = (primitives - 1u) / mesh.primitives_per_group + 1u;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}
	uint64_t mesh_args_addr = 0;
	if (mesh_gpu) {
		mesh_args_addr = mesh_gpu_request->args_addr;
		// The args are a GPU-read buffer of the cache; they are obtained before the registrations below so a later lookup finds the same one.
		(void)m_context.GetBufferCache().ObtainBuffer(mesh_args_addr, sizeof(vk::DrawIndexedIndirectCommand), false);
	}

	if (mesh_active && draw.indexed) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(
		    index_source.address, static_cast<uint64_t>(draw.index_count) *
		                              index_source.guest_element_size);
	}
	if (mesh_gpu) {
		// S472: the mesh shader reads the args itself (BDA), like the indices.
		(void)m_context.GetBufferCache().FindBuffer(mesh_args_addr, sizeof(vk::DrawIndexedIndirectCommand));
	}
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	const auto stages = [&] {
		CpPhases::Scope bind(CpPhases::Phase::DrawBind);
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
			descriptor_stages[stage_count++] = &bindings.vertex[i];
		}
		const bool s538_card = (ShaderRecompiler::S538::On() || ShaderRecompiler::S538::Force()) && state.ps_active &&
		                       state.ps_input_info.stage.module != nullptr &&
		                       state.ps_input_info.stage.module->info.shader_hash == ShaderRecompiler::S537::kCardHash;
		if (s538_card && ShaderRecompiler::S538::Force()) S538Diag::Force(state.ps_input_info.stage);
		if (state.ps_active) {
			PrepareBindings(state.ps_input_info.stage, bindings.pixel);
			descriptor_stages[stage_count++] = &bindings.pixel;
		}
		const auto stages = std::span {descriptor_stages.data(), stage_count};
		PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
		if (s538_card) {
			S538Diag::Learn(m_context, state.ps_input_info.stage, bindings.pixel);
			if (ShaderRecompiler::S538::On()) S538Diag::Note(m_context, state.ps_input_info.stage, bindings.pixel, buffer.GetShaders().GetPs());
		}
		return stages;
	}();
	if (ShaderRecompiler::S537::Dir() != nullptr && state.ps_active && state.ps_input_info.stage.module != nullptr &&
	    state.ps_input_info.stage.module->info.shader_hash == ShaderRecompiler::S537::kCardHash) {
		S537NoteTextures(m_context, bindings.pixel);
	}
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	if (!mesh_active) {
		vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		index_binding   = PrepareIndexBuffer(buffer, index_source);
	}
	std::pair<Buffer*, uint64_t> args_buffer {};
	if (emit.indirect.address != 0) {
		const auto size = static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
		                  (draw.indexed ? sizeof(vk::DrawIndexedIndirectCommand)
		                                : sizeof(vk::DrawIndirectCommand));
		args_buffer = m_context.GetBufferCache().ObtainBuffer(emit.indirect.address, size, false);
		EXIT_IF((args_buffer.second & 3u) != 0);
	}
	auto& pipeline = [&]() -> auto& {
		CpPhases::Scope pipe(CpPhases::Phase::DrawPipeline);
		return m_context.GetPipelineCache().GetGraphicsPipeline(buffer, state);
	}();
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering = [&] {
		CpPhases::Scope targets(CpPhases::Phase::DrawTargets);
		return AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
		                            feedback_aspects, stages);
	}();
	vk::DeviceAddress task_params = 0;
	if (mesh_task) {
		const MeshTaskParams params {
		    .records        = args_buffer.first->BufferDeviceAddress() + args_buffer.second,
		    .index_address  = index_source.address,
		    .stride         = emit.indirect.stride,
		    .index_size     = index_source.guest_element_size,
		    .index_limit    = draw.index_count,
		    .primitive_size = mesh.InputPrimitiveSize(),
		    .group_step     = mesh.primitives_per_group * mesh.InputPrimitiveStep(),
		    .max_groups     = limits.maxMeshWorkGroupCount[0],
		    .max_instances  = limits.maxMeshWorkGroupCount[1],
		    .max_total      = limits.maxMeshWorkGroupTotalCount,
		};
		auto& stream = m_context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		task_params  = stream.BufferDeviceAddress() + stream.Copy(&params, sizeof(params), 8);
	}

	// S327: the pass about to be needed differs from the open one (BeginRendering would close it below): split here, before this draw
	// records anything into the buffer (KYTY_EAGER_SUBMIT).
	m_context.GetCommandScheduler().MaybeEagerSplitAtPass(rendering);

	// S472: a ring slot for the mesh prepare compute (taken for the command buffer that records it).
	MeshIndirectRuntime::Prepared mesh_prepared;
	if (mesh_gpu) {
		const auto& mesh_stage = state.vertex_info[0];
		const auto [mesh_args_buffer, mesh_args_offset] =
		    m_context.GetBufferCache().ObtainBuffer(mesh_args_addr, sizeof(vk::DrawIndexedIndirectCommand), false);
		EXIT_IF(mesh_args_buffer == nullptr || (mesh_args_offset & 3u) != 0);
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		MeshIndirectRuntime::Job job;
		job.args_buffer       = mesh_args_buffer->Handle();
		job.args_offset       = mesh_args_offset;
		job.mesh              = {mesh_stage.mesh.InputPrimitiveSize(), mesh_stage.mesh.InputPrimitiveStep(), mesh_stage.mesh.primitives_per_group,
		                         mesh_stage.mesh.fast_launch};
		job.limits            = {limits.maxMeshWorkGroupCount[0], limits.maxMeshWorkGroupCount[1], limits.maxMeshWorkGroupTotalCount};
		job.index_buffer_size = mesh_gpu_request->index_buffer_size;
		job.native_fetch      = !mesh_stage.fetch_embedded;
		job.shadow            = false;
		mesh_prepared         = MeshIndirectRt().Reserve(job);
		if (!mesh_prepared.ok) {
			FrameTiming::Add(FrameTiming::Counter::MeshIndRingFullN);
			mesh_gpu_request->declined = true; // nothing of the draw is recorded: the caller reads the args and draws the old way
			return;
		}
	}

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	if (mesh_prepared.ok) {
		// The prepare compute runs outside a pass, before the draw: the draw takes its output as indirect args.
		MeshIndirectRecorder recorder {m_context.GetCommandScheduler(), buffer, rendering, MeshIndirectRt(), mesh_prepared, vk_buffer};
		(void)PvIndirect::RecordPrepare(recorder, m_context.GetCommandScheduler().RenderingOpen());
		FrameTiming::Add(FrameTiming::Counter::MeshIndGpuN);
	}
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	{
		CpPhases::Scope desc(CpPhases::Phase::DrawDesc);
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	}
	if (!mesh_active) {
		CommitIndexBuffer(vk_buffer, index_binding);
	}
	{
	CpPhases::Scope vkcmd(CpPhases::Phase::DrawVkcmd);
	using ShaderRecompiler::IR::PushData;
	const auto draw_dwords =
	    PushData::GraphicsDrawDwords(mesh_active, state.ps_input_info.object_primitive_id);
	if (draw_dwords != 0) {
		std::array<uint32_t, PushData::MeshDrawDwordCount> draw_data {};
		if (mesh_gpu) {
			// S472: the shader reads the counts and offsets from the guest args (Translate.cpp); see MeshIndirect::PushElementWord.
			draw_data = {draw.index_count,
			             static_cast<uint32_t>(mesh_args_addr),
			             static_cast<uint32_t>(mesh_args_addr >> 32u),
			             MeshIndirect::PushElementWord(index_source.guest_element_size, !state.vertex_info[0].fetch_embedded),
			             static_cast<uint32_t>(index_source.address),
			             static_cast<uint32_t>(index_source.address >> 32u)};
		} else if (mesh_task) {
			draw_data[0] = static_cast<uint32_t>(task_params);
			draw_data[1] = static_cast<uint32_t>(task_params >> 32u);
		} else if (mesh_active) {
			draw_data = {draw.index_count,
			             static_cast<uint32_t>(emit.vertex_offset),
			             emit.first_instance,
			             index_source.guest_element_size,
			             static_cast<uint32_t>(index_source.address),
			             static_cast<uint32_t>(index_source.address >> 32u)};
		} else {
			draw_data[0] = buffer.GetUserConfig().GetObjectId();
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        draw_dwords * sizeof(uint32_t), draw_data.data());
	}
	if (const auto planes = vertex_stages.back().user_clip_planes; planes != 0) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages,
		                        PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(planes) * 4u * sizeof(uint32_t),
		                        buffer.GetRegisters().GetUserClipPlanes().data());
	}

	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), state.depth_info,
	                           rendering);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);

	vk::PipelineStageFlags shader_write_stages = {};
	bool                   address_writes      = false;
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
		address_writes |= stage.stage.module->info.has_address_writes;
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (state.ps_active) {
		address_writes |= state.ps_input_info.stage.module->info.has_address_writes;
	}
	// S511: bounded buffer writes (the garden's BUFFER_ATOMIC) leave the barrier pending.
	// An address write has no range, so it keeps the immediate pair. Env off keeps both barriers.
	// A draw that does not itself write still flushes when it may read a pending range.
	const bool defer_on    = DeferBarrier::Enabled();
	const bool defer_write = defer_on && static_cast<bool>(shader_write_stages) && !address_writes;
	if (defer_on) {
		if (!buffer.HasPendingShaderWrites()) {
			m_pending_writes.clear();
		} else if (DrawReadsPending(stages, state.vertex_info[0], index_source, emit, mesh_args_addr)) {
			buffer.EndRendering();
			m_pending_writes.clear();
			FrameTiming::Add(FrameTiming::Counter::DeferWriteReadN);
		}
	}
	if (shader_write_stages && !defer_write) {
		// As for dispatches, buffer writes wait for earlier accesses to the memory.
		buffer.EndRendering();
		m_pending_writes.clear();
		ShaderWriteHazardBarrier(vk_buffer, shader_write_stages);
		if (defer_on) {
			FrameTiming::Add(FrameTiming::Counter::DeferWriteNowN);
		}
	}
	buffer.BeginRendering(rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	auto&         occlusion_queries = m_context.GetOcclusionQueries();
	const int32_t occlusion_query   = occlusion_queries.Begin(vk_buffer);
	if (mesh_gpu) {
		vk_buffer.drawMeshTasksIndirectEXT(mesh_prepared.out_buffer, mesh_prepared.out_offset, 1, 0);
	} else if (mesh_task) {
		vk_buffer.drawMeshTasksEXT(emit.indirect.count, 1, 1);
	} else if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else if (args_buffer.first == nullptr) {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	} else if (draw.indexed) {
		vk_buffer.drawIndexedIndirect(args_buffer.first->Handle(), args_buffer.second,
		                              emit.indirect.count, emit.indirect.stride);
	} else {
		vk_buffer.drawIndirect(args_buffer.first->Handle(), args_buffer.second,
		                       emit.indirect.count, emit.indirect.stride);
	}
	occlusion_queries.End(vk_buffer, occlusion_query);
	if (CpPhases::Enabled()) {
		DrawRepeat::Finger now;
		now.cb         = buffer.RecordingId();
		now.pipeline   = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkPipeline>(pipeline.pipeline)));
		now.generation = m_context.GetTextureCache().RegistrationGeneration();
		now.color_n    = state.color_count < 8 ? state.color_count : 8;
		for (uint32_t i = 0; i < now.color_n; i++) {
			now.colors[i] = state.color_info[i].image_id.index;
		}
		now.depth = state.depth_info.image_id.index;
		for (const auto* stage: stages) {
			for (const auto& image: stage->images) {
				now.images.push_back(image.image_id.index);
			}
			for (const auto sampler: stage->samplers) {
				now.samplers.push_back(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(static_cast<VkSampler>(sampler))));
			}
			for (const auto& source: stage->buffer_sources) {
				now.buffers.push_back(source.id.index);
			}
			now.shader_data.insert(now.shader_data.end(), stage->shader_data.begin(), stage->shader_data.end());
		}
		bool state_same = false;
		bool rt_same    = false;
		DrawRepeat::Observe(std::move(now), &state_same, &rt_same);
		if (state_same) {
			FrameTiming::Add(FrameTiming::Counter::DrawStateSameN);
		}
		if (rt_same) {
			FrameTiming::Add(FrameTiming::Counter::RtSameN);
		}
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);

	if (shader_write_stages && !defer_write) {
		buffer.EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	} else if (defer_write) {
		buffer.DeferShaderWriteBarrier(shader_write_stages);
		if (!NotePendingBufferWrites(stages)) {
			buffer.EndRendering();
			m_pending_writes.clear();
		}
		FrameTiming::Add(FrameTiming::Counter::DeferWriteDrawN);
	}
	}
}

// S479 (KYTY_CP_PIPE): the worker's GetGraphicsPrograms for a deferred draw, on the registers snapshotted with it. S490: upstream reads a draw's programs from the
// command buffer, so the worker runs the same function on the snapshot (PipelineCache::GetGraphicsProgramsOffThread: lookup only, resources in the job's own
// snapshot). Everything the walk read is logged as pages, which JoinJob checks against GPU-written bytes when the recorder asks for the result.
void CpPipe::RunGfpJob(CpPipe::GfpJob& job) {
	const int64_t t0 = FrameTiming::SteadyNowNs();
	CpPipe::WorkerScope worker_flag;
	ShaderRecompiler::IR::SetScratchPerThread(true);
	ShaderRecompiler::IR::SetSrtWalkerWorker(true);
	job.answered = false;
	job.out.fail = 0;
	job.tables.reads.clear();
	job.tables.words.clear();
	job.tables.unsafe = false;
	job.pages.clear();
	job.write_seq = Libs::LibKernel::Memory::BackingWriteSeq(); // before the first read
	ScoutTableLog::Active() = &job.tables;
	Libs::LibKernel::Memory::SetWorkerReadLog(&job.pages);
	bool answered = DrawHasValidVertexShader(*job.sh);
	if (!answered) job.out.fail = 4;
	answered = answered && job.cache->GetGraphicsProgramsOffThread(*job.ctx, *job.ucfg, *job.sh, job.out);
	Libs::LibKernel::Memory::SetWorkerReadLog(nullptr);
	ScoutTableLog::Active() = nullptr;
	// A guest load of the walker that faulted (a tracked page) was completed as zero: the walk is not the CP's.
	if (answered && job.tables.unsafe) {
		answered     = false;
		job.out.fail = 5;
	}
	if (answered && ShaderRecompiler::IR::SrtWalkerWorkerFaults() != 0) {
		answered     = false;
		job.out.fail = 7;
	}
	ShaderRecompiler::IR::SetSrtWalkerWorker(false);
	if (answered) {
		// The pages the compiled walker and the specialization reads touched (the backing reads logged theirs already).
		constexpr size_t kMaxPages = 4096;
		const auto       add_reads = [&](const ShaderRecompiler::IR::ResourceSnapshot& res) {
			for (const auto& [address, size]: res.specialization_reads) {
				if (!CpPipe::AppendPages(job.pages, address, size, kMaxPages)) answered = false;
			}
		};
		add_reads(job.out.vs_res);
		if (job.out.ps_active) add_reads(job.out.ps_res);
		std::sort(job.pages.begin(), job.pages.end());
		job.pages.erase(std::unique(job.pages.begin(), job.pages.end()), job.pages.end());
		if (job.pages.size() > kMaxPages) answered = false;
	}
	if (!answered) {
		const uint32_t code = job.out.fail == 0 ? 10u : (job.out.fail > 11u ? 11u : job.out.fail);
		FrameTiming::AddAlways(static_cast<FrameTiming::Counter>(static_cast<uint32_t>(FrameTiming::Counter::PipeFail1N) + code - 1u));
	}
	job.answered  = answered;
	job.worker_ns = FrameTiming::SteadyNowNs() - t0;
}

// S479 shadow: the result the CP computed (state) against the worker's (job), field by field of everything the draw is built from.
static void CompareWithJob(const DrawRenderState& state, const CpPipe::GfpJob& job) {
	using C = FrameTiming::Counter;
	FrameTiming::Add(C::PipeShadowN);
	const auto& out   = job.out;
	const bool  ne_vs = state.vertex_info[0].stage.module != out.vertex_info[0].stage.module;
	const bool  ne_ps = state.ps_active != out.ps_active || (state.ps_active && state.ps_input_info.stage.module != out.pixel_info.stage.module) ||
	                   state.mrt_mask != out.mrt_mask;
	const auto  same_res = [](const ShaderStageRuntime& a, const ShaderStageRuntime& b) {
        if (a.resources == nullptr || b.resources == nullptr) return a.resources == b.resources;
        // specialization_reads are not compared: the worker records every load of its walk (capture_reads), the CP only what the plan asks for
        return a.resources->flattened_srt == b.resources->flattened_srt && a.resources->buffers == b.resources->buffers && a.resources->images == b.resources->images &&
               a.resources->samplers == b.resources->samplers && a.resources->uniform_fill == b.resources->uniform_fill;
	};
	const bool ne_res = !same_res(state.vertex_info[0].stage, out.vertex_info[0].stage) || (state.ps_active && !same_res(state.ps_input_info.stage, out.pixel_info.stage));
	// The stage inputs: the static key decides the module, the rest of the fields the draw and its push data.
	const auto same_vs = [](const ShaderVertexInputInfo& x, const ShaderVertexInputInfo& y) {
		std::vector<uint32_t> kx, ky;
		BuildStageStaticKey(x, kx);
		BuildStageStaticKey(y, ky);
		if (kx != ky || x.logical_stage != y.logical_stage || x.fetch_attrib_reg != y.fetch_attrib_reg || x.fetch_buffer_reg != y.fetch_buffer_reg ||
		    x.buffers_num != y.buffers_num || x.fetch_external != y.fetch_external || x.fetch_embedded != y.fetch_embedded)
			return false;
		for (int i = 0; i < x.resources_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
			const auto &rx = x.resources_dst[i], &ry = y.resources_dst[i];
			if (std::memcmp(x.resources[i].fields, y.resources[i].fields, sizeof(x.resources[i].fields)) != 0 || rx.register_start != ry.register_start ||
			    rx.registers_num != ry.registers_num || rx.attr_id != ry.attr_id || rx.fetch_index != ry.fetch_index || rx.buffer_index != ry.buffer_index)
				return false;
		}
		for (int i = 0; i < x.buffers_num && i < ShaderVertexInputInfo::RES_MAX; i++) {
			if (x.buffers[i].addr != y.buffers[i].addr || x.buffers[i].stride != y.buffers[i].stride || x.buffers[i].num_records != y.buffers[i].num_records ||
			    x.buffers[i].fetch_index != y.buffers[i].fetch_index)
				return false;
		}
		return true;
	};
	const auto same_ps = [](const ShaderPixelInputInfo& x, const ShaderPixelInputInfo& y) {
		std::vector<uint32_t> kx, ky;
		BuildStageStaticKey(x, kx);
		BuildStageStaticKey(y, ky);
		return kx == ky && x.ps_sample_shading == y.ps_sample_shading && x.ps_execute_on_noop == y.ps_execute_on_noop && x.ps_single_sample == y.ps_single_sample &&
		       x.parameter_mode == y.parameter_mode && x.dual_source_blending == y.dual_source_blending && x.alpha_blend_source == y.alpha_blend_source;
	};
	const bool ne_info = !same_vs(state.vertex_info[0], out.vertex_info[0]) || (state.ps_active && !same_ps(state.ps_input_info, out.pixel_info));
	if (ne_vs || ne_ps || ne_res || ne_info) {
		FrameTiming::Add(C::PipeShadowNeN);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 16) {
			std::fprintf(stderr, "[cp-pipe] shadow ne: vs=%d ps=%d res=%d info=%d\n", ne_vs, ne_ps, ne_res, ne_info);
			std::fflush(stderr);
		}
	}
}

// Vulkan offsets for a prepared draw, or nullopt when its indirect records need the CPU:
// quad-list draws for counts, tessellation draws for the index/SGPR prolog and mesh draws without
// room for the task payload. Indirect records carry their own offsets.
static std::optional<DrawEmitInfo> ResolveDrawEmit(const HW::UserConfig&        ucfg,
                                                   const ShaderVertexInputInfo& vs,
                                                   DrawOffsetSource             source,
                                                   int32_t                      first_vertex,
                                                   uint32_t                     first_instance,
                                                   const DrawIndirectArgs&      indirect) {
	if ((indirect.address != 0 || source == DrawOffsetSource::IndirectArgs) &&
	    ucfg.GetIndexOffset() != 0) {
		static std::once_flag warning_once;
		std::call_once(warning_once, [&] {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU: indirect draw ignores GE_INDX_OFFSET=0x{:08x}; geometry may be incorrect.\n",
			    ucfg.GetIndexOffset()));
		});
	}
	if (indirect.address != 0) {
		if (!vs.indirect_draw ||
		    ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy ||
		    vs.indirect_offsets != indirect.offsets) {
			return std::nullopt;
		}
		return DrawEmitInfo {.indirect = indirect};
	}
	if (source == DrawOffsetSource::IndirectArgs) {
		// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
		return vs.has_dynamic_vertex_fetch ? DrawEmitInfo {first_vertex, first_instance} : DrawEmitInfo {};
	}
	const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), vs);
	return DrawEmitInfo {vertex_offset + first_vertex, instance_offset};
}

static bool PrimitiveRestartControlOn(const CommandBuffer& buffer) {
	if ((buffer.GetUserConfig().GetPrimitiveResetControl() & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: return true;
		default: return false;
	}
}

bool RenderExecutor::DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();
	FrameTiming::Add(FrameTiming::Counter::Draws);
	m_context.GetCommandScheduler().MaybeEagerSplit(); // KYTY_EAGER_SUBMIT
	// S472: the draw is declined before anything of it is recorded: the caller reads the args and draws the old way (counted by that call).
	const auto decline = [&] {
		args.indirect_gpu->declined = true;
		ResetBindings();
		FrameTiming::Add(FrameTiming::Counter::Draws, ~uint64_t {0});
		return true;
	};

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || (args.instance_count == 0 && args.indirect.address == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	alignas(DrawRenderState) unsigned char state_mem[sizeof(DrawRenderState)];
	DrawRenderState& state = *[&]() -> DrawRenderState* {
		if (DrsNoZero()) {
			return new (state_mem) DrawRenderState(DrawRenderState::Fast {}, args.indirect.offsets, args.indirect.address != 0);
		}
		auto* full                = new (state_mem) DrawRenderState();
		full->indirect_offsets    = args.indirect.offsets;
		full->indirect_draw       = args.indirect.address != 0;
		return full;
	}();
	if (!GetDrawTopology(ucfg, state.topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16: index_source.guest_element_size = 2; break;
		case Prospero::IndexType::kIndex32: index_source.guest_element_size = 4; break;
		case Prospero::IndexType::kIndex8: index_source.guest_element_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	if (args.indirect_gpu != nullptr) {
		// S472: a candidate's index range is the whole INDEX_BUFFER_SIZE: scanning it for a restart value would read indices the draw may never
		// fetch, so restart control on a strip / fan, a legacy quad list and a depth-to-color copy take the old path.
		if (PrimitiveRestartControlOn(buffer) || ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy ||
		    buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
			args.indirect_gpu->host_reason = IndirectGpu::Reason::PrimitiveType;
			return decline();
		}
		state.primitive_restart_enable = false;
	} else {
		state.primitive_restart_enable = ResolvePrimitiveRestart(buffer, index_source);
	}

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] =
			    state.primitive_restart_enable && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {true, args.index_count, args.instance_count};
	if (!PrepareDrawRenderState(buffer, draw, state)) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);
	MeshIndirectRequest* mesh_gpu_request = nullptr;
	if (args.indirect_gpu != nullptr) {
		// S472: only a mesh shader that can read the args itself takes the GPU path; any other host path declines.
		const auto& first_stage = state.vertex_info[0];
		const bool  mesh_gpu_ok = first_stage.stage.module->info.stage == ShaderType::Mesh && args.mesh_indirect != nullptr &&
		                         args.mesh_indirect->mode == MeshIndirect::Mode::On &&
		                         MeshIndirect::MeshCanGoGpu({first_stage.mesh.InputPrimitiveSize(), first_stage.mesh.InputPrimitiveStep(),
		                                                     first_stage.mesh.primitives_per_group, first_stage.mesh.fast_launch});
		if (!mesh_gpu_ok) {
			args.indirect_gpu->host_reason = IndirectGpu::Reason::Mesh;
			return decline();
		}
		mesh_gpu_request = args.mesh_indirect;
	}
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  args.base_vertex, args.first_instance, args.indirect);
	if (emit) {
		ExecutePreparedDraw(buffer, draw, state, *emit, index_source, mesh_gpu_request);
		if (mesh_gpu_request != nullptr && mesh_gpu_request->declined) {
			// The ring was full before anything of the draw was recorded: the caller reads the args and draws the old way.
			return decline();
		}
	}
	ResetBindings();
	return emit.has_value();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args) {
	FrameTiming::Add(FrameTiming::Counter::Draws);
	m_context.GetCommandScheduler().MaybeEagerSplit(); // KYTY_EAGER_SUBMIT
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (args.indirect.address == 0 && (args.vertex_count == 0 || args.instance_count == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {false, args.vertex_count, args.instance_count};

	alignas(DrawRenderState) unsigned char state_mem[sizeof(DrawRenderState)];
	DrawRenderState& state = *[&]() -> DrawRenderState* {
		if (DrsNoZero()) {
			return new (state_mem) DrawRenderState(DrawRenderState::Fast {}, args.indirect.offsets, args.indirect.address != 0);
		}
		auto* full             = new (state_mem) DrawRenderState();
		full->indirect_offsets = args.indirect.offsets;
		full->indirect_draw    = args.indirect.address != 0;
		return full;
	}();
	if (!GetDrawTopology(ucfg, state.topology)) {
		ResetBindings();
		return true;
	}
	if (!PrepareDrawRenderState(buffer, draw, state)) {
		ResetBindings();
		return true;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.module->info.param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  static_cast<int32_t>(args.first_vertex), args.first_instance,
	                                  args.indirect);
	if (emit) {
		ExecutePreparedDraw(buffer, draw, state, *emit, {});
	}
	ResetBindings();
	return emit.has_value();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, 1, true, true);
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

} // namespace Libs::Graphics
