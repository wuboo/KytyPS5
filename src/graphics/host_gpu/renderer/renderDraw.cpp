#include "graphics/host_gpu/renderer/renderDraw.h"
#include "graphics/host_gpu/renderer/syncCensus.h"

#include "common/assert.h"
#include "graphics/host_gpu/gpuLabelsVk.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/command_processor/cpScoutRuntime.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/appleDrop.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/diagDump.h"
#include "graphics/host_gpu/renderer/gfDiag.h"
#include "graphics/host_gpu/renderer/rtDump.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/perVertexPrototype.h"
#include "graphics/host_gpu/renderer/perVertexVsStoreRuntime.h"
#include "graphics/host_gpu/renderer/pvIndexBacking.h"
#include "graphics/host_gpu/renderer/pvIndirectRuntime.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/cpSampler.h"
#include "graphics/host_gpu/renderer/pvRangeGen.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

// S398 (KYTY_DIAG_SYNC=1): largest color target of the current flip (rt_max_w / rt_max_h); a change is logged.
static void NoteRenderTargetSize(uint32_t width, uint32_t height) {
	if (!SyncCensus::Enabled()) {
		return;
	}
	static uint64_t flip = ~0ull;
	static uint32_t logged_w = 0, logged_h = 0;
	static uint32_t logged_n = 0;
	const auto      now = FrameTiming::GuestFlips();
	if (now != flip) {
		flip = now;
		FrameTiming::Set(FrameTiming::Counter::RtMaxW, 0);
		FrameTiming::Set(FrameTiming::Counter::RtMaxH, 0);
	}
	FrameTiming::Max(FrameTiming::Counter::RtMaxW, width);
	FrameTiming::Max(FrameTiming::Counter::RtMaxH, height);
	const auto w = static_cast<uint32_t>(FrameTiming::Get(FrameTiming::Counter::RtMaxW));
	const auto h = static_cast<uint32_t>(FrameTiming::Get(FrameTiming::Counter::RtMaxH));
	if ((w != logged_w || h != logged_h) && logged_n < 200) {
		logged_w = w;
		logged_h = h;
		logged_n++;
		LOGF("[rt-size] flip=%" PRIu64 " max_color_target=%ux%u\n", now, w, h);
	}
}

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t                     index_offset,
                                                const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.fetch_embedded) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = *vs_input_info.stage.program;
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

static std::atomic<uint32_t> g_draw_state_log_count = 0;
static std::atomic<uint32_t> g_draw_input_log_count = 0;
static std::atomic<uint32_t> g_mrt_state_log_count  = 0;

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
	const auto& ps_resources   = ps_input_info.stage.program->info;
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
	    cc.mode, cc.op, bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend,
	    bc.color_comb_fcn, static_cast<int>(ps_resources.images.size()),
	    static_cast<int>(sampled_images),
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

// `registers` defaults to the live context; a batched prototype replay passes the snapshot taken
// when its draw was issued, since the guest may have rewritten the live registers since then.
static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering,
                                       const HW::Context* registers = nullptr) {
	KYTY_PROFILER_FUNCTION();

	const auto&        ctx = registers != nullptr ? *registers : buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto&        outputs = vs_input_info.stage.program->info.outputs;
	const bool         indexed_viewports =
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

	{
#if defined(__APPLE__)
		constexpr AppleDropHost host {.depth_clamp = false, .depth_bounds = false, .color_write_enable = false};
#else
		constexpr AppleDropHost host {};
#endif
		AppleDropDraw draw {};
		draw.z_clip_enabled    = ctx.GetClipControl().IsZClipEnabled();
		draw.depth_bounds_test = depth.depth_bounds_test_enable;
		draw.color_attachments = rendering.num_color_attachments;
		for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
			draw.color_attachments_bound += rendering.color_attachments[slot].image_view != nullptr ? 1u : 0u;
		}
		const uint32_t mask = AppleDropMask(draw, host);
		if ((mask & AppleDropDepthClamp) != 0u) FrameTiming::Add(FrameTiming::Counter::DropDepthClamp);
		if ((mask & AppleDropDepthBounds) != 0u) FrameTiming::Add(FrameTiming::Counter::DropDepthBounds);
		if ((mask & AppleDropColorWrite) != 0u) FrameTiming::Add(FrameTiming::Counter::DropColorWriteDyn);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	dynamic_state.SetColorWriteEnables(std::span {enable}.first(rendering.num_color_attachments));
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

static bool PixelShaderHasDepthOrCoverageSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

struct DrawRenderState {
	RenderDepthInfo                      depth_info;
	RenderColorInfo                      color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t                             color_count                              = 0;
	bool                                 ps_active                                = true;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	PipelineCache::GraphicsPrograms      programs;
};

struct DrawCallInfo {
	CommandBufferDebugOp debug_op       = CommandBufferDebugOp::DrawIndex;
	uint32_t             index_count    = 0;
	uint32_t             instance_count = 0;
	uint32_t             first_instance = 0;
	// S404 (KYTY_INDIRECT_GPU=1): the draw is recorded from the args in this buffer; index_count / instance_count are placeholders.
	uint64_t             indirect_addr   = 0; // guest address of the DrawIndexedIndirectArgs; 0 = a direct draw
	vk::Buffer           indirect_buffer = nullptr;
	vk::DeviceSize       indirect_offset = 0;

	[[nodiscard]] bool IsIndexed() const { return debug_op == CommandBufferDebugOp::DrawIndex; }
	[[nodiscard]] const char* Name() const { return IsIndexed() ? "DrawIndex" : "DrawIndexAuto"; }
};

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
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
		const auto owner = cache.m_slot_images.try_get(target.image_id);
		if (owner == nullptr || (!owner->registered && !owner->info.data.Empty()) ||
		    owner->binding.needs_rebind) {
			EXIT("color target changed after render-state discovery\n");
		}
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
		const auto extent           = target.Extent();
		NoteRenderTargetSize(extent.width, extent.height);
		state.width                 = std::min(state.width, extent.width);
		state.height                = std::min(state.height, extent.height);
		state.num_layers            = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view       = image_view;
		attachment.image_layout     = layout;
	}
	if (depth.image_id) {
		const auto owner = cache.m_slot_images.try_get(depth.image_id);
		if (owner == nullptr || !owner->registered || owner->binding.needs_rebind) {
			EXIT("depth target changed after render-state discovery\n");
		}
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
		const auto access               = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
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

static uint32_t DrawColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = ctx.HasObjectPrimitiveIdOutput() ? 1u << 7u : 0;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		const auto slot = ShaderPixelExportTarget(sh_regs.m_cbShaderMask, index);
		if (slot >= RENDER_COLOR_ATTACHMENTS_MAX) {
			break;
		}
		if (sh_regs.target_output_mode[index] != 0 &&
		    render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer,
                                                   uint32_t render_target_slice_offset) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, render_target_slice_offset, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

struct DrawEmitInfo {
	int32_t  vertex_offset  = 0;
	uint32_t first_vertex   = 0;
	uint32_t first_instance = 0;
};

struct DrawIndexBufferSource {
	uint64_t      address            = 0;
	const void*   host_data          = nullptr;
	uint64_t      size               = 0;
	vk::IndexType type               = vk::IndexType::eUint16;
	uint32_t      guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::DeviceSize size   = 0; // bytes of `buffer` (the descriptor of the VS-store draw must stay inside)
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

static void SetDrawDebugPhase(CommandBuffer& buffer, uint64_t submit_id, const DrawCallInfo& draw,
                              uint32_t phase) {
	buffer.SetDebugInfo(static_cast<uint32_t>(draw.debug_op), submit_id, phase, draw.index_count, 0,
	                    draw.instance_count, draw.first_instance);
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

static bool ResolvePrimitiveRestart(const CommandBuffer&         buffer,
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

static void RefreshShaders(CommandBuffer& buffer, const DrawCallInfo& draw,
                           uint32_t color_output_mask, DrawRenderState& state) {
	auto& ctx    = buffer.GetRegisters();
	auto& sh_ctx = buffer.GetShaders();

	const auto& vertex_shader_info = sh_ctx.GetVs();
	const auto& pixel_shader_info  = sh_ctx.GetPs();
	const auto& shader_regs        = ctx.GetShaderRegisters();

	state.programs      = {};
	state.ps_input_info = {};
	std::array<Prospero::ColorComponentMapping, RENDER_COLOR_ATTACHMENTS_MAX>
	    target_export_mapping {};
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		const auto& rt = ctx.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) != 0 && rt.base.addr != 0) {
			target_export_mapping[slot] =
			    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                 rt.info.channel_order)
			        .export_mapping;
		}
	}
	// Centroid equals center only when every active attachment has one sample.
	bool single_sample = true;
	for (uint32_t i = 0; i < state.color_count; ++i) {
		single_sample &= state.color_info[i].desc.info.samples == 1u;
	}
	if (state.depth_info.image_id) {
		single_sample &= state.depth_info.desc.info.samples == 1u;
	} else if (state.color_count == 0) {
		single_sample = render_sample_count(ctx.GetAaConfig().msaa_num_samples) == 1u;
	}
	auto& pipeline_cache = buffer.GetContext().GetPipelineCache();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "GetGraphicsPrograms");
	}
	CpScout::Prepared scout_prep;
	const auto        scout_kind = CpScout::Take(buffer, scout_prep);
	if (scout_kind == CpScout::TakeKind::Hit) {
		state.programs      = scout_prep.programs;
		state.vertex_info   = scout_prep.vertex_info;
		state.ps_input_info = scout_prep.pixel_info;
	} else {
		state.programs = pipeline_cache.GetGraphicsPrograms(
		    vertex_shader_info, pixel_shader_info, shader_regs, ctx, buffer.GetUserConfig(),
		    target_export_mapping, state.ps_active, state.vertex_info, state.ps_input_info,
		    single_sample);
		if (scout_kind == CpScout::TakeKind::Shadow) {
			FrameTiming::Add(FrameTiming::Counter::ScoutCmpN);
			const bool ne_vs = state.programs.vertex[0].id != scout_prep.programs.vertex[0].id;
			const bool ne_ps = state.programs.pixel.id != scout_prep.programs.pixel.id;
			// S471: the snapshots the descriptors are built from must be equal too (same programs with different resources would draw wrongly).
			const auto same_res = [](const ShaderStageRuntime& a, const ShaderStageRuntime& b) {
				if (a.resources == nullptr || b.resources == nullptr) return a.resources == b.resources;
				return a.resources->flattened_srt == b.resources->flattened_srt && a.resources->buffers == b.resources->buffers &&
				       a.resources->images == b.resources->images && a.resources->samplers == b.resources->samplers &&
				       a.resources->uniform_fill == b.resources->uniform_fill;
			};
			const bool ne_res = !same_res(state.vertex_info[0].stage, scout_prep.vertex_info[0].stage) ||
			                    (state.ps_active && !same_res(state.ps_input_info.stage, scout_prep.pixel_info.stage));
			if (ne_res) FrameTiming::Add(FrameTiming::Counter::ScoutNeResN);
			// The whole stage inputs (all but the snapshot pointer) must be the CP's own too: they decide the pipeline and the push data.
			const auto info_diff = [&](const auto& cp, const auto& scout, const char* name) {
				auto a = cp;
				auto b = scout;
				a.stage.resources = nullptr;
				b.stage.resources = nullptr;
				if (std::memcmp(&a, &b, sizeof(a)) == 0) return false;
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1) < 6) {
					const auto* pa = reinterpret_cast<const unsigned char*>(&a);
					const auto* pb = reinterpret_cast<const unsigned char*>(&b);
					std::fprintf(stderr, "[scout] info differs: %s size %zu offsets:", name, sizeof(a));
					int shown = 0;
					for (size_t i = 0; i < sizeof(a) && shown < 12; ++i) {
						if (pa[i] != pb[i]) {
							std::fprintf(stderr, " %zu(%02x/%02x)", i, pa[i], pb[i]);
							++shown;
						}
					}
					std::fprintf(stderr, "\n");
					std::fflush(stderr);
				}
				return true;
			};
			const bool ne_info = info_diff(state.vertex_info[0], scout_prep.vertex_info[0], "vs") ||
			                     (state.ps_active && info_diff(state.ps_input_info, scout_prep.pixel_info, "ps"));
			if (ne_info) FrameTiming::Add(FrameTiming::Counter::ScoutNeInfoN);
			if (ne_vs || ne_ps || ne_res || ne_info) FrameTiming::Add(FrameTiming::Counter::ScoutNeN);
			if (ne_vs) FrameTiming::Add(FrameTiming::Counter::ScoutNeVsN);
			if (ne_ps) FrameTiming::Add(FrameTiming::Counter::ScoutNePsN);
		}
	}
	if (!state.programs.vertex[0]) {
		// PrepareDrawRenderState() checks this right after calling us and skips the draw; bail out
		// here too since state.ps_input_info was left unpopulated and is not safe to dereference.
		return;
	}
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            uint32_t            render_target_slice_offset,
	                                        DrawRenderState& state) {
	KYTY_CP_SITE(DrawRenderState);
	const auto& shader_regs       = buffer.GetRegisters().GetShaderRegisters();
	const auto  color_output_mask = DrawColorOutputMask(buffer.GetRegisters());
	state.ps_active = buffer.GetShaders().GetPs().ps_regs.data_addr != 0 &&
	                  (color_output_mask != 0 ||
	                   PixelShaderHasDepthOrCoverageSideEffects(shader_regs));
	{
		CpPhases::Scope phase(CpPhases::Phase::DrawShaders);
		RefreshShaders(buffer, draw, color_output_mask, state);
	}
	if (!state.programs.vertex[0]) {
		// GetGraphicsPrograms() returns an empty GraphicsPrograms when it could not prepare the
		// vertex stage (currently: a mesh-shader draw on a host without mesh shader support).
		// Skip this draw the same way an empty render target does, below.
		return false;
	}
	uint32_t mrt_mask = 0;
	if (state.ps_active) {
		for (const auto& output: state.ps_input_info.stage.program->info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				mrt_mask |= 1u << output.location;
			}
		}
	}
	mrt_mask &= color_output_mask;
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderColorTarget");
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count],
			                         render_target_slice_offset, slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "ResolveRenderDepthTarget");
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);

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
	prepared.type = source.type;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		// Reserved to a whole word (the VS-store PS reads uint32 words); the pad is zeroed.
		const uint64_t reserved = (source.size + 3u) & ~uint64_t {3};
		auto [mapped, offset]   = stream.Map(reserved, 16);
		EXIT_IF(mapped == nullptr);
		std::memcpy(mapped, source.host_data, static_cast<size_t>(source.size));
		std::memset(mapped + source.size, 0, static_cast<size_t>(reserved - source.size));
		stream.Commit();
		prepared.offset = offset;
		prepared.buffer = stream.Handle();
		prepared.size   = stream.Size();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
		prepared.size   = buffer_ptr->Size();
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	for (uint32_t i = 0; i < prepared.count; i++) {
		EXIT_IF(prepared.buffers[i] == nullptr);
	}
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

	if (!draw.IsIndexed() && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

// S404: primitive restart is on for a strip / fan topology (ResolvePrimitiveRestart then looks at the indices).
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

static void EmitDrawPrimitives(const HW::UserConfig& ucfg, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	if (draw.indirect_addr != 0) {
		// S404: the host path of a qualified draw is one of the plain topologies (QuadListLegacy declined).
		EXIT_IF(!draw.IsIndexed() || draw.indirect_buffer == nullptr || ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy);
		vk_buffer.drawIndexedIndirect(draw.indirect_buffer, draw.indirect_offset, 1, sizeof(vk::DrawIndexedIndirectCommand));
		return;
	}
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
			if (draw.IsIndexed()) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count, emit.first_vertex,
				               emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.IsIndexed()) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count, i + emit.first_vertex,
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
	EXIT_NOT_IMPLEMENTED(!m_context.GetGraphics().maintenance8_enabled ||
	                     !Prospero::IsRectList(ucfg.GetPrimType()) ||
	                     draw.IsIndexed() || draw.index_count != 3 || draw.instance_count != 1 ||
	                     emit.first_vertex != 0 || emit.first_instance != 0 ||
	                     state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	                     color.desc.info.samples != 1 ||
	                     depth.desc.info.extent.width != color.Extent().width ||
	                     depth.desc.info.extent.height != color.Extent().height ||
	                     depth.desc.view_info.layer_count != 1 ||
	                     color.desc.view_info.layer_count != 1 ||
	                     hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	                     state.ps_input_info.ps_depth_export_enable ||
	                     state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		m_context.GetCommandScheduler().BeginRendering(rendering);
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
		m_context.GetCommandScheduler().EndRendering();
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
// dump_rt (control file): copies one image (mip/layer of a render target, or the depth aspect) into a
// host buffer; returns false when it cannot be copied (multisampled, no image, unsupported size).
static bool DumpRtCopy(RenderContext& context, ImageId id, uint32_t mip, uint32_t layer, vk::Extent2D extent,
                       bool depth, int mrt, std::vector<std::unique_ptr<Buffer>>& buffers,
                       std::vector<DiagDump::Target>& targets) {
	if (!id) return false;
	auto&      image = context.GetTextureCache().GetImage(id);
	const auto fmt   = depth ? DepthAspectTransferFormat(image.info.pixel_format) : image.info.pixel_format;
	const auto bytes = depth ? DepthAspectTransferBytes(image.info.pixel_format)
	                         : DiagDump::RtLayoutFor(static_cast<int>(fmt)).bytes;
	if (image.backing.image == nullptr || image.info.samples != 1 || bytes == 0 || extent.width == 0 || extent.height == 0) {
		return false;
	}
	const uint64_t size = static_cast<uint64_t>(extent.width) * extent.height * bytes;
	buffers.push_back(std::make_unique<Buffer>(context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::Download, 0,
	                                           vk::BufferUsageFlagBits::eTransferDst, size));
	vk::BufferImageCopy region {};
	region.imageSubresource.aspectMask     = depth ? vk::ImageAspectFlagBits::eDepth : vk::ImageAspectFlagBits::eColor;
	region.imageSubresource.mipLevel       = mip;
	region.imageSubresource.baseArrayLayer = layer;
	region.imageSubresource.layerCount     = 1;
	region.imageExtent                     = vk::Extent3D {extent.width, extent.height, 1};
	image.Download(std::span {&region, 1}, buffers.back()->Handle(), 0, size);
	DiagDump::Target target;
	target.addr      = image.info.data.address;
	target.mrt       = mrt;
	target.vk_format = static_cast<int>(fmt);
	target.width     = extent.width;
	target.height    = extent.height;
	target.data.resize(size);
	targets.push_back(std::move(target));
	return true;
}

// dump_rt: after a draw of a listed pixel shader reads every colour target and the depth target back
// (one flush + wait per call, diagnostics only) and hands the copies to DiagDump. Needs no open render pass.
static void DumpRenderTargets(RenderContext& context, uint64_t ps, const RenderColorInfo* colors, uint32_t color_count,
                              const RenderDepthInfo& depth) {
	std::vector<std::unique_ptr<Buffer>> buffers;
	std::vector<DiagDump::Target>        targets;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto& color = colors[i];
		DumpRtCopy(context, color.image_id, color.guest_mip_level, color.guest_array_layer, color.Extent(), false,
		           static_cast<int>(color.target_slot), buffers, targets);
	}
	if (depth.image_id) {
		DumpRtCopy(context, depth.image_id, 0, 0,
		           {std::max(depth.desc.info.extent.width, 1u), std::max(depth.desc.info.extent.height, 1u)}, true, -1, buffers,
		           targets);
	}
	if (targets.empty()) return;
	context.GetCommandScheduler().FlushAndWait();
	for (size_t i = 0; i < targets.size(); i++) {
		buffers[i]->Invalidate(0, buffers[i]->Size());
		if (buffers[i]->Mapped().size() >= targets[i].data.size()) {
			std::memcpy(targets[i].data.data(), buffers[i]->Mapped().data(), targets[i].data.size());
			DiagDump::Store(ps, std::move(targets[i]));
		}
	}
}

// dump_tex: decodes mip 0 / layer 0 of a BC7 texture from guest memory (detiled with the tile list the texture cache uploads
// with) and from the host image the draw sampled, and hands both to DiagDump::WriteDumpTex.
static void DumpTexOne(RenderContext& context, const DiagDump::DumpTexRequest& request) {
	DiagDump::DumpTexResult result;
	result.info = request.info;
	char note[640];
	auto& cache = context.GetTextureCache();
	auto& image = cache.GetImage(request.image);
	const auto& info = image.info;
	if (info.guest_format != Prospero::BufferFormat::kBc7UNorm && info.guest_format != Prospero::BufferFormat::kBc7Srgb) {
		std::snprintf(note, sizeof(note), "[dump-tex] not a BC7 texture (guest format %u): nothing to compare",
		              static_cast<uint32_t>(info.guest_format));
		result.notes.push_back(note);
		DiagDump::WriteDumpTex(result);
		return;
	}

	// (a) guest memory
	std::vector<GpuTileInfo> tiles;
	if (!cache.DiagUploadTiles(request.image, tiles)) {
		result.notes.push_back("[dump-tex] guest side unavailable: the image has no tiled upload (linear, depth or multisampled)");
	} else {
		const auto&     tile = tiles[0];
		TileBlockLayout block {};
		if (!TileGetBlockLayout(tile.family, tile.bytes_per_element, block) || tile.bytes_per_element != 16 ||
		    (tile.family != TileBlockFamily::Standard256B && tile.family != TileBlockFamily::Standard4KB &&
		     tile.family != TileBlockFamily::Standard64KB && tile.family != TileBlockFamily::Prt64KB)) {
			std::snprintf(note, sizeof(note), "[dump-tex] guest side unavailable: block family %u elem %u B not handled",
			              static_cast<uint32_t>(tile.family), tile.bytes_per_element);
			result.notes.push_back(note);
		} else {
			DiagDump::BcTileSpec spec;
			spec.block_width  = block.block_width;
			spec.block_height = block.block_height;
			spec.block_bytes  = block.block_size;
			spec.elem_bytes   = tile.bytes_per_element;
			spec.tiled_offset = tile.tiled_offset;
			spec.tiled_size   = tile.tiled_size;
			spec.width        = tile.width;
			spec.height       = tile.height;
			spec.tiled_width  = tile.tiled_width != 0 ? tile.tiled_width : tile.pitch;
			spec.tiled_height = tile.tiled_height != 0 ? tile.tiled_height : tile.height;
			spec.tail         = tile.tail;
			spec.tail_x       = tile.tail_x;
			spec.tail_y       = tile.tail_y;
			const auto offset = [&](uint32_t x, uint32_t y, uint32_t& within) { return TileGetBlockOffset(block, x, y, 0, within); };
			const auto* guest = reinterpret_cast<const uint8_t*>(info.data.address);
			const auto  missing = DiagDump::BcDetile(guest, info.data.size, spec, offset, result.guest);
			result.blocks_w = tile.width;
			result.blocks_h = tile.height;
			std::snprintf(note, sizeof(note),
			              "[dump-tex] guest layout family=%u block=%ux%u elements (%u B) mip0 offset=%" PRIu64 " size=%" PRIu64
			              " elements=%ux%u padded=%ux%u tail=%d image_size=%" PRIu64 " unreadable_elements=%" PRIu64,
			              static_cast<uint32_t>(tile.family), block.block_width, block.block_height, block.block_size, tile.tiled_offset,
			              tile.tiled_size, tile.width, tile.height, spec.tiled_width, spec.tiled_height, tile.tail ? 1 : 0, info.data.size,
			              missing);
			result.notes.push_back(note);

			// S125: every 64 KB tile of the image_size (mip tail + mips + mip 0) detiled alone, for the mosaic.
			if (!spec.tail && block.block_size != 0 && block.block_width * block.block_height * tile.bytes_per_element == block.block_size) {
				std::vector<std::vector<uint8_t>> pieces;
				for (uint64_t k = 0; (k + 1) * block.block_size <= info.data.size; k++) {
					pieces.emplace_back();
					DiagDump::BcDetile(guest, info.data.size, DiagDump::SingleTileSpec(spec, k), offset, pieces.back());
				}
				DiagDump::WriteTileMosaic(request.info, pieces, block.block_width, block.block_height);
			}
		}
	}

	// (b) host image: the BC blocks of mip 0
	const auto host_format = image.backing.format;
	if (image.backing.image == nullptr || image.info.samples != 1 ||
	    (host_format != vk::Format::eBc7UnormBlock && host_format != vk::Format::eBc7SrgbBlock)) {
		std::snprintf(note, sizeof(note), "[dump-tex] host side unavailable: backing format %d (image %s)", static_cast<int>(host_format),
		              image.backing.image == nullptr ? "not backed" : "backed");
		result.notes.push_back(note);
	} else {
		const uint32_t bw = (info.extent.width + 3u) / 4u, bh = (info.extent.height + 3u) / 4u;
		if (result.blocks_w != 0 && (bw != result.blocks_w || bh != result.blocks_h)) {
			std::snprintf(note, sizeof(note), "[dump-tex] host blocks %ux%u differ from the guest layout %ux%u: host side skipped", bw, bh,
			              result.blocks_w, result.blocks_h);
			result.notes.push_back(note);
		} else {
			const uint64_t size = static_cast<uint64_t>(bw) * bh * 16;
			Buffer         buffer(context.GetGraphics(), context.GetCommandScheduler(), MemoryUsage::Download, 0,
			                      vk::BufferUsageFlagBits::eTransferDst, size);
			vk::BufferImageCopy region {};
			region.imageSubresource.aspectMask     = vk::ImageAspectFlagBits::eColor;
			region.imageSubresource.mipLevel       = 0;
			region.imageSubresource.baseArrayLayer = 0;
			region.imageSubresource.layerCount     = 1;
			region.imageExtent                     = vk::Extent3D {info.extent.width, info.extent.height, 1};
			image.Download(std::span {&region, 1}, buffer.Handle(), 0, size);
			context.GetCommandScheduler().FlushAndWait();
			buffer.Invalidate(0, buffer.Size());
			if (buffer.Mapped().size() >= size) {
				result.host.assign(buffer.Mapped().data(), buffer.Mapped().data() + size);
				result.blocks_w = bw;
				result.blocks_h = bh;
			}
			std::snprintf(note, sizeof(note), "[dump-tex] host image=0x%" PRIx64 " backing_format=%d extent=%ux%u levels=%u read=%" PRIu64 " B",
			              reinterpret_cast<uint64_t>(static_cast<VkImage>(image.backing.image)), static_cast<int>(host_format),
			              info.extent.width, info.extent.height, info.resources.levels, result.host.size());
			result.notes.push_back(note);
		}
	}

	// (d) when the host copy was uploaded and what the CPU-write tracker knows since
	const char* uma = "0"; // UMA is not part of this build
	std::snprintf(note, sizeof(note),
	              "[dump-tex] state cur_flip=%" PRIu64 " host_upload_flip=%" PRIu64 " uploads=%" PRIu64 " cpu_dirty=%d maybe_dirty=%d gpu_modified=%d "
	              "buffer_modified=%d tracked=%d write_flip=%" PRIu64 " writes=%" PRIu64 " page_write_flip=%" PRIu64 " page_writes=%" PRIu64
	              " uma=%s%s",
	              FrameTiming::GuestFlips() + 1, image.diag.upload_flip, image.diag.uploads, image.IsDefinitelyCpuDirty() ? 1 : 0,
	              image.IsMaybeCpuDirty() ? 1 : 0, image.IsGpuModified() ? 1 : 0, image.IsBufferModified() ? 1 : 0, image.IsTracked() ? 1 : 0,
	              image.diag.write_flip, image.diag.writes, image.diag.page_flip, image.diag.page_writes, uma != nullptr ? uma : "0",
	              image.diag.uploads == 0 ? " (no upload recorded for this host image: created or refreshed by another path)" : "");
	result.notes.push_back(note);
	DiagDump::WriteDumpTex(result);
}

static void DumpTexAfterDraw(RenderContext& context) {
	auto requests = DiagDump::TakeDumpTexRequests();
	if (requests.empty()) return;
	context.GetCommandScheduler().EndRendering();
	for (const auto& request: requests) DumpTexOne(context, request);
}

// diag_tex: reads back mip 0 / layer 0 of every image queued by ResolveTexture (the host image the shader sampled).
static void DiagTexAfterDraw(RenderContext& context) {
	DumpTexAfterDraw(context);
	auto requests = DiagDump::TakeTexRequests();
	if (requests.empty()) return;
	context.GetCommandScheduler().EndRendering();
	std::vector<std::unique_ptr<Buffer>> buffers;
	std::vector<DiagDump::Target>        targets;
	std::vector<size_t>                  which;
	for (size_t i = 0; i < requests.size(); i++) {
		// The draw that resolved the image has just been recorded, so the slot is still alive.
		const auto& image  = context.GetTextureCache().GetImage(requests[i].image);
		const auto  extent = image.info.extent;
		const auto  before = targets.size();
		DumpRtCopy(context, requests[i].image, 0, 0, {std::max(extent.width, 1u), std::max(extent.height, 1u)},
		           image.info.IsDepth(), -3, buffers, targets);
		if (targets.size() != before) which.push_back(i);
	}
	if (!targets.empty()) context.GetCommandScheduler().FlushAndWait();
	std::vector<bool> stored(requests.size(), false);
	for (size_t i = 0; i < targets.size(); i++) {
		buffers[i]->Invalidate(0, buffers[i]->Size());
		if (buffers[i]->Mapped().size() >= targets[i].data.size()) {
			std::memcpy(targets[i].data.data(), buffers[i]->Mapped().data(), targets[i].data.size());
			DiagDump::StoreTex(requests[which[i]].info, std::move(targets[i]));
			stored[which[i]] = true;
		}
	}
	for (size_t i = 0; i < requests.size(); i++) {
		if (!stored[i]) DiagDump::StoreTex(requests[i].info, {});
	}
}

// diag_tex ...:all: closes the draw (or batch) for the slots ResolveTexture queued; the first colour target's address
// (else the depth target's) tells which pass the draw belongs to.
static void DiagTexDrawsDone(uint64_t index_count, const RenderColorInfo* colors, uint32_t color_count,
                             const RenderDepthInfo& depth) {
	const uint64_t rt = color_count != 0 ? colors[0].desc.info.data.address : depth.desc.info.data.address;
	DiagDump::TexDrawsDone(index_count, rt);
}

// Called after a draw (or a batch of prototype draws) of `ps_hash`: dumps when the control file asks.
static void DiagDumpAfterDraw(RenderContext& context, bool ps_active, uint64_t ps_hash, const RenderColorInfo* colors,
                              uint32_t color_count, const RenderDepthInfo& depth) {
	if (!ps_active || !DiagDump::Wants(ps_hash)) return;
	context.GetCommandScheduler().EndRendering();
	DumpRenderTargets(context, ps_hash, colors, color_count, depth);
}

#include "graphics/host_gpu/renderer/perVertexPrototypeDraw.inc"

// ---- S150: KYTY_PV_VSSTORE=1 ----
static void CountPvFallback(PvVsStoreFallback fallback) {
	using C = FrameTiming::Counter;
	switch (fallback) {
		case PvVsStoreFallback::None: break;
		case PvVsStoreFallback::Topology: FrameTiming::Add(C::PvFallbackTopology); break;
		case PvVsStoreFallback::Range: FrameTiming::Add(C::PvFallbackRange); break;
		case PvVsStoreFallback::Instanced: FrameTiming::Add(C::PvFallbackInstanced); break;
		case PvVsStoreFallback::Ring: FrameTiming::Add(C::PvFallbackRing); break;
		case PvVsStoreFallback::Param31: FrameTiming::Add(C::PvFallbackParam31); break;
		case PvVsStoreFallback::Interface: FrameTiming::Add(C::PvFallbackInterface); break;
		case PvVsStoreFallback::IndexTail: FrameTiming::Add(C::PvFallbackIndexTail); break;
	}
}

bool RenderExecutor::TryPvVsStoreDraw(const DrawCallInfo& draw, const DrawRenderState& state,
                                      vk::PrimitiveTopology topology, bool primitive_restart_enable,
                                      const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
                                      PvVsStoreDrawSetup& setup, const PvIndirectDraw* pv_indirect, PvIndirect::Decline* why) {
	// S408: a draw of mode On is declined to the CPU path, which counts its own fallback; here only the reason is reported.
	const bool pvi_on = pv_indirect != nullptr && pv_indirect->mode == PvIndirect::Mode::On;
	const auto fallback = [&](PvVsStoreFallback value, PvIndirect::Decline reason) {
		if (pvi_on && why != nullptr) {
			*why = reason;
		} else {
			CountPvFallback(value);
		}
	};
	if (!state.ps_active || state.vertex_info[0].logical_stage != ShaderType::Vertex ||
	    state.vertex_info[0].stage.program->stage == ShaderType::Mesh)
		return false;
	auto&       graphics = m_context.GetGraphics();
	const auto* programs = GetPvVsStorePrograms(graphics, state.programs);
	if (programs == nullptr) return false; // not a per-vertex pair the store knows
	if (programs->verdict != PvVsStoreInterfaceVerdict::Use) {
		fallback(ClassifyPvVsStoreFallback(programs->verdict), PvIndirect::Decline::Other);
		return false;
	}
	LogPvPairFirstUse(*programs, static_cast<uint32_t>(topology), draw.IsIndexed(), draw.instance_count);
	const uint32_t element_size =
	    index_source.host_data != nullptr ? sizeof(uint16_t) : index_source.guest_element_size;

	PvVsStoreRequest request;
	request.enabled            = true;
	request.triangle_list      = topology == vk::PrimitiveTopology::eTriangleList;
	request.indexed            = draw.IsIndexed();
	request.primitive_restart  = primitive_restart_enable;
	request.index_size         = draw.IsIndexed() ? element_size : 0;
	request.index_count        = draw.index_count;
	request.instance_count     = draw.instance_count;
	request.record_stride_vec4 = programs->stride_vec4;
	request.instanced_allowed  = PerVertexVsStoreInstanced();
	const bool scannable = request.triangle_list && request.indexed && !primitive_restart_enable &&
	                       (element_size == 2 || element_size == 4) && index_source.size != 0;
	bool range_reused = false; // S413 (KYTY_PV_IND_RANGE=1)
	if (scannable) {
		const void* idx_bytes = index_source.host_data != nullptr
		                            ? index_source.host_data
		                            : reinterpret_cast<const void*>(index_source.address);
		// The scan of a big index buffer is the expensive part; meshes recur, so it is memoized by the
		// content hash exactly like the prototype does.
		constexpr uint32_t kNoDropLimit = 0x7fffffffu; // only negative vertex numbers count as dropped
		// S406 (KYTY_PV_RANGE_GEN=0|1|shadow): the page generation of the index range can stand in for the hash of its bytes.
		const auto             gen_mode = PvRangeGen::EnvMode();
		const PvRangeGen::Key  gen_key {index_source.address, index_source.size, element_size, draw.index_count,
		                                emit.vertex_offset, kNoDropLimit};
		PvRangeGen::Probe      gen_probe;
		const PvRangeGen::Entry* gen_entry = nullptr;
		auto                   gen_outcome = PvRangeGen::Outcome::IneligibleHost;
		if (gen_mode != PvRangeGen::Mode::Off) {
			gen_probe.host_data = index_source.host_data != nullptr;
			if (!gen_probe.host_data) {
				const auto generation = m_context.GetBufferCache().QueryPageGeneration(index_source.address, index_source.size);
				gen_probe.registered  = generation.registered;
				gen_probe.clean       = generation.clean;
				gen_probe.max_stamp   = generation.max_stamp;
				gen_entry             = m_pv_range_gen.Find(gen_key);
			}
			gen_outcome = PvRangeGen::Decide(gen_probe, gen_entry);
			using C = FrameTiming::Counter;
			switch (gen_outcome) {
				case PvRangeGen::Outcome::IneligibleHost: FrameTiming::Add(C::PvGenIneligibleHostN); break;
				case PvRangeGen::Outcome::IneligibleUnregistered: FrameTiming::Add(C::PvGenIneligibleUnregN); break;
				case PvRangeGen::Outcome::IneligibleDirty: FrameTiming::Add(C::PvGenIneligibleDirtyN); break;
				case PvRangeGen::Outcome::Miss: FrameTiming::Add(C::PvGenMissN); break;
				case PvRangeGen::Outcome::Hit: FrameTiming::Add(C::PvGenHitN); break;
			}
		}
		PerVertexRecordRange range;
		bool                 known = false;
		// S413 (KYTY_PV_IND_RANGE): an On draw whose indices the GPU wrote takes the vertex range remembered for its key and does not read them
		// (the hash below drains the queue); the draws that hold the entry on the old path remember the range their scan finds.
		const auto              range_mode = PvIndirect::EnvRangeMode();
		PvIndirect::RangeEntry* range_entry = nullptr;
		PvIndirect::RangeKey    range_key;
		auto                    range_decision = PvIndirect::RangeDecision::NotApplicable;
		if (range_mode != PvIndirect::RangeMode::Off && pv_indirect != nullptr && (pvi_on || pv_indirect->range_record) &&
		    index_source.host_data == nullptr) {
			range_entry = PvIndirectRangeEntry(pv_indirect->entry_id);
			range_key   = {index_source.address, index_source.size, element_size, draw.index_count,
			               static_cast<uint32_t>(emit.vertex_offset)};
			if (range_entry != nullptr) {
				// S417: bytes the GPU wrote OR a page of the range the GPU owns (the read would fault although the bytes are clean).
				const bool dirty = pvi_on && PvIndirect::RangeReuseDirty(
				                                 !m_context.GetBufferCache().CanReadBacking(index_source.address, index_source.size),
				                                 m_context.GetBufferCache().IsRegionGpuModified(index_source.address, index_source.size));
				range_decision = PvIndirect::DecideRangeReuse(range_mode, pvi_on, dirty, *range_entry, range_key);
			}
		}
		if (range_decision == PvIndirect::RangeDecision::Reuse) {
			range        = {range_entry->range.first, range_entry->range.end, false};
			known        = true;
			range_reused = true;
			FrameTiming::Add(FrameTiming::Counter::PvHashBytesSkipped, index_source.size);
			FrameTiming::Add(FrameTiming::Counter::PvIndRangeReuseN);
		} else if (gen_mode == PvRangeGen::Mode::On && gen_outcome == PvRangeGen::Outcome::Hit) {
			range = {gen_entry->range.first, gen_entry->range.end, gen_entry->range.dropped};
			known = gen_entry->derivable;
			FrameTiming::Add(FrameTiming::Counter::PvHashBytesSkipped, index_source.size);
		} else {
			FrameTiming::Add(FrameTiming::Counter::PvVsstoreIndexBytes, index_source.size);
			// S417 (KYTY_PV_IDX_BACKING): clean index bytes on a protected page are copied from the backing, which neither faults nor waits.
			// The copy is what the hash and the scan see (mode 1); shadow keeps the guest read and compares it with the copy.
			const void*                       raw_bytes = idx_bytes;
			const void*                       copy_bytes = nullptr;
			thread_local std::vector<uint8_t> idx_copy;
			const auto                        backing_mode = PvIndexBacking::EnvMode();
			if (backing_mode != PvIndexBacking::Mode::Off && index_source.host_data == nullptr) {
				// S418: the backing is current only when the GPU has not written the bytes AND no copy of them is still in flight.
				int        backing_refusal = 0;
				const bool bytes_dirty =
				    !m_context.GetBufferCache().CanReadBacking(index_source.address, index_source.size, &backing_refusal);
				if (backing_refusal == 2) {
					FrameTiming::Add(FrameTiming::Counter::PvIdxBackingInflightN);
				}
				const auto source = PvIndexBacking::Decide(backing_mode, false, bytes_dirty);
				if (source == PvIndexBacking::Source::Raw) {
					FrameTiming::Add(FrameTiming::Counter::PvIdxBackingFallbackN);
				} else {
					idx_copy.resize(index_source.size);
					if (Libs::LibKernel::Memory::TryReadBacking(index_source.address, idx_copy.data(), index_source.size)) {
						copy_bytes = idx_copy.data();
						if (GuestFault::Enabled()) { // S421
							GuestFault::Reads().Note(index_source.address, index_source.size, GuestFault::Site::IndexBytes, FrameTiming::GuestFlips());
						}
						FrameTiming::Add(FrameTiming::Counter::PvIdxBackingN);
						FrameTiming::Add(FrameTiming::Counter::PvIdxBackingBytes, index_source.size);
						if (source == PvIndexBacking::Source::Backing) idx_bytes = copy_bytes;
					} else {
						FrameTiming::Add(FrameTiming::Counter::PvIdxBackingFallbackN);
					}
				}
			}
			uint64_t index_hash;
			{
				KYTY_CP_SITE(PvIndexHash);
				index_hash = XXH3_64bits(idx_bytes, index_source.size);
			}
			if (copy_bytes != nullptr && idx_bytes == raw_bytes) {
				// shadow: the hash above was the guest read; the copy was taken before it.
				const auto first_diff = PvIndexBacking::FirstDiff(copy_bytes, raw_bytes, index_source.size);
				if (first_diff == index_source.size) {
					FrameTiming::Add(FrameTiming::Counter::PvIdxBackingEqN);
				} else {
					FrameTiming::Add(FrameTiming::Counter::PvIdxBackingNeN);
					static std::atomic<uint32_t> ne_logged {0};
					if (ne_logged.fetch_add(1, std::memory_order_relaxed) < 20) {
						LOGF("[pv-idx-backing-ne] flip=%" PRIu64 " addr=0x%" PRIx64 " size=%" PRIu64 " first_diff=%" PRIu64 "\n",
						     FrameTiming::GuestFlips(), index_source.address, index_source.size, first_diff);
					}
				}
			}
			const PerVertexRangeMemo::Key memo_key {index_hash, index_source.size,
			                                        element_size, draw.index_count, emit.vertex_offset, kNoDropLimit};
			const auto memoized = m_prototype_range_memo.Find(memo_key);
			if (memoized.found) {
				range = memoized.range;
				known = memoized.derivable;
			} else {
				KYTY_CP_SITE(PvRangeScan);
				known = ComputePerVertexRecordRange(true, idx_bytes, index_source.size, element_size,
				                                    draw.index_count, emit.first_vertex, emit.vertex_offset,
				                                    kNoDropLimit, range);
				m_prototype_range_memo.Insert(memo_key, known, range);
			}
			if (gen_mode != PvRangeGen::Mode::Off) {
				const PvRangeGen::Range current {range.first, range.end, range.dropped};
				if (gen_outcome == PvRangeGen::Outcome::Hit) {
					// Shadow: what the hash would have skipped, and whether the generation told the truth.
					FrameTiming::Add(FrameTiming::Counter::PvHashBytesSkipped, index_source.size);
					const auto comparison = PvRangeGen::Compare(*gen_entry, index_hash, known, current);
					if (comparison.Mismatch()) {
						FrameTiming::Add(FrameTiming::Counter::PvGenMismatchN);
						static std::atomic<uint32_t> mismatch_logged {0};
						if (mismatch_logged.fetch_add(1, std::memory_order_relaxed) < 20) {
							LOGF("[pv-gen-mismatch] flip=%" PRIu64 " addr=0x%" PRIx64 " bytes=%" PRIu64 " index_count=%u hash_differs=%d "
							     "range_differs=%d stamp=%" PRIu64 " memo_range=%" PRIu64 "..%" PRIu64 " now=%" PRIu64 "..%" PRIu64 "\n",
							     FrameTiming::GuestFlips(), index_source.address, index_source.size, draw.index_count,
							     comparison.hash_differs ? 1 : 0, comparison.range_differs ? 1 : 0, gen_probe.max_stamp,
							     gen_entry->range.first, gen_entry->range.end, current.first, current.end);
						}
					}
				}
				if (gen_outcome == PvRangeGen::Outcome::Hit || gen_outcome == PvRangeGen::Outcome::Miss) {
					// The stamp was read before the bytes were hashed: a write in between only makes the entry miss later.
					m_pv_range_gen.Insert(gen_key, {gen_probe.max_stamp, index_hash, known, current});
				}
			}
		}
		if (range_entry != nullptr && !range_reused) {
			// S413: the CPU scan ran. shadow: against what is remembered; then remember it (an unusable scan forgets the entry's range).
			if (range_decision == PvIndirect::RangeDecision::Miss) FrameTiming::Add(FrameTiming::Counter::PvIndRangeMissN);
			const bool recordable = PvIndirect::RangeRecordable(known, range.dropped, range.first, range.end);
			if (range_mode == PvIndirect::RangeMode::Shadow && recordable) {
				const PvIndirect::Range scanned {range.first, range.end};
				const auto              comparison = PvIndirect::CompareRange(*range_entry, range_key, scanned);
				if (comparison == PvIndirect::RangeCompare::Equal) {
					FrameTiming::Add(FrameTiming::Counter::PvIndRangeEqN);
				} else if (comparison == PvIndirect::RangeCompare::NotEqual || comparison == PvIndirect::RangeCompare::NotEqualWithin) {
					FrameTiming::Add(FrameTiming::Counter::PvIndRangeNeN);
					if (comparison == PvIndirect::RangeCompare::NotEqualWithin) FrameTiming::Add(FrameTiming::Counter::PvIndRangeNeWithinN);
					static std::atomic<uint32_t> ne_logged {0};
					if (ne_logged.fetch_add(1, std::memory_order_relaxed) < 20) {
						LOGF("[pv-ind-range-ne] flip=%" PRIu64 " addr=0x%" PRIx64 " index_addr=0x%" PRIx64 " index_count=%u entry=%" PRIu64 "..%" PRIu64
						     " scan=%" PRIu64 "..%" PRIu64 " within=%d\n",
						     FrameTiming::GuestFlips(), pv_indirect->args_addr, index_source.address, draw.index_count, range_entry->range.first,
						     range_entry->range.end, scanned.first, scanned.end,
						     comparison == PvIndirect::RangeCompare::NotEqualWithin ? 1 : 0);
					}
				}
			}
			if (recordable) {
				PvIndirect::RecordRange(*range_entry, range_key, range.first, range.end);
			} else {
				range_entry->known = false;
			}
		}
		request.range_known   = known;
		request.range_dropped = range.dropped;
		request.range_first   = range.first;
		request.range_end     = range.end;
	}
	// A storage descriptor is bound from an aligned offset and the difference becomes first_index; the
	// guest address decides the offset modulo the element size, so an odd one stays on the prototype.
	const bool index_aligned =
	    element_size == 0 || index_source.host_data != nullptr || index_source.address % element_size == 0;
	// The descriptor of the index buffer must stay inside the buffer the cache hands out (its end is not
	// ours to choose): obtain the binding now (the draw obtains the same range again) and ask whether the
	// last word the PS reads fits. A host copy is padded to a whole word in PrepareIndexBuffer.
	bool index_tail_fits = true;
	if (index_aligned && scannable && index_source.host_data == nullptr) {
		KYTY_CP_SITE(PvObtainTail);
		const auto [index_buffer, index_offset] =
		    m_context.GetBufferCache().ObtainBuffer(index_source.address, index_source.size, false);
		index_tail_fits = PvVsStoreIndexDescriptorFor(index_offset, element_size, graphics.StorageMinAlignment(),
		                                              index_source.size, index_buffer->Size()).tail_fits;
	}
	// S388 (H1): the scan above read the index bytes on the CPU; indices the GPU wrote may differ by the time the draw runs.
	if (draw.IsIndexed() && index_source.size != 0) {
		const bool gpu_index =
		    index_source.host_data == nullptr &&
		    m_context.GetBufferCache().IsRegionGpuModified(index_source.address, index_source.size);
		if (gpu_index) {
			FrameTiming::Add(FrameTiming::Counter::PvVsStoreIdxGpu);
			static std::atomic<uint32_t> gpuidx_logged {0};
			if (gpuidx_logged.fetch_add(1, std::memory_order_relaxed) < 20) {
				LOGF("[pv-gpuidx] vs=0x%016llx ps=0x%016llx addr=0x%llx bytes=%llu inst=%u\n",
				     static_cast<unsigned long long>(state.programs.vertex[0].id),
				     static_cast<unsigned long long>(state.programs.pixel.id),
				     static_cast<unsigned long long>(index_source.address),
				     static_cast<unsigned long long>(index_source.size), draw.instance_count);
			}
		}
		if (PvVsStoreGpuIndexFallback(PerVertexVsStoreGpuIdxMode(), gpu_index)) {
			fallback(PvVsStoreFallback::Range, PvIndirect::Decline::Range);
			return false;
		}
	}
	if (!m_pv_vsstore) m_pv_vsstore = std::make_shared<PvVsStoreRuntime>(graphics, m_context.GetCommandScheduler());
	PvVsStoreRuntime::Region region;
	const auto decision = DecidePvVsStore(programs->verdict, request, index_aligned, [&](const PvVsStorePlan& plan) {
		KYTY_CP_SITE(PvPlace);
		region = m_pv_vsstore->Place(plan.records, programs->stride_vec4);
		return region.ok;
	}, index_tail_fits);
	if (!decision.use) {
		{
			PvIndirect::Decline reason = PvIndirect::Decline::Other;
			switch (decision.fallback) {
				case PvVsStoreFallback::Range: reason = PvIndirect::Decline::Range; break;
				case PvVsStoreFallback::IndexTail: reason = PvIndirect::Decline::IndexTail; break;
				case PvVsStoreFallback::Ring: reason = PvIndirect::Decline::Ring; break;
				case PvVsStoreFallback::Instanced:
					// A draw with several instances is classified as "instanced" whatever the size reason; the plan says which.
					// (the ring check runs after the plan accepted the draw, so an accepted plan means the ring)
					reason = decision.plan.verdict == PvVsStoreVerdict::Use           ? PvIndirect::Decline::Ring
					         : decision.plan.verdict == PvVsStoreVerdict::TooLarge     ? PvIndirect::Decline::TooLarge
					         : decision.plan.verdict == PvVsStoreVerdict::InstancedOff ? PvIndirect::Decline::Other
					                                                                   : PvIndirect::Decline::Range;
					break;
				default: break;
			}
			fallback(decision.fallback, reason);
		}
		// S151: the first draw that stays on the prototype for its topology says which one it was.
		static std::atomic<bool> topology_logged {false};
		if (decision.fallback == PvVsStoreFallback::Topology && !topology_logged.exchange(true)) {
			LOGF("PvVsStore: first topology fallback: topology=%u indexed=%d restart=%d index_size=%u indices=%u "
			     "VS=0x%016llx PS=0x%016llx\n",
			     static_cast<uint32_t>(topology), request.indexed ? 1 : 0, primitive_restart_enable ? 1 : 0,
			     request.index_size, draw.index_count, static_cast<unsigned long long>(state.programs.vertex[0].id),
			     static_cast<unsigned long long>(state.programs.pixel.id));
		}
		return false;
	}
	const auto& plan = decision.plan;
	setup.programs           = programs;
	setup.index_element_size = element_size;
	setup.region             = region;
	setup.draw.base          = region.base_record;
	setup.draw.span          = plan.span;
	setup.draw.first         = plan.first;
	setup.draw.first_instance = emit.first_instance;
	setup.draw.first_index   = 0; // set once the index buffer is bound (alignment)
	setup.draw.vertex_offset = emit.vertex_offset;
	setup.draw.index32       = element_size == 4;
	setup.draw.records       = static_cast<uint32_t>(plan.records);
	setup.range_reused       = range_reused;
	setup.range_first        = request.range_first;
	setup.range_end          = request.range_end;
	if (pv_indirect != nullptr && pv_indirect->mode != PvIndirect::Mode::Off) {
		// S408: the plan with cap instances is the plan of the prepared draw (mode On: `plan` is it); a shadow draw compares it with the real one.
		auto cap_request           = request;
		cap_request.instance_count = pv_indirect->cap;
		PvIndirectNotePlan(*pv_indirect, PlanPvVsStore(cap_request), plan, programs->stride_vec4);
	}
	if (PerVertexVsStoreGuard()) setup.guard = m_pv_vsstore->AcquireGuard(FrameTiming::GuestFlips());
	if (const auto& ctx = SyncCensus::IndirectPvCtx(); ctx.active) {
		// S406 (KYTY_DIAG_INDIRECT=1): the VS-store plan of an indirect draw next to the args before / after the drain.
		static std::atomic<uint32_t> logged {0};
		const auto                   flip = FrameTiming::GuestFlips();
		if (logged.fetch_add(1, std::memory_order_relaxed) < 200 || (flip >= 6300 && flip <= 6302)) {
			LOGF("[indirect-pv] flip=%" PRIu64 " eq=%d stale=%u,%u,%u,%u,%u fresh=%u,%u,%u,%u,%u first=%u span=%u records=%u instances=%u "
			     "ring_free=%" PRIu64 " index_count=%u\n",
			     flip, ctx.Equal() ? 1 : 0, ctx.stale[0], ctx.stale[1], ctx.stale[2], ctx.stale[3], ctx.stale[4], ctx.fresh[0],
			     ctx.fresh[1], ctx.fresh[2], ctx.fresh[3], ctx.fresh[4], static_cast<uint32_t>(plan.first),
			     static_cast<uint32_t>(plan.span), static_cast<uint32_t>(plan.records), draw.instance_count,
			     m_pv_vsstore->FreeRecords(), draw.index_count);
		}
	}
	return true;
}

// Binds set 1 (cap slab + index buffer) and pushes the VS-store block; the pipeline of the draw is the
// one made from setup.programs, so its layout has both.
static void BindPvVsStore(RenderContext& context, vk::CommandBuffer vk_buffer, const PipelineCache::Pipeline& pipeline,
                          PvVsStoreDrawSetup& setup, const PreparedIndexBuffer& index_binding,
                          const DrawIndexBufferSource& index_source) {
	auto&      graphics = context.GetGraphics();
	const auto bind     = PvVsStoreIndexDescriptorFor(index_binding.offset, setup.index_element_size,
	                                                  graphics.StorageMinAlignment(), index_source.size, index_binding.size);
	EXIT_IF(!bind.ok || index_binding.buffer == nullptr);
	setup.draw.first_index = bind.first_index;
	const bool guard = PerVertexVsStoreGuard();
	const vk::DescriptorBufferInfo infos[] {
	    {setup.region.buffer, 0, setup.region.buffer_size},
	    {index_binding.buffer, bind.offset, bind.range},
	    {setup.guard.buffer, setup.guard.offset, setup.guard.range},
	};
	const uint32_t binding_count = guard ? 3u : 2u;
	const auto set = context.GetDescriptorHeap().Commit(PvVsStoreSetLayout(graphics));
	vk::WriteDescriptorSet writes[3] {};
	for (uint32_t i = 0; i < binding_count; ++i) {
		writes[i].dstSet          = set;
		writes[i].dstBinding      = i;
		writes[i].descriptorCount = 1;
		writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		writes[i].pBufferInfo     = &infos[i];
	}
	graphics.device.updateDescriptorSets(binding_count, writes, 0, nullptr);
	vk_buffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 1, 1, &set, 0, nullptr);
	const auto block = PvVsStorePushBlock(setup.draw, setup.programs->stride_vec4);
	// S409: the bytes of records this draw has bound (both modes of KYTY_PV_VSSTORE_COMPACT, to measure the mechanism).
	FrameTiming::Add(FrameTiming::Counter::PvVsstoreRecordBytes, setup.draw.records * setup.programs->stride_vec4 * 16ull);
	if (setup.programs->compact) FrameTiming::Add(FrameTiming::Counter::PvCompactDrawsN);
	vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
	                        PvVsStorePushOffset(), sizeof(block), block.data());
	if (guard) {
		vk_buffer.pushConstants(pipeline.pipeline_layout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
		                        PvVsStorePushOffset() + kPvPushRecords * 4u, sizeof(uint32_t), &setup.draw.records);
	}
}

// S408: the commands of the prepare compute, in the order PvIndirect::RecordPrepare dictates.
namespace {
struct PvIndirectRecorder {
	CommandScheduler&                  scheduler;
	CommandBuffer&                     buffer;
	const RenderState&                 rendering;
	PvIndirectRuntime&                 runtime;
	const PvIndirectRuntime::Prepared& prepared;
	vk::CommandBuffer                  command;
	void EndPass() {
		// The pass that is open and is the one this draw continues is reopened with loads (a clear must not be repeated).
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

void RenderExecutor::ExecutePreparedDraw(uint64_t submit_id, CommandBuffer& buffer,
                                         const DrawCallInfo& draw_in, DrawRenderState& state,
                                         vk::PrimitiveTopology topology, const DrawEmitInfo& emit_in,
                                         const DrawIndexBufferSource& index_source_in,
	                                     bool primitive_restart_enable, PvIndirectDraw* pvi) {
	std::optional<CpPhases::SubTimer> seg(std::in_place, FrameTiming::Counter::DrawPrologNs); // S463 DIAG
	DrawCallInfo          draw         = draw_in; // S404: the indirect buffer is filled in below
	DrawEmitInfo          emit         = emit_in;
	DrawIndexBufferSource index_source = index_source_in;
	// S408 (KYTY_PV_INDIRECT): On draws from the args prepared on the GPU. The counts are the learned ones and the cap, only to make the
	// plan and the region; the shapes that must not change (first index, vertex offset, first instance) are checked by the prepare compute.
	const bool pvi_on     = pvi != nullptr && pvi->mode == PvIndirect::Mode::On;
	const bool pvi_shadow = pvi != nullptr && pvi->mode == PvIndirect::Mode::Shadow;
	if (pvi_on) {
		draw.index_count    = pvi->expected.index_count;
		draw.instance_count = pvi->cap;
		emit.vertex_offset  = static_cast<int32_t>(pvi->expected.vertex_offset);
		index_source.size   = uint64_t {draw.index_count} * index_source.guest_element_size;
	}
	PvVsStoreDrawSetup  pv_store;
	bool                use_pv_store = false;
	PvIndirect::Decline pvi_why      = PvIndirect::Decline::Other;
	if (PerVertexVsStoreEnabled()) {
		CpPhases::Scope phase(CpPhases::Phase::PvStore);
		use_pv_store = TryPvVsStoreDraw(draw, state, topology, primitive_restart_enable, emit, index_source, pv_store, pvi, &pvi_why);
	}
	if (pvi_on && !use_pv_store) {
		// Nothing has been recorded: the caller reads the args and draws the old way (never the prototype with placeholder counts).
		pvi->declined = true;
		pvi->why      = pvi_why;
		return;
	}
	if (pvi_shadow && !use_pv_store) FrameTiming::Add(FrameTiming::Counter::PvIndShadowNoStoreN);
	bool pvi_prepare = pvi_on || (pvi_shadow && use_pv_store);
	// (mode On counts the draw once it is certain not to be declined, below)
	const auto count_pv_draw = [&] {
		FrameTiming::Add(FrameTiming::Counter::PvVsStoreDraws);
		if (draw.instance_count > 1) FrameTiming::Add(FrameTiming::Counter::PvVsStoreInstanced);
	};
	if (use_pv_store && !pvi_on) count_pv_draw();
	// The per-vertex prototype takes the draw when the VS store does not; true when it handled the draw (ran or skipped it).
	const auto run_prototype = [&]() -> bool {
		if (state.vertex_info[0].logical_stage != ShaderType::Vertex) return false;
		if (const auto* prototype = GetPerVertexPrototypePrograms(
		        m_context.GetGraphics(), state.programs, state.vertex_info[0],
		        m_context.GetPipelineCache().DriverCache())) {
			const auto verdict = ClassifyPerVertexPrototypeDraw(
			    state.ps_active, topology, primitive_restart_enable, draw.index_count);
			if (verdict.reason != nullptr) {
				static std::atomic<uint32_t> logged {0};
				if (!verdict.run) FrameTiming::Add(FrameTiming::Counter::ProtoUnsupportedDraws);
				if (logged.fetch_add(1, std::memory_order_relaxed) < 20) {
					LOGF("PerVertexPrototype: %s draw (%s): topology=%u restart=%d indices=%u "
					     "VS=0x%016llx PS=0x%016llx\n",
					     verdict.run ? "adjusting" : "skipping", verdict.reason,
					     static_cast<uint32_t>(topology), primitive_restart_enable ? 1 : 0,
					     draw.index_count,
					     static_cast<unsigned long long>(state.programs.vertex[0].id),
					     static_cast<unsigned long long>(state.programs.pixel.id));
				}
			}
			// S132: KYTY_DIAG_PROTO_SKIP=1 drops supported prototype draws (tempo ablation only, the image is incomplete).
			static const bool diag_skip = FrameTiming::DiagFlagOn(std::getenv("KYTY_DIAG_PROTO_SKIP"));
			const auto action = FrameTiming::ProtoDrawDecision(verdict.run, diag_skip);
			if (action == FrameTiming::ProtoDrawAction::Skip) return true;
			if (action == FrameTiming::ProtoDrawAction::DiagSkipped) {
				FrameTiming::Add(FrameTiming::Counter::ProtoSkippedDraws);
				return true;
			}
			DrawCallInfo adjusted = draw;
			adjusted.index_count  = verdict.index_count;
			FrameTiming::Add(FrameTiming::Counter::PrototypeDraws);
			FrameTiming::Add(FrameTiming::Counter::PrototypeVertices,
			                 uint64_t {adjusted.index_count} * adjusted.instance_count);
			{
				FrameTiming::ScopedCounterNs proto_time(FrameTiming::Counter::CpProtoNs);
				ExecutePerVertexPrototype(submit_id, buffer, adjusted, state, topology, emit,
				                          index_source, primitive_restart_enable);
			}
			return true;
		}
		return false;
	};
	if (!use_pv_store && run_prototype()) return;
	// This draw is not itself a (possibly batchable) per-vertex-prototype draw: flush any pending
	// batch first, same reasoning as the PM4-interpreter-level flush in graphicsRun.cpp -- two
	// consecutive IT_DRAW_INDEX_2 packets reach the same opcode check there, so only the check
	// here (now that we know this one isn't prototype-eligible) can catch that case.
	{
		KYTY_CP_SITE(DrawFlushProto);
		FlushPendingPrototypeBatch(buffer);
	}
	auto& ucfg = buffer.GetUserConfig();
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	const auto vertex_stages =
	    std::span {state.vertex_info.data(), state.programs.VertexStageCount()};
	const bool mesh_active = state.vertex_info[0].stage.program->stage == ShaderType::Mesh;
	uint32_t   mesh_groups = 0;
	if (mesh_active) {
		const auto& mesh = state.vertex_info[0].mesh;
		EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.IsIndexed() || primitive_restart_enable));
		static std::atomic_bool restart_warned = false;
		if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
			std::printf("Warning: primitive restart is not implemented for mesh shaders; "
			            "continuing draw (primitive=%u indexed=%u)\n",
			            static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed());
		}
		if (mesh.primitives_per_group == 0) {
			EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
			     static_cast<uint32_t>(ucfg.GetPrimType()), draw.IsIndexed(),
			     primitive_restart_enable);
		}
		const auto primitives = mesh.InputPrimitiveCount(draw.index_count);
		if (primitives == 0 || draw.instance_count == 0) {
			return;
		}
		mesh_groups        = (primitives - 1u) / mesh.primitives_per_group + 1u;
		const auto& limits = m_context.GetGraphics().mesh_shader_properties;
		if (mesh_groups > limits.maxMeshWorkGroupCount[0] ||
		    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
		    static_cast<uint64_t>(mesh_groups) * draw.instance_count >
		        limits.maxMeshWorkGroupTotalCount) {
			EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", mesh_groups,
			     draw.instance_count);
		}
	}

	if (mesh_active && draw.IsIndexed()) {
		// Register the original guest indices for shader reads; PrepareGraphicsBindings
		// synchronizes registered BDA ranges before any draw commands are committed.
		(void)m_context.GetBufferCache().FindBuffer(index_source.address,
		                                            static_cast<uint64_t>(draw.index_count) *
		                                                index_source.guest_element_size);
	}
	LogDrawPhase(draw.Name(), "PrepareBindings");
	seg.reset();
	std::optional<CpPhases::Scope> bind_phase;
	bind_phase.emplace(CpPhases::Phase::DrawBind);
	auto&                            bindings = m_graphics_bindings;
	std::array<PreparedBindings*, 4> descriptor_stages {};
	uint32_t                         stage_count = 0;
	{
		CpPhases::SubTimer prep_timer(FrameTiming::Counter::BindPrepNs);
		for (uint32_t i = 0; i < vertex_stages.size(); i++) {
			PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
			descriptor_stages[stage_count++] = &bindings.vertex[i];
		}
		if (state.ps_active) {
			if (!bindings.pixel) bindings.pixel.emplace();
			PrepareBindings(state.ps_input_info.stage, *bindings.pixel);
			descriptor_stages[stage_count++] = &*bindings.pixel;
		}
	}
	const auto stages = std::span {descriptor_stages.data(), stage_count};
	PrepareGraphicsBindings(stages, std::span {state.color_info, state.color_count});
	bind_phase.reset();
	seg.emplace(FrameTiming::Counter::DrawVbNs);
	PreparedVertexBuffers vertex_bindings;
	PreparedIndexBuffer   index_binding;
	vk::Buffer            pvi_args_buffer = nullptr;
	uint64_t              pvi_args_offset = 0;
	if (!mesh_active) {
		LogDrawPhase(draw.Name(), "PrepareVertexBuffers");
		{
			KYTY_CP_SITE(DrawVertexBuffers);
			vertex_bindings = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		}
		KYTY_CP_SITE(DrawIndexBuffer);
		// S408: the prepare compute reads the args from the buffer cache the same way (mode On: the draw itself reads the prepared copy).
		const uint64_t args_lookup = draw.indirect_addr != 0 ? draw.indirect_addr : (pvi_prepare ? pvi->args_addr : 0);
		if (args_lookup != 0) {
			// S404: the args are a GPU-read buffer of the cache like the index buffer; obtaining one can merge the other, so the
			// index binding is taken between two lookups of the args and the last lookup is the one that is used.
			EXIT_IF(draw.indirect_addr != 0 && use_pv_store && !pvi_on);
			(void)m_context.GetBufferCache().ObtainBuffer(args_lookup, sizeof(vk::DrawIndexedIndirectCommand), false);
		}
		index_binding   = PrepareIndexBuffer(buffer, index_source);
		if (args_lookup != 0) {
			const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
			    args_lookup, sizeof(vk::DrawIndexedIndirectCommand), false);
			EXIT_IF(args_buffer == nullptr || (args_offset & 3u) != 0);
			if (draw.indirect_addr != 0) {
				draw.indirect_buffer = args_buffer->Handle();
				draw.indirect_offset = args_offset;
			}
			pvi_args_buffer = args_buffer->Handle();
			pvi_args_offset = args_offset;
		}
	}
	EXIT_IF(draw.indirect_addr != 0 && (mesh_active || (use_pv_store && !pvi_on)));
	// The index binding the draw really got (the cache may move or merge the range while the bindings above are
	// prepared) decides the VS-store tail again; a tail that does not fit goes to the prototype, as in the early check.
	if (use_pv_store && !mesh_active && draw.IsIndexed() && index_binding.buffer != nullptr &&
	    !PvVsStoreIndexBindingFits(index_binding.offset, pv_store.index_element_size,
	                               m_context.GetGraphics().StorageMinAlignment(), index_source.size,
	                               index_binding.size)) {
		use_pv_store = false;
		if (pvi_on) {
			// The resources of the draw are prepared but nothing of it is recorded: the old path prepares them again.
			pvi->declined = true;
			pvi->why      = PvIndirect::Decline::IndexTail;
			return;
		}
		CountPvFallback(PvVsStoreFallback::IndexTail);
		if (pvi_prepare) {
			pvi_prepare = false;
			FrameTiming::Add(FrameTiming::Counter::PvIndShadowNoStoreN);
		}
		if (run_prototype()) return;
	}
	seg.reset();
	if (draw.IsIndexed()) {
		LogDrawPhase(draw.Name(), "CreatePipeline");
	}
	std::optional<CpPhases::Scope> pipeline_phase;
	pipeline_phase.emplace(CpPhases::Phase::DrawPipeline);
	auto& pipeline = m_context.GetPipelineCache().GetGraphicsPipeline(
	    std::span {state.color_info, state.color_count}, state.depth_info, vertex_stages, buffer,
	    state.ps_active ? &state.ps_input_info : nullptr, topology, primitive_restart_enable,
	    use_pv_store ? pv_store.programs->graphics : state.programs);
	pipeline_phase.reset();
	vk::ImageAspectFlags feedback_aspects;
	pipeline_phase.emplace(CpPhases::Phase::DrawTargets);
	const auto rendering =
	    AcquireRenderTargets(buffer, state.color_info, state.color_count, state.depth_info,
	                         feedback_aspects, stages);
	pipeline_phase.reset();
	seg.emplace(FrameTiming::Counter::DrawPreEmitNs);
	// S327: the pass about to be needed differs from the open one (BeginRendering would close it below): split here, before this draw
	// records anything into the buffer.
	{
		KYTY_CP_SITE(DrawEagerSplit);
		m_context.GetCommandScheduler().MaybeEagerSplitAtPass(rendering);
	}
	CpPhases::Scope vkcmd_phase(CpPhases::Phase::DrawVkcmd); // to the end of the draw; CommitBindings below is its child

	// S408: a ring slot for the prepared args of this draw (taken for the command buffer that records it).
	PvIndirectRuntime::Prepared pvi_prepared;
	if (pvi_prepare) {
		EXIT_IF(pvi_args_buffer == nullptr);
		auto& runtime = PvIndirectRt();
		PvIndirectAdvance();
		PvIndirectRuntime::Job job {pvi_args_buffer, pvi_args_offset, pvi->expected, pvi->cap, pvi->entry_id, FrameTiming::GuestFlips()};
		if (pvi_on && pv_store.range_reused) {
			// S413: the prepare compute checks the draw's indices against the remembered range (the binding the VS-store reads).
			const auto bind = PvVsStoreIndexDescriptorFor(index_binding.offset, pv_store.index_element_size,
			                                              m_context.GetGraphics().StorageMinAlignment(), index_source.size,
			                                              index_binding.size);
			EXIT_IF(!bind.ok || index_binding.buffer == nullptr);
			job.range_check  = true;
			job.range        = {pv_store.range_first, pv_store.range_end};
			job.index_buffer = index_binding.buffer;
			job.index_offset = bind.offset;
			job.index_range  = bind.range;
			job.index_first  = bind.first_index;
			job.index_count  = pvi->expected.index_count;
			job.index32      = pv_store.index_element_size == 4;
		}
		pvi_prepared = runtime.Reserve(job);
		if (!pvi_prepared.ok && pvi_on) {
			pvi->declined = true;
			pvi->why      = PvIndirect::Decline::Ring;
			return;
		}
		if (pvi_on) count_pv_draw();
	}

	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory.
	auto vk_buffer = buffer.Handle();
	if (pvi_prepared.ok) {
		// The prepare compute runs outside a pass, before the draw: the draw (mode On) takes its output as indirect args.
		PvIndirectRecorder recorder {m_context.GetCommandScheduler(), buffer, rendering, PvIndirectRt(), pvi_prepared, vk_buffer};
		const bool         pass_break = PvIndirect::RecordPrepare(recorder, m_context.GetCommandScheduler().RenderingOpen());
		if (pass_break) FrameTiming::Add(FrameTiming::Counter::PvIndPassBreakN);
		if (pvi_on) {
			draw.indirect_buffer = pvi_prepared.out_buffer;
			draw.indirect_offset = pvi_prepared.out_offset;
			FrameTiming::Add(FrameTiming::Counter::PvIndN);
		}
		static std::atomic<uint32_t> logged {0};
		const auto                   flip = FrameTiming::GuestFlips();
		if (logged.fetch_add(1, std::memory_order_relaxed) < 50 || (flip >= 6300 && flip <= 6302)) {
			LOGF("[pv-ind] mode=%s flip=%" PRIu64 " args=0x%" PRIx64 " expected=%u,%u,%u,%u cap=%u vs=0x%016" PRIx64 " ps=0x%016" PRIx64
			     " stride=%u span=%u first=%u records=%u ring_free=%" PRIu64 " pass_break=%d\n",
			     pvi_on ? "1" : "shadow", flip, pvi->args_addr, pvi->expected.index_count, pvi->expected.first_index,
			     pvi->expected.vertex_offset, pvi->expected.first_instance, pvi->cap, pv_store.programs->vs_hash, pv_store.programs->ps_hash,
			     pv_store.programs->stride_vec4, pv_store.draw.span, pv_store.draw.first, pv_store.draw.records,
			     m_pv_vsstore ? m_pv_vsstore->FreeRecords() : uint64_t {0}, pass_break ? 1 : 0);
		}
	}
	SetDrawDebugPhase(buffer, submit_id, draw, draw.IsIndexed() ? 0x100u : 0x200u);
	if (!mesh_active) {
		CommitVertexBuffers(vk_buffer, vertex_bindings);
	}
	if (state.ps_active && !draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x300u);
	}
	{
		CpPhases::Scope desc_phase(CpPhases::Phase::DrawDesc);
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	}
	if (!mesh_active) {
		CommitIndexBuffer(vk_buffer, index_binding);
		if (use_pv_store) {
			m_pv_vsstore->Touch(pv_store.region);
			BindPvVsStore(m_context, vk_buffer, pipeline, pv_store, index_binding, index_source);
		}
	}
	const auto draw_dwords = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, state.ps_input_info.object_primitive_id);
	if (draw_dwords != 0) {
		std::array<uint32_t, ShaderRecompiler::IR::PushData::MeshDrawDwordCount> draw_data {};
		const auto draw_stages = vk::ShaderStageFlagBits::eFragment |
		                         (mesh_active ? vk::ShaderStageFlagBits::eMeshEXT
		                                      : vk::ShaderStageFlagBits::eVertex);
		if (mesh_active) {
			draw_data = {
			    draw.index_count,
			    draw.IsIndexed() ? static_cast<uint32_t>(emit.vertex_offset) : emit.first_vertex,
			    emit.first_instance, index_source.guest_element_size,
			    static_cast<uint32_t>(index_source.address),
			    static_cast<uint32_t>(index_source.address >> 32u)};
		} else {
			draw_data[0] = buffer.GetUserConfig().GetObjectId();
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, draw_stages, 0,
		                        draw_dwords * sizeof(uint32_t), draw_data.data());
	}

	seg.emplace(FrameTiming::Counter::DrawDynNs);
	auto& dynamic_state = m_context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), state.depth_info,
	                           rendering);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	dynamic_state.Commit(m_context.GetGraphics(), vk_buffer);

	LogDrawPhase(draw.Name(), "BeginRendering");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x400u);
	}
	m_context.GetCommandScheduler().BeginRendering(rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x500u);
	}
	seg.emplace(FrameTiming::Counter::DrawEmitNs);
	if (mesh_active) {
		vk_buffer.drawMeshTasksEXT(mesh_groups, draw.instance_count, 1);
	} else {
		EmitDrawPrimitives(ucfg, vk_buffer, draw, emit);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances, draw.instance_count);

	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x600u);
	}
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: vertex_stages) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	if (shader_write_stages) {
		m_context.GetCommandScheduler().EndRendering();
		ShaderWriteBarrier(vk_buffer, shader_write_stages);
	}
	{ // S459 DIAG: KYTY_DIAG_S459_FLIP=<n> lists every graphics draw (PS, target, blend) in guest flips n..n+2
		static const long s459_flip = std::getenv("KYTY_DIAG_S459_FLIP") ? std::atol(std::getenv("KYTY_DIAG_S459_FLIP")) : -1;
		const long        now_flip  = static_cast<long>(FrameTiming::GuestFlips() + 1);
		if (s459_flip >= 0 && now_flip >= s459_flip && now_flip < s459_flip + 3) {
			static FILE* s459_file = std::fopen(std::getenv("KYTY_DIAG_S459_FILE") ? std::getenv("KYTY_DIAG_S459_FILE") : "s459-draws.txt", "a");
			if (s459_file) {
				const auto& s459_hw = buffer.GetRegisters();
				std::fprintf(s459_file, "[s459-draw] flip=%ld ps=0x%016" PRIx64 " idx=%u colors=%u", now_flip,
				             state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0, static_cast<unsigned>(draw.index_count),
				             static_cast<unsigned>(state.color_count));
				for (uint32_t i = 0; i < state.color_count; i++) {
					const auto& c  = state.color_info[i];
					const auto& bc = s459_hw.GetBlendControl(c.target_slot);
					std::fprintf(s459_file, " | slot=%u fmt=%d %ux%u map=0x%02x blend=%d c=%u/%u/%u a=%u/%u/%u sep=%d", c.target_slot,
					             static_cast<int>(c.desc.info.pixel_format), c.Extent().width, c.Extent().height, c.export_mapping.packed,
					             bc.enable ? 1 : 0, bc.color_srcblend, bc.color_comb_fcn, bc.color_destblend, bc.alpha_srcblend,
					             bc.alpha_comb_fcn, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0);
				}
				std::fprintf(s459_file, " depth=%d\n", state.depth_info.image_id ? 1 : 0);
				std::fflush(s459_file);
			}
		}
	}
	DiagDumpAfterDraw(m_context, state.ps_active,
	                  state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0, state.color_info,
	                  state.color_count, state.depth_info);
	DiagTexAfterDraw(m_context);
	DiagTexDrawsDone(draw.index_count, state.color_info, state.color_count, state.depth_info);
	LogDrawPhase(draw.Name(), "DrawComplete");
	if (!draw.IsIndexed()) {
		SetDrawDebugPhase(buffer, submit_id, draw, 0x700u);
	}
}

void RenderExecutor::DrawIndex(uint64_t submit_id, CommandBuffer& buffer,
                               const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();
	auto* const igpu = args.indirect_gpu; // S404: KYTY_INDIRECT_GPU request (placeholder counts with mode On)
	const bool  igpu_on = igpu != nullptr && igpu->mode == IndirectGpu::Mode::On;
	FrameTiming::Add(FrameTiming::Counter::Draws);
	{
		KYTY_CP_SITE(DrawEagerSplit);
		m_context.GetCommandScheduler().MaybeEagerSplit();
	}

	std::optional<CpPhases::SubTimer> head_timer(std::in_place, FrameTiming::Counter::DrawHeadNs); // S463 DIAG
	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndex), submit_id,
	                    args.index_count, 0, 1, args.instance_count,
	                    reinterpret_cast<uint64_t>(args.index_addr));

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || args.instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
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

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		return;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16:
			index_source.type               = vk::IndexType::eUint16;
			index_source.guest_element_size = 2;
			break;
		case Prospero::IndexType::kIndex32:
			index_source.type               = vk::IndexType::eUint32;
			index_source.guest_element_size = 4;
			break;
		case Prospero::IndexType::kIndex8: index_source.guest_element_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	// S404: a candidate's index range is the whole INDEX_BUFFER_SIZE: scanning it for a restart value would read indices the
	// draw may never fetch, so restart control on a strip / fan declines (below) instead of being resolved.
	const bool primitive_restart = igpu_on ? false : ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] = primitive_restart && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndex, args.index_count, args.instance_count,
	                         args.first_instance};
	DrawRenderState    state {};
	head_timer.emplace(FrameTiming::Counter::DrawStateNs);
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}
	head_timer.reset();

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size, args.index_addr);

	DrawCallInfo   draw_to_run = draw;
	PvIndirectDraw pvi;       // S408: mode Off unless the draw takes (On) or shadows (Shadow) the prepared args
	auto* const    pv_request = args.pv_indirect;
	if (igpu != nullptr || (pv_request != nullptr && pv_request->mode != PvIndirect::Mode::Off)) {
		// S404: what the host path of this draw is, without looking at index_count / instance_count (nothing below may).
		IndirectGpu::HostPath host;
		const auto&           first_stage = state.vertex_info[0];
		const bool            vertex_stage = first_stage.logical_stage == ShaderType::Vertex;
		host.pv_store_possible = PerVertexVsStoreEnabled() && state.ps_active && vertex_stage &&
		                         first_stage.stage.program->stage != ShaderType::Mesh &&
		                         GetPvVsStorePrograms(m_context.GetGraphics(), state.programs) != nullptr;
		host.proto_possible = vertex_stage && GetPerVertexPrototypePrograms(m_context.GetGraphics(), state.programs, first_stage,
		                                                                    m_context.GetPipelineCache().DriverCache()) != nullptr;
		host.mesh              = first_stage.stage.program->stage == ShaderType::Mesh;
		host.copy_depth        = buffer.GetRegisters().GetRenderControl().copy_depth_to_color;
		host.quad_list_legacy  = ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy;
		host.restart           = PrimitiveRestartControlOn(buffer);
		host.native_fetch      = !first_stage.fetch_embedded;
		const auto host_reason = IndirectGpu::QualifyHost(host);
		if (igpu != nullptr) igpu->host_reason = host_reason;
		if (pv_request != nullptr && pv_request->mode != PvIndirect::Mode::Off) {
			// S408: a VS-store pair with nothing else in the way may take the prepared args (learns when the CPU has read them).
			const bool pv_candidate = host.pv_store_possible && PvIndirect::ResidualReason(host) == IndirectGpu::Reason::Eligible;
			pvi = PvIndirectDecide(*pv_request, pv_candidate);
			if (!pv_candidate && host.pv_store_possible && !pv_request->have_cpu_args) {
				FrameTiming::Add(FrameTiming::Counter::PvIndDeclineN);
				FrameTiming::Add(FrameTiming::Counter::PvIndDeclineResidualN);
			}
		}
		if (igpu_on) {
			if (host_reason != IndirectGpu::Reason::Eligible && pvi.mode != PvIndirect::Mode::On) {
				// Nothing has been recorded: the caller reads the args and draws the old way. The draw is counted by that second call.
				igpu->declined = true;
				ResetBindings();
				FrameTiming::Add(FrameTiming::Counter::Draws, ~uint64_t {0});
				return;
			}
			draw_to_run.indirect_addr = igpu->args_addr;
		}
	}

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);

	DrawEmitInfo emit {};
	// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
	const bool native_indirect = indirect && !state.vertex_info[0].fetch_embedded;
	emit.vertex_offset  = native_indirect ? 0 : vertex_offset + args.base_vertex;
	emit.first_instance = native_indirect ? 0 : instance_offset;

	ExecutePreparedDraw(submit_id, buffer, draw_to_run, state, topology, emit, index_source,
	                    primitive_restart, (pvi.mode != PvIndirect::Mode::Off || pvi.range_record) ? &pvi : nullptr);
	if (pvi.declined) {
		// S408: the prepared draw was given up before anything of it was recorded: the caller reads the args and draws the old way.
		using C = FrameTiming::Counter;
		FrameTiming::Add(C::PvIndDeclineN);
		switch (pvi.why) {
			case PvIndirect::Decline::Range: FrameTiming::Add(C::PvIndDeclineRangeN); break;
			case PvIndirect::Decline::Ring: FrameTiming::Add(C::PvIndDeclineRingN); break;
			case PvIndirect::Decline::IndexTail: FrameTiming::Add(C::PvIndDeclineIndexTailN); break;
			case PvIndirect::Decline::TooLarge: FrameTiming::Add(C::PvIndDeclineTooLargeN); break;
			default: FrameTiming::Add(C::PvIndDeclineOtherN); break;
		}
		static std::atomic<uint32_t> declined_logged {0};
		if (declined_logged.fetch_add(1, std::memory_order_relaxed) < 20) {
			LOGF("[pv-ind-decline] flip=%" PRIu64 " args=0x%" PRIx64 " reason=%s\n", FrameTiming::GuestFlips(), pvi.args_addr,
			     PvIndirect::DeclineName(pvi.why));
		}
		EXIT_IF(igpu == nullptr);
		igpu->declined = true;
		FrameTiming::Add(FrameTiming::Counter::Draws, ~uint64_t {0});
	}
	ResetBindings();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void RenderExecutor::DrawAuto(uint64_t submit_id, CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();
	FrameTiming::Add(FrameTiming::Counter::Draws);
	{
		KYTY_CP_SITE(DrawEagerSplit);
		m_context.GetCommandScheduler().MaybeEagerSplit();
	}

	EXIT_IF(buffer.IsInvalid());
	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	buffer.SetDebugInfo(static_cast<uint32_t>(CommandBufferDebugOp::DrawIndexAuto), submit_id,
	                    args.vertex_count, 0, args.first_vertex, args.instance_count,
	                    args.first_instance);

	Common::LockGuard lock(m_context.GetMutex());
	if (args.vertex_count == 0 || args.instance_count == 0) {
		return;
	}

	if (ConsumeMetadataColorOperation(buffer, args.render_target_slice_offset) ||
	    DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer, args.render_target_slice_offset)) {
		ResetBindings();
		return;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return;
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

	const DrawCallInfo draw {CommandBufferDebugOp::DrawIndexAuto, args.vertex_count,
	                         args.instance_count, args.first_instance};

	vk::PrimitiveTopology topology = vk::PrimitiveTopology::ePointList;
	if (!GetDrawTopology(ucfg, topology)) {
		ResetBindings();
		return;
	}
	DrawRenderState state {};
	if (!PrepareDrawRenderState(buffer, draw, args.render_target_slice_offset, state)) {
		ResetBindings();
		return;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.program->param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);

	const bool indirect = args.offset_source == DrawOffsetSource::IndirectArgs;
	const auto [vertex_offset, instance_offset] =
	    indirect ? std::pair<int32_t, uint32_t> {0, args.first_instance}
	             : ResolveDrawOffsets(ucfg.GetIndexOffset(), state.vertex_info[0]);
	DrawEmitInfo emit {};
	const bool native_indirect = indirect && !state.vertex_info[0].fetch_embedded;
	emit.first_vertex = native_indirect ? 0 :
	    static_cast<uint32_t>(vertex_offset + static_cast<int32_t>(args.first_vertex));
	emit.first_instance = native_indirect ? 0 : instance_offset;

	DrawIndexBufferSource index_source {};
	ExecutePreparedDraw(submit_id, buffer, draw, state, topology, emit, index_source, false);
	ResetBindings();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer,
                                         uint32_t       render_target_slice_offset) {
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
	ResolveRenderColorTarget(buffer, src, render_target_slice_offset, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, render_target_slice_offset, 1, true, true);
	if (!src.image_id || !dst.image_id) {
		return false;
	}
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source, {src.guest_mip_level, 1, src.guest_array_layer, 1},
	                    {dst.guest_mip_level, 1, dst.guest_array_layer, 1});
	return true;
}

} // namespace Libs::Graphics
