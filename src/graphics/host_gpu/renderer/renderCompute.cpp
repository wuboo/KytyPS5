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
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/gpuLabelsVk.h"
#include "graphics/host_gpu/renderer/computeMetaFill.h"
#include "graphics/host_gpu/renderer/image/imageInfo.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/narrowBarrierVk.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/cpSampler.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/rfGpuPatch.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <optional>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {
static bool FillSourcesDisjoint(std::span<const ShaderRecompiler::IR::DescriptorValue> sources,
                                 GuestRange destination, uint32_t output_buffer = UINT32_MAX) {
	for (uint32_t i = 0; i < sources.size(); ++i) {
		if (i == output_buffer) continue;
		const auto source = DecodeNativeDescriptor<ShaderBufferResource>(sources[i]);
		const auto bytes  = source.GetSize();
		if (source.Base48() < destination.End() && destination.address < source.Base48() + bytes)
			return false;
	}
	return true;
}

bool RenderExecutor::TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
                                                const CommandBuffer&          buffer) {
	const auto& program   = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	if (resources.buffers.size() != program.info.buffers.size()) {
		EXIT("compute runtime buffer count does not match shader metadata\n");
	}
	auto& cache = buffer.GetContext().GetTextureCache();
	for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
		const auto& resource   = program.info.buffers[i];
		const auto  descriptor = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
		// A metadata resource that is also read is not proven to be a full overwrite. Execute it
		// conservatively instead of replacing the dispatch with a coarse full-surface clear.
		if ((!resource.written || resource.read) && cache.IsMeta(descriptor.Base48())) {
			return false;
		}
	}

	if (!program.info.has_bitwise_xor) {
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& resource = program.info.buffers[i];
			if (resource.written) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
				if (cache.ClearMeta(descriptor.Base48())) {
					return true;
				}
			}
		}
	}
	return false;
}

bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                              uint32_t group_y, uint32_t group_z, uint32_t mode,
                              ShaderBufferResource& resolved_descriptor, uint32_t& resolved_clear,
                              uint64_t& resolved_size) {
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	if (fill.kind != ShaderRecompiler::IR::UniformFillKind::Buffer) {
		return false;
	}
	const auto element_size = fill.words * sizeof(uint32_t);
	const auto descriptor =
	    DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[fill.resource]);
	constexpr std::array formats {
	    Prospero::BufferFormat::k32UInt, Prospero::BufferFormat::k32_32UInt,
	    Prospero::BufferFormat::k32_32_32UInt, Prospero::BufferFormat::k32_32_32_32UInt};
	// Offset-or-swizzle bounds do not describe the element range proved by the shader.
	if (descriptor.Stride() != element_size || descriptor.Format() != formats[fill.words - 1] ||
	    descriptor.SwizzleEnabled() || descriptor.IndexStride() != 0 || descriptor.AddTid() ||
	    descriptor.OutOfBounds() == 3 || descriptor.Base48() == 0) {
		return false;
	}
	if (input.threads_num[0] == 0 || input.threads_num[0] != fill.group_stride[0] ||
	    input.threads_num[1] != 1 || input.threads_num[2] != 1 || group_x == 0 || group_y != 1 ||
	    group_z != 1 || mode != (input.dispatch_thread_dimensions ? 0x61u : 0x41u)) {
		return false;
	}
	const uint64_t invocations = input.dispatch_thread_dimensions
	                                 ? group_x
	                                 : static_cast<uint64_t>(group_x) * input.threads_num[0];
	const auto     size        = descriptor.GetSize();
	const auto elements = descriptor.NumRecords();
	const bool covered = fill.bounded_elements != 0
	                         ? fill.bounded_elements == elements && invocations >= elements
	                         : invocations == elements;
	if (!covered || invocations > uint64_t {UINT32_MAX} + 1 || size == 0 || size > UINT32_MAX ||
	    (input.dispatch_thread_dimensions &&
	     (group_x % input.threads_num[0] != 0 || input.dispatch_threads_num[0] != group_x ||
	      input.dispatch_threads_num[1] != 1 || input.dispatch_threads_num[2] != 1))) {
		return false;
	}
	if (!FillSourcesDisjoint(resources.buffers, {descriptor.Base48(), size}, fill.resource))
		return false;
	resolved_descriptor = descriptor;
	resolved_clear      = fill.value;
	resolved_size       = size;
	return true;
}

bool RenderExecutor::TryConsumeComputeFill(const ShaderComputeInputInfo& input,
                                          CommandBuffer& command, uint32_t group_x,
                                          uint32_t group_y, uint32_t group_z, uint32_t mode) {
	const auto& program   = input.stage.module->info;
	const auto& resources = *input.stage.resources;
	const auto& fill      = resources.uniform_fill;
	auto&       cache     = command.GetContext().GetTextureCache();
	if (fill.kind == ShaderRecompiler::IR::UniformFillKind::Image) {
		if (mode != 0x41u || input.dispatch_thread_dimensions || fill.value > 255 ||
		    input.threads_num[2] != 1)
			return false;
		const auto  descriptor = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[0]);
		const auto& resource   = program.info.images[0];
		if (descriptor.IsNull() || descriptor.Format() != Prospero::BufferFormat::k8UInt ||
		    descriptor.Type() != Prospero::ImageType::kColor2DArray || descriptor.MetaCompress() ||
		    descriptor.WriteCompress() || descriptor.BaseLevel() > descriptor.LastLevel() ||
		    descriptor.BaseLevel() > descriptor.MaxMip() ||
		    descriptor.BaseArray5() > descriptor.Depth() || descriptor.DstSelX() != 4)
			return false;
		const std::array extents {
		    std::max(1u, (descriptor.Width5() + 1u) >> descriptor.BaseLevel()),
		    std::max(1u, (descriptor.Height5() + 1u) >> descriptor.BaseLevel()),
		    descriptor.Depth() - descriptor.BaseArray5() + 1u};
		const std::array groups {group_x, group_y, group_z};
		for (uint32_t axis = 0; axis < 3; ++axis) {
			const uint64_t threads = input.threads_num[axis];
			// Guest image writes outside the descriptor dimensions are discarded. Only the
			// final workgroup may extend beyond the selected image view.
			if (threads == 0 || threads != fill.group_stride[axis] ||
			    groups[axis] != (extents[axis] + threads - 1) / threads ||
			    groups[axis] * threads > UINT32_MAX) return false;
		}
		const auto  binding     = ResolveTexture(resource, resources.images[0]);
		const auto& destination = binding.desc.info.data;
		if (!FillSourcesDisjoint(resources.buffers, destination)) return false;
		std::scoped_lock lock {cache.m_lock};
		const auto&      image = cache.GetImage(binding.image_id);
		const auto&      view  = binding.desc.view_info;
		if (image.backing.format != vk::Format::eD32SfloatS8Uint || image.info.samples != 1 ||
		    image.info.stencil != destination || view.base_level >= image.backing.mip_levels ||
		    view.base_layer >= image.backing.layers || view.layer_count != extents[2] ||
		    view.layer_count > image.backing.layers - view.base_layer ||
		    std::max(1u, image.info.extent.width >> view.base_level) != extents[0] ||
		    std::max(1u, image.info.extent.height >> view.base_level) != extents[1]) return false;
		const vk::ImageSubresourceRange range {vk::ImageAspectFlagBits::eStencil, view.base_level,
		                                       1, view.base_layer, view.layer_count};
		vk::ClearValue clear {};
		clear.depthStencil = vk::ClearDepthStencilValue {0.0f, fill.value};
		cache.ClearImage(command, binding.image_id, image.backing.format, range, clear);
		return true;
	}
	ShaderBufferResource descriptor;
	uint32_t             packed_clear = 0;
	uint64_t             size         = 0;
	if (!ResolveComputeBufferFill(input, group_x, group_y, group_z, mode, descriptor, packed_clear,
	                              size)) {
		return false;
	}
	const bool image_clear =
	    cache.ClearImageFromBuffer(command, descriptor.Base48(), size, packed_clear);
	if (!image_clear) {
		// No image lives there (DCC metadata, for one). KYTY_DCC_KNOWN=1: the dispatch only stores one value over the whole
		// buffer, which a transfer fill does without a compute pass, and the buffer cache then knows the bytes: reading DCC
		// clear codes needs no GPU drain. shadow: the dispatch stays; the buffer cache records the prediction.
		const auto known = DccFillPublish::EnvMode();
		auto& buffer_cache = command.GetContext().GetBufferCache();
		if (known != DccFillPublish::Mode::Off) {
			if ((descriptor.Base48() & 3u) != 0 || (size & 3u) != 0) {
				return false;
			}
			if (known == DccFillPublish::Mode::Shadow) {
				buffer_cache.NoteShadowFill(descriptor.Base48(), size, packed_clear);
				return false;
			}
			// Like the shader it replaces, write on the GPU: a CPU fill of a tracked range takes a write fault per page.
			buffer_cache.FillBuffer(descriptor.Base48(), size, packed_clear, false, true);
			FrameTiming::Add(FrameTiming::Counter::ComputeFillsReplaced);
			return true;
		}
		// Upstream (#1628), only for ranges our DCC_KNOWN path did not take (DCC_KNOWN=0). Preserve the native
		// dispatch for partial or ambiguous image backings, including pooled metadata aliases. The fill goes
		// through the GPU as well: a CPU fill of a range the guest tracks is a write fault per page.
		if ((descriptor.Base48() & 3u) != 0 || (size & 3u) != 0 ||
		    !m_context.IsMapped(descriptor.Base48(), size) ||
		    cache.IsRegionRegistered(descriptor.Base48(), size)) {
			return false;
		}
		buffer_cache.FillBuffer(descriptor.Base48(), size, packed_clear, false, true);
		FrameTiming::Add(FrameTiming::Counter::ComputeFillsReplaced);
	}
	static std::atomic<uint32_t> logged_clears {0};
	if (logged_clears.fetch_add(1, std::memory_order_relaxed) < 32) {
		LOGF("GraphicsRenderDispatchDirect: compute %s shader=0x%016" PRIx64
		     " addr=0x%016" PRIx64 " size=0x%016" PRIx64 " value=0x%08" PRIx32 "\n",
		     image_clear ? "image clear" : "buffer fill", input.stage.module->info.shader_hash,
		     descriptor.Base48(), size, packed_clear);
	}
	return true;
}

// Global loads read pages without a buffer as zero. Back the 32 KiB after each base the shader
// loads from scalar data before the dispatch reads it.
static void BackAddressBases(RenderContext& context, const ShaderStageRuntime& stage) {
	constexpr uint64_t Page = BufferCache::CACHING_PAGESIZE;
	const auto&        flat = stage.resources->flattened_srt;
	for (const auto [low, high]: stage.module->info.info.address_bases) {
		const auto base = flat[low] | uint64_t {flat[high]} << 32u;
		for (uint64_t index = 0; index <= 2; ++index) {
			const auto page = base + index * Page;
			if (context.IsMapped(page, 1)) (void)context.GetBufferCache().FindBuffer(page, 1);
		}
	}
}

static void BindSharedMemory(RenderContext& context, ShaderComputeInputInfo& input,
                             PreparedBindings& bindings, uint64_t indirect_args = 0) {
	if (input.stage.module->info.bindings.descriptor_counts[
	        static_cast<size_t>(ShaderRecompiler::IR::DescriptorBindingKind::SharedMemory)] == 0) {
		return;
	}
	auto& cache = context.GetBufferCache();
	if (indirect_args != 0) {
		cache.ReadMemory(indirect_args, sizeof(vk::DispatchIndirectCommand));
		std::memcpy(input.workgroup_counts, reinterpret_cast<const void*>(indirect_args),
		            sizeof(input.workgroup_counts));
	}
	// LDS has no contents to preserve between dispatches. The existing shader hazard
	// barriers also order other users of this GPU-only utility buffer.
	auto& storage = cache.GetUtilityBuffer(MemoryUsage::DeviceLocal);
	const auto limit = std::min<uint64_t>(storage.Size(),
	    context.GetGraphics().GetPhysicalDeviceProperties().limits.maxStorageBufferRange);
	uint64_t size = sizeof(uint32_t);
	if (std::ranges::find(input.workgroup_counts, 0u) == std::end(input.workgroup_counts)) {
		size = uint64_t {input.lds_size_dwords} * sizeof(uint32_t);
		EXIT_IF(size == 0 || size > limit);
		for (const auto count: input.workgroup_counts) {
			EXIT_IF(size > limit / count);
			size *= count;
		}
	}
	bindings.shared_memory = {storage.Handle(), 0, size};
}

static bool HasStorageImageWrites(const ShaderComputeInputInfo& input_info) {
	const auto& images = input_info.stage.module->info.info.images;
	return std::ranges::any_of(images, [](const auto& image) {
		return image.written && image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Storage;
	});
}

static bool HasStorageWrites(const ShaderComputeInputInfo& input_info,
                             const PreparedBindings&       bindings) {
	return bindings.shared_memory.buffer != nullptr || HasShaderBufferWrites(input_info.stage) ||
	       HasStorageImageWrites(input_info);
}

// S556 (KYTY_NARROW_BARRIER, narrowBarrier.h): the barriers around a guest dispatch. Level 0 records exactly the old pair.
// Before (only when the dispatch writes): level 1 narrows it when the dispatch writes no storage image (buffers, LDS and
// address writes only): every stage that can touch a buffer stays in the execution dependency (WAR), the buffer writes
// stay in the memory dependency (WAW); colour/depth output and their caches cannot hold this memory. The dispatch's own
// reads of earlier image writes are ordered by Image::Transit, as for a read-only dispatch (which has no barrier here).
// S565: Command is vk::CommandBuffer or the KYTY_CP_EMIT IR handle CpEmit::Cmd.
template <class Command>
static void DispatchWriteHazard(const Command& vk_buffer, bool image_writes) {
	FrameTiming::Add(FrameTiming::Counter::DispHazardN);
	if (image_writes) {
		FrameTiming::Add(FrameTiming::Counter::DispHazardImgN);
	}
	if (NarrowBarrier::NarrowWriteHazard(NarrowBarrier::Level(), image_writes)) {
		NarrowBarrier::WriteHazard(vk_buffer, vk::PipelineStageFlagBits2::eComputeShader);
		FrameTiming::Add(FrameTiming::Counter::NarrowBarrierN);
	} else {
		ShaderWriteHazardBarrierTo(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
	}
}

// After: a writing dispatch keeps the full barrier (its writes may be read by any later command, an image it wrote may
// become an attachment). After a read-only dispatch the barrier only orders its reads before later writers (WAR): no
// write is in its first scope, so level 2 records it without access masks (execution dependency only).
template <class Command>
static void DispatchAfter(const Command& vk_buffer, bool writes) {
	if (!writes) {
		FrameTiming::Add(FrameTiming::Counter::DispAccessRoN);
		if (NarrowBarrier::ExecOnlyAfterReadOnlyDispatch(NarrowBarrier::Level())) {
			vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eAllCommands,
			                          vk::DependencyFlags {}, 0, nullptr, 0, nullptr, 0, nullptr);
			FrameTiming::Add(FrameTiming::Counter::NarrowBarrierN);
			return;
		}
	}
	ShaderAccessBarrierTo(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
}

void RenderExecutor::DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x,
                                    uint32_t thread_group_y, uint32_t thread_group_z, uint32_t mode,
                                    bool async_compute) {
	CpPhases::Scope dispatch_phase(CpPhases::Phase::Dispatch);
	EXIT_IF(buffer.IsInvalid());
	FrameTiming::Add(FrameTiming::Counter::Dispatches);
	m_context.GetCommandScheduler().MaybeEagerSplit();
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& sh_ctx = buffer.GetShaders();

	if (thread_group_x == 0 || thread_group_y == 0 || thread_group_z == 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: skipping zero-sized dispatch groups=%ux%ux%u "
			     "mode=0x%08" PRIx32 " shader=0x%016" PRIx64 "\n",
			     thread_group_x, thread_group_y, thread_group_z, mode,
			     sh_ctx.GetCs().cs_regs.data_addr);
		}
		return;
	}

	Common::LockGuard lock(m_context.GetMutex());
	if (sh_ctx.GetCs().cs_regs.data_addr == 0) {
		LOGF("GraphicsRenderDispatchDirect: temporary: ignoring dispatch with null CS shader, "
		     "groups=%ux%ux%u mode=%u\n",
		     thread_group_x, thread_group_y, thread_group_z, mode);
		return;
	}

	constexpr uint32_t DISPATCH_INITIATOR_BASE_BITS     = 0x41u;
	constexpr uint32_t DISPATCH_INITIATOR_MODIFIER_BITS = 0xa038u;
	constexpr uint32_t DISPATCH_INITIATOR_KNOWN_MASK =
	    DISPATCH_INITIATOR_BASE_BITS | DISPATCH_INITIATOR_MODIFIER_BITS;

	const uint32_t unknown_mode_bits = mode & ~DISPATCH_INITIATOR_KNOWN_MASK;
	if (unknown_mode_bits != 0) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 32) {
			LOGF("GraphicsRenderDispatchDirect: unknown dispatch initiator bits "
			     "mode=0x%08" PRIx32 " unknown=0x%08" PRIx32 " shader=0x%016" PRIx64
			     " groups=%ux%ux%u\n",
			     mode, unknown_mode_bits, sh_ctx.GetCs().cs_regs.data_addr, thread_group_x,
			     thread_group_y, thread_group_z);
		}
	}

	const auto& cs_regs = sh_ctx.GetCs();

	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	const bool use_thread_dimensions =
	    (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	input_info.dispatch_thread_dimensions = use_thread_dimensions;
	input_info.workgroup_counts[0]        = thread_group_x;
	input_info.workgroup_counts[1]        = thread_group_y;
	input_info.workgroup_counts[2]        = thread_group_z;
	if (use_thread_dimensions) {
		const uint32_t group_sizes[] = {cs_regs.cs_regs.num_thread_x, cs_regs.cs_regs.num_thread_y,
		                                cs_regs.cs_regs.num_thread_z};
		for (uint32_t axis = 0; axis < 3u; ++axis) {
			const auto size = std::max(group_sizes[axis], 1u);
			const auto threads = input_info.workgroup_counts[axis];
			input_info.workgroup_counts[axis] = threads / size + (threads % size != 0u);
		}
	}
	m_context.GetImageHeap().BeginDraw();
	if (!m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info)) return;
	if (use_thread_dimensions) {
		input_info.dispatch_threads_num[0]    = thread_group_x;
		input_info.dispatch_threads_num[1]    = thread_group_y;
		input_info.dispatch_threads_num[2]    = thread_group_z;
	}

	const auto& program   = input_info.stage.module->info;
	RfOp::WriterHashGuard cs_writer_hash(static_cast<uint64_t>(program.shader_hash));
	const auto& resources = *input_info.stage.resources;
	if (resources.specialization_reads.empty()) {
		// The clear must not skip the fill: the fill is what the ledger records (S491).
		const bool meta = TryConsumeComputeMetaClear(input_info, buffer);
		bool       fill = false;
		if (RunUniformFillBesideMetaClear(meta)) {
			fill = TryConsumeComputeFill(input_info, buffer, thread_group_x, thread_group_y,
			                             thread_group_z, mode);
		}
		if (meta || fill) {
			ResetBindings();
			return;
		}
	}

	if (use_thread_dimensions) {
		thread_group_x = input_info.workgroup_counts[0];
		thread_group_y = input_info.workgroup_counts[1];
		thread_group_z = input_info.workgroup_counts[2];
	}

	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	if (program.bindings.dispatch_thread_dword != ShaderRecompiler::IR::PushData::NoStart) {
		std::copy(std::begin(input_info.dispatch_threads_num), std::end(input_info.dispatch_threads_num),
		          bindings.shader_data.begin() + program.bindings.dispatch_thread_dword);
	}
	PreparedBindings* descriptor_stage = &bindings;
	BackAddressBases(m_context, input_info.stage);
	FindBuffers(std::span {&descriptor_stage, 1u});
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	BindSharedMemory(m_context, input_info, bindings);
	RebindBuffers(bindings);

	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
	               std::span {&descriptor_stage, 1u});
	// S561 (KYTY_CP_EMIT): the handle is taken after CommitBindings (an IR site). S565: it follows the IR when the stream holds commands
	// (no drain: RESULT-S563 had 221 of the 292 handle drains per flip here), else records straight on.
	const auto vk_buffer = buffer.EmitIfPending();
	const bool storage_writes = HasStorageWrites(input_info, bindings);
	if (storage_writes) {
		// A host fence used to serialize every dispatch. Preserve its read-before-write ordering
		// while allowing the queue to execute asynchronously.
		DispatchWriteHazard(vk_buffer, HasStorageImageWrites(input_info));
	}
	if (Config::GraphicsDebugDumpEnabled()) {
		const auto sampled_images = std::count_if(
		    program.info.images.begin(), program.info.images.end(), [](const auto& image) {
			    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
		    });
		const uint32_t frame_num = static_cast<uint32_t>(m_context.GetGpu().GetFrameNum());
		LOGF("GraphicsRenderDispatchDirect: frame=%u shader=0x%016" PRIx64
		     " hash=0x%016" PRIx64 " tick=%" PRIu64 " completed_tick=%" PRIu64
		     " groups=%ux%ux%u mode=0x%08" PRIx32 " local=%ux%ux%u "
		     "buffers=%zu textures=%zu sampled=%zu storage=%zu samplers=%zu push=%u\n",
		     frame_num, sh_ctx.GetCs().cs_regs.data_addr, program.shader_hash,
		     m_context.GetCommandScheduler().CurrentTick(),
		     m_context.GetCommandScheduler().GetMasterSemaphore().KnownGpuTick(),
		     thread_group_x, thread_group_y, thread_group_z, mode,
		     input_info.threads_num[0], input_info.threads_num[1],
		     input_info.threads_num[2], program.info.buffers.size(), program.info.images.size(),
		     sampled_images, program.info.images.size() - sampled_images,
		     program.info.samplers.size(),
		     program.bindings.UsesPushData()
		         ? static_cast<uint32_t>(sizeof(ShaderRecompiler::IR::PushData))
		         : 0u);
		for (uint32_t i = 0; i < program.info.buffers.size(); i++) {
			const auto& buffer = program.info.buffers[i];
			const auto  r      = DecodeNativeDescriptor<ShaderBufferResource>(resources.buffers[i]);
			LOGF("  CS buffer[%u]: source=%u usage=%s addr=0x%012" PRIx64
			     " stride=%u records=%u format=%u\n",
			     i, buffer.source, buffer.written ? "read-write" : "read-only", r.Base48(),
			     r.Stride(), r.NumRecords(), r.RawFormat());
		}
		for (uint32_t i = 0; i < program.info.images.size(); i++) {
			const auto& image = program.info.images[i];
			const auto  r     = DecodeNativeDescriptor<ShaderTextureResource>(resources.images[i]);
			const auto  max_mip = image.r128 ? r.LastLevel() : r.MaxMip();
			LOGF("  CS texture[%u]: source=%u usage=%s sampled=%s addr=0x%010" PRIx64
			     " type=%u fmt=%u extent=%ux%u depth=%u levels=%u tile=%u\n",
			     i, image.source, image.written ? "read-write" : "read-only",
			     image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled
			         ? "true"
			         : "false",
			     r.Base40(), static_cast<uint32_t>(r.Type()), static_cast<uint32_t>(r.Format()),
			     static_cast<uint32_t>(r.Width5()) + 1u, static_cast<uint32_t>(r.Height5()) + 1u,
			     static_cast<uint32_t>(r.Depth()) + 1u,
			     r.Type() == Prospero::ImageType::kColor2DMsaa ||
			             r.Type() == Prospero::ImageType::kColor2DMsaaArray
			         ? 1u
			         : static_cast<uint32_t>(max_mip) + 1u,
			     static_cast<uint32_t>(r.TileMode()));
		}
		for (uint32_t i = 0; i < program.info.samplers.size(); i++) {
			const auto r = DecodeNativeDescriptor<ShaderSamplerResource>(
			    resources.samplers[program.info.samplers[i].snapshot_index]);
			LOGF("  CS sampler[%u]: source=%u clamp=%u/%u/%u filter=%u/%u/%u mip=%u "
			     "lod=%u-%u bias=%d\n",
			     i, program.info.samplers[i].source, static_cast<uint32_t>(r.ClampX()),
			     static_cast<uint32_t>(r.ClampY()), static_cast<uint32_t>(r.ClampZ()),
			     static_cast<uint32_t>(r.XyMagFilter()), static_cast<uint32_t>(r.XyMinFilter()),
			     static_cast<uint32_t>(r.ZFilter()), static_cast<uint32_t>(r.MipFilter()),
			     static_cast<uint32_t>(r.MinLod()), static_cast<uint32_t>(r.MaxLod()),
			     static_cast<int32_t>(r.LodBias()));
		}
	}

	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	constexpr uint32_t DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE = 1u << 3u;
	const uint64_t     group_threads = uint64_t {std::max(input_info.threads_num[0], 1u)} *
	                                   std::max(input_info.threads_num[1], 1u) *
	                                   std::max(input_info.threads_num[2], 1u);
	if ((mode & DISPATCH_INITIATOR_ORDERED_APPEND_ENABLE) != 0 && program.info.uses_gds &&
	    group_threads <= input_info.wave_size && !input_info.lds_storage) {
		// A single guest wave per group can run in launch order without a device-wide
		// spinlock. Keep the guest workgroup IDs when splitting the dispatch.
		for (uint32_t z = 0; z < thread_group_z; ++z) {
			for (uint32_t y = 0; y < thread_group_y; ++y) {
				for (uint32_t x = 0; x < thread_group_x; ++x) {
					if ((x | y | z) != 0u) {
						ShaderAccessBarrierTo(vk_buffer, vk::PipelineStageFlagBits::eComputeShader);
					}
					vk_buffer.dispatchBase(x, y, z, 1, 1, 1);
				}
			}
		}
	} else {
		vk_buffer.dispatch(thread_group_x, thread_group_y, thread_group_z);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	FrameTiming::NoteDispatch(FrameTiming::DispatchClass::Guest);

	// The removed host fence also ordered read-only dispatches before later writers.
	DispatchAfter(vk_buffer, storage_writes);
	GpuLabel::NoteCause(GpuLabel::Cause::Cs);
	ResetBindings();
}

// S570 (KYTY_RF_GPU_PATCH, rfGpuPatch.h) shadow: drain each deferred dword, walk the program again with the fresh bytes and compare the
// snapshots: equal except the patched slots, which must now hold the fresh dwords (rf_patch_ne_n otherwise). The second walk's snapshot is
// the one the dispatch uses, so shadow never draws with a stale value.
static bool RfPatchShadow(RenderContext& context, ShaderComputeInputInfo& input_info, const HW::ComputeShaderInfo& cs_regs) {
	namespace P = ShaderRecompiler::IR::RfGpuPatch;
	auto&       pending = P::ThreadPending();
	const auto* module  = input_info.stage.module;
	const auto  first   = *input_info.stage.resources; // copy: the second walk rewrites the program's snapshot
	std::array<uint32_t, P::kMaxEntries> fresh {};
	for (uint32_t i = 0; i < pending.count; ++i) {
		const auto& e = pending.entries[i];
		(void)context.GetBufferCache().ReadMemory(e.address, sizeof(uint32_t));
		if (!LibKernel::Memory::TryReadBacking(e.address, &fresh[i], sizeof(uint32_t))) fresh[i] = e.stale;
		if (fresh[i] != e.stale) FrameTiming::Add(FrameTiming::Counter::RfPatchStaleNeN);
	}
	FrameTiming::Add(FrameTiming::Counter::RfPatchShadowN);
	if (!context.GetPipelineCache().GetComputeProgram(cs_regs, input_info)) {
		FrameTiming::Add(FrameTiming::Counter::RfPatchNeN);
		return false;
	}
	const auto& second = *input_info.stage.resources;
	bool        agree  = input_info.stage.module == module && first.buffers == second.buffers && first.images == second.images &&
	             first.samplers == second.samplers && first.uniform_fill == second.uniform_fill &&
	             first.specialization_reads == second.specialization_reads &&
	             P::SnapshotsAgree(first.flattened_srt, second.flattened_srt, pending);
	for (uint32_t i = 0; agree && i < pending.count; ++i) {
		const auto& e = pending.entries[i];
		for (uint32_t k = 0; k < e.slot_count; ++k) {
			agree &= e.slots[k] < second.flattened_srt.size() && second.flattened_srt[e.slots[k]] == fresh[i];
		}
	}
	if (!agree) {
		FrameTiming::Add(FrameTiming::Counter::RfPatchNeN);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("[rf-patch] shadow: the walk with the fresh bytes differs beyond the patched slots (shader=0x%016" PRIx64 " entries=%u)\n",
			     module != nullptr ? static_cast<uint64_t>(module->info.shader_hash) : 0ull, pending.count);
		}
	}
	return true;
}

// S570: the GPU bytes of each deferred dword into its slots of the uploaded flattened SRT, before the dispatch reads them.
template <class Command>
static void RfPatchRecord(const Command& vk_buffer, std::span<const std::pair<vk::Buffer, uint64_t>> sources,
                          const vk::DescriptorBufferInfo& flat) {
	namespace P = ShaderRecompiler::IR::RfGpuPatch;
	const auto& pending = P::ThreadPending();
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                          nullptr);
	uint32_t copies = 0;
	for (uint32_t i = 0; i < pending.count && i < sources.size(); ++i) {
		const auto& e = pending.entries[i];
		if (e.slot_count == 0) continue; // S579: a superset load that only reaches conditions has nothing to copy
		std::array<vk::BufferCopy, P::kMaxSlots> regions {};
		for (uint32_t k = 0; k < e.slot_count; ++k) {
			regions[k] = vk::BufferCopy(sources[i].second, flat.offset + uint64_t {e.slots[k]} * sizeof(uint32_t), sizeof(uint32_t));
		}
		vk_buffer.copyBuffer(sources[i].first, flat.buffer, e.slot_count, regions.data());
		copies += e.slot_count;
	}
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eUniformRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, 1, &after, 0, nullptr, 0,
	                          nullptr);
	FrameTiming::Add(FrameTiming::Counter::RfPatchCopyN, copies);
	FrameTiming::Add(FrameTiming::Counter::RfPatchDispatchN);
}

// S579 K-rf v2 (KYTY_RF_GPU_PATCH=superset_shadow, rfGpuPatch.h): the superset walk is the one GetComputeProgram just did (deferred loads
// unknown). Drain each deferred dword, walk again with the fresh bytes (the true walk: the scope is disarmed) and check that the superset holds
// every resource, flat word and read range of the true walk unchanged (the extra ones are counted), the patched slots aside (their drained dword
// is in `fresh`; the GPU copy is checked by Krf2GpuCheck). On agreement the dispatch goes on with the superset snapshot and module (the
// production path, its bindings and copies); a disagreement counts rf_patch_ne_n and the dispatch uses the true walk.
static bool Krf2Shadow(RenderContext& context, ShaderComputeInputInfo& input_info, const HW::ComputeShaderInfo& cs_regs,
                       std::array<uint32_t, ShaderRecompiler::IR::RfGpuPatch::kMaxEntries>& fresh) {
	namespace P = ShaderRecompiler::IR::RfGpuPatch;
	auto&       pending = P::ThreadPending();
	const auto* module  = input_info.stage.module;
	auto        first   = *input_info.stage.resources; // copy: the true walk rewrites the program's snapshot
	for (uint32_t i = 0; i < pending.count; ++i) {
		const auto& e = pending.entries[i];
		(void)context.GetBufferCache().ReadMemory(e.address, sizeof(uint32_t));
		if (!LibKernel::Memory::TryReadBacking(e.address, &fresh[i], sizeof(uint32_t))) fresh[i] = e.stale;
		if (fresh[i] != e.stale) FrameTiming::Add(FrameTiming::Counter::RfPatchStaleNeN);
	}
	FrameTiming::Add(FrameTiming::Counter::RfPatchShadowN);
	if (!context.GetPipelineCache().GetComputeProgram(cs_regs, input_info)) {
		FrameTiming::Add(FrameTiming::Counter::RfPatchNeN);
		return false;
	}
	const auto&           truth = *input_info.stage.resources;
	std::vector<uint32_t> extra_buffers;
	const auto            buffers  = P::SupersetCoverResources(std::span(truth.buffers), std::span<const ShaderRecompiler::IR::DescriptorValue>(first.buffers), &extra_buffers);
	const auto            images   = P::SupersetCoverResources(std::span(truth.images), std::span<const ShaderRecompiler::IR::DescriptorValue>(first.images));
	const auto            samplers = P::SupersetCoverResources(std::span(truth.samplers), std::span<const ShaderRecompiler::IR::DescriptorValue>(first.samplers));
	const auto            flat     = P::SupersetCoverFlat(truth.flattened_srt, first.flattened_srt, pending, std::span(fresh).first(pending.count));
	const bool            ranges   = P::SupersetCoverRanges(std::span(truth.specialization_reads),
	                                                        std::span<const std::pair<uint64_t, uint64_t>>(first.specialization_reads));
	const bool agree = buffers.agree && images.agree && samplers.agree && flat.agree && ranges && truth.uniform_fill == first.uniform_fill &&
	                   truth.user_data == first.user_data;
	FrameTiming::Add(FrameTiming::Counter::Krf2ExtraResN, buffers.extra + images.extra + samplers.extra);
	FrameTiming::Add(FrameTiming::Counter::Krf2ExtraFlatN, flat.extra);
	if (module != nullptr) {
		uint32_t written = 0;
		for (const auto i: extra_buffers) {
			if (i < module->info.info.buffers.size() && module->info.info.buffers[i].written) ++written;
		}
		FrameTiming::Add(FrameTiming::Counter::Krf2ExtraWrN, written);
	}
	if (input_info.stage.module != module) FrameTiming::Add(FrameTiming::Counter::Krf2VariantN);
	if (!agree) {
		FrameTiming::Add(FrameTiming::Counter::RfPatchNeN);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("[krf2] shadow: the superset misses a resource of the true walk (shader=0x%016" PRIx64
			     " entries=%u buffers=%d images=%d samplers=%d flat=%d ranges=%d)\n",
			     module != nullptr ? static_cast<uint64_t>(module->info.shader_hash) : 0ull, pending.count, buffers.agree ? 1 : 0,
			     images.agree ? 1 : 0, samplers.agree ? 1 : 0, flat.agree ? 1 : 0, ranges ? 1 : 0);
		}
		return true; // the true walk's snapshot and module stay
	}
	// The superset again: the snapshot the stage points to is the program's own (non-const) one.
	*const_cast<ShaderRecompiler::IR::ResourceSnapshot*>(input_info.stage.resources) = std::move(first);
	input_info.stage.module                                                             = module;
	return true;
}

// S579 superset_shadow: after RfPatchRecord, the patched slots of the uploaded SRT go to the download ring; once the GPU is done with the
// submission, each must equal the dword the shadow drained (krf2_gpu_check_n words checked, rf_patch_ne_n otherwise).
template <class Command>
static void Krf2GpuCheck(RenderContext& context, const Command& vk_buffer, const vk::DescriptorBufferInfo& flat,
                         const std::array<uint32_t, ShaderRecompiler::IR::RfGpuPatch::kMaxEntries>& fresh) {
	namespace P         = ShaderRecompiler::IR::RfGpuPatch;
	const auto& pending = P::ThreadPending();
	std::array<vk::BufferCopy, P::kMaxEntries * P::kMaxSlots> regions {};
	std::array<uint32_t, P::kMaxEntries * P::kMaxSlots>       expected {};
	uint32_t                                                  n = 0;
	auto&      stream          = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Download);
	uint32_t   words           = 0;
	for (uint32_t i = 0; i < pending.count; ++i) words += pending.entries[i].slot_count;
	if (words == 0) return;
	auto [mapped, offset] = stream.Map(uint64_t {words} * sizeof(uint32_t), sizeof(uint32_t), false);
	if (mapped == nullptr) return;
	stream.Commit();
	for (uint32_t i = 0; i < pending.count; ++i) {
		const auto& e = pending.entries[i];
		for (uint32_t k = 0; k < e.slot_count; ++k, ++n) {
			regions[n]  = vk::BufferCopy(flat.offset + uint64_t {e.slots[k]} * sizeof(uint32_t), offset + uint64_t {n} * sizeof(uint32_t),
                                        sizeof(uint32_t));
			expected[n] = fresh[i];
		}
	}
	vk::MemoryBarrier before {};
	before.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	before.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer, {}, 1, &before, 0, nullptr, 0,
	                          nullptr);
	vk_buffer.copyBuffer(flat.buffer, stream.Handle(), n, regions.data());
	vk::MemoryBarrier after {};
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eHost, {}, 1, &after, 0, nullptr, 0, nullptr);
	context.GetCommandScheduler().DeferOperation([&stream, mapped, offset, n, expected] {
		stream.Invalidate(offset, uint64_t {n} * sizeof(uint32_t));
		uint32_t bad = 0;
		for (uint32_t i = 0; i < n; ++i) {
			uint32_t word = 0;
			std::memcpy(&word, mapped + uint64_t {i} * sizeof(uint32_t), sizeof(word));
			if (word != expected[i]) ++bad;
		}
		FrameTiming::Add(FrameTiming::Counter::Krf2GpuCheckN, n);
		if (bad != 0) {
			FrameTiming::Add(FrameTiming::Counter::RfPatchNeN);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
				LOGF("[krf2] shadow: %u of %u copied slot words differ from the drained dwords\n", bad, n);
			}
		}
	});
}

void RenderExecutor::DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
                                      bool async_compute) {
	CpPhases::Scope dispatch_phase(CpPhases::Phase::Dispatch);
	EXIT_IF(buffer.IsInvalid() || args_addr == 0 || (args_addr & 3u) != 0 ||
	        (mode & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0);
	FrameTiming::Add(FrameTiming::Counter::Dispatches);
	m_context.GetCommandScheduler().MaybeEagerSplit();
	m_context.GetCommandScheduler().PopPendingOperations();
	Common::LockGuard lock(m_context.GetMutex());
	vk::DispatchIndirectCommand args {};
	// S418: a copy of the args still in flight leaves the tracker clear and the backing stale (zeros from the previous use).
	int args_refusal = 0;
	const bool args_readable = m_context.GetBufferCache().CanReadBacking(args_addr, sizeof(args), &args_refusal);
	if (args_refusal == 2) {
		FrameTiming::Add(FrameTiming::Counter::DispIndirectBackingInflightN);
	}
	if (args_readable &&
	    LibKernel::Memory::TryReadBacking(args_addr, &args, sizeof(args)) &&
	    (args.x == 0 || args.y == 0 || args.z == 0)) {
		return;
	}
	const auto& cs_regs = buffer.GetShaders().GetCs();
	if (cs_regs.cs_regs.data_addr == 0) {
		return;
	}
	ShaderComputeInputInfo input_info {};
	input_info.async_compute = async_compute;
	input_info.indirect_args_address = args_addr;
	m_context.GetImageHeap().BeginDraw();
	// S570 (KYTY_RF_GPU_PATCH): the walks of this program may defer GPU-dirty dwords that only reach flattened SRT slots (rfGpuPatch.h).
	namespace RfPatch      = ShaderRecompiler::IR::RfGpuPatch;
	const auto rf_patch    = RfPatch::EnvMode();
	RfPatch::Scope rf_scope(rf_patch != RfPatch::Mode::Off);
	RfPatch::DiagDispatchArgs() = args_addr; // S573 DIAG (KYTY_DIAG_RF_WHY): which DISPATCH_INDIRECT the walker faults belong to
	const bool rf_program = m_context.GetPipelineCache().GetComputeProgram(cs_regs, input_info);
	RfPatch::DiagDispatchArgs() = 0;
	if (!rf_program) return;
	rf_scope.Disarm();
	const auto& rf_pending = RfPatch::ThreadPending();
	const bool  rf_patched = rf_patch != RfPatch::Mode::Off && rf_pending.count != 0;
	if (rf_patched && rf_patch == RfPatch::Mode::Shadow && !RfPatchShadow(m_context, input_info, cs_regs)) return;
	// S579 K-rf v2: superset entries that only reached conditions have no slot: no copy, no barrier.
	uint32_t rf_copy_slots = 0;
	for (uint32_t i = 0; rf_patched && i < rf_pending.count; ++i) rf_copy_slots += rf_pending.entries[i].slot_count;
	std::array<uint32_t, RfPatch::kMaxEntries> krf2_fresh {};
	if (rf_patched && rf_patch == RfPatch::Mode::SupersetShadow && !Krf2Shadow(m_context, input_info, cs_regs, krf2_fresh)) return;
	buffer.EndRendering();
	auto& pipeline = m_context.GetPipelineCache().GetComputePipeline(input_info);
	auto& bindings = m_compute_bindings;
	PrepareBindings(input_info.stage, bindings);
	PreparedBindings* descriptor_stage = &bindings;
	BackAddressBases(m_context, input_info.stage);
	FindBuffers(std::span {&descriptor_stage, 1u});
	const auto& program = input_info.stage.module->info;
	RfOp::WriterHashGuard cs_writer_hash(static_cast<uint64_t>(program.shader_hash));
	if (program.info.uses_dma) {
		m_context.PrepareBda();
	}
	BindSharedMemory(m_context, input_info, bindings, args_addr);
	if (program.info.uses_image_heap) {
		m_context.GetImageHeap().Rebind(*this);
	}
	RebindImages(bindings);
	// S570: the patch sources first (a buffer they create or merge cannot replace the arguments' buffer afterwards), then again after the
	// arguments (they exist now: no merge).
	std::array<std::pair<vk::Buffer, uint64_t>, RfPatch::kMaxEntries> rf_sources {};
	if (rf_copy_slots != 0) {
		for (uint32_t i = 0; i < rf_pending.count; ++i) {
			if (rf_pending.entries[i].slot_count == 0) continue;
			(void)m_context.GetBufferCache().ObtainBuffer(rf_pending.entries[i].address, sizeof(uint32_t), false);
		}
	}
	// Acquiring arguments can merge cache buffers; finalize shader bindings afterward.
	const auto [args_buffer, args_offset] = m_context.GetBufferCache().ObtainBuffer(
	    args_addr, sizeof(vk::DispatchIndirectCommand), false);
	EXIT_IF((args_offset & 3u) != 0);
	if (rf_copy_slots != 0) {
		for (uint32_t i = 0; i < rf_pending.count; ++i) {
			if (rf_pending.entries[i].slot_count == 0) continue;
			const auto [source, offset] = m_context.GetBufferCache().ObtainBuffer(rf_pending.entries[i].address, sizeof(uint32_t), false);
			rf_sources[i]               = {source->Handle(), offset};
		}
	}
	RebindBuffers(bindings);
	{
		KYTY_CP_SITE(DispDesc);
		CommitBindings(buffer, vk::PipelineBindPoint::eCompute, pipeline,
		               std::span {&descriptor_stage, 1u});
	}
	KYTY_CP_SITE(DispRecord);
	const auto vk_buffer = buffer.EmitIfPending(); // S565: as DispatchDirect
	if (rf_copy_slots != 0) {
		EXIT_IF(!bindings.flattened_srt.buffer);
		RfPatchRecord(vk_buffer, std::span(rf_sources.data(), rf_pending.count), bindings.flattened_srt);
		if (rf_patch == RfPatch::Mode::SupersetShadow) Krf2GpuCheck(m_context, vk_buffer, bindings.flattened_srt, krf2_fresh);
	}
	const bool storage_writes = HasStorageWrites(input_info, bindings);
	if (storage_writes) {
		DispatchWriteHazard(vk_buffer, HasStorageImageWrites(input_info));
	}
	if (NarrowBarrier::NarrowBufferSync(NarrowBarrier::Level())) {
		// S556: the arguments are buffer memory, so only shader stores and copy/fill/update can have written them;
		// eAllGraphics as the source also waited for colour/depth output, which cannot write a buffer.
		vk::MemoryBarrier2 narrow {};
		narrow.srcStageMask  = NarrowBarrier::kBufferWriteStages;
		narrow.srcAccessMask = NarrowBarrier::kBufferWrites;
		narrow.dstStageMask  = vk::PipelineStageFlagBits2::eDrawIndirect;
		narrow.dstAccessMask = vk::AccessFlagBits2::eIndirectCommandRead;
		NarrowBarrier::Record(vk_buffer, narrow);
		FrameTiming::Add(FrameTiming::Counter::NarrowBarrierN);
	} else {
		vk::MemoryBarrier barrier {};
		barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eTransferWrite;
		barrier.dstAccessMask = vk::AccessFlagBits::eIndirectCommandRead;
		vk_buffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics |
		                              vk::PipelineStageFlagBits::eComputeShader |
		                              vk::PipelineStageFlagBits::eTransfer,
		                          vk::PipelineStageFlagBits::eDrawIndirect, {},
		                          1, &barrier, 0, nullptr, 0, nullptr);
	}
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline.pipeline);
	GpuLabel::With(
	    vk_buffer, GpuLabel::Class::GuestCs, [&] { vk_buffer.dispatchIndirect(args_buffer->Handle(), args_offset); },
	    "G.cs.%08x.a%d.ind", static_cast<uint32_t>(program.shader_hash), async_compute ? 1 : 0);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::Dispatches);
	FrameTiming::NoteDispatch(FrameTiming::DispatchClass::Guest);
	DispatchAfter(vk_buffer, storage_writes);
	GpuLabel::NoteCause(GpuLabel::Cause::Cs);
	ResetBindings();
}

} // namespace Libs::Graphics
