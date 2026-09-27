// Live PerVertex execution with GPU compute unpack and native C++ pipeline.
void RenderExecutor::ExecutePerVertexPrototype(uint64_t submit_id, CommandBuffer& buffer,
                                               const DrawCallInfo& draw, DrawRenderState& state,
                                               vk::PrimitiveTopology        topology,
                                               const DrawEmitInfo&          emit,
                                               const DrawIndexBufferSource& index_source,
                                               bool primitive_restart_enable) {
	const auto& vs       = state.vertex_info[0];
	const auto* programs = GetPerVertexPrototypePrograms(
	    m_context.GetGraphics(), state.programs, vs, m_context.GetPipelineCache().DriverCache());
	EXIT_IF(programs == nullptr || !state.ps_active ||
	        topology != vk::PrimitiveTopology::eTriangleList || primitive_restart_enable ||
	        draw.index_count % 3 != 0);
	EXIT_IF(vs.logical_stage != ShaderType::Vertex || vs.buffers_num != 1 ||
	        vs.resources_num == 0 || vs.resources_num > 12);
	const bool capture_has_writes = HasShaderBufferWrites(vs.stage);
	EXIT_IF(capture_has_writes && !programs->native_capture);
	const auto&    input     = vs.buffers[0];
	const uint32_t num_attrs = vs.resources_num;
	EXIT_IF(input.stride == 0 || input.fetch_index != 0 ||
	        input.attr_num != static_cast<int>(num_attrs) || input.addr == 0);

	for (uint32_t i = 0; i < num_attrs; ++i) {
		EXIT_IF(input.attr_indices[i] != static_cast<int>(i));
		const auto& descriptor = vs.resources[i];
		const auto  format     = ShaderRecompiler::Format::GetFormatInfo(descriptor.Format());
		EXIT_IF(format.byte_size > input.stride ||
		        input.attr_offsets[i] > input.stride - format.byte_size ||
		        descriptor.SwizzleEnabled() || descriptor.AddTid());
		std::array<uint8_t, 16> bytes {};
		std::array<uint32_t, 4> decoded {};
		EXIT_IF(!DecodePerVertexPrototypeAttribute(descriptor.Format(), bytes, decoded));
	}

	const uint64_t count64 = static_cast<uint64_t>(draw.index_count) * draw.instance_count;
	// The non-native capture dispatches one workgroup per vertex, bounded by the 65535-group
	// dispatch limit; the native capture packs 64 vertices per group. Skip a draw past the limit
	// instead of exiting (it renders nothing, the rest of the frame continues).
	const uint64_t max_count = programs->native_capture ? (uint64_t {1} << 24u) : 65535u;
	if (count64 == 0 || count64 > max_count) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("PerVertexPrototype: skipping draw with %llu vertices (limit %llu, native=%d)\n",
			     static_cast<unsigned long long>(count64),
			     static_cast<unsigned long long>(max_count), programs->native_capture ? 1 : 0);
		}
		return;
	}
	const auto count     = static_cast<uint32_t>(count64);
	// Diagnostic: this path costs orders of magnitude more per draw than a native vertex stage
	// (a compute unpack + a compute capture + a graphics replay, each with its own barrier and,
	// because Vulkan forbids vkCmdDispatch inside an active dynamic-rendering region, its own
	// render-pass break). A scene with many objects going through it can spend most of a frame
	// here; see bench-notes.md 2026-09-27 for a measured example.
	FrameTiming::Add(FrameTiming::Counter::PrototypeDraws);
	FrameTiming::Add(FrameTiming::Counter::PrototypeVertices, count64);
	auto&      graphics  = m_context.GetGraphics();
	auto&      scheduler = m_context.GetCommandScheduler();
	auto&      cache     = m_context.GetBufferCache();
	scheduler.EndRendering();

	const auto     alignment       = graphics.StorageMinAlignment();
	const uint64_t vtx_total_bytes = static_cast<uint64_t>(input.stride) * input.num_records;
	const uint32_t index_element_size =
	    index_source.host_data != nullptr ? sizeof(uint16_t) : index_source.guest_element_size;

	// ---- Content-addressed capture cache (bench-notes.md 2026-09-27, "Kierunek B") ----
	// The capture compute stage is a pure function of {VS, PS, vertex bytes, index bytes, emit/
	// draw params}. Many objects in a scene (e.g. a field of static asteroid meshes) draw the
	// exact same content every frame; when nothing observable to the capture has changed, reuse
	// a cached capture and skip straight to the graphics replay. Shaders whose capture has
	// side-effecting buffer writes are excluded: skipping those would silently drop the write,
	// not just the geometry output.
	static const bool capture_cache_enabled = FrameTiming::OptEnabled("capture_cache");
	const bool         cache_eligible       = capture_cache_enabled && !capture_has_writes;
	uint64_t            cache_key           = 0;
	uint64_t            content_hash        = 0;
	uint64_t            vs_hash_for_cache   = 0;
	uint64_t            ps_hash_for_cache   = 0;
	CachedPrototypeCapture* hit             = nullptr;
	if (cache_eligible) {
		struct CaptureKeyMeta {
			uint64_t vs_hash;
			uint64_t ps_hash;
			uint32_t stride;
			uint32_t num_records;
			uint32_t num_attrs;
			uint32_t instance_count;
			uint32_t index_count;
			int32_t  vertex_offset;
			uint32_t first_vertex;
			uint32_t first_instance;
			uint32_t index_element_size;
		};
		vs_hash_for_cache = vs.stage.program->shader_hash;
		ps_hash_for_cache =
		    state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0;
		const CaptureKeyMeta meta {vs_hash_for_cache,
		                          ps_hash_for_cache,
		                          input.stride,
		                          input.num_records,
		                          num_attrs,
		                          draw.instance_count,
		                          draw.index_count,
		                          emit.vertex_offset,
		                          emit.first_vertex,
		                          emit.first_instance,
		                          draw.IsIndexed() ? index_element_size : 0u};
		const uint64_t meta_hash  = XXH3_64bits(&meta, sizeof(meta));
		const uint64_t attrs_hash = XXH3_64bits(vs.resources, sizeof(vs.resources[0]) * num_attrs);
		const uint64_t vertex_hash =
		    XXH3_64bits(reinterpret_cast<const void*>(input.addr), vtx_total_bytes);
		uint64_t index_hash = 0;
		if (draw.IsIndexed()) {
			const void* idx_bytes = index_source.host_data != nullptr
			                            ? index_source.host_data
			                            : reinterpret_cast<const void*>(index_source.address);
			index_hash = XXH3_64bits(idx_bytes, index_source.size);
		}
		const std::array<uint64_t, 4> parts {meta_hash, attrs_hash, vertex_hash, index_hash};
		content_hash = XXH3_64bits(parts.data(), parts.size() * sizeof(uint64_t));
		const std::array<uint64_t, 3> key_parts {input.addr, vs_hash_for_cache, ps_hash_for_cache};
		cache_key = XXH3_64bits(key_parts.data(), key_parts.size() * sizeof(uint64_t));

		auto it = m_prototype_capture_cache.find(cache_key);
		if (it != m_prototype_capture_cache.end() && it->second.captured != nullptr &&
		    it->second.content_hash == content_hash && it->second.addr == input.addr &&
		    it->second.vs_hash == vs_hash_for_cache && it->second.ps_hash == ps_hash_for_cache &&
		    it->second.count64 == count64 &&
		    it->second.record_stride_vec4 == programs->layout.record_stride_vec4) {
			hit                 = &it->second;
			hit->last_used_tick = scheduler.CurrentTick();
			FrameTiming::Add(FrameTiming::Counter::PrototypeCaptureCacheHits);
		}
	}

	PreparedBindings vertex_bindings;
	PreparedBindings pixel_bindings;
	PrepareBindings(vs.stage, vertex_bindings);
	PrepareBindings(state.ps_input_info.stage, pixel_bindings);
	PreparedBindings* stages[] {&vertex_bindings, &pixel_bindings};
	PrepareGraphicsBindings(stages, {state.color_info, state.color_count});

	std::unique_ptr<Buffer> captured;
	vk::DescriptorPool      capture_pool;
	vk::DescriptorSet       extra_set;
	uint64_t                captured_size_for_log = 0;
	vk::CommandBuffer       cmd                    = buffer.Handle();

	if (hit != nullptr) {
		extra_set              = hit->extra_set;
		captured_size_for_log  = hit->captured->Size();
	} else {
	// Obtain vertex buffer from GPU cache without host readback
	auto [vtx_buf_ptr, vtx_offset]    = cache.ObtainBuffer(input.addr, vtx_total_bytes, false);
	const uint64_t vtx_aligned_offset = Common::AlignDown(vtx_offset, alignment);
	const uint32_t vertex_byte_adjust = static_cast<uint32_t>(vtx_offset - vtx_aligned_offset);

	vk::Buffer     idx_vk_buf         = nullptr;
	uint64_t       idx_aligned_offset = 0;
	uint32_t       index_byte_adjust  = 0;
	uint64_t       idx_buf_available  = 0;
	if (draw.IsIndexed()) {
		EXIT_IF(index_element_size != 1 && index_element_size != 2 && index_element_size != 4);
		const uint64_t idx_total_bytes = index_source.size;
		if (index_source.host_data != nullptr) {
			auto&      stream  = cache.GetUtilityBuffer(MemoryUsage::Stream);
			const auto offset  = stream.Copy(index_source.host_data, idx_total_bytes, 16);
			idx_vk_buf         = stream.Handle();
			idx_aligned_offset = Common::AlignDown(offset, alignment);
			index_byte_adjust  = static_cast<uint32_t>(offset - idx_aligned_offset);
			idx_buf_available =
			    stream.Size() > idx_aligned_offset ? (stream.Size() - idx_aligned_offset) : 0;
		} else {
			auto [idx_buf_ptr, offset] =
			    cache.ObtainBuffer(index_source.address, idx_total_bytes, false);
			idx_vk_buf         = idx_buf_ptr->Handle();
			idx_aligned_offset = Common::AlignDown(offset, alignment);
			index_byte_adjust  = static_cast<uint32_t>(offset - idx_aligned_offset);
			idx_buf_available  = idx_buf_ptr->Size() > idx_aligned_offset
			                         ? (idx_buf_ptr->Size() - idx_aligned_offset)
			                         : 0;
		}
	} else {
		idx_vk_buf         = vtx_buf_ptr->Handle();
		idx_aligned_offset = 0;
		index_byte_adjust  = 0;
		idx_buf_available  = vtx_buf_ptr->Size();
	}

	// GPU-side intermediate buffers for unpack and capture
	auto id_buffer =
	    std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, count * 8);
	auto attribute_buffer = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0,
	                                                 AllFlags, count * num_attrs * 16);
	captured = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags,
	                                    count64 * programs->layout.record_stride_vec4 * 16);
	captured_size_for_log = captured->Size();

	const auto* unpack_pipeline =
	    GetPerVertexUnpackPipeline(graphics, m_context.GetPipelineCache().DriverCache());

	// Descriptor pool for the unpack set (4 storage buffers) and the capture-only extra set (3
	// storage buffers, real bindings to attribute_buffer/captured/id_buffer); both are per-draw
	// only, read solely by steps 1-2, and never cached.
	vk::DescriptorPool           unpack_pool;
	const vk::DescriptorPoolSize unpack_pool_size(vk::DescriptorType::eStorageBuffer, 7);
	vk::DescriptorPoolCreateInfo unpack_pool_info {};
	unpack_pool_info.maxSets       = 2;
	unpack_pool_info.poolSizeCount = 1;
	unpack_pool_info.pPoolSizes    = &unpack_pool_size;
	EXIT_IF(graphics.device.createDescriptorPool(&unpack_pool_info, nullptr, &unpack_pool) !=
	        vk::Result::eSuccess);

	// Descriptor pool for the replay-only extra set (3 storage buffers, all pointing at the
	// still-alive `captured` buffer -- see below). On a cache miss this, along with `captured`,
	// outlives this draw call when `cache_eligible` (owned by the cache entry below).
	const vk::DescriptorPoolSize capture_pool_size(vk::DescriptorType::eStorageBuffer, 3);
	vk::DescriptorPoolCreateInfo capture_pool_info {};
	capture_pool_info.maxSets       = 1;
	capture_pool_info.poolSizeCount = 1;
	capture_pool_info.pPoolSizes    = &capture_pool_size;
	EXIT_IF(graphics.device.createDescriptorPool(&capture_pool_info, nullptr, &capture_pool) !=
	        vk::Result::eSuccess);

	// Allocate and update unpack descriptor set
	vk::DescriptorSet             unpack_set;
	vk::DescriptorSetAllocateInfo unpack_alloc {};
	unpack_alloc.descriptorPool     = unpack_pool;
	unpack_alloc.descriptorSetCount = 1;
	unpack_alloc.pSetLayouts        = &unpack_pipeline->set_layout;
	EXIT_IF(graphics.device.allocateDescriptorSets(&unpack_alloc, &unpack_set) !=
	        vk::Result::eSuccess);

	const uint64_t idx_bytes_raw = draw.IsIndexed() ? (index_source.size + index_byte_adjust)
	                                                : (vtx_total_bytes + vertex_byte_adjust);
	const uint64_t idx_size_bounded =
	    std::min(Common::AlignUp(idx_bytes_raw, 4), idx_buf_available);

	const uint64_t vtx_bytes_raw = vtx_total_bytes + vertex_byte_adjust;
	const uint64_t vtx_buf_available =
	    vtx_buf_ptr->Size() > vtx_aligned_offset ? (vtx_buf_ptr->Size() - vtx_aligned_offset) : 0;
	const uint64_t vtx_size_bounded =
	    std::min(Common::AlignUp(vtx_bytes_raw, 4) + 4, vtx_buf_available);

	const vk::DescriptorBufferInfo unpack_data[] {
	    {idx_vk_buf, idx_aligned_offset, idx_size_bounded},
	    {vtx_buf_ptr->Handle(), vtx_aligned_offset, vtx_size_bounded},
	    {id_buffer->Handle(), 0, id_buffer->Size()},
	    {attribute_buffer->Handle(), 0, attribute_buffer->Size()}};
	std::array<vk::WriteDescriptorSet, 4> unpack_writes;
	for (uint32_t i = 0; i < 4; ++i) {
		unpack_writes[i].dstSet          = unpack_set;
		unpack_writes[i].dstBinding      = i;
		unpack_writes[i].descriptorCount = 1;
		unpack_writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		unpack_writes[i].pBufferInfo     = &unpack_data[i];
	}
	graphics.device.updateDescriptorSets(4, unpack_writes.data(), 0, nullptr);

	// Allocate and update the capture-only extra set (real bindings; used solely by step 2 below,
	// never cached and never rewritten again after this).
	vk::DescriptorSet             capture_extra_set;
	vk::DescriptorSetAllocateInfo capture_extra_alloc {};
	capture_extra_alloc.descriptorPool     = unpack_pool;
	capture_extra_alloc.descriptorSetCount = 1;
	capture_extra_alloc.pSetLayouts        = &programs->extra_layout;
	EXIT_IF(graphics.device.allocateDescriptorSets(&capture_extra_alloc, &capture_extra_set) !=
	        vk::Result::eSuccess);

	const vk::DescriptorBufferInfo capture_extra_data[] {
	    {attribute_buffer->Handle(), 0, attribute_buffer->Size()},
	    {captured->Handle(), 0, captured->Size()},
	    {id_buffer->Handle(), 0, id_buffer->Size()}};
	std::array<vk::WriteDescriptorSet, 3> capture_extra_writes;
	for (uint32_t i = 0; i < 3; ++i) {
		capture_extra_writes[i].dstSet          = capture_extra_set;
		capture_extra_writes[i].dstBinding      = i;
		capture_extra_writes[i].descriptorCount = 1;
		capture_extra_writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		capture_extra_writes[i].pBufferInfo     = &capture_extra_data[i];
	}
	graphics.device.updateDescriptorSets(3, capture_extra_writes.data(), 0, nullptr);

	// Allocate and update the replay-only extra set (`extra_set`, the outer variable): every
	// binding points at `captured`. Only binding 1 is statically read by the replay pipeline; 0
	// and 2 exist purely so the set holds no reference to attribute_buffer/id_buffer, which this
	// draw frees below while `extra_set` itself may be kept alive in the cache for many future
	// draws. Written exactly once -- never updated again while bound to any command buffer.
	vk::DescriptorSetAllocateInfo extra_alloc {};
	extra_alloc.descriptorPool     = capture_pool;
	extra_alloc.descriptorSetCount = 1;
	extra_alloc.pSetLayouts        = &programs->extra_layout;
	EXIT_IF(graphics.device.allocateDescriptorSets(&extra_alloc, &extra_set) !=
	        vk::Result::eSuccess);

	const vk::DescriptorBufferInfo replay_extra_data[] {{captured->Handle(), 0, captured->Size()},
	                                                     {captured->Handle(), 0, captured->Size()},
	                                                     {captured->Handle(), 0, captured->Size()}};
	std::array<vk::WriteDescriptorSet, 3> replay_extra_writes;
	for (uint32_t i = 0; i < 3; ++i) {
		replay_extra_writes[i].dstSet          = extra_set;
		replay_extra_writes[i].dstBinding      = i;
		replay_extra_writes[i].descriptorCount = 1;
		replay_extra_writes[i].descriptorType  = vk::DescriptorType::eStorageBuffer;
		replay_extra_writes[i].pBufferInfo     = &replay_extra_data[i];
	}
	graphics.device.updateDescriptorSets(3, replay_extra_writes.data(), 0, nullptr);

	// Prepare Unpack push constants
	UnpackPushConstants unpack_push {};
	unpack_push.total_invocations = count;
	unpack_push.index_count       = draw.index_count;
	unpack_push.first_vertex      = emit.first_vertex;
	unpack_push.vertex_offset     = emit.vertex_offset;
	unpack_push.first_instance    = emit.first_instance;
	unpack_push.vertex_stride     = input.stride;
	unpack_push.num_records       = input.num_records;

	uint32_t idx_flag = 0;
	if (draw.IsIndexed()) {
		idx_flag = 1u;
		if (index_element_size == 2) {
			idx_flag |= (1u << 1u);
		} else if (index_element_size == 4) {
			idx_flag |= (2u << 1u);
		}
	}
	unpack_push.packed_flags =
	    (index_byte_adjust & 0xffffu) | (idx_flag << 16u) | ((num_attrs & 0x0fu) << 19u);

	for (uint32_t i = 0; i < num_attrs; ++i) {
		const auto&    descriptor  = vs.resources[i];
		const auto     format_info = ShaderRecompiler::Format::GetFormatInfo(descriptor.Format());
		const uint32_t byte_offset = vertex_byte_adjust + input.attr_offsets[i];
		uint32_t       format_kind = 0;
		if (descriptor.Format() == Prospero::BufferFormat::k16_16_16_16Float) {
			format_kind = 3;
		} else if (descriptor.Format() == Prospero::BufferFormat::k16_16_16_16SNorm) {
			format_kind = 2;
		} else if (format_info.type == ShaderRecompiler::Format::ComponentType::Float) {
			format_kind = 1;
		} else if (format_info.type == ShaderRecompiler::Format::ComponentType::Uint) {
			format_kind = 4;
		} else if (format_info.type == ShaderRecompiler::Format::ComponentType::Sint) {
			format_kind = 5;
		}
		const uint32_t comp_count = format_info.component_count;
		unpack_push.attrs[i].meta =
		    (byte_offset & 0xffffu) | (comp_count << 17u) | (format_kind << 20u);

		uint32_t bit_counts = 0;
		for (uint32_t c = 0; c < 4; ++c) {
			bit_counts |= (format_info.component_bits[c] & 0xffu) << (c * 8u);
		}
		unpack_push.attrs[i].bit_counts = bit_counts;
	}

	cmd = buffer.Handle();

	// Step 1: GPU unpack compute pass
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, unpack_pipeline->pipeline);
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute, unpack_pipeline->pipeline_layout, 0, 1,
	                       &unpack_set, 0, nullptr);
	cmd.pushConstants(unpack_pipeline->pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
	                  sizeof(unpack_push), &unpack_push);
	KYTY_BENCH_TRACE_SITE();
	cmd.dispatch((count + 63) / 64, 1, 1);

	// Barrier: Unpack -> Capture compute
	vk::MemoryBarrier unpack_barrier {};
	unpack_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	unpack_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                    vk::PipelineStageFlagBits::eComputeShader, {}, 1, &unpack_barrier, 0,
	                    nullptr, 0, nullptr);

	// Step 2: Capture compute pass
	PreparedBindings* vertex_stage[] {&vertex_bindings};
	CommitBindings(buffer, vk::PipelineBindPoint::eCompute, programs->capture_pipeline,
	               vertex_stage);
	cmd = buffer.Handle();
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
	                       programs->capture_pipeline.pipeline_layout, 1, 1, &capture_extra_set, 0,
	                       nullptr);
	cmd.bindPipeline(vk::PipelineBindPoint::eCompute, programs->capture_pipeline.pipeline);
	KYTY_BENCH_TRACE_SITE();
	cmd.dispatch(programs->native_capture ? (count + 63u) / 64u : count, 1, 1);
	if (capture_has_writes) ShaderWriteBarrier(cmd, vk::PipelineStageFlagBits::eComputeShader);

	// Barrier: Capture compute -> Graphics replay
	vk::MemoryBarrier capture_barrier {};
	capture_barrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
	capture_barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
	cmd.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
	                    vk::PipelineStageFlagBits::eVertexShader |
	                        vk::PipelineStageFlagBits::eFragmentShader,
	                    {}, 1, &capture_barrier, 0, nullptr, 0, nullptr);

	// `id_buffer`/`attribute_buffer`/`unpack_pool` (which also backs `capture_extra_set`) are only
	// read by steps 1-2 above; reclaim them
	// now regardless of whether `captured`/`capture_pool`/`extra_set` end up cached below.
	scheduler.DeferOperation([attr = std::move(attribute_buffer), id = std::move(id_buffer),
	                          up = unpack_pool, dev = graphics.device]() mutable {
		attr.reset();
		id.reset();
		dev.destroyDescriptorPool(up);
	});
	} // if (hit != nullptr) ... else

	// Step 3: Graphics replay pass
	auto& pipeline = m_context.GetPipelineCache().GetGraphicsPipeline(
	    {state.color_info, state.color_count}, state.depth_info, {state.vertex_info.data(), 1},
	    buffer, &state.ps_input_info, topology, false, programs->graphics);
	vk::ImageAspectFlags feedback_aspects;
	const auto rendering = AcquireRenderTargets(buffer, state.color_info, state.color_count,
	                                            state.depth_info, feedback_aspects, stages);
	CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
	cmd = buffer.Handle();
	cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 1, 1,
	                       &extra_set, 0, nullptr);
	SetGraphicsDynamicParams(buffer, cmd, vs, state.depth_info, rendering);
	if (graphics.attachment_feedback_loop_enabled)
		cmd.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
	// Counts every attempt, not only ones that change CommandBuffer::m_render_state (the global
	// RenderPasses counter only counts the latter); compare the two to see how often this forced
	// break was actually redundant (same target as the previous draw).
	FrameTiming::Add(FrameTiming::Counter::PrototypeRenderPasses);
	scheduler.BeginRendering(rendering);
	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	KYTY_BENCH_TRACE_SITE();
	BenchTrace::Record(BenchTrace::Tick(), "proto-draw", vs.stage.program->shader_hash,
	                   state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0);
	cmd.draw(count, 1, 0, 0);
	scheduler.EndRendering();
	if (HasShaderBufferWrites(state.ps_input_info.stage))
		ShaderWriteBarrier(cmd, vk::PipelineStageFlagBits::eFragmentShader);

	static uint64_t draw_counter = 0;
	draw_counter++;
	LOGF("PerVertexPrototype: recorded draw submission=%llu vertices=%u instances=%u "
	     "capture_bytes=%llu layout_stride=%u cache_hit=%d\n",
	     static_cast<unsigned long long>(draw_counter), count, draw.instance_count,
	     static_cast<unsigned long long>(captured_size_for_log),
	     programs->layout.record_stride_vec4, hit != nullptr ? 1 : 0);

	if (hit == nullptr) {
		if (cache_eligible) {
			// Cache takes ownership of `captured`/`capture_pool`/`extra_set`. Whatever previously
			// occupied this slot, and the globally least-recently-used entry when the cache is at
			// capacity, are freed through the deferred-reclamation path (not immediately): an
			// already-recorded, not-yet-completed command buffer from an earlier frame may still
			// be reading them.
			static constexpr size_t kPrototypeCaptureCacheCap = 4096;
			auto&                   slot = m_prototype_capture_cache[cache_key];
			if (slot.captured != nullptr) {
				scheduler.DeferOperation([old_captured = std::move(slot.captured),
				                          old_pool = slot.pool, dev = graphics.device]() mutable {
					old_captured.reset();
					dev.destroyDescriptorPool(old_pool);
				});
			} else if (m_prototype_capture_cache.size() > kPrototypeCaptureCacheCap) {
				auto oldest = m_prototype_capture_cache.end();
				for (auto it = m_prototype_capture_cache.begin();
				     it != m_prototype_capture_cache.end(); ++it) {
					if (it->first == cache_key) continue;
					if (oldest == m_prototype_capture_cache.end() ||
					    it->second.last_used_tick < oldest->second.last_used_tick) {
						oldest = it;
					}
				}
				if (oldest != m_prototype_capture_cache.end()) {
					scheduler.DeferOperation([old_captured = std::move(oldest->second.captured),
					                          old_pool = oldest->second.pool,
					                          dev = graphics.device]() mutable {
						old_captured.reset();
						dev.destroyDescriptorPool(old_pool);
					});
					m_prototype_capture_cache.erase(oldest);
				}
			}
			slot.addr               = input.addr;
			slot.vs_hash            = vs_hash_for_cache;
			slot.ps_hash            = ps_hash_for_cache;
			slot.content_hash       = content_hash;
			slot.count64            = count64;
			slot.record_stride_vec4 = programs->layout.record_stride_vec4;
			slot.last_used_tick     = scheduler.CurrentTick();
			slot.captured           = std::move(captured);
			slot.pool               = capture_pool;
			slot.extra_set          = extra_set;
		} else {
			// Diagnostic (capture has side-effecting writes) or `KYTY_OPT_OFF=capture_cache`:
			// keep the original per-draw lifetime, same as before this cache existed.
			scheduler.DeferOperation([cap = std::move(captured), cp = capture_pool,
			                          dev = graphics.device]() mutable {
				cap.reset();
				dev.destroyDescriptorPool(cp);
			});
		}
	}
}
