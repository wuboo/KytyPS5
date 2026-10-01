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
	EXIT_IF(vs.logical_stage != ShaderType::Vertex || vs.buffers_num != 1 || vs.resources_num <= 0);
	if (vs.resources_num > static_cast<int>(kPerVertexPrototypeMaxAttributes)) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
			const uint64_t hash = vs.stage.program != nullptr ? vs.stage.program->shader_hash : 0;
			LOGF("PerVertexPrototype: skipping draw, VS 0x%016llx has %d attributes (limit %u)\n",
			     static_cast<unsigned long long>(hash), vs.resources_num,
			     kPerVertexPrototypeMaxAttributes);
		}
		return;
	}
	if (vs.resources_num > 12) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 4) {
			const uint64_t hash = vs.stage.program != nullptr ? vs.stage.program->shader_hash : 0;
			LOGF("PerVertexPrototype: VS 0x%016llx draws with %d attributes\n",
			     static_cast<unsigned long long>(hash), vs.resources_num);
		}
	}
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
	// CORRECTNESS FIX (2026-09-28, found via a since-removed diagnostic): the capture is only a
	// pure function of {vertex bytes, index bytes, a few draw/emit params} -- what the cache key
	// hashed -- if the captured VS reads NOTHING ELSE. It runs as compute with its full original
	// resource bindings (PrepareBindings(vs.stage, ...) below covers general buffers/images/
	// samplers, not just the vertex-fetch buffer decoded via vs.resources[]/vs.buffers[0]); if it
	// declares any such resource (e.g. a per-object transform matrix in a uniform buffer) whose
	// content isn't hashed too, two objects with identical vertex bytes but a different bound
	// resource could silently share a captured result that is only correct for one of them --
	// confirmed to actually happen in this game (a real VS hash logged with buffers=4). General
	// buffers are handled below (their bound bytes are hashed the same way the vertex buffer's
	// are); images/samplers are not (no simple byte range to hash), so a VS using either is still
	// excluded from caching entirely rather than guessed at.
	const bool capture_has_uncoverable_resources =
	    vs.stage.program != nullptr &&
	    (!vs.stage.program->info.images.empty() || !vs.stage.program->info.samplers.empty());
	const bool cache_eligible =
	    capture_cache_enabled && !capture_has_writes && !capture_has_uncoverable_resources;
	uint64_t            cache_key           = 0;
	uint64_t            content_hash        = 0;
	// Also used below by the batching key (Kierunek A), regardless of cache_eligible.
	const uint64_t vs_hash_for_cache = vs.stage.program->shader_hash;
	const uint64_t ps_hash_for_cache =
	    state.ps_active ? state.ps_input_info.stage.program->shader_hash : 0;
	CachedPrototypeCapture* hit = nullptr;
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
		// Hash the actual bound bytes of every general buffer resource the VS declares (e.g. a
		// per-object transform matrix), the same way the vertex buffer's bytes are hashed above --
		// see the correctness-fix note where cache_eligible is computed. Same decode as
		// FindBuffers() (descriptors.cpp): a buffer resource pointing below the first page, or with
		// a zero size, is degenerate/unbound and contributes nothing (matches FindBuffers treating
		// it as an empty BufferSource).
		uint64_t buffers_hash = 0;
		if (!vs.stage.program->info.buffers.empty()) {
			EXIT_IF(vs.stage.resources == nullptr);
			const auto& snapshot     = *vs.stage.resources;
			const auto& buffer_infos = vs.stage.program->info.buffers;
			// Sized to the actual declared count, not ShaderInfo::MaxBuffers: that constant is a
			// design intent, not a bound this vector's producer is verified to enforce.
			std::vector<uint64_t> per_buffer_hashes(buffer_infos.size(), 0);
			for (size_t i = 0; i < buffer_infos.size(); ++i) {
				const auto descriptor =
				    DecodeNativeDescriptor<ShaderBufferResource>(snapshot.buffers[i]);
				const auto address        = descriptor.Base48();
				const auto requested_size = descriptor.GetSize();
				if (address < BufferCache::CACHING_PAGESIZE || requested_size == 0) {
					continue;
				}
				const auto size = Libs::LibKernel::Memory::ClampRangeSize(address, requested_size);
				const std::array<uint64_t, 2> src {
				    address, XXH3_64bits(reinterpret_cast<const void*>(address), size)};
				per_buffer_hashes[i] = XXH3_64bits(src.data(), src.size() * sizeof(uint64_t));
			}
			buffers_hash =
			    XXH3_64bits(per_buffer_hashes.data(), buffer_infos.size() * sizeof(uint64_t));
		}
		const std::array<uint64_t, 5> parts {meta_hash, attrs_hash, vertex_hash, index_hash,
		                                     buffers_hash};
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
	    (index_byte_adjust & 0xffffu) | (idx_flag << 16u) | ((num_attrs & 0x1fu) << 19u);

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

	static uint64_t draw_counter = 0;
	draw_counter++;
	// Per-draw logging cost ~4% of this path in a scene with ~500 such draws per frame.
	if (draw_counter <= 64) {
		LOGF("PerVertexPrototype: recorded draw submission=%llu vertices=%u instances=%u "
		     "capture_bytes=%llu layout_stride=%u cache_hit=%d\n",
		     static_cast<unsigned long long>(draw_counter), count, draw.instance_count,
		     static_cast<unsigned long long>(captured_size_for_log),
		     programs->layout.record_stride_vec4, hit != nullptr ? 1 : 0);
	}

	// `extra_set` is fully decided from here on (either from the cache, or freshly captured
	// above); settle its ownership before choosing whether to replay now or batch it, so both
	// paths below only ever read it.
	const bool store_in_cache = hit == nullptr && cache_eligible;
	if (store_in_cache) {
		// Cache takes ownership of `captured`/`capture_pool`/`extra_set`. Whatever previously
		// occupied this slot, and the globally least-recently-used entry when the cache is at
		// capacity, are freed through the deferred-reclamation path (not immediately): an
		// already-recorded, not-yet-completed command buffer from an earlier frame may still
		// be reading them.
		static constexpr size_t kPrototypeCaptureCacheCap = 4096;
		auto&                   slot = m_prototype_capture_cache[cache_key];
		// A pending batch item may still use the capture this slot (or the evicted entry) holds --
		// its draw is only recorded when the batch flushes, possibly after a submit that already ran
		// the deferred free below. Confirmed as the cause of a guest crash with batching + register
		// writes passing (bench-notes.md 2026-09-28, "Kierunek A v2"): record the batch first.
		if (m_prototype_batch.active &&
		    (slot.captured != nullptr ||
		     m_prototype_capture_cache.size() > kPrototypeCaptureCacheCap)) {
			FlushPendingPrototypeBatch(buffer);
		}
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
	}

	// ---- Batching of the replay draw (bench-notes.md 2026-09-27, "Kierunek A") ----
	// Consecutive replay draws that share {VS, PS, render targets, topology} can share one
	// BeginRendering/EndRendering pair instead of one each. Whether the *next* PM4 packet is a
	// compatible draw is decided by the interpreter (graphicsRun.cpp) before it runs anything
	// else, and by ExecutePreparedDraw() before it runs a non-matching draw -- both flush this
	// batch first -- so a pending item here is never observed by anything else in the meantime.
	static const bool batch_opt_enabled = FrameTiming::OptEnabled("prototype_batch");
	// A shader using GDS (Global Data Share -- PS4/PS5 atomic counters / append-consume buffers)
	// makes CommitBindings() end the current dynamic-rendering scope internally, to fence the GDS
	// buffer outside of it (see its `descriptors.gds.buffer` branch, pipeline/descriptors.cpp).
	// The immediate-replay path below calls CommitBindings() *before* BeginRendering(), so that's
	// a harmless no-op there. FlushPendingPrototypeBatch() calls it per item *after*
	// BeginRendering() (needed so all of a batch's draws share one render pass), so for a
	// GDS-using item it silently ends the render pass mid-batch -- every bind/draw after that
	// point, for this item and any batched with it, then runs with no active render encoder,
	// the same encoder-corruption class suspected for bug #3. Found while hunting bug #4
	// (bench-notes.md 2026-09-28); live diagnostics ruled it out as bug #4's actual cause (that
	// crash's draw doesn't use GDS), but it is a real, independent correctness gap in its own
	// right, worth keeping fixed. PrepareBindings() above already populated `.gds.buffer` for
	// both stages, so checking it here is free.
	const bool item_uses_gds =
	    vertex_bindings.gds.buffer != nullptr || pixel_bindings.gds.buffer != nullptr;
	// Bug #4 (bench-notes.md 2026-09-28, "second round"): a specific VS/PS pair crashes
	// deterministically when it goes through the batched path at all, even alone in a batch of
	// one, for reasons that survived every mitigation tried (activation delay, batch-size cap,
	// GDS exclusion) -- narrowing it to something about this exact shader specifically, not
	// batching in general. KYTY_BENCH_SKIP_BATCH excludes it (and lets anything else be named
	// later the same way) from batching only, same mechanism as the existing
	// KYTY_BENCH_SKIP_DRAW/KYTY_BENCH_SKIP_CS bench switches -- the draw itself still renders via
	// the immediate path below, nothing visible is lost, only this one shader's forced
	// render-pass break stays unbatched.
	const bool item_skipped_from_batch = BenchTrace::SkipBatch(vs_hash_for_cache, ps_hash_for_cache);
	const bool batch_enabled = batch_opt_enabled && !item_uses_gds && !item_skipped_from_batch;
	if (!batch_enabled && m_prototype_batch.active) {
		// Falling through to the immediate-replay path below despite an active pending batch --
		// only possible now for a GDS-using or SKIP_BATCH-listed item (batch_opt_enabled is a
		// process-wide constant, so when it alone gates this, m_prototype_batch.active can never
		// be true here). Flush first so draw order is preserved; the interpreter's own flush
		// hooks don't know about these exclusions, since from PM4-opcode level this looks like
		// any other eligible draw.
		FlushPendingPrototypeBatch(buffer);
	}
	// Exact byte equality (not just image identity) of color/depth info: these also carry
	// blend-bypass/depth-test/stencil state that feeds GetGraphicsPipeline per item below, and
	// which the shared BeginRendering doesn't otherwise account for. Comparing every byte can
	// only ever cause an unnecessary miss (from incidentally differing padding), never a false
	// match, the same tradeoff PipelineStaticParameters::operator==() already makes elsewhere.
	bool colors_match = batch_enabled && m_prototype_batch.active &&
	                    m_prototype_batch.color_count == state.color_count;
	for (uint32_t i = 0; colors_match && i < state.color_count; ++i) {
		colors_match = std::memcmp(&m_prototype_batch.color_info[i], &state.color_info[i],
		                           sizeof(RenderColorInfo)) == 0;
	}
	const bool batch_matches =
	    colors_match && m_prototype_batch.vs_hash == vs_hash_for_cache &&
	    m_prototype_batch.ps_hash == ps_hash_for_cache &&
	    m_prototype_batch.ps_active == state.ps_active && m_prototype_batch.topology == topology &&
	    m_prototype_batch.programs == programs &&
	    std::memcmp(&m_prototype_batch.depth_info, &state.depth_info, sizeof(RenderDepthInfo)) == 0;
	if (batch_enabled) {
		if (m_prototype_batch.active && !batch_matches) {
			FlushPendingPrototypeBatch(buffer);
		}
		if (!m_prototype_batch.active) {
			m_prototype_batch.active      = true;
			m_prototype_batch.vs_hash     = vs_hash_for_cache;
			m_prototype_batch.ps_hash     = ps_hash_for_cache;
			m_prototype_batch.color_count = state.color_count;
			for (uint32_t i = 0; i < state.color_count; ++i) {
				m_prototype_batch.color_info[i] = state.color_info[i];
			}
			m_prototype_batch.depth_info     = state.depth_info;
			m_prototype_batch.ps_active      = state.ps_active;
			m_prototype_batch.ps_input_info  = state.ps_input_info;
			m_prototype_batch.topology       = topology;
			m_prototype_batch.programs       = programs;
			m_prototype_batch.command_buffer_at_start = buffer.Handle();
		} else if (m_prototype_batch.command_buffer_at_start != buffer.Handle()) {
			// Diagnostic (bug #4 hunt): an implicit submit happened between this batch's first
			// item and this one, without ever going through a flush hook.
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("[bench] prototype_batch: command buffer changed mid-batch (was %p, now %p), "
				     "items=%zu\n",
				     static_cast<void*>(static_cast<VkCommandBuffer>(m_prototype_batch.command_buffer_at_start)),
				     static_cast<void*>(static_cast<VkCommandBuffer>(buffer.Handle())),
				     m_prototype_batch.items.size());
			}
		}
		PendingPrototypeReplayItem item;
		// Same pipeline lookup as the immediate path below, done now rather than at flush time:
		// GetGraphicsPipeline() reads live registers, which may already describe a later draw by the
		// time the batch is flushed. No render pass is open here -- the capture dispatch above ended it.
		item.pipeline = &m_context.GetPipelineCache().GetGraphicsPipeline(
		    {state.color_info, state.color_count}, state.depth_info, {state.vertex_info.data(), 1},
		    buffer, &state.ps_input_info, topology, false, programs->graphics);
		item.registers = std::make_shared<HW::Context>(buffer.GetRegisters());
		item.extra_set       = extra_set;
		item.count           = count;
		item.vs               = vs;
		// `vs.pixel_input`, if set, also points into the caller's stack-local DrawRenderState (see
		// the `runtime` note below). Nothing reads it once a program is already compiled -- it
		// only feeds shader specialization, which already happened for programs->graphics when
		// GetPerVertexPrototypePrograms() first compiled it -- but null it out anyway so a latent
		// read elsewhere fails loudly (null deref) instead of touching freed stack memory.
		item.vs.pixel_input   = nullptr;
		item.vertex_bindings  = std::move(vertex_bindings);
		item.pixel_bindings   = std::move(pixel_bindings);
		// `PreparedBindings::runtime` is a raw pointer that PrepareBindings() above set into
		// `vs.stage`/`state.ps_input_info.stage` -- both part of the caller's stack-local
		// DrawRenderState, gone as soon as that caller (DrawIndex/DrawAuto) returns, which happens
		// right after this draw does. It gets repointed at storage this item/batch actually owns,
		// but NOT here: `m_prototype_batch.items` is a std::vector, and every push_back beyond its
		// current capacity moves every already-appended item (this one included) to a new address,
		// which would leave a pointer into *this* item's own address (a self-reference) dangling.
		// FlushPendingPrototypeBatch() fixes `runtime` up right before each item's address is
		// final (no more items will be appended once flushing starts).
		if (!store_in_cache && hit == nullptr) {
			item.owned_captured = std::move(captured);
			item.owned_pool     = capture_pool;
		}
		m_prototype_batch.items.push_back(std::move(item));
		return;
	}

	// Batching disabled (or KYTY_OPT_OFF=prototype_batch): replay immediately, exactly as before
	// this batching existed.
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

	if (!store_in_cache) {
		// Diagnostic (capture has side-effecting writes) or `KYTY_OPT_OFF=capture_cache`:
		// keep the original per-draw lifetime, same as before this cache existed.
		scheduler.DeferOperation([cap = std::move(captured), cp = capture_pool,
		                          dev = graphics.device]() mutable {
			cap.reset();
			dev.destroyDescriptorPool(cp);
		});
	}
}

void RenderExecutor::FlushPendingPrototypeBatch(CommandBuffer& buffer) {
	if (!m_prototype_batch.active) {
		return;
	}
	auto& batch = m_prototype_batch;
	m_prototype_batch.active = false; // reentrancy guard: none of the calls below re-enter here
	                                  // with `active` still true, but keep this false regardless
	if (batch.items.empty()) {
		batch.items.clear();
		return;
	}
	// Diagnostic (bug #4 hunt, 2026-09-28): confirm/deny whether an implicit submit -- one not
	// tied to any PM4 opcode our flush hooks intercept -- happened between this batch's first
	// item's capture and this flush. If so, that item's capture_barrier is recorded into an
	// already-submitted, separate command buffer and does not protect this flush's replay draw.
	if (batch.command_buffer_at_start != buffer.Handle()) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
			LOGF("[bench] prototype_batch: FLUSHING INTO A DIFFERENT COMMAND BUFFER than this "
			     "batch started in (was %p, now %p), items=%zu -- capture barrier for at least "
			     "the first item is not in this command buffer\n",
			     static_cast<void*>(static_cast<VkCommandBuffer>(batch.command_buffer_at_start)),
			     static_cast<void*>(static_cast<VkCommandBuffer>(buffer.Handle())),
			     batch.items.size());
		}
	}
	auto& graphics  = m_context.GetGraphics();
	auto& scheduler = m_context.GetCommandScheduler();
	scheduler.EndRendering();

	// Pipelines were resolved when each item was appended (see there); `runtime` is fixed up here,
	// once each item's address in `batch.items` is final (see the note where items are appended).
	for (auto& item: batch.items) {
		item.vertex_bindings.runtime = &item.vs.stage;
		item.pixel_bindings.runtime  = &batch.ps_input_info.stage;
	}

	// See the design note where items are appended: bindings/feedback-loop detection depend on
	// which resources are actually bound, which could in principle differ between items sharing
	// the same {VS, PS, targets}; every item here shares the exact same compiled VS+PS program
	// (same vs_hash+ps_hash), so its static resource-usage pattern -- and therefore whether it
	// reads its own render target -- is the same for all of them. Acquiring once, from the first
	// item, is exactly as correct as acquiring per item would be, and is what makes batching the
	// render pass possible at all.
	PreparedBindings*     first_stages[] {&batch.items[0].vertex_bindings,
	                                     &batch.items[0].pixel_bindings};
	vk::ImageAspectFlags feedback_aspects;
	const auto           rendering =
	    AcquireRenderTargets(buffer, batch.color_info.data(), batch.color_count, batch.depth_info,
	                         feedback_aspects, first_stages);
	// Bug #4 (bench-notes.md 2026-09-28): CommitBindings() below runs inside the render pass, and
	// its image layout transitions (Image::Transit) end rendering whenever they record a barrier --
	// confirmed live, right before the crash, on a texture-sampling UI sprite. The immediate path
	// never hits this because it commits before BeginRendering. Do the transitions for every item
	// here, after the render targets are acquired (same order as the immediate path) and before the
	// pass begins; CommitBindings() then finds nothing left to transition.
	for (size_t i = 0; i < batch.items.size(); ++i) {
		PreparedBindings* stages[] {&batch.items[i].vertex_bindings,
		                            &batch.items[i].pixel_bindings};
		TransitionBoundResources(buffer, stages, batch.items[i].pipeline->prototype_vertex_capture);
	}
	FrameTiming::Add(FrameTiming::Counter::PrototypeRenderPasses);
	scheduler.BeginRendering(rendering);
	for (size_t i = 0; i < batch.items.size(); ++i) {
		auto&      item     = batch.items[i];
		auto&      pipeline = *item.pipeline;
		PreparedBindings* stages[] {&item.vertex_bindings, &item.pixel_bindings};
		CommitBindings(buffer, vk::PipelineBindPoint::eGraphics, pipeline, stages);
		if (!buffer.IsRendering()) {
			// Something in CommitBindings (Image::Transit, the GDS barrier) ended the batch's
			// render pass; the bind/draw below would run with no active render encoder.
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1, std::memory_order_relaxed) < 64) {
				LOGF("[bench-batch-closed] item=%zu/%zu vs=0x%016llx ps=0x%016llx\n", i,
				     batch.items.size(), static_cast<unsigned long long>(batch.vs_hash),
				     static_cast<unsigned long long>(batch.ps_hash));
			}
		}
		auto cmd = buffer.Handle();
		cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipeline.pipeline_layout, 1, 1,
		                       &item.extra_set, 0, nullptr);
		SetGraphicsDynamicParams(buffer, cmd, item.vs, batch.depth_info, rendering,
		                         item.registers.get());
		if (graphics.attachment_feedback_loop_enabled)
			cmd.setAttachmentFeedbackLoopEnableEXT(feedback_aspects);
		cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
		KYTY_BENCH_TRACE_SITE();
		BenchTrace::Record(BenchTrace::Tick(), "proto-draw", batch.vs_hash, batch.ps_hash);
		cmd.draw(item.count, 1, 0, 0);
		if (batch.ps_active && HasShaderBufferWrites(batch.ps_input_info.stage))
			ShaderWriteBarrier(cmd, vk::PipelineStageFlagBits::eFragmentShader);
	}
	scheduler.EndRendering();

	for (auto& item: batch.items) {
		if (item.owned_captured != nullptr) {
			scheduler.DeferOperation([cap = std::move(item.owned_captured), cp = item.owned_pool,
			                          dev = graphics.device]() mutable {
				cap.reset();
				dev.destroyDescriptorPool(cp);
			});
		}
	}
	batch.items.clear();
}
