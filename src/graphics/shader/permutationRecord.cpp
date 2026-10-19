#include "graphics/shader/permutationRecord.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

namespace Libs::Graphics::ProgramDisk {

namespace IR = ShaderRecompiler::IR;

// A field added to one of these structures must be written below (and kProgramDiskFormat bumped): the build stops here until the size is updated.
// The sizes are those of libc++ on a 64-bit target.
#if defined(_LIBCPP_VERSION) && defined(__LP64__)
static_assert(sizeof(IR::ResourceSpecialization::Buffer) == 28, "ResourceSpecialization::Buffer changed: update permutationRecord.cpp");
static_assert(sizeof(IR::ResourceSpecialization::Image) == 56, "ResourceSpecialization::Image changed: update permutationRecord.cpp");
static_assert(sizeof(IR::ResourceSpecialization::Sampler) == 12, "ResourceSpecialization::Sampler changed: update permutationRecord.cpp");
static_assert(sizeof(IR::ResourceSpecialization) == 72, "ResourceSpecialization changed: update permutationRecord.cpp");
static_assert(sizeof(IR::BufferResource) == 80, "BufferResource changed: update permutationRecord.cpp");
static_assert(sizeof(IR::ImageResource) == 112, "ImageResource changed: update permutationRecord.cpp");
static_assert(sizeof(IR::SamplerResource) == 56, "SamplerResource changed: update permutationRecord.cpp");
static_assert(sizeof(IR::SampledResourcePair) == 12, "SampledResourcePair changed: update permutationRecord.cpp");
static_assert(sizeof(IR::StageInput) == 48, "StageInput changed: update permutationRecord.cpp");
static_assert(sizeof(IR::StageOutput) == 40, "StageOutput changed: update permutationRecord.cpp");
static_assert(sizeof(IR::Bindings) == 248, "Bindings changed: update permutationRecord.cpp");
static_assert(sizeof(IR::ShaderInfo) == 240, "ShaderInfo changed: update permutationRecord.cpp");
static_assert(sizeof(IR::CompiledShaderInfo) == 528, "CompiledShaderInfo changed: update permutationRecord.cpp");
static_assert(sizeof(PvOriginalRecord) == 32 && sizeof(PvStoreRecord) == 56, "PV records changed: update permutationRecord.cpp");
#endif

namespace {

constexpr uint64_t kMaxCount = 1ull << 26;

class Writer {
public:
	void U8(uint8_t value) { m_out.push_back(value); }
	void Bool(bool value) { U8(value ? 1 : 0); }
	void U32(uint32_t value) { Raw(&value, sizeof(value)); }
	void I32(int32_t value) { Raw(&value, sizeof(value)); }
	void U64(uint64_t value) { Raw(&value, sizeof(value)); }
	template <typename E>
	void Enum(E value) { U32(static_cast<uint32_t>(value)); }
	void Raw(const void* data, size_t size) {
		const auto* bytes = static_cast<const uint8_t*>(data);
		m_out.insert(m_out.end(), bytes, bytes + size);
	}
	void Words(const std::vector<uint32_t>& words) {
		U64(words.size());
		Raw(words.data(), words.size() * sizeof(uint32_t));
	}
	void String(const std::string& text) {
		U64(text.size());
		Raw(text.data(), text.size());
	}
	std::vector<uint8_t>& Bytes() { return m_out; }

private:
	std::vector<uint8_t> m_out;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> data): m_data(data) {}
	bool Raw(void* out, size_t size) {
		if (m_data.size() - m_pos < size) return false;
		std::memcpy(out, m_data.data() + m_pos, size);
		m_pos += size;
		return true;
	}
	bool U8(uint8_t* value) { return Raw(value, 1); }
	bool Bool(bool* value) {
		uint8_t byte = 0;
		if (!U8(&byte) || byte > 1) return false;
		*value = byte != 0;
		return true;
	}
	bool U32(uint32_t* value) { return Raw(value, sizeof(*value)); }
	bool I32(int32_t* value) { return Raw(value, sizeof(*value)); }
	bool U64(uint64_t* value) { return Raw(value, sizeof(*value)); }
	template <typename E>
	bool Enum(E* value) {
		uint32_t raw = 0;
		if (!U32(&raw)) return false;
		*value = static_cast<E>(raw);
		return true;
	}
	bool Count(uint64_t* count, size_t element_bytes) {
		return U64(count) && *count <= kMaxCount && *count * element_bytes <= m_data.size() - m_pos;
	}
	bool Words(std::vector<uint32_t>* words) {
		uint64_t count = 0;
		if (!Count(&count, sizeof(uint32_t))) return false;
		words->resize(count);
		return Raw(words->data(), count * sizeof(uint32_t));
	}
	bool String(std::string* text) {
		uint64_t count = 0;
		if (!Count(&count, 1)) return false;
		text->resize(count);
		return Raw(text->data(), count);
	}
	[[nodiscard]] size_t Remaining() const { return m_data.size() - m_pos; }

private:
	std::span<const uint8_t> m_data;
	size_t                   m_pos = 0;
};

// ---- specialization -----------------------------------------------------------------------------------------------------------------------

void Put(Writer& w, const IR::ResourceSpecialization& spec) {
	w.U64(spec.buffers.size());
	for (const auto& b: spec.buffers) {
		w.U32(b.packed_stride);
		w.Enum(b.descriptor_format);
		w.U32(b.descriptor_swizzle);
		w.Bool(b.zero_stride_oob);
		w.Bool(b.scalar_empty);
		w.U32(b.indirect_root);
		w.U32(b.indirect_mapping_offset);
		w.U32(b.indirect_search_iterations);
	}
	w.U64(spec.images.size());
	for (const auto& i: spec.images) {
		w.Enum(i.numeric_class);
		w.Enum(i.dimension);
		w.U32(i.mip_count);
		w.Enum(i.conversion_format);
		w.U32(i.shader_swizzle);
		w.U32(i.indirect_root);
		w.U32(i.indirect_mapping_offset);
		w.U32(i.indirect_search_iterations);
		w.Bool(i.cube);
		w.Bool(i.fmask);
		w.U32(i.min_lod_resinfo.shift);
		for (const uint32_t extent: i.min_lod_resinfo.extent) w.U32(extent);
		w.U32(i.min_lod_resinfo.levels);
	}
	w.U64(spec.samplers.size());
	for (const auto& s: spec.samplers) {
		w.U32(s.indirect_root);
		w.U32(s.indirect_mapping_offset);
		w.U32(s.indirect_search_iterations);
	}
}

bool Get(Reader& r, IR::ResourceSpecialization* spec) {
	uint64_t count = 0;
	if (!r.Count(&count, 8)) return false;
	spec->buffers.resize(count);
	for (auto& b: spec->buffers) {
		if (!r.U32(&b.packed_stride) || !r.Enum(&b.descriptor_format) || !r.U32(&b.descriptor_swizzle) || !r.Bool(&b.zero_stride_oob) ||
		    !r.Bool(&b.scalar_empty) || !r.U32(&b.indirect_root) || !r.U32(&b.indirect_mapping_offset) || !r.U32(&b.indirect_search_iterations)) {
			return false;
		}
	}
	if (!r.Count(&count, 8)) return false;
	spec->images.resize(count);
	for (auto& i: spec->images) {
		if (!r.Enum(&i.numeric_class) || !r.Enum(&i.dimension) || !r.U32(&i.mip_count) || !r.Enum(&i.conversion_format) ||
		    !r.U32(&i.shader_swizzle) || !r.U32(&i.indirect_root) || !r.U32(&i.indirect_mapping_offset) || !r.U32(&i.indirect_search_iterations) ||
		    !r.Bool(&i.cube) || !r.Bool(&i.fmask) || !r.U32(&i.min_lod_resinfo.shift)) {
			return false;
		}
		for (uint32_t& extent: i.min_lod_resinfo.extent) {
			if (!r.U32(&extent)) return false;
		}
		if (!r.U32(&i.min_lod_resinfo.levels)) return false;
	}
	if (!r.Count(&count, 12)) return false;
	spec->samplers.resize(count);
	for (auto& s: spec->samplers) {
		if (!r.U32(&s.indirect_root) || !r.U32(&s.indirect_mapping_offset) || !r.U32(&s.indirect_search_iterations)) return false;
	}
	return true;
}

// ---- compiled info ------------------------------------------------------------------------------------------------------------------------

void Put(Writer& w, const IR::ShaderInfo& info) {
#ifdef KYTY_PERMREC_MUTANT_DROP_INFO_VECTORS
	w.U64(0), w.U64(0), w.U64(0), w.U64(0), w.U64(0), w.U64(0);
	w.Words({});
#else
	w.U64(info.buffers.size());
	for (const auto& b: info.buffers) {
		w.U32(b.source);
		w.U32(b.descriptor_index);
		w.U32(b.first_use_pc);
		w.U32(b.max_byte_extent);
		w.U32(b.packed_stride);
		w.Enum(b.descriptor_format);
		w.U32(b.descriptor_swizzle);
		w.U32(b.image_alias);
		w.Bool(b.read), w.Bool(b.written), w.Bool(b.atomic), w.Bool(b.formatted), w.Bool(b.scalar), w.Bool(b.live);
		w.U32(b.indirect_root);
		w.U32(b.indirect_mapping_offset);
		w.U32(b.indirect_search_iterations);
		w.Words(b.indirect_resources);
	}
	w.U64(info.images.size());
	for (const auto& i: info.images) {
		w.U32(i.source);
		w.U32(i.descriptor_index);
		w.U32(i.first_use_pc);
		w.Enum(i.resource_class);
		w.Enum(i.numeric_class);
		w.Enum(i.dimension);
		w.Enum(i.mip_mode);
		w.U32(i.mip_count);
		w.Enum(i.conversion_format);
		w.U32(i.shader_swizzle);
		w.Bool(i.read), w.Bool(i.written), w.Bool(i.atomic), w.Bool(i.atomic64), w.Bool(i.unfiltered_access_only), w.Bool(i.depth_compare);
		w.Bool(i.cube), w.Bool(i.r128), w.Bool(i.queries_size);
		w.U32(i.min_lod_resinfo.shift);
		for (const uint32_t extent: i.min_lod_resinfo.extent) w.U32(extent);
		w.U32(i.min_lod_resinfo.levels);
		w.U32(i.indirect_root);
		w.U32(i.indirect_mapping_offset);
		w.U32(i.indirect_search_iterations);
		w.Words(i.indirect_resources);
	}
	w.U64(info.samplers.size());
	for (const auto& s: info.samplers) {
		w.U32(s.source);
		w.U32(s.first_use_pc);
		w.U32(s.snapshot_index);
		w.U32(s.indirect_root);
		w.U32(s.indirect_mapping_offset);
		w.U32(s.indirect_search_iterations);
		w.Words(s.indirect_resources);
		w.Bool(s.force_point_filtering), w.Bool(s.depth_compare), w.Bool(s.integer_border), w.Bool(s.gather_lod);
	}
	w.U64(info.sampled_pairs.size());
	for (const auto& p: info.sampled_pairs) {
		w.U32(p.image), w.U32(p.sampler), w.U32(p.first_use_pc);
	}
	w.U64(info.inputs.size());
	for (const auto& in: info.inputs) {
		w.Enum(in.kind);
		w.U32(in.location);
		w.U32(in.component_count);
		w.String(in.debug_name);
		w.Bool(in.per_vertex);
	}
	w.U64(info.outputs.size());
	for (const auto& out: info.outputs) {
		w.Enum(out.kind);
		w.U32(out.index);
		w.U32(out.location);
		w.String(out.debug_name);
	}
	w.Words(info.user_data_registers);
#endif
	w.Raw(info.vertex_fetch_components.data(), info.vertex_fetch_components.size());
	w.I32(info.vertex_offset_sgpr);
	w.I32(info.instance_offset_sgpr);
	const bool flags[] = {info.has_bitwise_xor, info.uses_dma, info.uses_swizzle, info.uses_lds, info.uses_gds, info.uses_flattened_srt,
	                      info.uses_dispatch_threads, info.bvh, info.subgroup_ballot, info.subgroup_vote, info.subgroup_barrier,
	                      info.subgroup_shuffle, info.subgroup_arithmetic, info.subgroup_local_invocation_id, info.compute_derivatives,
	                      info.image_gather_extended, info.function_lds, info.function_scratch, info.pixel_valid_mask, info.buffer_int64_atomics,
	                      info.buffer_u8, info.buffer_u16, info.shared_int64_atomics, info.coherent_buffers, info.float64};
	uint64_t     bits    = 0;
	for (size_t i = 0; i < std::size(flags); i++) bits |= uint64_t {flags[i]} << i;
	w.U64(bits);
}

bool Get(Reader& r, IR::ShaderInfo* info) {
	uint64_t count = 0;
	if (!r.Count(&count, 8)) return false;
	info->buffers.resize(count);
	for (auto& b: info->buffers) {
		if (!r.U32(&b.source) || !r.U32(&b.descriptor_index) || !r.U32(&b.first_use_pc) || !r.U32(&b.max_byte_extent) || !r.U32(&b.packed_stride) ||
		    !r.Enum(&b.descriptor_format) || !r.U32(&b.descriptor_swizzle) || !r.U32(&b.image_alias) || !r.Bool(&b.read) || !r.Bool(&b.written) ||
		    !r.Bool(&b.atomic) || !r.Bool(&b.formatted) || !r.Bool(&b.scalar) || !r.Bool(&b.live) || !r.U32(&b.indirect_root) ||
		    !r.U32(&b.indirect_mapping_offset) || !r.U32(&b.indirect_search_iterations) || !r.Words(&b.indirect_resources)) {
			return false;
		}
	}
	if (!r.Count(&count, 8)) return false;
	info->images.resize(count);
	for (auto& i: info->images) {
		if (!r.U32(&i.source) || !r.U32(&i.descriptor_index) || !r.U32(&i.first_use_pc) || !r.Enum(&i.resource_class) || !r.Enum(&i.numeric_class) ||
		    !r.Enum(&i.dimension) || !r.Enum(&i.mip_mode) || !r.U32(&i.mip_count) || !r.Enum(&i.conversion_format) || !r.U32(&i.shader_swizzle) ||
		    !r.Bool(&i.read) || !r.Bool(&i.written) || !r.Bool(&i.atomic) || !r.Bool(&i.atomic64) || !r.Bool(&i.unfiltered_access_only) ||
		    !r.Bool(&i.depth_compare) || !r.Bool(&i.cube) || !r.Bool(&i.r128) || !r.Bool(&i.queries_size) || !r.U32(&i.min_lod_resinfo.shift)) {
			return false;
		}
		for (uint32_t& extent: i.min_lod_resinfo.extent) {
			if (!r.U32(&extent)) return false;
		}
		if (!r.U32(&i.min_lod_resinfo.levels) || !r.U32(&i.indirect_root) || !r.U32(&i.indirect_mapping_offset) ||
		    !r.U32(&i.indirect_search_iterations) || !r.Words(&i.indirect_resources)) {
			return false;
		}
	}
	if (!r.Count(&count, 8)) return false;
	info->samplers.resize(count);
	for (auto& s: info->samplers) {
		if (!r.U32(&s.source) || !r.U32(&s.first_use_pc) || !r.U32(&s.snapshot_index) || !r.U32(&s.indirect_root) || !r.U32(&s.indirect_mapping_offset) ||
		    !r.U32(&s.indirect_search_iterations) || !r.Words(&s.indirect_resources) || !r.Bool(&s.force_point_filtering) || !r.Bool(&s.depth_compare) ||
		    !r.Bool(&s.integer_border) || !r.Bool(&s.gather_lod)) {
			return false;
		}
	}
	if (!r.Count(&count, 12)) return false;
	info->sampled_pairs.resize(count);
	for (auto& p: info->sampled_pairs) {
		if (!r.U32(&p.image) || !r.U32(&p.sampler) || !r.U32(&p.first_use_pc)) return false;
	}
	if (!r.Count(&count, 8)) return false;
	info->inputs.resize(count);
	for (auto& in: info->inputs) {
		if (!r.Enum(&in.kind) || !r.U32(&in.location) || !r.U32(&in.component_count) || !r.String(&in.debug_name) || !r.Bool(&in.per_vertex)) return false;
	}
	if (!r.Count(&count, 8)) return false;
	info->outputs.resize(count);
	for (auto& out: info->outputs) {
		if (!r.Enum(&out.kind) || !r.U32(&out.index) || !r.U32(&out.location) || !r.String(&out.debug_name)) return false;
	}
	if (!r.Words(&info->user_data_registers)) return false;
	if (!r.Raw(info->vertex_fetch_components.data(), info->vertex_fetch_components.size()) || !r.I32(&info->vertex_offset_sgpr) ||
	    !r.I32(&info->instance_offset_sgpr)) {
		return false;
	}
	uint64_t bits = 0;
	if (!r.U64(&bits)) return false;
	bool* flags[] = {&info->has_bitwise_xor, &info->uses_dma, &info->uses_swizzle, &info->uses_lds, &info->uses_gds, &info->uses_flattened_srt,
	                 &info->uses_dispatch_threads, &info->bvh, &info->subgroup_ballot, &info->subgroup_vote, &info->subgroup_barrier,
	                 &info->subgroup_shuffle, &info->subgroup_arithmetic, &info->subgroup_local_invocation_id, &info->compute_derivatives,
	                 &info->image_gather_extended, &info->function_lds, &info->function_scratch, &info->pixel_valid_mask, &info->buffer_int64_atomics,
	                 &info->buffer_u8, &info->buffer_u16, &info->shared_int64_atomics, &info->coherent_buffers, &info->float64};
	if ((bits >> std::size(flags)) != 0) return false;
	for (size_t i = 0; i < std::size(flags); i++) *flags[i] = ((bits >> i) & 1u) != 0;
	return true;
}

void Put(Writer& w, const IR::Bindings& b) {
#ifdef KYTY_PERMREC_MUTANT_STALE_BINDINGS
	// the bindings of no emission at all: what a default constructed program would carry
	const IR::Bindings stale;
	const IR::Bindings& use = stale;
#else
	const IR::Bindings& use = b;
#endif
	for (const uint32_t count: use.descriptor_counts) w.U32(count);
	w.U64(use.descriptor_mask);
	w.U32(use.push_data_start_dword);
	w.U32(use.dispatch_thread_dword);
	w.U32(use.memory_offset_dword);
}

bool Get(Reader& r, IR::Bindings* b) {
	for (uint32_t& count: b->descriptor_counts) {
		if (!r.U32(&count)) return false;
	}
	return r.U64(&b->descriptor_mask) && r.U32(&b->push_data_start_dword) && r.U32(&b->dispatch_thread_dword) && r.U32(&b->memory_offset_dword);
}

void Put(Writer& w, const IR::CompiledShaderInfo& c) {
	w.Enum(c.stage);
	w.U64(c.shader_hash);
	w.U32(c.wave_size);
	w.U32(c.user_data_base);
	w.U32(c.user_data_count);
	w.U32(c.scratch_dwords);
	w.U32(c.param_export_mask);
	w.Bool(c.has_address_writes);
	Put(w, c.info);
	Put(w, c.bindings);
}

bool Get(Reader& r, IR::CompiledShaderInfo* c) {
	return r.Enum(&c->stage) && r.U64(&c->shader_hash) && r.U32(&c->wave_size) && r.U32(&c->user_data_base) && r.U32(&c->user_data_count) &&
	       r.U32(&c->scratch_dwords) && r.U32(&c->param_export_mask) && r.Bool(&c->has_address_writes) && Get(r, &c->info) && Get(r, &c->bindings);
}

// ---- per-vertex ----------------------------------------------------------------------------------------------------------------------------

void Put(Writer& w, const PvRecord& pv) {
#ifdef KYTY_PERMREC_MUTANT_DROP_PV_RECORD
	w.Bool(false), w.Bool(false);
	return;
#endif
	w.Bool(pv.original.has_value());
	if (pv.original) {
		w.U32(pv.original->stage);
		w.Bool(pv.original->per_vertex);
		w.Words(pv.original->capture_words);
	}
	w.Bool(pv.store.has_value());
	if (pv.store) {
		w.U32(pv.store->stage);
		w.Words(pv.store->words);
		w.U32(pv.store->mask);
		w.U32(pv.store->corners);
		w.U64(pv.store->hash);
		w.U32(pv.store->compact_spec_id);
	}
}

bool Get(Reader& r, PvRecord* pv) {
	bool present = false;
	if (!r.Bool(&present)) return false;
	if (present) {
		PvOriginalRecord original;
		if (!r.U32(&original.stage) || !r.Bool(&original.per_vertex) || !r.Words(&original.capture_words)) return false;
		pv->original = std::move(original);
	}
	if (!r.Bool(&present)) return false;
	if (present) {
		PvStoreRecord store;
		if (!r.U32(&store.stage) || !r.Words(&store.words) || !r.U32(&store.mask) || !r.U32(&store.corners) || !r.U64(&store.hash) ||
		    !r.U32(&store.compact_spec_id)) {
			return false;
		}
		pv->store = std::move(store);
	}
	return true;
}

} // namespace

std::vector<uint8_t> SerializeRequest(const IR::ResourceSpecialization& specialization, uint32_t push_data_start_dword) {
	Writer w;
	Put(w, specialization);
#ifndef KYTY_PERMREC_MUTANT_IGNORE_PUSH_START
	w.U32(push_data_start_dword);
#else
	(void)push_data_start_dword;
#endif
	return std::move(w.Bytes());
}

std::vector<uint8_t> SerializeRecord(const PermutationRecord& record) {
	Writer w;
	Put(w, record.specialization);
	w.U32(record.push_data_start_dword);
	w.Words(record.spirv);
	Put(w, record.info);
	Put(w, record.pv);
	return std::move(w.Bytes());
}

bool DeserializeRecord(std::span<const uint8_t> payload, PermutationRecord* record) {
	Reader            r(payload);
	PermutationRecord out;
	if (!Get(r, &out.specialization) || !r.U32(&out.push_data_start_dword) || !r.Words(&out.spirv) || !Get(r, &out.info) || !Get(r, &out.pv) ||
	    r.Remaining() != 0) {
		return false;
	}
	*record = std::move(out);
	return true;
}

bool Answers(const PermutationRecord& record, const IR::ResourceSpecialization& specialization, uint32_t push_data_start_dword) {
#ifdef KYTY_PERMREC_MUTANT_SKIP_REQUEST_CHECK
	(void)record, (void)specialization, (void)push_data_start_dword;
	return true;
#else
#ifdef KYTY_PERMREC_MUTANT_IGNORE_SPEC_COMPARE
	const bool same_spec = true;
#else
	const bool same_spec = record.specialization == specialization;
#endif
#ifdef KYTY_PERMREC_MUTANT_IGNORE_PUSH_START
	const bool same_push = true;
#else
	const bool same_push = record.push_data_start_dword == push_data_start_dword;
#endif
	return same_spec && same_push;
#endif
}

std::string DescribeDifference(std::span<const uint8_t> stored, std::span<const uint8_t> fresh) {
	PermutationRecord a, b;
	if (!DeserializeRecord(stored, &a)) return "stored payload does not decode";
	if (!DeserializeRecord(fresh, &b)) return "fresh payload does not decode";
	char text[160];
	if (!(a.specialization == b.specialization) || a.push_data_start_dword != b.push_data_start_dword) return "request";
	if (a.spirv != b.spirv) {
		size_t at = 0;
		while (at < a.spirv.size() && at < b.spirv.size() && a.spirv[at] == b.spirv[at]) at++;
		std::snprintf(text, sizeof(text), "spirv word %zu (stored words=%zu a=%08x, fresh words=%zu b=%08x)", at, a.spirv.size(),
		              at < a.spirv.size() ? a.spirv[at] : 0u, b.spirv.size(), at < b.spirv.size() ? b.spirv[at] : 0u);
		return text;
	}
	if (!(a.info.info == b.info.info)) return "info";
	if (!(a.info.bindings == b.info.bindings)) return "bindings";
	if (a.info.stage != b.info.stage || a.info.shader_hash != b.info.shader_hash || a.info.wave_size != b.info.wave_size ||
	    a.info.user_data_base != b.info.user_data_base || a.info.user_data_count != b.info.user_data_count ||
	    a.info.scratch_dwords != b.info.scratch_dwords || a.info.param_export_mask != b.info.param_export_mask ||
	    a.info.has_address_writes != b.info.has_address_writes) {
		return "compiled info scalars";
	}
	if (a.pv.original != b.pv.original) return "pv.original";
	if (a.pv.store != b.pv.store) return "pv.store";
	return "same";
}

Request MakeRequest(const std::filesystem::path& dir, const Key& fingerprint, const ProgramIdentity& identity,
                    const IR::ResourceSpecialization& specialization, uint32_t push_data_start_dword) {
	Request request;
	request.stage       = identity.stage;
	request.program     = MakeProgramKey(fingerprint, identity);
	request.permutation = MakePermutationKey(request.program, SerializeRequest(specialization, push_data_start_dword));
	request.path        = FilePath(dir, identity.stage, request.program, request.permutation);
	return request;
}

Found Find(const Request& request, const IR::ResourceSpecialization& specialization, uint32_t push_data_start_dword, PermutationRecord* record) {
	bool       rejected = false;
	const auto payload  = Load(request.path, request.program, request.permutation, &rejected);
	if (!payload) return rejected ? Found::Rejected : Found::Miss;
	PermutationRecord decoded;
	if (!DeserializeRecord(*payload, &decoded) || !Answers(decoded, specialization, push_data_start_dword)) return Found::Rejected;
	*record = std::move(decoded);
	return Found::Hit;
}

std::vector<uint8_t> MakeFileBytes(const Request& request, const PermutationRecord& record) {
	return SerializeFile(request.program, request.permutation, SerializeRecord(record));
}

Verdict VerifyAgainst(const Request& request, const PermutationRecord& fresh, std::string* difference) {
	bool       rejected = false;
	const auto stored   = Load(request.path, request.program, request.permutation, &rejected);
	if (!stored) return rejected ? Verdict::Rejected : Verdict::NoFile;
	const auto fresh_bytes = SerializeRecord(fresh);
	if (*stored == fresh_bytes) return Verdict::Equal;
	if (difference != nullptr) *difference = DescribeDifference(*stored, fresh_bytes);
	return Verdict::NotEqual;
}

} // namespace Libs::Graphics::ProgramDisk
