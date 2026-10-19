// S434 (S443): the record of KYTY_PROGRAM_DISK_CACHE stage 1. A record with every field set to a value of its own must come back equal from its
// bytes (and from a file); a record must answer exactly the request it was made for (specialization and push data start, compared field by field);
// a file that decodes to another request, is damaged, or is of another key must be refused; verify must see a changed word and name it.
// Mutants: see src/graphics/shader/permutationRecord.h.

#include "graphics/shader/permutationRecord.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <unistd.h>

using namespace Libs::Graphics;
using namespace Libs::Graphics::ProgramDisk;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

struct Counter {
	uint32_t value = 100;
	uint32_t Next() { return ++value; }
};

IR::ResourceSpecialization MakeSpecialization(uint32_t seed) {
	Counter c {seed};
	IR::ResourceSpecialization spec;
	for (int i = 0; i < 3; i++) {
		IR::ResourceSpecialization::Buffer b;
		b.packed_stride            = c.Next();
		b.descriptor_format        = static_cast<Prospero::BufferFormat>(i + 1);
		b.descriptor_swizzle       = c.Next();
		b.zero_stride_oob          = (i & 1) != 0;
		b.scalar_empty             = (i & 2) != 0;
		b.indirect_root            = c.Next();
		b.indirect_mapping_offset  = c.Next();
		b.indirect_search_iterations = c.Next();
		spec.buffers.push_back(b);
	}
	for (int i = 0; i < 2; i++) {
		IR::ResourceSpecialization::Image im;
		im.numeric_class              = static_cast<Prospero::TextureNumericClass>(1 + i);
		im.dimension                  = static_cast<Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension>(2 + i);
		im.mip_count                  = c.Next();
		im.conversion_format          = static_cast<Prospero::BufferFormat>(3 + i);
		im.shader_swizzle             = c.Next();
		im.indirect_root              = c.Next();
		im.indirect_mapping_offset    = c.Next();
		im.indirect_search_iterations = c.Next();
		im.cube                       = i == 0;
		im.fmask                      = i == 1;
		im.min_lod_resinfo.shift      = c.Next();
		im.min_lod_resinfo.extent[0]  = c.Next();
		im.min_lod_resinfo.extent[1]  = c.Next();
		im.min_lod_resinfo.extent[2]  = c.Next();
		im.min_lod_resinfo.levels     = c.Next();
		spec.images.push_back(im);
	}
	for (int i = 0; i < 2; i++) {
		spec.samplers.push_back({c.Next(), c.Next(), c.Next()});
	}
	return spec;
}

PermutationRecord MakeRecord(uint32_t seed, uint32_t push) {
	Counter           c {seed + 1000};
	PermutationRecord r;
	r.specialization        = MakeSpecialization(seed);
	r.push_data_start_dword = push;
	for (uint32_t i = 0; i < 50; i++) r.spirv.push_back(0x07230000u + i * 31 + seed);

	auto& ci             = r.info;
	ci.stage             = ShaderType::Pixel;
	ci.shader_hash       = 0x1122334455667788ull + seed;
	ci.wave_size         = 32;
	ci.user_data_base    = 8;
	ci.user_data_count   = 12;
	ci.scratch_dwords    = c.Next();
	ci.param_export_mask = 0x00f0f00fu;
	ci.has_address_writes = true;
	auto& info           = ci.info;
	for (int i = 0; i < 3; i++) {
		IR::BufferResource b;
		b.source = c.Next(), b.descriptor_index = c.Next(), b.first_use_pc = c.Next(), b.max_byte_extent = c.Next(), b.packed_stride = c.Next();
		b.descriptor_format = static_cast<Prospero::BufferFormat>(2 + i), b.descriptor_swizzle = c.Next(), b.image_alias = c.Next();
		b.read = true, b.written = (i & 1) != 0, b.atomic = (i & 2) != 0, b.formatted = i == 0, b.scalar = i == 1, b.live = true;
		b.indirect_root = c.Next(), b.indirect_mapping_offset = c.Next(), b.indirect_search_iterations = c.Next();
		for (int k = 0; k <= i; k++) b.indirect_resources.push_back(c.Next());
		info.buffers.push_back(b);
	}
	for (int i = 0; i < 2; i++) {
		IR::ImageResource im;
		im.source = c.Next(), im.descriptor_index = c.Next(), im.first_use_pc = c.Next();
		im.resource_class = i == 0 ? IR::ImageResourceClass::Sampled : IR::ImageResourceClass::Storage;
		im.numeric_class  = static_cast<Prospero::TextureNumericClass>(1 + i);
		im.dimension      = static_cast<Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension>(2 + i);
		im.mip_mode       = i == 0 ? IR::ImageMipMode::None : IR::ImageMipMode::Dynamic;
		im.mip_count = c.Next(), im.conversion_format = static_cast<Prospero::BufferFormat>(5 + i), im.shader_swizzle = c.Next();
		im.read = true, im.written = i == 1, im.atomic = i == 1, im.atomic64 = i == 0, im.unfiltered_access_only = i == 1, im.depth_compare = i == 0;
		im.cube = i == 1, im.r128 = i == 0, im.queries_size = true;
		im.min_lod_resinfo.shift = c.Next(), im.min_lod_resinfo.extent[0] = c.Next(), im.min_lod_resinfo.extent[1] = c.Next();
		im.min_lod_resinfo.extent[2] = c.Next(), im.min_lod_resinfo.levels = c.Next();
		im.indirect_root = c.Next(), im.indirect_mapping_offset = c.Next(), im.indirect_search_iterations = c.Next();
		im.indirect_resources = {c.Next(), c.Next()};
		info.images.push_back(im);
	}
	for (int i = 0; i < 2; i++) {
		IR::SamplerResource s;
		s.source = c.Next(), s.first_use_pc = c.Next(), s.snapshot_index = c.Next();
		s.indirect_root = c.Next(), s.indirect_mapping_offset = c.Next(), s.indirect_search_iterations = c.Next();
		s.indirect_resources = {c.Next()};
		s.force_point_filtering = i == 0, s.depth_compare = i == 1, s.integer_border = i == 0, s.gather_lod = i == 1;
		info.samplers.push_back(s);
	}
	info.sampled_pairs = {{c.Next(), c.Next(), c.Next()}, {c.Next(), c.Next(), c.Next()}};
	{
		IR::StageInput a;
		a.kind = IR::StageInputKind::Parameter, a.location = 3, a.component_count = 4, a.debug_name = "in_color", a.per_vertex = true;
		IR::StageInput b;
		b.kind = IR::StageInputKind::FragCoord, b.location = 0, b.component_count = 2, b.debug_name = "", b.per_vertex = false;
		info.inputs = {a, b};
		IR::StageOutput o;
		o.kind = IR::StageOutputKind::Mrt, o.index = 1, o.location = 2, o.debug_name = "out_target";
		info.outputs = {o};
	}
	info.user_data_registers = {8, 9, 10, 14};
	for (size_t i = 0; i < info.vertex_fetch_components.size(); i++) info.vertex_fetch_components[i] = static_cast<uint8_t>(i * 3 + 1);
	info.vertex_offset_sgpr   = 5;
	info.instance_offset_sgpr = -1;
	info.has_bitwise_xor = true, info.uses_dma = true, info.uses_swizzle = false, info.uses_lds = true, info.uses_gds = false;
	info.uses_flattened_srt = true, info.uses_dispatch_threads = false, info.bvh = true, info.subgroup_ballot = true, info.subgroup_vote = false;
	info.subgroup_barrier = true, info.subgroup_shuffle = false, info.subgroup_arithmetic = true, info.subgroup_local_invocation_id = false;
	info.compute_derivatives = true, info.image_gather_extended = false, info.function_lds = true, info.function_scratch = false;
	info.pixel_valid_mask = true, info.buffer_int64_atomics = false, info.buffer_u8 = true, info.buffer_u16 = false, info.shared_int64_atomics = true;
	info.coherent_buffers = false, info.float64 = true;
	auto& bindings = ci.bindings;
	for (auto& count: bindings.descriptor_counts) count = c.Next();
	bindings.descriptor_mask       = 0x8000000000000005ull;
	bindings.push_data_start_dword = push;
	bindings.dispatch_thread_dword = c.Next();
	bindings.memory_offset_dword   = c.Next();

	PvOriginalRecord original;
	original.stage = static_cast<uint32_t>(ShaderType::Vertex), original.per_vertex = false;
	for (uint32_t i = 0; i < 20; i++) original.capture_words.push_back(0x1000u + i);
	r.pv.original = original;
	PvStoreRecord store;
	store.stage = static_cast<uint32_t>(ShaderType::Vertex);
	for (uint32_t i = 0; i < 30; i++) store.words.push_back(0x2000u + i);
	store.mask = 0x35, store.corners = 5, store.hash = 0xabcdef0123456789ull, store.compact_spec_id = 77;
	r.pv.store = store;
	return r;
}

bool Same(const PermutationRecord& a, const PermutationRecord& b) {
	return a.specialization == b.specialization && a.push_data_start_dword == b.push_data_start_dword && a.spirv == b.spirv &&
	       a.info.stage == b.info.stage && a.info.shader_hash == b.info.shader_hash && a.info.wave_size == b.info.wave_size &&
	       a.info.user_data_base == b.info.user_data_base && a.info.user_data_count == b.info.user_data_count &&
	       a.info.scratch_dwords == b.info.scratch_dwords && a.info.param_export_mask == b.info.param_export_mask &&
	       a.info.has_address_writes == b.info.has_address_writes && a.info.info == b.info.info && a.info.bindings == b.info.bindings && a.pv == b.pv;
}

void TestRoundTrip() {
	const auto record = MakeRecord(1, 6);
	const auto bytes  = SerializeRecord(record);
	PermutationRecord back;
	Check(DeserializeRecord(bytes, &back), "the record decodes");
	Check(back.specialization == record.specialization, "specialization round trips");
	Check(back.spirv == record.spirv, "spirv round trips");
	Check(back.info.info == record.info.info, "ShaderInfo (resources, inputs, outputs, flags) round trips");
	Check(back.info.info.buffers == record.info.info.buffers && back.info.info.images == record.info.info.images &&
	          back.info.info.samplers == record.info.info.samplers,
	      "the resource vectors round trip");
	Check(back.info.bindings == record.info.bindings, "bindings round trip");
	Check(back.pv == record.pv, "per-vertex records round trip");
	Check(Same(back, record), "the whole record round trips");
	Check(SerializeRecord(back) == bytes, "serialization is deterministic and stable under a round trip");

	// Empty vectors and absent per-vertex records.
	PermutationRecord bare;
	PermutationRecord bare_back;
	Check(DeserializeRecord(SerializeRecord(bare), &bare_back) && Same(bare, bare_back), "an empty record round trips");

	// Short and long payloads are refused.
	bool accepted = false;
	for (size_t size = 0; size < bytes.size(); size += 7) {
		PermutationRecord tmp;
		if (DeserializeRecord(std::span(bytes).first(size), &tmp)) accepted = true;
	}
	PermutationRecord tmp;
	if (DeserializeRecord(std::span(bytes).first(bytes.size() - 1), &tmp)) accepted = true;
	Check(!accepted, "a truncated payload is refused");
	auto longer = bytes;
	longer.push_back(0);
	Check(!DeserializeRecord(longer, &tmp), "a payload with trailing bytes is refused");
}

void TestAnswers() {
	const auto record = MakeRecord(1, 6);
	Check(Answers(record, record.specialization, 6), "a record answers its own request");
	auto other = record.specialization;
	other.buffers[1].packed_stride++;
	Check(!Answers(record, other, 6), "another buffer stride is another request");
	other = record.specialization;
	other.images[0].min_lod_resinfo.extent[2]++;
	Check(!Answers(record, other, 6), "another MinLod extent is another request");
	other = record.specialization;
	other.samplers.pop_back();
	Check(!Answers(record, other, 6), "another sampler count is another request");
	Check(!Answers(record, record.specialization, 7), "another push data start is another request");
	Check(SerializeRequest(record.specialization, 6) != SerializeRequest(record.specialization, 7), "the push data start is in the request bytes");
	Check(SerializeRequest(record.specialization, 6) != SerializeRequest(other, 6), "the specialization is in the request bytes");
}

std::filesystem::path TempDir(const char* name) {
	auto dir = std::filesystem::temp_directory_path() / (std::string("kyty_permrec_") + name + "_" + std::to_string(getpid()));
	std::filesystem::remove_all(dir);
	std::filesystem::create_directories(dir);
	return dir;
}

ProgramIdentity Identity(const std::vector<uint32_t>& state, const std::vector<uint32_t>& code) {
	ProgramIdentity id;
	id.stage = static_cast<uint32_t>(ShaderType::Pixel), id.params_hash = 0x77, id.user_data_count = 12, id.code_size = static_cast<uint32_t>(code.size() * 4);
	id.static_state = state, id.user_data_base = 8, id.wave_size = 32, id.code = code;
	return id;
}

void TestStore() {
	const auto dir = TempDir("store");
	const Key  fingerprint {11, 22};
	const std::vector<uint32_t> state = {1, 2, 3};
	const std::vector<uint32_t> code  = {0xc0de0001, 0xc0de0002};
	const auto record  = MakeRecord(1, 6);
	const auto request = MakeRequest(dir, fingerprint, Identity(state, code), record.specialization, 6);

	PermutationRecord found;
	Check(Find(request, record.specialization, 6, &found) == Found::Miss, "nothing on disk yet: a miss");
	{
		const auto bytes = MakeFileBytes(request, record);
		Check(Save(request.path, bytes), "the file is saved");
	}
	Check(Find(request, record.specialization, 6, &found) == Found::Hit && Same(found, record), "the stored record comes back");
	Check(request.path.parent_path().filename() == "prog_v1", "in prog_v1");

	// Other requests are other files.
	auto spec2 = record.specialization;
	spec2.buffers[0].zero_stride_oob = !spec2.buffers[0].zero_stride_oob;
	const auto request_spec = MakeRequest(dir, fingerprint, Identity(state, code), spec2, 6);
	Check(request_spec.path != request.path && request_spec.permutation != request.permutation, "another specialization is another key");
	Check(Find(request_spec, spec2, 6, &found) == Found::Miss, "another specialization: a miss");
	const auto request_push = MakeRequest(dir, fingerprint, Identity(state, code), record.specialization, 7);
	Check(request_push.permutation != request.permutation && request_push.path != request.path, "another push data start is another key");
	Check(Find(request_push, record.specialization, 7, &found) == Found::Miss, "another push data start: a miss");
	const auto request_fp = MakeRequest(dir, Key {11, 23}, Identity(state, code), record.specialization, 6);
	Check(Find(request_fp, record.specialization, 6, &found) == Found::Miss, "another fingerprint: a miss");
	auto state2 = state;
	state2[2]++;
	const auto request_state = MakeRequest(dir, fingerprint, Identity(state2, code), record.specialization, 6);
	Check(Find(request_state, record.specialization, 6, &found) == Found::Miss, "another static state: a miss");
	auto code2 = code;
	code2[1]++;
	const auto request_code = MakeRequest(dir, fingerprint, Identity(state, code2), record.specialization, 6);
	Check(Find(request_code, record.specialization, 6, &found) == Found::Miss, "other guest code: a miss");

	// A file at the path of this request that holds the record of another request (a copied or renamed file): refused, not adopted.
	{
		const auto foreign = MakeRecord(2, 6);
		Check(Save(request.path, SerializeFile(request.program, request.permutation, SerializeRecord(foreign))), "a foreign record is saved under our key");
		Check(Find(request, record.specialization, 6, &found) == Found::Rejected, "a record that answers another specialization is refused");
		auto other_push = record;
		other_push.push_data_start_dword = 9;
		Check(Save(request.path, SerializeFile(request.program, request.permutation, SerializeRecord(other_push))), "a record of another push start is saved");
		Check(Find(request, record.specialization, 6, &found) == Found::Rejected, "a record that answers another push data start is refused");
	}
	// A damaged file.
	{
		auto bytes = MakeFileBytes(request, record);
		bytes[bytes.size() / 2] ^= 0x40;
		Check(Save(request.path, bytes), "a damaged file is saved");
		Check(Find(request, record.specialization, 6, &found) == Found::Rejected, "a damaged file is refused");
	}
	std::filesystem::remove_all(dir);
}

void TestVerify() {
	const auto dir = TempDir("verify");
	const Key  fingerprint {1, 2};
	const std::vector<uint32_t> state = {4};
	const std::vector<uint32_t> code  = {5, 6, 7};
	const auto record  = MakeRecord(3, 2);
	const auto request = MakeRequest(dir, fingerprint, Identity(state, code), record.specialization, 2);
	std::string difference;
	Check(VerifyAgainst(request, record, &difference) == Verdict::NoFile, "verify: no file yet");
	Check(Save(request.path, MakeFileBytes(request, record)), "save");
	Check(VerifyAgainst(request, record, &difference) == Verdict::Equal, "verify: the same record is equal");

	auto changed = record;
	changed.spirv[17] ^= 0x100;
	Check(VerifyAgainst(request, changed, &difference) == Verdict::NotEqual, "verify: a changed SPIR-V word is not equal");
	Check(difference.find("spirv word 17") != std::string::npos, "verify names the first different word");
	changed = record;
	changed.info.info.images[1].mip_count++;
	Check(VerifyAgainst(request, changed, &difference) == Verdict::NotEqual && difference == "info", "verify: a changed resource is named 'info'");
	changed = record;
	changed.info.bindings.descriptor_counts[2]++;
	Check(VerifyAgainst(request, changed, &difference) == Verdict::NotEqual && difference == "bindings", "verify: changed bindings are named");
	changed = record;
	changed.pv.store->mask ^= 1;
	Check(VerifyAgainst(request, changed, &difference) == Verdict::NotEqual && difference == "pv.store", "verify: a changed VS-store record is named");
	changed = record;
	changed.pv.original->capture_words[3]++;
	Check(VerifyAgainst(request, changed, &difference) == Verdict::NotEqual && difference == "pv.original", "verify: a changed prototype record is named");
	changed = record;
	changed.pv.store.reset();
	Check(VerifyAgainst(request, changed, &difference) == Verdict::NotEqual, "verify: a missing per-vertex record is not equal");

	auto bytes = MakeFileBytes(request, record);
	bytes[40] ^= 1;
	Check(Save(request.path, bytes), "save damaged");
	Check(VerifyAgainst(request, record, &difference) == Verdict::Rejected, "verify: a damaged file is rejected");
	std::filesystem::remove_all(dir);
}

} // namespace

int main() {
	TestRoundTrip();
	TestAnswers();
	TestStore();
	TestVerify();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("PermutationRecordTests: OK");
	return 0;
}
