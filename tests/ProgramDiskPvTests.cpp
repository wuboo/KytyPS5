// S434 (S443): KYTY_PROGRAM_DISK_CACHE with the real recompiler. A compute shader is translated and compiled; the CompiledShaderInfo and the SPIR-V
// go through a record and a file and must come back as the compile produced them (words, bindings, info). The per-vertex records that the
// compile hooks leave behind must survive Import -> Export unchanged (the importer is what a hit uses instead of RememberPerVertexVariants).
// Mutants (WILL_FAIL, whole emulator rebuilt with the define): KYTY_PROGDISK_MUTANT_PV_IMPORT_DROPS_STORE, KYTY_PROGDISK_MUTANT_PV_IMPORT_DROPS_CAPTURE.

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/perVertexPrototype.h"
#include "graphics/host_gpu/renderer/perVertexVsStoreRuntime.h"
#include "graphics/shader/permutationRecord.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shader.h"

#include <array>
#include <cstdio>
#include <cstdlib>
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

void EnsureConfigInitialized() {
	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions options;
	options.printf_direction = Config::LogDirection::Silent;
	Config::Load(options);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();
}

void TestRealCompile() {
	static const ShaderComputeInputInfo compute {};
	static std::array<uint32_t, 64>     user_data {};
	const uint32_t                      code[] = {0xbf810000u}; // s_endpgm

	ShaderRecompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x1234;
	options.user_data          = user_data;
	options.input_info.compute = &compute;
	options.dump_ir            = false;
	auto translated            = ShaderRecompiler::TranslateProgram(code, options);
	auto plan                  = IR::ExtractResourcePlan(translated.program);
	IR::ResourceSnapshot       resources;
	IR::ResourceSpecialization specialization;
	const IR::SrtRuntime       runtime {.user_data = options.user_data, .shader_base = reinterpret_cast<uint64_t>(code)};
	Check(IR::MaterializeResources(plan, runtime, resources, specialization), "the resources materialize");
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), options, specialization, 6);
	Check(!result.spirv.empty(), "the compile produced SPIR-V");

	PermutationRecord record;
	record.specialization        = specialization;
	record.push_data_start_dword = 6;
	record.spirv                 = result.spirv;
	record.info                  = std::move(result.program).TakeCompiledInfo();
	Check(record.info.stage == ShaderType::Compute && record.info.wave_size == 64, "the compiled info is that of the compute shader");

	const auto dir = std::filesystem::temp_directory_path() / ("kyty_progdisk_pv_" + std::to_string(getpid()));
	std::filesystem::remove_all(dir);
	ProgramIdentity identity;
	identity.stage = static_cast<uint32_t>(ShaderType::Compute), identity.params_hash = options.shader_hash, identity.code_size = sizeof(code);
	identity.wave_size = 64, identity.code = std::span<const uint32_t>(code);
	const auto request = MakeRequest(dir, Key {1, 2}, identity, specialization, 6);
	Check(Save(request.path, MakeFileBytes(request, record)), "the file is written");

	PermutationRecord back;
	Check(Find(request, specialization, 6, &back) == Found::Hit, "the file is found");
	Check(back.spirv == record.spirv, "the SPIR-V words come back as compiled");
	Check(back.info.bindings == record.info.bindings, "the bindings come back as compiled");
	Check(back.info.info == record.info.info, "the shader info comes back as compiled");
	Check(back.info.stage == record.info.stage && back.info.shader_hash == record.info.shader_hash &&
	          back.info.user_data_count == record.info.user_data_count && back.info.scratch_dwords == record.info.scratch_dwords,
	      "the stage metadata comes back as compiled");
	std::string difference;
	Check(VerifyAgainst(request, record, &difference) == Verdict::Equal, "verify finds the compile equal to its own file");
	std::filesystem::remove_all(dir);
}

void TestPvImportExport() {
	PvRecord record;
	PvOriginalRecord original;
	original.stage      = static_cast<uint32_t>(ShaderType::Vertex);
	original.per_vertex = false;
	for (uint32_t i = 0; i < 12; i++) original.capture_words.push_back(0xcafe0000u + i);
	record.original = original;
	PvStoreRecord store;
	store.stage = static_cast<uint32_t>(ShaderType::Vertex);
	for (uint32_t i = 0; i < 9; i++) store.words.push_back(0xbeef0000u + i);
	store.mask = 0x1d, store.corners = 3, store.hash = 0x0102030405060708ull, store.compact_spec_id = 41;
	record.store = store;
	std::vector<uint32_t> spirv;
	for (uint32_t i = 0; i < 20; i++) spirv.push_back(0x07230000u + i);

	Check(!ExportPerVertexVariants(910).original && !ExportPerVertexVariants(910).store, "a program that was never compiled has no per-vertex records");
	ImportPerVertexVariants(910, record, spirv);
	const auto back = ExportPerVertexVariants(910);
	Check(back.original.has_value() && back.original->capture_words == original.capture_words && back.original->stage == original.stage &&
	          back.original->per_vertex == original.per_vertex,
	      "the prototype record survives import and export");
	Check(back.store.has_value() && back.store->words == store.words && back.store->mask == store.mask && back.store->corners == store.corners &&
	          back.store->hash == store.hash && back.store->compact_spec_id == store.compact_spec_id && back.store->stage == store.stage,
	      "the VS-store record survives import and export");
	Check(back == record, "the whole per-vertex record is the same");

	// A program with only the store record, and one with none.
	PvRecord only_store;
	only_store.store = store;
	ImportPerVertexVariants(911, only_store, spirv);
	const auto back_store = ExportPerVertexVariants(911);
	Check(!back_store.original.has_value() && back_store.store.has_value(), "only the records that exist are imported");
	ImportPerVertexVariants(912, PvRecord {}, spirv);
	const auto back_none = ExportPerVertexVariants(912);
	Check(!back_none.original.has_value() && !back_none.store.has_value(), "an empty record imports nothing");
}

} // namespace

int main() {
	setenv("KYTY_SHADER_LOOP_LIMIT", "0", 1);
	EnsureConfigInitialized();
	TestRealCompile();
	TestPvImportExport();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("ProgramDiskPvTests: OK");
	return 0;
}
