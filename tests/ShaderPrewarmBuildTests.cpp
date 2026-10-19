// S435 (S446): the translation that the worker thread makes from a manifest entry (PrewarmBuild: the code copied at the mapping, the static state, the raw bytes of
// the stage input info) must give what the command processor's own translation of the same draw gives: the same resource plan and, with the same specialization,
// the same SPIR-V. A compute shader with an input info that is not the default is translated both ways. The mutant (whole emulator rebuilt with
// KYTY_PREWARM_MUTANT_BUILD_DROPS_INFO) translates with the default input info and must fail.

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderPrewarm.h"
#include "graphics/shader/shaderPrewarmBuild.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace Libs::Graphics;
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

constexpr uint32_t kUserDataCount = 10;
constexpr uint32_t kPushStart     = 6;
// A compute pattern fill whose bounds come from the dispatch thread dimensions (the same words as shaderCfgTests.cpp's thread_dimension_shader): what the
// translation emits depends on the input info (dispatch thread dimensions, thread ids, group ids), so a translation made with another info shows.
const uint32_t kCode[] = {
    0xbfa00002u, 0x7e040c09u, 0x81ea0980u, 0xd7460000u,
    0x04010c0au, 0x7e020205u, 0x7e045702u, 0x100404ffu,
    0x4f7ffffeu, 0x7e040f02u, 0x7ed60502u, 0x936a6b6au,
    0x9aea6b6au, 0x4a04046au, 0xd56a0002u, 0x00020102u,
    0xd5690002u, 0x00020409u, 0x4c040500u, 0x7d860409u,
    0x4e060409u, 0x02040702u, 0x7d860409u, 0x4e060409u,
    0x02040702u, 0x7d840481u, 0x7d8404f9u, 0x06868882u,
    0x7d8404f9u, 0x06868a83u, 0x02020204u, 0xd5010001u,
    0x00200d01u, 0xd5010001u, 0x00280f01u, 0xe0102000u,
    0x80000100u, 0xbf810000u,
};
// The user data of the draw (the direct buffer descriptor and the dispatch constants); the worker translates with zeros of the same count.
const std::vector<uint32_t> kDrawUserData = {0x00200000u, 4u << 16, 64u, 0x14204u, 0xff000000u, 0u, 0u, 0x08000000u, 64u, 1u};

ShaderComputeInputInfo MakeInfo(bool distinctive) {
	ShaderComputeInputInfo info {};
	if (!distinctive) return info;
	info.threads_num[0]             = 64;
	info.threads_num[1]             = 1;
	info.threads_num[2]             = 1;
	info.workgroup_register         = 10;
	info.thread_ids_num             = 1;
	info.group_id[0]                = true;
	info.dispatch_thread_dimensions = true;
	info.wave_size                  = 64;
	return info;
}

ShaderRecompiler::CompileOptions Options(const ShaderComputeInputInfo& info, const std::vector<uint32_t>& user_data) {
	ShaderRecompiler::CompileOptions options;
	options.stage              = ShaderType::Compute;
	options.wave_size          = 64;
	options.shader_hash        = 0x77aa;
	options.user_data          = user_data;
	options.input_info.compute = &info;
	options.dump_ir            = false;
	return options;
}

// What the command processor does for a new program: translate with the draw's input info, extract the plan, materialize, emit.
std::vector<uint32_t> OwnSpirv(const ShaderComputeInputInfo& info, uint64_t* plan_mask = nullptr, std::string* ir = nullptr) {
	const auto                  options    = Options(info, kDrawUserData);
	auto                        translated = ShaderRecompiler::TranslateProgram(kCode, options);
	if (ir != nullptr) *ir = IR::ProgramToString(translated.program); // before the plan extraction empties the blocks
	auto                        plan       = IR::ExtractResourcePlan(translated.program);
	if (plan_mask != nullptr) *plan_mask = IR::PlanUsedUserDataMask(plan);
	IR::ResourceSnapshot       resources;
	IR::ResourceSpecialization specialization;
	const IR::SrtRuntime       runtime {.user_data = options.user_data, .shader_base = reinterpret_cast<uint64_t>(kCode)};
	Check(IR::MaterializeResources(plan, runtime, resources, specialization), "own: the resources materialize");
	return ShaderRecompiler::CompileProgram(std::move(translated), options, specialization, kPushStart).spirv;
}

ShaderPrewarm::ManifestEntry MakeEntry(const ShaderComputeInputInfo& info) {
	ShaderPrewarm::ManifestEntry entry;
	entry.stage           = static_cast<uint32_t>(ShaderType::Compute);
	entry.code_hash       = 0x77aa;
	entry.user_data_count = kUserDataCount;
	entry.code_size_words = static_cast<uint32_t>(std::size(kCode));
	entry.wave_size       = 64;
	BuildStageStaticKey(info, entry.static_state);
	entry.input_info = PrewarmInfoBytes(info);
	return entry;
}

// The command processor takes the worker's translation and plan and emits with its own options and the specialization it materialized.
std::vector<uint32_t> PrewarmedSpirv(PrewarmProgram& built, const ShaderComputeInputInfo& draw_info, uint64_t* plan_mask = nullptr) {
	const auto                  options = Options(draw_info, kDrawUserData);
	if (plan_mask != nullptr) *plan_mask = IR::PlanUsedUserDataMask(built.plan);
	IR::ResourceSnapshot       resources;
	IR::ResourceSpecialization specialization;
	const IR::SrtRuntime       runtime {.user_data = options.user_data, .shader_base = reinterpret_cast<uint64_t>(kCode)};
	Check(IR::MaterializeResources(built.plan, runtime, resources, specialization), "prewarmed: the resources materialize");
	return ShaderRecompiler::CompileProgram(std::move(built.translated), options, specialization, kPushStart).spirv;
}

void TestSameAsOwn() {
	const auto draw    = MakeInfo(true);
	const auto other   = MakeInfo(false);
	std::string own_ir, default_ir;
	const auto own     = OwnSpirv(draw, nullptr, &own_ir);
	const auto default_info_spirv = OwnSpirv(other, nullptr, &default_ir);
	Check(!own.empty(), "own: SPIR-V produced");
	Check(own != default_info_spirv, "the input info moves the SPIR-V");
	Check(own_ir != default_ir, "the input info moves the translated IR (otherwise this test cannot tell the worker's translation from a default one)");

	// Built on a worker thread of the mailbox, as in the emulator.
	ShaderPrewarm::Mailbox<std::unique_ptr<PrewarmProgram>> box(1);
	const auto                                              entry = MakeEntry(draw);
	std::thread::id                                         ran_on;
	const std::vector<uint32_t>                             code(std::begin(kCode), std::end(kCode));
	box.Submit(ShaderPrewarm::EntryKey(entry), [&]() -> std::optional<std::unique_ptr<PrewarmProgram>> {
		ran_on = std::this_thread::get_id();
		auto built = PrewarmBuild(entry, code, true);
		if (!built) return std::nullopt;
		return built;
	});
	box.Drain(); // the command processor comes after the worker finished (a Claim of a queued job would cancel it)
	auto claimed = box.Claim(ShaderPrewarm::EntryKey(entry));
	Check(claimed.payload && *claimed.payload, "the worker's translation is there");
	Check(ran_on != std::thread::id() && ran_on != std::this_thread::get_id(), "the translation ran on the worker thread");
	if (!claimed.payload || !*claimed.payload) return;
	auto&    built = **claimed.payload;
	Check(built.entry == entry, "the result carries the entry it was built for");
	uint64_t own_mask = 0, built_mask = 0;
	const auto built_ir = built.ir_text;
	(void)OwnSpirv(draw, &own_mask);
	const auto prewarmed = PrewarmedSpirv(built, draw, &built_mask);
	Check(own_mask == built_mask, "the same resource plan (used user data mask)");
	Check(built_ir == own_ir, "the worker's translation is the same IR as the command processor's own");
	Check(ShaderPrewarm::CompareSpirv(own, prewarmed).empty(), "the worker's translation emits the same SPIR-V as the command processor's own");
}

void TestRefusals() {
	const auto                  draw = MakeInfo(true);
	const std::vector<uint32_t> code(std::begin(kCode), std::end(kCode));
	auto                        entry = MakeEntry(draw);
	Check(PrewarmBuild(entry, code) != nullptr, "the entry builds");
	auto short_info = entry;
	short_info.input_info.pop_back();
	Check(PrewarmBuild(short_info, code) == nullptr, "an input info of another size (another build of the structs) is refused");
	auto wrong_size = entry;
	wrong_size.code_size_words++;
	Check(PrewarmBuild(wrong_size, code) == nullptr, "a code of another length than the entry is refused");
	auto wrong_stage = entry;
	wrong_stage.stage = 0x7fffu;
	Check(PrewarmBuild(wrong_stage, code) == nullptr, "an unknown stage is refused");
	const auto layout = PrewarmLayout();
	Check(layout.compute_info_size == sizeof(ShaderComputeInputInfo) && layout.vertex_info_size == sizeof(ShaderVertexInputInfo) &&
	          layout.pixel_info_size == sizeof(ShaderPixelInputInfo),
	      "the manifest layout is the sizes of the three input infos");
}

} // namespace

int main() {
	setenv("KYTY_SHADER_LOOP_LIMIT", "0", 1);
	EnsureConfigInitialized();
	TestSameAsOwn();
	TestRefusals();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("ShaderPrewarmBuildTests: OK");
	return 0;
}
