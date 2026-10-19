#include "graphics/shader/shaderPrewarmBuild.h"

#include <cstring>

namespace Libs::Graphics {

namespace {

const char* PrewarmStageLabel(ShaderType stage, const char** name) {
	switch (stage) {
		case ShaderType::Vertex: *name = "vs"; return "ShaderRecompiler VS";
		case ShaderType::Mesh: *name = "ms"; return "ShaderRecompiler MS";
		case ShaderType::Local: *name = "ls"; return "ShaderRecompiler LS";
		case ShaderType::TessellationControl: *name = "hs"; return "ShaderRecompiler HS";
		case ShaderType::TessellationEvaluation: *name = "ds"; return "ShaderRecompiler DS";
		case ShaderType::Pixel: *name = "ps"; return "ShaderRecompiler PS";
		case ShaderType::Compute: *name = "cs"; return "ShaderRecompiler CS";
		default: *name = nullptr; return nullptr;
	}
}

} // namespace

ShaderPrewarm::ManifestLayout PrewarmLayout() {
	return {static_cast<uint32_t>(sizeof(ShaderVertexInputInfo)), static_cast<uint32_t>(sizeof(ShaderPixelInputInfo)),
	        static_cast<uint32_t>(sizeof(ShaderComputeInputInfo))};
}

// The worker's part. Runs on a worker thread: reads only `entry` and `code` (copied at ShaderMapUserData), writes only its result; touches no
// ProgramCache state, no Vulkan, no guest memory. nullptr: the program is not pre-warmable (call source) or the entry does not fit this build.
std::unique_ptr<PrewarmProgram> PrewarmBuild(const ShaderPrewarm::ManifestEntry& entry, std::span<const uint32_t> code, bool with_ir_text) {
	const auto  stage = static_cast<ShaderType>(entry.stage);
	const char* stage_name = nullptr;
	const char* label      = PrewarmStageLabel(stage, &stage_name);
	if (label == nullptr || entry.code_size_words != code.size()) return nullptr;

	ShaderVertexInputInfo  vertex;
	ShaderPixelInputInfo   pixel;
	ShaderComputeInputInfo compute;
	ShaderRecompiler::CompileOptions options;
	const auto load = [&](auto& info) {
		if (entry.input_info.size() != sizeof(info)) return false;
#ifndef KYTY_PREWARM_MUTANT_BUILD_DROPS_INFO // the mutant translates with a default input info instead of the stored one
		std::memcpy(static_cast<void*>(&info), entry.input_info.data(), sizeof(info));
#endif
		info.stage = {};
		return true;
	};
	if (stage == ShaderType::Pixel) {
		if (!load(pixel)) return nullptr;
		options.input_info.pixel = &pixel;
	} else if (stage == ShaderType::Compute) {
		if (!load(compute)) return nullptr;
		options.input_info.compute = &compute;
	} else {
		if (!load(vertex)) return nullptr;
		options.input_info.vertex = &vertex;
	}
	const std::vector<uint32_t> user_data(entry.user_data_count, 0u); // the translation reads only the size of the user data (S434)
	options.stage                = stage;
	options.maintenance8_enabled = entry.maintenance8;
	options.shader_hash          = entry.code_hash;
	options.user_data            = user_data;
	options.dump_ir              = false; // the mode is off when the shader log or the debug dump is on
	options.early_dump           = false;
	options.emit_debug_info      = false;
	options.dump_label           = label;
	options.user_data_base       = entry.user_data_base;
	options.wave_size            = entry.wave_size;

	auto result    = std::make_unique<PrewarmProgram>();
	result->entry  = entry;
	auto source    = ShaderRecompiler::PrepareShaderSource(code, options);
	if (source.call) return nullptr; // its IR depends on guest memory
	result->source     = std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source));
	result->translated = ShaderRecompiler::TranslateProgram(result->source->decoded, options);
	if (with_ir_text) result->ir_text = ShaderRecompiler::IR::ProgramToString(result->translated.program); // the plan extraction empties the blocks
	result->plan       = ShaderRecompiler::IR::ExtractResourcePlan(result->translated.program);
	return result;
}

} // namespace Libs::Graphics
