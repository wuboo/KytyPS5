#ifndef KYTY_GRAPHICS_SHADER_SHADERPREWARMBUILD_H_
#define KYTY_GRAPHICS_SHADER_SHADERPREWARMBUILD_H_

// S435 (KYTY_SHADER_PREWARM): the part of the first use of a shader that the worker thread does (shaderPrewarm.h has the hand-over and the manifest).

#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderPrewarm.h"

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <cstring>
#include <vector>

namespace Libs::Graphics {

// What the worker leaves for the command processor - the source (kept alive: the translation may refer to its decoded program), the translation
// and the resource plan - under the manifest entry it was built for.
struct PrewarmProgram {
	ShaderPrewarm::ManifestEntry                    entry;
	std::unique_ptr<ShaderRecompiler::ShaderSource> source;
	ShaderRecompiler::TranslateResult               translated;
	ShaderRecompiler::IR::ResourcePlan              plan;
	std::string                                     ir_text; // ProgramToString of the translation before the plan is extracted (shadow only)
};

[[nodiscard]] ShaderPrewarm::ManifestLayout PrewarmLayout();

// The raw bytes of a stage input info as the manifest keeps them (the runtime pointers of the draw cleared).
template <typename InputInfo>
[[nodiscard]] std::vector<uint8_t> PrewarmInfoBytes(const InputInfo& info) {
	static_assert(std::is_trivially_copyable_v<InputInfo>, "the input info is stored as raw bytes");
	InputInfo copy = info;
	copy.stage     = {};
	std::vector<uint8_t> bytes(sizeof(InputInfo));
	std::memcpy(bytes.data(), &copy, sizeof(InputInfo));
	return bytes;
}

// Runs on a worker thread: reads only `entry` and `code` (copied at ShaderMapUserData), writes only its result; touches no ProgramCache state, no
// Vulkan, no guest memory. nullptr: the program is not pre-warmable (call source) or the entry does not fit this build (stage, sizes).
[[nodiscard]] std::unique_ptr<PrewarmProgram> PrewarmBuild(const ShaderPrewarm::ManifestEntry& entry, std::span<const uint32_t> code, bool with_ir_text = false);

} // namespace Libs::Graphics

#endif /* KYTY_GRAPHICS_SHADER_SHADERPREWARMBUILD_H_ */
