// S432: KYTY_PV_LAYOUT_FAST. The vertex layout of the per-vertex prototype read straight from the SPIR-V words must equal the one derived from the
// friendly-names disassembly (the reference), for every shape of output interface the emitter produces and for the corner cases of the scan.
// Also prints the cost of both routes on a vertex shader of ~5000 instructions. Mutants: KYTY_PVLAYOUT_MUTANT_{DROP_ATTR, CLIP_IGNORED,
// KEEP_BUILTIN_VARS, NO_OUTPUT_FILTER}.

#include "graphics/host_gpu/renderer/perVertexTransform.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <spirv-tools/libspirv.hpp>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

void Check(bool condition, const std::string& message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message.c_str());
		g_failures++;
	}
}

constexpr const char* kTypes = R"(
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%uint = OpTypeInt 32 0
%uint_4 = OpConstant %uint 4
%arr = OpTypeArray %float %uint_4
%ptr_out_v4 = OpTypePointer Output %v4float
%ptr_in_v4 = OpTypePointer Input %v4float
%ptr_out_arr = OpTypePointer Output %arr
%zero = OpConstantNull %v4float
)";

struct Variant {
	const char* name;
	std::string interface_text; // OpName / OpDecorate lines
	std::string variables;      // OpVariable lines
	std::string interface_ids;  // ids for the OpEntryPoint
	bool        clip;
};

std::vector<uint32_t> Assemble(const std::string& text) {
	spvtools::SpirvTools  tools(SPV_ENV_VULKAN_1_2);
	std::vector<uint32_t> words;
	if (!tools.Assemble(text, &words, SPV_TEXT_TO_BINARY_OPTION_NONE)) {
		std::fprintf(stderr, "assembly failed for:\n%s\n", text.c_str());
		std::exit(2);
	}
	return words;
}

std::string VertexShader(const Variant& v, int body_adds = 0) {
	std::string text = "OpCapability Shader\n";
	if (v.clip) text += "OpCapability ClipDistance\n";
	text += "OpMemoryModel Logical GLSL450\nOpEntryPoint Vertex %main \"main\" " + v.interface_ids + "\nOpName %main \"main\"\n";
	text += v.interface_text;
	text += kTypes;
	text += v.variables;
	text += "%main = OpFunction %void None %fn\n%entry = OpLabel\n%a = OpLoad %v4float %in_pos\n";
	std::string previous = "%a";
	for (int i = 0; i < body_adds; i++) {
		const std::string id = "%t" + std::to_string(i);
		text += id + " = OpFAdd %v4float " + previous + " %a\n";
		previous = id;
	}
	text += "OpStore %param0 " + previous + "\nOpReturn\nOpFunctionEnd\n";
	return text;
}

const std::string kBaseNames = "OpName %in_pos \"in_pos\"\nOpName %outPerVertex \"outPerVertex\"\nOpName %param0 \"param0\"\n";
const std::string kBaseVars  = "%in_pos = OpVariable %ptr_in_v4 Input\n%outPerVertex = OpVariable %ptr_out_v4 Output\n%param0 = OpVariable %ptr_out_v4 Output\n";

std::vector<Variant> Variants() {
	std::vector<Variant> out;
	out.push_back({"params and clip distance",
	               kBaseNames + "OpName %param3 \"param3\"\nOpName %gl_ClipDistance \"gl_ClipDistance\"\nOpName %gl_PointSize \"gl_PointSize\"\n"
	                            "OpDecorate %in_pos Location 0\nOpDecorate %param0 Location 0\nOpDecorate %param3 Location 3\n"
	                            "OpDecorate %gl_ClipDistance BuiltIn ClipDistance\nOpDecorate %outPerVertex Location 8\nOpDecorate %gl_PointSize Location 9\n",
	               kBaseVars + "%param3 = OpVariable %ptr_out_v4 Output\n%gl_ClipDistance = OpVariable %ptr_out_arr Output\n%gl_PointSize = OpVariable %ptr_out_v4 Output\n",
	               "%in_pos %outPerVertex %param0 %param3 %gl_ClipDistance %gl_PointSize", true});
	out.push_back({"no clip distance",
	               kBaseNames + "OpName %param5 \"param5\"\nOpDecorate %param0 Location 0\nOpDecorate %param5 Location 5\n",
	               kBaseVars + "%param5 = OpVariable %ptr_out_v4 Output\n", "%in_pos %outPerVertex %param0 %param5", false});
	out.push_back({"input with a location is not an output",
	               kBaseNames + "OpName %in2 \"in2\"\nOpDecorate %in_pos Location 0\nOpDecorate %in2 Location 2\nOpDecorate %param0 Location 1\n",
	               kBaseVars + "%in2 = OpVariable %ptr_in_v4 Input\n", "%in_pos %in2 %outPerVertex %param0", false});
	out.push_back({"sanitized and unnamed variables",
	               kBaseNames + "OpName %dotted \"my.param\"\nOpDecorate %param0 Location 0\nOpDecorate %dotted Location 4\nOpDecorate %unnamed Location 6\n",
	               kBaseVars + "%dotted = OpVariable %ptr_out_v4 Output\n%unnamed = OpVariable %ptr_out_v4 Output\n", "%in_pos %outPerVertex %param0 %dotted %unnamed",
	               false});
	out.push_back({"output with an initializer is not matched",
	               kBaseNames + "OpName %init \"init\"\nOpDecorate %param0 Location 0\nOpDecorate %init Location 2\n",
	               kBaseVars + "%init = OpVariable %ptr_out_v4 Output %zero\n", "%in_pos %outPerVertex %param0 %init", false});
	out.push_back({"output without a location",
	               kBaseNames + "OpName %nolocation \"nolocation\"\nOpDecorate %param0 Location 0\n",
	               kBaseVars + "%nolocation = OpVariable %ptr_out_v4 Output\n", "%in_pos %outPerVertex %param0 %nolocation", false});
	out.push_back({"location 31 is reserved",
	               kBaseNames + "OpName %p31 \"p31\"\nOpDecorate %param0 Location 0\nOpDecorate %p31 Location 31\n",
	               kBaseVars + "%p31 = OpVariable %ptr_out_v4 Output\n", "%in_pos %outPerVertex %param0 %p31", false});
	out.push_back({"no outputs at all", "OpName %in_pos \"in_pos\"\nOpDecorate %in_pos Location 0\n",
	               "%in_pos = OpVariable %ptr_in_v4 Input\n%param0 = OpVariable %ptr_in_v4 Input\n", "%in_pos %param0", false});
	return out;
}

bool SameLayout(const PerVertexLayout& a, const PerVertexLayout& b) {
	return a.num_params == b.num_params && a.has_clip == b.has_clip && a.clip_slot == b.clip_slot && a.record_stride_vec4 == b.record_stride_vec4 &&
	       a.location_to_slot == b.location_to_slot && a.param_locations_and_slots == b.param_locations_and_slots;
}

struct Derived {
	bool                            ok = false;
	PerVertexLayout                 layout;
	std::map<uint32_t, std::string> params;
};

Derived ViaDisassembly(const std::vector<uint32_t>& words) {
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
	std::string          source;
	Check(tools.Disassemble(words, &source, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES), "disassemble");
	Derived out;
	out.ok = DerivePerVertexVertexLayout(source, out.layout, out.params);
	return out;
}

Derived ViaWords(const std::vector<uint32_t>& words) {
	Derived out;
	out.ok = DerivePerVertexVertexLayoutFromSpirv(words, out.layout, out.params);
	return out;
}

void TestEquivalence() {
	for (const auto& variant: Variants()) {
		const auto words = Assemble(VertexShader(variant));
		const auto slow  = ViaDisassembly(words);
		const auto fast  = ViaWords(words);
		Check(slow.ok == fast.ok, std::string("verdict equal: ") + variant.name);
		if (slow.ok && fast.ok) {
			Check(SameLayout(slow.layout, fast.layout), std::string("layout equal: ") + variant.name);
			Check(slow.params == fast.params, std::string("parameter variables equal: ") + variant.name);
		}
	}
	// The shapes are not vacuous: the first one has two parameters and a clip distance, and the verdicts differ across the set.
	const auto first = ViaWords(Assemble(VertexShader(Variants()[0])));
	Check(first.ok && first.layout.num_params == 2 && first.layout.has_clip && first.layout.clip_slot == 3 && first.layout.record_stride_vec4 == 4,
	      "reference shape: 2 parameters, clip slot 3, stride 4");
	const auto reserved = ViaWords(Assemble(VertexShader(Variants()[6])));
	Check(!reserved.ok, "location 31 refused");
	const auto inputs = ViaWords(Assemble(VertexShader(Variants()[2])));
	Check(inputs.ok && inputs.layout.num_params == 1 && inputs.layout.location_to_slot.count(2) == 0, "an Input variable is not a parameter");
	const auto sanitized = ViaWords(Assemble(VertexShader(Variants()[3])));
	Check(sanitized.ok && sanitized.params.count(4) == 1 && sanitized.params.at(4) == "%my_param", "friendly name of a dotted OpName");
}

void TestMalformedInput() {
	std::vector<uint32_t> words = Assemble(VertexShader(Variants()[0]));
	PerVertexLayout       layout;
	std::map<uint32_t, std::string> params;
	std::vector<uint32_t> truncated(words.begin(), words.begin() + 3);
	Check(!DerivePerVertexVertexLayoutFromSpirv(truncated, layout, params), "a header-only module has no outputs");
	std::vector<uint32_t> bad_magic = words;
	bad_magic[0] ^= 1;
	Check(!DerivePerVertexVertexLayoutFromSpirv(bad_magic, layout, params), "bad magic refused");
	std::vector<uint32_t> overrun(words.begin(), words.end() - 2);
	overrun.push_back(0x00ff0000u | 5u); // an instruction that claims 255 words
	Check(!DerivePerVertexVertexLayoutFromSpirv(overrun, layout, params), "an instruction past the end refused");
	Check(ParsePvLayoutFastMode(nullptr) == PvLayoutFastMode::Off && ParsePvLayoutFastMode("0") == PvLayoutFastMode::Off &&
	          ParsePvLayoutFastMode("1") == PvLayoutFastMode::On && ParsePvLayoutFastMode("shadow") == PvLayoutFastMode::Shadow &&
	          ParsePvLayoutFastMode("x") == PvLayoutFastMode::Off,
	      "mode parsing");
}

double MicrosPerCall(int iterations, const auto& call) {
	const auto begin = std::chrono::steady_clock::now();
	for (int i = 0; i < iterations; i++) call();
	return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - begin).count() / iterations;
}

void TestCostOnLargeShader() {
	const auto words = Assemble(VertexShader(Variants()[0], 5000));
	const auto slow  = ViaDisassembly(words);
	const auto fast  = ViaWords(words);
	Check(slow.ok && fast.ok && SameLayout(slow.layout, fast.layout), "large shader: layouts equal");
	const double slow_us = MicrosPerCall(50, [&] { (void)ViaDisassembly(words); });
	const double fast_us = MicrosPerCall(50, [&] { (void)ViaWords(words); });
	std::printf("pv-layout cost on %zu words: disassemble+derive %.1f us, from words %.1f us (%.0fx)\n", words.size(), slow_us, fast_us,
	            slow_us / fast_us);
	Check(fast_us * 3 < slow_us, "reading the words is at least 3x cheaper than disassembling");
}

} // namespace

int main() {
	TestEquivalence();
	TestMalformedInput();
	TestCostOnLargeShader();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failures\n", g_failures);
		return 1;
	}
	std::puts("pv layout fast tests passed");
	return 0;
}
