#include "graphics/host_gpu/renderer/perVertexPrototype.h"

#include "common/file.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/perVertexEmbeddedSpv.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "kytyGitVersion.h"
#include "per_vertex_unpack_spv.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <map>
#include <regex>
#include <set>
#include <spirv-tools/libspirv.hpp>
#include <spirv/unified1/spirv.hpp>
#include <sstream>
#include <string>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {
namespace {
uint32_t HalfToFloatBits(uint16_t half) {
	const uint32_t sign     = static_cast<uint32_t>(half & 0x8000u) << 16u;
	const uint32_t fraction = half & 0x03ffu;
	const uint32_t exponent = (half >> 10u) & 0x1fu;
	if (exponent == 0u) {
		if (fraction == 0u) return sign;
		uint32_t normalized        = fraction;
		int32_t  unbiased_exponent = -14;
		while ((normalized & 0x0400u) == 0u) {
			normalized <<= 1u;
			--unbiased_exponent;
		}
		return sign | (static_cast<uint32_t>(unbiased_exponent + 127) << 23u) |
		       ((normalized & 0x03ffu) << 13u);
	}
	if (exponent == 0x1fu) return sign | 0x7f800000u | (fraction << 13u);
	return sign | ((exponent - 15u + 127u) << 23u) | (fraction << 13u);
}

struct Original {
	ShaderType            stage;
	bool                  per_vertex;
	std::vector<uint32_t> words;
	std::vector<uint32_t> capture_words;
};
std::map<uint64_t, Original>                                        originals;
std::map<std::pair<uint64_t, uint64_t>, PerVertexPrototypePrograms> variants;
PerVertexUnpackPipeline                                             s_unpack_pipeline {};
bool                                                                s_unpack_initialized = false;
} // namespace

#include "graphics/host_gpu/renderer/perVertexTransform.h"

bool PerVertexPrototypeEnabled() {
	return std::getenv("KYTY_PER_VERTEX_PROTOTYPE") != nullptr;
}

bool PerVertexPushStartFix() {
	static const bool fix = [] {
		const char* value = std::getenv("KYTY_PV_PUSH_START");
		const bool  zero  = value != nullptr && std::strcmp(value, "zero") == 0;
		EXIT_IF(value != nullptr && !zero && std::strcmp(value, "fix") != 0);
		LOGF("[pv-push] mode=%s\n", zero ? "zero" : "fix");
		return !zero;
	}();
	return fix;
}

uint32_t PerVertexReemitPushStart(uint32_t push_data_start_dword) {
	return PerVertexPushStartFix() ? push_data_start_dword : 0u;
}

bool PerVertexReemitMaintenance8(bool maintenance8_enabled) {
	return PerVertexPushStartFix() ? maintenance8_enabled : false;
}

bool PvVariantPushMatches(const ShaderRecompiler::IR::Bindings& regular,
                          const ShaderRecompiler::IR::Bindings& variant) {
#if defined(KYTY_PVPUSH_MUTANT_NO_CHECK)
	return true;
#else
	return regular.push_data_start_dword == variant.push_data_start_dword &&
	       regular.ShaderDataDwords() == variant.ShaderDataDwords();
#endif
}

void NotePvPushVariant(const char* role, uint64_t shader_hash,
                       const ShaderRecompiler::IR::Bindings& variant) {
	using ShaderRecompiler::IR::PushData;
	const uint32_t start = variant.push_data_start_dword;
	if (start != PushData::NoStart && start != 0) {
		FrameTiming::Add(FrameTiming::Counter::PvPushNonzeroPrograms);
	}
	if (start == PushData::NoStart) {
		LOGF("[pv-push] role=%s hash=0x%016" PRIx64 " start=none dwords=%u\n", role, shader_hash,
		     variant.ShaderDataDwords());
	} else {
		LOGF("[pv-push] role=%s hash=0x%016" PRIx64 " start=%u dwords=%u\n", role, shader_hash,
		     start, variant.ShaderDataDwords());
	}
}

std::vector<uint32_t> EmitPerVertexCaptureVariant(ShaderRecompiler::IR::Program&                program,
                                                  const ShaderRecompiler::CompileOptions&       options,
                                                  const ShaderRecompiler::Spirv::VertexCaptureInfo& capture,
                                                  uint32_t push_data_start_dword,
                                                  bool     maintenance8_enabled) {
#if defined(KYTY_PVPUSH_MUTANT_ZERO)
	push_data_start_dword = 0;
	maintenance8_enabled  = false;
#endif
	return ShaderRecompiler::Spirv::EmitProgram(program, options.input_info, push_data_start_dword,
	                                            maintenance8_enabled, false, &capture);
}

bool PerVertexUniqueEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PV_UNIQUE");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

bool PerVertexVsStoreEnabled() {
	static const bool enabled = [] {
		const char* value = std::getenv("KYTY_PV_VSSTORE");
		return value != nullptr && value[0] == '1' && value[1] == '\0';
	}();
	return enabled;
}

const PvVsStorePairFilter& PerVertexVsStorePairs() {
	static const PvVsStorePairFilter filter = [] {
		PvVsStorePairFilter parsed;
		const char*         value = std::getenv("KYTY_PV_VSSTORE_PAIRS");
		EXIT_IF(!ParsePvVsStorePairs(value, parsed));
		LOGF("[pv-pairs] mode=%s\n", value != nullptr ? value : "all");
		return parsed;
	}();
	return filter;
}

bool PerVertexVsStoreGuard() {
	static const bool guard = [] {
		bool parsed = false;
		EXIT_IF(!ParsePvVsStoreGuard(std::getenv("KYTY_PV_VSSTORE_GUARD"), parsed));
		LOGF("[pv-guard] mode=%d\n", parsed ? 1 : 0);
		return parsed;
	}();
	return guard;
}

bool PerVertexVsStoreInstanced() {
	static const bool allowed = [] {
		bool parsed = true;
		EXIT_IF(!ParsePvVsStoreInstanced(std::getenv("KYTY_PV_VSSTORE_INSTANCED"), parsed));
		LOGF("[pv-instanced] mode=%d\n", parsed ? 1 : 0);
		return parsed;
	}();
	return allowed;
}

PvVsStoreGpuIdxMode PerVertexVsStoreGpuIdxMode() {
	static const PvVsStoreGpuIdxMode mode = [] {
		const char*         value = std::getenv("KYTY_PV_VSSTORE_GPUIDX");
		PvVsStoreGpuIdxMode parsed;
		EXIT_IF(!ParsePvVsStoreGpuIdxMode(value, parsed));
		LOGF("[pv-gpuidx] mode=%s\n", parsed == PvVsStoreGpuIdxMode::Fallback ? "fallback" : "allow");
		return parsed;
	}();
	return mode;
}

PerVertexDrawVerdict ClassifyPerVertexPrototypeDraw(bool ps_active, vk::PrimitiveTopology topology,
                                                    bool primitive_restart, uint32_t index_count) {
	if (!ps_active) return {false, index_count, "no pixel shader"};
	if (topology != vk::PrimitiveTopology::eTriangleList) return {false, index_count, "topology"};
	if (primitive_restart) return {false, index_count, "primitive restart"};
	const uint32_t whole = index_count - index_count % 3;
	if (whole == 0) return {false, index_count, "less than one triangle"};
	return {true, whole, whole != index_count ? "partial triangle dropped" : nullptr};
}

bool HasPerVertexPrototypeInput(std::span<const uint32_t> words) {
	if (words.size() < 5 || words[0] != spv::MagicNumber) return false;
	bool found = false;
	for (size_t offset = 5; offset < words.size();) {
		const auto count = words[offset] >> 16;
		if (count == 0 || count > words.size() - offset) return false;
		if ((words[offset] & 0xffffu) == spv::OpDecorate && count == 3 &&
		    words[offset + 2] == spv::DecorationPerVertexKHR)
			found = true;
		offset += count;
	}
	return found;
}

bool DecodePerVertexPrototypeAttribute(Prospero::BufferFormat   format,
                                       std::span<const uint8_t> bytes,
                                       std::span<uint32_t, 4>   components) {
	const auto info = ShaderRecompiler::Format::GetFormatInfo(format);
	using Type      = ShaderRecompiler::Format::ComponentType;
	if (info.component_count == 0 || info.component_count > 4 || bytes.size() < info.byte_size)
		return false;
	const bool integer = info.type == Type::Uint || info.type == Type::Sint;
	if (info.type != Type::Float && info.type != Type::Unorm && !integer &&
	    !(format == Prospero::BufferFormat::k16_16_16_16SNorm && info.type == Type::Snorm))
		return false;
	for (uint32_t c = 0; c < info.component_count; ++c) {
		if (integer && info.component_bits[c] != 8 && info.component_bits[c] != 16 &&
		    info.component_bits[c] != 32)
			return false;
		if (info.type == Type::Float && format != Prospero::BufferFormat::k16_16_16_16Float &&
		    info.component_bits[c] != 32)
			return false;
		if (format == Prospero::BufferFormat::k16_16_16_16Float && info.component_bits[c] != 16)
			return false;
		if (info.type == Type::Unorm &&
		    (info.component_bits[c] == 0 || info.component_bits[c] > 16))
			return false;
		if (format == Prospero::BufferFormat::k16_16_16_16SNorm && info.component_bits[c] != 16)
			return false;
	}
	for (uint32_t c = 0; c < 4; ++c) {
		uint32_t value = c == 3 ? (integer ? 1u : 0x3f800000u) : 0u;
		if (c < info.component_count) {
			if (format == Prospero::BufferFormat::k16_16_16_16Float) {
				uint16_t raw = 0;
				std::memcpy(&raw, bytes.data() + info.component_bit_offset[c] / 8, sizeof(raw));
				value = HalfToFloatBits(raw);
			} else if (info.type == Type::Float) {
				std::memcpy(&value, bytes.data() + info.component_bit_offset[c] / 8, 4);
			} else if (format == Prospero::BufferFormat::k16_16_16_16SNorm) {
				int16_t raw = 0;
				std::memcpy(&raw, bytes.data() + info.component_bit_offset[c] / 8, sizeof(raw));
				value =
				    std::bit_cast<uint32_t>(std::max(static_cast<float>(raw) / 32767.0f, -1.0f));
			} else {
				uint32_t raw = 0;
				for (uint32_t bit = 0; bit < info.component_bits[c]; ++bit) {
					const auto position = info.component_bit_offset[c] + bit;
					raw |= ((bytes[position / 8] >> (position % 8)) & 1u) << bit;
				}
				if (integer) {
					const auto sign = 1u << (info.component_bits[c] - 1u);
					value           = info.type == Type::Sint ? (raw ^ sign) - sign : raw;
				} else {
					const auto mask = (1u << info.component_bits[c]) - 1u;
					value =
					    std::bit_cast<uint32_t>(static_cast<float>(raw) / static_cast<float>(mask));
				}
			}
		}
		components[c] = value;
	}
	return true;
}

void RememberPerVertexPrototypeShader(const ShaderProgram&                    program,
                                      ShaderRecompiler::CompileResult&        result,
                                      const ShaderRecompiler::CompileOptions& options,
                                      uint32_t                                host_subgroup_size,
                                      uint32_t                                push_data_start_dword,
                                      bool                                    maintenance8_enabled) {
	if (!PerVertexPrototypeEnabled()) return;
	const auto  stage      = options.stage;
	const auto& words      = result.spirv;
	const bool  per_vertex = stage == ShaderType::Pixel && HasPerVertexPrototypeInput(words);
	if (stage == ShaderType::Vertex || per_vertex) {
		auto [it, inserted] =
		    originals.emplace(program.id, Original {stage, per_vertex, words, {}});
		if (!inserted || stage != ShaderType::Vertex) return;
		const auto& vs = *options.input_info.vertex;
		using namespace ShaderRecompiler;
		// Keep unsupported capture interfaces on the existing guarded prototype path.
		if ((host_subgroup_size != 32 && host_subgroup_size != 64) || vs.buffers_num != 1 ||
		    vs.resources_num <= 0 ||
		    vs.resources_num > static_cast<int>(kPerVertexPrototypeMaxAttributes) ||
		    std::ranges::any_of(result.program.info.inputs,
		                        [&](const IR::StageInput& input) {
			                        return (input.kind != IR::StageInputKind::VertexIndex &&
			                                input.kind != IR::StageInputKind::InstanceIndex &&
			                                input.kind != IR::StageInputKind::Parameter) ||
			                               (input.kind == IR::StageInputKind::Parameter &&
			                                input.location >=
			                                    static_cast<uint32_t>(vs.resources_num));
		                        }) ||
		    std::ranges::any_of(result.program.info.outputs,
		                        [](const IR::StageOutput& output) {
			                        return output.kind != IR::StageOutputKind::Position &&
			                               output.kind != IR::StageOutputKind::Parameter;
		                        }) ||
		    std::ranges::any_of(result.program.memory_info, [](const IR::MemoryInfo& memory) {
			    return memory.kind == IR::ResourceKind::Lds;
		    }))
			return;
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
		std::string          vs_source;
		EXIT_IF(!tools.Disassemble(words, &vs_source, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));
		PerVertexLayout                 layout;
		std::map<uint32_t, std::string> params;
		if (!DerivePerVertexVertexLayout(vs_source, layout, params)) return;
		const Spirv::VertexCaptureInfo capture {
		    host_subgroup_size, static_cast<uint32_t>(vs.resources_num), layout.record_stride_vec4,
		    layout.clip_slot, layout.location_to_slot};
		it->second.capture_words = EmitPerVertexCaptureVariant(result.program, options, capture,
		                                                       push_data_start_dword, maintenance8_enabled);
		NotePvPushVariant("capture", options.shader_hash, result.program.bindings);
		// Validation of our own emitter output is ~30% of a vertex permutation's cost; bench-only
		// switch KYTY_PV_SKIP_VALIDATE=1 skips it (the first draw would still be validated by Vulkan).
		static const bool skip_validate = std::getenv("KYTY_PV_SKIP_VALIDATE") != nullptr;
		EXIT_IF(!skip_validate && !tools.Validate(it->second.capture_words));
	}
}

const PerVertexUnpackPipeline* GetPerVertexUnpackPipeline(GraphicContext&   graphics,
                                                          vk::PipelineCache driver_cache) {
	if (s_unpack_initialized) return &s_unpack_pipeline;

	s_unpack_pipeline.module = CompileSPV(PERVERTEX_UNPACK_SPV, graphics.device);

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}};
	vk::DescriptorSetLayoutCreateInfo layout_info {};
	layout_info.bindingCount = 4;
	layout_info.pBindings    = bindings;
	EXIT_IF(graphics.device.createDescriptorSetLayout(
	            &layout_info, nullptr, &s_unpack_pipeline.set_layout) != vk::Result::eSuccess);

	const vk::PushConstantRange  push(vk::ShaderStageFlagBits::eCompute, 0,
	                                  sizeof(UnpackPushConstants));
	vk::PipelineLayoutCreateInfo pipe_layout_info {};
	pipe_layout_info.setLayoutCount         = 1;
	pipe_layout_info.pSetLayouts            = &s_unpack_pipeline.set_layout;
	pipe_layout_info.pushConstantRangeCount = 1;
	pipe_layout_info.pPushConstantRanges    = &push;
	EXIT_IF(graphics.device.createPipelineLayout(&pipe_layout_info, nullptr,
	                                             &s_unpack_pipeline.pipeline_layout) !=
	        vk::Result::eSuccess);

	vk::ComputePipelineCreateInfo compute_info {};
	compute_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	compute_info.stage.module = s_unpack_pipeline.module;
	compute_info.stage.pName  = "main";
	compute_info.layout       = s_unpack_pipeline.pipeline_layout;
	vk::Result res = graphics.device.createComputePipelines(driver_cache, 1, &compute_info, nullptr,
	                                                        &s_unpack_pipeline.pipeline);
	if (res != vk::Result::eSuccess && driver_cache != nullptr) {
		LOGF("PerVertexPrototype: unpack createComputePipelines failed with driver_cache (%s), "
		     "retrying with nullptr\n",
		     vk::to_string(res).c_str());
		res = graphics.device.createComputePipelines(nullptr, 1, &compute_info, nullptr,
		                                             &s_unpack_pipeline.pipeline);
	}
	EXIT_IF(res != vk::Result::eSuccess);

	s_unpack_initialized = true;
	return &s_unpack_pipeline;
}

const PerVertexPrototypePrograms* GetPerVertexPrototypePrograms(
    GraphicContext& graphics, const PipelineCache::GraphicsPrograms& programs,
    const ShaderVertexInputInfo& vs_info, vk::PipelineCache driver_cache) {
	if (!PerVertexPrototypeEnabled()) return nullptr;
	if (vs_info.logical_stage != ShaderType::Vertex) return nullptr;
	const auto ps = originals.find(programs.pixel.id);
	if (ps == originals.end() || !ps->second.per_vertex) return nullptr;
	const auto vs = originals.find(programs.vertex[0].id);
	EXIT_IF(vs == originals.end() || vs->second.stage != ShaderType::Vertex ||
	        programs.VertexStageCount() != 1);
	const auto key = std::pair {programs.vertex[0].id, programs.pixel.id};
	if (auto it = variants.find(key); it != variants.end()) return &it->second;

	// Include the capture variant (and its host wave width) in the persistent cache identity.
	const auto& capture_source =
	    vs->second.capture_words.empty() ? vs->second.words : vs->second.capture_words;
	const uint64_t vs_hash =
	    XXH3_64bits(capture_source.data(), capture_source.size() * sizeof(uint32_t));
	const uint64_t ps_hash =
	    XXH3_64bits(ps->second.words.data(), ps->second.words.size() * sizeof(uint32_t));
	const auto        title_id           = PipelineCacheTitleId();
	const std::string effective_title_id = title_id.empty() ? "default" : title_id;

	PerVertexLayout       layout {};
	std::vector<uint32_t> cap_words, frag_words, replay_words;
	std::string           vs_dis, ps_dis; // filled on a bytecode cache miss; the unique variant disassembles ps_dis itself if empty
	const auto            t0               = std::chrono::steady_clock::now();
	bool                  loaded_from_disk = TryLoadTransformedShadersFromDisk(
	    effective_title_id, vs_hash, ps_hash, layout, cap_words, frag_words, replay_words);

	if (loaded_from_disk) {
		const auto load_us = std::chrono::duration_cast<std::chrono::microseconds>(
		                         std::chrono::steady_clock::now() - t0)
		                         .count();
		LOGF("PerVertexPrototype: bytecode cache HIT for VS=0x%016llx PS=0x%016llx (loaded in %lld "
		     "us)\n",
		     static_cast<unsigned long long>(vs_hash), static_cast<unsigned long long>(ps_hash),
		     static_cast<long long>(load_us));
	} else {
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
		tools.SetMessageConsumer([](spv_message_level_t, const char*,
		                            const spv_position_t& position, const char* message) {
			LOGF("PerVertexPrototype SPIR-V word %zu: %s\n", position.index, message);
		});
		EXIT_IF(!tools.Disassemble(vs->second.words.data(), vs->second.words.size(), &vs_dis,
		                           SPV_BINARY_TO_TEXT_OPTION_INDENT |
		                               SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));
		EXIT_IF(!tools.Disassemble(ps->second.words.data(), ps->second.words.size(), &ps_dis,
		                           SPV_BINARY_TO_TEXT_OPTION_INDENT |
		                               SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));

		std::map<uint32_t, std::string> vs_param_vars;
		if (!DerivePerVertexLayout(vs_dis, ps_dis, layout, vs_param_vars)) {
			LOGF("PerVertexPrototype: interface contract validation failed for VS=%llu PS=%llu -> "
			     "rejecting\n",
			     static_cast<unsigned long long>(key.first),
			     static_cast<unsigned long long>(key.second));
			return nullptr;
		}

		std::string frag_dis   = LowerFragmentToBufferReplay(ps_dis, layout);
		std::string replay_dis = GenerateReplayVertexSpvasm(layout);

		cap_words = vs->second.capture_words;
		if (cap_words.empty()) {
			const auto cap_dis = LowerVertexToCompute(vs_dis, layout, vs_param_vars);
			EXIT_IF(!tools.Assemble(cap_dis, &cap_words,
			                        SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
		}
		EXIT_IF(
		    !tools.Assemble(frag_dis, &frag_words, SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
		EXIT_IF(!tools.Assemble(replay_dis, &replay_words,
		                        SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
		EXIT_IF(!tools.Validate(cap_words));
		EXIT_IF(!tools.Validate(frag_words));
		EXIT_IF(!tools.Validate(replay_words));

		SaveTransformedShadersToDisk(effective_title_id, vs_hash, ps_hash, layout, cap_words,
		                             frag_words, replay_words);
		const auto comp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
		                         std::chrono::steady_clock::now() - t0)
		                         .count();
		LOGF("PerVertexPrototype: bytecode cache MISS for VS=0x%016llx PS=0x%016llx (lowered & "
		     "assembled in %lld ms)\n",
		     static_cast<unsigned long long>(vs_hash), static_cast<unsigned long long>(ps_hash),
		     static_cast<long long>(comp_ms));
	}

	// S134: the unique-vertex replay shaders (the classic ones stay for the draws that fall back).
	std::vector<uint32_t> unique_frag_words, unique_replay_words;
	if (PerVertexUniqueEnabled()) {
		PerVertexLayout       unique_layout {};
		std::vector<uint32_t> unique_cap_words;
		const bool            unique_loaded = TryLoadTransformedShadersFromDisk(
            effective_title_id, vs_hash, ps_hash, unique_layout, unique_cap_words,
            unique_frag_words, unique_replay_words, true);
		if (!unique_loaded) {
			spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
			if (ps_dis.empty()) {
				EXIT_IF(!tools.Disassemble(ps->second.words.data(), ps->second.words.size(),
				                           &ps_dis,
				                           SPV_BINARY_TO_TEXT_OPTION_INDENT |
				                               SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES));
			}
			const std::string unique_frag_dis   = LowerFragmentToBufferReplay(ps_dis, layout, true);
			const std::string unique_replay_dis = GenerateReplayVertexSpvasm(layout, true);
			EXIT_IF(!tools.Assemble(unique_frag_dis, &unique_frag_words,
			                        SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
			EXIT_IF(!tools.Assemble(unique_replay_dis, &unique_replay_words,
			                        SPV_TEXT_TO_BINARY_OPTION_PRESERVE_NUMERIC_IDS));
			EXIT_IF(!tools.Validate(unique_frag_words));
			EXIT_IF(!tools.Validate(unique_replay_words));
			SaveTransformedShadersToDisk(effective_title_id, vs_hash, ps_hash, layout, cap_words,
			                             unique_frag_words, unique_replay_words, true);
		}
	}

	PerVertexPrototypePrograms value {};
	value.native_capture            = !vs->second.capture_words.empty();
	value.graphics                  = programs;
	value.layout                    = layout;
	value.capture                   = CompileSPV(cap_words, graphics.device);
	value.graphics.vertex[0].module = CompileSPV(replay_words, graphics.device);
	value.graphics.pixel.module     = CompileSPV(frag_words, graphics.device);
	value.graphics.vertex[0].id |= 1ull << 63;
	value.graphics.pixel.id |= 1ull << 63;
	if (!unique_replay_words.empty()) {
		value.unique_available                 = true;
		value.graphics_unique                  = programs;
		value.graphics_unique.vertex[0].module = CompileSPV(unique_replay_words, graphics.device);
		value.graphics_unique.pixel.module     = CompileSPV(unique_frag_words, graphics.device);
		value.graphics_unique.vertex[0].id |= 1ull << 62;
		value.graphics_unique.pixel.id |= 1ull << 62;
	}

	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
	    {1, vk::DescriptorType::eStorageBuffer, 1,
	     vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex |
	         vk::ShaderStageFlagBits::eFragment},
	    {2, vk::DescriptorType::eStorageBuffer, 1,
	     vk::ShaderStageFlagBits::eCompute |
	         (PerVertexUniqueEnabled() ? vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment
	                                   : vk::ShaderStageFlags {})}};
	vk::DescriptorSetLayoutCreateInfo create {};
	create.bindingCount = 3;
	create.pBindings    = bindings;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &value.extra_layout) !=
	        vk::Result::eSuccess);
	value.graphics.prototype_extra_layout = value.extra_layout;
	value.graphics_unique.prototype_extra_layout = value.extra_layout;

	// Build capture compute pipeline once for this pair
	value.capture_pipeline.prototype_vertex_capture = true;
	value.capture_pipeline.uses_push_descriptors    = false;
	std::vector<vk::DescriptorSetLayoutBinding> resources;
	const auto& vs_bindings = vs_info.stage.program->bindings;
	for (auto mask = vs_bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto index = static_cast<size_t>(std::countr_zero(mask));
		const auto kind  = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
		resources.emplace_back(
		    ShaderRecompiler::IR::NativeBinding(ShaderType::Vertex, kind),
		    NativeDescriptorType(kind), vs_bindings.descriptor_counts[index],
		    vk::ShaderStageFlagBits::eCompute);
	}
	vk::DescriptorSetLayoutCreateInfo resource_info {};
	resource_info.bindingCount = static_cast<uint32_t>(resources.size());
	resource_info.pBindings    = resources.data();
	EXIT_IF(graphics.device.createDescriptorSetLayout(
	            &resource_info, nullptr, &value.capture_pipeline.descriptor_set_layout) !=
	        vk::Result::eSuccess);

	const vk::DescriptorSetLayout layouts[] {value.capture_pipeline.descriptor_set_layout,
	                                         value.extra_layout};
	const vk::PushConstantRange   push(vk::ShaderStageFlagBits::eCompute, 0,
	                                   ShaderRecompiler::IR::NativePushConstantSize);
	vk::PipelineLayoutCreateInfo  layout_info {};
	layout_info.setLayoutCount         = 2;
	layout_info.pSetLayouts            = layouts;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &push;
	EXIT_IF(graphics.device.createPipelineLayout(&layout_info, nullptr,
	                                             &value.capture_pipeline.pipeline_layout) !=
	        vk::Result::eSuccess);

	vk::ComputePipelineCreateInfo pipeline_info {};
	pipeline_info.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipeline_info.stage.module = value.capture;
	pipeline_info.stage.pName  = "main";
	pipeline_info.layout       = value.capture_pipeline.pipeline_layout;
	vk::Result pipe_res        = graphics.device.createComputePipelines(
	    driver_cache, 1, &pipeline_info, nullptr, &value.capture_pipeline.pipeline);
	if (pipe_res != vk::Result::eSuccess && driver_cache != nullptr) {
		LOGF("PerVertexPrototype: capture createComputePipelines failed with driver_cache (%s), "
		     "retrying with nullptr\n",
		     vk::to_string(pipe_res).c_str());
		pipe_res = graphics.device.createComputePipelines(nullptr, 1, &pipeline_info, nullptr,
		                                                  &value.capture_pipeline.pipeline);
	}
	EXIT_IF(pipe_res != vk::Result::eSuccess);

	LOGF("PerVertexPrototype: compiled native C++ pair VS=%llu PS=%llu\n",
	     static_cast<unsigned long long>(key.first), static_cast<unsigned long long>(key.second));
	return &variants.emplace(key, value).first->second;
}

} // namespace Libs::Graphics
