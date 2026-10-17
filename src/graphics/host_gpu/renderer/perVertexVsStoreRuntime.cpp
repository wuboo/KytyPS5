#include "graphics/host_gpu/renderer/perVertexVsStoreRuntime.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/perVertexPrototype.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"

#include <map>
#include <spirv-tools/libspirv.hpp>

namespace Libs::Graphics {
namespace {

// 4M vec4 = 64 MiB per slab; a frame of the game needs a few slabs at most (counted in pv_fallback_ring).
constexpr uint64_t kSlabVec4s = uint64_t {1} << 22u;
constexpr uint32_t kMaxSlabs  = 8;

struct StoreShader {
	ShaderType            stage = ShaderType::Unknown;
	std::vector<uint32_t> words;       // empty: emission or validation failed
	uint32_t              mask = 0;    // host locations the stage stores (VS) / reads per vertex (PS)
};
std::map<uint64_t, StoreShader>                                    s_shaders;
std::map<std::pair<uint64_t, uint64_t>, PvVsStorePrograms>         s_pairs;
vk::DescriptorSetLayout                                            s_set_layout = nullptr;

} // namespace

std::vector<uint32_t> EmitPvVsStoreVariant(ShaderRecompiler::IR::Program&          program,
                                           const ShaderRecompiler::CompileOptions& options,
                                           const ShaderRecompiler::Spirv::PvStoreInfo& info,
                                           uint32_t push_data_start_dword, bool maintenance8_enabled) {
#if defined(KYTY_PVPUSH_MUTANT_ZERO)
	push_data_start_dword = 0;
	maintenance8_enabled  = false;
#endif
	return ShaderRecompiler::Spirv::EmitProgram(program, options.input_info, push_data_start_dword,
	                                            maintenance8_enabled, false, nullptr, &info);
}

void RememberPvVsStoreShader(const ShaderProgram& program, ShaderRecompiler::CompileResult& result,
                             const ShaderRecompiler::CompileOptions& options,
                             uint32_t push_data_start_dword, bool maintenance8_enabled) {
	if (!PerVertexVsStoreEnabled()) return;
	using Role = ShaderRecompiler::Spirv::PvStoreInfo::Role;
	const bool vertex = options.stage == ShaderType::Vertex && options.input_info.vertex != nullptr;
	const bool pixel  = options.stage == ShaderType::Pixel && options.input_info.pixel != nullptr &&
	                   HasPerVertexPrototypeInput(result.spirv);
	if (!vertex && !pixel) return;
	StoreShader entry;
	entry.stage = options.stage;
	ShaderRecompiler::Spirv::PvStoreInfo info;
	info.role          = vertex ? Role::Vertex : Role::Pixel;
	info.coherent      = true; // Coherent cap + a device barrier after the VS stores (the probe of RESULT-262 needs them)
	info.location_mask = &entry.mask;
	auto words         = EmitPvVsStoreVariant(result.program, options, info, push_data_start_dword, maintenance8_enabled);
	NotePvPushVariant(vertex ? "vsstore-vs" : "vsstore-ps", options.shader_hash, result.program.bindings);
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
	tools.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
		LOGF("PvVsStore: %s 0x%016llx SPIR-V word %zu: %s\n", vertex ? "VS" : "PS",
		     static_cast<unsigned long long>(program.id), position.index, message);
	});
	if (tools.Validate(words)) {
		entry.words = std::move(words);
	} else {
		LOGF("PvVsStore: %s 0x%016llx variant rejected, its draws keep the prototype\n", vertex ? "VS" : "PS",
		     static_cast<unsigned long long>(program.id));
	}
	s_shaders.emplace(program.id, std::move(entry));
}

void RememberPerVertexVariants(const ShaderProgram& program, ShaderRecompiler::CompileResult& result,
                               const ShaderRecompiler::CompileOptions& options, uint32_t host_subgroup_size,
                               uint32_t push_data_start_dword) {
	const uint32_t start = PerVertexReemitPushStart(push_data_start_dword);
	const bool     m8    = PerVertexReemitMaintenance8(options.maintenance8_enabled);
	// Each re-emission rewrites program.bindings; the first must not change the input of the second.
	const auto emitted_bindings = result.program.bindings;
	// A variant that does not read its shader data where the regular program does would see wrong dwords.
	const bool check = PerVertexPushStartFix();
	RememberPerVertexPrototypeShader(program, result, options, host_subgroup_size, start, m8);
	EXIT_IF(check && !PvVariantPushMatches(emitted_bindings, result.program.bindings));
	result.program.bindings = emitted_bindings;
	RememberPvVsStoreShader(program, result, options, start, m8);
	EXIT_IF(check && !PvVariantPushMatches(emitted_bindings, result.program.bindings));
	EXIT_IF(result.program.bindings.descriptor_mask != emitted_bindings.descriptor_mask);
	result.program.bindings = emitted_bindings;
}

vk::DescriptorSetLayout PvVsStoreSetLayout(GraphicContext& graphics) {
	if (s_set_layout != nullptr) return s_set_layout;
	const vk::DescriptorSetLayoutBinding bindings[] {
	    {0, vk::DescriptorType::eStorageBuffer, 1,
	     vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment},
	    {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eFragment}};
	vk::DescriptorSetLayoutCreateInfo create {};
	create.bindingCount = 2;
	create.pBindings    = bindings;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &s_set_layout) != vk::Result::eSuccess);
	return s_set_layout;
}

uint32_t PvVsStorePushOffset() {
	return ShaderRecompiler::IR::NativePushConstantSize;
}

const PvVsStorePrograms* GetPvVsStorePrograms(GraphicContext& graphics, const PipelineCache::GraphicsPrograms& programs) {
	if (!PerVertexVsStoreEnabled() || programs.VertexStageCount() != 1) return nullptr;
	const auto key = std::pair {programs.vertex[0].id, programs.pixel.id};
	if (const auto it = s_pairs.find(key); it != s_pairs.end()) {
		return it->second.stride_vec4 == UINT32_MAX ? nullptr : &it->second;
	}
	PvVsStorePrograms value;
	const auto        vs = s_shaders.find(key.first);
	const auto        ps = s_shaders.find(key.second);
	if (vs == s_shaders.end() || ps == s_shaders.end() || vs->second.words.empty() || ps->second.words.empty()) {
		value.stride_vec4 = UINT32_MAX; // negative cache: not a candidate
		s_pairs.emplace(key, value);
		return nullptr;
	}
	value.verdict = ClassifyPvVsStoreInterface(vs->second.mask, ps->second.mask);
	if (value.verdict == PvVsStoreInterfaceVerdict::Use) {
		value.stride_vec4               = PvVsStoreStrideFor(vs->second.mask);
		value.graphics                  = programs;
		value.graphics.vertex[0].module = CompileSPV(vs->second.words, graphics.device);
		value.graphics.pixel.module     = CompileSPV(ps->second.words, graphics.device);
		value.graphics.vertex[0].id |= 1ull << 61;
		value.graphics.pixel.id |= 1ull << 61;
		value.graphics.pv_store_layout = PvVsStoreSetLayout(graphics);
		LOGF("PvVsStore: pair VS=%llu PS=%llu stride=%u vec4\n", static_cast<unsigned long long>(key.first),
		     static_cast<unsigned long long>(key.second), value.stride_vec4);
	}
	return &s_pairs.emplace(key, std::move(value)).first->second;
}

PvVsStoreRuntime::PvVsStoreRuntime(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_graphics(graphics), m_scheduler(scheduler), m_slabs(kSlabVec4s, kMaxSlabs) {}

PvVsStoreRuntime::~PvVsStoreRuntime() = default;

uint64_t PvVsStoreRuntime::SlabVec4s() {
	return kSlabVec4s;
}

PvVsStoreRuntime::Region PvVsStoreRuntime::Place(uint64_t records, uint32_t stride_vec4) {
	Region region;
	if (stride_vec4 == 0 || records == 0) return region;
	// The shaders address record * stride + slot, so a region starts at a multiple of the stride.
	const uint64_t vec4s     = records * stride_vec4 + stride_vec4;
	const auto     placement = m_slabs.Place(vec4s, m_scheduler.CurrentTick(),
	                                         [&](uint64_t tick) { return m_scheduler.IsFree(tick); });
	if (!placement.ok) return region;
	if (placement.created) {
		EXIT_IF(placement.slab != m_buffers.size());
		m_buffers.push_back(std::make_unique<Buffer>(m_graphics, m_scheduler, MemoryUsage::Stream, 0, AllFlags,
		                                             kSlabVec4s * 16));
	}
	const uint64_t aligned = (uint64_t {placement.base} + stride_vec4 - 1) / stride_vec4 * stride_vec4;
	region.ok          = true;
	region.slab        = placement.slab;
	region.base_record = static_cast<uint32_t>(aligned / stride_vec4);
	region.buffer      = m_buffers.at(placement.slab)->Handle();
	region.buffer_size = m_buffers.at(placement.slab)->Size();
	return region;
}

void PvVsStoreRuntime::Touch(const Region& region) {
	m_slabs.Touch(region.slab, m_scheduler.CurrentTick());
}

} // namespace Libs::Graphics
