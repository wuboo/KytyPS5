// S150: GPU harness for KYTY_PV_VSSTORE (graphics/host_gpu/renderer/perVertexVsStore.h). The real SPIR-V
// emitter produces the VS-store VS and PS from an IR vertex shader and an ISA pixel shader, and the
// pair is drawn on the host GPU (MoltenVK) with the exact descriptor/push layout of the renderer:
//   set 0 = empty, set 1 = {cap SSBO (VS|FS), guest index SSBO (FS)}, push = native data + 8 dwords.
// Checks (bit for bit, no tolerance):
//   1. image: every probed pixel equals the PerVertexKHR semantics computed on the CPU from the vertex
//      data, the index buffer and the draw parameters (corner 0/1/2 read through P0/P10/P20),
//   2. cap: the records the VS wrote hold {gl_VertexIndex, gl_InstanceIndex} of the cases, which
//      settles whether gl_VertexIndex includes vertex_offset and gl_InstanceIndex first_instance.
// Cases: 16-bit and 32-bit indices, vertex_offset > 0 and < 0, first_instance != 0, 2 instances, an index
// buffer bound at an offset that is not a multiple of minStorageBufferOffsetAlignment.
// Run: per_vertex_vsstore_gpu_tests [--dump] [--stress] [--compact]  (--compact, S409: layouts with gaps in the PS mask, old and compact record)  (--stress: 2M triangles x 2..4 instances, parameter-buffer overflow)
//   previously:  per_vertex_vsstore_gpu_tests [--dump]   (--dump needs no GPU: prints the emitted SPIR-V).
// NOT the prototype: the prototype chain is compared in the game (A/B frames); this harness compares
// with the CPU model of the per-vertex semantics. The emitter mutant (KYTY_PVVS_MUTANT_FIRST_THIRD)
// must fail it.

#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/subsystems.h"
#include "common/threads.h"
#include "graphics/host_gpu/renderer/perVertexVsStore.h"
#include "graphics/host_gpu/renderer/perVertexVsStoreRuntime.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/backend/spirv/SpirvEmitter.h"
#include "graphics/shader/recompiler/ir/IREmitter.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <spirv-tools/libspirv.hpp>
#include <vulkan/vulkan.h>

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {
namespace {

using namespace ShaderRecompiler;

bool g_stress = false; // --stress: the >= 2M triangle cases (S151)

[[noreturn]] void Fail(const char* message) {
	std::fprintf(stderr, "per-vertex-vsstore-gpu-tests: FAILED: %s\n", message);
	std::exit(1);
}

void Check(bool ok, const char* message) {
	if (!ok) Fail(message);
}

void CheckVk(VkResult result, const char* what) {
	if (result != VK_SUCCESS) {
		std::fprintf(stderr, "per-vertex-vsstore-gpu-tests: FAILED: %s (VkResult %d)\n", what, static_cast<int>(result));
		std::exit(1);
	}
}

constexpr uint32_t EncodeSopp(uint32_t opcode, uint32_t simm = 0) {
	return (0x17fu << 23u) | ((opcode & 0x7fu) << 16u) | (simm & 0xffffu);
}
constexpr uint32_t EncodeVintrp(uint32_t opcode, uint32_t vdst, uint32_t attr, uint32_t chan, uint32_t vsrc) {
	return (0x32u << 26u) | ((opcode & 0x3u) << 16u) | ((vdst & 0xffu) << 18u) | ((attr & 0x3fu) << 10u) |
	       ((chan & 0x3u) << 8u) | (vsrc & 0xffu);
}
constexpr uint32_t EncodeExp0(uint32_t target, uint32_t en) {
	return (0x3eu << 26u) | ((target & 0x3fu) << 4u) | (en & 0xfu) | (1u << 11u);
}
constexpr uint32_t EncodeExp1(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
	return (a & 0xffu) | ((b & 0xffu) << 8u) | ((c & 0xffu) << 16u) | ((d & 0xffu) << 24u);
}

void EnsureConfigInitialized() {
	static bool initialized = false;
	if (initialized) return;
	static Common::Subsystems subsystems;
	Common::InitializeThreads();
	subsystems.Initialize<Config::Lifecycle>();
	Config::ConfigOptions options;
	options.printf_direction = Config::LogDirection::Silent;
	Config::Load(options);
	subsystems.Initialize<Log::Lifecycle>();
	ShaderInit();
	initialized = true;
}

// ---- shaders -------------------------------------------------------------------------------------

struct Shaders {
	std::vector<uint32_t> vs;
	std::vector<uint32_t> ps;
	uint32_t              stride_vec4 = 0;
	uint32_t              vs_mask = 0;
	uint32_t              ps_mask = 0;
	int32_t               ids_slot = -1; // slot of the {vertex, instance} output in a record; -1: not stored (compact)
};

// S409: which host locations the VS writes and the PS reads. The PS reads corner0.x and corner0.w of the first location, corner1.y of
// the second and corner2.z of the third (a layout with one read location reads all four from it). Location k of the reads holds
// the vertex color with `add[k]` added to the bits of every component (+ the instance index to x of the first), so a slot or a
// location mixed up by the PS shows in the image. The ids location holds {gl_VertexIndex, gl_InstanceIndex} and is read by nobody;
// `junk` locations hold a constant and are read by nobody.
struct Layout {
	const char* name;
	uint32_t    reads[3];  // host locations the PS reads
	uint32_t    read_count;
	uint32_t    add[3];
	uint32_t    ids;
	uint32_t    junk[2];
	uint32_t    junk_count;
};
const Layout kLayouts[] = {
    {"orig: VS {0,1}, PS {0}", {0, 0, 0}, 1, {0, 0, 0}, 1, {0, 0}, 0},
    {"gap: VS {0,3,5}, PS {3}", {3, 0, 0}, 1, {0, 0, 0}, 5, {0, 0}, 1},
    {"gap: VS {0,2,4,9}, PS {0,2,9} (mask 0x205)", {0, 2, 9}, 3, {0, 7, 13}, 4, {0, 0}, 0},
    {"gap: VS {1,2,6,9,12}, PS {2,9} (mask 0x204)", {2, 9, 0}, 2, {0, 7, 0}, 12, {1, 6}, 2},
};
const Layout* g_layout = &kLayouts[0];

// VS: position = attr0, the read locations hold attr1 (see Layout), the ids location {gl_VertexIndex, gl_InstanceIndex, 0, 0}.
// PS: out = {corner0.x, corner1.y, corner2.z, corner0.w} of the read locations.
// compact (S409): the record holds the locations of the PS mask only: the VS gets the mask as its specialization literal and the
// PS ranks its locations in its own mask.
Shaders BuildShaders(const Layout& layout, bool compact) {
	using IR::Value;
	using O = IR::ValueOpcode;
	Shaders out;

	// distinct locations written by the VS, in guest parameter index order
	struct Param { uint32_t location; int kind; uint32_t add; }; // kind 0 = color, 1 = ids, 2 = junk
	std::vector<Param> params;
	for (uint32_t k = 0; k < layout.read_count; ++k) {
		bool seen = false;
		for (const auto& p: params) seen = seen || p.location == layout.reads[k];
		if (!seen) params.push_back({layout.reads[k], 0, layout.add[k]});
	}
	params.push_back({layout.ids, 1, 0});
	for (uint32_t k = 0; k < layout.junk_count; ++k) params.push_back({layout.junk[k], 2, 0});

	IR::Program vs;
	vs.stage                  = ShaderType::Vertex;
	vs.wave_size              = 64;
	vs.srt_plan_complete      = vs.resource_tracking_complete = vs.shader_info_complete = true;
	vs.block_storage.push_back(std::make_unique<IR::Block>());
	vs.blocks.push_back(vs.block_storage.back().get());
	vs.info.inputs  = {{IR::StageInputKind::VertexIndex},
	                   {IR::StageInputKind::InstanceIndex},
	                   {IR::StageInputKind::Parameter, 0, 4},
	                   {IR::StageInputKind::Parameter, 1, 4}};
	vs.info.outputs = {{IR::StageOutputKind::Position}};
	vs.export_info  = {{.kind = IR::ExportTargetKind::Position, .en = 15}};
	for (uint32_t i = 0; i < params.size(); ++i) {
		vs.info.outputs.push_back({IR::StageOutputKind::Parameter, i, params[i].location});
		vs.export_info.push_back({.kind = IR::ExportTargetKind::Parameter, .index = i, .en = 15});
	}
	IR::IREmitter emit(vs.blocks.front());
	const auto vertex   = emit.Emit(O::GetBuiltin, {Value(static_cast<uint32_t>(IR::StageInputKind::VertexIndex)), Value(0u)});
	const auto instance = emit.Emit(O::GetBuiltin, {Value(static_cast<uint32_t>(IR::StageInputKind::InstanceIndex)), Value(0u)});
	std::array<Value, 4> position, color;
	for (uint32_t c = 0; c < 4; c++) position[c] = emit.Emit(O::GetAttribute, {Value(0u), Value(c)});
	for (uint32_t c = 0; c < 4; c++) color[c] = emit.Emit(O::GetAttribute, {Value(1u), Value(c)});
	const auto pos = emit.Emit(O::CompositeConstructU32x4, {position[0], position[1], position[2], position[3]});
	emit.Emit(O::SetAttribute, {pos, Value(true)}, IR::ExportFlags {0});
	for (uint32_t i = 0; i < params.size(); ++i) {
		Value value;
		if (params[i].kind == 0) {
			std::array<Value, 4> c4 = color;
			if (params[i].add != 0) {
				for (uint32_t c = 0; c < 4; c++) c4[c] = emit.Emit(O::IAdd32, {color[c], Value(params[i].add)});
			}
			// the first read location also carries the instance index in x
			if (params[i].location == layout.reads[0]) c4[0] = emit.Emit(O::IAdd32, {c4[0], instance});
			value = emit.Emit(O::CompositeConstructU32x4, {c4[0], c4[1], c4[2], c4[3]});
		} else if (params[i].kind == 1) {
			value = emit.Emit(O::CompositeConstructU32x4, {vertex, instance, Value(0u), Value(0u)});
		} else {
			value = emit.Emit(O::CompositeConstructU32x4, {Value(0x1234u), Value(0x5678u), Value(0u), Value(0u)});
		}
		emit.Emit(O::SetAttribute, {value, Value(true)}, IR::ExportFlags {i + 1});
	}
	static const ShaderVertexInputInfo vertex_info {};
	ShaderStageInputInfo               vs_input {.vertex = &vertex_info};

	uint32_t vs_mask = 0, spec_id = 0;
	Spirv::PvStoreInfo vs_store;
	vs_store.role          = Spirv::PvStoreInfo::Role::Vertex;
	vs_store.coherent      = true;
	vs_store.location_mask = &vs_mask;
	vs_store.compact_spec    = compact;
	vs_store.compact_mask_id = &spec_id;
	out.vs                 = Spirv::EmitProgram(vs, vs_input, 0, false, false, nullptr, &vs_store);

	// PS (the interpolation setup of TestPvVsStoreEmission): v_interp_mov_f32 with P0 / P10 / P20.
	const uint32_t a1 = layout.read_count > 1 ? 1 : 0, a2 = layout.read_count > 2 ? 2 : 0;
	const uint32_t ps_code[] = {EncodeVintrp(2, 12, 0, 0, 2), EncodeVintrp(2, 13, a1, 1, 0),
	                            EncodeVintrp(2, 14, a2, 2, 1), EncodeVintrp(2, 15, 0, 3, 2),
	                            EncodeExp0(0x00, 0xf),        EncodeExp1(12, 13, 14, 15),
	                            EncodeSopp(0x01)};
	static ShaderPixelInputInfo ps_info {};
	ps_info = {};
	ps_info.input_num                 = layout.read_count;
	ps_info.ps_system_input_base      = 2;
	ps_info.custom_interpolation_mask = (1u << layout.read_count) - 1u;
	ps_info.ps_perspective_center_vgpr = 0;
	for (uint32_t i = 0; i < std::size(ps_info.interpolator_settings); i++) ps_info.interpolator_settings[i] = i;
	for (uint32_t i = 0; i < layout.read_count; i++) ps_info.interpolator_settings[i] = 0x00000420u | layout.reads[i];
	static const std::array<uint32_t, 64> user_data = [] {
		std::array<uint32_t, 64> data {};
		data[3] = 3u << 28u;
		return data;
	}();
	CompileOptions options;
	options.stage            = ShaderType::Pixel;
	options.user_data        = user_data;
	options.input_info.pixel = &ps_info;
	auto translated          = TranslateProgram(ps_code, options);
	auto plan                = IR::ExtractResourcePlan(translated.program);
	IR::ResourceSnapshot       resources;
	IR::ResourceSpecialization specialization;
	const IR::SrtRuntime       runtime {.user_data = options.user_data,
	                                    .shader_base = reinterpret_cast<uint64_t>(ps_code),
	                                    .read_specialization_memory = [](void*, uint64_t, std::span<uint32_t>) { return false; }};
	Check(IR::MaterializeResources(plan, runtime, resources, specialization), "PS resources did not materialize");
	auto compiled = CompileProgram(std::move(translated), options, specialization, 0);
	uint32_t ps_mask = 0;
	Spirv::PvStoreInfo ps_store;
	ps_store.role          = Spirv::PvStoreInfo::Role::Pixel;
	ps_store.coherent      = true;
	ps_store.location_mask = &ps_mask;
	out.ps                 = Spirv::EmitProgram(compiled.program, options.input_info, 0, false, false, nullptr, &ps_store);

	Check(ClassifyPvVsStoreInterface(vs_mask, ps_mask) == PvVsStoreInterfaceVerdict::Use, "VS/PS pair is not a VS-store candidate");
	uint32_t expect_ps = 0;
	for (uint32_t k = 0; k < layout.read_count; ++k) expect_ps |= 1u << layout.reads[k];
	if (ps_mask != expect_ps) std::fprintf(stderr, "PS location mask %08x, expected %08x\n", ps_mask, expect_ps);
	Check(ps_mask == expect_ps, "the PS does not read the locations of the layout");
	out.vs_mask = vs_mask;
	out.ps_mask = ps_mask;
	if (compact) {
		// the PS of a compact record takes its slots from its own mask; the VS gets the mask of the pair
		FillPvCompactPixelInfo(ps_store, ps_mask, vs_mask);
		ps_store.location_mask = nullptr;
		out.ps = Spirv::EmitProgram(compiled.program, options.input_info, 0, false, false, nullptr, &ps_store);
		Check(spec_id != 0 && PvCompactPatchMask(out.vs, spec_id, ps_mask), "the VS has no mask specialization constant");
		out.stride_vec4 = PvCompactStride(ps_mask);
		out.ids_slot    = -1;
	} else {
		out.stride_vec4 = PvVsStoreStrideFor(vs_mask);
		out.ids_slot    = static_cast<int32_t>(layout.ids);
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
	Check(tools.Validate(out.vs), "VS-store VS does not validate");
	Check(tools.Validate(out.ps), "VS-store PS does not validate");
	return out;
}

// ---- scene (CPU model) ---------------------------------------------------------------------------

constexpr uint32_t kDim      = 32; // render target
constexpr uint32_t kCell     = 16; // one triangle per cell: 2x2 cells
constexpr uint32_t kVertices = 24; // 4 triangles x 3 corners, plus 12 spare vertices for offsets
constexpr uint32_t kStrideB  = 32; // float4 position, float4 color

struct Vertex {
	float position[4];
	float color[4];
};

std::array<Vertex, kVertices> MakeVertices() {
	std::array<Vertex, kVertices> v {};
	for (uint32_t i = 0; i < kVertices; ++i) {
		const uint32_t tri = (i / 3u) % 4u;
		const uint32_t cx  = tri % 2u, cy = tri / 2u;
		const float    x0  = static_cast<float>(cx * kCell), y0 = static_cast<float>(cy * kCell);
		const float    px[3] = {x0 + 2.f, x0 + 14.f, x0 + 2.f};
		const float    py[3] = {y0 + 2.f, y0 + 2.f, y0 + 14.f};
		const uint32_t k     = i % 3u;
		v[i].position[0] = px[k] / (kDim / 2.f) - 1.f;
		v[i].position[1] = py[k] / (kDim / 2.f) - 1.f;
		v[i].position[2] = 0.5f;
		v[i].position[3] = 1.f;
		// distinct, normal floats: any mix-up of a vertex, a corner or a component shows
		v[i].color[0] = 1.0f + 0.03125f * static_cast<float>(i);
		v[i].color[1] = 2.0f + 0.0625f * static_cast<float>(i);
		v[i].color[2] = 3.0f + 0.125f * static_cast<float>(i);
		v[i].color[3] = 4.0f + 0.25f * static_cast<float>(i);
	}
	return v;
}

struct Case {
	const char*           name;
	bool                  index32;
	std::vector<uint32_t> indices;       // per triangle corner, relative to vertex_offset
	int32_t               vertex_offset;
	uint32_t              first_instance;
	uint32_t              instance_count;
	uint32_t              bind_byte_offset; // where the index data starts inside the index buffer
};

uint32_t Bits(float f) {
	uint32_t u;
	std::memcpy(&u, &f, 4);
	return u;
}

// Expected pixel of triangle t of the draw: {corner0.x(+inst), corner1.y, corner2.z, corner0.w}.
std::array<uint32_t, 4> ExpectedPixel(const std::array<Vertex, kVertices>& v, const Case& c, uint32_t tri) {
	const uint32_t last_instance = c.first_instance + c.instance_count - 1;
	const auto     vertex = [&](uint32_t corner) { return static_cast<uint32_t>(static_cast<int32_t>(c.indices[tri * 3 + corner]) + c.vertex_offset); };
	std::array<uint32_t, 4> out {};
	const auto& add = g_layout->add;
	out[0] = Bits(v[vertex(0)].color[0]) + add[0] + last_instance;
	out[1] = Bits(v[vertex(1)].color[1]) + (g_layout->read_count > 1 ? add[1] : 0u);
	out[2] = Bits(v[vertex(2)].color[2]) + (g_layout->read_count > 2 ? add[2] : 0u);
	out[3] = Bits(v[vertex(0)].color[3]) + add[0];
	return out;
}

// ---- Vulkan --------------------------------------------------------------------------------------

#define VK_FUNCS(X)                                                                                                        \
	X(vkCreateDevice) X(vkGetDeviceProcAddr) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceMemoryProperties)          \
	X(vkGetPhysicalDeviceQueueFamilyProperties) X(vkGetPhysicalDeviceFeatures) X(vkGetPhysicalDeviceProperties)

#define VK_DEVICE_FUNCS(X)                                                                                                 \
	X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkGetBufferMemoryRequirements) X(vkAllocateMemory) X(vkBindBufferMemory)      \
	X(vkMapMemory) X(vkCreateDescriptorSetLayout) X(vkCreatePipelineLayout) X(vkCreateShaderModule)                         \
	X(vkCreateGraphicsPipelines) X(vkCreateDescriptorPool) X(vkAllocateDescriptorSets) X(vkUpdateDescriptorSets)           \
	X(vkCreateCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) X(vkEndCommandBuffer)                       \
	X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdBeginRenderPass) X(vkCmdEndRenderPass)    \
	X(vkCmdBindVertexBuffers) X(vkCmdBindIndexBuffer) X(vkCmdDrawIndexed) X(vkCmdCopyImageToBuffer) X(vkQueueSubmit)      \
	X(vkQueueWaitIdle) X(vkCreateRenderPass) X(vkCreateFramebuffer) X(vkCreateImage) X(vkGetImageMemoryRequirements)       \
	X(vkBindImageMemory) X(vkCreateImageView) X(vkCmdPipelineBarrier)

#define DECLARE(name) PFN_##name name##_ = nullptr;
VK_FUNCS(DECLARE)
VK_DEVICE_FUNCS(DECLARE)
#undef DECLARE

struct Gpu {
	VkInstance                       instance = VK_NULL_HANDLE;
	VkPhysicalDevice                 physical = VK_NULL_HANDLE;
	VkDevice                         device   = VK_NULL_HANDLE;
	VkQueue                          queue    = VK_NULL_HANDLE;
	uint32_t                         family   = 0;
	VkPhysicalDeviceMemoryProperties memory {};
	VkDeviceSize                     storage_alignment = 16;
};

struct Buf {
	VkBuffer       buffer = VK_NULL_HANDLE;
	VkDeviceMemory memory = VK_NULL_HANDLE;
	void*          mapped = nullptr;
	size_t         size   = 0;
};

uint32_t MemoryType(const Gpu& gpu, uint32_t bits, VkMemoryPropertyFlags flags) {
	for (uint32_t i = 0; i < gpu.memory.memoryTypeCount; ++i) {
		if ((bits & (1u << i)) && (gpu.memory.memoryTypes[i].propertyFlags & flags) == flags) return i;
	}
	Fail("no suitable memory type");
}

Buf MakeBuffer(const Gpu& gpu, size_t size, VkBufferUsageFlags usage) {
	Buf b;
	b.size = size;
	VkBufferCreateInfo info {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	info.size  = size;
	info.usage = usage;
	CheckVk(vkCreateBuffer_(gpu.device, &info, nullptr, &b.buffer), "vkCreateBuffer");
	VkMemoryRequirements req;
	vkGetBufferMemoryRequirements_(gpu.device, b.buffer, &req);
	VkMemoryAllocateInfo alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	alloc.allocationSize  = req.size;
	alloc.memoryTypeIndex = MemoryType(gpu, req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	CheckVk(vkAllocateMemory_(gpu.device, &alloc, nullptr, &b.memory), "vkAllocateMemory");
	CheckVk(vkBindBufferMemory_(gpu.device, b.buffer, b.memory, 0), "vkBindBufferMemory");
	CheckVk(vkMapMemory_(gpu.device, b.memory, 0, size, 0, &b.mapped), "vkMapMemory");
	std::memset(b.mapped, 0xA5, size);
	return b;
}

Gpu InitGpu() {
	const char* driver = std::getenv("SDL_VULKAN_LIBRARY");
	void*       lib    = dlopen(driver != nullptr ? driver : "libMoltenVK.dylib", RTLD_NOW | RTLD_LOCAL);
	if (lib == nullptr) Fail("cannot load MoltenVK; set SDL_VULKAN_LIBRARY");
	auto get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
	if (get_instance_proc == nullptr) Fail("dlsym vkGetInstanceProcAddr");
	auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(get_instance_proc(VK_NULL_HANDLE, "vkCreateInstance"));
	auto enumerate_extensions = reinterpret_cast<PFN_vkEnumerateInstanceExtensionProperties>(
	    get_instance_proc(VK_NULL_HANDLE, "vkEnumerateInstanceExtensionProperties"));
	if (enumerate_extensions == nullptr) Fail("vkEnumerateInstanceExtensionProperties");
	uint32_t extension_count = 0;
	CheckVk(enumerate_extensions(nullptr, &extension_count, nullptr), "vkEnumerateInstanceExtensionProperties");
	std::vector<VkExtensionProperties> available(extension_count);
	CheckVk(enumerate_extensions(nullptr, &extension_count, available.data()), "vkEnumerateInstanceExtensionProperties list");
	bool has_portability = false;
	for (const auto& e: available) {
		if (std::strcmp(e.extensionName, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) == 0) has_portability = true;
	}
	const char* extensions[] {VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME};
	VkApplicationInfo app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_3;
	VkInstanceCreateInfo info {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	info.pApplicationInfo        = &app;
	info.flags                   = has_portability ? VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR : 0;
	info.enabledExtensionCount   = has_portability ? 1u : 0u;
	info.ppEnabledExtensionNames = has_portability ? extensions : nullptr;
	Gpu gpu;
	CheckVk(create_instance(&info, nullptr, &gpu.instance), "vkCreateInstance");
#define LOAD_INSTANCE(name) name##_ = reinterpret_cast<PFN_##name>(get_instance_proc(gpu.instance, #name));
	VK_FUNCS(LOAD_INSTANCE)
#undef LOAD_INSTANCE
	uint32_t count = 0;
	CheckVk(vkEnumeratePhysicalDevices_(gpu.instance, &count, nullptr), "vkEnumeratePhysicalDevices");
	Check(count > 0, "no physical device");
	std::vector<VkPhysicalDevice> devices(count);
	CheckVk(vkEnumeratePhysicalDevices_(gpu.instance, &count, devices.data()), "vkEnumeratePhysicalDevices list");
	gpu.physical = devices[0];
	vkGetPhysicalDeviceMemoryProperties_(gpu.physical, &gpu.memory);
	VkPhysicalDeviceProperties props {};
	vkGetPhysicalDeviceProperties_(gpu.physical, &props);
	gpu.storage_alignment = props.limits.minStorageBufferOffsetAlignment;
	uint32_t qcount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties_(gpu.physical, &qcount, nullptr);
	std::vector<VkQueueFamilyProperties> queues(qcount);
	vkGetPhysicalDeviceQueueFamilyProperties_(gpu.physical, &qcount, queues.data());
	for (uint32_t i = 0; i < qcount; ++i) {
		if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
			gpu.family = i;
			break;
		}
	}
	VkPhysicalDeviceFeatures supported {};
	vkGetPhysicalDeviceFeatures_(gpu.physical, &supported);
	Check(supported.vertexPipelineStoresAndAtomics, "device lacks vertexPipelineStoresAndAtomics (the VS cannot store)");
	VkPhysicalDeviceFeatures features {};
	features.vertexPipelineStoresAndAtomics = VK_TRUE;
	features.fragmentStoresAndAtomics       = supported.fragmentStoresAndAtomics;
	features.shaderClipDistance             = supported.shaderClipDistance; // the emitted VS has a clip guard
	features.geometryShader                 = supported.geometryShader; // PrimitiveId in the PS needs the Geometry capability
	float                   priority = 1.f;
	VkDeviceQueueCreateInfo queue_info {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	queue_info.queueFamilyIndex = gpu.family;
	queue_info.queueCount       = 1;
	queue_info.pQueuePriorities = &priority;
	VkDeviceCreateInfo device_info {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	device_info.queueCreateInfoCount = 1;
	device_info.pQueueCreateInfos    = &queue_info;
	device_info.pEnabledFeatures     = &features;
	CheckVk(vkCreateDevice_(gpu.physical, &device_info, nullptr, &gpu.device), "vkCreateDevice");
#define LOAD_DEVICE(name) name##_ = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr_(gpu.device, #name));
	VK_DEVICE_FUNCS(LOAD_DEVICE)
#undef LOAD_DEVICE
	vkGetDeviceQueue_(gpu.device, gpu.family, 0, &gpu.queue);
	return gpu;
}

int RunGpu(const Gpu& gpu, const Shaders& shaders) {
	const auto vertices = MakeVertices();
	auto       module   = [&](const std::vector<uint32_t>& words, const char* what) {
        VkShaderModuleCreateInfo info {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        info.codeSize = words.size() * 4;
        info.pCode    = words.data();
        VkShaderModule m = VK_NULL_HANDLE;
        CheckVk(vkCreateShaderModule_(gpu.device, &info, nullptr, &m), what);
        return m;
	};
	const VkShaderModule vs_module = module(shaders.vs, "VS module");
	const VkShaderModule ps_module = module(shaders.ps, "PS module");

	// layouts: set 0 empty, set 1 = cap (VS|FS) + indices (FS); push = native data + the VS-store block
	VkDescriptorSetLayoutCreateInfo empty_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	VkDescriptorSetLayout           empty_layout = VK_NULL_HANDLE;
	CheckVk(vkCreateDescriptorSetLayout_(gpu.device, &empty_info, nullptr, &empty_layout), "empty set layout");
	const VkDescriptorSetLayoutBinding bindings[2] {
	    {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
	    {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
	VkDescriptorSetLayoutCreateInfo set_info {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
	set_info.bindingCount = 2;
	set_info.pBindings    = bindings;
	VkDescriptorSetLayout pv_layout = VK_NULL_HANDLE;
	CheckVk(vkCreateDescriptorSetLayout_(gpu.device, &set_info, nullptr, &pv_layout), "pv set layout");
	const VkDescriptorSetLayout layouts[2] {empty_layout, pv_layout};
	const uint32_t              push_offset = IR::NativePushConstantSize;
	const VkPushConstantRange   push {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
	                                  static_cast<uint32_t>(push_offset + kPvVsStorePushDwords * 4)};
	VkPipelineLayoutCreateInfo  pl_info {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pl_info.setLayoutCount         = 2;
	pl_info.pSetLayouts            = layouts;
	pl_info.pushConstantRangeCount = 1;
	pl_info.pPushConstantRanges    = &push;
	VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
	CheckVk(vkCreatePipelineLayout_(gpu.device, &pl_info, nullptr, &pipeline_layout), "pipeline layout");

	// render target
	VkAttachmentDescription color {};
	color.format         = VK_FORMAT_R32G32B32A32_SFLOAT;
	color.samples        = VK_SAMPLE_COUNT_1_BIT;
	color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
	color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
	color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
	color.finalLayout    = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
	const VkAttachmentReference color_ref {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
	VkSubpassDescription        subpass {};
	subpass.pipelineBindPoint    = VK_PIPELINE_BIND_POINT_GRAPHICS;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments    = &color_ref;
	VkRenderPassCreateInfo rp_info {VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
	rp_info.attachmentCount = 1;
	rp_info.pAttachments    = &color;
	rp_info.subpassCount    = 1;
	rp_info.pSubpasses      = &subpass;
	VkRenderPass render_pass = VK_NULL_HANDLE;
	CheckVk(vkCreateRenderPass_(gpu.device, &rp_info, nullptr, &render_pass), "render pass");
	VkImageCreateInfo image_info {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
	image_info.imageType     = VK_IMAGE_TYPE_2D;
	image_info.format        = VK_FORMAT_R32G32B32A32_SFLOAT;
	image_info.extent        = {kDim, kDim, 1};
	image_info.mipLevels     = 1;
	image_info.arrayLayers   = 1;
	image_info.samples       = VK_SAMPLE_COUNT_1_BIT;
	image_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
	image_info.usage         = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
	image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VkImage image = VK_NULL_HANDLE;
	CheckVk(vkCreateImage_(gpu.device, &image_info, nullptr, &image), "image");
	VkMemoryRequirements image_req;
	vkGetImageMemoryRequirements_(gpu.device, image, &image_req);
	VkMemoryAllocateInfo image_alloc {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	image_alloc.allocationSize  = image_req.size;
	image_alloc.memoryTypeIndex = MemoryType(gpu, image_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	VkDeviceMemory image_memory = VK_NULL_HANDLE;
	CheckVk(vkAllocateMemory_(gpu.device, &image_alloc, nullptr, &image_memory), "image memory");
	CheckVk(vkBindImageMemory_(gpu.device, image, image_memory, 0), "bind image");
	VkImageViewCreateInfo view_info {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
	view_info.image            = image;
	view_info.viewType         = VK_IMAGE_VIEW_TYPE_2D;
	view_info.format           = VK_FORMAT_R32G32B32A32_SFLOAT;
	view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	VkImageView view = VK_NULL_HANDLE;
	CheckVk(vkCreateImageView_(gpu.device, &view_info, nullptr, &view), "image view");
	VkFramebufferCreateInfo fb_info {VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
	fb_info.renderPass      = render_pass;
	fb_info.attachmentCount = 1;
	fb_info.pAttachments    = &view;
	fb_info.width           = kDim;
	fb_info.height          = kDim;
	fb_info.layers          = 1;
	VkFramebuffer framebuffer = VK_NULL_HANDLE;
	CheckVk(vkCreateFramebuffer_(gpu.device, &fb_info, nullptr, &framebuffer), "framebuffer");

	// pipeline: native vertex input (location 0 / 1 = float4), triangle list, no culling
	VkPipelineShaderStageCreateInfo stages[2] {};
	stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vs_module, "main", nullptr};
	stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, ps_module, "main", nullptr};
	const VkVertexInputBindingDescription   binding {0, kStrideB, VK_VERTEX_INPUT_RATE_VERTEX};
	const VkVertexInputAttributeDescription attributes[2] {{0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16}};
	VkPipelineVertexInputStateCreateInfo vi {VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
	vi.vertexBindingDescriptionCount   = 1;
	vi.pVertexBindingDescriptions      = &binding;
	vi.vertexAttributeDescriptionCount = 2;
	vi.pVertexAttributeDescriptions    = attributes;
	VkPipelineInputAssemblyStateCreateInfo ia {VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
	ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
	const VkViewport viewport {0.f, 0.f, float(kDim), float(kDim), 0.f, 1.f};
	const VkRect2D   scissor {{0, 0}, {kDim, kDim}};
	VkPipelineViewportStateCreateInfo vp {VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
	vp.viewportCount = 1;
	vp.pViewports    = &viewport;
	vp.scissorCount  = 1;
	vp.pScissors     = &scissor;
	VkPipelineRasterizationStateCreateInfo rs {VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
	rs.polygonMode = VK_POLYGON_MODE_FILL;
	rs.cullMode    = VK_CULL_MODE_NONE;
	rs.frontFace   = VK_FRONT_FACE_COUNTER_CLOCKWISE;
	rs.lineWidth   = 1.f;
	VkPipelineMultisampleStateCreateInfo ms {VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
	ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
	VkPipelineColorBlendAttachmentState blend {};
	blend.colorWriteMask = 0xf;
	VkPipelineColorBlendStateCreateInfo cb {VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
	cb.attachmentCount = 1;
	cb.pAttachments    = &blend;
	VkGraphicsPipelineCreateInfo gi {VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
	gi.stageCount          = 2;
	gi.pStages             = stages;
	gi.pVertexInputState   = &vi;
	gi.pInputAssemblyState = &ia;
	gi.pViewportState      = &vp;
	gi.pRasterizationState = &rs;
	gi.pMultisampleState   = &ms;
	gi.pColorBlendState    = &cb;
	gi.layout              = pipeline_layout;
	gi.renderPass          = render_pass;
	VkPipeline pipeline = VK_NULL_HANDLE;
	CheckVk(vkCreateGraphicsPipelines_(gpu.device, VK_NULL_HANDLE, 1, &gi, nullptr, &pipeline), "graphics pipeline");

	VkCommandPoolCreateInfo cp_info {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	cp_info.queueFamilyIndex = gpu.family;
	VkCommandPool pool = VK_NULL_HANDLE;
	CheckVk(vkCreateCommandPool_(gpu.device, &cp_info, nullptr, &pool), "command pool");
	VkCommandBufferAllocateInfo cb_alloc {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cb_alloc.commandPool        = pool;
	cb_alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cb_alloc.commandBufferCount = 1;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	CheckVk(vkAllocateCommandBuffers_(gpu.device, &cb_alloc, &cmd), "command buffer");
	const VkDescriptorPoolSize pool_size {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64}; // 2 bindings per set, up to 32 sets
	VkDescriptorPoolCreateInfo dp_info {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
	dp_info.maxSets       = 32;
	dp_info.poolSizeCount = 1;
	dp_info.pPoolSizes    = &pool_size;
	VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
	CheckVk(vkCreateDescriptorPool_(gpu.device, &dp_info, nullptr, &descriptor_pool), "descriptor pool");

	Buf vertex_buffer = MakeBuffer(gpu, sizeof(Vertex) * kVertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
	std::memcpy(vertex_buffer.mapped, vertices.data(), sizeof(Vertex) * kVertices);
	Buf readback = MakeBuffer(gpu, kDim * kDim * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

	// Triangles t = 0..3 use consecutive corners; case index lists permute vertices and add spare ones.
	std::vector<Case> cases;
	cases.push_back({"16-bit, vertex_offset=+3, 1 instance", false, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 0, 1, 0});
	cases.push_back({"32-bit, first_instance=1, 2 instances", true, {2, 0, 1, 5, 3, 4, 8, 6, 7, 11, 9, 10}, 0, 1, 2, 0});
	cases.push_back({"16-bit, vertex_offset=-2, index buffer at byte 18 (unaligned descriptor)", false,
	                 {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, -2, 0, 1, 18});
	// odd-element descriptor starts (the index buffer moves into first_index); the triangles keep whole vertex groups
	cases.push_back({"16-bit, vertex_offset=0, index buffer at byte 2", false,
	                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 0, 0, 1, 2});
	cases.push_back({"16-bit, vertex_offset=-3, index buffer at byte 6", false,
	                 {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, -3, 0, 1, 6});
	cases.push_back({"16-bit, vertex_offset=+3, index buffer at byte 22", false,
	                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 3, 0, 1, 22});
	cases.push_back({"32-bit, vertex_offset=+6, first_instance=3, 2 instances, index buffer at byte 36", true,
	                 {1, 2, 0, 4, 5, 3, 7, 8, 6, 10, 11, 9}, 6, 3, 2, 36});

	// S151 (--stress): >= 2M triangles x 2..4 instances in one draw, so that the tile parameter buffer of the TBDR GPU
	// overflows mid-draw. Triangle t sits in cell t % 4 with whole-vertex corners, so the last writer of every cell is
	// known (the last 4 triangles) and the result must still match the CPU model bit for bit.
	if (g_stress) {
		constexpr uint32_t kStressTriangles = 2'000'000u;
		for (const bool index32: {true, false}) {
			Case big {index32 ? "STRESS 32-bit, 2M triangles, 4 instances, first_instance=1"
			                  : "STRESS 16-bit, 2M triangles, 2 instances, index buffer at byte 6",
			          index32, {}, 0, index32 ? 1u : 0u, index32 ? 4u : 2u, index32 ? 0u : 6u};
			big.indices.resize(size_t {kStressTriangles} * 3);
			for (uint32_t t = 0; t < kStressTriangles; ++t)
				for (uint32_t k = 0; k < 3; ++k) big.indices[size_t {t} * 3 + k] = (t % 4u) * 3u + k;
			cases.push_back(std::move(big));
		}
	}

	int failures = 0;
	for (const auto& c: cases) {
		const uint32_t element = c.index32 ? 4u : 2u;
		const uint32_t count   = static_cast<uint32_t>(c.indices.size());
		Buf index_buffer       = MakeBuffer(gpu, c.bind_byte_offset + count * element + 64, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
		for (uint32_t i = 0; i < count; ++i) {
			auto* dst = static_cast<uint8_t*>(index_buffer.mapped) + c.bind_byte_offset + i * element;
			if (c.index32) {
				std::memcpy(dst, &c.indices[i], 4);
			} else {
				const uint16_t v = static_cast<uint16_t>(c.indices[i]);
				std::memcpy(dst, &v, 2);
			}
		}
		// the cap region: base 5 records into the slab, `span` records per instance
		const uint32_t first = static_cast<uint32_t>(static_cast<int32_t>(*std::min_element(c.indices.begin(), c.indices.end())) + c.vertex_offset);
		const uint32_t last  = static_cast<uint32_t>(static_cast<int32_t>(*std::max_element(c.indices.begin(), c.indices.end())) + c.vertex_offset);
		PvVsStoreDraw draw;
		draw.base           = 5;
		draw.span           = last - first + 1;
		draw.first          = first;
		draw.first_instance = c.first_instance;
		draw.vertex_offset  = c.vertex_offset;
		draw.index32        = c.index32;
		const auto bind     = PvVsStoreBindIndex(c.bind_byte_offset, element, gpu.storage_alignment);
		Check(bind.ok, "index offset cannot be bound");
		draw.first_index = bind.first_index;
		const uint64_t records = uint64_t {draw.span} * c.instance_count;
		Buf cap = MakeBuffer(gpu, (draw.base + records + 4) * shaders.stride_vec4 * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

		VkDescriptorSetAllocateInfo alloc {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		alloc.descriptorPool     = descriptor_pool;
		alloc.descriptorSetCount = 1;
		alloc.pSetLayouts        = &pv_layout;
		VkDescriptorSet set = VK_NULL_HANDLE;
		CheckVk(vkAllocateDescriptorSets_(gpu.device, &alloc, &set), "allocate set");
		const VkDescriptorBufferInfo infos[2] {{cap.buffer, 0, cap.size},
		                                       {index_buffer.buffer, bind.offset, PvVsStoreIndexDescriptorFor(c.bind_byte_offset, element, gpu.storage_alignment, count * element, index_buffer.size).range}};
		VkWriteDescriptorSet writes[2] {};
		for (uint32_t i = 0; i < 2; ++i) {
			writes[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
			writes[i].dstSet          = set;
			writes[i].dstBinding      = i;
			writes[i].descriptorCount = 1;
			writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writes[i].pBufferInfo     = &infos[i];
		}
		vkUpdateDescriptorSets_(gpu.device, 2, writes, 0, nullptr);

		VkCommandBufferBeginInfo begin {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		CheckVk(vkBeginCommandBuffer_(cmd, &begin), "begin");
		const VkClearValue clear {{{0.f, 0.f, 0.f, 0.f}}};
		VkRenderPassBeginInfo rp_begin {VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
		rp_begin.renderPass      = render_pass;
		rp_begin.framebuffer     = framebuffer;
		rp_begin.renderArea      = {{0, 0}, {kDim, kDim}};
		rp_begin.clearValueCount = 1;
		rp_begin.pClearValues    = &clear;
		vkCmdBeginRenderPass_(cmd, &rp_begin, VK_SUBPASS_CONTENTS_INLINE);
		vkCmdBindPipeline_(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
		const VkDeviceSize zero = 0;
		vkCmdBindVertexBuffers_(cmd, 0, 1, &vertex_buffer.buffer, &zero);
		vkCmdBindIndexBuffer_(cmd, index_buffer.buffer, c.bind_byte_offset, c.index32 ? VK_INDEX_TYPE_UINT32 : VK_INDEX_TYPE_UINT16);
		vkCmdBindDescriptorSets_(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 1, 1, &set, 0, nullptr);
		const auto block = PvVsStorePushBlock(draw, shaders.stride_vec4);
		vkCmdPushConstants_(cmd, pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, push_offset, sizeof(block), block.data());
		// same as the renderer: firstIndex 0, the index buffer is bound at its offset
		vkCmdDrawIndexed_(cmd, count, c.instance_count, 0, c.vertex_offset, c.first_instance);
		vkCmdEndRenderPass_(cmd);
		VkBufferImageCopy copy {};
		copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
		copy.imageExtent      = {kDim, kDim, 1};
		vkCmdCopyImageToBuffer_(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buffer, 1, &copy);
		CheckVk(vkEndCommandBuffer_(cmd), "end");
		VkSubmitInfo submit {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &cmd;
		CheckVk(vkQueueSubmit_(gpu.queue, 1, &submit, VK_NULL_HANDLE), "submit");
		CheckVk(vkQueueWaitIdle_(gpu.queue), "wait idle");

		int bad_pixels = 0, bad_records = 0;
		const auto* pixels = static_cast<const uint32_t*>(readback.mapped);
		const uint32_t tri_count = static_cast<uint32_t>(c.indices.size() / 3);
		for (uint32_t t = tri_count - 4; t < tri_count; ++t) { // the last writer of each cell
			// the triangle lands in the cell of its vertices (vertex v is in cell (v / 3) % 4)
			const uint32_t v0 = static_cast<uint32_t>(static_cast<int32_t>(c.indices[t * 3]) + c.vertex_offset);
			const uint32_t cell = (v0 / 3u) % 4u;
			const uint32_t px = (cell % 2) * kCell + 4, py = (cell / 2) * kCell + 4;
			const auto     expected = ExpectedPixel(vertices, c, t);
			for (uint32_t ch = 0; ch < 4; ++ch) {
				const uint32_t got = pixels[(py * kDim + px) * 4 + ch];
				if (got != expected[ch]) {
					if (bad_pixels++ < 6) std::fprintf(stderr, "  [%s] tri %u ch %u: got %08x expected %08x\n", c.name, t, ch, got, expected[ch]);
				}
			}
		}
		// cap: record(instance, vertex) holds {vertex, instance, 0, 0} in the slot of the ids location (not stored by a compact record)
		const auto* cap_words = static_cast<const uint32_t*>(cap.mapped);
		std::vector<bool> used_vertex(last + 1, false);
		for (auto idx: c.indices) used_vertex[static_cast<uint32_t>(static_cast<int32_t>(idx) + c.vertex_offset)] = true;
		for (uint32_t i = 0; i < c.instance_count; ++i) {
			for (uint32_t vtx = first; vtx <= last; ++vtx) {
				if (!used_vertex[vtx]) continue;
				const uint32_t record = PvVsStoreVertexRecord(draw.base, draw.span, draw.first, draw.first_instance, c.first_instance + i, vtx);
				if (shaders.ids_slot < 0) continue;
				const uint32_t* slot1 = cap_words + (record * shaders.stride_vec4 + static_cast<uint32_t>(shaders.ids_slot)) * 4;
				if (slot1[0] != vtx || slot1[1] != c.first_instance + i) {
					if (bad_records++ < 6) {
						std::fprintf(stderr, "  [%s] record(inst %u, vertex %u): VS stored {vertex %u, instance %u}\n", c.name,
						             c.first_instance + i, vtx, slot1[0], slot1[1]);
					}
				}
			}
		}
		// nothing outside the region of the draw was written: the stride is what the shaders use, a store above it would land here
		int overrun = 0;
		for (size_t w = 0; w < cap.size / 4; ++w) {
			const size_t record_word = w / (shaders.stride_vec4 * 4u);
			const bool   inside      = record_word >= draw.base && record_word < draw.base + records;
			if (!inside && cap_words[w] != 0xA5A5A5A5u) overrun++;
		}
		std::printf("%s: %s (pixels bad %d, records bad %d, words written outside the region %d)\n",
		            bad_pixels == 0 && bad_records == 0 && overrun == 0 ? "PASS" : "FAIL", c.name, bad_pixels, bad_records, overrun);
		failures += (bad_pixels != 0) + (bad_records != 0) + (overrun != 0);
	}
	std::printf("arm result: %s\n", failures == 0 ? "PASS" : "FAIL");
	return failures == 0 ? 0 : 1;
}

} // namespace
} // namespace Libs::Graphics

int main(int argc, char** argv) {
	using namespace Libs::Graphics;
	bool compact_run = false, dump = false;
	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--stress") == 0) g_stress = true;
		if (std::strcmp(argv[i], "--compact") == 0) compact_run = true;
		if (std::strcmp(argv[i], "--dump") == 0) dump = true;
	}
	EnsureConfigInitialized();
	// default: the layout of S150 with the old record. --compact (S409): every layout with the old and with the compact record,
	// all against the same CPU model, so the pixels of the compact arm equal those of the old one.
	struct Arm { const Layout* layout; bool compact; };
	std::vector<Arm> arms;
	if (compact_run) {
		for (const auto& layout: kLayouts) {
			arms.push_back({&layout, false});
			arms.push_back({&layout, true});
		}
	} else {
		arms.push_back({&kLayouts[0], false});
	}
	if (dump) {
		const Shaders shaders = BuildShaders(*arms.back().layout, arms.back().compact);
		spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
		std::string          text;
		for (const auto* words: {&shaders.vs, &shaders.ps}) {
			tools.Disassemble(*words, &text, SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES);
			std::printf("---- %s\n%s\n", words == &shaders.vs ? "VS" : "PS", text.c_str());
		}
		return 0;
	}
	const Gpu gpu = InitGpu();
	int failures = 0;
	for (const auto& arm: arms) {
		g_layout = arm.layout;
		const Shaders shaders = BuildShaders(*arm.layout, arm.compact);
		std::printf("==== %s, %s record: VS mask 0x%x, PS mask 0x%x, stride %u\n", arm.layout->name, arm.compact ? "compact" : "old",
		            shaders.vs_mask, shaders.ps_mask, shaders.stride_vec4);
		failures += RunGpu(gpu, shaders);
	}
	std::printf("per_vertex_vsstore_gpu_tests (%zu arms): %s\n", arms.size(), failures == 0 ? "PASS" : "FAIL");
	return failures == 0 ? 0 : 1;
}
