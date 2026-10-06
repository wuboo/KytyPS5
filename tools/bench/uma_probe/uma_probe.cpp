// uma_probe: standalone spike (S67) - can guest memory be a VkBuffer without a copy through
// VK_EXT_external_memory_host on MoltenVK?  Loads libMoltenVK.dylib directly (no Vulkan loader, no emulator).
//
//   uma_probe --dylib <libMoltenVK.dylib> [--size-mb 1024] [--backing shm|anon] [--mutant] [--no-canary]
//
// Checks (exit code 0 only if all pass): main import + BDA dispatch + CPU read without copy, alignment canary,
// checker positive control (verify before dispatch must FAIL), CPU->GPU visibility after import (no copy).
#include <dlfcn.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan_core.h>

#include "uma_probe_spv.h"

#define CHECK(x)                                                                                         \
	do {                                                                                                 \
		VkResult r_ = (x);                                                                               \
		if (r_ != VK_SUCCESS) {                                                                          \
			printf("FATAL %s:%d %s = %d\n", __FILE__, __LINE__, #x, static_cast<int>(r_));              \
			exit(2);                                                                                     \
		}                                                                                                \
	} while (0)

static double NowMs() {
	using namespace std::chrono;
	return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

struct MemStat {
	double wired_mb, resident_mb, footprint_mb;
};

static MemStat GetMemStat() {
	MemStat            s {};
	vm_statistics64_data_t vm {};
	mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
	host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &cnt);
	s.wired_mb = static_cast<double>(vm.wire_count) * static_cast<double>(vm_page_size) / 1048576.0;
	task_vm_info_data_t ti {};
	cnt = TASK_VM_INFO_COUNT;
	task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&ti), &cnt);
	s.resident_mb  = static_cast<double>(ti.resident_size) / 1048576.0;
	s.footprint_mb = static_cast<double>(ti.phys_footprint) / 1048576.0;
	return s;
}

static void PrintMem(const char* tag) {
	auto s = GetMemStat();
	printf("MEM %-28s system_wired=%.0f MB  proc_resident=%.0f MB  proc_footprint=%.0f MB\n", tag, s.wired_mb,
	       s.resident_mb, s.footprint_mb);
}

// ---- Vulkan entry points ------------------------------------------------------------------------------------
static PFN_vkGetInstanceProcAddr                     pGetInstanceProcAddr;
static PFN_vkCreateInstance                          pCreateInstance;
static PFN_vkEnumeratePhysicalDevices                pEnumeratePhysicalDevices;
static PFN_vkGetPhysicalDeviceProperties2            pGetPhysicalDeviceProperties2;
static PFN_vkGetPhysicalDeviceMemoryProperties       pGetPhysicalDeviceMemoryProperties;
static PFN_vkEnumerateDeviceExtensionProperties      pEnumerateDeviceExtensionProperties;
static PFN_vkGetPhysicalDeviceQueueFamilyProperties  pGetPhysicalDeviceQueueFamilyProperties;
static PFN_vkCreateDevice                            pCreateDevice;
static PFN_vkGetDeviceProcAddr                       pGetDeviceProcAddr;

#define DEV_FUNCS(X)                                                                                     \
	X(GetDeviceQueue) X(AllocateMemory) X(FreeMemory) X(MapMemory) X(UnmapMemory) X(CreateBuffer)       \
	X(DestroyBuffer) X(GetBufferMemoryRequirements) X(BindBufferMemory) X(GetBufferDeviceAddress)        \
	X(CreateShaderModule) X(DestroyShaderModule) X(CreatePipelineLayout) X(DestroyPipelineLayout)        \
	X(CreateComputePipelines) X(DestroyPipeline) X(CreateCommandPool) X(DestroyCommandPool)              \
	X(AllocateCommandBuffers) X(BeginCommandBuffer) X(EndCommandBuffer) X(CmdBindPipeline)               \
	X(CmdPushConstants) X(CmdDispatch) X(CreateFence) X(DestroyFence) X(WaitForFences) X(QueueSubmit)    \
	X(DeviceWaitIdle) X(DestroyDevice) X(GetMemoryHostPointerPropertiesEXT) X(ResetFences)
#define X(n) static PFN_vk##n p##n;
DEV_FUNCS(X)
#undef X

// ---- guest-like memory --------------------------------------------------------------------------------------
struct GuestMem {
	uint8_t* a    = nullptr; // view imported into Vulkan ("guest view")
	uint8_t* b    = nullptr; // second, independent view of the same pages ("backing alias" the CPU reads through)
	size_t   size = 0;
	int      fd   = -1;
};

static uint8_t* MapAligned(size_t size, size_t align, int fd) {
	// Rosetta (x86_64) only guarantees 4 KB mmap alignment; reserve, then MAP_FIXED an aligned view.
	void* res = mmap(nullptr, size + align, PROT_NONE, MAP_ANON | MAP_PRIVATE, -1, 0);
	if (res == MAP_FAILED) return nullptr;
	auto base = (reinterpret_cast<uintptr_t>(res) + align - 1) & ~(static_cast<uintptr_t>(align) - 1);
	int  fl   = MAP_FIXED | (fd >= 0 ? MAP_SHARED : (MAP_ANON | MAP_PRIVATE));
	void* p   = mmap(reinterpret_cast<void*>(base), size, PROT_READ | PROT_WRITE, fl, fd, 0);
	return p == MAP_FAILED ? nullptr : static_cast<uint8_t*>(p);
}

static bool MakeGuestMem(GuestMem& g, size_t size, bool shm, size_t align) {
	g.size = size;
	if (shm) {
		char name[64];
		snprintf(name, sizeof(name), "/uma-probe-%d-%zx", getpid(), size);
		g.fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (g.fd < 0) return false;
		shm_unlink(name);
		if (ftruncate(g.fd, static_cast<off_t>(size)) != 0) return false;
	}
	g.a = MapAligned(size, align, g.fd);
	if (!g.a) return false;
	if (shm) {
		g.b = MapAligned(size, align, g.fd);
		if (!g.b) return false;
	} else {
		g.b = g.a; // anonymous private memory has no second view
	}
	return true;
}

// ---- helpers ------------------------------------------------------------------------------------------------
struct Ctx {
	VkDevice         dev = nullptr;
	VkPhysicalDevice phys = nullptr;
	VkPhysicalDeviceMemoryProperties memprops {};
	uint32_t         queue_family = 0;
	VkQueue          queue = nullptr;
};

struct Imported {
	VkDeviceMemory mem = VK_NULL_HANDLE;
	VkBuffer       buf = VK_NULL_HANDLE;
	VkDeviceAddress addr = 0;
	VkResult       alloc_result = VK_SUCCESS;
	double         import_ms = 0, create_ms = 0;
};

static uint32_t PickType(const Ctx& c, uint32_t bits, VkMemoryPropertyFlags want) {
	for (uint32_t i = 0; i < c.memprops.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (c.memprops.memoryTypes[i].propertyFlags & want) == want) return i;
	for (uint32_t i = 0; i < c.memprops.memoryTypeCount; i++)
		if (bits & (1u << i)) return i;
	return ~0u;
}

// Imports [ptr, ptr+size) as VkDeviceMemory + VkBuffer (+ device address). Never aborts: returns the VkResult.
static Imported Import(const Ctx& c, void* ptr, size_t size, bool want_bda = true) {
	Imported out;
	VkMemoryHostPointerPropertiesEXT hp {VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
	VkResult r = pGetMemoryHostPointerPropertiesEXT(c.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, ptr, &hp);
	if (r != VK_SUCCESS) {
		out.alloc_result = r;
		return out;
	}
	VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bi.size  = size;
	bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
	           (want_bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0);
	double t0 = NowMs();
	r = pCreateBuffer(c.dev, &bi, nullptr, &out.buf);
	if (r != VK_SUCCESS) {
		out.alloc_result = r;
		return out;
	}
	VkMemoryRequirements req {};
	pGetBufferMemoryRequirements(c.dev, out.buf, &req);
	out.create_ms = NowMs() - t0;
	uint32_t type = PickType(c, hp.memoryTypeBits & req.memoryTypeBits,
	                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
	VkImportMemoryHostPointerInfoEXT imp {VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
	imp.handleType   = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
	imp.pHostPointer = ptr;
	VkMemoryAllocateFlagsInfo fi {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
	fi.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
	fi.pNext = &imp;
	VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	ai.pNext           = want_bda ? static_cast<void*>(&fi) : static_cast<void*>(&imp);
	ai.allocationSize  = size; // must stay a multiple of minImportedHostPointerAlignment
	ai.memoryTypeIndex = type;
	t0 = NowMs();
	r  = pAllocateMemory(c.dev, &ai, nullptr, &out.mem);
	out.import_ms = NowMs() - t0;
	out.alloc_result = r;
	if (r != VK_SUCCESS) return out;
	r = pBindBufferMemory(c.dev, out.buf, out.mem, 0);
	if (r != VK_SUCCESS) {
		out.alloc_result = r;
		return out;
	}
	if (want_bda) {
		VkBufferDeviceAddressInfo di {VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
		di.buffer = out.buf;
		out.addr  = pGetBufferDeviceAddress(c.dev, &di);
	}
	return out;
}

static void Release(const Ctx& c, Imported& i) {
	if (i.buf) pDestroyBuffer(c.dev, i.buf, nullptr);
	if (i.mem) pFreeMemory(c.dev, i.mem, nullptr);
	i = {};
}

// Mirror of uma_probe.comp.
static void Expect(uint32_t i, uint32_t quarter_words, uint32_t mult, uint32_t seed, uint64_t& word, uint32_t& value) {
	uint32_t q = i & 3u, t = ((i >> 2) * mult) & (quarter_words - 1u);
	word  = static_cast<uint64_t>(q) * quarter_words + t;
	value = (i * 2654435761u) ^ seed;
}

static uint64_t Verify(const uint32_t* words, uint32_t n, uint32_t quarter_words, uint32_t mult, uint32_t seed) {
	uint64_t bad = 0;
	for (uint32_t i = 0; i < n; i++) {
		uint64_t w;
		uint32_t v;
		Expect(i, quarter_words, mult, seed, w, v);
		if (words[w] != v) bad++;
	}
	return bad;
}

int main(int argc, char** argv) {
	std::string dylib   = "libMoltenVK.dylib";
	size_t      size_mb = 1024;
	bool        shm     = true, mutant = false, canary = true, import_only = false;
	for (int i = 1; i < argc; i++) {
		std::string a = argv[i];
		if (a == "--dylib" && i + 1 < argc) dylib = argv[++i];
		else if (a == "--size-mb" && i + 1 < argc) size_mb = strtoull(argv[++i], nullptr, 10);
		else if (a == "--backing" && i + 1 < argc) shm = std::string(argv[++i]) != "anon";
		else if (a == "--mutant") mutant = true;
		else if (a == "--no-canary") canary = false;
		else if (a == "--import-only") import_only = true;
		else {
			printf("usage: uma_probe --dylib <libMoltenVK.dylib> [--size-mb N(power of two)] [--backing shm|anon] [--mutant] [--no-canary] [--import-only]\n");
			return 2;
		}
	}
	if (size_mb == 0 || (size_mb & (size_mb - 1)) != 0) {
		printf("--size-mb must be a power of two\n");
		return 2;
	}
	const size_t size = size_mb << 20;
	int          fails = 0;
	auto         verdict = [&](const char* name, bool ok) {
        printf("RESULT %-44s %s\n", name, ok ? "PASS" : "FAIL");
        if (!ok) fails++;
	};
	printf("uma_probe arch=%s size=%zu MB backing=%s host_page=%d\n",
#if defined(__x86_64__)
	       "x86_64",
#else
	       "arm64",
#endif
	       size_mb, shm ? "shm(MAP_SHARED,2 views)" : "anon", static_cast<int>(vm_page_size));

	void* lib = dlopen(dylib.c_str(), RTLD_NOW | RTLD_LOCAL);
	if (!lib) {
		printf("FATAL dlopen(%s): %s\n", dylib.c_str(), dlerror());
		return 2;
	}
	pGetInstanceProcAddr = reinterpret_cast<PFN_vkGetInstanceProcAddr>(dlsym(lib, "vkGetInstanceProcAddr"));
	if (!pGetInstanceProcAddr) {
		printf("FATAL no vkGetInstanceProcAddr\n");
		return 2;
	}
	pCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(pGetInstanceProcAddr(nullptr, "vkCreateInstance"));

	// MoltenVK is the ICD itself: no portability enumeration
	const char*         inst_ext[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
	VkApplicationInfo   app {VK_STRUCTURE_TYPE_APPLICATION_INFO};
	app.apiVersion = VK_API_VERSION_1_2;
	VkInstanceCreateInfo ici {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		ici.pApplicationInfo        = &app;
	ici.enabledExtensionCount   = 1;
	ici.ppEnabledExtensionNames = inst_ext;
	VkInstance inst;
	CHECK(pCreateInstance(&ici, nullptr, &inst));
#define IL(n) p##n = reinterpret_cast<PFN_vk##n>(pGetInstanceProcAddr(inst, "vk" #n))
	IL(EnumeratePhysicalDevices);
	IL(GetPhysicalDeviceProperties2);
	IL(GetPhysicalDeviceMemoryProperties);
	IL(EnumerateDeviceExtensionProperties);
	IL(GetPhysicalDeviceQueueFamilyProperties);
	IL(CreateDevice);
	IL(GetDeviceProcAddr);
#undef IL
	Ctx      c;
	uint32_t n = 1;
	VkResult er = pEnumeratePhysicalDevices(inst, &n, &c.phys);
	if (n == 0 || (er != VK_SUCCESS && er != VK_INCOMPLETE)) {
		printf("FATAL no physical device\n");
		return 2;
	}
	VkPhysicalDeviceExternalMemoryHostPropertiesEXT emh {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
	VkPhysicalDeviceMaintenance3Properties m3 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
	emh.pNext = &m3;
	VkPhysicalDeviceProperties2 props {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
	props.pNext = &emh;
	pGetPhysicalDeviceProperties2(c.phys, &props);
	printf("GPU: %s  api=%u.%u\n", props.properties.deviceName, VK_VERSION_MAJOR(props.properties.apiVersion),
	       VK_VERSION_MINOR(props.properties.apiVersion));
	printf("minImportedHostPointerAlignment = %" PRIu64 "\n", static_cast<uint64_t>(emh.minImportedHostPointerAlignment));
	printf("maxMemoryAllocationSize = %" PRIu64 " MB (guest backing in main3 = 13824 MB)\n", static_cast<uint64_t>(m3.maxMemoryAllocationSize >> 20));
	const size_t align = emh.minImportedHostPointerAlignment ? emh.minImportedHostPointerAlignment : 16384;
	pGetPhysicalDeviceMemoryProperties(c.phys, &c.memprops);

	uint32_t ne = 0;
	pEnumerateDeviceExtensionProperties(c.phys, nullptr, &ne, nullptr);
	std::vector<VkExtensionProperties> exts(ne);
	pEnumerateDeviceExtensionProperties(c.phys, nullptr, &ne, exts.data());
	bool has_emh = false, has_subset = false;
	for (auto& e: exts) {
		if (!strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME)) has_emh = true;
		if (!strcmp(e.extensionName, "VK_KHR_portability_subset")) has_subset = true;
	}
	verdict("device exposes VK_EXT_external_memory_host", has_emh);
	if (!has_emh) return 1;

	uint32_t nq = 0;
	pGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, nullptr);
	std::vector<VkQueueFamilyProperties> qf(nq);
	pGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, qf.data());
	for (uint32_t i = 0; i < nq; i++)
		if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			c.queue_family = i;
			break;
		}
	float                   prio = 1.0f;
	VkDeviceQueueCreateInfo qci {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = c.queue_family;
	qci.queueCount       = 1;
	qci.pQueuePriorities = &prio;
	std::vector<const char*> dext = {VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
	if (has_subset) dext.push_back("VK_KHR_portability_subset");
	VkPhysicalDeviceVulkan12Features f12 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
	f12.bufferDeviceAddress = VK_TRUE;
	VkPhysicalDeviceFeatures2 f2 {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
	f2.pNext                  = &f12;
	f2.features.shaderInt64   = VK_TRUE;
	VkDeviceCreateInfo dci {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
	dci.pNext                   = &f2;
	dci.queueCreateInfoCount    = 1;
	dci.pQueueCreateInfos       = &qci;
	dci.enabledExtensionCount   = static_cast<uint32_t>(dext.size());
	dci.ppEnabledExtensionNames = dext.data();
	CHECK(pCreateDevice(c.phys, &dci, nullptr, &c.dev));
#define X(nm) p##nm = reinterpret_cast<PFN_vk##nm>(pGetDeviceProcAddr(c.dev, "vk" #nm));
	DEV_FUNCS(X)
#undef X
	pGetDeviceQueue(c.dev, c.queue_family, 0, &c.queue);

	// ---- pipeline -------------------------------------------------------------------------------------------
	VkShaderModuleCreateInfo smi {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
	smi.codeSize = uma_probe_spv_len;
	smi.pCode    = reinterpret_cast<const uint32_t*>(uma_probe_spv);
	VkShaderModule sm;
	CHECK(pCreateShaderModule(c.dev, &smi, nullptr, &sm));
	struct PC {
		uint64_t big, hdr;
		uint32_t quarter_words, mult;
	};
	VkPushConstantRange pcr {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PC)};
	VkPipelineLayoutCreateInfo pli {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
	pli.pushConstantRangeCount = 1;
	pli.pPushConstantRanges    = &pcr;
	VkPipelineLayout pl;
	CHECK(pCreatePipelineLayout(c.dev, &pli, nullptr, &pl));
	VkComputePipelineCreateInfo cpi {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
	cpi.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	cpi.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
	cpi.stage.module = sm;
	cpi.stage.pName  = "main";
	cpi.layout       = pl;
	VkPipeline pipe;
	CHECK(pCreateComputePipelines(c.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &pipe));
	VkCommandPoolCreateInfo cpci {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	cpci.queueFamilyIndex = c.queue_family;
	cpci.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	VkCommandPool pool;
	CHECK(pCreateCommandPool(c.dev, &cpci, nullptr, &pool));
	VkCommandBufferAllocateInfo cbai {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cbai.commandPool        = pool;
	cbai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cbai.commandBufferCount = 1;
	VkCommandBuffer cb;
	CHECK(pAllocateCommandBuffers(c.dev, &cbai, &cb));
	VkFenceCreateInfo fci {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
	VkFence           fence;
	CHECK(pCreateFence(c.dev, &fci, nullptr, &fence));

	// ---- 1. CANARY: a misaligned pointer must be a hard import error or still alias (never a silent copy) -----
	if (canary) {
		GuestMem tmp, th;
		if (MakeGuestMem(tmp, align * 4, shm, align) && MakeGuestMem(th, align * 4, shm, align)) {
			struct Case {
				const char* name;
				size_t      off, size;
			} cases[] = {{"+0 aligned (control)", 0, align * 2}, {"+4096 (4 KB aligned only)", 4096, align * 2},
			             {"+64 (cache-line aligned)", 64, align * 2}};
			for (auto& cs: cases) {
				uint8_t* bp = tmp.a + cs.off;
				uint8_t* hp = th.a + cs.off;
				Imported ibig = Import(c, bp, cs.size), ihdr = Import(c, hp, cs.size);
				if (ibig.alloc_result != VK_SUCCESS || ihdr.alloc_result != VK_SUCCESS) {
					printf("CANARY %-28s import REJECTED (VkResult=%d/%d)\n", cs.name, static_cast<int>(ibig.alloc_result),
					       static_cast<int>(ihdr.alloc_result));
					verdict((std::string("canary ") + cs.name + ": rejected or aliasing").c_str(), cs.off != 0);
				} else {
					const uint32_t cn = static_cast<uint32_t>(cs.size / 4 / 4) * 4; // bijective for this quarter size
					const uint32_t cq = static_cast<uint32_t>(cs.size / 4 / 4);
					const uint32_t cs_seed = 0xC0DE0000u + static_cast<uint32_t>(cs.off);
					auto*          hw      = reinterpret_cast<uint32_t*>(hp);
					hw[0] = cs_seed; // after import
					hw[1] = 0;
					PC       cpc {ibig.addr, ihdr.addr, cq, 2654435769u | 1u};
					VkCommandBufferBeginInfo cbg {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
					CHECK(pBeginCommandBuffer(cb, &cbg));
					pCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
					pCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(cpc), &cpc);
					pCmdDispatch(cb, (cn + 255) / 256, 1, 1);
					CHECK(pEndCommandBuffer(cb));
					VkSubmitInfo csi {VK_STRUCTURE_TYPE_SUBMIT_INFO};
					csi.commandBufferCount = 1;
					csi.pCommandBuffers    = &cb;
					CHECK(pResetFences(c.dev, 1, &fence));
					CHECK(pQueueSubmit(c.queue, 1, &csi, fence));
					CHECK(pWaitForFences(c.dev, 1, &fence, VK_TRUE, 60ull * 1000000000ull));
					// invocations beyond cn/.. are masked by the bijection; only i < cn is verified (cn == 4*quarter words)
					uint64_t bad = Verify(reinterpret_cast<uint32_t*>(bp), cn, cq, 2654435769u | 1u, cs_seed);
					bool     wb  = hw[1] == 0xC0FFEEu;
					printf("CANARY %-28s import ACCEPTED: gpu->cpu bad=%" PRIu64 "/%u  cpu->gpu+writeback %s => %s\n", cs.name, bad, cn,
					       wb ? "ok" : "MISSING", (bad == 0 && wb) ? "aliases (no copy)" : "SILENT COPY / incoherent");
					verdict((std::string("canary ") + cs.name + ": rejected or aliasing").c_str(), bad == 0 && wb);
				}
				Release(c, ibig);
				Release(c, ihdr);
			}
			Imported sz = Import(c, tmp.a, align + 4096, false); // size not a multiple of the page
			printf("CANARY unaligned size (%zu): VkResult=%d (info only)\n", align + 4096, static_cast<int>(sz.alloc_result));
			Release(c, sz);
		}
	}

	// ---- 2. main import ------------------------------------------------------------------------------------
	GuestMem big, hdr;
	if (!MakeGuestMem(big, size, shm, align) || !MakeGuestMem(hdr, align, shm, align)) {
		printf("FATAL cannot create %zu MB guest-like mapping\n", size_mb);
		return 2;
	}
	printf("guest views: big a=%p b=%p  hdr a=%p b=%p\n", static_cast<void*>(big.a), static_cast<void*>(big.b),
	       static_cast<void*>(hdr.a), static_cast<void*>(hdr.b));
	PrintMem("before import");
	Imported ib = Import(c, big.a, size);
	Imported ih = Import(c, hdr.a, align);
	printf("IMPORT big: VkResult=%d vkAllocateMemory=%.3f ms (create+reqs %.3f ms) address=0x%" PRIx64 "\n",
	       static_cast<int>(ib.alloc_result), ib.import_ms, ib.create_ms, static_cast<uint64_t>(ib.addr));
	printf("IMPORT hdr: VkResult=%d vkAllocateMemory=%.3f ms address=0x%" PRIx64 "\n", static_cast<int>(ih.alloc_result),
	       ih.import_ms, static_cast<uint64_t>(ih.addr));
	verdict("main 1 GB-class import succeeds", ib.alloc_result == VK_SUCCESS && ih.alloc_result == VK_SUCCESS);
	if (ib.alloc_result != VK_SUCCESS || ih.alloc_result != VK_SUCCESS) return 1;
	verdict("buffer device addresses are non-zero", ib.addr != 0 && ih.addr != 0);
	PrintMem("after import (untouched)");

	if (import_only) { // import + residency numbers only (no page is touched): safe for sizes near the whole backing
		Release(c, ib);
		Release(c, ih);
		PrintMem("after free (import-only)");
		printf("SUMMARY fails=%d (import-only)\n", fails);
		return fails ? 1 : 0;
	}

	// reference: a normal driver allocation of the same size (time + wiring) for comparison
	{
		VkBufferCreateInfo bi {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		bi.size  = size;
		bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		VkBuffer rb;
		CHECK(pCreateBuffer(c.dev, &bi, nullptr, &rb));
		VkMemoryRequirements rq;
		pGetBufferMemoryRequirements(c.dev, rb, &rq);
		VkMemoryAllocateInfo ai {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		ai.allocationSize  = rq.size;
		ai.memoryTypeIndex = PickType(c, rq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
		VkDeviceMemory rm;
		double         t0 = NowMs();
		VkResult       rr = pAllocateMemory(c.dev, &ai, nullptr, &rm);
		printf("REFERENCE vkAllocateMemory(%zu MB host-visible, driver-owned) = %.3f ms VkResult=%d\n", size_mb, NowMs() - t0,
		       static_cast<int>(rr));
		PrintMem("after reference alloc");
		if (rr == VK_SUCCESS) pFreeMemory(c.dev, rm, nullptr);
		pDestroyBuffer(c.dev, rb, nullptr);
	}

	// ---- 3. CPU stores AFTER import, GPU dispatch, CPU reads WITHOUT copy -----------------------------------
	const uint32_t quarter_words = static_cast<uint32_t>(size / 4 / 4);
	const uint32_t mult          = 2654435769u | 1u; // odd
	const uint32_t N             = 1u << 20;
	const uint32_t seed          = 0x5EED0000u ^ static_cast<uint32_t>(getpid());
	auto*          h_a           = reinterpret_cast<uint32_t*>(hdr.a);
	auto*          h_b           = reinterpret_cast<uint32_t*>(hdr.b);
	auto*          w_b           = reinterpret_cast<uint32_t*>(big.b);
	h_a[0]                       = seed; // written through view A after the import; GPU must see it (no copy)
	h_a[1]                       = 0;

	// positive control for the checker: nothing dispatched yet -> verification MUST report (almost) all words bad.
	uint64_t bad_pre = Verify(w_b, N, quarter_words, mult, seed);
	printf("CONTROL verify-before-dispatch: %" PRIu64 " / %u bad\n", bad_pre, N);
	verdict("checker positive control (pre-dispatch bad > 99%)", bad_pre > static_cast<uint64_t>(N) * 99 / 100);

	PC pc {ib.addr, ih.addr, quarter_words, mult};
	VkCommandBufferBeginInfo bgi {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	CHECK(pBeginCommandBuffer(cb, &bgi));
	pCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
	pCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
	pCmdDispatch(cb, N / 256, 1, 1);
	CHECK(pEndCommandBuffer(cb));
	VkSubmitInfo si {VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers    = &cb;
	double t0             = NowMs();
	CHECK(pResetFences(c.dev, 1, &fence)); // the canary section leaves it signaled
	CHECK(pQueueSubmit(c.queue, 1, &si, fence));
	if (mutant) {
		// MUTANT: CPU reads before the fence. Mismatches are possible (race); the number is informational.
		uint64_t bad_race = Verify(w_b, N, quarter_words, mult, seed);
		printf("MUTANT read-before-fence: %" PRIu64 " / %u bad (race: any value is legal)\n", bad_race, N);
	}
	CHECK(pWaitForFences(c.dev, 1, &fence, VK_TRUE, 60ull * 1000000000ull));
	double gpu_ms = NowMs() - t0;
	printf("DISPATCH %u invocations submit->fence %.3f ms\n", N, gpu_ms);
	PrintMem("after dispatch");

	double   v0     = NowMs();
	uint64_t bad_b  = Verify(w_b, N, quarter_words, mult, seed);
	double   v_ms   = NowMs() - v0;
	uint64_t bad_a  = Verify(reinterpret_cast<uint32_t*>(big.a), N, quarter_words, mult, seed);
	printf("VERIFY via view B (independent alias): %" PRIu64 " / %u bad (%.1f ms);  via imported view A: %" PRIu64 " bad\n", bad_b, N,
	       v_ms, bad_a);
	verdict("GPU writes visible to CPU without copy (view B)", bad_b == 0);
	verdict("GPU writes visible through the imported view A", bad_a == 0);
	verdict("CPU stores after import visible to GPU (seed, no copy)", bad_b == 0 /* values embed the seed */);
	verdict("GPU->CPU header write-back (2nd import)", h_b[1] == 0xC0FFEEu);

	// second dispatch with a changed seed: proves the mapping stays live (not a one-shot snapshot)
	h_b[0] = seed ^ 0xFFFFFFFFu; // CPU writes through view B this time
	CHECK(pResetFences(c.dev, 1, &fence));
	CHECK(pQueueSubmit(c.queue, 1, &si, fence));
	CHECK(pWaitForFences(c.dev, 1, &fence, VK_TRUE, 60ull * 1000000000ull));
	uint64_t bad2 = Verify(w_b, N, quarter_words, mult, seed ^ 0xFFFFFFFFu);
	printf("VERIFY second dispatch (seed rewritten via view B): %" PRIu64 " bad\n", bad2);
	verdict("second dispatch sees the new seed (live aliasing)", bad2 == 0);

	// ---- 4. teardown, wiring after free ---------------------------------------------------------------------
	pDeviceWaitIdle(c.dev);
	Release(c, ib);
	Release(c, ih);
	PrintMem("after free (mappings still live)");
	munmap(big.a, size);
	if (big.b != big.a) munmap(big.b, size);
	PrintMem("after munmap");
	printf("SUMMARY fails=%d\n", fails);
	return fails ? 1 : 0;
}
