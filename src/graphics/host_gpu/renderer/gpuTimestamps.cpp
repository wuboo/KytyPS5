#include "graphics/host_gpu/renderer/gpuTimestamps.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <vk_mem_alloc.h>

#include "common/assert.h"
#include "common/diagOut.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/gpuProbeSpv.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/presentation/frameTiming.h"

namespace Libs::Graphics::GpuTs {

namespace {

template <typename... A>
void DiagPrintf(const char* fmt, A... args) {
	char line[700];
	std::snprintf(line, sizeof(line), fmt, args...);
	Common::DiagOut::Write(line);
}

struct Pending {
	int32_t  slot;
	uint64_t tick;
	uint8_t  reason;
	uint32_t debug_op;
	BufStats stats;
};

struct Agg {
	uint64_t n = 0, ns = 0, draws = 0;
};

struct State {
	std::mutex                    mutex;
	bool                          on = false, inited = false, ready = false, broken = false;
	vk::QueryPool                 pool = nullptr;
	double                        period = 1.0;
	uint64_t                      mask = ~0ull;
	std::array<bool, kSlots>      busy {};
	uint32_t                      next = 0;
	std::deque<Pending>           pending;
	Acc                           acc;
	std::array<uint64_t, 6>       hist_n {}, hist_ns {};
	uint64_t                      window_n = 0, window_busy = 0, window_idle = 0;
	uint64_t                      window_begin = 0;
	std::vector<uint64_t>         bounds;
	size_t                        next_bound = 0;
	uint32_t                      poll_tick = 0, raw_logged = 0;
	std::array<uint64_t, 12>      reason_n {}, reason_ns {}, long_n {}, long_ns {}; // by GpuDone::SubmitReason (S477 D3b): all submits / submits over 3 ms
	// S478: per pass shape / per long submit shape, the clock probe, the part of a submit that is no render pass
	std::map<std::string, Agg>    pass_agg, long_agg;
	uint64_t                      rem_ns = 0, rem_n = 0;
	std::vector<uint32_t>         probe_ns;
	uint64_t                      probe_sum = 0;
	uint32_t                      probe_iters = 0, probe_every = 2, probe_ctr = 0;
	bool                          probe_ok = false;
	uint32_t                      fill_iters = 0;   // S478 H-ZEGAR (c): KYTY_GPU_FILL=N, a thread submits an N-step ALU chain whenever the GPU has no guest work
	uint32_t                      fill_pause_us = 0; // KYTY_GPU_FILL_PAUSE_US: pause between two fillers
	bool                          fill_started = false;
	vk::DescriptorSetLayout       probe_set_layout = nullptr;
	vk::PipelineLayout            probe_layout = nullptr;
	vk::Pipeline                  probe_pipeline = nullptr;
	VkBuffer                      probe_buffer = nullptr;
	VmaAllocation                 probe_alloc = nullptr;
};

State& S() {
	static State s;
	return s;
}

void InitEnv(State& s) {
	s.inited      = true;
	const char* v = std::getenv("KYTY_GPU_TIMESTAMPS");
	s.on          = v != nullptr && v[0] == '1' && v[1] == '\0';
	if (const char* p = std::getenv("KYTY_GPU_PROBE"); p != nullptr && p[0] != '\0') s.probe_iters = static_cast<uint32_t>(std::strtoul(p, nullptr, 10));
	if (const char* p = std::getenv("KYTY_GPU_FILL"); p != nullptr && p[0] != '\0') s.fill_iters = static_cast<uint32_t>(std::strtoul(p, nullptr, 10));
	if (const char* p = std::getenv("KYTY_GPU_FILL_PAUSE_US"); p != nullptr && p[0] != '\0') s.fill_pause_us = static_cast<uint32_t>(std::strtoul(p, nullptr, 10));
	if (const char* p = std::getenv("KYTY_GPU_PROBE_EVERY"); p != nullptr && p[0] != '\0') s.probe_every = std::max(1u, static_cast<uint32_t>(std::strtoul(p, nullptr, 10)));
	const char* at = std::getenv("KYTY_GPU_TIMESTAMPS_AT");
	const char* list = at != nullptr && at[0] != '\0' ? at : "6300,6800,6990";
	char*       end  = nullptr;
	for (const char* p = list; *p != '\0';) {
		const auto b = std::strtoull(p, &end, 10);
		if (end == p) break;
		s.bounds.push_back(b);
		p = (*end == ',') ? end + 1 : end;
	}
}

void CreateProbe(State& s, GraphicContext& graphics) {
	const vk::DescriptorSetLayoutBinding binding {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr};
	vk::DescriptorSetLayoutCreateInfo    set_info {};
	set_info.flags        = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
	set_info.bindingCount = 1;
	set_info.pBindings    = &binding;
	if (graphics.device.createDescriptorSetLayout(&set_info, nullptr, &s.probe_set_layout) != vk::Result::eSuccess) return;
	const vk::PushConstantRange  range {vk::ShaderStageFlagBits::eCompute, 0, 4};
	vk::PipelineLayoutCreateInfo layout_info {};
	layout_info.setLayoutCount         = 1;
	layout_info.pSetLayouts            = &s.probe_set_layout;
	layout_info.pushConstantRangeCount = 1;
	layout_info.pPushConstantRanges    = &range;
	if (graphics.device.createPipelineLayout(&layout_info, nullptr, &s.probe_layout) != vk::Result::eSuccess) return;
	const auto                        module = CompileSPV(kProbeSpv, graphics.device);
	vk::ComputePipelineCreateInfo     pipe {};
	pipe.stage.stage  = vk::ShaderStageFlagBits::eCompute;
	pipe.stage.module = module;
	pipe.stage.pName  = "main";
	pipe.layout       = s.probe_layout;
	const auto result = graphics.device.createComputePipelines(nullptr, 1, &pipe, nullptr, &s.probe_pipeline);
	graphics.device.destroyShaderModule(module, nullptr);
	if (result != vk::Result::eSuccess) return;
	vk::BufferCreateInfo buffer {};
	buffer.size  = 4096;
	buffer.usage = vk::BufferUsageFlagBits::eStorageBuffer;
	VmaAllocationCreateInfo alloc {};
	alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
	if (vmaCreateBuffer(graphics.allocator, reinterpret_cast<const VkBufferCreateInfo*>(&buffer), &alloc, &s.probe_buffer, &s.probe_alloc, nullptr) != VK_SUCCESS) return;
	s.probe_ok = true;
	DiagPrintf("[gpu-ts] clock probe on: %u iterations, every %u-th buffer\n", s.probe_iters, s.probe_every);
}

void EnsurePool(State& s, GraphicContext& graphics) {
	if (s.ready || s.broken) return;
	const auto family = graphics.physical_device.getQueueFamilyProperties();
	const uint32_t bits = graphics.queue_family < family.size() ? family[graphics.queue_family].timestampValidBits : 0;
	if (bits == 0) {
		DiagPrintf("[gpu-ts] queue family has no timestamp support: disabled\n");
		s.broken = true;
		return;
	}
	s.mask   = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
	s.period = static_cast<double>(graphics.physical_device_properties.limits.timestampPeriod);
	vk::QueryPoolCreateInfo info {};
	info.queryType  = vk::QueryType::eTimestamp;
	info.queryCount = kSlots * kQueriesPerSlot;
	auto result     = graphics.device.createQueryPool(&info, nullptr, &s.pool);
	if (result != vk::Result::eSuccess) {
		DiagPrintf("[gpu-ts] createQueryPool failed: disabled\n");
		s.broken = true;
		return;
	}
	if (s.probe_iters != 0 || s.fill_iters != 0) CreateProbe(s, graphics);
	s.ready = true;
	DiagPrintf("[gpu-ts] enabled: period=%.3f ns valid_bits=%u slots=%u\n", s.period, bits, kSlots);
}

void DumpDetail(State& s, double fl);

void Dump(State& s, uint64_t flip) {
	static const char* const names[] = {"<100us", "<300us", "<1ms", "<3ms", "<10ms", ">=10ms"};
	const double             fl      = static_cast<double>(flip > s.window_begin ? flip - s.window_begin : 1);
	char                     line[600];
	int                      len = std::snprintf(line, sizeof(line), "[gpu-ts] window flips %llu-%llu submits/flip=%.2f busy_ms/flip=%.3f idle_ms/flip=%.3f hist(n/flip|ms/flip):",
	                                             static_cast<unsigned long long>(s.window_begin), static_cast<unsigned long long>(flip), static_cast<double>(s.window_n) / fl,
	                                             static_cast<double>(s.window_busy) / fl / 1e6, static_cast<double>(s.window_idle) / fl / 1e6);
	for (size_t i = 0; i < 6 && len > 0 && static_cast<size_t>(len) < sizeof(line); i++) {
		len += std::snprintf(line + len, sizeof(line) - static_cast<size_t>(len), " %s=%.2f|%.3f", names[i], static_cast<double>(s.hist_n[i]) / fl,
		                     static_cast<double>(s.hist_ns[i]) / fl / 1e6);
	}
	DiagPrintf("%s\n", line);
	static const char* const rn[] = {"Drain", "LazyMax", "Lazy1ms", "LazyIdle", "Eager", "EagerPass", "EagerMid", "Eager3Mid", "Eager3Flush", "LowWater", "Flip", "Other"};
	int rl = std::snprintf(line, sizeof(line), "[gpu-ts] by submit reason (all n/flip|ms/flip ; over 3 ms n/flip|ms/flip):");
	for (size_t i = 0; i < 12 && rl > 0 && static_cast<size_t>(rl) < sizeof(line); i++) {
		if (s.reason_n[i] == 0) continue;
		rl += std::snprintf(line + rl, sizeof(line) - static_cast<size_t>(rl), " %s=%.2f|%.2f;%.2f|%.2f", rn[i], static_cast<double>(s.reason_n[i]) / fl,
		                    static_cast<double>(s.reason_ns[i]) / fl / 1e6, static_cast<double>(s.long_n[i]) / fl, static_cast<double>(s.long_ns[i]) / fl / 1e6);
	}
	DiagPrintf("%s\n", line);
	DumpDetail(s, fl);
	s.reason_n.fill(0);
	s.reason_ns.fill(0);
	s.long_n.fill(0);
	s.long_ns.fill(0);
	s.hist_n.fill(0);
	s.hist_ns.fill(0);
	s.window_n = s.window_busy = s.window_idle = 0;
	s.window_begin = flip;
}

std::atomic<uint64_t> g_fill_n {0};

// S478 H-ZEGAR (c): while the guest has nothing queued (every tick handed out is done) keep the GPU busy with a short ALU chain, one filler at a time.
void FillerLoop(GraphicContext* graphics, MasterSemaphore* master, vk::PipelineLayout layout, vk::Pipeline pipeline, VkBuffer buffer, uint32_t iters, uint32_t pause_us) {
	vk::CommandPoolCreateInfo pool_info {};
	pool_info.queueFamilyIndex = graphics->queue_family;
	vk::CommandPool pool = nullptr;
	if (graphics->device.createCommandPool(&pool_info, nullptr, &pool) != vk::Result::eSuccess) return;
	vk::CommandBufferAllocateInfo alloc {};
	alloc.commandPool        = pool;
	alloc.level              = vk::CommandBufferLevel::ePrimary;
	alloc.commandBufferCount = 1;
	vk::CommandBuffer command = nullptr;
	if (graphics->device.allocateCommandBuffers(&alloc, &command) != vk::Result::eSuccess) return;
	vk::CommandBufferBeginInfo begin {};
	command.begin(&begin);
	command.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
	const vk::DescriptorBufferInfo info {buffer, 0, 4096};
	vk::WriteDescriptorSet         write {};
	write.dstBinding      = 0;
	write.descriptorCount = 1;
	write.descriptorType  = vk::DescriptorType::eStorageBuffer;
	write.pBufferInfo     = &info;
	command.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, layout, 0, write);
	command.pushConstants(layout, vk::ShaderStageFlagBits::eCompute, 0, 4, &iters);
	command.dispatch(1, 1, 1);
	command.end();
	vk::FenceCreateInfo fence_info {};
	vk::Fence           fence = nullptr;
	if (graphics->device.createFence(&fence_info, nullptr, &fence) != vk::Result::eSuccess) return;
	bool in_flight = false;
	for (;;) {
		if (in_flight) {
			if (graphics->device.getFenceStatus(fence) != vk::Result::eSuccess) {
				std::this_thread::yield();
				continue;
			}
			in_flight = false;
			(void)graphics->device.resetFences(1, &fence);
			if (pause_us != 0) std::this_thread::sleep_for(std::chrono::microseconds(pause_us));
		}
		master->Refresh();
		if (!master->IsFree(master->CurrentTick() - 1)) {
			std::this_thread::yield();
			continue;
		}
		vk::SubmitInfo submit {};
		submit.commandBufferCount = 1;
		submit.pCommandBuffers    = &command;
		{
			Common::LockGuard lock(graphics->queue_mutex);
			if (graphics->queue.submit(1, &submit, fence) != vk::Result::eSuccess) return;
		}
		in_flight = true;
		g_fill_n.fetch_add(1, std::memory_order_relaxed);
	}
}

const char* ReasonName(uint8_t r) {
	static const char* const rn[] = {"Drain", "LazyMax", "Lazy1ms", "LazyIdle", "Eager", "EagerPass", "EagerMid", "Eager3Mid", "Eager3Flush", "LowWater", "Flip", "Other"};
	return rn[r < 12 ? r : 11];
}

std::string PassKey(const PassRec& r) {
	char t[48];
	std::snprintf(t, sizeof(t), "%ux%u c%u%s%s", r.w, r.h, r.ncol, (r.flags & 1) ? " d" : "", (r.flags & 2) ? " clr" : "");
	return t;
}

// S478: the passes and the clock probe of one finished buffer (their queries are written once the buffer is done); `sum_ns` = end - start of the whole buffer
void Detail(State& s, GraphicContext& graphics, const Pending& p, uint64_t sum_ns) {
	const uint32_t base = static_cast<uint32_t>(p.slot) * kQueriesPerSlot;
	uint64_t       probe_ns = 0, pass_sum = 0;
	if (p.stats.probed) {
		uint64_t   d[2] = {};
		if (graphics.device.getQueryPoolResults(s.pool, base + 2, 2, sizeof(d), d, sizeof(uint64_t), vk::QueryResultFlagBits::e64) == vk::Result::eSuccess) {
			const uint64_t a = d[0] & s.mask, b = d[1] & s.mask;
			if (b >= a) {
				probe_ns = static_cast<uint64_t>(static_cast<double>(b - a) * s.period);
				if (s.probe_ns.size() < 200000) s.probe_ns.push_back(static_cast<uint32_t>(probe_ns));
				s.probe_sum += probe_ns;
			}
		}
	}
	const uint32_t np = std::min(p.stats.npass, kMaxPass);
	std::string    first;
	for (uint32_t i = 0; i < np; i++) {
		uint64_t d[2] = {};
		if (graphics.device.getQueryPoolResults(s.pool, base + 4 + 2 * i, 2, sizeof(d), d, sizeof(uint64_t), vk::QueryResultFlagBits::e64) != vk::Result::eSuccess) continue;
		const uint64_t a = d[0] & s.mask, b = d[1] & s.mask;
		if (b < a) continue;
		const auto     ns    = static_cast<uint64_t>(static_cast<double>(b - a) * s.period);
		const uint32_t next  = i + 1 < np ? p.stats.pass[i + 1].draw0 : p.stats.draws;
		auto&          agg   = s.pass_agg[PassKey(p.stats.pass[i])];
		agg.n++;
		agg.ns += ns;
		agg.draws += next - p.stats.pass[i].draw0;
		pass_sum += ns;
		if (i == 0) first = PassKey(p.stats.pass[i]);
	}
	if (sum_ns > probe_ns + pass_sum) {
		s.rem_ns += sum_ns - probe_ns - pass_sum;
		s.rem_n++;
	}
	if (sum_ns > 3000000) {
		char key[160];
		const uint32_t dr = p.stats.draws;
		std::snprintf(key, sizeof(key), "%s op=%u passes=%u draws=%s disp=%u first=%s", ReasonName(p.reason), p.debug_op, p.stats.npass,
		              dr == 0 ? "0" : dr < 10 ? "1-9" : dr < 100 ? "10-99" : "100+", p.stats.disp, first.c_str());
		auto& agg = s.long_agg[key];
		agg.n++;
		agg.ns += sum_ns - probe_ns;
		agg.draws += dr;
	}
}

void DumpDetail(State& s, double fl) {
	char line[700];
	auto top = [&](std::map<std::string, Agg>& m, const char* tag, size_t limit) {
		std::vector<std::pair<std::string, Agg>> v(m.begin(), m.end());
		std::sort(v.begin(), v.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		for (size_t i = 0; i < v.size() && i < limit; i++) {
			const auto& a = v[i].second;
			std::snprintf(line, sizeof(line), "[gpu-ts-%s] %-62s n/flip=%.2f ms/flip=%.3f us_each=%.1f draws_each=%.1f", tag, v[i].first.c_str(), static_cast<double>(a.n) / fl,
			              static_cast<double>(a.ns) / fl / 1e6, static_cast<double>(a.ns) / static_cast<double>(a.n) / 1e3, static_cast<double>(a.draws) / static_cast<double>(a.n));
			DiagPrintf("%s\n", line);
		}
		m.clear();
	};
	top(s.pass_agg, "pass", 14);
	top(s.long_agg, "long", 10);
	{
		static uint64_t last_fill = 0;
		const uint64_t  now_fill  = g_fill_n.load(std::memory_order_relaxed);
		if (now_fill != 0) {
			std::snprintf(line, sizeof(line), "[gpu-ts-fill] fillers submitted n/flip=%.2f", static_cast<double>(now_fill - last_fill) / fl);
			DiagPrintf("%s\n", line);
		}
		last_fill = now_fill;
	}
	std::snprintf(line, sizeof(line), "[gpu-ts-rem] part of buffers outside render passes (compute/copies/barriers): n/flip=%.2f ms/flip=%.3f", static_cast<double>(s.rem_n) / fl,
	              static_cast<double>(s.rem_ns) / fl / 1e6);
	DiagPrintf("%s\n", line);
	s.rem_ns = s.rem_n = 0;
	if (!s.probe_ns.empty()) {
		std::sort(s.probe_ns.begin(), s.probe_ns.end());
		const auto q = [&](double f) { return static_cast<double>(s.probe_ns[static_cast<size_t>(f * static_cast<double>(s.probe_ns.size() - 1))]) / 1e3; };
		std::snprintf(line, sizeof(line), "[gpu-ts-probe] n=%zu n/flip=%.2f ms/flip=%.3f us: min=%.1f p10=%.1f p50=%.1f p90=%.1f max=%.1f mean=%.1f", s.probe_ns.size(),
		              static_cast<double>(s.probe_ns.size()) / fl, static_cast<double>(s.probe_sum) / fl / 1e6, q(0), q(0.1), q(0.5), q(0.9), q(1.0),
		              static_cast<double>(s.probe_sum) / static_cast<double>(s.probe_ns.size()) / 1e3);
		DiagPrintf("%s\n", line);
		s.probe_ns.clear();
		s.probe_sum = 0;
	}
}

} // namespace

bool Enabled() {
	auto& s = S();
	if (!s.inited) InitEnv(s);
	return s.on && !s.broken;
}

int32_t OnBegin(GraphicContext& graphics, vk::CommandBuffer buffer, BufStats& stats) {
	stats = BufStats {};
	if (!Enabled()) return -1;
	auto&            s = S();
	std::lock_guard  lock(s.mutex);
	EnsurePool(s, graphics);
	if (!s.ready) return -1;
	const auto slot = static_cast<int32_t>(s.next % kSlots);
	if (s.busy[static_cast<size_t>(slot)]) {
		FrameTiming::Add(FrameTiming::Counter::GpuTsSkipN);
		return -1;
	}
	s.busy[static_cast<size_t>(slot)] = true;
	s.next++;
	const uint32_t base = static_cast<uint32_t>(slot) * kQueriesPerSlot;
	buffer.resetQueryPool(s.pool, base, kQueriesPerSlot);
	buffer.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, s.pool, base);
	if (s.probe_ok && s.probe_iters != 0 && (s.probe_ctr++ % s.probe_every) == 0) { // S478 H-ZEGAR: a fixed ALU chain timed at the start of the buffer (~ 1 / shader clock)
		buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.pool, base + 2);
		vk::MemoryBarrier2 barrier {};
		barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
		barrier.dstStageMask = vk::PipelineStageFlagBits2::eComputeShader;
		vk::DependencyInfo dependency {};
		dependency.memoryBarrierCount = 1;
		dependency.pMemoryBarriers    = &barrier;
		buffer.pipelineBarrier2(dependency);
		buffer.bindPipeline(vk::PipelineBindPoint::eCompute, s.probe_pipeline);
		const vk::DescriptorBufferInfo info {s.probe_buffer, 0, 4096};
		vk::WriteDescriptorSet         write {};
		write.dstBinding      = 0;
		write.descriptorCount = 1;
		write.descriptorType  = vk::DescriptorType::eStorageBuffer;
		write.pBufferInfo     = &info;
		buffer.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, s.probe_layout, 0, write);
		buffer.pushConstants(s.probe_layout, vk::ShaderStageFlagBits::eCompute, 0, 4, &s.probe_iters);
		buffer.dispatch(1, 1, 1);
		barrier.srcStageMask = vk::PipelineStageFlagBits2::eComputeShader;
		barrier.dstStageMask = vk::PipelineStageFlagBits2::eAllCommands;
		buffer.pipelineBarrier2(dependency);
		buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.pool, base + 3);
		stats.probed = true;
	}
	return slot;
}

void OnEnd(GraphicContext&, vk::CommandBuffer buffer, int32_t slot) {
	if (slot < 0) return;
	auto& s = S();
	buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, s.pool, static_cast<uint32_t>(slot) * kQueriesPerSlot + 1);
}

void OnPassBegin(vk::CommandBuffer buffer, int32_t slot, BufStats& stats, uint32_t w, uint32_t h, uint32_t ncol, uint32_t flags) {
	const uint32_t i = stats.npass++;
	if (i >= kMaxPass) return;
	stats.pass[i] = PassRec {static_cast<uint16_t>(w), static_cast<uint16_t>(h), static_cast<uint8_t>(ncol), static_cast<uint8_t>(flags), stats.draws};
	stats.pass_open = true;
	if (slot >= 0) buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, S().pool, static_cast<uint32_t>(slot) * kQueriesPerSlot + 4 + 2 * i);
}

void OnPassEnd(vk::CommandBuffer buffer, int32_t slot, BufStats& stats) {
	if (!stats.pass_open) return;
	stats.pass_open = false;
	const uint32_t i = stats.npass - 1;
	if (i >= kMaxPass) return;
	if (slot >= 0) buffer.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, S().pool, static_cast<uint32_t>(slot) * kQueriesPerSlot + 5 + 2 * i);
}

void OnSubmit(int32_t slot, uint64_t tick, uint8_t reason, const BufStats& stats, uint32_t debug_op) {
	if (slot < 0) return;
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	auto            it = s.pending.end();
	while (it != s.pending.begin() && (it - 1)->tick > tick) --it;
	s.pending.insert(it, Pending {slot, tick, reason, debug_op, stats});
}

void Poll(GraphicContext& graphics, MasterSemaphore& master) {
	if (!Enabled()) return;
	auto&           s = S();
	std::lock_guard lock(s.mutex);
	if (!s.ready) return;
	if ((s.poll_tick++ & 7u) == 0) master.Refresh();
	if (s.fill_iters != 0 && s.probe_ok && !s.fill_started) {
		s.fill_started = true;
		std::thread(FillerLoop, &graphics, &master, s.probe_layout, s.probe_pipeline, s.probe_buffer, s.fill_iters, s.fill_pause_us).detach();
		DiagPrintf("[gpu-ts] filler thread started: %u iterations, pause %u us\n", s.fill_iters, s.fill_pause_us);
	}
	while (!s.pending.empty()) {
		const auto p = s.pending.front();
		if (!master.IsFree(p.tick)) break;
		uint64_t data[2] = {};
		const auto result = graphics.device.getQueryPoolResults(s.pool, static_cast<uint32_t>(p.slot) * kQueriesPerSlot, 2, sizeof(data), data, sizeof(uint64_t),
		                                                       vk::QueryResultFlagBits::e64);
		if (result == vk::Result::eNotReady) break;
		s.pending.pop_front();
		s.busy[static_cast<size_t>(p.slot)] = false;
		if (result != vk::Result::eSuccess) continue;
		const auto     start = data[0] & s.mask, end = data[1] & s.mask;
		if (s.raw_logged < 60 || end < start || (end - start) * s.period > 1e9) { // anomalies and the first pairs, raw
			if (s.raw_logged < 400) {
				s.raw_logged++;
				DiagPrintf("[gpu-ts] raw slot=%d tick=%llu start=%llu end=%llu dur_us=%.1f gap_to_prev_end_us=%.1f\n", p.slot, static_cast<unsigned long long>(p.tick),
				           static_cast<unsigned long long>(start), static_cast<unsigned long long>(end), static_cast<double>(end - start) * s.period / 1e3,
				           s.acc.have ? (static_cast<double>(start) - static_cast<double>(s.acc.prev_end)) * s.period / 1e3 : 0.0);
			}
		}
		if (end < start || static_cast<double>(end - start) * s.period > 2e9) { // a pair that cannot be a submit: counted, not accumulated
			FrameTiming::Add(FrameTiming::Counter::GpuTsSkipN);
			continue;
		}
		const auto     before_busy = s.acc.busy_ns, before_idle = s.acc.idle_ns, before_sum = s.acc.sum_ns, before_ov = s.acc.overlap_n;
		const uint64_t busy = Accumulate(s.acc, start, end, s.period);
		using C = FrameTiming::Counter;
		FrameTiming::Add(C::GpuTsN);
		FrameTiming::Add(C::GpuTsBusyNs, s.acc.busy_ns - before_busy);
		FrameTiming::Add(C::GpuTsIdleNs, s.acc.idle_ns - before_idle);
		FrameTiming::Add(C::GpuTsSumNs, s.acc.sum_ns - before_sum);
		FrameTiming::Add(C::GpuTsOverlapN, s.acc.overlap_n - before_ov);
		const auto b = Bucket(s.acc.sum_ns - before_sum);
		s.hist_n[b]++;
		s.hist_ns[b] += busy;
		{
			const size_t r = p.reason < 12 ? p.reason : 11;
			s.reason_n[r]++;
			s.reason_ns[r] += busy;
			if (s.acc.sum_ns - before_sum > 3000000) {
				s.long_n[r]++;
				s.long_ns[r] += busy;
			}
		}
		s.window_n++;
		s.window_busy += s.acc.busy_ns - before_busy;
		s.window_idle += s.acc.idle_ns - before_idle;
		Detail(s, graphics, p, s.acc.sum_ns - before_sum);
	}
	const auto flip = FrameTiming::GuestFlips();
	if (s.next_bound < s.bounds.size() && flip >= s.bounds[s.next_bound]) {
		s.next_bound++;
		Dump(s, flip);
	}
}

void CpDelay() {
	static const uint64_t ns = [] {
		const char* v = std::getenv("KYTY_DIAG_CP_DELAY_NS");
		return v != nullptr && v[0] != '\0' ? std::strtoull(v, nullptr, 10) : 0ull;
	}();
	if (ns == 0) return;
	const int64_t t0 = FrameTiming::SteadyNowNs();
	while (static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0) < ns) {
	}
}

} // namespace Libs::Graphics::GpuTs
