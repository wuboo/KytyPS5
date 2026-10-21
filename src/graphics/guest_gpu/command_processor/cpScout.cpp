#include "graphics/guest_gpu/command_processor/cpScoutRuntime.h"

#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics::CpScout {
namespace {

struct Slot {
	HW::Shader                           sh;
	HW::UserConfig                       ucfg;
	uint32_t                             rt_mask = 0;
	uint64_t                             rt_addr[8] {};
	uint32_t                             rt_format[8] {};
	bool                                 blend0 = false;
	uint32_t                             aa     = 0;
	bool                                 clip_disable = false;
	bool                                 provoking_last = false;
	DrawSnap                             snap;
	PipelineCache::GraphicsPrograms      programs;
	std::array<ShaderVertexInputInfo, 3> vertex_info {};
	ShaderPixelInputInfo                 pixel_info {};
	std::atomic<bool>                    ready {false};
};

struct Job {
	CommandProcessor*         cp    = nullptr;
	PipelineCache*            cache = nullptr;
	std::span<const uint32_t> commands;
	HW::Context               ctx;
	HW::Shader                sh;
	HW::UserConfig            ucfg;
	HW::UserSgprType          marker = HW::UserSgprType::Unknown;
	std::atomic<bool>         pending {false};
};

struct Live {
	std::vector<std::unique_ptr<Slot>> slots;
	std::atomic<uint32_t>              ready_n {0};
	std::atomic<bool>                  done {true};
	std::atomic<bool>                  stopped {false};
	uint32_t                           cp_draw = 0;
};

std::mutex              g_mu;
std::condition_variable g_cv;
Job                     g_job;
Live                    g_live;
std::thread             g_worker;
std::atomic<bool>       g_stop {false};
bool                    g_started = false;

bool ReadGuestWord(uint64_t address, uint32_t& word) {
	if (address < 0x40000ull) return false;
	std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	return true;
}

bool PixelSideEffects(const HW::ShaderRegisters& sh_regs) {
	const auto& db = sh_regs.db_shader_control;
	return db.shader_kill_enable || db.shader_z_export_enable || db.shader_mask_export_enable ||
	       db.shader_dual_export_enable || db.shader_execute_on_noop;
}

uint32_t ColorOutputMask(const HW::Context& ctx) {
	const auto& sh_regs     = ctx.GetShaderRegisters();
	const auto  write_mask  = ctx.GetRenderTargetMask() & sh_regs.m_cbShaderMask;
	uint32_t    output_mask = ctx.HasObjectPrimitiveIdOutput() ? 1u << 7u : 0;
	for (uint32_t index = 0; index < 8; index++) {
		const auto slot = ShaderPixelExportTarget(sh_regs.m_cbShaderMask, index);
		if (slot >= 8) break;
		if (sh_regs.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			output_mask |= 1u << slot;
		}
	}
	return output_mask;
}

void FillSlotRegs(Slot& slot, const HW::Context& ctx, const HW::Shader& sh, const HW::UserConfig& ucfg) {
	slot.sh   = sh;
	slot.ucfg = ucfg;
	slot.rt_mask = ctx.GetRenderTargetMask();
	slot.blend0  = ctx.GetBlendControl(0).enable;
	slot.aa      = ctx.GetAaConfig().msaa_num_samples;
	slot.clip_disable   = ctx.GetClipControl().clip_disable;
	slot.provoking_last = ctx.GetModeControl().provoking_vtx_last;
	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt = ctx.GetRenderTarget(i);
		slot.rt_addr[i]   = rt.base.addr;
		slot.rt_format[i] = static_cast<uint32_t>(rt.info.format) |
		                    (static_cast<uint32_t>(rt.info.channel_type) << 16u);
	}
}

// Returns the counter of the first differing field, or Count when equal.
FrameTiming::Counter RegsDiff(const Slot& slot, const HW::Context& ctx, const HW::Shader& sh, const HW::UserConfig& ucfg) {
	using C = FrameTiming::Counter;
	if (sh.GetPs().ps_regs.data_addr != slot.sh.GetPs().ps_regs.data_addr) return C::ScoutRegsPsAddrN;
	if (sh.GetVs().gs_regs.data_addr != slot.sh.GetVs().gs_regs.data_addr) return C::ScoutRegsGsAddrN;
	if (std::memcmp(sh.GetPs().ps_user_sgpr.value, slot.sh.GetPs().ps_user_sgpr.value,
	                sizeof(sh.GetPs().ps_user_sgpr.value)) != 0)
		return C::ScoutRegsPsSgprN;
	if (std::memcmp(sh.GetVs().gs_user_sgpr.value, slot.sh.GetVs().gs_user_sgpr.value,
	                sizeof(sh.GetVs().gs_user_sgpr.value)) != 0)
		return C::ScoutRegsGsSgprN;
	if (ucfg.GetPrimType() != slot.ucfg.GetPrimType()) return C::ScoutRegsPrimN;
	if (ctx.GetRenderTargetMask() != slot.rt_mask) return C::ScoutRegsRtN;
	if (ctx.GetBlendControl(0).enable != slot.blend0) return C::ScoutRegsOtherN;
	if (ctx.GetAaConfig().msaa_num_samples != slot.aa) return C::ScoutRegsOtherN;
	if (ctx.GetClipControl().clip_disable != slot.clip_disable) return C::ScoutRegsOtherN;
	if (ctx.GetModeControl().provoking_vtx_last != slot.provoking_last) return C::ScoutRegsOtherN;
	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt = ctx.GetRenderTarget(i);
		if (rt.base.addr != slot.rt_addr[i]) return C::ScoutRegsRtN;
		const auto packed = static_cast<uint32_t>(rt.info.format) |
		                    (static_cast<uint32_t>(rt.info.channel_type) << 16u);
		if (packed != slot.rt_format[i]) return C::ScoutRegsRtN;
	}
	return C::Count;
}

void ComputePrograms(PipelineCache& cache, const HW::Context& ctx, const HW::Shader& sh,
                     const HW::UserConfig& ucfg, Slot& slot) {
	const auto color_output_mask = ColorOutputMask(ctx);
	const bool ps_active = sh.GetPs().ps_regs.data_addr != 0 &&
	                       (color_output_mask != 0 || PixelSideEffects(ctx.GetShaderRegisters()));
	std::array<Prospero::ColorComponentMapping, 8> mapping {};
	for (uint32_t slot_i = 0; slot_i < 8; slot_i++) {
		const auto& rt = ctx.GetRenderTarget(slot_i);
		if ((color_output_mask & (1u << slot_i)) != 0 && rt.base.addr != 0) {
			mapping[slot_i] = TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type,
			                                               rt.info.channel_order)
			                      .export_mapping;
		}
	}
	const bool single_sample = render_sample_count(ctx.GetAaConfig().msaa_num_samples) == 1u;
	slot.programs            = cache.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(),
	                                                     ctx, ucfg, mapping, ps_active, slot.vertex_info,
	                                                     slot.pixel_info, single_sample);
}

void NoteStopReason(uint32_t opcode, bool parse) {
	using C = FrameTiming::Counter;
	C c = C::ScoutStopOtherN;
	if (parse) c = C::ScoutStopParseN;
	else if (opcode == Pm4::IT_WAIT_REG_MEM || opcode == Pm4::IT_WAIT_REG_MEM_64) c = C::ScoutStopWaitN;
	else if (opcode == Pm4::IT_COND_EXEC) c = C::ScoutStopCondN;
	else if (opcode == Pm4::IT_DRAW_INDIRECT_MULTI || opcode == Pm4::IT_DRAW_INDEX_INDIRECT_MULTI) c = C::ScoutStopMultiN;
	else if (opcode == Pm4::IT_WRITE_CONST_RAM || opcode == Pm4::IT_DUMP_CONST_RAM || opcode == Pm4::IT_WAIT_ON_CE_COUNTER ||
	         opcode == Pm4::IT_WAIT_ON_DE_COUNTER_DIFF) c = C::ScoutStopCeN;
	else if (opcode == Pm4::IT_DISPATCH_DIRECT || opcode == Pm4::IT_DISPATCH_INDIRECT || opcode == Pm4::IT_DISPATCH_DRAW) c = C::ScoutStopDispatchN;
	else if (opcode == Pm4::IT_WRITE_DATA) c = C::ScoutStopWriteDataN;
	FrameTiming::Add(c);
	if (c == C::ScoutStopOtherN) {
		static std::atomic<uint64_t> seen[4] {};
		const uint64_t               bit = 1ull << (opcode & 63u);
		if ((seen[(opcode >> 6u) & 3u].fetch_or(bit) & bit) == 0) std::printf("[scout] stop on opcode 0x%x\n", opcode);
	}
}

void ApplySet(CommandProcessor& cp, uint32_t header, const uint32_t* packet, uint32_t remaining_dw,
              uint32_t total_dw) {
	const auto opcode = (header >> 8u) & 0xffu;
	auto       call   = g_cp_op_func[opcode];
	if (call == nullptr) return;
	call(cp, header & ~1u, packet + 1, remaining_dw, total_dw);
}

void WorkerLoop() {
	for (;;) {
		Job job;
		{
			std::unique_lock lock(g_mu);
			g_cv.wait(lock, [] { return g_stop.load() || g_job.pending.load(); });
			if (g_stop.load() && !g_job.pending.load()) return;
			job.cp       = g_job.cp;
			job.cache    = g_job.cache;
			job.commands = g_job.commands;
			job.ctx      = g_job.ctx;
			job.sh       = g_job.sh;
			job.ucfg     = g_job.ucfg;
			job.marker   = g_job.marker;
			g_job.pending.store(false);
			g_job.cp = nullptr;
		}
		CommandProcessor::ScoutRegs tls;
		tls.ctx    = &job.ctx;
		tls.sh     = &job.sh;
		tls.ucfg   = &job.ucfg;
		tls.marker = &job.marker;
		CommandProcessor::ScoutRegsTls() = &tls;

		std::array<uint32_t, kShNum> sh_words {};
		struct Cursor {
			std::span<const uint32_t> commands;
			uint32_t                  offset = 0;
		};
		std::vector<Cursor> stack;
		stack.push_back({job.commands, 0});
		bool stopped = false;

		auto apply_sh_words = [&](uint32_t offset, uint32_t value) {
			if (offset < kShNum) sh_words[offset] = value;
		};

		while (!stack.empty() && !g_stop.load()) {
			auto& cursor = stack.back();
			if (cursor.offset >= cursor.commands.size()) {
				stack.pop_back();
				continue;
			}
			const auto* packet       = cursor.commands.data() + cursor.offset;
			const auto  remaining_dw = static_cast<uint32_t>(cursor.commands.size() - cursor.offset);
			const auto  header       = packet[0];
			const auto  opcode       = (header >> 8u) & 0xffu;
			if (header == 0x80000000u || Pm4::PacketSizeDw(header) == 1u) {
				cursor.offset++;
				continue;
			}
			if (remaining_dw < 2) {
				NoteStopReason(opcode, true);
				stopped = true;
				break;
			}
			const auto packet_dw = Pm4::PacketSizeDw(header);
			if (packet_dw == 0 || packet_dw > remaining_dw) {
				NoteStopReason(opcode, true);
				stopped = true;
				break;
			}
			if (IsStopOpcode(opcode)) {
				NoteStopReason(opcode, false);
				stopped = true;
				break;
			}
			if (opcode == Pm4::IT_INDIRECT_BUFFER && packet_dw >= 4u) {
				const uint32_t control = packet[3];
				const auto     num_dw  = control & 0xfffffu;
				auto*          ib =
				    reinterpret_cast<const uint32_t*>(packet[1] | (static_cast<uint64_t>(packet[2]) << 32u));
				cursor.offset += packet_dw;
				if (num_dw != 0 && ib != nullptr) {
					const bool chain = (control & (1u << 20u)) != 0;
					if (chain) cursor = {{ib, num_dw}, 0};
					else stack.push_back({{ib, num_dw}, 0});
				}
				continue;
			}
			if (opcode == Pm4::IT_SET_SH_REG) {
				const uint32_t cmd_offset = packet[1];
				if (cmd_offset != Pm4::SH_NOP) {
					const uint32_t n = packet_dw - 2u;
					for (uint32_t i = 0; i < n; i++) apply_sh_words(cmd_offset + i, packet[2 + i]);
				}
				if (job.cp != nullptr) ApplySet(*job.cp, header, packet, remaining_dw,
				                               static_cast<uint32_t>(cursor.commands.size()));
			} else if (opcode == Pm4::IT_SET_SH_REG_INDIRECT || opcode == Pm4::IT_SET_CONTEXT_REG ||
			           opcode == Pm4::IT_SET_CONTEXT_REG_INDIRECT || opcode == Pm4::IT_SET_UCONFIG_REG ||
			           opcode == Pm4::IT_SET_UCONFIG_REG_INDIRECT || opcode == Pm4::IT_SET_UCONFIG_REG_INDEX ||
			           opcode == Pm4::IT_SET_CONFIG_REG) {
				if (job.cp != nullptr) ApplySet(*job.cp, header, packet, remaining_dw,
				                               static_cast<uint32_t>(cursor.commands.size()));
			} else if (IsDrawOpcode(opcode)) {
				auto slot = std::make_unique<Slot>();
				FillSlotRegs(*slot, job.ctx, job.sh, job.ucfg);
				slot->snap.sh = sh_words;
				NoteUserDataReads(slot->snap, ReadGuestWord);
				if (job.cache != nullptr) {
					ComputePrograms(*job.cache, job.ctx, job.sh, job.ucfg, *slot);
				}
				slot->ready.store(true, std::memory_order_release);
				FrameTiming::Add(FrameTiming::Counter::ScoutWalkDrawsN);
				{
					std::lock_guard lock(g_mu);
					g_live.slots.push_back(std::move(slot));
					g_live.ready_n.fetch_add(1, std::memory_order_release);
				}
				g_cv.notify_all();
			} else if (!IsIgnoreOpcode(opcode)) {
				NoteStopReason(opcode, false);
				stopped = true;
				break;
			}
			cursor.offset += packet_dw;
		}

		CommandProcessor::ScoutRegsTls() = nullptr;
		{
			std::lock_guard lock(g_mu);
			g_live.stopped.store(stopped, std::memory_order_relaxed);
			g_live.done.store(true, std::memory_order_release);
			if (stopped) FrameTiming::Add(FrameTiming::Counter::ScoutStopN);
		}
		g_cv.notify_all();
	}
}

void EnsureWorker() {
	if (g_started) return;
	g_started = true;
	g_live.done.store(true);
	g_job.pending.store(false);
	g_worker = std::thread(WorkerLoop);
}

} // namespace

void Begin(CommandProcessor& cp, PipelineCache& cache, std::span<const uint32_t> commands) {
	if (Current() == Mode::Off) return;
	FrameTiming::Add(FrameTiming::Counter::ScoutBeginN);
	EnsureWorker();
	{
		std::unique_lock lock(g_mu);
		g_cv.wait(lock, [] { return g_live.done.load(); });
		g_live.slots.clear();
		g_live.ready_n.store(0);
		g_live.stopped.store(false);
		g_live.done.store(false);
		g_live.cp_draw = 0;
		g_job.cp       = &cp;
		g_job.cache    = &cache;
		g_job.commands = commands;
		cp.CopyRegs(g_job.ctx, g_job.sh, g_job.ucfg, g_job.marker);
		g_job.pending.store(true);
	}
	g_cv.notify_all();
}

void Shutdown() {
	{
		std::lock_guard lock(g_mu);
		g_stop.store(true);
		g_job.pending.store(false);
		g_live.done.store(true);
	}
	g_cv.notify_all();
	if (g_worker.joinable()) g_worker.join();
	g_started = false;
	g_stop.store(false);
}

TakeKind Take(CommandBuffer& buffer, Prepared& out) {
	if (Current() == Mode::Off) return TakeKind::Miss;
	const uint32_t idx = g_live.cp_draw++;
	Slot*          slot = nullptr;
	FrameTiming::Add(FrameTiming::Counter::ScoutTakeN);
	{
		// Wait until the worker has this draw or has stopped. A miss-without-wait
		// lets CP enter GetGraphicsPrograms and starve the worker on m_get_mutex,
		// so every later draw also misses.
		std::unique_lock lock(g_mu);
		g_cv.wait(lock, [idx] {
			return g_stop.load() || g_live.done.load(std::memory_order_acquire) ||
			       (idx < g_live.slots.size() && g_live.slots[idx] &&
			        g_live.slots[idx]->ready.load(std::memory_order_acquire));
		});
		if (idx < g_live.slots.size() && g_live.slots[idx] &&
		    g_live.slots[idx]->ready.load(std::memory_order_acquire)) {
			slot = g_live.slots[idx].get();
		}
	}
	if (slot == nullptr) {
		FrameTiming::Add(FrameTiming::Counter::ScoutMissN);
		FrameTiming::Add(g_live.stopped.load(std::memory_order_relaxed) ? FrameTiming::Counter::ScoutMissNoslotStopN
		                                                                : FrameTiming::Counter::ScoutMissNoslotRunN);
		return TakeKind::Miss;
	}
	const auto regs_diff = RegsDiff(*slot, buffer.GetRegisters(), buffer.GetShaders(), buffer.GetUserConfig());
	if (regs_diff != FrameTiming::Counter::Count) {
		FrameTiming::Add(FrameTiming::Counter::ScoutMissN);
		FrameTiming::Add(FrameTiming::Counter::ScoutMissRegsN);
		FrameTiming::Add(regs_diff);
		return TakeKind::Miss;
	}
	if (!Matches(slot->snap, ReadGuestWord)) {
		FrameTiming::Add(FrameTiming::Counter::ScoutMissN);
		FrameTiming::Add(FrameTiming::Counter::ScoutMissMemN);
		return TakeKind::Miss;
	}
	out.programs    = slot->programs;
	out.vertex_info = slot->vertex_info;
	out.pixel_info  = slot->pixel_info;
	if (Current() == Mode::Shadow) return TakeKind::Shadow;
	FrameTiming::Add(FrameTiming::Counter::ScoutHitN);
	return TakeKind::Hit;
}

} // namespace Libs::Graphics::CpScout
