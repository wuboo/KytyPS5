#include "graphics/guest_gpu/command_processor/cpScoutRuntime.h"

#include "graphics/guest_gpu/command_processor/cpScoutValidate.h"

#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/shader.h"
#include "kernel/memory.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <memory>
#include <optional>
#include <mutex>
#include <span>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dbghelp.h>
#include <tlhelp32.h>
#pragma comment(lib, "dbghelp.lib")
#endif

namespace Libs::Graphics::CpScout {
namespace {

// S471 DIAG: the stacks of every thread of the process (symbols from the pdb next to the exe), printed to stderr when the monitor sees a stall.
void DumpAllStacks() {
#ifdef _WIN32
	const HANDLE process = GetCurrentProcess();
	SymSetOptions(SYMOPT_LOAD_LINES | SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
	SymInitialize(process, nullptr, TRUE);
	const DWORD self = GetCurrentThreadId();
	HANDLE      snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snap == INVALID_HANDLE_VALUE) return;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
		if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) continue;
		HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) continue;
		if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
			CONTEXT context {};
			context.ContextFlags = CONTEXT_FULL;
			if (GetThreadContext(thread, &context)) {
				std::fprintf(stderr, "[scout] STACK tid=%lu\n", static_cast<unsigned long>(entry.th32ThreadID));
				STACKFRAME64 frame {};
				frame.AddrPC.Offset    = context.Rip;
				frame.AddrPC.Mode      = AddrModeFlat;
				frame.AddrFrame.Offset = context.Rbp;
				frame.AddrFrame.Mode   = AddrModeFlat;
				frame.AddrStack.Offset = context.Rsp;
				frame.AddrStack.Mode   = AddrModeFlat;
				for (int depth = 0; depth < 24; depth++) {
					if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &context, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
					if (frame.AddrPC.Offset == 0) break;
					alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 256] {};
					auto*                     symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
					symbol->SizeOfStruct             = sizeof(SYMBOL_INFO);
					symbol->MaxNameLen               = 255;
					DWORD64 displacement             = 0;
					DWORD   line_displacement        = 0;
					IMAGEHLP_LINE64 line {};
					line.SizeOfStruct = sizeof(line);
					if (SymFromAddr(process, frame.AddrPC.Offset, &displacement, symbol)) {
						const bool has_line = SymGetLineFromAddr64(process, frame.AddrPC.Offset, &line_displacement, &line) != 0;
						std::fprintf(stderr, "[scout]   %s+0x%llx %s:%lu\n", symbol->Name, static_cast<unsigned long long>(displacement),
						             has_line ? line.FileName : "", has_line ? static_cast<unsigned long>(line.LineNumber) : 0ul);
					} else {
						std::fprintf(stderr, "[scout]   0x%llx\n", static_cast<unsigned long long>(frame.AddrPC.Offset));
					}
				}
			}
			ResumeThread(thread);
		}
		CloseHandle(thread);
	}
	CloseHandle(snap);
	std::fflush(stderr);
#endif
}

struct Slot {
	HW::Shader                           sh;
	HW::UserConfig                       ucfg;
	HW::ShaderRegisters                  shr;
	HW::RenderTarget                     rt[8];
	HW::BlendControl                     blend[8];
	HW::ClipControl                      clip;
	HW::ModeControl                      mode;
	HW::AaConfig                         aa_cfg;
	HW::ScreenViewport                   viewport;
	uint32_t                             rt_mask = 0;
	DrawSnap                             snap;
	PipelineCache::GraphicsPrograms      programs;
	std::array<ShaderVertexInputInfo, 3> vertex_info {};
	ShaderPixelInputInfo                 pixel_info {};
	// S471: the slot owns the snapshots its stage inputs point at (the worker's private ones are overwritten by its next call), and the table reads of the walk.
	ShaderRecompiler::IR::ResourceSnapshot vs_res;
	ShaderRecompiler::IR::ResourceSnapshot ps_res;
	ScoutTableLog::Log                     tables;
	bool                                   dirty = false; // the worker could not compute it safely: the CP does
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
// S471: a worker that stops answering must not hang the CP: Take / Begin give up after kWatchdogMs, say where the worker was, and leave the scout off for the rest of the run.
constexpr uint32_t        kWatchdogMs = 10000;
std::atomic<bool>         g_dead {false};
std::atomic<uint32_t>     g_dbg_phase {0}; // 0 waiting for a job, 1 walking packets, 2 in GetGraphicsPrograms
std::atomic<uint32_t>     g_dbg_op {0};
std::atomic<uint32_t>     g_dbg_draws {0};

std::atomic<uint64_t>     g_dbg_tick {0};
std::thread               g_monitor;

// S471 DIAG (KYTY_CP_SCOUT_PROF=1): a poor man's profiler of every thread, flips 6400..6440, one folded stack per thread per sample on stderr ("[prof] tid a;b;c").
void ProfSample() {
#ifdef _WIN32
	struct One {
		DWORD                 tid;
		std::vector<DWORD64>  pcs;
	};
	std::vector<One> ones;
	const HANDLE     process = GetCurrentProcess();
	const DWORD      self    = GetCurrentThreadId();
	HANDLE           snap    = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
	if (snap == INVALID_HANDLE_VALUE) return;
	THREADENTRY32 entry {};
	entry.dwSize = sizeof(entry);
	for (BOOL ok = Thread32First(snap, &entry); ok; ok = Thread32Next(snap, &entry)) {
		if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) continue;
		HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE, entry.th32ThreadID);
		if (thread == nullptr) continue;
		if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
			CONTEXT context {};
			context.ContextFlags = CONTEXT_FULL;
			if (GetThreadContext(thread, &context)) {
				One one {entry.th32ThreadID, {}};
				STACKFRAME64 frame {};
				frame.AddrPC.Offset    = context.Rip;
				frame.AddrPC.Mode      = AddrModeFlat;
				frame.AddrFrame.Offset = context.Rbp;
				frame.AddrFrame.Mode   = AddrModeFlat;
				frame.AddrStack.Offset = context.Rsp;
				frame.AddrStack.Mode   = AddrModeFlat;
				for (int depth = 0; depth < 14; depth++) {
					if (!StackWalk64(IMAGE_FILE_MACHINE_AMD64, process, thread, &frame, &context, nullptr, SymFunctionTableAccess64, SymGetModuleBase64, nullptr)) break;
					if (frame.AddrPC.Offset == 0) break;
					one.pcs.push_back(frame.AddrPC.Offset);
				}
				ones.push_back(std::move(one));
			}
			ResumeThread(thread);
		}
		CloseHandle(thread);
	}
	CloseHandle(snap);
	static std::unordered_map<DWORD64, std::string> names;
	for (const auto& one: ones) {
		std::string line;
		for (const auto pc: one.pcs) {
			auto it = names.find(pc);
			if (it == names.end()) {
				alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 200] {};
				auto*                     symbol = reinterpret_cast<SYMBOL_INFO*>(buffer);
				symbol->SizeOfStruct             = sizeof(SYMBOL_INFO);
				symbol->MaxNameLen               = 199;
				DWORD64 displacement             = 0;
				it = names.emplace(pc, SymFromAddr(process, pc, &displacement, symbol) ? std::string(symbol->Name) : std::string("?")).first;
			}
			line += it->second;
			line += ';';
		}
		std::fprintf(stderr, "[prof] %lu %s\n", static_cast<unsigned long>(one.tid), line.c_str());
	}
#endif
}

// Says (once per stall) what the worker is doing when nothing moved for 10 s, whoever is stuck.
void MonitorLoop() {
	static const bool prof = [] { const char* v = std::getenv("KYTY_CP_SCOUT_PROF"); return v != nullptr && v[0] == '1'; }();
	if (prof) {
#ifdef _WIN32
		SymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
		SymInitialize(GetCurrentProcess(), nullptr, TRUE);
#endif
		while (!g_stop.load()) {
			const auto flips = FrameTiming::GuestFlips();
			if (flips >= 6400 && flips < 6440) {
				ProfSample();
				std::this_thread::sleep_for(std::chrono::milliseconds(15));
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		}
		return;
	}
	uint64_t last = ~uint64_t {0};
	uint32_t same = 0;
	bool     said = false;
	while (!g_stop.load()) {
		std::this_thread::sleep_for(std::chrono::seconds(5));
		const auto tick = g_dbg_tick.load();
		same            = tick == last ? same + 1 : 0;
		last            = tick;
		if (same >= 2 && !said) {
			said = true;
			std::fprintf(stderr, "[scout] MONITOR no progress 10 s: worker phase=%u last_op=0x%x draws=%u mark=%u cp_in_gfp=%d flips=%llu\n", g_dbg_phase.load(), g_dbg_op.load(),
			             g_dbg_draws.load(), PipelineCache::ScoutScope::Marked(), PipelineCache::ScoutScope::CpIsInGfp() ? 1 : 0, static_cast<unsigned long long>(FrameTiming::GuestFlips()));
			std::fflush(stderr);
			DumpAllStacks();
		}
		if (same == 0) said = false;
	}
}

void Dead(const char* where) {
	if (g_dead.exchange(true)) return;
	std::fprintf(stderr, "[scout] WATCHDOG at %s: worker phase=%u last_op=0x%x draws=%u slots=%zu done=%d stopped=%d first=%u\n", where, g_dbg_phase.load(),
	             g_dbg_op.load(), g_dbg_draws.load(), g_live.slots.size(), g_live.done.load() ? 1 : 0, g_live.stopped.load() ? 1 : 0, g_live.cp_draw);
	std::fflush(stderr);
}

// A user-data word that is not a pointer must not fault the worker: the read goes through the guest backing store, which
// fails for unmapped (and unbacked) addresses instead of faulting. Such addresses are not recorded.
bool ReadGuestWord(uint64_t address, uint32_t& word) {
	if (address < 0x40000ull) return false;
	return Libs::LibKernel::Memory::TryReadBacking(address, &word, sizeof(word));
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
	slot.sh      = sh;
	slot.ucfg    = ucfg;
	slot.shr     = ctx.GetShaderRegisters();
	slot.rt_mask = ctx.GetRenderTargetMask();
	for (uint32_t i = 0; i < 8; i++) {
		slot.rt[i]    = ctx.GetRenderTarget(i);
		slot.blend[i] = ctx.GetBlendControl(i);
	}
	slot.clip     = ctx.GetClipControl();
	slot.mode     = ctx.GetModeControl();
	slot.aa_cfg   = ctx.GetAaConfig();
	slot.viewport = ctx.GetScreenViewport();
}

template <class T>
bool Same(const T& a, const T& b) {
	return std::memcmp(&a, &b, sizeof(T)) == 0;
}

bool SameSgpr(const HW::UserSgprInfo& a, const HW::UserSgprInfo& b) {
	return a.count == b.count && std::memcmp(a.value, b.value, sizeof(a.value)) == 0 &&
	       std::memcmp(a.type, b.type, sizeof(a.type)) == 0;
}

// Returns the counter of the first differing field of everything GetGraphicsPrograms reads, or Count when equal.
// Whole structs are compared byte for byte; a spurious miss from padding shows up in the counters.
FrameTiming::Counter RegsDiff(const Slot& slot, const HW::Context& ctx, const HW::Shader& sh, const HW::UserConfig& ucfg) {
	using C = FrameTiming::Counter;
	const auto& ps = sh.GetPs();
	const auto& vs = sh.GetVs();
	const auto& sp = slot.sh.GetPs();
	const auto& sv = slot.sh.GetVs();
	if (!Same(ps.ps_regs, sp.ps_regs)) return C::ScoutRegsPsAddrN;
	if (!Same(vs.es_regs, sv.es_regs) || !Same(vs.ls_regs, sv.ls_regs) || !Same(vs.hs_regs, sv.hs_regs) ||
	    !Same(vs.gs_regs, sv.gs_regs))
		return C::ScoutRegsGsAddrN;
	if (!SameSgpr(ps.ps_user_sgpr, sp.ps_user_sgpr)) return C::ScoutRegsPsSgprN;
	if (!SameSgpr(vs.gs_user_sgpr, sv.gs_user_sgpr) || !SameSgpr(vs.hs_user_sgpr, sv.hs_user_sgpr)) return C::ScoutRegsGsSgprN;
	if (ucfg.GetPrimType() != slot.ucfg.GetPrimType()) return C::ScoutRegsPrimN;
	if (ctx.GetRenderTargetMask() != slot.rt_mask) return C::ScoutRegsRtN;
	for (uint32_t i = 0; i < 8; i++) {
		if (!Same(ctx.GetRenderTarget(i), slot.rt[i])) return C::ScoutRegsRtN;
		if (!Same(ctx.GetBlendControl(i), slot.blend[i])) return C::ScoutRegsOtherN;
	}
	if (!Same(ctx.GetShaderRegisters(), slot.shr)) return C::ScoutRegsOtherN;
	if (!Same(ctx.GetClipControl(), slot.clip) || !Same(ctx.GetModeControl(), slot.mode) ||
	    !Same(ctx.GetAaConfig(), slot.aa_cfg))
		return C::ScoutRegsOtherN;
	if (slot.clip.clip_disable && !Same(ctx.GetScreenViewport(), slot.viewport)) return C::ScoutRegsOtherN;
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
	PipelineCache::ScoutScope::Reset();
	ScoutTableLog::Log log;
	ScoutTableLog::Active() = &log;
	std::optional<FrameTiming::ScopedCounterNs> gfp_timer(std::in_place, FrameTiming::Counter::ScoutWorkerGfpNs);
	slot.programs           = cache.GetGraphicsPrograms(sh.GetVs(), sh.GetPs(), ctx.GetShaderRegisters(),
	                                                    ctx, ucfg, mapping, ps_active, slot.vertex_info,
	                                                    slot.pixel_info, single_sample);
	ScoutTableLog::Active() = nullptr;
	gfp_timer.reset();
	using C = FrameTiming::Counter;
	if (PipelineCache::ScoutScope::Failed() || !slot.programs.vertex[0] || (ps_active && !slot.programs.pixel) ||
	    slot.vertex_info[0].stage.resources == nullptr || (ps_active && slot.pixel_info.stage.resources == nullptr)) {
		slot.dirty = true;
		switch (PipelineCache::ScoutScope::FailReason()) {
			case 2: FrameTiming::Add(C::ScoutBailMemoN); break;
			case 3: FrameTiming::Add(C::ScoutBailShapeN); break;
			case 4: FrameTiming::Add(C::ScoutBailMeshN); break;
			case 5: FrameTiming::Add(C::ScoutBailTableN); break;
			default: FrameTiming::Add(C::ScoutBailEntryN); break;
		}
		return;
	}
	if (log.unsafe) {
		slot.dirty = true;
		FrameTiming::Add(C::ScoutBailTableN);
		return;
	}
	slot.tables = std::move(log);
	// The stage inputs point at the worker's private snapshots, which its next call overwrites: the slot keeps copies.
	slot.vs_res                          = *slot.vertex_info[0].stage.resources;
	slot.vertex_info[0].stage.resources = &slot.vs_res;
	if (ps_active) {
		slot.ps_res                       = *slot.pixel_info.stage.resources;
		slot.pixel_info.stage.resources = &slot.ps_res;
	}
}

// The CP's own readers for the replay of a slot (see cpScoutValidate.h).
bool ReplayStrict(uint64_t address, std::span<uint32_t> values) {
	return Libs::LibKernel::Memory::TryReadQuiet(address, values.data(), values.size_bytes());
}

bool ReplayOrdinary(uint64_t address, std::span<uint32_t> values) {
	if (address < 0x40000ull) {
		std::fill(values.begin(), values.end(), 0u);
		return true;
	}
	return Libs::LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes());
}

// True when the slot's result is what the CP would compute now.
bool ValidateSlot(const Slot& slot) {
	using C = FrameTiming::Counter;
	uint32_t      reads = 0;
	Verdict       verdict = ReadSetVerdict(slot.vs_res.srt_memo, ReplayStrict, ReplayOrdinary, &reads);
	if (verdict == Verdict::Ok && slot.pixel_info.stage.resources != nullptr) {
		verdict = ReadSetVerdict(slot.ps_res.srt_memo, ReplayStrict, ReplayOrdinary, &reads);
	}
	FrameTiming::Add(C::ScoutValReadsN, reads);
	if (verdict == Verdict::Busy) {
		FrameTiming::Add(C::ScoutValBusyN);
		return false;
	}
	if (verdict != Verdict::Ok) {
		FrameTiming::Add(C::ScoutValChangedN);
		return false;
	}
	const bool tables_ok = TablesUnchanged(slot.tables, [](uint32_t site, uint64_t address, uint32_t* dst, uint32_t dwords) {
		Libs::LibKernel::Memory::CpGuestRead(static_cast<Graphics::CpReadBacking::Site>(site), address, dst, dwords * sizeof(uint32_t));
	});
	if (!tables_ok) {
		FrameTiming::Add(C::ScoutValTableN);
		return false;
	}
	return true;
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

// Opcodes that do not change any register GetGraphicsPrograms reads; the guest memory they write is covered by the re-read in Take.
bool IsMemoryOnlyOpcode(uint32_t op) {
	return op == Pm4::IT_DISPATCH_DIRECT || op == Pm4::IT_DISPATCH_INDIRECT || op == Pm4::IT_DISPATCH_DRAW ||
	       op == Pm4::IT_WRITE_DATA || op == Pm4::IT_COPY_DATA || op == Pm4::IT_DMA_DATA || op == Pm4::IT_CP_DMA ||
	       op == Pm4::IT_WAIT_REG_MEM || op == Pm4::IT_WAIT_REG_MEM_64 || op == Pm4::IT_MEM_SEMAPHORE ||
	       op == Pm4::IT_SET_BASE;
}

void ApplySet(CommandProcessor& cp, uint32_t header, const uint32_t* packet, uint32_t remaining_dw,
              uint32_t total_dw) {
	const auto opcode = (header >> 8u) & 0xffu;
	auto       call   = g_cp_op_func[opcode];
	if (call == nullptr) return;
	call(cp, header & ~1u, packet + 1, remaining_dw, total_dw);
}

void WorkerLoop() {
	{
		const char* raw = std::getenv("KYTY_CP_SCOUT_RAW");
		Libs::LibKernel::Memory::SetScoutRawReadThread(raw != nullptr && raw[0] == '1');
	}
#ifdef _WIN32
	if (const char* prio = std::getenv("KYTY_CP_SCOUT_PRIO"); prio != nullptr && prio[0] != '\0') { // S471 DIAG: -2 idle ... 0 normal ... 2 above normal
		SetThreadPriority(GetCurrentThread(), std::atoi(prio));
	}
#endif
	PipelineCache::ScoutScope scout_scope; // S471: this thread's GetGraphicsPrograms is the private, read-only-on-shared-state one
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
		g_dbg_phase.store(1, std::memory_order_relaxed);

		std::array<uint32_t, kShNum> sh_words {};
		HW::Context                  saved_ctx;
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
			g_dbg_op.store(opcode, std::memory_order_relaxed);
			g_dbg_tick.fetch_add(1, std::memory_order_relaxed);
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
			if (IsStopOpcode(opcode) && !IsMemoryOnlyOpcode(opcode)) {
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
			if (opcode == Pm4::IT_CLEAR_STATE) {
				job.ctx.Reset();
			} else if (opcode == Pm4::IT_NOP) {
				// Custom codes ride on NOP; only those that change the registers matter (the others only touch guest memory, which Take re-reads).
				const auto r = KYTY_PM4_R(header);
				if (r == Pm4::R_CONTEXT_STATE && packet_dw >= 3u) {
					switch (static_cast<ContextStateOperation>(packet[1])) {
						case ContextStateOperation::Clear: job.ctx.Reset(); break;
						case ContextStateOperation::Push:
							saved_ctx = job.ctx;
							break;
						case ContextStateOperation::Pop: job.ctx = saved_ctx; break;
						case ContextStateOperation::PushClear:
							saved_ctx = job.ctx;
							job.ctx.Reset();
							break;
						default: NoteStopReason(opcode, true); stopped = true; break;
					}
					if (stopped) break;
				} else if (r == Pm4::R_DISPATCH_RESET) {
					job.ctx.Reset();
					job.sh.Reset();
					job.ucfg.Reset();
					saved_ctx.Reset();
					job.marker = HW::UserSgprType::Unknown;
					sh_words.fill(0);
				}
			} else if (opcode == Pm4::IT_SET_SH_REG) {
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
				const FrameTiming::ScopedCounterNs draw_timer(FrameTiming::Counter::ScoutWorkerDrawNs);
				auto slot = std::make_unique<Slot>();
				FillSlotRegs(*slot, job.ctx, job.sh, job.ucfg);
				slot->snap.sh = sh_words;
				NoteUserDataReads(slot->snap, ReadGuestWord);
				static const bool no_gfp = [] { const char* v = std::getenv("KYTY_CP_SCOUT_NOGFP"); return v != nullptr && v[0] == '1'; }(); // S470 DIAG: walk and apply the registers only
				if (job.cache != nullptr && !no_gfp) {
					g_dbg_phase.store(2, std::memory_order_relaxed);
					g_dbg_draws.store(static_cast<uint32_t>(g_live.slots.size()), std::memory_order_relaxed);
					ComputePrograms(*job.cache, job.ctx, job.sh, job.ucfg, *slot);
					g_dbg_phase.store(1, std::memory_order_relaxed);
				}
				slot->ready.store(true, std::memory_order_release);
				FrameTiming::Add(FrameTiming::Counter::ScoutWalkDrawsN);
				{
					std::lock_guard lock(g_mu);
					g_live.slots.push_back(std::move(slot));
					g_live.ready_n.fetch_add(1, std::memory_order_release);
				}
				g_cv.notify_all();
			} else if (!IsIgnoreOpcode(opcode) && !IsMemoryOnlyOpcode(opcode)) {
				NoteStopReason(opcode, false);
				stopped = true;
				break;
			}
			cursor.offset += packet_dw;
		}

		CommandProcessor::ScoutRegsTls() = nullptr;
		g_dbg_phase.store(0, std::memory_order_relaxed);
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
	g_monitor = std::thread(MonitorLoop);
}

} // namespace

void Begin(CommandProcessor& cp, PipelineCache& cache, std::span<const uint32_t> commands) {
	if (Current() == Mode::Off || g_dead.load()) return;
	FrameTiming::Add(FrameTiming::Counter::ScoutBeginN);
	g_dbg_tick.fetch_add(1, std::memory_order_relaxed);
	EnsureWorker();
	{
		std::unique_lock lock(g_mu);
		if (!g_cv.wait_for(lock, std::chrono::milliseconds(kWatchdogMs), [] { return g_live.done.load(); })) {
			Dead("Begin");
			return;
		}
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
	if (g_monitor.joinable()) g_monitor.join();
	g_started = false;
	g_stop.store(false);
}

TakeKind Take(CommandBuffer& buffer, Prepared& out) {
	if (Current() == Mode::Off || g_dead.load()) return TakeKind::Miss;
	using C = FrameTiming::Counter;
	FrameTiming::Add(C::ScoutTakeN);
	constexpr uint32_t kWindow = 8;
	Slot*              cand[kWindow] {};
	uint32_t           n       = 0;
	bool               stopped = false;
	const uint32_t     first   = g_live.cp_draw;
	{
		// Wait until the worker has the next draw or has finished. A miss-without-wait lets CP enter
		// GetGraphicsPrograms and starve the worker on m_get_mutex, so every later draw also misses.
		std::unique_lock lock(g_mu);
		const FrameTiming::ScopedCounterNs wait_timer(FrameTiming::Counter::ScoutTakeWaitNs);
		if (!g_cv.wait_for(lock, std::chrono::milliseconds(kWatchdogMs), [first] {
			    return g_stop.load() || g_live.done.load(std::memory_order_acquire) ||
			           (first < g_live.slots.size() && g_live.slots[first] &&
			            g_live.slots[first]->ready.load(std::memory_order_acquire));
		    })) {
			Dead("Take");
			return TakeKind::Miss;
		}
		for (uint32_t i = first; i < g_live.slots.size() && n < kWindow; i++) {
			if (g_live.slots[i] && g_live.slots[i]->ready.load(std::memory_order_acquire)) cand[n++] = g_live.slots[i].get();
		}
		stopped = g_live.stopped.load(std::memory_order_relaxed);
	}
	if (n == 0) {
		FrameTiming::Add(C::ScoutMissN);
		FrameTiming::Add(stopped ? C::ScoutMissNoslotStopN : C::ScoutMissNoslotRunN);
		return TakeKind::Miss;
	}
	// The draw index is not trusted: CP takes only the draws that reach RefreshShaders, the walk also sees
	// draws CP drops. The slot is picked by the registers it was computed from and the guest dwords it read.
	const auto& regs_ctx = buffer.GetRegisters();
	const auto& regs_sh  = buffer.GetShaders();
	const auto& regs_uc  = buffer.GetUserConfig();
	C           first_diff = C::Count;
	bool        mem_miss   = false;
	Slot*       slot       = nullptr;
	uint32_t    at         = 0;
	for (uint32_t j = 0; j < n; j++) {
		const auto diff = RegsDiff(*cand[j], regs_ctx, regs_sh, regs_uc);
		if (diff != C::Count) {
			if (j == 0) first_diff = diff;
			continue;
		}
		if (!Matches(cand[j]->snap, ReadGuestWord)) {
			mem_miss = true;
			continue;
		}
		slot = cand[j];
		at   = j;
		break;
	}
	if (slot == nullptr) {
		FrameTiming::Add(C::ScoutMissN);
		if (mem_miss) {
			FrameTiming::Add(C::ScoutMissMemN);
		} else {
			FrameTiming::Add(C::ScoutMissRegsN);
			FrameTiming::Add(first_diff);
		}
		return TakeKind::Miss;
	}
	static const bool no_gfp = [] { const char* v = std::getenv("KYTY_CP_SCOUT_NOGFP"); return v != nullptr && v[0] == '1'; }(); // S470 DIAG
	g_live.cp_draw = first + at + 1;
	if (no_gfp) return TakeKind::Miss;
	if (at != 0) FrameTiming::Add(C::ScoutSkipN, at);
	if (slot->dirty) { // S471: the worker could not answer this one safely
		FrameTiming::Add(C::ScoutMissN);
		FrameTiming::Add(C::ScoutMissDirtyN);
		return TakeKind::Miss;
	}
	const bool valid = [&] {
		const FrameTiming::ScopedCounterNs validate_timer(FrameTiming::Counter::ScoutValidateNs);
		return ValidateSlot(*slot);
	}();
	if (!valid) { // S471: what the worker read is not what the CP reads now
		FrameTiming::Add(C::ScoutMissN);
		FrameTiming::Add(C::ScoutMissMemN);
		return TakeKind::Miss;
	}
	out.programs    = slot->programs;
	out.vertex_info = slot->vertex_info;
	out.pixel_info  = slot->pixel_info;
	if (Current() == Mode::Shadow) return TakeKind::Shadow;
	FrameTiming::Add(C::ScoutHitN);
	return TakeKind::Hit;
}

} // namespace Libs::Graphics::CpScout
