// S479 (KYTY_CP_PIPE): the FIFO between the PM4 parser (front) and the handlers that draw and signal (recorder). Both run on the CP thread; the point of the FIFO
// is that a draw is no longer executed where it is parsed, so the work GetGraphicsPrograms needs can be done by other threads in between (cpPipeWorkers.cpp).
//
// What runs where (CpPipe::Classify): register writes at parse time as before; a draw or dispatch is snapshotted (the three register structs and the CP's index/instance
// state) and executed later on the snapshot; signals, barriers, CPU writes and flips are copied and executed later in stream order; WAIT_REG_MEM is answered from the
// writes still pending in the FIFO (PendingWrites) or drains; anything else drains first. The end of every stream and every suspend drains, so what leaves
// CommandProcessor::Process is exactly what the old code left.

#include "common/assert.h"
#include "common/cpTuning.h"
#include "common/diagOut.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/cpPipe.h"
#include "graphics/guest_gpu/command_processor/cpPipeJob.h"
#include "graphics/guest_gpu/command_processor/cpPipeNoDrain.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/frameTiming.h"
#include "kernel/memory.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#if defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

namespace {

// S570 micro: KYTY_CP_TIMERS_LITE=1 drops the two clock reads per FIFO item (pipe_exec_ns) and the one per pipeline job (submit time for
// pipe_queue_ns) on the CP; those two columns stay 0. Nothing else changes (pure bookkeeping).
bool CpTimersLite() {
	static const bool lite = [] {
		const char* v = std::getenv("KYTY_CP_TIMERS_LITE");
		return v != nullptr && v[0] == '1' && v[1] == '\0';
	}();
	return lite;
}

using CpPipe::EnvMode;

size_t EnvWorkers() {
	static const size_t workers = [] {
		const char* text  = std::getenv("KYTY_CP_PIPE_WORKERS");
		const long  value = text != nullptr ? std::strtol(text, nullptr, 10) : 3;
		return static_cast<size_t>(value < 0 ? 0 : (value > 8 ? 8 : value));
	}();
	return workers;
}

// S543 (KYTY_CP_AFFINITY_PIPE=1, default off, needs KYTY_CP_AFFINITY=<lp>): the pipeline workers stay off the physical core of the CP thread, like the guest threads.
bool EnvAffinityPipe() {
	static const bool on = [] {
		const char* text = std::getenv("KYTY_CP_AFFINITY_PIPE");
		return text != nullptr && text[0] == '1' && text[1] == '\0';
	}();
	return on;
}

CpPipe::SignalPolicy EnvSignals() {
	static const CpPipe::SignalPolicy policy = CpPipe::ParseSignalPolicy(std::getenv("KYTY_CP_PIPE_SIGNALS"));
	return policy;
}

bool EnvShard() {
	static const bool shard = [] {
		const char* text = std::getenv("KYTY_CP_PIPE_SHARD");
		return text != nullptr && text[0] == '1';
	}();
	return shard;
}

size_t EnvDepth() {
	static const size_t depth = [] {
		const char* text = std::getenv("KYTY_CP_PIPE_DEPTH");
		const long  value = text != nullptr ? std::strtol(text, nullptr, 10) : 64;
		return static_cast<size_t>(value < 4 ? 4 : (value > 256 ? 256 : value));
	}();
	return depth;
}

} // namespace

namespace CpPipe {

GfpJob*& CurrentJob() {
	static thread_local GfpJob* job = nullptr;
	return job;
}

namespace {

inline void CpuRelax() {
#if defined(_MSC_VER) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

// The pool of threads that run GfpJobs. A job is a pointer into a FIFO item; the worker never touches the CP. One queue for all workers, or (shard) one per worker with the
// job routed by its shader: the same worker then sees the consecutive draws of a shader, which is what its private memo of the walk needs.
class Workers {
public:
	static Workers& Instance() {
		static Workers* pool = new Workers; // never destroyed: the process ends with threads parked
		return *pool;
	}

	void Start(size_t count, bool shard) {
		const std::lock_guard lock(m_start_mutex);
		if (m_started) return;
		m_started = true;
		m_shard   = shard;
		m_count   = count;
		m_queues  = std::vector<Queue>(shard ? count : 1);
		for (size_t i = 0; i < count; i++) {
			m_threads.emplace_back([this, i] { Run(i); });
			m_threads.back().detach();
		}
	}

	void Submit(GfpJob* job, uint64_t shard_key) {
		Queue& queue = m_queues[m_shard ? static_cast<size_t>(shard_key % m_count) : 0];
		{
			const std::lock_guard lock(queue.mutex);
			queue.jobs.push_back(job);
			queue.size.fetch_add(1, std::memory_order_release);
		}
		if (queue.sleepers.load(std::memory_order_acquire) != 0) queue.cv.notify_one();
	}

private:
	struct Queue {
		std::mutex              mutex;
		std::condition_variable cv;
		std::deque<GfpJob*>     jobs;
		std::atomic<size_t>     size {0};
		std::atomic<int>        sleepers {0};
	};

	static GfpJob* Pop(Queue& queue) {
		const std::lock_guard lock(queue.mutex);
		if (queue.jobs.empty()) return nullptr;
		GfpJob* job = queue.jobs.front();
		queue.jobs.pop_front();
		queue.size.fetch_sub(1, std::memory_order_release);
		return job;
	}

	void Run(size_t index) {
		FrameTiming::SetThreadMute(true);
		if (EnvAffinityPipe()) Common::CpTuning::ApplyToGuestThread();
		Queue& queue = m_queues[m_shard ? index : 0];
		for (;;) {
			GfpJob* job = nullptr;
			// Spin first: the CP submits a job every few microseconds while a stream is parsed; a sleeping worker costs the CP a wake-up call each time.
			const auto spin_end = std::chrono::steady_clock::now() + std::chrono::microseconds(100);
			for (uint32_t i = 0; job == nullptr; i++) {
				if (queue.size.load(std::memory_order_acquire) != 0) {
					job = Pop(queue);
					if (job != nullptr) break;
				}
				CpuRelax();
				if ((i & 63u) == 63u && std::chrono::steady_clock::now() >= spin_end) break;
			}
			if (job == nullptr) {
				std::unique_lock lock(queue.mutex);
				queue.sleepers.fetch_add(1, std::memory_order_release);
				queue.cv.wait(lock, [&queue] { return !queue.jobs.empty(); });
				queue.sleepers.fetch_sub(1, std::memory_order_release);
				job = queue.jobs.front();
				queue.jobs.pop_front();
				queue.size.fetch_sub(1, std::memory_order_release);
			}
			if (job->submit_ns != 0) FrameTiming::AddAlways(FrameTiming::Counter::PipeQueueNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - job->submit_ns));
			RunGfpJob(*job);
			FrameTiming::AddAlways(FrameTiming::Counter::PipeWorkerNs, static_cast<uint64_t>(job->worker_ns));
			job->state.store(GfpJob::Done, std::memory_order_release);
		}
	}

	std::mutex               m_start_mutex;
	bool                     m_started = false;
	bool                     m_shard   = false;
	size_t                   m_count   = 0;
	std::vector<Queue>       m_queues;
	std::vector<std::thread> m_threads;
};

void WaitDone(GfpJob& job) {
	if (job.state.load(std::memory_order_acquire) == GfpJob::Done) return;
	const int64_t t0 = FrameTiming::SteadyNowNs();
	for (uint32_t i = 0; job.state.load(std::memory_order_acquire) != GfpJob::Done; i++) {
		CpuRelax();
		if ((i & 255u) == 255u) {
			std::this_thread::yield();
			if (FrameTiming::SteadyNowNs() - t0 > 2'000'000'000ll) {
				static std::atomic<uint32_t> logged {0};
				if (logged.fetch_add(1) < 4) LOGF("[cp-pipe] stall: a worker has not finished a job for 2 s\n");
				i = 0;
			}
		}
	}
	FrameTiming::Add(FrameTiming::Counter::PipeStallNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
}

} // namespace

bool JoinJob(GfpJob& job) {
	using C = FrameTiming::Counter;
	if (job.state.load(std::memory_order_acquire) == GfpJob::Idle) return false;
	WaitDone(job);
	FrameTiming::Add(C::PipeJobN);
	if (!job.answered) {
		FrameTiming::Add(C::PipeJobBailN);
		return false;
	}
	// A page the worker read that holds GPU-written bytes the backing does not have yet (or an unpublished download of them): the CP computes the programs itself, which drains.
	for (const uint64_t page: job.pages) {
		if (Libs::LibKernel::Memory::QueryShaderGpuDirty(page, 4096)) {
			FrameTiming::Add(C::PipeJoinDirtyN);
			return false;
		}
	}
	// A write into the backing after the worker started (the publish of a download, an upload) that touched a page it read: the bytes it saw may be older than the draw's.
	if (!job.pages.empty() && Libs::LibKernel::Memory::BackingWrittenSince(job.write_seq, job.pages.data(), job.pages.size())) {
		FrameTiming::Add(C::PipeJoinPubN);
		return false;
	}
	return true;
}

} // namespace CpPipe

// One deferred packet. Draws and dispatches carry the state they were parsed with.
struct PipeItem {
	CpPipe::OpClass       cls    = CpPipe::OpClass::Ordered;
	uint32_t              opcode = 0;
	std::vector<uint32_t> dw; // the whole packet, header included
	HW::Context           ctx;
	HW::Shader            sh;
	HW::UserConfig        ucfg;
	CpPipe::CpDrawState   ds; // index type, sizes, bases, the resolved instance count
	CpPipe::GfpJob        job; // a graphics draw: the worker's GetGraphicsPrograms on this item's snapshot
	// S549 (front thread): the FIFO item that has to run before the job may start (0: none), the Sync item's live packet and result, the shadow hash of the snapshot.
	uint64_t        gate        = 0;
	uint32_t        sync_kind   = 0;
	const uint32_t* live_packet = nullptr;
	uint32_t        remaining_dw = 0;
	uint32_t        total_dw     = 0;
	uint32_t        result_dw    = 0;
	uint64_t        shadow_hash  = 0;
};

struct CommandProcessor::PipeState {
	explicit PipeState(size_t depth) : ring(depth), depth(depth), items(depth) {}

	CpPipe::Ring          ring;
	size_t                depth;
	std::vector<PipeItem> items;
	CpPipe::PendingWrites pending;
	CpPipe::GateTracker   gates;
	struct Held {
		CpPipe::GfpJob* job;
		uint64_t        gate; // the FIFO item that has to run before the job may start
	};
	std::deque<Held> held;      // jobs waiting for a gate, oldest first (their gates only grow)
	bool             executing = false; // an item runs: the pipeline is transparent for the handlers it calls
	// S549 (KYTY_CP_FRONT_THREAD): the hand-over ring (items keep living in `items`, slot = seq % depth), the stream the front parses, the recorder's cursor over the
	// committed items whose job waits for its gate.
	std::unique_ptr<CpPipe::FrontRing> front;
	bool                               front_shadow = false;
	std::atomic<uint32_t>              front_go {0};
	Pm4Execution*                      front_exec = nullptr;
	GuestGpu*                          front_gpu  = nullptr;
	uint64_t                           rel_cursor = 1;
};

namespace {
void SubmitJob(CpPipe::GfpJob* job) {
	job->submit_ns = CpTimersLite() ? 0 : FrameTiming::SteadyNowNs(); // S570: 0 = not timed
	job->state.store(CpPipe::GfpJob::Queued, std::memory_order_relaxed);
	CpPipe::Workers::Instance().Submit(job, job->shard_key);
}
} // namespace

CommandProcessor::CommandProcessor(RenderContext& renderer, int interrupt_event_id) : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::PipeInit() {
	if (m_pipe != nullptr || IsAsyncComputeQueue() || EnvMode() == CpPipe::Mode::Off) return;
	m_pipe = std::make_unique<PipeState>(EnvDepth());
	if (EnvWorkers() != 0) CpPipe::Workers::Instance().Start(EnvWorkers(), EnvShard());
	LOGF("[cp-pipe] on: mode=%s depth=%zu workers=%zu shard=%d signals=%d\n", EnvMode() == CpPipe::Mode::Shadow ? "shadow" : "1", m_pipe->depth, EnvWorkers(), EnvShard() ? 1 : 0, static_cast<int>(EnvSignals()));
	if (CpPipe::NoDrainEnvMode() != CpPipe::NoDrainMode::Off) { // S604
		LOGF("[cp-pipe] nodrain: mode=%s (the precise view of the pending writes decides the table and wait drains)\n", CpPipe::NoDrainModeName(CpPipe::NoDrainEnvMode()));
		std::fprintf(stderr, "[cp-pipe] nodrain: mode=%s\n", CpPipe::NoDrainModeName(CpPipe::NoDrainEnvMode()));
	}
	// S558 (KYTY_CP_PREDECODE): the workers decode the T#s of the jobs they answer; the recorder takes them only with KYTY_CP_PIPE=1 (shadow: none used).
	if (Predecode::EnvMode() != Predecode::Mode::Off) {
		LOGF("[cp-predecode] on: mode=%s pipe=%s workers=%zu sizeof(PreTex)=%zu slim=%d\n", Predecode::ModeName(Predecode::EnvMode()),
		     EnvMode() == CpPipe::Mode::Shadow ? "shadow" : "1", EnvWorkers(), sizeof(PreTex), Predecode::SlimEnabled() ? 1 : 0);
		std::fprintf(stderr, "[cp-predecode] on: mode=%s pipe=%s workers=%zu sizeof(PreTex)=%zu slim=%d\n", Predecode::ModeName(Predecode::EnvMode()),
		             EnvMode() == CpPipe::Mode::Shadow ? "shadow" : "1", EnvWorkers(), sizeof(PreTex), Predecode::SlimEnabled() ? 1 : 0);
	}
	// S549: the front thread (parked until the first stream). Not with the S479 shadow (it compares on the CP thread).
	if (CpPipe::FrontEnvMode() != CpPipe::FrontMode::Off && EnvMode() == CpPipe::Mode::On) {
		m_pipe->front        = std::make_unique<CpPipe::FrontRing>(m_pipe->depth);
		m_pipe->front_shadow = CpPipe::FrontEnvMode() == CpPipe::FrontMode::Shadow;
		std::thread([this] { PipeFrontThread(); }).detach(); // never ends: the graphics CP lives as long as the process
		LOGF("[cp-front] on: mode=%s depth=%zu numinst_lazy=%s\n", m_pipe->front_shadow ? "shadow" : "1", m_pipe->depth, NumInstLazy::ModeName(NumInstLazy::EnvMode()));
		std::fprintf(stderr, "[cp-front] on: mode=%s depth=%zu numinst_lazy=%s\n", m_pipe->front_shadow ? "shadow" : "1", m_pipe->depth,
		             NumInstLazy::ModeName(NumInstLazy::EnvMode()));
	}
}

void CommandProcessor::PipeEnqueue(CpPipe::OpClass cls, uint32_t opcode, const uint32_t* packet, uint32_t packet_dw) {
	using C = FrameTiming::Counter;
	auto& p = *m_pipe;
	if (p.front != nullptr) {
		PipeFrontEnqueue(cls, opcode, packet, packet_dw);
		return;
	}
	if (p.ring.Size() >= p.depth) {
		FrameTiming::Add(C::PipeFullN);
		PipeExecuteFront();
	}
	auto& it  = p.items[p.ring.Reserve()];
	it.cls    = cls;
	it.opcode = opcode;
	it.dw.assign(packet, packet + packet_dw);
	if (cls == CpPipe::OpClass::Draw || cls == CpPipe::OpClass::Dispatch) {
		const FrameTiming::ScopedCounterNs timer(C::PipeSnapshotNs);
		it.ctx                    = m_ctx;
		it.sh                     = m_sh_ctx;
		it.ucfg                   = m_ucfg;
		const uint32_t num_instances = NumInstances(); // resolves an inherited count now, as the unpiped path does at the draw
		it.ds                        = m_ds;
		it.ds.num_instances          = num_instances;
	}
	// S490: a worker computes the draw's programs on the snapshot (PipelineCache::GetGraphicsProgramsOffThread: a lookup in the shared program cache); the recorder joins it.
	if (cls == CpPipe::OpClass::Draw && CpPipe::IsGraphicsDraw(opcode) && EnvWorkers() != 0) {
		it.job.cache     = &m_renderer.GetPipelineCache();
		it.job.ctx_copy  = it.ctx;
		it.job.sh_copy   = it.sh;
		it.job.ucfg_copy = it.ucfg;
		it.job.ctx       = &it.job.ctx_copy;
		it.job.sh        = &it.job.sh_copy;
		it.job.ucfg      = &it.job.ucfg_copy;
		it.job.shard_key = it.sh.GetPs().ps_regs.data_addr != 0 ? it.sh.GetPs().ps_regs.data_addr : it.sh.GetVs().es_regs.data_addr;
		const uint64_t gate = p.gates.GateFor(EnvSignals(), p.ring.FrontSeq());
		if (gate != 0) {
			it.job.state.store(CpPipe::GfpJob::Held, std::memory_order_relaxed);
			p.held.push_back({&it.job, gate});
		} else {
			SubmitJob(&it.job);
		}
	}
	const uint64_t seq = p.ring.Commit();
	p.gates.NoteItem(cls, opcode, seq);
	FrameTiming::Add(C::PipeEnqN);
	FrameTiming::Add(cls == CpPipe::OpClass::Draw ? C::PipeDrawN : (cls == CpPipe::OpClass::Dispatch ? C::PipeDispatchN : C::PipeOrderedN));

	// The writes this item will do, for the WAIT_REG_MEM that follows it in the stream (S604: the shared decoder, cpPipeNoDrain.h).
	CpPipe::NoteItemWrites(p.pending, opcode, packet, packet_dw, IsAsyncComputeQueue(), seq, CpPipe::NoDrainEnvMode() != CpPipe::NoDrainMode::Off);
}

void CommandProcessor::PipeExecuteFront() {
	using C = FrameTiming::Counter;
	auto&          p   = *m_pipe;
	auto&          it  = p.items[p.ring.Front()];
	const uint64_t seq = p.ring.FrontSeq();
	const bool     timed = !CpTimersLite();
	const auto     t0  = timed ? FrameTiming::SteadyNowNs() : int64_t {0};
	p.executing        = true;

	const auto handler = g_cp_op_func[it.opcode];
	EXIT_IF(handler == nullptr);
	const bool state_item = it.cls == CpPipe::OpClass::Draw || it.cls == CpPipe::OpClass::Dispatch;
	if (state_item) {
		GetScheduler().Rebind(it.ctx, it.ucfg, it.sh);
		m_exec_sh   = &it.sh;
		m_exec_ucfg = &it.ucfg;
		std::swap(m_ds, it.ds);
	}
	{
		if (it.job.state.load(std::memory_order_relaxed) == CpPipe::GfpJob::Held) { // not released yet (cannot happen: the gate is older): never wait for a job nobody runs
			for (auto held = p.held.begin(); held != p.held.end(); ++held) {
				if (held->job == &it.job) {
					p.held.erase(held);
					break;
				}
			}
			SubmitJob(&it.job);
		}
		if (it.job.state.load(std::memory_order_relaxed) != CpPipe::GfpJob::Idle) CpPipe::CurrentJob() = &it.job;
		CpPhases::Scope handler_phase(CpPhases::Phase::Handler);
		if (CpPhases::Enabled()) {
			handler_phase.SetSub(it.cls == CpPipe::OpClass::Draw ? C::HdlDrawNs : (it.cls == CpPipe::OpClass::Dispatch ? C::HdlDispNs : C::HdlOtherNs),
			                     it.cls == CpPipe::OpClass::Draw ? C::HdlDrawN : (it.cls == CpPipe::OpClass::Dispatch ? C::HdlDispN : C::HdlOtherN));
		}
		const auto packet_dw = static_cast<uint32_t>(it.dw.size());
		if (CpSampler::Active().load(std::memory_order_relaxed)) {
			CpSampler::CpOpcode().store(it.opcode, std::memory_order_relaxed);
		}
		const bool rf_op = RfOp::Enabled();
		if (rf_op) {
			RfOp::BeginPacket(it.opcode, it.dw.data(), packet_dw);
		}
		const auto consumed  = handler(*this, it.dw[0] & ~1u, it.dw.data() + 1, packet_dw, packet_dw) + 1;
		if (rf_op) {
			RfOp::EndPacket();
		}
		if (CpSampler::Active().load(std::memory_order_relaxed)) {
			CpSampler::CpOpcode().store(CpSampler::kNoOpcode, std::memory_order_relaxed);
		}
		if (consumed != packet_dw) {
			FrameTiming::Add(C::PipeLenMismatchN);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 8) {
				LOGF("[cp-pipe] handler of opcode 0x%x consumed %u dwords of a %u dword packet\n", it.opcode, consumed, packet_dw);
			}
		}
	}
	CpPipe::CurrentJob() = nullptr;
	if (it.job.state.load(std::memory_order_relaxed) != CpPipe::GfpJob::Idle) { // the draw never asked for its programs (it was skipped before): the worker may still read the snapshot
		CpPipe::WaitDone(it.job);
		it.job.state.store(CpPipe::GfpJob::Idle, std::memory_order_relaxed);
	}
	if (state_item) {
		std::swap(m_ds, it.ds);
		m_exec_sh   = nullptr;
		m_exec_ucfg = nullptr;
		GetScheduler().Rebind(m_ctx, m_ucfg, m_sh_ctx);
	}
	p.pending.Retire(seq);
	p.ring.Pop();
	p.executing = false;
	while (!p.held.empty() && p.held.front().gate < p.ring.FrontSeq()) { // the gate has run: the job may read what it needs
		SubmitJob(p.held.front().job);
		p.held.pop_front();
	}
	if (timed) FrameTiming::Add(C::PipeExecNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
}

namespace {

// KYTY_CP_PIPE diag: what cut the windows. Key = opcode (NOP: 256 + R), 0x1f0 = guest command, 0x1f1 = wait, 0x1f2 = end of stream; dumped every 500 flips to the diag file.
std::atomic<uint64_t> g_drain_by_key[512];
std::atomic<uint64_t> g_drain_items;
std::atomic<uint64_t> g_dump_flip {0};

// S604 (KYTY_PIPE_NODRAIN=1|shadow) DIAG: per drain key, the culprit in the S479 view and in the precise view (or "avoided"); one front thread writes, the dump
// below runs on it too (atomics only so that a second CP could never tear an entry). Keys never change once set; a full table drops (counted).
struct NoDrainHistEntry {
	std::atomic<uint64_t> key {0};
	std::atomic<uint64_t> n {0};
};
constexpr uint32_t    kNoDrainHist = 256;
NoDrainHistEntry      g_nd_hist[kNoDrainHist];
std::atomic<uint64_t> g_nd_hist_dropped {0};

void NoDrainHistAdd(uint32_t drain_key, CpPipe::Culprit s479, bool avoided, CpPipe::Culprit precise) {
	const uint64_t key = CpPipe::NoDrainHistKey(drain_key, s479, avoided, precise);
	uint32_t       i   = static_cast<uint32_t>((key ^ (key >> 17u) ^ (key >> 40u)) * 0x9e3779b1u) % kNoDrainHist;
	for (uint32_t probe = 0; probe < kNoDrainHist; probe++, i = (i + 1u) % kNoDrainHist) {
		uint64_t seen = g_nd_hist[i].key.load(std::memory_order_relaxed);
		if (seen == 0 && g_nd_hist[i].key.compare_exchange_strong(seen, key, std::memory_order_relaxed)) seen = key;
		if (seen == key) {
			g_nd_hist[i].n.fetch_add(1, std::memory_order_relaxed);
			return;
		}
	}
	g_nd_hist_dropped.fetch_add(1, std::memory_order_relaxed);
}

void DumpNoDrainHistogram(uint64_t flips, uint64_t last) {
	char line[200];
	std::snprintf(line, sizeof(line), "[pipe-nodrain] flip=%" PRIu64 " since=%" PRIu64 " mode=%s dropped=%" PRIu64 "\n", flips, last,
	              CpPipe::NoDrainModeName(CpPipe::NoDrainEnvMode()), g_nd_hist_dropped.load(std::memory_order_relaxed));
	Common::DiagOut::Write(line);
	for (auto& e: g_nd_hist) {
		const uint64_t key = e.key.load(std::memory_order_relaxed);
		const uint64_t n   = key != 0 ? e.n.exchange(0, std::memory_order_relaxed) : 0;
		if (n == 0) continue;
		const auto s479_src   = static_cast<uint32_t>(key & 0xffffu);
		const auto s479_why   = CpPipe::ReasonLetter(static_cast<CpPipe::Reason>((key >> 16u) & 0x3u));
		const auto prec_src   = static_cast<uint32_t>((key >> 18u) & 0xffffu);
		const auto prec_why   = CpPipe::ReasonLetter(static_cast<CpPipe::Reason>((key >> 34u) & 0x3u));
		const bool avoided    = ((key >> 36u) & 0x1u) != 0;
		const auto drain_key  = static_cast<uint32_t>((key >> 40u) & 0x1ffu);
		char       precise[32];
		if (avoided) {
			std::snprintf(precise, sizeof(precise), "avoided");
		} else {
			std::snprintf(precise, sizeof(precise), "0x%04x/%c", prec_src, prec_why);
		}
		std::snprintf(line, sizeof(line), "[pipe-nodrain]   key=0x%x s479=0x%04x/%c precise=%s n=%" PRIu64 " per_flip=%.2f\n", drain_key, s479_src, s479_why, precise, n,
		              static_cast<double>(n) / static_cast<double>(flips - last));
		Common::DiagOut::Write(line);
	}
}

void DumpDrainHistogram() {
	const uint64_t flips = FrameTiming::GuestFlips();
	uint64_t last = g_dump_flip.load(std::memory_order_relaxed);
	if (flips < last + 500u || !g_dump_flip.compare_exchange_strong(last, flips)) return;
	char line[160];
	std::snprintf(line, sizeof(line), "[pipe-drain] flip=%" PRIu64 " since=%" PRIu64 " items=%" PRIu64 "\n", flips, last, g_drain_items.exchange(0));
	Common::DiagOut::Write(line);
	for (uint32_t key = 0; key < 512; key++) {
		const uint64_t n = g_drain_by_key[key].exchange(0);
		if (n == 0) continue;
		std::snprintf(line, sizeof(line), "[pipe-drain]   key=0x%x n=%" PRIu64 " per_flip=%.2f\n", key, n, static_cast<double>(n) / static_cast<double>(flips - last));
		Common::DiagOut::Write(line);
	}
	if (CpPipe::NoDrainEnvMode() != CpPipe::NoDrainMode::Off) DumpNoDrainHistogram(flips, last);
}

uint64_t ReadGuestValue(uint64_t address, uint64_t size) {
	return size == 8u ? *reinterpret_cast<const volatile uint64_t*>(address) : *reinterpret_cast<const volatile uint32_t*>(address);
}

FrameTiming::Counter NoDrainTableCounter(uint32_t opcode) {
	switch (opcode) {
		case Pm4::IT_SET_CONTEXT_REG_INDIRECT: return FrameTiming::Counter::NoDrainCxN;
		case Pm4::IT_SET_UCONFIG_REG_INDIRECT: return FrameTiming::Counter::NoDrainUcN;
		default: return FrameTiming::Counter::NoDrainShN;
	}
}

} // namespace

void CommandProcessor::PipeDrain(unsigned reason, uint32_t key) {
	using C = FrameTiming::Counter;
	if (m_pipe != nullptr && m_pipe->front != nullptr) { // S549: only the front drains (it waits for the recorder); a handler the recorder runs has nothing ahead of it
		if (OnFrontThread()) PipeFrontDrain(reason, key);
		return;
	}
	if (m_pipe == nullptr || m_pipe->executing || m_pipe->ring.Empty()) return;
	FrameTiming::Add(reason == 0 ? C::PipeDrainClsN : (reason == 1 ? C::PipeDrainWaitN : C::PipeDrainEndN));
	g_drain_by_key[key & 511u].fetch_add(1, std::memory_order_relaxed);
	g_drain_items.fetch_add(m_pipe->ring.Size(), std::memory_order_relaxed);
	DumpDrainHistogram();
	while (!m_pipe->ring.Empty()) PipeExecuteFront();
	m_pipe->pending.Clear();
}

bool CommandProcessor::PipeOverlaps(uint64_t address, uint64_t size) const {
	if (m_pipe != nullptr && m_pipe->front != nullptr) m_pipe->pending.RetireUpTo(m_pipe->front->Done()); // S549: the front's map, retired up to the recorder's progress
	return m_pipe != nullptr && m_pipe->pending.Overlaps(address, size);
}

// The table of a SET_*_REG_INDIRECT is read at parse time: a write still in the FIFO that touches it has to run first. S604 (KYTY_PIPE_NODRAIN): which view decides.
void CommandProcessor::PipeTableCheck(uint32_t opcode, const uint32_t* payload) {
	using C = FrameTiming::Counter;
	uint64_t table = 0, table_size = 0;
	if (!CpPipe::IndirectTableRange(payload, &table, &table_size)) return;
	const auto mode = CpPipe::NoDrainEnvMode();
	if (mode == CpPipe::NoDrainMode::Off) {
		if (PipeOverlaps(table, table_size)) PipeDrain(0, opcode);
		return;
	}
	auto& p = *m_pipe;
	if (p.front != nullptr) {
		if (!OnFrontThread()) return; // as PipeDrain: only the front drains
		p.pending.RetireUpTo(p.front->Done());
	} else if (p.executing) {
		return;
	}
	CpPipe::Culprit s479, precise;
	const bool      s479_overlap    = p.pending.OverlapsS479(table, table_size, &s479);
	const bool      precise_overlap = p.pending.OverlapsPrecise(table, table_size, &precise);
	if (!s479_overlap) {
		if (precise_overlap) { // the precise view is never wider than the S479 one (tests); counted, and drained to be safe
			FrameTiming::Add(C::NoDrainMonoNeN);
			PipeDrain(0, opcode);
		}
		return;
	}
	NoDrainHistAdd(opcode, s479, !precise_overlap, precise);
	if (precise_overlap) {
		FrameTiming::Add(C::NoDrainKeptN);
		PipeDrain(0, opcode);
		return;
	}
	FrameTiming::Add(NoDrainTableCounter(opcode));
	if (mode == CpPipe::NoDrainMode::On) return; // the handler reads the table now: no write in the FIFO touches it
	// Shadow: the bytes the handler would have read without the drain against the bytes it reads after it.
	const uint64_t before = CpPipe::HashBytes(reinterpret_cast<const void*>(table), table_size);
	PipeDrain(0, opcode);
	const uint64_t after = CpPipe::HashBytes(reinterpret_cast<const void*>(table), table_size);
	FrameTiming::Add(C::NoDrainShadowN);
	if (before != after) {
		FrameTiming::Add(C::NoDrainNeN);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 8) {
			LOGF("[pipe-nodrain] shadow: the table of opcode 0x%x at 0x%016" PRIx64 " (%" PRIu64 " bytes) changed across the drain (S479 culprit 0x%04x)\n", opcode, table,
			     table_size, s479.src);
		}
	}
}

bool CommandProcessor::PipeWaitVirtual(uint64_t address, uint64_t size, uint32_t func, uint64_t ref, uint64_t mask, bool* satisfied) {
	using C  = FrameTiming::Counter;
	auto&    p = *m_pipe;
	if (p.front != nullptr) {
		if (!OnFrontThread()) return false; // a wait the recorder runs (none today): nothing ahead of it
		p.pending.RetireUpTo(p.front->Done());
	} else if (p.executing) {
		return false;
	}
	const auto      mode  = CpPipe::NoDrainEnvMode();
	uint64_t        value = 0;
	CpPipe::Culprit s479;
	const auto      verdict = p.pending.Resolve(address, size, &value, mode != CpPipe::NoDrainMode::Off ? &s479 : nullptr);
	if (!CpPipe::WaitNeedsDrain(verdict, value, ref, mask, func)) {
		if (mode != CpPipe::NoDrainMode::Off) { // the precise view is never more careful than the S479 one (tests); counted
			uint64_t   precise_value   = 0;
			const auto precise_verdict = p.pending.ResolvePrecise(address, size, &precise_value);
			if (CpPipe::WaitNeedsDrain(precise_verdict, precise_value, ref, mask, func)) FrameTiming::Add(C::NoDrainMonoNeN);
		}
		if (verdict == CpPipe::PendingWrites::Verdict::NoPending) return false;
		FrameTiming::Add(C::PipeWaitVirtN);
		*satisfied = true;
		return true;
	}
	if (mode == CpPipe::NoDrainMode::Off) {
		PipeDrain(1, 0x1f1);
		return false;
	}
	if (verdict == CpPipe::PendingWrites::Verdict::Value) s479.reason = CpPipe::Reason::Fail;
	uint64_t        precise_value = 0;
	CpPipe::Culprit precise;
	const auto      precise_verdict = p.pending.ResolvePrecise(address, size, &precise_value, &precise);
	const bool      precise_drain   = CpPipe::WaitNeedsDrain(precise_verdict, precise_value, ref, mask, func);
	if (precise_drain && precise_verdict == CpPipe::PendingWrites::Verdict::Value) precise.reason = CpPipe::Reason::Fail;
	NoDrainHistAdd(0x1f1, s479, !precise_drain, precise);
	if (precise_drain) {
		FrameTiming::Add(C::NoDrainKeptN);
		PipeDrain(1, 0x1f1);
		return false;
	}
	FrameTiming::Add(C::NoDrainWaitN);
	if (mode == CpPipe::NoDrainMode::On) {
		if (precise_verdict == CpPipe::PendingWrites::Verdict::NoPending) return false; // the handler reads guest memory: no write in the FIFO touches it
		FrameTiming::Add(C::PipeWaitVirtN);
		*satisfied = true;
		return true;
	}
	// Shadow: what the wait would have decided without the drain against what it decides after it (the handler then reads the memory as before).
	const uint64_t before = ReadGuestValue(address, size);
	PipeDrain(1, 0x1f1);
	const uint64_t after = ReadGuestValue(address, size);
	FrameTiming::Add(C::NoDrainShadowN);
	if (!CpPipe::ShadowWaitAgrees(precise_verdict, precise_value, before, after, ref, mask, func)) {
		FrameTiming::Add(C::NoDrainNeN);
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 8) {
			LOGF("[pipe-nodrain] shadow: the wait on 0x%016" PRIx64 " (%" PRIu64 " bytes, func %u ref 0x%" PRIx64 " mask 0x%" PRIx64 ") decides otherwise after the drain:"
			     " precise %s 0x%" PRIx64 ", before 0x%" PRIx64 ", after 0x%" PRIx64 " (S479 culprit 0x%04x)\n",
			     address, size, func, ref, mask, precise_verdict == CpPipe::PendingWrites::Verdict::Value ? "value" : "memory", precise_value, before, after, s479.src);
		}
	}
	return false;
}


// ---------------------------------------------------------------------------------------------------------------------------------------------------------------
// S549 (KYTY_CP_FRONT_THREAD=1|shadow): the front on its own thread. Thread_Gpu (the recorder) executes the items; the front parses, runs the register handlers, takes
// the snapshots, submits the worker jobs and answers the waits. Who owns what: notes in artifacts/S549/notes.md. In short: the CP's registers and draw state belong
// to the front; the recorder reads its item's copies (m_exec_*, Ds()) and touches the live state only in a Sync item, while the front waits (HandoffScope).

namespace {

thread_local const CommandProcessor* t_front_cp = nullptr; // the CP whose front this thread is

enum : uint32_t { kSyncNone = 0, kSyncHandler = 1, kSyncCommands = 2, kSyncEnd = 3 };
constexpr uint64_t kReleaseWindow = 24;

uint64_t SnapshotHash(const PipeItem& it) {
	uint64_t h = CpPipe::HashBytes(&it.ctx, sizeof(it.ctx));
	h          = CpPipe::HashBytes(&it.sh, sizeof(it.sh), h);
	h          = CpPipe::HashBytes(&it.ucfg, sizeof(it.ucfg), h);
	h          = CpPipe::HashBytes(&it.ds, sizeof(it.ds), h);
	return CpPipe::HashBytes(it.dw.data(), it.dw.size() * sizeof(uint32_t), h);
}

} // namespace

bool CommandProcessor::PipeFrontOn() const {
	return m_pipe != nullptr && m_pipe->front != nullptr;
}

bool CommandProcessor::OnFrontThread() const {
	return t_front_cp == this;
}

bool CommandProcessor::PipeFrontCommandsPending() const {
	return m_pipe->front_gpu != nullptr && m_pipe->front_gpu->m_pending_commands.load(std::memory_order_acquire) != 0;
}

void CommandProcessor::PipeFrontThread() {
	t_front_cp = this;
	Common::CpTuning::ApplyToFrontThread(); // named "CpFront" for the thread-cpu snapshot; off the CP's core when KYTY_CP_AFFINITY is set
	FrameTiming::SetThreadBuffered(true);   // flushed at the end of every stream, before the End item
	auto&    p    = *m_pipe;
	uint32_t seen = 0;
	for (;;) {
		uint32_t go = p.front_go.load(std::memory_order_acquire);
		for (uint32_t i = 0; go == seen && i < 8000u; i++) { // streams come in bursts: ~100 us of spinning before parking
			CpPipe::FrontCpuRelax();
			go = p.front_go.load(std::memory_order_acquire);
		}
		while (go == seen) {
			p.front_go.wait(seen, std::memory_order_acquire);
			go = p.front_go.load(std::memory_order_acquire);
		}
		seen          = go;
		const auto t0 = FrameTiming::SteadyNowNs();
		FrontParse(*p.front_exec);
		FrameTiming::Add(FrameTiming::Counter::FrontRunNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
		FrameTiming::FlushThreadCounters();
		PipeFrontEnd();
	}
}

void CommandProcessor::PipeFrontEnqueue(CpPipe::OpClass cls, uint32_t opcode, const uint32_t* packet, uint32_t packet_dw) {
	using C     = FrameTiming::Counter;
	auto& p     = *m_pipe;
	auto& ring  = *p.front;
	{
		const auto t0 = FrameTiming::SteadyNowNs();
		if (ring.WaitSlot()) {
			FrameTiming::Add(C::PipeFullN);
			FrameTiming::Add(C::FrontFullWaitN);
			FrameTiming::Add(C::FrontFullWaitNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
		}
	}
	const uint64_t seq = ring.NextSeq();
	auto&          it  = p.items[ring.Slot(seq)];
	it.cls             = cls;
	it.opcode          = opcode;
	it.sync_kind       = kSyncNone;
	it.gate            = 0;
	it.dw.assign(packet, packet + packet_dw);
	if (cls == CpPipe::OpClass::Draw || cls == CpPipe::OpClass::Dispatch) {
		const FrameTiming::ScopedCounterNs timer(C::PipeSnapshotNs);
		it.ctx                       = m_ctx;
		it.sh                        = m_sh_ctx;
		it.ucfg                      = m_ucfg;
		SnapshotNumInstances(it.ds); // S577: 0 resolves an inherited count now; KYTY_NUMINST_LAZY=1 leaves it to the recorder
	}
	if (cls == CpPipe::OpClass::Draw && CpPipe::IsGraphicsDraw(opcode) && EnvWorkers() != 0) {
		it.job.cache     = &m_renderer.GetPipelineCache();
		it.job.ctx_copy  = it.ctx;
		it.job.sh_copy   = it.sh;
		it.job.ucfg_copy = it.ucfg;
		it.job.ctx       = &it.job.ctx_copy;
		it.job.sh        = &it.job.sh_copy;
		it.job.ucfg      = &it.job.ucfg_copy;
		it.job.shard_key = it.sh.GetPs().ps_regs.data_addr != 0 ? it.sh.GetPs().ps_regs.data_addr : it.sh.GetVs().es_regs.data_addr;
		const uint64_t gate = p.gates.GateFor(EnvSignals(), ring.Done() + 1); // a stale `done` only holds the job longer
		if (gate != 0) {
			it.gate = gate;
			it.job.state.store(CpPipe::GfpJob::Held, std::memory_order_relaxed); // the recorder releases it (PipeFrontRelease) or submits it when it executes the item
		} else {
			SubmitJob(&it.job);
		}
	}
	if (p.front_shadow) it.shadow_hash = SnapshotHash(it);
	p.pending.RetireUpTo(ring.Done());
	ring.Commit();
	p.gates.NoteItem(cls, opcode, seq);
	FrameTiming::Add(C::PipeEnqN);
	FrameTiming::Add(cls == CpPipe::OpClass::Draw ? C::PipeDrawN : (cls == CpPipe::OpClass::Dispatch ? C::PipeDispatchN : C::PipeOrderedN));

	// The writes this item will do, for the WAIT_REG_MEM that follows it in the stream (as PipeEnqueue).
	CpPipe::NoteItemWrites(p.pending, opcode, packet, packet_dw, IsAsyncComputeQueue(), seq, CpPipe::NoDrainEnvMode() != CpPipe::NoDrainMode::Off);
}

// S577 (KYTY_NUMINST_LAZY, numInstLazy.h). Off: NumInstances() as before (the read of an indirect draw's record faults when the GPU wrote it).
// 1: the item carries {address, epoch}, the recorder reads at the first item that inherits the count. shadow: both.
void CommandProcessor::SnapshotNumInstances(CpPipe::CpDrawState& ds) {
	const auto snap     = NumInstLazy::TakeSnapshot(NumInstLazy::EnvMode(), m_ni_front, m_ds.num_instances, [this] { return NumInstances(); });
	ds                  = m_ds;
	ds.num_instances    = snap.value;
	ds.ni_addr          = snap.addr;
	ds.ni_epoch         = snap.epoch;
	if (snap.skipped) FrameTiming::Add(FrameTiming::Counter::NumInstLazySkipN);
}

uint32_t CommandProcessor::PipeFrontSync(uint32_t kind, uint32_t opcode, const uint32_t* packet, uint32_t remaining_dw, uint32_t total_dw) {
	using C       = FrameTiming::Counter;
	auto&      p  = *m_pipe;
	auto&      ring = *p.front;
	const auto t0 = FrameTiming::SteadyNowNs();
	if (ring.Done() < ring.LastCommitted()) { // the drain the unthreaded path does before the handler (counted alike)
		FrameTiming::Add(C::PipeDrainClsN);
		const uint32_t key = kind == kSyncCommands ? 0x1f0u : (opcode == Pm4::IT_NOP && packet != nullptr ? 256u + KYTY_PM4_R(packet[0]) : opcode);
		g_drain_by_key[key & 511u].fetch_add(1, std::memory_order_relaxed);
		g_drain_items.fetch_add(ring.LastCommitted() - ring.Done(), std::memory_order_relaxed);
		DumpDrainHistogram();
	}
	ring.WaitSlot();
	const uint64_t seq = ring.NextSeq();
	auto&          it  = p.items[ring.Slot(seq)];
	it.cls             = CpPipe::OpClass::Drain;
	it.opcode          = opcode;
	it.sync_kind       = kind;
	it.gate            = 0;
	it.live_packet     = packet;
	it.remaining_dw    = remaining_dw;
	it.total_dw        = total_dw;
	it.result_dw       = 0;
	ring.Commit();
	ring.WaitDone(seq);
	p.pending.Clear(); // every item before it has run
	FrameTiming::Add(C::FrontSyncN);
	FrameTiming::Add(C::FrontSyncNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
	return it.result_dw;
}

void CommandProcessor::PipeFrontDrain(unsigned reason, uint32_t key) {
	using C    = FrameTiming::Counter;
	auto& p    = *m_pipe;
	auto& ring = *p.front;
	if (reason == 2u) return; // the end of the stream: the End item follows, Thread_Gpu returns after it
	if (ring.Done() >= ring.LastCommitted()) return;
	FrameTiming::Add(reason == 0 ? C::PipeDrainClsN : C::PipeDrainWaitN);
	g_drain_by_key[key & 511u].fetch_add(1, std::memory_order_relaxed);
	g_drain_items.fetch_add(ring.LastCommitted() - ring.Done(), std::memory_order_relaxed);
	DumpDrainHistogram();
	const auto t0 = FrameTiming::SteadyNowNs();
	ring.WaitDone(ring.LastCommitted());
	p.pending.Clear();
	FrameTiming::Add(C::FrontDrainN);
	FrameTiming::Add(C::FrontDrainNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
}

void CommandProcessor::PipeFrontEnd() {
	auto& p    = *m_pipe;
	auto& ring = *p.front;
	if (ring.Done() < ring.LastCommitted()) FrameTiming::Add(FrameTiming::Counter::PipeDrainEndN);
	ring.WaitSlot();
	auto& it       = p.items[ring.Slot(ring.NextSeq())];
	it.cls         = CpPipe::OpClass::Drain;
	it.sync_kind   = kSyncEnd;
	it.gate        = 0;
	it.live_packet = nullptr;
	p.pending.Clear(); // Thread_Gpu returns after the End item: nothing is pending at the next stream
	ring.Commit();     // the last write of this stream by the front: the execution belongs to Thread_Gpu again after the End item
}

void CommandProcessor::PipeFrontRun(Pm4Execution& execution, GuestGpu* gpu) {
	using C    = FrameTiming::Counter;
	auto& p    = *m_pipe;
	auto& ring = *p.front;
	p.front_exec = &execution;
	p.front_gpu  = gpu;
	p.front_go.fetch_add(1, std::memory_order_release);
	p.front_go.notify_one();
	const CpPipe::RecorderScope recorder; // the CP state is the front's from here to the End item
	const auto idle = [&p] {
		// A guest thread's command (a fault readback of the front thread itself, an unmap) while nothing is left to record: the FIFO is empty, so it sees every item.
		if (p.front_gpu != nullptr && p.front_gpu->m_pending_commands.load(std::memory_order_acquire) != 0) {
			FrameTiming::Add(C::FrontIdleCmdN);
			p.front_gpu->ProcessCommands(); // the gate stays shut: a command that touched the front's state would stop the emulator, not race
		}
	};
	for (uint64_t seq = ring.Done() + 1;; seq++) {
		if (!ring.HasItem(seq)) { // a real wait (the cached tail was refreshed first)
			const auto t0 = FrameTiming::SteadyNowNs();
			ring.WaitItem(seq, idle);
			FrameTiming::Add(C::FrontRecWaitNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
		}
		const bool end = p.items[ring.Slot(seq)].sync_kind == kSyncEnd;
		if (!end) PipeFrontExecute(seq);
		ring.MarkDone(seq);
		if (end) break;
		PipeFrontRelease(seq);
	}
	GetScheduler().Rebind(m_ctx, m_ucfg, m_sh_ctx);
	p.front_exec = nullptr;
}

// After item `done`: start the held jobs whose gate has run. The items it looks at are committed and not executed yet, so the front does not write them.
void CommandProcessor::PipeFrontRelease(uint64_t done) {
	auto& p    = *m_pipe;
	auto& ring = *p.front;
	if (p.rel_cursor <= done) p.rel_cursor = done + 1;
	// Only the next items: their slots were written long ago (no cross-core loads of what the front writes now); a held job further ahead has a later gate anyway
	// or is released when the window reaches it, still ~24 items before its draw.
	const uint64_t window = done + kReleaseWindow;
	const uint64_t tail   = ring.TailSeen() < window ? ring.TailSeen() : window;
	for (; p.rel_cursor <= tail; p.rel_cursor++) {
		auto& it = p.items[ring.Slot(p.rel_cursor)];
		if (it.sync_kind != kSyncNone || it.job.state.load(std::memory_order_acquire) != CpPipe::GfpJob::Held) continue;
		if (it.gate > done) break; // gates grow with the items
		uint8_t held = CpPipe::GfpJob::Held;
		if (it.job.state.compare_exchange_strong(held, CpPipe::GfpJob::Queued, std::memory_order_acq_rel)) SubmitJob(&it.job);
	}
}

void CommandProcessor::PipeFrontExecute(uint64_t seq) {
	using C        = FrameTiming::Counter;
	auto&      p   = *m_pipe;
	auto&      it  = p.items[p.front->Slot(seq)];
	const bool timed = !CpTimersLite();
	const auto t0  = timed ? FrameTiming::SteadyNowNs() : int64_t {0};
	FrameTiming::Add(C::FrontItemsN);
	if (it.sync_kind != kSyncNone) { // a hand-over: the front waits, the live state is ours
		const CpPipe::HandoffScope handoff;
		GetScheduler().Rebind(m_ctx, m_ucfg, m_sh_ctx);
		if (it.sync_kind == kSyncCommands) {
			if (p.front_gpu != nullptr) {
				CpPhases::Scope phase(CpPhases::Phase::Cmds);
				p.front_gpu->ProcessCommands();
			}
		} else {
			const auto handler = g_cp_op_func[it.opcode];
			EXIT_IF(handler == nullptr || it.live_packet == nullptr);
			CpPhases::Scope handler_phase(CpPhases::Phase::Handler);
			it.result_dw = handler(*this, it.live_packet[0] & ~1u, it.live_packet + 1, it.remaining_dw, it.total_dw) + 1;
		}
		return;
	}
	if (p.front_shadow) {
		FrameTiming::Add(C::FrontShadowN);
		if (SnapshotHash(it) != it.shadow_hash) {
			FrameTiming::Add(C::FrontShadowNeN);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 8) LOGF("[cp-front] shadow: the snapshot of item %" PRIu64 " (opcode 0x%x) changed after the front committed it\n", seq, it.opcode);
		}
	}
	p.executing           = true;
	const bool state_item = it.cls == CpPipe::OpClass::Draw || it.cls == CpPipe::OpClass::Dispatch;
	if (state_item) {
		GetScheduler().Rebind(it.ctx, it.ucfg, it.sh);
		m_exec_sh   = &it.sh;
		m_exec_ucfg = &it.ucfg;
		m_exec_ds   = &it.ds;
	}
	{
		uint8_t held = CpPipe::GfpJob::Held;
		if (it.job.state.load(std::memory_order_acquire) == CpPipe::GfpJob::Held &&
		    it.job.state.compare_exchange_strong(held, CpPipe::GfpJob::Queued, std::memory_order_acq_rel)) {
			SubmitJob(&it.job); // its gate is older than the item and has run
		}
		if (it.job.state.load(std::memory_order_relaxed) != CpPipe::GfpJob::Idle) CpPipe::CurrentJob() = &it.job;
		CpPhases::Scope handler_phase(CpPhases::Phase::Handler);
		if (CpPhases::Enabled()) {
			handler_phase.SetSub(it.cls == CpPipe::OpClass::Draw ? C::HdlDrawNs : (it.cls == CpPipe::OpClass::Dispatch ? C::HdlDispNs : C::HdlOtherNs),
			                     it.cls == CpPipe::OpClass::Draw ? C::HdlDrawN : (it.cls == CpPipe::OpClass::Dispatch ? C::HdlDispN : C::HdlOtherN));
		}
		const auto handler   = g_cp_op_func[it.opcode];
		EXIT_IF(handler == nullptr);
		const auto packet_dw = static_cast<uint32_t>(it.dw.size());
		if (CpSampler::Active().load(std::memory_order_relaxed)) CpSampler::CpOpcode().store(it.opcode, std::memory_order_relaxed);
		const bool rf_op = RfOp::Enabled();
		if (rf_op) RfOp::BeginPacket(it.opcode, it.dw.data(), packet_dw);
		const auto consumed = handler(*this, it.dw[0] & ~1u, it.dw.data() + 1, packet_dw, packet_dw) + 1;
		if (rf_op) RfOp::EndPacket();
		if (CpSampler::Active().load(std::memory_order_relaxed)) CpSampler::CpOpcode().store(CpSampler::kNoOpcode, std::memory_order_relaxed);
		if (consumed != packet_dw) {
			FrameTiming::Add(C::PipeLenMismatchN);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 8) LOGF("[cp-pipe] handler of opcode 0x%x consumed %u dwords of a %u dword packet\n", it.opcode, consumed, packet_dw);
		}
	}
	CpPipe::CurrentJob() = nullptr;
	if (it.job.state.load(std::memory_order_relaxed) != CpPipe::GfpJob::Idle) { // the draw never asked for its programs: the worker may still read the snapshot
		CpPipe::WaitDone(it.job);
		it.job.state.store(CpPipe::GfpJob::Idle, std::memory_order_relaxed);
	}
	if (state_item) {
		m_exec_sh   = nullptr;
		m_exec_ucfg = nullptr;
		m_exec_ds   = nullptr;
		GetScheduler().Rebind(m_ctx, m_ucfg, m_sh_ctx); // pointers only: nothing outside a draw or dispatch reads the bound registers
	}
	p.executing = false;
	if (timed) FrameTiming::Add(C::PipeExecNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
}

} // namespace Libs::Graphics
