// The FIFO between the PM4 parser and the handlers that draw and signal. Both run on the CP thread.
// A graphics draw is not executed where it is parsed, so another thread can look up its programs
// in between. Register writes still run at parse time. Signals, barriers, CPU writes and flips are
// copied and run later, in stream order. WAIT_REG_MEM is answered from the writes still in the
// FIFO, or the FIFO is drained first. Anything else drains first. The end of every stream and
// every suspend drains, so what leaves CommandProcessor::Process is what the old code left.

#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/command_processor/cpPipe.h"
#include "graphics/guest_gpu/command_processor/cpPipeJob.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "kernel/memory.h"

#include <condition_variable>
#include <cstring>
#include <deque>
#include <mutex>
#include <thread>
#if defined(_MSC_VER) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace Libs::Graphics {

namespace {

constexpr size_t kWorkers = 3;
constexpr size_t kDepth   = 64;

using CpPipe::Enabled;

inline void CpuRelax() {
#if defined(_MSC_VER) || defined(__x86_64__)
	_mm_pause();
#else
	std::this_thread::yield();
#endif
}

} // namespace

namespace CpPipe {

GfpJob*& CurrentJob() {
	static thread_local GfpJob* job = nullptr;
	return job;
}

namespace {

class Workers {
public:
	static Workers& Instance() {
		static Workers* pool = new Workers;
		return *pool;
	}

	void Start() {
		const std::lock_guard lock(m_start_mutex);
		if (m_started) return;
		m_started = true;
		for (size_t i = 0; i < kWorkers; i++) {
			m_threads.emplace_back([this] { Run(); });
			m_threads.back().detach();
		}
	}

	void Submit(GfpJob* job) {
		{
			const std::lock_guard lock(m_mutex);
			m_jobs.push_back(job);
		}
		m_ready.notify_one();
	}

private:
	Workers() = default;

	void Run() {
		for (;;) {
			GfpJob* job = nullptr;
			{
				std::unique_lock lock(m_mutex);
				m_ready.wait(lock, [&] { return !m_jobs.empty(); });
				job = m_jobs.front();
				m_jobs.pop_front();
			}
			RunGfpJob(*job);
			job->state.store(GfpJob::Done, std::memory_order_release);
		}
	}

	std::mutex              m_start_mutex;
	std::mutex              m_mutex;
	std::condition_variable m_ready;
	std::deque<GfpJob*>     m_jobs;
	std::vector<std::thread> m_threads;
	bool                    m_started = false;
};

void WaitDone(GfpJob& job) {
	if (job.state.load(std::memory_order_acquire) == GfpJob::Done) return;
	for (uint32_t i = 0; job.state.load(std::memory_order_acquire) != GfpJob::Done; i++) {
		CpuRelax();
		if ((i & 255u) == 255u) std::this_thread::yield();
	}
}

void SubmitJob(GfpJob* job) {
	job->state.store(GfpJob::Queued, std::memory_order_relaxed);
	Workers::Instance().Submit(job);
}

} // namespace

void StartWorkers() { Workers::Instance().Start(); }
void EnqueueJob(GfpJob* job) { SubmitJob(job); }
void WaitForJob(GfpJob& job) { WaitDone(job); }

bool JoinJob(GfpJob& job) {
	if (job.state.load(std::memory_order_acquire) == GfpJob::Idle) return false;
	WaitDone(job);
	if (!job.answered) return false;
	if (!job.pages.empty() &&
	    Libs::LibKernel::Memory::BackingWrittenSince(job.write_seq, job.pages.data(), job.pages.size())) {
		return false;
	}
	return true;
}

} // namespace CpPipe

// One deferred packet. Draws and dispatches carry the state they were parsed with.
struct PipeItem {
	CpPipe::OpClass       cls    = CpPipe::OpClass::Ordered;
	uint32_t              opcode = 0;
	std::vector<uint32_t> dw;
	HW::Context           ctx;
	HW::Shader            sh;
	HW::UserConfig        ucfg;
	uint32_t              index_type_and_size    = 0;
	uint32_t              index_buffer_size      = 0;
	uint32_t              num_instances          = 1;
	uint64_t              num_instances_addr     = 0;
	uint64_t              index_base_addr        = 0;
	uint64_t              draw_indirect_base     = 0;
	uint64_t              dispatch_indirect_base = 0;
	CpPipe::GfpJob        job;
};

struct CommandProcessor::PipeState {
	explicit PipeState(size_t depth) : ring(depth), depth(depth), items(depth) {}

	CpPipe::Ring          ring;
	size_t                depth;
	std::vector<PipeItem> items;
	CpPipe::PendingWrites pending;
	CpPipe::GateTracker   gates;
	struct Held {
		CpPipe::GfpJob* job  = nullptr;
		uint64_t        gate = 0;
	};
	std::deque<Held> held;
	bool             executing = false;
};

CommandProcessor::CommandProcessor(RenderContext& renderer, int interrupt_event_id)
    : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::PipeInit() {
	if (m_pipe != nullptr || IsAsyncComputeQueue() || !CpPipe::Enabled()) return;
	m_pipe = std::make_unique<PipeState>(kDepth);
	CpPipe::StartWorkers();
	Libs::LibKernel::Memory::SetBackingWriteLogEnabled(true);
	ShaderRecompiler::IR::SetSrtWorkerLoad([](uint64_t address, uint32_t* value) {
		return Libs::LibKernel::Memory::TryReadBackingWorker(address, value, sizeof(uint32_t));
	});
	LOGF("cp pipe: on (depth %zu, workers %zu)\n", kDepth, kWorkers);
}

void CommandProcessor::PipeEnqueue(CpPipe::OpClass cls, uint32_t opcode, const uint32_t* packet, uint32_t packet_dw) {
	auto& p = *m_pipe;
	if (p.ring.Full()) PipeExecuteFront();
	auto& it  = p.items[p.ring.Reserve()];
	it.cls    = cls;
	it.opcode = opcode;
	it.dw.assign(packet, packet + packet_dw);
	it.job.state.store(CpPipe::GfpJob::Idle, std::memory_order_relaxed);
	it.job.answered = false;
	it.job.pages.clear();
	if (cls == CpPipe::OpClass::Draw || cls == CpPipe::OpClass::Dispatch) {
		it.ctx                     = m_ctx;
		it.sh                      = m_sh_ctx;
		it.ucfg                    = m_ucfg;
		it.index_type_and_size     = m_index_type_and_size;
		it.index_buffer_size       = m_index_buffer_size;
		it.num_instances           = NumInstances();
		it.num_instances_addr      = 0;
		it.index_base_addr         = m_index_base_addr;
		it.draw_indirect_base      = m_draw_indirect_args_base_addr;
		it.dispatch_indirect_base  = m_dispatch_indirect_args_base_addr;
	}
	if (cls == CpPipe::OpClass::Draw && CpPipe::IsGraphicsDraw(opcode)) {
		it.job.cache     = &m_renderer.GetPipelineCache();
		it.job.ctx_copy  = it.ctx;
		it.job.sh_copy   = it.sh;
		it.job.ucfg_copy = it.ucfg;
		it.job.ctx       = &it.job.ctx_copy;
		it.job.sh        = &it.job.sh_copy;
		it.job.ucfg      = &it.job.ucfg_copy;
		const uint64_t gate = p.gates.GateFor(CpPipe::SignalPolicy::Fifo, p.ring.FrontSeq());
		if (gate != 0) {
			it.job.state.store(CpPipe::GfpJob::Held, std::memory_order_relaxed);
			p.held.push_back({&it.job, gate});
		} else {
			CpPipe::EnqueueJob(&it.job);
		}
	}
	const uint64_t seq = p.ring.Commit();
	p.gates.NoteItem(cls, opcode, seq);

	constexpr uint64_t kEverything = ~uint64_t {0} >> 1u;
	const auto*        payload     = packet + 1;
	const auto         r           = opcode == Pm4::IT_NOP ? KYTY_PM4_R(packet[0]) : 0u;
	const bool         async       = IsAsyncComputeQueue();
	uint64_t           address     = 0;
	auto               note_signal = [&](const CpPipe::SignalWrite& w) {
		if (w.writes) p.pending.Note(address, w.size, w.known, w.value, seq);
	};
	switch (opcode) {
		case Pm4::IT_NOP:
			if (r == Pm4::R_FLIP) p.pending.Note(0, kEverything, false, 0, seq);
			break;
		case Pm4::IT_RELEASE_MEM:
			if (packet_dw >= 8u) note_signal(CpPipe::DecodeReleaseMem(payload, async, &address));
			break;
		case Pm4::IT_EVENT_WRITE:
			if (packet_dw > 3u) p.pending.Note(0, kEverything, false, 0, seq);
			break;
		case Pm4::IT_WRITE_DATA:
			if (packet_dw >= 5u) {
				const auto w = CpPipe::DecodeWriteData(payload, packet_dw);
				p.pending.Note(w.address, w.size, false, 0, seq);
			}
			break;
		case Pm4::IT_DMA_DATA:
			if (packet_dw >= 7u) {
				const auto w = CpPipe::DecodeDmaData(payload);
				p.pending.Note(w.address, w.size, false, 0, seq);
			}
			break;
		case Pm4::IT_COPY_DATA:
			if (packet_dw >= 6u) {
				const auto w = CpPipe::DecodeCopyData(payload);
				p.pending.Note(w.address, w.size, false, 0, seq);
			}
			break;
		default: break;
	}
}

void CommandProcessor::PipeExecuteFront() {
	auto&          p   = *m_pipe;
	auto&          it  = p.items[p.ring.Front()];
	const uint64_t seq = p.ring.FrontSeq();
	p.executing        = true;

	const auto handler = g_cp_op_func[it.opcode];
	EXIT_IF(handler == nullptr);
	const bool state_item = it.cls == CpPipe::OpClass::Draw || it.cls == CpPipe::OpClass::Dispatch;
	if (state_item) {
		GetScheduler().Rebind(it.ctx, it.ucfg, it.sh);
		m_exec_sh   = &it.sh;
		m_exec_ucfg = &it.ucfg;
		std::swap(m_index_type_and_size, it.index_type_and_size);
		std::swap(m_index_buffer_size, it.index_buffer_size);
		std::swap(m_num_instances, it.num_instances);
		std::swap(m_num_instances_addr, it.num_instances_addr);
		std::swap(m_index_base_addr, it.index_base_addr);
		std::swap(m_draw_indirect_args_base_addr, it.draw_indirect_base);
		std::swap(m_dispatch_indirect_args_base_addr, it.dispatch_indirect_base);
	}
	if (it.job.state.load(std::memory_order_relaxed) == CpPipe::GfpJob::Held) {
		for (auto held = p.held.begin(); held != p.held.end(); ++held) {
			if (held->job == &it.job) {
				p.held.erase(held);
				break;
			}
		}
		CpPipe::EnqueueJob(&it.job);
	}
	if (it.job.state.load(std::memory_order_relaxed) != CpPipe::GfpJob::Idle) CpPipe::CurrentJob() = &it.job;
	const auto packet_dw = static_cast<uint32_t>(it.dw.size());
	const auto consumed  = handler(*this, it.dw[0] & ~1u, it.dw.data() + 1, packet_dw, packet_dw) + 1;
	if (consumed != packet_dw) {
		static std::atomic<uint32_t> logged {0};
		if (logged.fetch_add(1) < 8) {
			LOGF("cp pipe: handler of opcode 0x%x consumed %u dwords of a %u dword packet\n", it.opcode, consumed, packet_dw);
		}
	}
	CpPipe::CurrentJob() = nullptr;
	if (it.job.state.load(std::memory_order_relaxed) != CpPipe::GfpJob::Idle) {
		CpPipe::WaitForJob(it.job);
		it.job.state.store(CpPipe::GfpJob::Idle, std::memory_order_relaxed);
	}
	if (state_item) {
		std::swap(m_index_type_and_size, it.index_type_and_size);
		std::swap(m_index_buffer_size, it.index_buffer_size);
		std::swap(m_num_instances, it.num_instances);
		std::swap(m_num_instances_addr, it.num_instances_addr);
		std::swap(m_index_base_addr, it.index_base_addr);
		std::swap(m_draw_indirect_args_base_addr, it.draw_indirect_base);
		std::swap(m_dispatch_indirect_args_base_addr, it.dispatch_indirect_base);
		m_exec_sh   = nullptr;
		m_exec_ucfg = nullptr;
		GetScheduler().Rebind(m_ctx, m_ucfg, m_sh_ctx);
	}
	p.pending.Retire(seq);
	p.ring.Pop();
	p.executing = false;
	while (!p.held.empty() && p.held.front().gate < p.ring.FrontSeq()) {
		CpPipe::EnqueueJob(p.held.front().job);
		p.held.pop_front();
	}
}

void CommandProcessor::PipeDrain() {
	if (m_pipe == nullptr || m_pipe->executing || m_pipe->ring.Empty()) return;
	while (!m_pipe->ring.Empty()) PipeExecuteFront();
	m_pipe->pending.Clear();
}

bool CommandProcessor::PipeOverlaps(uint64_t address, uint64_t size) const {
	return m_pipe != nullptr && m_pipe->pending.Overlaps(address, size);
}

bool CommandProcessor::PipeWaitVirtual(uint64_t address, uint64_t size, uint32_t func, uint64_t ref, uint64_t mask, bool* satisfied) {
	auto& p = *m_pipe;
	if (p.executing) return false;
	uint64_t value = 0;
	switch (p.pending.Resolve(address, size, &value)) {
		case CpPipe::PendingWrites::Verdict::NoPending: return false;
		case CpPipe::PendingWrites::Verdict::Value:
			if (CpPipe::WaitCompare(value, ref, mask, func)) {
				*satisfied = true;
				return true;
			}
			break;
		case CpPipe::PendingWrites::Verdict::Unknown: break;
	}
	PipeDrain();
	return false;
}

} // namespace Libs::Graphics
