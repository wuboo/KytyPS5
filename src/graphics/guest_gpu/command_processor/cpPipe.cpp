// S479 (KYTY_CP_PIPE): the FIFO between the PM4 parser (front) and the handlers that draw and signal (recorder). Both run on the CP thread; the point of the FIFO
// is that a draw is no longer executed where it is parsed, so the work GetGraphicsPrograms needs can be done by other threads in between (cpPipeWorkers.cpp).
//
// What runs where (CpPipe::Classify): register writes at parse time as before; a draw or dispatch is snapshotted (the three register structs and the CP's index/instance
// state) and executed later on the snapshot; signals, barriers, CPU writes and flips are copied and executed later in stream order; WAIT_REG_MEM is answered from the
// writes still pending in the FIFO (PendingWrites) or drains; anything else drains first. The end of every stream and every suspend drains, so what leaves
// CommandProcessor::Process is exactly what the old code left.

#include "common/assert.h"
#include "common/diagOut.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/cpPipe.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/presentation/cpPhaseScope.h"
#include "graphics/presentation/frameTiming.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace Libs::Graphics {

namespace {

CpPipe::Mode EnvMode() {
	static const CpPipe::Mode mode = CpPipe::ParseMode(std::getenv("KYTY_CP_PIPE"));
	return mode;
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

// One deferred packet. Draws and dispatches carry the state they were parsed with.
struct PipeItem {
	CpPipe::OpClass       cls    = CpPipe::OpClass::Ordered;
	uint32_t              opcode = 0;
	std::vector<uint32_t> dw; // the whole packet, header included
	HW::Context           ctx;
	HW::Shader            sh;
	HW::UserConfig        ucfg;
	uint32_t              index_type_and_size  = 0;
	uint32_t              index_buffer_size    = 0;
	uint32_t              num_instances        = 1;
	uint64_t              index_base_addr      = 0;
	uint64_t              draw_indirect_base   = 0;
	uint64_t              dispatch_indirect_base = 0;
};

struct CommandProcessor::PipeState {
	explicit PipeState(size_t depth) : ring(depth), depth(depth), items(depth) {}

	CpPipe::Ring          ring;
	size_t                depth;
	std::vector<PipeItem> items;
	CpPipe::PendingWrites pending;
	bool                  executing = false; // an item runs: the pipeline is transparent for the handlers it calls
};

CommandProcessor::CommandProcessor(RenderContext& renderer, int interrupt_event_id) : m_renderer(renderer), m_interrupt_event_id(interrupt_event_id) {}

CommandProcessor::~CommandProcessor() = default;

void CommandProcessor::PipeInit() {
	if (m_pipe != nullptr || IsAsyncComputeQueue() || EnvMode() == CpPipe::Mode::Off) return;
	m_pipe = std::make_unique<PipeState>(EnvDepth());
	LOGF("[cp-pipe] on: mode=%s depth=%zu\n", EnvMode() == CpPipe::Mode::Shadow ? "shadow" : "1", m_pipe->depth);
}

void CommandProcessor::PipeEnqueue(CpPipe::OpClass cls, uint32_t opcode, const uint32_t* packet, uint32_t packet_dw) {
	using C = FrameTiming::Counter;
	auto& p = *m_pipe;
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
		it.index_type_and_size    = m_index_type_and_size;
		it.index_buffer_size      = m_index_buffer_size;
		it.num_instances          = m_num_instances;
		it.index_base_addr        = m_index_base_addr;
		it.draw_indirect_base     = m_draw_indirect_args_base_addr;
		it.dispatch_indirect_base = m_dispatch_indirect_args_base_addr;
	}
	const uint64_t seq = p.ring.Commit();
	FrameTiming::Add(C::PipeEnqN);
	FrameTiming::Add(cls == CpPipe::OpClass::Draw ? C::PipeDrawN : (cls == CpPipe::OpClass::Dispatch ? C::PipeDispatchN : C::PipeOrderedN));

	// The writes this item will do, for the WAIT_REG_MEM that follows it in the stream.
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
			if (r == Pm4::R_RELEASE_MEM && packet_dw >= 8u) note_signal(CpPipe::DecodeReleaseMem(payload, async, &address));
			else if (r == Pm4::R_FLIP || r == Pm4::R_ZERO) p.pending.Note(0, kEverything, false, 0, seq); // a flip writes labels
			break;
		case Pm4::IT_EVENT_WRITE_EOP:
			if (packet_dw >= 6u) note_signal(CpPipe::DecodeEventWriteEop(payload, async, &address));
			break;
		case Pm4::IT_EVENT_WRITE_EOS:
			if (packet_dw >= 5u) note_signal(CpPipe::DecodeEventWriteEos(payload, async, &address));
			break;
		case Pm4::IT_EVENT_WRITE:
			if (packet_dw > 3u) p.pending.Note(0, kEverything, false, 0, seq); // event types that write an address
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
	using C = FrameTiming::Counter;
	auto&          p   = *m_pipe;
	auto&          it  = p.items[p.ring.Front()];
	const uint64_t seq = p.ring.FrontSeq();
	const auto     t0  = FrameTiming::SteadyNowNs();
	p.executing        = true;

	const auto handler = g_cp_op_func[it.opcode];
	EXIT_IF(handler == nullptr);
	const bool state_item = it.cls == CpPipe::OpClass::Draw || it.cls == CpPipe::OpClass::Dispatch;
	if (it.opcode != Pm4::IT_DRAW_INDEX_2 && it.opcode != Pm4::IT_DISPATCH_DRAW_PREAMBLE && it.opcode != Pm4::IT_DRAW_INDEX_AUTO) {
		// Everything but a draw must see the effects of the draws before it applied (the same rule ProcessPm4 has for the pending prototype batch).
		m_renderer.GetRenderExecutor().FlushPendingPrototypeBatch(CurrentBuffer());
	}
	if (state_item) {
		GetScheduler().Rebind(it.ctx, it.ucfg, it.sh);
		m_exec_sh   = &it.sh;
		m_exec_ucfg = &it.ucfg;
		std::swap(m_index_type_and_size, it.index_type_and_size);
		std::swap(m_index_buffer_size, it.index_buffer_size);
		std::swap(m_num_instances, it.num_instances);
		std::swap(m_index_base_addr, it.index_base_addr);
		std::swap(m_draw_indirect_args_base_addr, it.draw_indirect_base);
		std::swap(m_dispatch_indirect_args_base_addr, it.dispatch_indirect_base);
	}
	{
		CpPhases::Scope handler_phase(CpPhases::Phase::Handler);
		if (CpPhases::Enabled()) {
			handler_phase.SetSub(it.cls == CpPipe::OpClass::Draw ? C::HdlDrawNs : (it.cls == CpPipe::OpClass::Dispatch ? C::HdlDispNs : C::HdlOtherNs),
			                     it.cls == CpPipe::OpClass::Draw ? C::HdlDrawN : (it.cls == CpPipe::OpClass::Dispatch ? C::HdlDispN : C::HdlOtherN));
		}
		const auto packet_dw = static_cast<uint32_t>(it.dw.size());
		const auto consumed  = handler(*this, it.dw[0] & ~1u, it.dw.data() + 1, packet_dw, packet_dw) + 1;
		if (consumed != packet_dw) {
			FrameTiming::Add(C::PipeLenMismatchN);
			static std::atomic<uint32_t> logged {0};
			if (logged.fetch_add(1) < 8) {
				LOGF("[cp-pipe] handler of opcode 0x%x consumed %u dwords of a %u dword packet\n", it.opcode, consumed, packet_dw);
			}
		}
	}
	if (state_item) {
		std::swap(m_index_type_and_size, it.index_type_and_size);
		std::swap(m_index_buffer_size, it.index_buffer_size);
		std::swap(m_num_instances, it.num_instances);
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
	FrameTiming::Add(C::PipeExecNs, static_cast<uint64_t>(FrameTiming::SteadyNowNs() - t0));
}

namespace {

// KYTY_CP_PIPE diag: what cut the windows. Key = opcode (NOP: 256 + R), 0x1f0 = guest command, 0x1f1 = wait, 0x1f2 = end of stream; dumped every 500 flips to the diag file.
std::atomic<uint64_t> g_drain_by_key[512];
std::atomic<uint64_t> g_drain_items;
std::atomic<uint64_t> g_dump_flip {0};

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
}

} // namespace

void CommandProcessor::PipeDrain(unsigned reason, uint32_t key) {
	using C = FrameTiming::Counter;
	if (m_pipe == nullptr || m_pipe->executing || m_pipe->ring.Empty()) return;
	FrameTiming::Add(reason == 0 ? C::PipeDrainClsN : (reason == 1 ? C::PipeDrainWaitN : C::PipeDrainEndN));
	g_drain_by_key[key & 511u].fetch_add(1, std::memory_order_relaxed);
	g_drain_items.fetch_add(m_pipe->ring.Size(), std::memory_order_relaxed);
	DumpDrainHistogram();
	while (!m_pipe->ring.Empty()) PipeExecuteFront();
	m_pipe->pending.Clear();
}

bool CommandProcessor::PipeWaitVirtual(uint64_t address, uint64_t size, uint32_t func, uint64_t ref, uint64_t mask, bool* satisfied) {
	auto&    p = *m_pipe;
	if (p.executing) return false;
	uint64_t value = 0;
	switch (p.pending.Resolve(address, size, &value)) {
		case CpPipe::PendingWrites::Verdict::NoPending: return false;
		case CpPipe::PendingWrites::Verdict::Value:
			if (CpPipe::WaitCompare(value, ref, mask, func)) {
				FrameTiming::Add(FrameTiming::Counter::PipeWaitVirtN);
				*satisfied = true;
				return true;
			}
			break;
		case CpPipe::PendingWrites::Verdict::Unknown: break;
	}
	PipeDrain(1, 0x1f1);
	return false;
}

} // namespace Libs::Graphics
