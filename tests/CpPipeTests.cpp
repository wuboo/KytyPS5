// S479: the pure part of the CP pipeline (cpPipe.h). Every check below fails under one of the KYTY_CPPIPE_MUTANT_* defines (CMake: cp_pipe_mutant_*, WILL_FAIL).

#include "graphics/guest_gpu/command_processor/cpPipe.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CpPipe;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpPipeTests: failed: %s\n", text);
		std::exit(1);
	}
}

void ClassifyTests() {
	Check(Classify(Pm4::IT_SET_SH_REG, 0, 0, 5) == OpClass::Inline, "SET_SH_REG runs at parse");
	Check(Classify(Pm4::IT_SET_CONTEXT_REG, 0, 0, 5) == OpClass::Inline, "SET_CONTEXT_REG runs at parse");
	Check(Classify(Pm4::IT_INDEX_TYPE, 0, 0, 2) == OpClass::Inline, "INDEX_TYPE runs at parse");
	Check(Classify(Pm4::IT_DRAW_INDEX_2, 0, 0, 9) == OpClass::Draw, "DRAW_INDEX_2 is a draw");
	Check(Classify(Pm4::IT_DRAW_INDEX_AUTO, 0, 0, 3) == OpClass::Draw, "DRAW_INDEX_AUTO is a draw");
	Check(Classify(Pm4::IT_DISPATCH_DIRECT, 0, 0, 5) == OpClass::Dispatch, "DISPATCH_DIRECT is a dispatch");
	Check(Classify(Pm4::IT_WAIT_REG_MEM, 0, 0, 7) == OpClass::Wait, "WAIT_REG_MEM is a wait");
	Check(Classify(Pm4::IT_WAIT_REG_MEM_64, 0, 0, 9) == OpClass::Wait, "WAIT_REG_MEM_64 is a wait");
	// signals and CPU writes wait in the FIFO behind the draws that precede them
	Check(Classify(Pm4::IT_RELEASE_MEM, 0, 0, 8) == OpClass::Ordered, "RELEASE_MEM is ordered");
	Check(Classify(Pm4::IT_EVENT_WRITE, 0, 0, 4) == OpClass::Ordered, "EVENT_WRITE is ordered");
	Check(Classify(Pm4::IT_WRITE_DATA, 0, 0, 6) == OpClass::Ordered, "WRITE_DATA is ordered");
	Check(Classify(Pm4::IT_DMA_DATA, 0, 0, 7) == OpClass::Ordered, "DMA_DATA is ordered");
	Check(Classify(Pm4::IT_NOP, Pm4::R_FLIP, 0, 6) == OpClass::Ordered, "FLIP is ordered");
	// markers: upstream no longer executes them (S487), all stay inline
	Check(Classify(Pm4::IT_NOP, Pm4::R_ZERO, 0x68750004u, 3) == OpClass::Inline, "marker tag 4 is inline");
	Check(Classify(Pm4::IT_NOP, Pm4::R_ZERO, 0x68750777u, 3) == OpClass::Inline, "marker flip tag is inline");
	Check(Classify(Pm4::IT_NOP, Pm4::R_ZERO, 0, 2) == OpClass::Inline, "plain NOP is inline");
	// everything the table does not know drains
	Check(Classify(Pm4::IT_SET_SH_REG_INDIRECT, 0, 0, 5) == OpClass::Inline && ReadsIndirectTable(Pm4::IT_SET_SH_REG_INDIRECT), "indirect register sets run at parse, after a table check");
	Check(Classify(Pm4::IT_SET_CONTEXT_REG_INDIRECT, 0, 0, 5) == OpClass::Inline && !ReadsIndirectTable(Pm4::IT_SET_CONTEXT_REG), "only the indirect sets read a table");
	Check(Classify(Pm4::IT_COND_EXEC, 0, 0, 5) == OpClass::Drain, "COND_EXEC drains");
	Check(Classify(Pm4::IT_SET_PREDICATION, 0, 0, 5) == OpClass::Drain, "SET_PREDICATION drains");
	Check(Classify(Pm4::IT_DRAW_INDIRECT, 0, 0, 5) == OpClass::Drain, "DRAW_INDIRECT drains");
	Check(Classify(0xee, 0, 0, 5) == OpClass::Drain, "an unknown opcode drains");
	Check(Classify(Pm4::IT_NOP, Pm4::R_WAIT_FLIP_DONE, 0, 4) == OpClass::Drain, "WAIT_FLIP_DONE drains");
	Check(IsGraphicsDraw(Pm4::IT_DRAW_INDEX_2) && !IsGraphicsDraw(Pm4::IT_DISPATCH_DIRECT), "graphics draw predicate");
}

void WaitTests() {
	Check(WaitCompare(0x1234u, 0x34u, 0xffu, 3), "masked equal");
	Check(!WaitCompare(0x1234u, 0x1234u, 0xffu, 3), "masked: the ref has bits outside the mask");
	Check(WaitCompare(5, 5, ~uint64_t {0}, 5) && !WaitCompare(4, 5, ~uint64_t {0}, 5), ">=");
	Check(WaitCompare(0xff00u, 0u, 0xffu, 3), "the mask hides the high bits (equal to zero)");
}

void PendingWritesTests() {
	PendingWrites pending;
	uint64_t      value = 0;
	Check(pending.Resolve(0x1000, 4, &value) == PendingWrites::Verdict::NoPending, "empty: nothing pending");
	pending.Note(0x1000, 4, true, 0x77u, 1);
	Check(pending.Resolve(0x1000, 4, &value) == PendingWrites::Verdict::Value && value == 0x77u, "exact range: the value");
	Check(pending.Resolve(0x2000, 4, &value) == PendingWrites::Verdict::NoPending, "other address: nothing pending");
	Check(pending.Resolve(0x1000, 8, &value) == PendingWrites::Verdict::Unknown, "wider read overlaps: unknown");
	Check(pending.Resolve(0x1002, 4, &value) == PendingWrites::Verdict::Unknown, "shifted read overlaps: unknown");
	Check(pending.Resolve(0x1004, 4, &value) == PendingWrites::Verdict::NoPending, "adjacent range is not an overlap");
	// the newest write wins
	pending.Note(0x1000, 4, true, 0x99u, 2);
	Check(pending.Resolve(0x1000, 4, &value) == PendingWrites::Verdict::Value && value == 0x99u, "newest write wins");
	// a write whose bytes the packet does not fix
	pending.Note(0x3000, 8, false, 0, 3);
	Check(pending.Resolve(0x3000, 8, &value) == PendingWrites::Verdict::Unknown, "unknown bytes: unknown");
	// one item retires, the others stay (RETIRE_ALL forgets them)
	pending.Retire(2);
	Check(pending.Resolve(0x1000, 4, &value) == PendingWrites::Verdict::Value && value == 0x77u, "retiring item 2 uncovers the older write");
	Check(pending.Resolve(0x3000, 8, &value) == PendingWrites::Verdict::Unknown, "retiring item 2 keeps item 3");
	pending.Retire(1);
	pending.Retire(3);
	Check(pending.Size() == 0, "all retired");
}

void IndirectTableTests() {
	uint32_t payload[4] = {0x1003u, 0x7u, 0, 0x8004u};
	uint64_t address = 0, size = 0;
	Check(IndirectTableRange(payload, &address, &size) && address == 0x7'0000'1000ull && size == 4u * 8u, "indirect table range");
	payload[3] = 0;
	Check(!IndirectTableRange(payload, &address, &size), "an empty table has no range");
	PendingWrites pending;
	pending.Note(0x7'0000'1010ull, 4, false, 0, 1);
	payload[3] = 4;
	IndirectTableRange(payload, &address, &size);
	Check(pending.Overlaps(address, size), "a pending write inside the table is an overlap");
	Check(!pending.Overlaps(0x7'0000'1020ull, 8), "a table behind the write is not");
}

void SignalDecodeTests() {
	// RELEASE_MEM: data_sel 1 = 32-bit immediate, 2 = 64-bit immediate, 0 = interrupt only
	uint32_t     p[7] = {0, 0, 0x1000, 0, 0x11223344u, 0x55667788u, 0};
	uint64_t     address = 0;
	p[1]               = 1u << 29u;
	auto s             = DecodeReleaseMem(p, false, &address);
	Check(s.writes && s.size == 4 && s.known && s.value == 0x11223344u && address == 0x1000, "RELEASE_MEM data_sel 1");
	p[1] = 2u << 29u;
	s    = DecodeReleaseMem(p, false, &address);
	Check(s.writes && s.size == 8 && s.known && s.value == 0x5566778811223344ull, "RELEASE_MEM data_sel 2");
	p[1] = 0;
	Check(!DecodeReleaseMem(p, false, &address).writes, "RELEASE_MEM data_sel 0 writes nothing");
	p[1] = (1u << 29u) | (1u << 24u);
	Check(!DecodeReleaseMem(p, false, &address).writes, "interrupt selector 1 on the graphics queue writes nothing");
	Check(DecodeReleaseMem(p, true, &address).writes, "interrupt selector 1 on an async queue writes");
	p[1] = 3u << 29u;
	s    = DecodeReleaseMem(p, false, &address);
	Check(s.writes && !s.known, "a timestamp is not a known value");
	// EVENT_WRITE_EOP
	uint32_t e[5] = {0, 0x2000, 0, 0xaaaaaaaau, 0xbbbbbbbbu};
	e[2]          = 2u << 29u;
	auto eop      = DecodeEventWriteEop(e, false, &address);
	Check(eop.writes && eop.size == 8 && eop.known && eop.value == 0xbbbbbbbbaaaaaaaaull && address == 0x2000, "EOP immediate 64");
	e[2] = 4u << 29u;
	Check(!DecodeEventWriteEop(e, false, &address).known, "EOP reference clock is not known");
	// CPU writes
	uint32_t w[8] = {0, 0x4000, 0, 1, 2, 3, 0, 0};
	auto     wr   = DecodeWriteData(w, 7u);
	Check(wr.address == 0x4000 && wr.size == 12u, "WRITE_DATA range");
	w[0] = 1u << 16u;
	Check(DecodeWriteData(w, 7u).size == 4u, "WRITE_DATA one address");
	uint32_t d[6] = {0, 0, 0, 0x5000, 0, 0x100};
	Check(DecodeDmaData(d).address == 0x5000 && DecodeDmaData(d).size == 0x100, "DMA_DATA range");
}

void GateTests() {
	Check(ParseSignalPolicy(nullptr) == SignalPolicy::Fifo && ParseSignalPolicy("writes") == SignalPolicy::Writes && ParseSignalPolicy("fifo") == SignalPolicy::Fifo &&
	          ParseSignalPolicy("parse") == SignalPolicy::Parse && ParseSignalPolicy("x") == SignalPolicy::Fifo,
	      "signal policy parse");
	GateTracker gate;
	Check(gate.GateFor(SignalPolicy::Fifo, 1) == 0, "nothing pending: no gate");
	gate.NoteItem(OpClass::Ordered, Pm4::IT_EVENT_WRITE, 3);      // a barrier
	gate.NoteItem(OpClass::Draw, Pm4::IT_DRAW_INDEX_2, 4);        // a draw is no gate
	Check(gate.GateFor(SignalPolicy::Parse, 1) == 0, "parse: never a gate");
	Check(gate.GateFor(SignalPolicy::Writes, 1) == 0, "writes: a barrier is not a gate");
	Check(gate.GateFor(SignalPolicy::Fifo, 1) == 3, "fifo: the barrier is the gate");
	gate.NoteItem(OpClass::Ordered, Pm4::IT_WRITE_DATA, 5);
	Check(gate.GateFor(SignalPolicy::Writes, 1) == 5 && gate.GateFor(SignalPolicy::Fifo, 1) == 5, "a CPU write gates writes and fifo");
	Check(gate.GateFor(SignalPolicy::Writes, 6) == 0, "a gate that has run is gone");
	gate.NoteItem(OpClass::Dispatch, Pm4::IT_DISPATCH_DIRECT, 7);
	Check(gate.GateFor(SignalPolicy::Writes, 6) == 7, "a dispatch gates writes");
}

void RingTests() {
	Ring ring(4);
	Check(ring.Empty() && !ring.Full(), "ring: starts empty");
	for (int i = 0; i < 4; i++) {
		const size_t slot = ring.Reserve();
		Check(slot == static_cast<size_t>(i), "ring: slots in order");
		Check(ring.Commit() == static_cast<uint64_t>(i + 1), "ring: sequence numbers");
	}
	Check(ring.Full() && ring.Size() == 4, "ring: full");
	Check(ring.Front() == 0 && ring.FrontSeq() == 1, "ring: front is the oldest");
	ring.Pop();
	Check(!ring.Full() && ring.Front() == 1 && ring.FrontSeq() == 2, "ring: pop advances the front");
	const size_t slot = ring.Reserve();
	Check(slot == 0, "ring: wraps");
	ring.Commit();
	Check(ring.At(3) == 0 && ring.Size() == 4, "ring: At indexes from the front");
}

void WorkerGateTests() {
	struct FakeCp {
		uint32_t index_type = 0;
	} cp;
	Check(WriteCpMember(cp.index_type, 1u) && cp.index_type == 1u, "the CP thread writes its own state");
	std::thread worker([&] {
		WorkerScope scope;
		const bool  wrote = WriteCpMember(cp.index_type, 2u);
		Check(!wrote && cp.index_type == 1u, "a worker thread does not write the CP");
	});
	worker.join();
	Check(WriteCpMember(cp.index_type, 3u) && cp.index_type == 3u, "the flag is per thread");
	Check(ParseMode("1") == Mode::On && ParseMode("shadow") == Mode::Shadow && ParseMode("0") == Mode::Off && ParseMode(nullptr) == Mode::Off, "mode parse");
}

} // namespace

int main() {
	ClassifyTests();
	WaitTests();
	PendingWritesTests();
	SignalDecodeTests();
	IndirectTableTests();
	GateTests();
	RingTests();
	WorkerGateTests();
	std::puts("CpPipeTests: ok");
	return 0;
}
