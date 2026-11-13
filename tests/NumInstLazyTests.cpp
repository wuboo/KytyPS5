// S577 (KYTY_NUMINST_LAZY, numInstLazy.h): the inherited NUM_INSTANCES of an indirect draw, read eagerly by the CP front (0) or lazily by the recorder at
// the first item that inherits it (1), or both (shadow). A model of the CP (front, FIFO, recorder; indirect draws are hand-over items the recorder runs while
// the front waits; an eager read of a GPU-written record waits until the FIFO is empty) runs the same PM4-like streams as the unpiped CP and checks:
//   - 1 gives the values of the unpiped CP (NumInstances() at the draw) on every stream, the front never reads guest memory, one read per consumed epoch;
//   - 0 gives the values of the eager reference (the read at the first Draw/Dispatch after the indirect draw);
//   - where 1 and 0 differ the watch saw a write over the record (numinst_taint_use_n), and shadow counts exactly those consumers (numinst_ne_n);
//   - NUM_INSTANCES clears the record, a new indirect draw is a new epoch, a draw with its own count and a dispatch never read.
// Every check below fails under one of the mutants (CMake: numinst_lazy_mutant_*, WILL_FAIL): STALE_EPOCH, REREAD, PACKET_KEEPS_ADDR, NO_TAINT.

#include "graphics/guest_gpu/command_processor/numInstLazy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics::NumInstLazy;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "NumInstLazyTests: failed: %s\n", text);
		std::exit(1); // not abort: ctest WILL_FAIL does not invert a crash
	}
}

enum class Kind : uint8_t {
	Indirect,     // DRAW_INDEX_INDIRECT*: leaves `addr` (its record's instanceCount) as the register; a hand-over item
	Packet,       // NUM_INSTANCES `value` (inline on the front)
	Dispatch,     // DISPATCH_*: a state item, never reads the count; `writes` are its GPU writes
	DrawInherit,  // DRAW_INDEX_2 (short) / DRAW_INDEX_OFFSET_2 / DRAW_INDEX_AUTO: inherits the count
	DrawExplicit, // DRAW_INDEX_2 with its own count `value`
	CpWrite,      // WRITE_DATA / DMA_DATA: an ordered item, writes guest memory on the recorder
};

struct Op {
	Kind                                       kind  = Kind::Dispatch;
	uint64_t                                   addr  = 0;
	uint32_t                                   value = 0;
	std::vector<std::pair<uint64_t, uint32_t>> writes; // dword writes of a dispatch / draw
};

using Memory = std::map<uint64_t, uint32_t>;

uint32_t Load(const Memory& mem, uint64_t addr) {
	const auto it = mem.find(addr);
	return it != mem.end() ? it->second : 0u;
}

struct Result {
	std::vector<uint32_t> consumed;     // the count each consumer drew with
	uint32_t              front_reads  = 0;
	uint32_t              eager_reads  = 0; // NumInstances() reads (numinst_eager_n)
	uint32_t              lazy_reads   = 0; // numinst_read_n
	uint32_t              lazy_hits    = 0; // numinst_hit_n
	uint32_t              skips        = 0; // numinst_lazy_skip_n
	uint32_t              shadow_n     = 0;
	uint32_t              ne           = 0;
	uint32_t              taint_uses   = 0;
	std::vector<bool>     consumer_tainted; // per consumer: the epoch it resolved / reused was tainted at its read
};

// The unpiped CP: NumInstances() at the draw that inherits the count.
std::vector<uint32_t> Unpiped(const std::vector<Op>& ops, Memory mem) {
	std::vector<uint32_t> out;
	uint64_t              addr = 0;
	uint32_t              reg  = 1;
	for (const auto& op: ops) {
		switch (op.kind) {
			case Kind::Indirect: addr = op.addr; break;
			case Kind::Packet:
				reg  = op.value == 0 ? 1u : op.value;
				addr = 0;
				break;
			case Kind::DrawInherit:
				if (addr != 0) {
					reg  = Load(mem, addr);
					addr = 0;
				}
				out.push_back(reg);
				break;
			case Kind::DrawExplicit:
			case Kind::Dispatch:
			case Kind::CpWrite: break;
		}
		if (op.kind == Kind::CpWrite) mem[op.addr] = op.value;
		for (const auto& [a, v]: op.writes) mem[a] = v;
	}
	return out;
}

// The eager reference (the pipe before S577): the read at the first Draw/Dispatch after the indirect draw, before that item runs.
std::vector<uint32_t> Eager(const std::vector<Op>& ops, Memory mem) {
	std::vector<uint32_t> out;
	uint64_t              addr = 0;
	uint32_t              reg  = 1;
	for (const auto& op: ops) {
		const bool state = op.kind == Kind::Dispatch || op.kind == Kind::DrawInherit || op.kind == Kind::DrawExplicit;
		if (op.kind == Kind::Indirect) addr = op.addr;
		if (op.kind == Kind::Packet) {
			reg  = op.value == 0 ? 1u : op.value;
			addr = 0;
		}
		if (state && addr != 0) {
			reg  = Load(mem, addr);
			addr = 0;
		}
		if (op.kind == Kind::DrawInherit) out.push_back(reg);
		if (op.kind == Kind::CpWrite) mem[op.addr] = op.value;
		for (const auto& [a, v]: op.writes) mem[a] = v;
	}
	return out;
}

// The piped CP with the front on its own thread (cpPipe.cpp PipeFrontEnqueue / PipeFrontExecute, graphicsRun.cpp ExecNumInstances): the front runs up
// to `depth` items ahead (a random lag), an indirect draw is a hand-over (the FIFO drains, the recorder runs it on the live state), an eager read of the
// record waits for the FIFO to drain (the fault's readback runs on the idle recorder).
struct Item {
	Op       op;
	uint32_t num_instances = 1;
	uint64_t ni_addr       = 0;
	uint64_t ni_epoch      = 0;
};

Result Piped(Mode mode, const std::vector<Op>& ops, Memory mem, uint32_t seed) {
	Result           r;
	std::mt19937     rng(seed);
	std::deque<Item> fifo;
	// Front-owned (the CP's live state): m_num_instances_addr, m_ds.num_instances, m_ni_front.
	uint64_t   num_instances_addr = 0;
	uint32_t   reg                = 1;
	FrontState front;
	// Recorder-owned.
	Resolver                     resolver;
	Watch                        watch;
	std::map<uint64_t, bool>     epoch_tainted; // the taint each epoch had at its read

	const auto note = [&](uint64_t a, uint32_t v) {
		watch.NoteGpuWrite(a, 4);
		mem[a] = v;
	};
	const auto execute = [&](const Item& it) {
		uint32_t count = 0;
		if (it.op.kind == Kind::DrawInherit) {
			if (it.ni_addr == 0) {
				count = it.num_instances;
				r.consumer_tainted.push_back(false);
			} else {
				const bool tainted = watch.TaintedFor(it.ni_epoch);
				bool       did_read = false;
				const uint32_t lazy = resolver.Resolve(it.ni_addr, it.ni_epoch, [&](uint64_t a) { return Load(mem, a); }, &did_read);
				if (did_read) {
					r.lazy_reads++;
					if (tainted) r.taint_uses++;
					epoch_tainted[it.ni_epoch] = tainted;
					if (watch.epoch.load() == it.ni_epoch) watch.Disarm();
				} else {
					r.lazy_hits++;
				}
				r.consumer_tainted.push_back(epoch_tainted[it.ni_epoch]);
				count = lazy;
				if (mode == Mode::Shadow) {
					r.shadow_n++;
					if (lazy != it.num_instances) r.ne++;
					count = it.num_instances;
				}
			}
			r.consumed.push_back(count);
		}
		if (it.op.kind == Kind::CpWrite) note(it.op.addr, it.op.value);
		for (const auto& [a, v]: it.op.writes) note(a, v);
	};
	const auto drain = [&] {
		while (!fifo.empty()) {
			execute(fifo.front());
			fifo.pop_front();
		}
	};
	const auto eager = [&]() -> uint32_t {
		if (num_instances_addr != 0) {
			drain(); // the fault: the readback runs once the recorder is idle
			r.eager_reads++;
			r.front_reads++;
			reg                = Load(mem, num_instances_addr);
			num_instances_addr = 0;
		}
		return reg;
	};
	const size_t depth = 1 + rng() % 6;
	for (const auto& op: ops) {
		switch (op.kind) {
			case Kind::Indirect: {
				drain(); // a hand-over: the recorder runs it on the live state while the front waits
				num_instances_addr = op.addr;
				front.OnIndirectDraw(op.addr);
				if (mode != Mode::Off) watch.Arm(op.addr, front.epoch); // after the draw is recorded
				break;
			}
			case Kind::Packet:
				reg                = op.value == 0 ? 1u : op.value;
				num_instances_addr = 0;
				front.OnPacket();
				break;
			case Kind::CpWrite: fifo.push_back({op}); break;
			case Kind::Dispatch:
			case Kind::DrawInherit:
			case Kind::DrawExplicit: {
				const Snap s = TakeSnapshot(mode, front, reg, eager);
				if (s.skipped) r.skips++;
				Item it {op, s.value, s.addr, s.epoch};
				fifo.push_back(it);
				break;
			}
		}
		while (fifo.size() > depth || (!fifo.empty() && rng() % 4 == 0)) { // the recorder catches up at its own pace
			execute(fifo.front());
			fifo.pop_front();
		}
	}
	drain();
	return r;
}

Memory InitialMemory() {
	return {{0x1004, 7}, {0x2004, 3}, {0x3004, 0}};
}

void ModeTests() {
	Check(ParseMode(nullptr) == Mode::Off, "no env: 0");
	Check(ParseMode("0") == Mode::Off, "0");
	Check(ParseMode("1") == Mode::On, "1");
	Check(ParseMode("shadow") == Mode::Shadow, "shadow");
	Check(ParseMode("2") == Mode::Off, "anything else: 0");
}

void WatchTests() {
	Watch w;
	w.NoteGpuWrite(0x1000, 0x100);
	Check(!w.TaintedFor(1), "an unarmed watch sees nothing");
	w.Arm(0x1004, 1);
	w.NoteGpuWrite(0x1000, 4);
	Check(!w.TaintedFor(1), "a write ending at the record is not over it");
	w.NoteGpuWrite(0x1008, 4);
	Check(!w.TaintedFor(1), "a write starting after the record is not over it");
	w.NoteGpuWrite(0x1007, 1);
	Check(w.TaintedFor(1), "a byte write into the record taints it");
	Check(!w.TaintedFor(2), "the taint belongs to its epoch");
	w.Arm(0x1004, 2);
	Check(!w.TaintedFor(2), "arming a new epoch clears the taint");
	w.NoteGpuWrite(0x0, 0x10000);
	Check(w.TaintedFor(2), "a buffer-sized write over the record taints it");
	w.Disarm();
	Check(!w.TaintedFor(2), "a disarmed watch reports nothing");
}

void DirectedTests() {
	const Memory m = InitialMemory();
	// A dispatch after the indirect draw: 0 reads (the fault of S575), 1 does not.
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x1004}, {Kind::Dispatch}, {Kind::Dispatch}};
		const auto off = Piped(Mode::Off, ops, m, 1);
		const auto on  = Piped(Mode::On, ops, m, 1);
		Check(off.eager_reads == 1 && off.front_reads == 1, "0: the first dispatch after the indirect draw reads the record");
		Check(on.front_reads == 0 && on.lazy_reads == 0, "1: no consumer, no read");
		Check(on.skips == 1, "1: one skipped read per epoch (numinst_lazy_skip_n)");
	}
	// NUM_INSTANCES after the indirect draw: the packet's value, no read.
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x1004}, {Kind::Packet, 0, 5}, {Kind::DrawInherit}};
		const auto on = Piped(Mode::On, ops, m, 2);
		Check(on.consumed.size() == 1 && on.consumed[0] == 5, "1: NUM_INSTANCES replaces the indirect draw's count");
		Check(on.lazy_reads == 0, "1: NUM_INSTANCES leaves nothing to read");
	}
	// One read per epoch, later consumers reuse it although the record changed.
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x1004}, {Kind::DrawInherit}, {Kind::Dispatch, 0, 0, {{0x1004, 9}}}, {Kind::DrawInherit}};
		const auto on = Piped(Mode::On, ops, m, 3);
		Check(on.consumed.size() == 2 && on.consumed[0] == 7 && on.consumed[1] == 7, "1: the epoch's value holds for every consumer");
		Check(on.lazy_reads == 1 && on.lazy_hits == 1, "1: one read, then the memo");
		Check(on.consumed == Unpiped(ops, m), "1: as the unpiped CP");
	}
	// A new indirect draw at the same address is a new epoch.
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x1004}, {Kind::DrawInherit}, {Kind::Dispatch, 0, 0, {{0x1004, 8}}}, {Kind::Indirect, 0x1004},
		                           {Kind::DrawInherit}};
		const auto on = Piped(Mode::On, ops, m, 4);
		Check(on.consumed.size() == 2 && on.consumed[0] == 7 && on.consumed[1] == 8, "1: the second indirect draw's count is read again");
		Check(on.lazy_reads == 2, "1: two epochs, two reads");
	}
	// A draw with its own count and a later inheriting draw.
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x2004}, {Kind::DrawExplicit, 0, 4}, {Kind::DrawInherit}};
		const auto on = Piped(Mode::On, ops, m, 5);
		Check(on.consumed.size() == 1 && on.consumed[0] == 3, "1: an explicit draw does not consume the inherited count");
		Check(on.lazy_reads == 1, "1: only the inheriting draw reads");
	}
	// A GPU write over the record between the indirect draw and the consumer: the only case where 1 and 0 differ.
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x1004}, {Kind::Dispatch, 0, 0, {{0x1004, 9}}}, {Kind::DrawInherit}};
		const auto off = Piped(Mode::Off, ops, m, 6);
		const auto on  = Piped(Mode::On, ops, m, 6);
		const auto sh  = Piped(Mode::Shadow, ops, m, 6);
		Check(off.consumed.size() == 1 && off.consumed[0] == 7, "0: the value before the dispatch");
		Check(on.consumed.size() == 1 && on.consumed[0] == 9, "1: the value at the consumer (the unpiped CP's)");
		Check(on.taint_uses == 1, "1: the watch saw the write over the record (numinst_taint_use_n)");
		Check(sh.consumed.size() == 1 && sh.consumed[0] == 7 && sh.ne == 1 && sh.shadow_n == 1, "shadow: draws with 0's value, counts the mismatch");
	}
	// A write next to the record does not taint it.
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x1004}, {Kind::Dispatch, 0, 0, {{0x1008, 9}, {0x1000, 9}}}, {Kind::DrawInherit}};
		const auto on = Piped(Mode::On, ops, m, 7);
		Check(on.consumed.size() == 1 && on.consumed[0] == 7 && on.taint_uses == 0, "1: neighbours of the record are not the record");
	}
	// A CP write (WRITE_DATA) over the record before the consumer.
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x1004}, {Kind::Dispatch}, {Kind::CpWrite, 0x1004, 2}, {Kind::DrawInherit}};
		const auto on = Piped(Mode::On, ops, m, 8);
		Check(on.consumed.size() == 1 && on.consumed[0] == 2 && on.taint_uses == 1, "1: a CP write over the record is seen and tainted");
	}
	// A zero record is inherited as zero (the old path does not map it to 1; only the packet does).
	{
		const std::vector<Op> ops {{Kind::Indirect, 0x3004}, {Kind::DrawInherit}};
		Check(Piped(Mode::On, ops, m, 9).consumed == std::vector<uint32_t> {0}, "1: a zero record stays zero");
		Check(Piped(Mode::Off, ops, m, 9).consumed == std::vector<uint32_t> {0}, "0: a zero record stays zero");
	}
}

std::vector<Op> RandomStream(std::mt19937& rng, size_t n) {
	const uint64_t        records[] = {0x1004, 0x2004, 0x3004};
	std::vector<Op>       ops;
	for (size_t i = 0; i < n; i++) {
		Op         op;
		const auto roll = rng() % 100;
		if (roll < 12) {
			op.kind = Kind::Indirect;
			op.addr = records[rng() % 3];
		} else if (roll < 20) {
			op.kind  = Kind::Packet;
			op.value = rng() % 4;
		} else if (roll < 50) {
			op.kind = Kind::Dispatch;
		} else if (roll < 75) {
			op.kind = Kind::DrawInherit;
		} else if (roll < 90) {
			op.kind  = Kind::DrawExplicit;
			op.value = 1 + rng() % 4;
		} else {
			op.kind  = Kind::CpWrite;
			op.addr  = records[rng() % 3] + (rng() % 2 == 0 ? 0 : 4);
			op.value = rng() % 16;
		}
		if ((op.kind == Kind::Dispatch || op.kind == Kind::DrawExplicit || op.kind == Kind::DrawInherit) && rng() % 3 == 0) {
			op.writes.push_back({records[rng() % 3] + (rng() % 2 == 0 ? 0 : 8), rng() % 16});
		}
		ops.push_back(op);
	}
	return ops;
}

void RandomTests() {
	std::mt19937 rng(577);
	uint32_t     diverged = 0;
	uint32_t     reads_saved = 0;
	for (uint32_t round = 0; round < 3000; round++) {
		const auto   ops  = RandomStream(rng, 8 + rng() % 40);
		const Memory m    = InitialMemory();
		const auto   ref  = Unpiped(ops, m);
		const auto   eref = Eager(ops, m);
		const auto   off  = Piped(Mode::Off, ops, m, round);
		const auto   on   = Piped(Mode::On, ops, m, round + 1);
		const auto   sh   = Piped(Mode::Shadow, ops, m, round + 2);
		Check(off.consumed == eref, "random: 0 = the eager reference");
		Check(on.consumed == ref, "random: 1 = the unpiped CP");
		Check(sh.consumed == eref, "random: shadow draws with 0's values");
		Check(on.front_reads == 0, "random: with 1 the front never reads the record");
		Check(off.eager_reads == on.skips, "random: 1 skips exactly the reads 0 does (numinst_lazy_skip_n = numinst_eager_n of 0)");
		Check(on.consumed.size() == off.consumed.size(), "random: same consumers");
		uint32_t mismatches = 0;
		for (size_t i = 0; i < on.consumed.size(); i++) {
			if (on.consumed[i] != off.consumed[i]) {
				mismatches++;
				Check(on.consumer_tainted[i], "random: 1 differs from 0 only where the watch saw a write over the record");
			}
		}
		Check(sh.ne == mismatches, "random: shadow counts exactly the consumers where 1 and 0 differ");
		diverged += mismatches;
		Check(on.lazy_reads <= on.skips, "random: every lazy read belongs to an epoch the front skipped");
		reads_saved += on.skips - on.lazy_reads;
	}
	Check(diverged > 0, "random: the streams include writes over a pending record");
	Check(reads_saved > 0, "random: the streams include epochs nobody consumed");
}

} // namespace

int main() {
	ModeTests();
	WatchTests();
	DirectedTests();
	RandomTests();
	std::printf("NumInstLazyTests: OK\n");
	return 0;
}
