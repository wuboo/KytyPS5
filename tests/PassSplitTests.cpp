// S612 (KYTY_PASS_SPLIT, passSplit.h): the order of a draw's tail split into R and P, and the ledger of its sections. Every check ends in exit(1)
// when it fails, so each mutant (CMake pass_split_mutant_*, WILL_FAIL) is caught by a failed check.
//
// Tail oracle: a fake command buffer, images with layouts and the executor's deferred-write list (the semantics of renderDraw.cpp: a transition
// that needs a barrier ends an open pass, a pass end records the deferred write barrier, BeginRendering of another pass ends the open one, the
// eager split submits at a pass change, an image-heap commit transitions the heap image). Random draw streams (stages with GDS and bound images,
// render targets, vertex/index buffers, push data, buffer writes immediate/deferred/by address, reads of pending ranges, a list of 3 pending
// writes) run twice from the same world, with RunTail off and with RunTail split (serial draws as the emulator picks them, NeedsSerial). The
// state each draw is recorded with (bound vertex/index buffer, descriptor contents with the layouts they hold, push data, dynamic state, pipeline,
// pass, deferred barrier, layouts of every image) must be the same, the final world must be the same, and the command log must be the same once
// the one reordering the split makes is undone (the vertex-buffer bind moved after the binding transitions of its draw). In shadow the checks
// the emulator makes (pipeline at its off position, the decisions again at their off position) must never differ.
//   DECIDE_EARLY, BUILD_BEFORE_TRANSIT, DRAW_BEFORE_DYNAMIC, NO_EMIT_BEFORE, NO_NOTE_END, SERIAL_IGNORED: caught by the tail oracle.
// Ledger oracle: a fake clock and random draws of laps (split and classify laps), mutations (hard / soft, inside and outside draws), nested scopes,
// serial and no-P draws; the slots Flush moves must equal an independent replay of the script (each interval filed by the mutations inside it).
//   SOFT_AS_CLEAN, P_AS_R, NOTE_OUTSIDE, NESTED_SCOPE: caught by the ledger oracle.
// The hand cases, the tables and the laps of RunTail catch most of them as well; -DKYTY_PASSSPLIT_ORACLE_ONLY builds the two oracles alone (a
// control: they catch every mutant without the rest).

#include "graphics/host_gpu/renderer/passSplit.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <utility>
#include <vector>

namespace {

namespace PS = Libs::Graphics::PassSplit;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "PassSplitTests: failed: %s\n", text);
		std::fflush(stderr);
		std::exit(1);
	}
}

uint64_t g_now = 0;
uint64_t FakeTicks() {
	return g_now;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// The tail.

enum Kind : int { VB, Barrier, Gds, EndPass, DeferredFlush, Submit, PushConst, PushDesc, IB, PushDraw, Dyn, Hazard, Begin, BindPipe, DrawCmd, WriteBarrier, DeferMark, HeapCommit };

struct Cmd {
	int  draw = 0;
	int  kind = 0;
	int  a = 0, b = 0;
	bool r_tag = false; // made by the binding transitions (BindingsWhole's / BindingsTransit's barriers, GDS barriers, the pass ends they cause)
	bool operator==(const Cmd& o) const { return draw == o.draw && kind == o.kind && a == o.a && b == o.b && r_tag == o.r_tag; }
};

struct Range {
	int begin = 0, end = 0;
	bool operator==(const Range& o) const { return begin == o.begin && end == o.end; }
};

struct Bind {
	int img = 0, want = 0;
};
struct Stage {
	bool              gds = false;
	std::vector<Bind> binds;
};
struct DrawDesc {
	int                id = 0;
	int                pass = 0;
	std::vector<int>   rts;
	int                rt_layout = 1;
	std::vector<Stage> stages;
	bool               vb = false, ib = false, push = false;
	bool               write_stages = false, address_writes = false;
	std::vector<Range> reads, writes;
	bool               heap = false;
	int                heap_img = 0;
	bool               eager = false;
};

struct Snapshot {
	int              vb, ib, pushc, pushdesc, dyn, pipe, pass, draw;
	bool             open, deferred;
	std::vector<int> layouts;
	bool operator==(const Snapshot& o) const {
		return vb == o.vb && ib == o.ib && pushc == o.pushc && pushdesc == o.pushdesc && dyn == o.dyn && pipe == o.pipe && pass == o.pass &&
		       draw == o.draw && open == o.open && deferred == o.deferred && layouts == o.layouts;
	}
};

constexpr int    kImages     = 6;
constexpr size_t kPendingCap = 3;

struct World {
	bool                  open = false, deferred = false, defer_on = true;
	int                   pass = -1, submits = 0;
	int                   vb = -1, ib = -1, pushc = -1, pushdesc = -1, dyn = -1;
	std::vector<int>      layout = std::vector<int>(kImages, 0);
	std::vector<Range>    pending;
	std::vector<Cmd>      log;
	std::vector<Snapshot> snaps;
	uint64_t              ne = 0, shadow = 0;
};

struct Plan {
	bool read_flush = false, write_now = false, defer_write = false, note_failed = false;
};

bool Overlaps(const std::vector<Range>& a, const std::vector<Range>& b) {
	for (const auto& x: a) {
		for (const auto& y: b) {
			if (x.begin < y.end && y.begin < x.end) return true;
		}
	}
	return false;
}

// The decisions of DecideBefore / DecideAfter on a given state (also the shadow's recomputation on a copy).
void DecideBeforeOn(const DrawDesc& d, bool defer_on, bool deferred, std::vector<Range>& pending, Plan& plan) {
	plan.defer_write = defer_on && d.write_stages && !d.address_writes;
	plan.read_flush  = false;
	if (defer_on) {
		if (!deferred) {
			pending.clear();
		} else if (Overlaps(d.reads, pending)) {
			plan.read_flush = true;
			pending.clear();
		}
	}
	plan.write_now = d.write_stages && !plan.defer_write;
	if (plan.write_now) pending.clear();
}
void DecideAfterOn(const DrawDesc& d, std::vector<Range>& pending, Plan& plan) {
	plan.note_failed = false;
	if (!plan.defer_write) return;
	for (const auto& w: d.writes) pending.push_back(w);
	if (pending.size() > kPendingCap) {
		plan.note_failed = true;
		pending.clear();
	}
}

struct TailOps {
	TailOps(World& world, const DrawDesc& draw): w(world), d(draw) {}
	World&          w;
	const DrawDesc& d;
	Plan            plan;
	int             pipe = -1;
	std::vector<std::vector<int>> captured; // [stage][bind]: the layout the descriptor holds
	// shadow
	int                early_pipe = -1;
	std::vector<Range> saved_pending;

	void Log(int kind, int a = 0, int b = 0, bool r_tag = false) { w.log.push_back({d.id, kind, a, b, r_tag}); }
	void End(bool r_tag) {
		if (!w.open) return;
		Log(EndPass, 0, 0, r_tag);
		w.open = false;
		if (w.deferred) {
			Log(DeferredFlush, 0, 0, r_tag);
			w.deferred = false;
		}
	}
	void Transit(int img, int layout, bool r_tag) {
		if (w.layout[img] == layout) return;
		End(r_tag);
		Log(Barrier, img, layout, r_tag);
		w.layout[img] = layout;
		PS::Note(PS::Mut::Transit);
		g_now += 3;
	}
	void Transits() {
		captured.assign(d.stages.size(), {});
		for (size_t s = 0; s < d.stages.size(); s++) {
			if (d.stages[s].gds) {
				End(true);
				Log(Gds, static_cast<int>(s), 0, true);
			}
			for (const auto& b: d.stages[s].binds) {
				Transit(b.img, b.want, true);
				captured[s].push_back(w.layout[b.img]);
			}
		}
	}
	void CommitHeap() {
		Transit(d.heap_img, 7, true);
		Log(HeapCommit, d.heap_img, 0, true);
	}
	void Build() {
		uint32_t hash = 17; // unsigned: wraps
		for (size_t s = 0; s < d.stages.size(); s++) {
			for (size_t k = 0; k < d.stages[s].binds.size(); k++) {
				// A build before the transitions (a mutant) has no kept layout: -1, as stale as it gets (no out-of-range read: exit(1), not a crash).
				const int kept = s < captured.size() && k < captured[s].size() ? captured[s][k] : -1;
				hash           = hash * 31u + static_cast<uint32_t>(d.stages[s].binds[k].img * 8 + kept);
			}
		}
		if (d.push) {
			Log(PushConst, d.id);
			w.pushc = d.id;
		}
		Log(PushDesc, static_cast<int>(hash & 0x7fffffffu));
		w.pushdesc = static_cast<int>(hash & 0x7fffffffu);
		g_now += 5;
	}

	void operator()(PS::Op::Pipeline) {
		pipe = d.id * 7 + 3;
		g_now += 2;
		if (early_pipe != -1 && early_pipe != pipe) w.ne++;
	}
	void operator()(PS::Op::AcquireTargets) {
		for (const int rt: d.rts) Transit(rt, d.rt_layout, false);
		g_now += 4;
	}
	bool operator()(PS::Op::Prologue) {
		if (d.eager && w.open && w.pass != d.pass) {
			End(false);
			Log(Submit);
			PS::Note(PS::Mut::Submit);
			w.submits++;
			w.vb = w.ib = w.pushc = w.pushdesc = w.dyn = -1; // a new command buffer
		}
		g_now += 1;
		return true;
	}
	void operator()(PS::Op::CommitVertex) {
		if (d.vb) {
			Log(VB, d.id);
			w.vb = d.id;
		}
		g_now += 1;
	}
	void operator()(PS::Op::BindingsWhole) {
		if (d.heap) CommitHeap();
		Transits();
		Build();
	}
	void operator()(PS::Op::BindingsTransit) { Transits(); }
	void operator()(PS::Op::BindingsBuild) {
		if (d.heap) CommitHeap(); // only a mutant splits a heap draw: its commit then follows the transitions
		Build();
	}
	void operator()(PS::Op::CommitIndex) {
		if (d.ib) {
			Log(IB, d.id);
			w.ib = d.id;
		}
	}
	void operator()(PS::Op::PushDraw) { Log(PushDraw, d.id); }
	void operator()(PS::Op::Dynamic) {
		Log(Dyn, d.id);
		w.dyn = d.id;
		g_now += 2;
	}
	void operator()(PS::Op::DecideBefore) { DecideBeforeOn(d, w.defer_on, w.deferred, w.pending, plan); }
	void operator()(PS::Op::EmitBefore) {
		if (plan.read_flush) End(false);
		if (plan.write_now) {
			End(false);
			Log(Hazard, d.id);
		}
	}
	void operator()(PS::Op::Draw) {
		if (!(w.open && w.pass == d.pass)) {
			End(false);
			Log(Begin, d.pass);
			w.open = true;
			w.pass = d.pass;
		}
		Log(BindPipe, pipe);
		Log(DrawCmd, d.id);
		Snapshot snap {w.vb, w.ib, w.pushc, w.pushdesc, w.dyn, pipe, w.pass, d.id, w.open, w.deferred, w.layout};
		w.snaps.push_back(snap);
		g_now += 6;
	}
	void operator()(PS::Op::AfterBarrier) {
		if (plan.write_now) {
			End(false);
			Log(WriteBarrier, d.id);
		} else if (plan.defer_write) {
			w.deferred = true;
			Log(DeferMark, d.id);
		}
	}
	void operator()(PS::Op::DecideAfter) { DecideAfterOn(d, w.pending, plan); }
	void operator()(PS::Op::AfterEnd) {
		if (plan.defer_write && plan.note_failed) End(false);
	}
	void operator()(PS::Op::ShadowEarly) {
		early_pipe     = d.id * 7 + 3;
		saved_pending  = w.pending;
	}
	void operator()(PS::Op::ShadowCheck) {
		Plan again;
		auto copy = saved_pending;
		DecideBeforeOn(d, w.defer_on, w.deferred, copy, again);
		DecideAfterOn(d, copy, again);
		w.shadow++;
		if (again.read_flush != plan.read_flush || again.write_now != plan.write_now || again.note_failed != plan.note_failed || copy != w.pending) {
			w.ne++;
		}
	}
};

DrawDesc RandomDraw(std::mt19937_64& rng, int id) {
	auto     pick = [&](int n) { return static_cast<int>(rng() % static_cast<uint64_t>(n)); };
	DrawDesc d;
	d.id   = id;
	d.pass = pick(3);
	for (int i = pick(3); i > 0; i--) d.rts.push_back(pick(kImages));
	d.rt_layout = 1 + pick(2);
	d.stages.resize(1 + pick(3));
	for (auto& s: d.stages) {
		s.gds = pick(8) == 0;
		for (int i = pick(4); i > 0; i--) s.binds.push_back({pick(kImages), 1 + pick(4)});
	}
	d.vb             = pick(4) != 0;
	d.ib             = pick(2) != 0;
	d.push           = pick(2) != 0;
	d.write_stages   = pick(3) == 0;
	d.address_writes = d.write_stages && pick(4) == 0;
	for (int i = pick(3); i > 0; i--) {
		const int b = pick(32);
		d.reads.push_back({b, b + 1 + pick(6)});
	}
	for (int i = 1 + pick(2); i > 0; i--) {
		const int b = pick(32);
		d.writes.push_back({b, b + 1 + pick(6)});
	}
	d.heap     = pick(10) == 0;
	d.heap_img = pick(kImages);
	d.eager    = pick(5) == 0;
	return d;
}

// Undoes the one reordering of the split: the VB bind of a draw goes after the last binding-transition command of the draw, if any.
std::vector<Cmd> Canonical(const std::vector<Cmd>& log) {
	std::vector<Cmd> out;
	size_t           i = 0;
	while (i < log.size()) {
		size_t j = i;
		while (j < log.size() && log[j].draw == log[i].draw) j++;
		std::vector<Cmd> seg(log.begin() + static_cast<std::ptrdiff_t>(i), log.begin() + static_cast<std::ptrdiff_t>(j));
		const auto vb = std::find_if(seg.begin(), seg.end(), [](const Cmd& c) { return c.kind == VB; });
		if (vb != seg.end()) {
			const Cmd v = *vb;
			seg.erase(vb);
			size_t last_r = SIZE_MAX;
			for (size_t k = 0; k < seg.size(); k++) {
				if (seg[k].r_tag) last_r = k;
			}
			if (last_r != SIZE_MAX) {
				seg.insert(seg.begin() + static_cast<std::ptrdiff_t>(last_r + 1), v);
			} else {
				// No binding transitions: the bind stays before the first command of P that follows it (push data / descriptors).
				const auto at = std::find_if(seg.begin(), seg.end(), [](const Cmd& c) { return c.kind == PushConst || c.kind == PushDesc; });
				seg.insert(at, v);
			}
		}
		out.insert(out.end(), seg.begin(), seg.end());
		i = j;
	}
	return out;
}

void RunStream(World& w, const std::vector<DrawDesc>& draws, bool split, bool shadow) {
	for (const auto& d: draws) {
		TailOps    ops {w, d};
		const bool this_split = split && !PS::NeedsSerial(false, d.heap);
		Check(PS::RunTail(ops, this_split, shadow && this_split), "a tail declined");
	}
}

void TailOracle() {
	uint64_t compared = 0, reordered = 0, shadows = 0, serial = 0;
	for (uint64_t seed = 1; seed <= 4; seed++) {
		std::mt19937_64 rng(seed * 0x9e3779b97f4a7c15ull);
		for (int stream = 0; stream < 2500; stream++) {
			std::vector<DrawDesc> draws;
			const int             n = 1 + static_cast<int>(rng() % 40);
			for (int i = 0; i < n; i++) draws.push_back(RandomDraw(rng, i + 1));
			World start;
			start.defer_on = (rng() % 8) != 0;
			for (auto& l: start.layout) l = static_cast<int>(rng() % 3);
			for (const bool shadow: {false, true}) {
				World off = start, cut = start;
				RunStream(off, draws, false, false);
				RunStream(cut, draws, true, shadow);
				Check(off.snaps.size() == cut.snaps.size(), "tail: the number of draws recorded differs");
				for (size_t i = 0; i < off.snaps.size(); i++) Check(off.snaps[i] == cut.snaps[i], "tail: the state a draw is recorded with differs");
				Check(off.layout == cut.layout && off.pending == cut.pending && off.open == cut.open && off.deferred == cut.deferred &&
				          off.pass == cut.pass && off.submits == cut.submits,
				      "tail: the final world differs");
				Check(Canonical(off.log) == Canonical(cut.log), "tail: the command log differs beyond the vertex-buffer bind");
				if (off.log != cut.log) reordered++;
				Check(cut.ne == 0, "tail: a shadow check differs");
				Check(off.ne == 0 && off.shadow == 0, "tail: off ran a shadow check");
				if (shadow) {
					size_t splits = 0;
					for (const auto& d: draws) splits += PS::NeedsSerial(false, d.heap) ? 0 : 1;
					Check(cut.shadow == splits, "tail: shadow did not check every split draw");
					shadows += cut.shadow;
				}
				compared++;
			}
			for (const auto& d: draws) serial += d.heap ? 1 : 0;
		}
	}
	Check(reordered > 1000, "tail: too few streams where the split reorders (the generator does not reach the case)");
	std::printf("tail oracle: %llu stream pairs, %llu reordered (equal once canonical), %llu shadow checks, %llu serial draws\n",
	            static_cast<unsigned long long>(compared), static_cast<unsigned long long>(reordered), static_cast<unsigned long long>(shadows),
	            static_cast<unsigned long long>(serial));
}

// Hand case: a deferred write pending, an open pass, a binding transition that ends it (flushing the deferred barrier) and a read of the
// pending range: off sees no pending barrier at its decision, so it only clears the list; a split that decided before the transition would flush.
void TailHandCases() {
	World w;
	w.open     = true;
	w.pass     = 0;
	w.deferred = true;
	w.pending  = {{0, 4}};
	DrawDesc d;
	d.id     = 1;
	d.pass   = 0;
	d.stages = {Stage {false, {{2, 3}}}};
	d.reads  = {{1, 2}};
	d.vb     = true;
	World off = w, cut = w;
	RunStream(off, {d}, false, false);
	RunStream(cut, {d}, true, true);
	Check(Canonical(off.log) == Canonical(cut.log) && off.snaps == cut.snaps, "hand: transition ends the pass before the decision");
	Check(cut.ne == 0, "hand: shadow");
	// Off: [VB, EndPass(r), DeferredFlush(r), Barrier(r), PushDesc, PushDraw, Dyn, Begin, BindPipe, DrawCmd]; split: VB after the barrier.
	Check(off.log.size() == 10 && off.log[0].kind == VB && cut.log[3].kind == VB && cut.log[2].kind == Barrier, "hand: the logs have the expected shape");
	// A full pending list: the pass end after the deferred mark.
	World f;
	f.pending = {{0, 1}, {2, 3}, {4, 5}};
	f.deferred = true;
	DrawDesc g;
	g.id           = 2;
	g.pass         = 1;
	g.write_stages = true;
	g.writes       = {{10, 12}};
	World off2 = f, cut2 = f;
	RunStream(off2, {g}, false, false);
	RunStream(cut2, {g}, true, false);
	Check(off2.log == cut2.log && !cut2.open && cut2.pending.empty(), "hand: a full pending list ends the pass after the draw");
}

// The laps of RunTail inside a draw scope (classify): Acq, Cbr, P, Tail get the time of their ops; P is clean.
void TailLaps() {
	PS::g_ticks = &FakeTicks;
	World    w;
	DrawDesc d;
	d.id     = 1;
	d.pass   = 0;
	d.rts    = {0};
	d.stages = {Stage {false, {{1, 3}}}};
	d.vb     = true;
	PS::Flush(1.0, [](uint32_t, uint64_t) {});
	{
		PS::DrawScope scope(PS::Mode::Classify);
		g_now += 10; // head
		TailOps ops {w, d};
		Check(PS::RunTail(ops, true, false), "laps: tail declined");
		g_now += 1; // tail
	}
	std::map<uint32_t, uint64_t> got;
	PS::Flush(1.0, [&](uint32_t slot, uint64_t value) { got[slot] += value; });
	const auto sec = [&](PS::Sec s, PS::Bucket b) { return got[PS::SlotSecFirst + static_cast<uint32_t>(s) * 3u + b]; };
	Check(sec(PS::Sec::Head, PS::Clean) == 10, "laps: head");
	Check(sec(PS::Sec::Acq, PS::Hard) == 4 + 3 + 1, "laps: acq (one RT transition, the prologue)");
	Check(sec(PS::Sec::Cbr, PS::Hard) == 3, "laps: cbr (one binding transition)");
	Check(sec(PS::Sec::P, PS::Clean) == 2 + 1 + 5 + 2 + 6, "laps: p");
	Check(sec(PS::Sec::Tail, PS::Clean) == 1, "laps: tail");
	Check(got[PS::SlotPNs] == 16 && got[PS::SlotRNs] == 10 + 8 + 3 + 1, "laps: r / p");
	Check(got[PS::SlotPDrawN] == 1 && got[PS::SlotDrawN] == 1 && got[PS::SlotPHardN] == 0 && got[PS::SlotNopN] == 0, "laps: counts");
	Check(got[PS::SlotLapN] == 4, "laps: laps (acq, cbr, p, tail; the scope's closing lap is not counted)");
	Check(got[PS::SlotMutFirst + static_cast<uint32_t>(PS::Mut::Transit)] == 2, "laps: transitions noted");
	PS::g_ticks = &PS::Ticks;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// The ledger oracle.

struct Expect {
	std::map<uint32_t, uint64_t> slots;
	void Add(uint32_t slot, uint64_t v) {
		if (v != 0) slots[slot] += v;
	}
};

void LedgerOracle() {
	PS::g_ticks         = &FakeTicks;
	PS::g_sample_period = 1; // the replay files every draw
	PS::Flush(1.0, [](uint32_t, uint64_t) {});
	uint64_t scripts = 0;
	for (uint64_t seed = 1; seed <= 4; seed++) {
		std::mt19937_64 rng(seed * 0x51ed270b2f5a4c3dull);
		auto            pick = [&](int n) { return static_cast<int>(rng() % static_cast<uint64_t>(n)); };
		for (int script = 0; script < 3000; script++) {
			Expect e;
			// Oracle state of the open interval.
			const int draws = 1 + pick(6);
			for (int k = 0; k < draws; k++) {
				// Mutations outside a draw: never counted.
				for (int i = pick(3); i > 0; i--) PS::Note(static_cast<PS::Mut>(pick(static_cast<int>(PS::kMutCount))));
				g_now += static_cast<uint64_t>(pick(5));
				const bool     classify = pick(2) == 0;
				PS::DrawScope  scope(classify ? PS::Mode::Classify : PS::Mode::On);
				PS::Sec        cur      = PS::Sec::Head;
				bool           hard = false, soft = false, reached_p = false;
				uint64_t       begin = g_now, laps = 0;
				const auto     close = [&](PS::Sec next) {
                    const int b = hard ? PS::Hard : (soft ? PS::Soft : PS::Clean);
                    e.Add(PS::SlotSecFirst + static_cast<uint32_t>(cur) * 3u + static_cast<uint32_t>(b), g_now - begin);
                    e.Add(cur == PS::Sec::P ? PS::SlotPNs : PS::SlotRNs, g_now - begin);
                    if (cur == PS::Sec::P && b == PS::Hard) e.Add(PS::SlotPHardN, 1);
                    if (cur == PS::Sec::P && b == PS::Soft) e.Add(PS::SlotPSoftN, 1);
                    cur   = next;
                    begin = g_now;
                    hard = soft = false;
				};
				const int steps = pick(12);
				for (int s = 0; s < steps; s++) {
					g_now += static_cast<uint64_t>(pick(50));
					switch (pick(6)) {
						case 0:
						case 1: { // a mutation
							const auto kind = static_cast<PS::Mut>(pick(static_cast<int>(PS::kMutCount)));
							PS::Note(kind);
							(PS::kMutHard[static_cast<size_t>(kind)] ? hard : soft) = true;
							e.Add(PS::SlotMutFirst + static_cast<uint32_t>(kind), 1);
							break;
						}
						case 2: { // a classify lap (ignored by a split-only draw)
							const auto next = static_cast<PS::Sec>(1 + pick(9)); // job..cbr
							PS::LapC(next);
							if (classify) {
								close(next);
								laps++;
							}
							break;
						}
						case 3:
							if (!reached_p) {
								PS::EnterP();
								close(PS::Sec::P);
								laps++;
								reached_p = true;
								e.Add(PS::SlotPDrawN, 1);
							} else {
								PS::Lap(PS::Sec::Tail);
								close(PS::Sec::Tail);
								laps++;
							}
							break;
						case 4: { // a nested scope (a draw inside a draw): nothing
							PS::DrawScope inner(PS::Mode::Classify);
							g_now += 7;
							break;
						}
						default:
							PS::NoteSerial();
							e.Add(PS::SlotSerialN, 1);
							break;
					}
				}
				g_now += static_cast<uint64_t>(pick(20));
				close(PS::Sec::Head); // the scope's end, below
				e.Add(PS::SlotLapN, laps);
				e.Add(PS::SlotDrawN, 1);
				if (!reached_p) e.Add(PS::SlotNopN, 1);
			}
			std::map<uint32_t, uint64_t> got;
			PS::Flush(1.0, [&](uint32_t slot, uint64_t value) { got[slot] += value; });
			// Head/R include the closing interval, which the scope files when it ends (after `close` above computed it at the same time).
			Check(got == e.slots, "ledger: the flushed slots differ from the replay");
			scripts++;
		}
	}
	// Uncalibrated clock: nothing moves, the ledger keeps its counts.
	{
		PS::DrawScope scope(PS::Mode::On);
	}
	bool moved = false;
	PS::Flush(0.0, [&](uint32_t, uint64_t) { moved = true; });
	Check(!moved, "ledger: an uncalibrated flush moved values");
	std::map<uint32_t, uint64_t> got;
	PS::Flush(2.0, [&](uint32_t slot, uint64_t value) { got[slot] += value; });
	Check(got[PS::SlotDrawN] == 1, "ledger: the draw kept across an uncalibrated flush");
	PS::g_ticks = &PS::Ticks;
	std::printf("ledger oracle: %llu scripts\n", static_cast<unsigned long long>(scripts));
}

// on / shadow time one draw in g_sample_period and count it that many times; classify times all.
void LedgerSampling() {
	PS::g_ticks = &FakeTicks;
	PS::Flush(1.0, [](uint32_t, uint64_t) {});
	for (const uint32_t period: {4u, 8u}) {
		PS::g_sample_period = period;
		for (int i = 0; i < 40 * static_cast<int>(period); i++) {
			PS::DrawScope scope(PS::Mode::On);
			g_now += 10;
			PS::EnterP();
			PS::Note(PS::Mut::Stream);
			g_now += 5;
		}
		std::map<uint32_t, uint64_t> got;
		PS::Flush(1.0, [&](uint32_t slot, uint64_t value) { got[slot] += value; });
		const uint64_t n = 40ull * period;
		Check(got[PS::SlotDrawN] == n && got[PS::SlotPDrawN] == n && got[PS::SlotLapN] == n, "sampling: counts");
		Check(got[PS::SlotRNs] == 10 * n && got[PS::SlotPNs] == 5 * n && got[PS::SlotPSoftN] == n, "sampling: times");
		Check(got[PS::SlotMutFirst + static_cast<uint32_t>(PS::Mut::Stream)] == n, "sampling: mutations");
	}
	PS::g_sample_period = 8;
	for (int i = 0; i < 12; i++) {
		PS::DrawScope scope(PS::Mode::Classify);
		g_now += 3;
	}
	std::map<uint32_t, uint64_t> got;
	PS::Flush(1.0, [&](uint32_t slot, uint64_t value) { got[slot] += value; });
	Check(got[PS::SlotDrawN] == 12 && got[PS::SlotNopN] == 12 && got[PS::SlotRNs] == 36, "sampling: classify times every draw");
	PS::g_ticks = &PS::Ticks;
}

void ModeTable() {
	Check(PS::ParseMode(nullptr) == PS::Mode::Off && PS::ParseMode("") == PS::Mode::Off && PS::ParseMode("0") == PS::Mode::Off, "mode: off");
	Check(PS::ParseMode("1") == PS::Mode::On && PS::ParseMode("on") == PS::Mode::On, "mode: on");
	Check(PS::ParseMode("shadow") == PS::Mode::Shadow && PS::ParseMode("classify") == PS::Mode::Classify, "mode: shadow / classify");
	Check(PS::ParseMode("2") == PS::Mode::Off && PS::ParseMode("Shadow") == PS::Mode::Off, "mode: unknown is off");
	Check(!PS::NeedsSerial(false, false) && PS::NeedsSerial(true, false) && PS::NeedsSerial(false, true), "serial: mesh gpu and image heap");
	Check(PS::SlotCount == 11 + PS::kSecCount * 3 + PS::kMutCount, "slots: count");
	// Before any draw of a split mode the thread notes nothing (level 0); an off scope does not change that.
	{
		PS::DrawScope off(PS::Mode::Off);
		PS::Note(PS::Mut::Upload);
	}
	Check(PS::t_thread.level == 0 && PS::t_thread.hard == 0 && !PS::t_thread.draw, "level 0: nothing noted");
}

} // namespace

int main() {
#if !defined(KYTY_PASSSPLIT_ORACLE_ONLY)
	ModeTable();
	TailHandCases();
#endif
	TailOracle();
#if !defined(KYTY_PASSSPLIT_ORACLE_ONLY)
	TailLaps();
#endif
	LedgerOracle();
#if !defined(KYTY_PASSSPLIT_ORACLE_ONLY)
	LedgerSampling();
#endif
	std::printf("PassSplitTests: ok\n");
	return 0;
}
