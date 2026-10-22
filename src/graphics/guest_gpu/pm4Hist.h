#ifndef KYTY_GRAPHICS_GUEST_GPU_PM4HIST_H_
#define KYTY_GRAPHICS_GUEST_GPU_PM4HIST_H_

// S477 D1 (KYTY_DIAG_PM4_HIST): histogram of the PM4 opcodes the CP walks, classification of every packet for the R13 pipeline plan
// (draw / dispatch / register write / signal / cpu write / wait / predicate / marker / other) and the lengths of the runs of draws between "drain"
// packets under three policies. Pure counters on the CP thread (the only caller), no allocation after the first Enable() check; the report is
// printed with LOGF at the flips listed in the variable (default 6300,6800,6990; "1" = the default) and the counters are reset after each report,
// so a report is the window between two boundaries.
//
// Policies (a run ends at a packet that the policy treats as a drain; its length counts the draws since the previous drain). Signals (RELEASE_MEM/EOP/EOS/
// FLIP) and barriers (EVENT_WRITE, ACQUIRE_MEM, PFP_SYNC_ME, SURFACE_SYNC, MEM_SEMAPHORE) are ordered ops of the FIFO unless the policy says otherwise:
//   A  plan stage 1: drains = dispatch, cpu write (WRITE_DATA/DMA/COPY_DATA), predicate, other (unknown opcodes)
//   B  stage 2 (dispatch in the FIFO): drains = cpu write, predicate, other
//   C  everything ordered in the FIFO, cpu writes wait for reads only: drains = predicate, other
//   D  A + WAIT_REG_MEM                    (the virtual-write map fails every time)
//   E  D + signals + barriers              (executed at parse time: policy "parse")

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "graphics/guest_gpu/pm4.h"

namespace Libs::Graphics::Pm4Hist {

enum class Cls : uint32_t { Draw, Dispatch, Reg, Marker, Signal, Barrier, CpuWrite, Wait, Pred, Struct, Other, Count };

inline const char* ClsName(Cls c) {
	static const char* const n[] = {"draw", "dispatch", "reg", "marker", "signal", "barrier", "cpuwrite", "wait", "pred", "struct", "other"};
	return n[static_cast<uint32_t>(c)];
}

// opcode = IT_*, r = the R_* subtype of an IT_NOP (0 otherwise)
inline Cls Classify(uint32_t opcode, uint32_t r) {
	using namespace Pm4;
	switch (opcode) {
		case IT_DRAW_INDEX_2: case IT_DRAW_INDEX_AUTO: case IT_DRAW_INDEX_OFFSET_2: case IT_DRAW_INDIRECT: case IT_DRAW_INDEX_INDIRECT:
		case IT_DRAW_INDIRECT_MULTI: case IT_DRAW_INDEX_INDIRECT_MULTI: case IT_DISPATCH_DRAW: case IT_DISPATCH_DRAW_PREAMBLE:
			return Cls::Draw;
		case IT_DISPATCH_DIRECT: case IT_DISPATCH_INDIRECT: return Cls::Dispatch;
		case IT_SET_SH_REG: case IT_SET_CONTEXT_REG: case IT_SET_UCONFIG_REG: case IT_SET_UCONFIG_REG_INDEX: case IT_SET_CONFIG_REG:
		case IT_SET_SH_REG_INDIRECT: case IT_SET_CONTEXT_REG_INDIRECT: case IT_SET_UCONFIG_REG_INDIRECT: case IT_NUM_INSTANCES:
		case IT_INDEX_TYPE: case IT_INDEX_BASE: case IT_INDEX_BUFFER_SIZE: case IT_SET_BASE: case IT_CLEAR_STATE: case IT_CONTEXT_CONTROL:
			return Cls::Reg;
		case IT_RELEASE_MEM: case IT_EVENT_WRITE_EOP: case IT_EVENT_WRITE_EOS: return Cls::Signal;
		case IT_EVENT_WRITE: case IT_ACQUIRE_MEM: case IT_PFP_SYNC_ME: case IT_SURFACE_SYNC: case IT_MEM_SEMAPHORE: return Cls::Barrier;
		case IT_WRITE_DATA: case IT_DMA_DATA: case IT_COPY_DATA: case IT_CP_DMA: case IT_WRITE_CONST_RAM: case IT_DUMP_CONST_RAM:
			return Cls::CpuWrite;
		case IT_WAIT_REG_MEM: case IT_WAIT_REG_MEM_64: return Cls::Wait;
		case IT_COND_EXEC: case IT_SET_PREDICATION: return Cls::Pred;
		case IT_INDIRECT_BUFFER: case IT_INDIRECT_BUFFER_CNST: return Cls::Struct;
		case IT_NOP:
			switch (r) {
				case R_RELEASE_MEM: case R_FLIP: return Cls::Signal;
				case R_WRITE_DATA: case R_DMA_DATA: return Cls::CpuWrite;
				case R_WAIT_FLIP_DONE: return Cls::Other;
				default: return Cls::Marker;
			}
		default: return Cls::Other;
	}
}

constexpr uint32_t kPolicies = 5;
constexpr uint32_t kBuckets = 10; // run length: 0, 1, 2, 3-4, 5-8, 9-16, 17-32, 33-64, 65-128, >128
inline uint32_t Bucket(uint32_t n) {
	if (n <= 2) return n;
	uint32_t b = 3, hi = 4;
	while (n > hi && b + 1 < kBuckets) {
		hi <<= 1u;
		b++;
	}
	return b;
}
inline const char* BucketName(uint32_t b) {
	static const char* const n[] = {"0", "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", ">128"};
	return n[b];
}

struct Policy {
	uint32_t                          run = 0;
	uint64_t                          runs = 0, draws = 0;
	std::array<uint64_t, kBuckets>    by_run {}, by_draw {}; // runs / draws that sit in a run of that length
	std::array<uint64_t, 256 + 16>    ends {};               // what ended the run: opcode (NOP: 256 + r)
};

struct State {
	std::array<uint64_t, 256 + 16> op_n {}, op_dw {}; // opcode (NOP: 256 + r)
	std::array<uint64_t, static_cast<size_t>(Cls::Count)> cls_n {};
	Policy                          pol[kPolicies];
	uint64_t                        packets = 0;
	uint64_t                        wait_eval = 0, wait_sat = 0, wait_retry = 0, wait_distinct = 0, wait_first_sat = 0;
	uint64_t                        wait_dst_same_call = 0, wait_dst_other_call = 0, wait_dst_none = 0;
	uint64_t                        calls = 0;
	uint64_t                        window_begin = 0;
	uintptr_t                       last_unsat = 0;
	struct Rel { uintptr_t a = 0; uint64_t call = 0; };
	std::array<Rel, 4096>           rel {};
	uint32_t                        rel_pos = 0;
	std::vector<uint64_t>           bounds;
	size_t                          next_bound = 0;
	bool                            on = false, inited = false;
};

inline State& S() {
	static State s;
	return s;
}

inline bool Enabled() {
	auto& s = S();
	if (!s.inited) {
		s.inited = true;
		const char* v = std::getenv("KYTY_DIAG_PM4_HIST");
		if (v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0) {
			s.on = true;
			const char* list = std::strcmp(v, "1") == 0 ? "6300,6800,6990" : v;
			char*       end  = nullptr;
			for (const char* p = list; *p != '\0';) {
				const auto b = std::strtoull(p, &end, 10);
				if (end == p) break;
				s.bounds.push_back(b);
				p = (*end == ',') ? end + 1 : end;
			}
		}
	}
	return s.on;
}

inline void BeginCall() {
	S().calls++;
}

inline void Reset(State& s, uint64_t flip) {
	const auto bounds = s.bounds;
	const auto nb     = s.next_bound;
	s                 = State {};
	s.bounds          = bounds;
	s.next_bound      = nb;
	s.inited          = true;
	s.on              = true;
	s.window_begin    = flip;
}

template <typename Log>
void Dump(State& s, uint64_t flip, Log log) {
	const double fl = static_cast<double>(flip > s.window_begin ? flip - s.window_begin : 1);
	char         b[400];
	std::snprintf(b, sizeof(b), "[pm4-hist] window flips %llu-%llu packets=%llu (%.1f/flip) calls=%llu (%.1f/flip)\n",
	              static_cast<unsigned long long>(s.window_begin), static_cast<unsigned long long>(flip), static_cast<unsigned long long>(s.packets),
	              static_cast<double>(s.packets) / fl, static_cast<unsigned long long>(s.calls), static_cast<double>(s.calls) / fl);
	log(b);
	for (size_t c = 0; c < s.cls_n.size(); c++) {
		std::snprintf(b, sizeof(b), "[pm4-hist] class %-8s n/flip=%.2f\n", ClsName(static_cast<Cls>(c)), static_cast<double>(s.cls_n[c]) / fl);
		log(b);
	}
	for (size_t i = 0; i < s.op_n.size(); i++) {
		if (s.op_n[i] == 0) continue;
		if (i < 256) {
			std::snprintf(b, sizeof(b), "[pm4-hist] op 0x%02zx cls=%s n/flip=%.2f dw/pkt=%.1f\n", i, ClsName(Classify(static_cast<uint32_t>(i), 0)),
			              static_cast<double>(s.op_n[i]) / fl, static_cast<double>(s.op_dw[i]) / static_cast<double>(s.op_n[i]));
		} else {
			std::snprintf(b, sizeof(b), "[pm4-hist] op NOP/R=0x%02zx cls=%s n/flip=%.2f dw/pkt=%.1f\n", i - 256,
			              ClsName(Classify(Pm4::IT_NOP, static_cast<uint32_t>(i - 256))), static_cast<double>(s.op_n[i]) / fl,
			              static_cast<double>(s.op_dw[i]) / static_cast<double>(s.op_n[i]));
		}
		log(b);
	}
	static const char* const pn[] = {"A(stage1)", "B(+dispatch)", "C(all ordered)", "D(A+wait)", "E(parse signals)"};
	for (int p = 0; p < static_cast<int>(kPolicies); p++) {
		auto& pl = s.pol[p];
		// median by run and by draw
		auto median = [&](const std::array<uint64_t, kBuckets>& h, uint64_t total) {
			uint64_t acc = 0;
			for (uint32_t k = 0; k < kBuckets; k++) {
				acc += h[k];
				if (acc * 2 >= total) return k;
			}
			return kBuckets - 1;
		};
		std::snprintf(b, sizeof(b), "[pm4-hist] policy %s runs/flip=%.1f draws/flip=%.1f mean_run=%.2f median_run_bucket(by run)=%s median_run_bucket(by draw)=%s\n", pn[p],
		              static_cast<double>(pl.runs) / fl, static_cast<double>(pl.draws) / fl,
		              pl.runs ? static_cast<double>(pl.draws) / static_cast<double>(pl.runs) : 0.0, BucketName(median(pl.by_run, pl.runs)),
		              BucketName(median(pl.by_draw, pl.draws)));
		log(b);
		std::string line = "[pm4-hist] policy " + std::string(1, "ABCDE"[p]) + " run-length buckets (runs|draws):";
		for (uint32_t k = 0; k < kBuckets; k++) {
			std::snprintf(b, sizeof(b), " %s=%llu|%llu", BucketName(k), static_cast<unsigned long long>(pl.by_run[k]),
			              static_cast<unsigned long long>(pl.by_draw[k]));
			line += b;
		}
		log((line + "\n").c_str());
		line = "[pm4-hist] policy " + std::string(1, "ABCDE"[p]) + " ends by opcode (n/flip):";
		for (size_t i = 0; i < pl.ends.size(); i++) {
			if (pl.ends[i] == 0) continue;
			if (i < 256) std::snprintf(b, sizeof(b), " 0x%02zx=%.2f", i, static_cast<double>(pl.ends[i]) / fl);
			else std::snprintf(b, sizeof(b), " NOP/R%02zx=%.2f", i - 256, static_cast<double>(pl.ends[i]) / fl);
			line += b;
		}
		log((line + "\n").c_str());
	}
	std::snprintf(b, sizeof(b),
	              "[pm4-hist] wait_reg_mem evals/flip=%.2f satisfied=%.2f unsatisfied_evals=%.2f distinct=%.2f first_read_satisfied=%.1f%% dst_is_release_same_call=%.2f dst_is_release_other_call=%.2f dst_unknown=%.2f (per flip)\n",
	              static_cast<double>(s.wait_eval) / fl, static_cast<double>(s.wait_sat) / fl, static_cast<double>(s.wait_eval - s.wait_sat) / fl,
	              static_cast<double>(s.wait_distinct) / fl,
	              s.wait_distinct ? 100.0 * static_cast<double>(s.wait_first_sat) / static_cast<double>(s.wait_distinct) : 0.0,
	              static_cast<double>(s.wait_dst_same_call) / fl, static_cast<double>(s.wait_dst_other_call) / fl, static_cast<double>(s.wait_dst_none) / fl);
	log(b);
}

inline void EndRun(Policy& p, uint32_t end_key) {
	p.runs++;
	p.draws += p.run;
	p.by_run[Bucket(p.run)]++;
	p.by_draw[Bucket(p.run)] += p.run;
	p.ends[end_key]++;
	p.run = 0;
}

// One packet about to be handled.
inline void Packet(uint32_t opcode, uint32_t r, uint32_t dw) {
	auto&      s   = S();
	const auto key = opcode == Pm4::IT_NOP ? 256u + (r & 15u) : (opcode & 255u);
	s.packets++;
	s.op_n[key]++;
	s.op_dw[key] += dw;
	const auto cls = Classify(opcode, r);
	s.cls_n[static_cast<size_t>(cls)]++;
	if (cls == Cls::Draw) {
		for (auto& p : s.pol) p.run++;
		return;
	}
	const bool oth = cls == Cls::Other || cls == Cls::Pred;
	const bool d_a = oth || cls == Cls::Dispatch || cls == Cls::CpuWrite;
	const bool d_b = oth || cls == Cls::CpuWrite;
	const bool d_c = oth;
	const bool d_d = d_a || cls == Cls::Wait;
	const bool d_e = d_d || cls == Cls::Signal || cls == Cls::Barrier;
	if (d_a) EndRun(s.pol[0], key);
	if (d_b) EndRun(s.pol[1], key);
	if (d_c) EndRun(s.pol[2], key);
	if (d_d) EndRun(s.pol[3], key);
	if (d_e) EndRun(s.pol[4], key);
}

// A WAIT_REG_MEM was evaluated (addr = the host address of the watched dword/qword).
inline void Wait(uintptr_t addr, bool satisfied) {
	auto& s = S();
	s.wait_eval++;
	const bool retry = s.last_unsat == addr;
	if (retry) s.wait_retry++;
	else s.wait_distinct++;
	if (satisfied) {
		s.wait_sat++;
		if (!retry) s.wait_first_sat++;
		s.last_unsat = 0;
	} else {
		s.last_unsat = addr;
	}
	if (!retry) {
		bool same = false, other = false;
		for (const auto& e : s.rel) {
			if (e.a == addr) {
				if (e.call == s.calls) same = true;
				else other = true;
			}
		}
		if (same) s.wait_dst_same_call++;
		else if (other) s.wait_dst_other_call++;
		else s.wait_dst_none++;
	}
}

inline void ReleaseDst(uintptr_t addr) {
	auto& s = S();
	s.rel[s.rel_pos++ % s.rel.size()] = {addr, s.calls};
}

template <typename Log>
void MaybeDump(uint64_t flip, Log log) {
	auto& s = S();
	if (s.next_bound < s.bounds.size() && flip >= s.bounds[s.next_bound]) {
		s.next_bound++;
		Dump(s, flip, log);
		Reset(s, flip);
	}
}

} // namespace Libs::Graphics::Pm4Hist

#endif
