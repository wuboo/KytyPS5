#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RDHAZARD_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RDHAZARD_H_

// S477 D2 (KYTY_DIAG_RDHAZARD=1, DIAG, default off): oracle of the read hazard of "the worker reads after the front".
// Right after a GetGraphicsPrograms/GetComputeProgram stage walked its SRT on the CP (SrtMemo: every guest read with its value) the reads are copied
// here with the draw number. Later the same addresses are read again from the backing (the worker's view: no drain, no wait) at four points:
//   t0   immediately (the backing the worker would see right now vs what the CP saw after its drain)
//   k    32 draws later (the worker is a little behind)
//   sig  before the next signal write of the stream (RELEASE_MEM/EOP: the fifo policy executes it only after the reads)
//   end  at the end of the Process() call (the parse policy: signals and cpu writes of the whole submission already happened)
// A read is "dirty" when the GPU thread has unflushed / in-flight bytes there (the worker would bail out: no hazard), "ne" when it is not dirty and the
// bytes differ from the CP's (a real hazard), "busy" when the backing cannot be read now. Counts are reads and draws-stages, printed with LOGF at the
// flips of the variable's value (default 6300,6800,6990) and reset. Needs KYTY_SRT_FAST=shadow semantics for the memo (forced by SrtFast::Mode()).
// Everything runs on the CP thread; nothing here changes what the CP does.

#include <algorithm>
#include <array>
#include <span>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/diagOut.h"
#include "common/logging/log.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/recompiler/ir/SrtMemo.h"
#include "graphics/shader/recompiler/srtReader.h"
#include "kernel/memory.h"

namespace Libs::Graphics::RdHazard {

enum Point : uint32_t { T0, K, SIG, END, kPoints };
template <typename... A>
inline void DiagPrintf(const char* fmt, A... args) {
	char line[700];
	std::snprintf(line, sizeof(line), fmt, args...);
	Common::DiagOut::Write(line);
}

inline const char* PointName(uint32_t p) {
	static const char* const n[] = {"t0", "k32", "sig", "end"};
	return n[p];
}

struct Rec {
	uint64_t                                      draw  = 0;
	uint32_t                                      stage = 0;
	uint32_t                                      done  = 0; // bit per Point
	std::vector<ShaderRecompiler::IR::SrtMemo::Read> reads;
	std::vector<uint32_t>                         words;
};

struct Counts {
	uint64_t recs = 0, reads = 0, ne_reads = 0, dirty_reads = 0, busy_reads = 0, failed_reads = 0;
	uint64_t recs_ne = 0, recs_dirty = 0, recs_busy = 0;
};

struct State {
	bool                  on = false, inited = false;
	uint64_t              draw = 0, stage_n[3] = {};
	uint64_t              nomemo = 0;
	std::vector<Rec>      recs;
	std::array<Counts, kPoints> c {};
	uint64_t              window_begin = 0, logged = 0;
	std::vector<uint64_t> bounds;
	size_t                next_bound = 0;
};

inline State& S() {
	static State s;
	return s;
}

inline bool Enabled() {
	auto& s = S();
	if (!s.inited) {
		s.inited      = true;
		const char* v = std::getenv("KYTY_DIAG_RDHAZARD");
		if (v != nullptr && v[0] != '\0' && std::strcmp(v, "0") != 0) {
			s.on            = true;
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

inline void CheckRec(State& s, Rec& r, Point point) {
	if (r.done & (1u << point)) return;
	r.done |= 1u << point;
	auto&    c = s.c[point];
	bool     any_ne = false, any_dirty = false, any_busy = false;
	c.recs++;
	std::array<uint32_t, 32> small_buf;
	std::vector<uint32_t>    large;
	for (const auto& read: r.reads) {
		c.reads++;
		if (!read.ok) {
			c.failed_reads++;
			continue;
		}
		std::span<uint32_t> out;
		if (read.count <= small_buf.size()) {
			out = std::span(small_buf).first(read.count);
		} else {
			large.resize(read.count);
			out = large;
		}
		if (read.ordinary && read.address < SrtReader::kGuestMemoryMin) {
			std::fill(out.begin(), out.end(), 0u);
		} else {
			const uint64_t bytes = static_cast<uint64_t>(read.count) * 4u;
			if (Libs::LibKernel::Memory::QueryShaderGpuDirty(read.address, bytes)) {
				c.dirty_reads++;
				any_dirty = true;
				continue;
			}
			if (!Libs::LibKernel::Memory::TryReadBacking(read.address, out.data(), bytes)) {
				c.busy_reads++;
				any_busy = true;
				continue;
			}
		}
		const auto* expect = r.words.data() + read.first;
		if (std::memcmp(out.data(), expect, static_cast<size_t>(read.count) * 4u) != 0) {
			c.ne_reads++;
			any_ne = true;
			if (s.logged < 24) {
				s.logged++;
				uint32_t i = 0;
				while (i < read.count && out[i] == expect[i]) i++;
				DiagPrintf("[rd-hazard] ne point=%s stage=%u draw=%llu addr=0x%llx dw=%u/%u cp=%08x now=%08x\n", PointName(point), r.stage,
				     static_cast<unsigned long long>(r.draw), static_cast<unsigned long long>(read.address), i, read.count, expect[i], out[i]);
			}
		}
	}
	if (any_ne) c.recs_ne++;
	if (any_dirty) c.recs_dirty++;
	if (any_busy) c.recs_busy++;
}

// The stage just walked on the CP. Copies the memo and does the t0 check.
inline void NoteStage(uint32_t stage, const ShaderRecompiler::IR::SrtMemo& memo) {
	auto& s = S();
	if (stage < 3) s.stage_n[stage]++;
	if (!memo.valid) {
		s.nomemo++;
		return;
	}
	Rec r;
	r.draw  = s.draw;
	r.stage = stage;
	r.reads = memo.reads;
	r.words = memo.words;
	CheckRec(s, r, T0);
	s.recs.push_back(std::move(r));
}

inline void OnDraw() {
	auto& s = S();
	s.draw++;
	for (auto& r: s.recs) {
		if (r.draw + 32 <= s.draw) CheckRec(s, r, K);
	}
}

inline void AtSignal() {
	auto& s = S();
	for (auto& r: s.recs) CheckRec(s, r, SIG);
}

inline void AtCallEnd() {
	auto& s = S();
	for (auto& r: s.recs) {
		CheckRec(s, r, K); // a record younger than 32 draws at the end of the call: its k point is the end
		CheckRec(s, r, SIG);
		CheckRec(s, r, END);
	}
	s.recs.clear();
}

inline void MaybeDump(uint64_t flip) {
	auto& s = S();
	if (s.next_bound >= s.bounds.size() || flip < s.bounds[s.next_bound]) return;
	s.next_bound++;
	const double fl = static_cast<double>(flip > s.window_begin ? flip - s.window_begin : 1);
	DiagPrintf("[rd-hazard] window flips %llu-%llu stage_walks vs=%.1f ps=%.1f cs=%.1f per flip, no_memo=%.2f per flip\n", static_cast<unsigned long long>(s.window_begin),
	     static_cast<unsigned long long>(flip), static_cast<double>(s.stage_n[0]) / fl, static_cast<double>(s.stage_n[1]) / fl, static_cast<double>(s.stage_n[2]) / fl,
	     static_cast<double>(s.nomemo) / fl);
	for (uint32_t p = 0; p < kPoints; p++) {
		const auto& c = s.c[p];
		DiagPrintf("[rd-hazard] point=%-4s recs=%llu reads=%llu failed=%llu dirty_reads=%llu busy_reads=%llu NE_READS=%llu recs_ne=%llu recs_dirty=%llu recs_busy=%llu (ne recs %.4f%%)\n",
		     PointName(p), static_cast<unsigned long long>(c.recs), static_cast<unsigned long long>(c.reads), static_cast<unsigned long long>(c.failed_reads),
		     static_cast<unsigned long long>(c.dirty_reads), static_cast<unsigned long long>(c.busy_reads), static_cast<unsigned long long>(c.ne_reads),
		     static_cast<unsigned long long>(c.recs_ne), static_cast<unsigned long long>(c.recs_dirty), static_cast<unsigned long long>(c.recs_busy),
		     c.recs ? 100.0 * static_cast<double>(c.recs_ne) / static_cast<double>(c.recs) : 0.0);
	}
	s.c = {};
	s.stage_n[0] = s.stage_n[1] = s.stage_n[2] = 0;
	s.nomemo       = 0;
	s.window_begin = flip;
}

} // namespace Libs::Graphics::RdHazard

#endif
