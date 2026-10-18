// S408 (KYTY_PV_INDIRECT): the per-address state and the status reports of the prepared args of VS-store indirect draws.
// The decision rules are in pvIndirect.h; the recording of a draw is in renderDraw.cpp (ExecutePreparedDraw).

#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/perVertexVsStoreRuntime.h"
#include "graphics/host_gpu/renderer/pvIndirectRuntime.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/frameTiming.h"

#include <atomic>
#include <cinttypes>

namespace Libs::Graphics {

PvIndirectRuntime& RenderExecutor::PvIndirectRt() {
	if (!m_pvi_rt) m_pvi_rt = std::make_shared<PvIndirectRuntime>(m_context.GetGraphics(), m_context.GetCommandScheduler());
	return *m_pvi_rt;
}

PvIndirect::RangeEntry* RenderExecutor::PvIndirectRangeEntry(uint32_t entry_id) {
	return entry_id < m_pvi_count ? &m_pvi_entries[entry_id].entry.range : nullptr;
}

PvIndirectDraw RenderExecutor::PvIndirectDecide(const PvIndirectRequest& request, bool pv_candidate) {
	using C = FrameTiming::Counter;
	PvIndirectDraw draw;
	if (!pv_candidate || request.mode == PvIndirect::Mode::Off) return draw;
	PvIndirectAdvance();

	PvIndirectEntryState* state = nullptr;
	for (uint32_t i = 0; i < m_pvi_count; i++) {
		if (m_pvi_entries[i].addr == request.args_addr) {
			state = &m_pvi_entries[i];
			break;
		}
	}
	if (state == nullptr && m_pvi_count < PvIndirect::kMaxEntries) {
		state             = &m_pvi_entries[m_pvi_count];
		state->addr       = request.args_addr;
		state->entry.id   = m_pvi_count++;
	}
	if (state == nullptr) {
		// More args addresses than status slots: these stay on the old path.
		if (!request.have_cpu_args) {
			FrameTiming::Add(C::PvIndDeclineN);
			FrameTiming::Add(C::PvIndDeclineOtherN);
		}
		return draw;
	}
	auto&          entry   = state->entry;
	const uint64_t flip    = FrameTiming::GuestFlips();
	const auto     verdict = PvIndirect::Decide(entry, flip);
	const auto     before  = entry.fields;
	const auto     count   = [&](PvIndirect::Verdict v) {
        switch (v) {
            case PvIndirect::Verdict::Learn: FrameTiming::Add(C::PvIndLearnN); break;
            case PvIndirect::Verdict::Cooldown: FrameTiming::Add(C::PvIndCooldownN); break;
            case PvIndirect::Verdict::Off:
                FrameTiming::Add(C::PvIndDeclineN);
                FrameTiming::Add(C::PvIndDeclineOffN);
                break;
            case PvIndirect::Verdict::Use: break;
        }
	};

	if (request.have_cpu_args) {
		// The old path: the CPU has read the args. Learn them (also during a cooldown, so the fields are the current ones when it ends).
		const bool learned = PvIndirect::Learn(entry, request.cpu_args, request.index_buffer_size, PvIndirect::EnvCap());
		if (learned && PvIndirect::EnvRangeMode() != PvIndirect::RangeMode::Off) {
			// S413: this draw's index scan runs after the args drain; its key (base, learned count and offset) is the key of the On draw.
			draw.range_record = true;
			draw.args_addr    = request.args_addr;
			draw.entry_id     = entry.id;
		}
		if (request.mode != PvIndirect::Mode::Shadow) return draw;
		count(verdict);
		auto& seen = state->seen[flip % state->seen.size()];
		seen       = {flip, true, request.cpu_args, false};
		if (verdict != PvIndirect::Verdict::Use || !learned) return draw;
		draw.mode      = PvIndirect::Mode::Shadow;
		draw.args_addr = request.args_addr;
		draw.expected  = before;
		draw.cap       = entry.cap;
		draw.entry_id  = entry.id;
		return draw;
	}
	// The first call of a candidate: only mode On takes the draw.
	if (request.mode != PvIndirect::Mode::On) return draw;
	count(verdict);
	if (verdict != PvIndirect::Verdict::Use) return draw;
	draw.mode      = PvIndirect::Mode::On;
	draw.args_addr = request.args_addr;
	draw.expected  = entry.fields;
	draw.cap       = entry.cap;
	draw.entry_id  = entry.id;
	return draw;
}

void RenderExecutor::PvIndirectNotePlan(const PvIndirectDraw& draw, const PvVsStorePlan& cap_plan, const PvVsStorePlan& real_plan,
                                        uint32_t stride_vec4) {
	using C = FrameTiming::Counter;
	if (draw.entry_id >= m_pvi_count) return;
	auto& state = m_pvi_entries[draw.entry_id];
	if (cap_plan.Use()) state.cap_limit = PvIndirect::CapLimit(PvVsStoreRuntime::SlabVec4s(), cap_plan.span, stride_vec4);
	if (draw.mode == PvIndirect::Mode::Shadow) {
		FrameTiming::Add(PvIndirect::PlanEq(cap_plan, real_plan) ? C::PvIndShadowPlanEqN : C::PvIndShadowPlanNeN);
	}
}

void RenderExecutor::PvIndirectAdvance() {
	if (!m_pvi_rt) return;
	using C                = FrameTiming::Counter;
	const uint64_t now     = FrameTiming::GuestFlips();
	const bool     shadow  = PvIndirect::EnvMode() == PvIndirect::Mode::Shadow;
	m_pvi_rt->AdvanceFlip(now, [&](const PvIndirectRuntime::Report& report) {
		if (report.entry_id >= m_pvi_count) return;
		auto& state = m_pvi_entries[report.entry_id];
		const auto& status = report.status;
		const auto  expected = state.entry.fields;
		const auto  cap_before = state.entry.cap;
		const auto  effect = PvIndirect::OnStatus(state.entry, status, now, state.cap_limit);
		FrameTiming::Add(C::PvIndRangeViolationN, status.range_violation);
		FrameTiming::Set(C::PvIndCap, state.entry.cap);
		FrameTiming::Max(C::PvIndMaxInstance, status.max_instance);
		static std::atomic<uint32_t> logged {0};
		if (shadow) {
			FrameTiming::Add(C::PvIndShadowN, status.draws);
			FrameTiming::Add(C::PvIndShadowFieldsOkN, status.draws - std::min(status.draws, status.mismatch));
			FrameTiming::Add(C::PvIndShadowFieldsMismatchN, status.mismatch);
			FrameTiming::Add(C::PvIndShadowOverflowN, status.overflow);
			auto& seen = state.seen[report.flip % state.seen.size()];
			if (status.draws > 1) {
				FrameTiming::Add(C::PvIndShadowMultiN);
			} else if (status.draws == 1 && seen.have && seen.flip == report.flip) {
				const bool equal = seen.cpu_args == report.last_in;
				FrameTiming::Add(equal ? C::PvIndShadowArgsEqN : C::PvIndShadowArgsNeN);
				if (!equal && logged.fetch_add(1, std::memory_order_relaxed) < 20) {
					LOGF("[pv-ind-args-ne] flip=%" PRIu64 " addr=0x%" PRIx64 " cpu=%u,%u,%u,%u,%u gpu=%u,%u,%u,%u,%u\n", report.flip, state.addr,
					     seen.cpu_args.index_count, seen.cpu_args.instance_count, seen.cpu_args.first_index, seen.cpu_args.vertex_offset,
					     seen.cpu_args.first_instance, report.last_in.index_count, report.last_in.instance_count, report.last_in.first_index,
					     report.last_in.vertex_offset, report.last_in.first_instance);
				}
			}
			if (status.draws != 0) seen.have = false;
		} else {
			FrameTiming::Add(C::PvIndMismatchN, status.mismatch);
			FrameTiming::Add(C::PvIndOverflowN, status.overflow);
		}
		if ((status.mismatch != 0 || status.overflow != 0 || status.range_violation != 0) && logged.fetch_add(1, std::memory_order_relaxed) < 20) {
			LOGF("[pv-ind-status] flip=%" PRIu64 " addr=0x%" PRIx64 " draws=%u mismatch=%u overflow=%u max_instance=%u expected=%u,%u,%u,%u "
			     "in=%u,%u,%u,%u,%u cap=%u->%u cooldown=%d off=%d range_violation=%u\n",
			     report.flip, state.addr, status.draws, status.mismatch, status.overflow, status.max_instance, expected.index_count,
			     expected.first_index, expected.vertex_offset, expected.first_instance, report.last_in.index_count,
			     report.last_in.instance_count, report.last_in.first_index, report.last_in.vertex_offset, report.last_in.first_instance,
			     cap_before, state.entry.cap, effect.cooldown ? 1 : 0, effect.off ? 1 : 0, status.range_violation);
		}
	});
}

} // namespace Libs::Graphics
