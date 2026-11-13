#include "graphics/host_gpu/renderer/w2Diag.h"

#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/cpWaitDiag.h"
#include "graphics/presentation/cpSampler.h"
#include "graphics/presentation/frameTiming.h"

#include <atomic>
#include <cinttypes>
#include <cstdio>

namespace Libs::Graphics::W2Diag {

namespace {

SiteTable<>&                         Table() {
	static SiteTable<> table;
	return table;
}
std::atomic<const CommandScheduler*> g_renderer {nullptr};
uint64_t                             g_last_dump_bucket = 0; // CP thread only
constexpr uint64_t                   kDumpEvery         = 100; // guest flips
constexpr size_t                     kDumpRows          = 48;

// stderr, not LOGF: the bench log is the emulator's stderr (LOGF is silent there).
void Dump(uint64_t flip) {
	const auto rows  = Table().Snapshot();
	uint64_t   total = 0;
	for (const auto& r: rows) total += r.cp + r.other;
	std::fprintf(stderr, "[w2-diag] flip=%" PRIu64 " handle_total=%" PRIu64 " sites=%zu dropped=%" PRIu64 "\n", flip, total, rows.size(), Table().Dropped());
	for (size_t i = 0; i < rows.size() && i < kDumpRows; i++) {
		std::fprintf(stderr, "[w2-diag-site] flip=%" PRIu64 " rank=%zu cp=%" PRIu64 " other=%" PRIu64 " site=%s:%u\n", flip, i + 1, rows[i].cp, rows[i].other,
		     rows[i].file.c_str(), rows[i].line);
	}
}

} // namespace

void NoteHandle(const char* file, uint32_t line) noexcept {
	const bool cp = GuestGpu::IsGpuThread();
	Table().Add(file, line, cp);
	FrameTiming::AddAlways(cp ? FrameTiming::Counter::W2HandleN : FrameTiming::Counter::W2HandleOtherN);
	if (!cp) return;
	const uint64_t bucket = FrameTiming::GuestFlips() / kDumpEvery;
	if (bucket != g_last_dump_bucket) {
		if (g_last_dump_bucket == 0) {
			std::fprintf(stderr, "[w2-diag] on: Handle() sites, CP waits, destroy_unsub (KYTY_W2_DIAG=1), dump every %" PRIu64 " flips\n", kDumpEvery);
		}
		g_last_dump_bucket = bucket;
		Dump(bucket * kDumpEvery);
		std::fflush(stderr);
	}
}

void NoteDrainSite(const char* file, uint32_t line) noexcept {
	static SiteTable<512> table;
	static uint64_t       last_bucket = 0; // one thread at a time: the drains of the recording thread
	const bool            cp          = GuestGpu::IsGpuThread();
	table.Add(file, line, cp);
	if (!cp) return;
	const uint64_t bucket = FrameTiming::GuestFlips() / kDumpEvery;
	if (bucket == last_bucket) return;
	if (last_bucket == 0) std::fprintf(stderr, "[emit-drain-site] on (KYTY_CP_EMIT_SITES=1): Handle() drains of a non-empty IR, cumulative, every %" PRIu64 " flips\n", kDumpEvery);
	last_bucket     = bucket;
	const auto rows = table.Snapshot();
	for (size_t i = 0; i < rows.size() && i < kDumpRows; i++) {
		std::fprintf(stderr, "[emit-drain-site] flip=%" PRIu64 " rank=%zu cp=%" PRIu64 " other=%" PRIu64 " site=%s:%u\n", bucket * kDumpEvery, i + 1, rows[i].cp,
		             rows[i].other, rows[i].file.c_str(), rows[i].line);
	}
	std::fflush(stderr);
}

void NoteWait(bool not_free, bool blocked) noexcept {
	using C = FrameTiming::Counter;
	FrameTiming::AddAlways(C::W2WaitCallN);
	if (!not_free) return;
	FrameTiming::AddAlways(C::W2WaitNotfreeN);
	const auto cause = static_cast<uint32_t>(CpWait::CurrentCause());
	if (cause < CpWait::kCauseCount) {
		FrameTiming::AddAlways(static_cast<C>(static_cast<uint32_t>(C::W2WaitRfGpuN) + cause));
	}
	if (blocked) FrameTiming::AddAlways(C::W2WaitBlockN);
}

void NoteSubmitWait() noexcept {
	if (!GuestGpu::IsGpuThread()) return;
	FrameTiming::AddAlways(FrameTiming::Counter::W2WaitSubmitN);
}

void NoteDestroy(const char* kind) noexcept {
	using C = FrameTiming::Counter;
	if (!GuestGpu::IsGpuThread()) {
		FrameTiming::AddAlways(C::W2DestroyOtherThreadN);
		return;
	}
	if (CommandScheduler::InDeferredOperation()) {
		FrameTiming::AddAlways(C::W2DestroyDeferredN);
		return;
	}
	const auto* scheduler = g_renderer.load(std::memory_order_acquire);
	if (scheduler == nullptr || !scheduler->OpenBufferRecorded()) {
		FrameTiming::AddAlways(C::W2DestroyImmediateN);
		return;
	}
	FrameTiming::AddAlways(C::W2DestroyUnsubN);
	static uint32_t logged = 0; // CP thread only
	if (logged < 32) {
		logged++;
		std::fprintf(stderr, "[w2-diag-destroy] flip=%" PRIu64 " kind=%s cp_site=%s opcode=0x%x\n", FrameTiming::GuestFlips(), kind,
		     CpSampler::SiteName(CpSampler::CpSite().load(std::memory_order_relaxed)), CpSampler::CpOpcode().load(std::memory_order_relaxed));
	}
}

void SetRendererScheduler(const CommandScheduler* scheduler) noexcept {
	g_renderer.store(scheduler, std::memory_order_release);
}

} // namespace Libs::Graphics::W2Diag
