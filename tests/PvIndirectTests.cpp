// S408: KYTY_PV_INDIRECT (pvIndirect.h): the prepared args, the entry life cycle, the cap limit and the order of the recorded commands.
// Mutants (each must fail): CLAMP_OFF (the instance count is not clamped to cap), IGNORE_FIELDS (a draw whose fields differ is not
// zeroed), NO_COOLDOWN (a mismatch does not send the entry back to the CPU read), CAP_NO_GROW (an overflow does not grow cap).

#include "graphics/host_gpu/renderer/pvIndirect.h"

#include <cstdio>
#include <string>

using namespace Libs::Graphics;
using namespace Libs::Graphics::PvIndirect;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

struct FakeRecorder {
	std::string log;
	void        EndPass() { log += "E"; }
	void        BindCompute() { log += "B"; }
	void        Push() { log += "P"; }
	void        Dispatch() { log += "D"; }
	void        BarrierToIndirect() { log += "X"; }
};

int main() {
	Check(ParseMode(nullptr) == Mode::Off && ParseMode("0") == Mode::Off && ParseMode("1") == Mode::On && ParseMode("shadow") == Mode::Shadow &&
	          ParseMode("yes") == Mode::Off,
	      "mode parse: unset = 0");
	Check(EnvMode() == Mode::Off, "without the variable the mode is 0");
	Check(ParseCap(nullptr) == 8192 && ParseCap("4096") == 4096 && ParseCap("0") == 8192 && ParseCap("abc") == 8192 && ParseCap("99999999") == 8192,
	      "cap parse");

	// ---- PrepareArgs: draw A of RESULT-407: 192, N, 0, 0, 0 ----
	const Fields expected {192, 0, 0, 0};
	{
		const auto ok = PrepareArgs({192, 483, 0, 0, 0}, expected, 8192);
		Check(ok.out == std::array<uint32_t, 5> {192, 483, 0, 0, 0}, "ok: args copied");
		Check(ok.status.mismatch == 0 && ok.status.overflow == 0 && ok.status.max_instance == 483 && ok.status.draws == 1, "ok: status");
	}
	{
		const auto over = PrepareArgs({192, 20000, 0, 0, 0}, expected, 8192);
		Check(over.out[1] == 8192, "overflow: instance count clamped to cap");
		Check(over.out[0] == 192 && over.status.overflow == 1 && over.status.max_instance == 20000 && over.status.mismatch == 0, "overflow: counted, draw kept");
		Check(PrepareArgs({192, 8192, 0, 0, 0}, expected, 8192).status.overflow == 0, "exactly cap is no overflow");
	}
	// Every field that differs zeroes the draw.
	const Args base {192, 100, 0, 0, 0};
	Args       variants[4] = {base, base, base, base};
	variants[0].index_count    = 193;
	variants[1].first_index    = 3;
	variants[2].vertex_offset  = 0xfffffff0u; // negative offset
	variants[3].first_instance = 1;
	for (const auto& variant: variants) {
		const auto result = PrepareArgs(variant, expected, 8192);
		Check(result.out == std::array<uint32_t, 5> {0, 0, 0, 0, 0}, "a differing field zeroes the whole draw");
		Check(result.status.mismatch == 1 && result.status.overflow == 0, "a differing field is a mismatch");
	}
	Check(PrepareArgs({192, 0, 0, 0, 0}, expected, 8192).out[1] == 0, "zero instances stay zero");

	// ---- entry life cycle ----
	Entry entry;
	Check(Decide(entry, 100) == Verdict::Learn, "unknown args: learn");
	Check(!Learn(entry, {0, 5, 0, 0, 0}, 192, 8192) && !entry.learned, "index_count 0 cannot be prepared");
	Check(!Learn(entry, {192, 5, 4, 0, 0}, 192, 8192) && !entry.learned, "first_index != 0 cannot be prepared");
	Check(!Learn(entry, {192, 5, 0, 0, 2}, 192, 8192) && !entry.learned, "first_instance != 0 cannot be prepared");
	Check(!Learn(entry, {300, 5, 0, 0, 0}, 192, 8192) && !entry.learned, "index_count above INDEX_BUFFER_SIZE cannot be prepared");
	Check(!Learn(entry, {192, 5, 0, 0, 0}, 0, 8192) && !entry.learned, "unknown index buffer size cannot be prepared");
	Check(Learn(entry, {192, 483, 0, 0, 0}, 192, 8192), "draw A args are preparable");
	Check(entry.fields == expected && entry.cap == 8192 && entry.max_seen == 483, "learned fields and the initial cap");
	Check(Decide(entry, 6300) == Verdict::Use, "learned: use");
	Check(InitialCap(483, 8192) == 8192 && InitialCap(7660, 8192) == 16384 && InitialCap(8065, 4096) == 16384 && InitialCap(1, 100) == 100,
	      "initial cap: env cap or 2 * max seen as a power of two");

	// A mismatch (read two flips later) sends the entry back to the CPU for 64 flips, then it is used again.
	const auto status_mismatch = PrepareArgs({192, 100, 0, 0, 1}, expected, 8192).status;
	const auto effect          = OnStatus(entry, status_mismatch, 6302, 100000);
	Check(effect.cooldown && !effect.grew, "mismatch: cooldown");
	Check(Decide(entry, 6303) == Verdict::Cooldown && Decide(entry, 6302 + kCooldownFlips - 1) == Verdict::Cooldown, "cooldown lasts 64 flips");
	Check(Decide(entry, 6302 + kCooldownFlips) == Verdict::Use, "after the cooldown the entry is used again");
	Check(Learn(entry, {192, 50, 0, 0, 0}, 192, 8192) && Decide(entry, 6303) == Verdict::Cooldown, "learning during the cooldown does not end it");

	// Overflow grows cap (power of two, twice the largest count) up to the limit; beyond it the entry is off.
	Entry grow;
	Learn(grow, {192, 10, 0, 0, 0}, 192, 8192);
	const auto overflow_status = PrepareArgs({192, 9000, 0, 0, 0}, expected, grow.cap).status;
	const auto grown           = OnStatus(grow, overflow_status, 10, 1u << 20);
	Check(grown.grew && !grown.off && grow.cap == 32768, "overflow 9000 > 8192: cap = next_pow2(2 * 9000) = 32768");
	Check(grow.max_seen == 9000, "max seen follows the status");
	Entry limited;
	Learn(limited, {192, 10, 0, 0, 0}, 192, 8192);
	const auto limited_effect = OnStatus(limited, PrepareArgs({192, 9000, 0, 0, 0}, expected, limited.cap).status, 10, 12000);
	Check(limited_effect.grew && limited.cap == 12000, "growth stops at the cap limit");
	Entry off;
	Learn(off, {192, 10, 0, 0, 0}, 192, 8192);
	const auto off_effect = OnStatus(off, PrepareArgs({192, 30000, 0, 0, 0}, expected, off.cap).status, 10, 12000);
	Check(off_effect.off && off.off && Decide(off, 1000000) == Verdict::Off, "more instances than the limit: the entry is off for good");

	// ---- limits ----
	Check(CapLimit(1u << 22, 45, 4) == (((1u << 22) / 4) - 1) / 45, "cap limit from the slab");
	Check(CapLimit(1u << 22, 0, 4) == 0 && CapLimit(1u << 22, 45, 0) == 0, "cap limit: no span / stride");
	{
		const uint32_t limit = CapLimit(1u << 22, 45, 4);
		Check(uint64_t {limit} * 45 * 4 + 4 <= (1u << 22), "a region of cap_limit instances fits the slab");
	}

	// ---- plans ----
	PvVsStorePlan cap_plan, real_plan;
	cap_plan.verdict = real_plan.verdict = PvVsStoreVerdict::Use;
	cap_plan.span = real_plan.span = 45;
	cap_plan.first = real_plan.first = 7;
	cap_plan.records = 45 * 8192;
	real_plan.records = 45 * 483;
	Check(PlanEq(cap_plan, real_plan), "plan with cap = the real plan");
	real_plan.records = 45 * 9000;
	Check(!PlanEq(cap_plan, real_plan), "the real draw has more records than the region");
	real_plan.records = 45 * 483;
	real_plan.first = 8;
	Check(!PlanEq(cap_plan, real_plan), "different first vertex");

	// ---- the host path ----
	IndirectGpu::HostPath host;
	host.pv_store_possible = true;
	host.proto_possible    = true;
	Check(IndirectGpu::QualifyHost(host) == IndirectGpu::Reason::PerVertexStore, "QualifyHost declines the VS-store pair");
	Check(ResidualReason(host) == IndirectGpu::Reason::Eligible, "nothing but the VS-store / prototype in the way");
	host.restart = true;
	Check(ResidualReason(host) == IndirectGpu::Reason::PrimitiveRestart, "restart control stays a reason");
	host.restart       = false;
	host.native_fetch  = true;
	Check(ResidualReason(host) == IndirectGpu::Reason::NativeFetch, "native fetch stays a reason");

	// ---- recorded order: (end pass) bind, push, dispatch, barrier; nothing is drawn by the prepare ----
	{
		FakeRecorder open;
		Check(RecordPrepare(open, true) && open.log == "EBPDX", "pass open: end it, then bind / push / dispatch / barrier");
		FakeRecorder closed;
		Check(!RecordPrepare(closed, false) && closed.log == "BPDX", "pass closed: no pass break");
		Check(open.log.find('X') > open.log.find('D'), "the barrier follows the dispatch");
	}

	if (g_failures != 0) return 1;
	std::fprintf(stderr, "PvIndirectTests: ok\n");
	return 0;
}
