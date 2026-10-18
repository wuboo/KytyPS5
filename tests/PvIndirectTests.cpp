// S408: KYTY_PV_INDIRECT (pvIndirect.h): the prepared args, the entry life cycle, the cap limit and the order of the recorded commands.
// Mutants (each must fail): CLAMP_OFF (the instance count is not clamped to cap), IGNORE_FIELDS (a draw whose fields differ is not
// zeroed), RANGE_REUSE_IGNORES_KEY (S413: a remembered range is reused for another index_count), REUSE_IGNORES_PAGE (S417: a protected page with clean bytes does not count as dirty), RANGECHECK_OFF_BY_ONE (S413: the end of the
// range is inclusive), NO_COOLDOWN (a mismatch does not send the entry back to the CPU read), CAP_NO_GROW (an overflow does not grow cap).

#include "graphics/host_gpu/renderer/pvIndirect.h"

#include <cstdio>
#include <string>
#include <vector>

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


	// ---- S413: KYTY_PV_IND_RANGE ----
	Check(ParseRangeMode(nullptr) == RangeMode::Off && ParseRangeMode("0") == RangeMode::Off && ParseRangeMode("1") == RangeMode::On &&
	          ParseRangeMode("shadow") == RangeMode::Shadow && ParseRangeMode("on") == RangeMode::Off,
	      "range mode parse: unset = 0");
	Check(EnvRangeMode() == RangeMode::Off, "without the variable the range mode is 0");
	{
		const RangeKey key {0x571b2da60ull, 192 * 2, 2, 192, 0};
		RangeEntry     entry;
		Check(DecideRangeReuse(RangeMode::On, true, true, entry, key) == RangeDecision::Miss, "nothing remembered: miss");
		Check(!RangeRecordable(false, false, 0, 10) && !RangeRecordable(true, true, 0, 10) && !RangeRecordable(true, false, 5, 5) &&
		          !RangeRecordable(true, false, 0, 0x100000000ull) && RangeRecordable(true, false, 0, 10),
		      "only a derivable, undropped, non-empty range that fits 32 bits is remembered");
		RecordRange(entry, key, 100, 400);
		Check(DecideRangeReuse(RangeMode::On, true, true, entry, key) == RangeDecision::Reuse, "same key, dirty indices: reuse");
		Check(DecideRangeReuse(RangeMode::Off, true, true, entry, key) == RangeDecision::NotApplicable, "mode 0: never");
		Check(DecideRangeReuse(RangeMode::Shadow, true, true, entry, key) == RangeDecision::NotApplicable, "shadow never reuses");
		Check(DecideRangeReuse(RangeMode::On, false, true, entry, key) == RangeDecision::NotApplicable, "not an On draw: never");
		Check(DecideRangeReuse(RangeMode::On, true, false, entry, key) == RangeDecision::NotApplicable, "indices not GPU-dirty: the CPU scan");
		RangeKey other = key;
		other.index_count = 193;
		Check(DecideRangeReuse(RangeMode::On, true, true, entry, other) == RangeDecision::Miss, "another index_count: miss");
		other = key;
		other.vertex_offset = 4;
		Check(DecideRangeReuse(RangeMode::On, true, true, entry, other) == RangeDecision::Miss, "another vertex_offset: miss");
		other = key;
		other.index_addr += 2;
		Check(DecideRangeReuse(RangeMode::On, true, true, entry, other) == RangeDecision::Miss, "another address: miss");
		other = key;
		other.index_bytes += 2;
		other.element_size = 4;
		Check(DecideRangeReuse(RangeMode::On, true, true, entry, other) == RangeDecision::Miss, "another size / element: miss");

		Check(CompareRange(RangeEntry {}, key, {100, 400}) == RangeCompare::NoEntry, "shadow: nothing remembered yet");
		Check(CompareRange(entry, other, {100, 400}) == RangeCompare::KeyDiffers, "shadow: another key");
		Check(CompareRange(entry, key, {100, 400}) == RangeCompare::Equal, "shadow: same range");
		Check(CompareRange(entry, key, {120, 380}) == RangeCompare::NotEqualWithin, "shadow: scan inside the remembered range");
		Check(CompareRange(entry, key, {100, 401}) == RangeCompare::NotEqual && CompareRange(entry, key, {99, 400}) == RangeCompare::NotEqual,
		      "shadow: scan wider than the remembered range");
	}
	{
		// RangeCheck: the arithmetic of the shader. 16-bit and 32-bit indices, the vertex offset added in 32 bits.
		const std::vector<uint16_t> idx16 = {10, 11, 12, 12, 11, 399};
		const Range                 range {10, 400};
		Check(RangeCheck(idx16.data(), 6, 2, 0, range), "all indices inside [first, end)");
		Check(!RangeCheck(idx16.data(), 6, 2, 1, range), "index 399 + 1 = 400 is the exclusive end: violation");
		Check(RangeCheck(idx16.data(), 5, 2, 1, range), "the first five +1 are inside");
		Check(!RangeCheck(idx16.data(), 6, 2, 0xffffffffu, range), "offset -1: 10 - 1 = 9 is below first");
		Check(RangeCheck(idx16.data(), 6, 2, 0xfffffff6u, {0, 400}), "offset -10 with first 0: 0..389 inside");
		Check(RangeCheck(idx16.data(), 0, 2, 0, range), "no indices: nothing to violate");
		const std::vector<uint32_t> idx32 = {100000, 100001, 100399};
		Check(RangeCheck(idx32.data(), 3, 4, 0, {100000, 100400}), "32-bit indices inside");
		Check(!RangeCheck(idx32.data(), 3, 4, 0, {100000, 100399}), "32-bit: last index equals the exclusive end");
		Check(!RangeCheck(idx32.data(), 3, 4, 0xfffe0000u, {0, 100400}), "negative vertex number is a violation");
		// only the first `count` indices matter (the draw reads index_count of them)
		Check(RangeCheck(idx16.data(), 5, 2, 0, {10, 13}) && !RangeCheck(idx16.data(), 6, 2, 0, {10, 13}), "count limits the check");

		// PrepareArgsChecked = PrepareArgs + the check
		const Fields   fields {6, 0, 0, 0};
		const Args     in {6, 483, 0, 0, 0};
		const auto     ok = PrepareArgsChecked(in, fields, 8192, idx16.data(), 2, &range);
		Check(ok.out == PrepareArgs(in, fields, 8192).out && ok.status.range_violation == 0, "inside the range: the same as PrepareArgs");
		const std::vector<uint16_t> bad = {10, 11, 400, 12, 11, 12};
		const auto violated = PrepareArgsChecked(in, fields, 8192, bad.data(), 2, &range);
		Check(violated.out == std::array<uint32_t, 5> {0, 0, 0, 0, 0} && violated.status.range_violation == 1 && violated.status.mismatch == 0 &&
		          violated.status.overflow == 0 && violated.status.draws == 1,
		      "an index at the exclusive end zeroes the draw and counts range_violation only");
		const auto unchecked = PrepareArgsChecked(in, fields, 8192, bad.data(), 2, nullptr);
		Check(unchecked.out[0] == 6 && unchecked.status.range_violation == 0, "no range: no check (draws without reuse)");
		const auto mismatch = PrepareArgsChecked({7, 1, 0, 0, 0}, fields, 8192, bad.data(), 2, &range);
		Check(mismatch.status.mismatch == 1 && mismatch.status.range_violation == 0, "a field mismatch is reported as a mismatch");
		// the draw has fewer indices than the learned count: only those are checked
		const auto shorter = PrepareArgsChecked({3, 1, 0, 0, 0}, {6, 0, 0, 0}, 8192, bad.data(), 2, &range);
		Check(shorter.status.mismatch == 1, "index_count is a learned field: a shorter draw is a mismatch");
	}
	{
		// A violation (read two flips later) forgets the range and sends the entry to the CPU read for 64 flips.
		Entry entry;
		Learn(entry, {192, 10, 0, 0, 0}, 192, 8192);
		RecordRange(entry.range, {0x1000, 384, 2, 192, 0}, 5, 300);
		Status violation;
		violation.draws           = 1;
		violation.range_violation = 1;
		const auto effect         = OnStatus(entry, violation, 7000, 100000);
		Check(effect.range_violation && effect.cooldown && !entry.range.known, "range violation: range forgotten, cooldown");
		Check(Decide(entry, 7001) == Verdict::Cooldown && Decide(entry, 7000 + kCooldownFlips) == Verdict::Use, "the cooldown is the 64 flips of a mismatch");
		Status clean;
		clean.draws = 1;
		RecordRange(entry.range, {0x1000, 384, 2, 192, 0}, 5, 300);
		Check(!OnStatus(entry, clean, 7100, 100000).range_violation && entry.range.known, "a clean status keeps the range");
	}

	{
		// S417: the reuse gate follows the fault, not only the bytes.
		Check(RangeReuseDirty(true, false) && RangeReuseDirty(false, true) && RangeReuseDirty(true, true), "bytes or page dirty: reuse gate open");
		Check(!RangeReuseDirty(false, false), "clean bytes on a clean page: the gate stays closed");
		RangeEntry entry;
		const RangeKey key {0x1000, 384, 2, 192, 0};
		RecordRange(entry, key, 5, 300);
		const bool dirty = RangeReuseDirty(/*bytes*/ false, /*page*/ true);
		Check(DecideRangeReuse(RangeMode::On, true, dirty, entry, key) == RangeDecision::Reuse, "clean bytes on a GPU-owned page: the range is reused");
	}

	if (g_failures != 0) return 1;
	std::fprintf(stderr, "PvIndirectTests: ok\n");
	return 0;
}
