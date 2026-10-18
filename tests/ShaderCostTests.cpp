// S433 (S441): first-use cost counters of shaders. The lead of a shader (registered -> first lookup) is reported once, a re-registration with the
// same code does not reset it, other code does; lookups count as misses exactly when they did not return a cached permutation.
// Mutants: see src/graphics/shader/shaderCost.h.

#include "graphics/shader/shaderCost.h"

#include <cstdint>
#include <cstdio>

using namespace Libs::Graphics::ShaderCost;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

constexpr uint64_t kMs = 1'000'000ull;

void TestBuckets() {
	Check(LeadBucket(0) == 0, "0 ms -> b0");
	Check(LeadBucket(kMs - 1) == 0, "just below 1 ms -> b0");
	Check(LeadBucket(kMs) == 1, "1 ms -> b1");
	Check(LeadBucket(5 * kMs) == 2, "5 ms -> b2");
	Check(LeadBucket(99 * kMs) == 4, "99 ms -> b4");
	Check(LeadBucket(100 * kMs) == 5, "100 ms -> b5");
	Check(LeadBucket(999 * kMs) == 6, "999 ms -> b6");
	Check(LeadBucket(1000 * kMs) == 7, "1 s -> last bucket");
	Check(LeadBucket(~0ull) == kLeadBuckets - 1, "huge -> last bucket");
}

void TestLead() {
	LeadState state = LeadOnMapped(nullptr, 0, 0xAA, 1000);
	Check(!state.used && state.map_ns == 1000, "fresh state");
	const auto first = LeadOnUse(state, 1000 + 40 * kMs);
	Check(first.has_value() && *first == 40 * kMs, "first use reports the lead");
	Check(!LeadOnUse(state, 1000 + 80 * kMs).has_value(), "second use reports nothing");
	Check(!LeadOnUse(state, 1000 + 90 * kMs).has_value(), "third use reports nothing");

	// Same code registered again after it was used: still used, same first-seen time.
	LeadState again = LeadOnMapped(&state, 0xAA, 0xAA, 5000 * kMs);
	Check(again.used && again.map_ns == 1000, "re-registration with the same code keeps state");
	Check(!LeadOnUse(again, 6000 * kMs).has_value(), "no second report after re-registration");

	// Registered twice before the first use: the lead counts from the first registration.
	LeadState unused = LeadOnMapped(nullptr, 0, 0xBB, 100);
	LeadState twice  = LeadOnMapped(&unused, 0xBB, 0xBB, 100 + 30 * kMs);
	const auto lead  = LeadOnUse(twice, 100 + 60 * kMs);
	Check(lead.has_value() && *lead == 60 * kMs, "lead counts from the first registration");

	// Other code at the same address starts over.
	LeadState other = LeadOnMapped(&state, 0xAA, 0xCC, 9000 * kMs);
	Check(!other.used && other.map_ns == 9000 * kMs, "other code resets");
	const auto lead2 = LeadOnUse(other, 9000 * kMs + 3 * kMs);
	Check(lead2.has_value() && *lead2 == 3 * kMs, "lead of the new code");

	// A clock that went backwards must not wrap.
	LeadState back = LeadOnMapped(nullptr, 0, 1, 500);
	const auto zero = LeadOnUse(back, 100);
	Check(zero.has_value() && *zero == 0, "negative lead clamps to 0");
}

void TestClassify() {
	const auto hit = ClassifyLookup(true, true, false);
	Check(hit.program_miss == 0 && hit.perm_miss == 0 && hit.call_program == 0, "hit counts nothing");
	const auto hit_call = ClassifyLookup(true, true, true);
	Check(hit_call.program_miss == 0 && hit_call.perm_miss == 0 && hit_call.call_program == 0, "a call-source hit counts nothing");
	const auto new_program = ClassifyLookup(false, false, false);
	Check(new_program.program_miss == 1 && new_program.perm_miss == 1 && new_program.call_program == 0, "new program counts program + permutation");
	const auto new_perm = ClassifyLookup(true, false, false);
	Check(new_perm.program_miss == 0 && new_perm.perm_miss == 1, "new permutation of a known program counts the permutation only");
	const auto call_miss = ClassifyLookup(true, false, true);
	Check(call_miss.perm_miss == 1 && call_miss.call_program == 1, "a call program miss is marked");

	// Summed over a mixed sequence the counters equal the number of misses, not of lookups.
	uint32_t programs = 0, perms = 0, lookups = 0;
	const bool seq[][2] = {{false, false}, {true, true}, {true, true}, {true, false}, {true, true}, {true, true}, {false, false}};
	for (const auto& s: seq) {
		const auto c = ClassifyLookup(s[0], s[1], false);
		programs += c.program_miss;
		perms += c.perm_miss;
		lookups++;
	}
	Check(lookups == 7 && programs == 2 && perms == 3, "counters grow once per miss");
}

} // namespace

int main() {
	TestBuckets();
	TestLead();
	TestClassify();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("shader_cost: OK\n");
	return 0;
}
