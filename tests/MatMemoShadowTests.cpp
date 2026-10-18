// S406: shadow memo after the walk (matMemoShadow.h).
// Mutant KYTY_MATMEMO_MUTANT_IGNORE_FLAT (the flattened SRT is not part of the input) must make `unsafe` non-zero in the test
// that varies only the flat buffer, i.e. fail it.

#include "graphics/host_gpu/renderer/pipeline/matMemoShadow.h"

#include <cstdio>
#include <vector>

using namespace Libs::Graphics::MatMemo;

static int g_failures = 0;
static void Check(bool ok, const char* what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		++g_failures;
	}
}

static InputParts Parts(const std::vector<uint32_t>& user, uint64_t mask, const std::vector<uint32_t>& flat) {
	InputParts parts;
	parts.user_data = user;
	parts.used_mask = mask;
	parts.flat      = flat;
	return parts;
}

int main() {
	const std::vector<uint32_t> user  = {10, 20, 30, 40};
	const std::vector<uint32_t> flat  = {1, 2, 3};
	const std::vector<uint32_t> flat2 = {1, 2, 4};

	{ // the same input twice: in_hit, same output, not unsafe
		Shadow     shadow;
		const auto key = InputKey(Parts(user, 0b0011, flat));
		auto       r1  = shadow.Note(key, {100, 7});
		Check(!r1.in_hit && !r1.out_hit && !r1.unsafe && !r1.rebase, "first call misses everything");
		auto r2 = shadow.Note(key, {100, 7});
		Check(r2.in_hit && r2.out_hit && !r2.unsafe, "second call: input hit, output hit");
	}
	{ // user_data the plan does not use does not change the input
		auto           other = user;
		other[3]             = 999;
		const uint64_t a     = InputKey(Parts(user, 0b0011, flat));
		Check(a == InputKey(Parts(other, 0b0011, flat)), "unused user_data word ignored");
		Check(a != InputKey(Parts(other, 0b1011, flat)), "used user_data word counts");
	}
	{ // different buffer bases only: no input hit, no output hit, rebase
		Shadow     shadow;
		const auto k1 = InputKey(Parts(user, 0b1, flat));
		auto       other = user;
		other[0]         = 11; // the pointer moved
		const auto k2    = InputKey(Parts(other, 0b1, flat));
		shadow.Note(k1, {100, 7});
		auto r = shadow.Note(k2, {101, 7});
		Check(!r.in_hit && !r.out_hit && r.rebase, "bases differ: rebase");
	}
	{ // different input, same output: output hit
		Shadow shadow;
		shadow.Note(1, {100, 7});
		auto r = shadow.Note(2, {100, 7});
		Check(!r.in_hit && r.out_hit && !r.rebase, "result-level hit");
	}
	{ // the flat buffer is part of the input: changing only it must not produce an input hit (no unsafe)
		Shadow     shadow;
		const auto k1 = InputKey(Parts(user, 0b11, flat));
		const auto k2 = InputKey(Parts(user, 0b11, flat2));
		shadow.Note(k1, {100, 7});
		auto r = shadow.Note(k2, {200, 8}); // the walk output a different layout: descriptors differ
		Check(!r.unsafe, "flat difference must not give an unsafe hit");
	}
	{ // same input, different output is flagged
		Shadow shadow;
		shadow.Note(5, {100, 7});
		auto r = shadow.Note(5, {200, 8});
		Check(r.in_hit && r.unsafe, "same input, different output is unsafe");
	}
	{ // MRU-4: the fifth distinct input evicts the oldest
		Shadow shadow;
		for (uint64_t i = 1; i <= 5; i++) shadow.Note(i, {i, i});
		Check(shadow.Size() == 4, "four ways");
		Check(!shadow.Note(1, {1, 1}).in_hit, "oldest evicted");
		Check(shadow.Note(5, {5, 5}).in_hit, "newest kept");
	}
	{ // workgroup counts, remap flag and push cursor are part of the input
		InputParts a = Parts(user, 0b1, flat);
		InputParts b = a;
		b.push_cursor = 4;
		Check(InputKey(a) != InputKey(b), "push cursor in the input");
		InputParts c = a;
		c.min_lod_remap = true;
		Check(InputKey(a) != InputKey(c), "min lod remap in the input");
		const std::vector<uint32_t> wg = {1, 2, 3};
		InputParts d = a;
		d.workgroup_counts = wg;
		Check(InputKey(a) != InputKey(d), "workgroup counts in the input");
	}
	{ // report: top misses first
		Report report;
		report.Add(0xA, 0, {false, false, false, false});
		report.Add(0xA, 0, {true, false, true, false});
		report.Add(0xB, 1, {false, false, false, false});
		report.Add(0xB, 1, {false, false, false, false});
		const auto top = report.TopMisses(1);
		Check(top.size() == 1 && top[0].hash == 0xB && top[0].Misses() == 2, "top miss program");
	}
	if (g_failures != 0) return 1;
	std::fprintf(stderr, "MatMemoShadowTests: ok\n");
	return 0;
}
