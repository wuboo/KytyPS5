// S434: every KYTY_* name of kKnownKytyEnv is classified in exactly one of kNonCodegenKytyEnv / kCodegenKytyEnv (the disk cache of programs keys
// on the codegen ones), and nothing is classified that is not known. Adding a switch to knownEnv.h without deciding fails this test. Mutant
// (WILL_FAIL): KYTY_KNOWNENV_MUTANT_UNCLASSIFIED adds a known name that is in neither list.

#include "common/knownEnv.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>

namespace {

int g_failures = 0;

void Fail(const char* what, std::string_view name) {
	std::fprintf(stderr, "FAIL: %s: %s\n", what, std::string(name).c_str());
	g_failures++;
}

template <typename List>
bool Has(const List& list, std::string_view name) {
	return std::find(std::begin(list), std::end(list), name) != std::end(list);
}

template <typename List>
void NoDuplicates(const List& list, const char* what) {
	for (auto a = std::begin(list); a != std::end(list); ++a) {
		if (std::count(std::begin(list), std::end(list), *a) != 1) Fail(what, *a);
	}
}

} // namespace

int main() {
	using namespace Common;
	for (const auto name: kKnownKytyEnv) {
		const bool non = Has(kNonCodegenKytyEnv, name);
		const bool gen = Has(kCodegenKytyEnv, name);
		if (!non && !gen) Fail("known name is neither codegen nor non-codegen (classify it in knownEnv.h)", name);
		if (non && gen) Fail("known name is in both lists", name);
	}
	for (const auto name: kNonCodegenKytyEnv) {
		if (!Has(kKnownKytyEnv, name)) Fail("non-codegen name is not a known name", name);
	}
	for (const auto name: kCodegenKytyEnv) {
		if (!Has(kKnownKytyEnv, name)) Fail("codegen name is not a known name", name);
	}
	NoDuplicates(kKnownKytyEnv, "duplicate in kKnownKytyEnv");
	NoDuplicates(kNonCodegenKytyEnv, "duplicate in kNonCodegenKytyEnv");
	NoDuplicates(kCodegenKytyEnv, "duplicate in kCodegenKytyEnv");
	// The switches that decide what the recompiler emits must stay on the codegen side.
	for (const auto name: {"KYTY_OPT_OFF", "KYTY_OPT_ON", "KYTY_SHADER_LOOP_LIMIT", "KYTY_SRT_READER", "KYTY_PV_VSSTORE", "KYTY_PER_VERTEX_PROTOTYPE",
	                       "KYTY_SPV_NOCONTRACT", "KYTY_NGG_VCOUNT", "KYTY_VS_LANE_FROM_INDEX", "KYTY_RT_MISS"}) {
		if (!Has(kCodegenKytyEnv, name)) Fail("a code-generating switch is not classified as codegen", name);
	}
	if (g_failures != 0) return 1;
	std::puts("EnvClassificationTests: OK");
	return 0;
}
