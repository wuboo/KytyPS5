// S469: KYTY_BIND_PREP_MEMO. The keyed table and the inputs hash of pipeline/bindPrepMemo.h on the CPU.
// Mutants (WILL_FAIL): KYTY_BINDPREP_MUTANT_{IGNORES_INPUTS, IGNORES_CONTROL_GEN, HASH_SKIPS_SAMPLERS, HASH_SKIPS_USER_DATA}.

#include "graphics/host_gpu/renderer/pipeline/bindPrepMemo.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace Libs::Graphics::BindPrepMemo;

static int g_failed = 0;

#define CHECK(cond)                                                                           \
	do {                                                                                      \
		if (!(cond)) {                                                                        \
			std::fprintf(stderr, "BindPrepMemoTests: failed: %s (line %d)\n", #cond, __LINE__); \
			g_failed++;                                                                       \
		}                                                                                     \
	} while (0)

struct Inputs {
	std::array<uint32_t, 8> image {1, 2, 3, 4, 5, 6, 7, 8};
	std::array<uint32_t, 4> sampler {10, 11, 12, 13};
	std::vector<uint32_t>   user_data {100, 200, 300};
	uint64_t                shader = 0xabcdef;
	uint32_t                sampler_flags = 0;
};

static uint64_t Hash(const Inputs& in) {
	InputsHash h;
	h.Program(in.shader, 4, 1, 1, static_cast<uint32_t>(in.user_data.size()));
	h.ImageResource(0x0102, 0x03);
	h.Image(in.image.data(), static_cast<uint32_t>(in.image.size()));
	h.Sampler(in.sampler.data(), static_cast<uint32_t>(in.sampler.size()), in.sampler_flags);
	uint32_t index = 0;
	for (const auto w: in.user_data) h.UserData(index++, w);
	return h.Digest();
}

struct Payload {
	int value = 0;
};

int main() {
	CHECK(ParseMode(nullptr) == Mode::Off);
	CHECK(ParseMode("") == Mode::Off);
	CHECK(ParseMode("0") == Mode::Off);
	CHECK(ParseMode("1") == Mode::On);
	CHECK(ParseMode("shadow") == Mode::Shadow);
	CHECK(ParseMode("yes") == Mode::Off);

	// The hash follows every input.
	const Inputs base;
	const auto   h0 = Hash(base);
	CHECK(Hash(base) == h0);
	{ Inputs x = base; x.image[3] ^= 1u; CHECK(Hash(x) != h0); }
	{ Inputs x = base; x.sampler[2] ^= 1u; CHECK(Hash(x) != h0); }
	{ Inputs x = base; x.sampler_flags = 1; CHECK(Hash(x) != h0); }
	{ Inputs x = base; x.user_data[1] ^= 1u; CHECK(Hash(x) != h0); }
	{ Inputs x = base; x.shader ^= 1u; CHECK(Hash(x) != h0); }
	{ Inputs x = base; std::swap(x.user_data[0], x.user_data[2]); CHECK(Hash(x) != h0); }

	// The table: a hit needs the owner, the inputs and the control-file generation.
	Table<Payload> table;
	int            owner_a = 0, owner_b = 0;
	CHECK(table.Find(&owner_a, h0, 1) == nullptr);
	table.Store(&owner_a, h0, 1, Payload {7});
	CHECK(table.Find(&owner_a, h0, 1) != nullptr && table.Find(&owner_a, h0, 1)->value == 7);
	CHECK(table.Find(&owner_b, h0, 1) == nullptr);
	CHECK(table.Find(&owner_a, h0 + 1, 1) == nullptr);
	CHECK(table.Find(&owner_a, h0, 2) == nullptr);
	table.Store(&owner_a, h0 + 1, 1, Payload {8});
	CHECK(table.Find(&owner_a, h0, 1) == nullptr);
	CHECK(table.Find(&owner_a, h0 + 1, 1) != nullptr && table.Find(&owner_a, h0 + 1, 1)->value == 8);
	table.Erase(&owner_a);
	CHECK(table.Find(&owner_a, h0 + 1, 1) == nullptr);

	// Bounded: the table is cleared when a new program arrives at the limit.
	std::vector<int> owners(Table<Payload>::kMaxPrograms + 1);
	for (size_t i = 0; i < Table<Payload>::kMaxPrograms; i++) table.Store(&owners[i], 1, 1, Payload {1});
	CHECK(table.Size() == Table<Payload>::kMaxPrograms);
	table.Store(&owners[Table<Payload>::kMaxPrograms], 1, 1, Payload {2});
	CHECK(table.Size() == 1);

	if (g_failed == 0) std::printf("BindPrepMemoTests: ok\n");
	return g_failed == 0 ? 0 : 1;
}
