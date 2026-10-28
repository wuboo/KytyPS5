#include "graphics/host_gpu/renderer/cache/landSync.h"

#include <cassert>
#include <cstdint>

using Libs::Graphics::LandSync::ClearHit;
using Libs::Graphics::LandSync::ColorClearHit;
using Libs::Graphics::LandSync::DccMetadataObtainIsWrite;
using Libs::Graphics::LandSync::Effective;
using Libs::Graphics::LandSync::EnablesGuest;
using Libs::Graphics::LandSync::Mode;
using Libs::Graphics::LandSync::Parse;

int main() {
	assert(Parse(nullptr) == Mode::Off);
	assert(Parse("0") == Mode::Off);
	assert(Parse("1") == Mode::On);
	assert(Parse("shadow") == Mode::Shadow);
	assert(Parse("yes") == Mode::Off);

	assert(Effective(Mode::On, nullptr) == Mode::On);
	assert(Effective(Mode::On, "") == Mode::On);
	assert(Effective(Mode::On, "0") == Mode::Off);
	assert(Effective(Mode::Off, "1") == Mode::On);
	assert(Effective(Mode::On, "shadow") == Mode::Shadow);

	assert(EnablesGuest(Mode::On));
	assert(EnablesGuest(Mode::Shadow));
	assert(!EnablesGuest(Mode::Off));
	assert(!DccMetadataObtainIsWrite());

	const uint32_t uniform = 0x20202020u;
	const uint32_t words[] {uniform, uniform, uniform, uniform};
	const ClearHit hit = ColorClearHit(words, 4, 1u << 1u);
	assert(hit.hit);
	assert(hit.slot == 1u);

	const uint32_t mixed[] {uniform, 0x21212121u};
	assert(!ColorClearHit(mixed, 2, 1u << 1u).hit);
	assert(!ColorClearHit(words, 4, 1u << 0u).hit); // slot 1 not selected
	const uint32_t unaligned = 0x01010101u;
	assert(!ColorClearHit(&unaligned, 1, 0xffu).hit);
	assert(!ColorClearHit(words, 0, 0xffu).hit);
	return 0;
}
