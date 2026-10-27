// S493: KYTY_BIND_DRAW_REPEAT header (bindDrawRepeat.h) — input hash and image-stamp validation.

#include "graphics/host_gpu/renderer/pipeline/bindDrawRepeat.h"
#include "graphics/host_gpu/renderer/pipeline/texResolveMemo.h"

#include <cstdio>
#include <map>

using namespace Libs::Graphics::BindDrawRepeat;
using namespace Libs::Graphics::TexMemo;

static int g_failed = 0;

#define CHECK(cond)                                                                                \
	do {                                                                                           \
		if (!(cond)) {                                                                             \
			std::fprintf(stderr, "BindDrawRepeatTests: failed: %s (line %d)\n", #cond, __LINE__); \
			g_failed++;                                                                        \
		}                                                                                          \
	} while (0)

using Id = uint32_t;

struct FakeHooks {
	std::map<Id, ImageState> images;
	bool                     pages_dirty = false;
	uint64_t                 page_stamp = 1;
	ImageState State(Id id) {
		const auto found = images.find(id);
		return found == images.end() ? ImageState {} : found->second;
	}
	bool RegistrationsSince(uint64_t /*address*/, uint64_t /*size*/, uint64_t generation) {
		return page_stamp > generation;
	}
};

int main() {
	InputsHash h1;
	h1.Program(0xabc, 1, 2, 1, 4);
	h1.ImageResource(1, 2);
	const uint32_t img[] = {1, 2, 3};
	h1.Image(img, 3);
	const uint32_t sam[] = {9};
	h1.Sampler(sam, 1, 0);
	h1.UserData(0, 42);
	const uint64_t d1 = h1.Digest();

	InputsHash h2;
	h2.Program(0xabc, 1, 2, 1, 4);
	h2.ImageResource(1, 2);
	h2.Image(img, 3);
	h2.Sampler(sam, 1, 0);
	h2.UserData(0, 42);
	CHECK(h2.Digest() == d1);

	InputsHash h3;
	h3.Program(0xabc, 1, 2, 1, 4);
	h3.ImageResource(1, 2);
	h3.Image(img, 3);
	h3.Sampler(sam, 1, 0);
	h3.UserData(0, 43);
	CHECK(h3.Digest() != d1);

	ImageStamp<Id> stamp {5, 0x1000, 0x4000, 1, false};
	FakeHooks hooks;
	hooks.images[5] = {true, true, false};
	CHECK(ImageStillValid(stamp, hooks));
	hooks.pages_dirty = true;
	hooks.page_stamp    = 2;
	CHECK(!ImageStillValid(stamp, hooks));
	hooks.page_stamp = 1;
	hooks.images[5].needs_rebind = true;
	CHECK(!ImageStillValid(stamp, hooks));

	Table<int> table;
	CHECK(table.Find(nullptr, d1, 1) == nullptr);
	table.Store(nullptr, d1, 1, 7);
	CHECK(table.Find(nullptr, d1, 1) != nullptr && *table.Find(nullptr, d1, 1) == 7);
	CHECK(table.Find(nullptr, d1, 2) == nullptr);

	return g_failed == 0 ? 0 : 1;
}
