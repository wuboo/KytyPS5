// S438: KYTY_TEX_RESOLVE_MEMO. The decision header (pipeline/texResolveMemo.h) against fake hooks that model the registration generations of
// the texture cache. Mutants: KYTY_TEXMEMO_MUTANT_{IGNORES_EPOCH, IGNORES_REBIND, SKIPS_CLEAR, STORE_OVERLAP, STORE_INEXACT}.

#include "graphics/host_gpu/renderer/pipeline/texResolveMemo.h"

#include <cstdint>
#include <cstdio>
#include <map>

using namespace Libs::Graphics::TexMemo;

static int g_failed = 0;

#define CHECK(cond)                                                                    \
	do {                                                                               \
		if (!(cond)) {                                                                 \
			std::fprintf(stderr, "TexResolveMemoTests: failed: %s (line %d)\n", #cond, __LINE__); \
			g_failed++;                                                                \
		}                                                                              \
	} while (0)

using Id = uint32_t;

// A texture cache reduced to what the memo asks: image slots, the registration generation and the generation stamped on the pages of the request.
struct FakeCache {
	std::map<Id, ImageState> images;
	uint64_t                 generation = 1;
	uint64_t                 page_stamp = 0; // generation of the last registration change on the pages of the image
	int                      locks = 0, touches = 0, clears = 0, state_calls = 0, since_calls = 0;
	bool                     locked = false;
	bool                     clear_with_lock = false;
	Id                       touched = 0, cleared = 0;

	struct Guard {
		FakeCache& c;
		explicit Guard(FakeCache& cache): c(cache) { c.locked = true; c.locks++; }
		Guard(const Guard&) = delete;
		~Guard() { c.locked = false; }
	};
	Guard Lock() { return Guard {*this}; }
	ImageState State(Id id) {
		state_calls++;
		CHECK(locked);
		const auto found = images.find(id);
		return found == images.end() ? ImageState {} : found->second;
	}
	bool RegistrationsSince(uint64_t since) {
		since_calls++;
		CHECK(locked);
		return page_stamp > since;
	}
	void Touch(Id id) { CHECK(locked); touches++; touched = id; }
	void ColorClear(Id id) { clear_with_lock |= locked; clears++; cleared = id; }

	// What FindImage does for a registration change on the pages of the image.
	void Register() { page_stamp = ++generation; }
};

static Entry<Id> Remember(FakeCache& cache, Id id) { return {id, cache.generation, true}; }

static void T1_HitSameKey() {
	FakeCache cache;
	cache.images[7] = {true, true, false};
	const auto entry = Remember(cache, 7);
	CHECK(Reuse(entry, cache, true) == Verdict::Hit);
	CHECK(cache.touches == 1 && cache.touched == 7);
	CHECK(cache.clears == 1 && cache.cleared == 7);
	CHECK(Reuse(entry, cache, true) == Verdict::Hit); // nothing changed: still a hit
	CHECK(cache.touches == 2 && cache.clears == 2);
}

static void T2_RegistrationOnPageMisses() {
	FakeCache cache;
	cache.images[7] = {true, true, false};
	const auto entry = Remember(cache, 7);
	cache.Register(); // an image was registered or freed on the pages
	CHECK(Reuse(entry, cache, true) == Verdict::Stale);
	CHECK(cache.touches == 0 && cache.clears == 0);
	// The slot of a freed image reused by another one (same id): the free and the insert both stamp the pages.
	FakeCache recycled;
	recycled.images[7] = {true, true, false};
	const auto old_entry = Remember(recycled, 7);
	recycled.images.erase(7);
	recycled.Register();
	recycled.images[7] = {true, true, false};
	recycled.Register();
	CHECK(Reuse(old_entry, recycled, true) == Verdict::Stale);
	// An entry made after the change holds.
	CHECK(Reuse(Remember(recycled, 7), recycled, true) == Verdict::Hit);
}

static void T3_RebindAndUnregisteredMiss() {
	FakeCache cache;
	cache.images[7] = {true, true, true};
	CHECK(Reuse(Remember(cache, 7), cache, true) == Verdict::Stale); // needs_rebind
	cache.images[7] = {true, false, false};
	CHECK(Reuse(Remember(cache, 7), cache, true) == Verdict::Stale); // not registered
	cache.images.erase(7);
	CHECK(Reuse(Remember(cache, 7), cache, true) == Verdict::Stale); // slot empty
	CHECK(cache.touches == 0 && cache.clears == 0);
}

static void T4_ColorClearStillRuns() {
	FakeCache cache;
	cache.images[3] = {true, true, false};
	CHECK(Reuse(Remember(cache, 3), cache, true) == Verdict::Hit);
	CHECK(cache.clears == 1 && cache.cleared == 3);
	CHECK(!cache.clear_with_lock); // the cache takes its own lock in MaterializeColorClear
	// shadow only asks: no touch, no clear
	FakeCache peek;
	peek.images[3] = {true, true, false};
	CHECK(Reuse(Remember(peek, 3), peek, false) == Verdict::Hit);
	CHECK(peek.touches == 0 && peek.clears == 0);
	// a miss does neither
	CHECK(Reuse(Entry<Id> {}, peek, true) == Verdict::NoEntry);
	CHECK(peek.touches == 0 && peek.clears == 0);
}

static void T5_OnlyExactResultsAreStored() {
	Found good {true, false, false, false};
	CHECK(CanStore(good));
	Found overlap = good;
	overlap.view_adjusted = true; // moved to a mip/layer of a larger image
	CHECK(!CanStore(overlap));
	Found inexact = good;
	inexact.exact = false; // a BC alias is left behind or the image was a partial match
	CHECK(!CanStore(inexact));
	Found null_image = good;
	null_image.null_image = true;
	CHECK(!CanStore(null_image));
	Found video_out = good;
	video_out.video_out = true;
	CHECK(!CanStore(video_out));
}

static void T6_ModeAndShadow() {
	CHECK(ParseMode(nullptr) == Mode::Off);
	CHECK(ParseMode("") == Mode::Off);
	CHECK(ParseMode("0") == Mode::Off);
	CHECK(ParseMode("1") == Mode::On);
	CHECK(ParseMode("shadow") == Mode::Shadow);
	CHECK(ParseMode("yes") == Mode::Off);
	Entry<Id> entry {5, 1, true};
	CHECK(ShadowAgrees(entry, Id {5}, true));
	CHECK(!ShadowAgrees(entry, Id {6}, true));
	CHECK(!ShadowAgrees(entry, Id {5}, false));
}

static void T7_NoEntryAsksNothing() {
	FakeCache cache;
	CHECK(Reuse(Entry<Id> {}, cache, true) == Verdict::NoEntry);
	CHECK(cache.state_calls == 0 && cache.since_calls == 0);
}

int main() {
	T1_HitSameKey();
	T2_RegistrationOnPageMisses();
	T3_RebindAndUnregisteredMiss();
	T4_ColorClearStillRuns();
	T5_OnlyExactResultsAreStored();
	T6_ModeAndShadow();
	T7_NoEntryAsksNothing();
	if (g_failed != 0) {
		std::fprintf(stderr, "TexResolveMemoTests: %d failed\n", g_failed);
		return 1;
	}
	std::printf("TexResolveMemoTests: ok\n");
	return 0;
}
