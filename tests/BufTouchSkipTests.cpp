// S562 KYTY_BUF_TOUCH_SKIP (bufTouchSkip.h) on Common::LeastRecentlyUsedCache: after every touch the item's LRU tick is at least the touch tick
// (what LeastRecentlyUsedCache::Touch guarantees and the GC relies on), and the eviction order (ForEachItemBelow) equals the order of plain touches.
// Events: repeated touches in one tick, a new tick, a buffer re-registered under another LRU id with an older tick, the same id reused.
// Mutant (must fail, WILL_FAIL): IGNORE_ID (the skip ignores that the LRU id changed).

#include "common/lruCache.h"
#include "graphics/host_gpu/renderer/cache/bufTouchSkip.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

using namespace Libs::Graphics;

void Check(bool ok, const char* what, int step) {
	if (!ok) {
		std::fprintf(stderr, "BufTouchSkipTests: failed: %s (step %d)\n", what, step);
		std::exit(1);
	}
}

struct Item {
	size_t   lru_id         = 0;
	uint64_t lru_touch_tick = UINT64_MAX;
	size_t   lru_touch_id   = SIZE_MAX;
};

using Lru = Common::LeastRecentlyUsedCache<int, uint64_t>;

std::vector<int> Order(Lru& lru, uint64_t below) {
	std::vector<int> order;
	lru.ForEachItemBelow(below, [&](int object) { order.push_back(object); });
	return order;
}

} // namespace

int main() {
	for (const auto mode: {TexMemo::Mode::On, TexMemo::Mode::Shadow}) {
		Lru                 plain;
		Lru                 skip;
		std::vector<Item>   a(4);
		std::vector<Item>   b(4);
		BufTouchSkip::Stats stats;
		for (int i = 0; i < 4; i++) {
			a[i].lru_id = plain.Insert(i, 0);
			b[i].lru_id = skip.Insert(i, 0);
		}
		int step = 0;
		const auto touch = [&](int i, uint64_t tick) {
			plain.Touch(a[i].lru_id, tick);
			BufTouchSkip::Touch(b[i], skip, tick, mode, stats);
			Check(skip.TickOf(b[i].lru_id) >= tick, "item tick below the touch tick", step);
			Check(Order(plain, tick + 1) == Order(skip, tick + 1), "eviction order differs", step);
			step++;
		};
		touch(0, 1);
		touch(0, 1);
		touch(1, 1);
		touch(0, 1);
		touch(2, 2);
		touch(0, 2);
		touch(0, 2);
		// Item 3 re-registered: its LRU item freed, inserted again under another id with an older tick (the id of item 1 freed before it).
		touch(3, 2);
		plain.Free(a[3].lru_id);
		skip.Free(b[3].lru_id);
		plain.Free(a[1].lru_id);
		skip.Free(b[1].lru_id);
		a[3].lru_id = plain.Insert(3, 1);
		b[3].lru_id = skip.Insert(3, 1);
		Check(b[3].lru_id != b[3].lru_touch_id, "test setup: the re-registered item must get another id", step);
		touch(3, 2);
		touch(3, 2);
		touch(0, 3);
		touch(3, 3);
		Check(stats.ne == 0, "shadow disagreements", step);
		Check(stats.skip >= 4, "too few skips", step);
		std::printf("BufTouchSkipTests: mode %s: %d touches, skip %llu ne %llu\n", TexMemo::ModeName(mode), step,
		            static_cast<unsigned long long>(stats.skip), static_cast<unsigned long long>(stats.ne));
	}
	std::printf("BufTouchSkipTests: OK\n");
	return 0;
}
