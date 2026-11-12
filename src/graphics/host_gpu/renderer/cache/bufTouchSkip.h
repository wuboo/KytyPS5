#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFTOUCHSKIP_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFTOUCHSKIP_H_

// S562 KYTY_BUF_TOUCH_SKIP=0|1|shadow: the buffer counterpart of KYTY_LRU_TOUCH_SKIP (S552, images). BufferCache::ObtainBuffer touches the
// buffer's LRU item on every call (~64 k/flip in the garden); LeastRecentlyUsedCache::Touch returns without change when the item already has
// the tick, but reads the item (a deque element, usually a cache miss) to find out. The buffer remembers the tick and the LRU id it last
// touched with; the same pair again is a no-op Touch (the item's tick only grows, and a new LRU id after a re-register is a new pair).
// Off: unchanged. 1: skip. shadow: touch as ever; count the would-be skips that the item's tick confirms and the ones it does not (ne).
//
// Header only and templated on the item and LRU types, so tests/BufTouchSkipTests.cpp runs it on Common::LeastRecentlyUsedCache.
// Mutant (tests, WILL_FAIL): KYTY_BUFTOUCH_MUTANT_IGNORE_ID.

#include "graphics/host_gpu/renderer/pipeline/texResolveMemo.h"

#include <cstdint>
#include <cstdlib>

namespace Libs::Graphics::BufTouchSkip {

using Mode = TexMemo::Mode;

[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = TexMemo::ParseMode(std::getenv("KYTY_BUF_TOUCH_SKIP"));
	return mode;
}

struct Stats {
	uint64_t skip = 0;
	uint64_t ne   = 0;
};

// item: lru_id, lru_touch_tick, lru_touch_id. lru: TickOf(id), Touch(id, tick).
template <class Item, class Lru, class Tick>
void Touch(Item& item, Lru& lru, Tick tick, Mode mode, Stats& stats) {
#if defined(KYTY_BUFTOUCH_MUTANT_IGNORE_ID)
	const bool same = item.lru_touch_tick == tick;
#else
	const bool same = item.lru_touch_tick == tick && item.lru_touch_id == item.lru_id;
#endif
	if (mode != Mode::Off && same) {
		if (mode == Mode::On) {
			stats.skip++;
			return;
		}
		if (lru.TickOf(item.lru_id) >= tick) {
			stats.skip++;
		} else {
			stats.ne++;
		}
	}
	lru.Touch(item.lru_id, tick);
	item.lru_touch_tick = tick;
	item.lru_touch_id   = item.lru_id;
}

} // namespace Libs::Graphics::BufTouchSkip

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_BUFTOUCHSKIP_H_
