#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMPROBE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMPROBE_H_

// S613 (KYTY_STREAM_PROBE=0|1, default 0): a probe that changes nothing. It sizes the next pool after the recorder diet: the stream path of
// BufferCache::ObtainBuffer (a small CPU-modified range is copied into the stream buffer at every binding: ~13,5 k copies per garden flip, S612
// classify `upload`; buf + vbib ~6,9 ms/flip of the recorder). A copy that repeats the same guest range in the same command buffer (scheduler tick)
// with the same bytes could reuse the earlier stream offset (the ring keeps a region until the GPU is past its tick, so within one tick the earlier
// copy is still there). For every stream copy the probe hashes the guest bytes (XXH3 of a private copy, CopySparseMemory) and looks the range up
// in a direct-mapped table keyed by (address, size):
//   rep   = the same range was copied earlier in this tick, same = and its bytes are the same (the dedupe pool), xtick_same = copied in an
//   earlier tick with the same bytes (not reusable: the ring may recycle it, counted to see the frame-level repeat).
// frames.csv: stream_probe_n, stream_probe_rep_n, stream_probe_same_n, stream_probe_same_bytes, stream_probe_xtick_same_n.
// The probe reads and hashes every copied range on the recorder (~0,3-0,6 ms/flip): a probe run is not a host measurement.
//
// Pure part (the table); tests/RecDietTests.cpp pins it against an exact model. Mutants (WILL_FAIL): KYTY_STREAMPROBE_MUTANT_TICK_IGNORED (a copy of an
// earlier tick counts as this tick), KYTY_STREAMPROBE_MUTANT_SIZE_IGNORED (the key is the address only), KYTY_STREAMPROBE_MUTANT_HASH_IGNORED (a
// repeat counts as same whatever the bytes).

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::StreamProbe {

enum class Mode : uint8_t { Off = 0, On = 1 };

[[nodiscard]] inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0 || std::strcmp(text, "on") == 0) return Mode::On;
	return Mode::Off;
}

// The first call logs `Environment: KYTY_STREAM_PROBE=<value> mode=<off|on>`.
inline Mode EnvMode() {
	static const Mode mode = [] {
		const char* value = std::getenv("KYTY_STREAM_PROBE");
		const Mode  m     = ParseMode(value);
		std::fprintf(stderr, "Environment: KYTY_STREAM_PROBE=%s mode=%s\n", value != nullptr ? value : "", m == Mode::On ? "on" : "off");
		std::fflush(stderr);
		return m;
	}();
	return mode;
}

enum class Kind : uint8_t { First, Repeat, Same, EarlierSame, EarlierOther };

struct Stats {
	uint64_t seen = 0, rep = 0, same = 0, same_bytes = 0, xtick_same = 0;
};

// One entry per slot; a new range takes the slot (the older range is forgotten: the probe can only undercount).
template <size_t Slots>
class Table {
	static_assert(Slots != 0 && (Slots & (Slots - 1)) == 0, "Slots must be a power of two");

public:
	[[nodiscard]] static size_t SlotOf(uint64_t address, uint64_t size) {
#if defined(KYTY_STREAMPROBE_MUTANT_SIZE_IGNORED)
		(void)size;
		const uint64_t key = address;
#else
		const uint64_t key = address ^ (size * 0x9E3779B97F4A7C15ull);
#endif
		return static_cast<size_t>((key ^ (key >> 17u) ^ (key >> 31u)) & (Slots - 1));
	}

	Kind Note(uint64_t address, uint64_t size, uint64_t tick, uint64_t hash, Stats& stats) {
		stats.seen++;
		Entry&     e     = m_entries[SlotOf(address, size)];
#if defined(KYTY_STREAMPROBE_MUTANT_SIZE_IGNORED)
		const bool known = e.used && e.address == address;
#else
		const bool known = e.used && e.address == address && e.size == size;
#endif
#if defined(KYTY_STREAMPROBE_MUTANT_HASH_IGNORED)
		const bool same_bytes = true;
#else
		const bool same_bytes = e.hash == hash;
#endif
#if defined(KYTY_STREAMPROBE_MUTANT_TICK_IGNORED)
		const bool same_tick = true;
#else
		const bool same_tick = e.tick == tick;
#endif
		Kind kind = Kind::First;
		if (known && same_tick) {
			stats.rep++;
			kind = Kind::Repeat;
			if (same_bytes) {
				stats.same++;
				stats.same_bytes += size;
				kind = Kind::Same;
			}
		} else if (known) {
			kind = same_bytes ? Kind::EarlierSame : Kind::EarlierOther;
			if (same_bytes) stats.xtick_same++;
		}
		e = {true, address, size, tick, hash};
		return kind;
	}

private:
	struct Entry {
		bool     used    = false;
		uint64_t address = 0, size = 0, tick = 0, hash = 0;
	};
	Entry m_entries[Slots] {};
};

} // namespace Libs::Graphics::StreamProbe

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_STREAMPROBE_H_ */
