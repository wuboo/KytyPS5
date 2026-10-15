// S361 KYTY_DESC_DYN: the descriptor-set model on the CPU. Built with -DKYTY_DESCDYN_MUTANT_<X>=1 each variant must fail (WILL_FAIL in CMake):
// OFFSET_IN_KEY, NO_GEN, SHADOW_IGNORES_RANGE.
#include "graphics/host_gpu/renderer/pipeline/descDyn.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::DescDyn;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DescDynTests: failed: %s\n", message);
		std::exit(1);
	}
}

// A draw: srt and shader_data are the two dynamic uploads (stream buffer 0x500, per-draw offset), one guest buffer and one image are fixed.
SetContent MakeSet(uint64_t srt_offset, uint64_t data_offset, uint64_t srt_range = 256, uint64_t guest_buffer = 0x900, uint64_t view = 0x77) {
	SetContent c;
	c.Header(3, 0, 1, 8);
	c.DynamicBuffer(0x500, srt_range, srt_offset);
	c.Header(4, 0, 1, 8);
	c.DynamicBuffer(0x500, 64, data_offset);
	c.Header(5, 0, 1, 7);
	c.Buffer(guest_buffer, 4096, 8192);
	c.Header(6, 0, 1, 1);
	c.Image(0, view, 5);
	return c;
}

void TestParseMode() {
	Check(ParseMode(nullptr) == Mode::Off && ParseMode("0") == Mode::Off && ParseMode("") == Mode::Off && ParseMode("x") == Mode::Off, "off");
	Check(ParseMode("1") == Mode::On && ParseMode("shadow") == Mode::Shadow, "on / shadow");
}

// The key has no dynamic offsets: another upload offset is the same set; another range, buffer or view is not.
void TestKeyWithoutOffsets() {
	const auto base = MakeSet(0, 256).Digest();
	Check(MakeSet(1024, 1280).Digest() == base, "other upload offsets: same set");
	Check(MakeSet(0, 256, 512).Digest() != base, "another range of the srt window");
	Check(MakeSet(0, 256, 256, 0x901).Digest() != base, "another guest buffer");
	Check(MakeSet(0, 256, 256, 0x900, 0x78).Digest() != base, "another image view");
}

// A reuse across draws with different upload offsets: the second draw finds the set of the first.
void TestReuseAcrossOffsets() {
	SetCache<int> cache;
	cache.Put({MakeSet(0, 256).Digest(), 1, 1, 11, {}});
	const auto* hit = cache.Find(MakeSet(2048, 2304).Digest(), 1, 1);
	Check(hit != nullptr && hit->set == 11, "same set, other offsets, hits");
	Check(cache.Find(MakeSet(0, 256, 512).Digest(), 1, 1) == nullptr, "another range misses");
}

// A set written against another pool or another resource generation is never bound again.
void TestGenerations() {
	SetCache<int> cache;
	const auto    hash = MakeSet(0, 256).Digest();
	cache.Put({hash, 4, 9, 11, {}});
	Check(cache.Find(hash, 4, 9) != nullptr, "same generations hit");
	Check(cache.Find(hash, 5, 9) == nullptr, "a new pool misses");
	Check(cache.Find(hash, 4, 10) == nullptr, "a new resource generation misses");
}

// A new pool drops the sets of the old one; the cache keeps at least 16 sets of one pool, the oldest is evicted first.
void TestLru() {
	SetCache<int> cache;
	for (uint64_t i = 0; i < kMinSetsPerLayout; i++) {
		cache.Put({100 + i, 1, 1, static_cast<int>(i), {}});
	}
	Check(cache.Size() == kMinSetsPerLayout, "16 sets are kept");
	for (uint64_t i = 0; i < kMinSetsPerLayout; i++) {
		Check(cache.Find(100 + i, 1, 1) != nullptr, "all 16 hit");
	}
	// The loop above touched 100 first and 115 last: 100 is the least recently used.
	cache.Put({999, 1, 1, 99, {}});
	Check(cache.Size() == kMinSetsPerLayout, "bounded");
	Check(cache.Find(100, 1, 1) == nullptr && cache.Find(115, 1, 1) != nullptr, "the least recently used goes");
	cache.Put({1000, 2, 1, 100, {}});
	Check(cache.Size() == 1, "a new pool generation clears the older sets");
}

// Shadow: a content check that sees a differing range and handle, and passes for equal content.
void TestShadowContent() {
	const auto a = MakeSet(0, 256);
	const auto b = MakeSet(4096, 4352);
	Check(ContentMismatch(a.Words(), b.Words()) == 0, "equal content (offsets are not in it)");
	Check(ContentMismatch(a.Words(), MakeSet(0, 256, 512).Words()) != 0, "different srt range");
	Check(ContentMismatch(a.Words(), MakeSet(0, 256, 256, 0x901).Words()) != 0, "different guest buffer");
	Check(ContentMismatch(a.Words(), MakeSet(0, 256, 256, 0x900, 0x78).Words()) != 0, "different view");
}

// Shadow window compare: base + dynamic offset against what the old path bound.
void TestShadowWindows() {
	const BufferWindow base {0x500, 0, 256};
	Check(WindowsEqual(base, 512, {0x500, 512, 256}), "same window");
	Check(!WindowsEqual(base, 512, {0x500, 768, 256}), "other offset");
	Check(!WindowsEqual(base, 512, {0x501, 512, 256}), "other buffer");
	Check(!WindowsEqual(base, 512, {0x500, 512, 512}), "other range");
}

void TestOffsetOrder() {
	std::vector<std::pair<uint32_t, uint32_t>> pairs {{9, 90}, {3, 30}, {5, 50}};
	std::vector<uint32_t>                      offsets;
	SortDynamicOffsets(pairs, offsets);
	Check(offsets.size() == 3 && offsets[0] == 30 && offsets[1] == 50 && offsets[2] == 90, "ascending binding order");
}

void TestVerifiedOnce() {
	VerifiedSet verified(4);
	Check(!verified.Contains(7), "unknown combination");
	verified.Add(7);
	Check(verified.Contains(7), "verified once");
	for (uint64_t i = 10; i < 20; i++) verified.Add(i);
	Check(verified.Size() <= 4, "bounded");
}

} // namespace

int main() {
	TestParseMode();
	TestKeyWithoutOffsets();
	TestReuseAcrossOffsets();
	TestGenerations();
	TestLru();
	TestShadowContent();
	TestShadowWindows();
	TestOffsetOrder();
	TestVerifiedOnce();
	std::printf("DescDynTests: ok\n");
	return 0;
}
