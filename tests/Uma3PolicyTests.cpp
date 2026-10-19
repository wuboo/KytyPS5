#include "graphics/host_gpu/memory/uma3Policy.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Built five times: normally, and once per mutant (KYTY_UMA_MUTANT_TEXEL_WINDOW,
// KYTY_UMA_MUTANT_WINDOW_NO_ADJUST, KYTY_UMA_MUTANT_SKIP_LEGACY,
// KYTY_UMA_MUTANT_GLOBAL_IGNORES_WRITES, KYTY_UMA_MUTANT_NARROW_IGNORES_WRITTEN). Every mutant build must FAIL (ctest WILL_FAIL).

namespace {

namespace Uma3 = Libs::Graphics::Uma3;
using Fence    = Libs::Graphics::WriteFenceTable;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "Uma3PolicyTests: failed: %s\n", message);
		std::exit(1); // not abort(): ctest WILL_FAIL does not invert signals
	}
}

void TestRoute() {
	int  asked = 0;
	auto image = [&](bool answer) {
		return [&asked, answer] {
			++asked;
			return answer;
		};
	};
	// Plain buffer (descriptor, vertex, index, indirect arguments): always a window.
	Check(Uma3::RouteBuffer(true, false, false, image(true)) == Uma3::Route::Window,
	      "plain read is a window");
	Check(Uma3::RouteBuffer(true, false, true, image(true)) == Uma3::Route::Window,
	      "plain write is a window");
	Check(asked == 0, "the texture cache is not asked for a plain buffer");
	// Not backed by the arena in one run: copy path.
	Check(Uma3::RouteBuffer(false, false, false, image(false)) == Uma3::Route::LegacyUncovered,
	      "uncovered range keeps the copy path");
	// Texel buffer / DMA source: window unless the memory holds an image.
	Check(Uma3::RouteBuffer(true, true, false, image(false)) == Uma3::Route::Window,
	      "texel read without an image is a window");
	Check(Uma3::RouteBuffer(true, true, false, image(true)) == Uma3::Route::LegacyImageSync,
	      "texel read of image memory needs the image-to-buffer copy");
	// A write never needs the image copy (the image is invalidated instead).
	Check(Uma3::RouteBuffer(true, true, true, image(true)) == Uma3::Route::Window,
	      "texel write is a window");
}

void TestSplitWindow() {
	constexpr uint64_t max_range = 1ull << 27;
	auto               s         = Uma3::SplitWindow(0x12345, 256, 0x1000, max_range);
	Check(s.ok && s.aligned == 0x12300 && s.adjustment == 0x45, "offset split at 256");
	s = Uma3::SplitWindow(0x4000, 256, 0x1000, max_range);
	Check(s.ok && s.aligned == 0x4000 && s.adjustment == 0, "aligned offset is untouched");
	s = Uma3::SplitWindow(0x40f0, 16, 64, max_range);
	Check(s.ok && s.aligned == 0x40f0 && s.adjustment == 0, "16-byte alignment");
	s = Uma3::SplitWindow(0x40ff, 16, 64, max_range);
	Check(s.ok && s.aligned == 0x40f0 && s.adjustment == 15, "15-byte remainder");
	// A remainder the shader cannot carry (>= 256) is refused.
	s = Uma3::SplitWindow(0x4000 + 300, 512, 64, max_range, 256);
	Check(!s.ok, "adjustment beyond the shader limit");
	// The adjusted size must still fit the range limit.
	s = Uma3::SplitWindow(0x4000 + 8, 16, max_range - 4, max_range);
	Check(!s.ok, "size plus adjustment exceeds the range limit");
	// The descriptor window always starts aligned (what the GPU requires).
	for (uint64_t offset = 0; offset < 600; offset += 7) {
		const auto w = Uma3::SplitWindow(offset, 64, 16, max_range, 64);
		if (w.ok) {
			Check(w.aligned % 64 == 0 && w.aligned + w.adjustment == offset,
			      "window starts aligned and keeps the bytes");
		}
	}
}

void TestLegacyRanges() {
	Uma3::LegacyRanges legacy;
	Check(legacy.Empty() && Uma3::TrackerCanSkip(legacy, 0x1000, 0x1000),
	      "nothing registered: skip");
	legacy.Add(0x10000, 0x4000);
	Check(!Uma3::TrackerCanSkip(legacy, 0x10000, 4), "registered start: no skip");
	Check(!Uma3::TrackerCanSkip(legacy, 0x13fff, 1), "registered last byte: no skip");
	Check(Uma3::TrackerCanSkip(legacy, 0x14000, 0x1000), "just after: skip");
	Check(Uma3::TrackerCanSkip(legacy, 0xf000, 0x1000), "just before: skip");
	Check(!Uma3::TrackerCanSkip(legacy, 0xf000, 0x1001), "overlap by one byte: no skip");
	legacy.Add(0x14000, 0x4000); // adjacent: merges
	legacy.Add(0x30000, 0x1000);
	Check(legacy.Count() == 2, "adjacent runs merge");
	Check(!Uma3::TrackerCanSkip(legacy, 0x17000, 1), "merged run");
	legacy.Add(0x12000, 0x20000); // swallows both
	Check(legacy.Count() == 1 && !Uma3::TrackerCanSkip(legacy, 0x31000, 1), "overlap swallows");
	Check(Uma3::TrackerCanSkip(legacy, 0x32000, 0x100), "beyond the end: skip");
}

void TestUnknownWriterTick() {
	Uma3::UnknownWriterTick stage1;
	stage1.Submit(5, false, false);
	Check(stage1.Value() == 5, "stage 1: any submit with direct access counts");
	Uma3::UnknownWriterTick stage3;
	stage3.Submit(5, false, true);
	Check(stage3.Value() == 0, "stage 3: a submit without pointer stores does not");
	stage3.Submit(6, true, true);
	Check(stage3.Value() == 6, "stage 3: a pointer-storing submit does");
	stage3.Submit(7, false, true);
	Check(stage3.Value() == 6, "stage 3: the next reader-only submit leaves it");
	stage3.Submit(4, true, true);
	Check(stage3.Value() == 6, "ticks never go back");
}

void TestPendingTimelines() {
	Fence fence;
	constexpr uint64_t label = 0x400000;
	// An EOP label that is written when tick 7 retires.
	fence.Record(label, 8, 0, 7);
	uint64_t retired = 6;
	const auto is_retired = [&](uint32_t, uint64_t tick) { return tick <= retired; };
	Check(Uma3::PendingTimelines(fence.MaxTicks(label, 4), 0, is_retired) == 1u,
	      "WAIT_REG_MEM on the label waits for tick 7");
	Check(Uma3::PendingTimelines(fence.MaxTicks(label + 0x1000, 4), 0, is_retired) == 0u,
	      "an unrelated range does not wait");
	retired = 7;
	Check(Uma3::PendingTimelines(fence.MaxTicks(label, 4), 0, is_retired) == 0u,
	      "retired: nothing to wait for");
	// The unknown-range writer applies to every range, on the graphics timeline.
	retired = 7;
	Check(Uma3::PendingTimelines(fence.MaxTicks(label + 0x1000, 4), 9, is_retired) == 1u,
	      "an in-flight pointer writer makes every range pending");
	Check(Uma3::PendingTimelines(fence.MaxTicks(label + 0x1000, 4), 7, is_retired) == 0u,
	      "a retired pointer writer does not");
	// A writer on another timeline waits on that timeline only.
	fence.Record(0x900000, 8, 2, 3);
	const auto only_t0 = [](uint32_t timeline, uint64_t) { return timeline == 0; };
	Check(Uma3::PendingTimelines(fence.MaxTicks(0x900000, 4), 0, only_t0) == (1u << 2),
	      "timeline 2 writer: wait on timeline 2");
}


// KYTY_OPT_ON=bda_wait_narrow: ranges of cells that were never written as a window / fill / copy /
// label ignore the global pointer-writer tick; written cells and the "all ranges" read keep it.
void TestBdaWaitNarrow() {
	Uma3::WrittenCells written;
	constexpr uint64_t window = 0x40000000;           // a window written once
	constexpr uint64_t other  = 0x80000000;           // never written
	written.Mark(window + 0x1000, 0x2000);
	Check(written.Any(window, 0x1000000 - 1) && !written.Any(other, 0x1000), "cell bookkeeping");
	// Marks across a cell boundary cover both cells.
	written.Mark(0x100000 * 5 - 8, 16);
	Check(written.Any(0x100000 * 4, 16) && written.Any(0x100000 * 5, 16), "mark across a cell boundary");
	const uint64_t global = 9;
	Check(Uma3::EffectiveUnknownTick(global, false, written, other, 64) == global,
	      "option off: the global tick always applies");
	Check(Uma3::EffectiveUnknownTick(global, true, written, other, 64) == 0,
	      "narrow: a range in an unwritten cell ignores the global tick");
	Check(Uma3::EffectiveUnknownTick(global, true, written, window + 0x1800, 64) == global,
	      "narrow: a range in a written cell keeps the global tick");
	Check(Uma3::EffectiveUnknownTick(global, true, written, other, 0) == global,
	      "narrow: size 0 (all ranges) keeps the global tick");
	// End to end with PendingTimelines: the in-flight pointer writer (tick 9) stalls only written cells.
	const auto is_retired = [](uint32_t, uint64_t tick) { return tick <= 7; };
	Fence      fence;
	Check(Uma3::PendingTimelines(fence.MaxTicks(other, 64),
	                             Uma3::EffectiveUnknownTick(global, true, written, other, 64), is_retired) == 0u,
	      "narrow: read of an unwritten range does not wait");
	Check(Uma3::PendingTimelines(fence.MaxTicks(window + 0x1800, 64),
	                             Uma3::EffectiveUnknownTick(global, true, written, window + 0x1800, 64),
	                             is_retired) == 1u,
	      "narrow: read of a written range waits for the pointer writer");
	// A recorded writer of the range itself is never narrowed away.
	fence.Record(other, 64, 0, 8);
	Check(Uma3::PendingTimelines(fence.MaxTicks(other, 64),
	                             Uma3::EffectiveUnknownTick(global, true, written, other, 64), is_retired) == 1u,
	      "narrow: the range's own writer still waits");
	// Wait attribution: range writer pending -> Range; only the global tick -> Global.
	Fence global_only;
	Check(Uma3::ClassifyWait(fence.MaxTicks(other, 64), is_retired) == Uma3::WaitReason::Range, "reason range");
	Check(Uma3::ClassifyWait(global_only.MaxTicks(other, 64), is_retired) == Uma3::WaitReason::Global,
	      "reason global");
}

} // namespace

int main() {
	TestRoute();
	TestSplitWindow();
	TestLegacyRanges();
	TestUnknownWriterTick();
	TestPendingTimelines();
	TestBdaWaitNarrow();
	std::puts("Uma3PolicyTests: ok");
	return 0;
}
