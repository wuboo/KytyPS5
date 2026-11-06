#ifndef GRAPHICS_PRESENTATION_BENCH_FRAME_LOG_H_
#define GRAPHICS_PRESENTATION_BENCH_FRAME_LOG_H_

#include <cstdint>

namespace Libs::Graphics {

// Writes one CSV row per guest flip when KYTY_FRAME_LOG names a file. Bench builds only.
void BenchNoteGuestFlip();

// One successful host present. The pad script counts these from the first pad read.
void BenchNotePresented();

uint64_t BenchGuestFlips();
uint64_t BenchPresentedFrames();

} // namespace Libs::Graphics

#endif
