#ifndef GRAPHICS_PRESENTATION_BENCH_FRAME_LOG_H_
#define GRAPHICS_PRESENTATION_BENCH_FRAME_LOG_H_

namespace Libs::Graphics {

// Writes one CSV row per guest flip when KYTY_FRAME_LOG names a file. Bench builds only.
void BenchNoteGuestFlip();

} // namespace Libs::Graphics

#endif
