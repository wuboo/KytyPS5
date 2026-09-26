#ifndef KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_
#define KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_

#include <cstdint>

namespace Libs::Graphics::FrameTiming {

// Called once per successfully presented frame. Emits a Tracy frame mark (with --profile) and,
// when the KYTY_FRAME_LOG environment variable names a file, appends
// "<frame>,<host_ns>,<flips>,<gpu_busy_ns>" to it: cumulative guest flips and cumulative time
// the GPU command thread spent processing, so a reader can tell re-presented frames from real
// ones and see how busy the GPU thread is even when the frame rate is capped by the vblank.
void OnFramePresented();

// Called once per guest flip that reached the screen (as opposed to a re-presented frame).
void OnGuestFlip();

// Adds time the GPU command thread spent processing a submission or command.
void AddGpuBusy(uint64_t ns);

// Number of frames presented since startup.
uint64_t PresentedFrames();

} // namespace Libs::Graphics::FrameTiming

#endif /* KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_ */
