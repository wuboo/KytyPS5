#ifndef KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_
#define KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_

#include <cstdint>

namespace Libs::Graphics::FrameTiming {

// Called once per successfully presented frame. Emits a Tracy frame mark (with --profile) and,
// when the KYTY_FRAME_LOG environment variable names a file, appends "<frame>,<host_ns>" to it.
void OnFramePresented();

// Number of frames presented since startup.
uint64_t PresentedFrames();

} // namespace Libs::Graphics::FrameTiming

#endif /* KYTY_GRAPHICS_PRESENTATION_FRAMETIMING_H_ */
