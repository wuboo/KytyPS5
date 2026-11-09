#ifndef KYTY_LIBS_PADSCRIPT_H_
#define KYTY_LIBS_PADSCRIPT_H_

#include <cstddef>

namespace Libs::Controller {
struct PadData;
} // namespace Libs::Controller

namespace Libs::Controller::PadScript {

// Scripted pad input for reproducible benchmark runs, driven through the host-input controller.
//   KYTY_PAD_SCRIPT       inline script, or @path to a file (grammar in padScriptParser.h)
//   KYTY_PAD_HOLD         default hold of a seconds point, in ms (default 300)
//   KYTY_PAD_FRAME_HOLD   default hold of a frame point, in presented frames (default 8)
//   KYTY_PAD_SCRIPT_LOG   when set, log every change of the scripted state
// Seconds and frames are counted from the first pad read by the game.
void Start();
void Stop();

// Called by the pad read entry points; the first call sets the script origin.
void OnPadRead();

// Pad recording for replaying a real play session in a benchmark run, keyed by guest flip:
//   KYTY_PAD_RECORD   path; every change of the first player's pad state is appended as
//                     "flip buttons lx ly rx ry l2 r2 ox oy oz ow ax ay az gx gy gz"
//   KYTY_PAD_REPLAY   path of such a recording; the first player's reads return the recorded
//                     state of the current guest flip instead of the host pad
// Called on every filled PadData of the first player; true when a replay overrode it.
bool RecordOrReplay(PadData* data);

// Guest text output (stdout/stderr). A recording notes each "Level has started" line with its flip
// as "L flip name"; a replay re-anchors its flip clock on the matching level start, so a load that
// takes a different number of flips does not shift the inputs that follow it.
void OnGuestText(const void* text, size_t size);

} // namespace Libs::Controller::PadScript

#endif /* KYTY_LIBS_PADSCRIPT_H_ */
