#ifndef KYTY_LIBS_PADSCRIPT_H_
#define KYTY_LIBS_PADSCRIPT_H_

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

} // namespace Libs::Controller::PadScript

#endif /* KYTY_LIBS_PADSCRIPT_H_ */
