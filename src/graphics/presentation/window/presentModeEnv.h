#ifndef KYTY_GRAPHICS_PRESENTATION_WINDOW_PRESENTMODEENV_H_
#define KYTY_GRAPHICS_PRESENTATION_WINDOW_PRESENTMODEENV_H_

// S585 (KYTY_PRESENT_MODE=fifo|mailbox|immediate): the host swapchain's present mode without editing the command line. The bench scripts
// hardcode --present-mode Immediate (S584 notes); the env wins over the flag when set, unset / anything else = the flag (as before).
// Pure; tests/GuestWaitTests.cpp pins the parse.

#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::PresentModeEnv {

enum class Override : unsigned char { None, Fifo, Mailbox, Immediate };

inline Override Parse(const char* text) {
	if (text == nullptr) return Override::None;
	if (std::strcmp(text, "fifo") == 0 || std::strcmp(text, "Fifo") == 0) return Override::Fifo;
	if (std::strcmp(text, "mailbox") == 0 || std::strcmp(text, "Mailbox") == 0) return Override::Mailbox;
	if (std::strcmp(text, "immediate") == 0 || std::strcmp(text, "Immediate") == 0) return Override::Immediate;
	return Override::None;
}

inline Override Env() {
	static const Override value = Parse(std::getenv("KYTY_PRESENT_MODE"));
	return value;
}

// `Mode` is Config::PresentMode (Fifo, Mailbox, Immediate); kept generic so the test needs no config.
template <typename Mode>
Mode Resolve(Mode from_flag, Override value) {
	switch (value) {
		case Override::Fifo: return Mode::Fifo;
		case Override::Mailbox: return Mode::Mailbox;
		case Override::Immediate: return Mode::Immediate;
		default: return from_flag;
	}
}

} // namespace Libs::Graphics::PresentModeEnv

#endif
