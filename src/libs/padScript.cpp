#include "libs/padScript.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/presentation/frameTiming.h"
#include "libs/controller.h"
#include "libs/padScriptParser.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Controller::PadScript {

namespace {

using Clock = std::chrono::steady_clock;

std::vector<Entry>    g_entries;
std::atomic<bool>     g_stop {false};
std::atomic<bool>     g_origin_claimed {false};
std::atomic<bool>     g_origin_ready {false};
std::atomic<int64_t>  g_origin_ns {0};
std::atomic<uint64_t> g_origin_frame {0};
bool                  g_log = false;

double EnvNumber(const char* name, double fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr || *value == '\0') {
		return fallback;
	}
	char*        end    = nullptr;
	const double parsed = std::strtod(value, &end);
	EXIT_IF(end == value || *end != '\0' || parsed <= 0.0);
	return parsed;
}

std::string LoadScript(const char* value) {
	if (value[0] != '@') {
		return value;
	}
	std::ifstream file(value + 1);
	if (!file) {
		EXIT("KYTY_PAD_SCRIPT: can't open %s\n", value + 1);
	}
	std::stringstream text;
	text << file.rdbuf();
	return text.str();
}

int StickValue(int8_t direction) {
	return direction < 0 ? 0 : (direction > 0 ? 255 : 128);
}

void Run() {
	static constexpr Axis AXES[4] = {Axis::LeftX, Axis::LeftY, Axis::RightX, Axis::RightY};

	uint32_t applied_buttons = 0;
	int8_t   applied_sticks[4] {};

	while (!g_stop.load(std::memory_order_relaxed)) {
		std::this_thread::sleep_for(std::chrono::milliseconds(2));
		if (!g_origin_ready.load(std::memory_order_acquire)) {
			continue;
		}
		const double seconds = static_cast<double>(Clock::now().time_since_epoch().count() -
		                                           g_origin_ns.load(std::memory_order_relaxed)) *
		                       static_cast<double>(Clock::period::num) /
		                       static_cast<double>(Clock::period::den);
		const double frames  = static_cast<double>(Graphics::FrameTiming::PresentedFrames() -
		                                           g_origin_frame.load(std::memory_order_relaxed));

		uint32_t buttons = 0;
		int8_t   sticks[4] {};
		for (const auto& entry: g_entries) {
			if (IsActive(entry, seconds, frames)) {
				buttons |= entry.buttons;
				for (int axis = 0; axis < 4; axis++) {
					if (entry.sticks[axis] != 0) {
						sticks[axis] = entry.sticks[axis];
					}
				}
			}
		}

		bool changed = false;
		for (uint32_t bit = 1; bit != 0; bit <<= 1u) {
			if (((buttons ^ applied_buttons) & bit) != 0) {
				SetButton(HOST_INPUT_CONTROLLER_ID, bit, (buttons & bit) != 0);
				changed = true;
			}
		}
		// Triggers are analog on the pad; mirror the digital L2/R2 onto their axes.
		const uint32_t trigger_bits[2] = {0x00000100u, 0x00000200u};
		const Axis     trigger_axes[2] = {Axis::TriggerLeft, Axis::TriggerRight};
		for (int i = 0; i < 2; i++) {
			if (((buttons ^ applied_buttons) & trigger_bits[i]) != 0) {
				SetAxis(HOST_INPUT_CONTROLLER_ID, trigger_axes[i],
				        (buttons & trigger_bits[i]) != 0 ? 255 : 0);
			}
		}
		for (int axis = 0; axis < 4; axis++) {
			if (sticks[axis] != applied_sticks[axis]) {
				SetAxis(HOST_INPUT_CONTROLLER_ID, AXES[axis], StickValue(sticks[axis]));
				applied_sticks[axis] = sticks[axis];
				changed              = true;
			}
		}
		applied_buttons = buttons;

		if (changed && g_log) {
			LOGF("[pad-script] t=%.3fs frame=%.0f buttons=0x%08x sticks=%d,%d,%d,%d\n", seconds,
			     frames, buttons, sticks[0], sticks[1], sticks[2], sticks[3]);
		}
	}
}

} // namespace

void Start() {
	const char* value = std::getenv("KYTY_PAD_SCRIPT");
	if (value == nullptr || *value == '\0') {
		return;
	}
	Defaults defaults;
	defaults.point_seconds = EnvNumber("KYTY_PAD_HOLD", 300.0) / 1000.0;
	defaults.point_frames  = EnvNumber("KYTY_PAD_FRAME_HOLD", 8.0);

	std::string error;
	if (!Parse(LoadScript(value), defaults, g_entries, error)) {
		EXIT("KYTY_PAD_SCRIPT: %s\n", error.c_str());
	}
	g_log = std::getenv("KYTY_PAD_SCRIPT_LOG") != nullptr;
	LOGF("[pad-script] %zu entries loaded\n", g_entries.size());

	// Detached: the emulator often leaves through exit(), where a joinable global thread
	// would terminate the process.
	g_stop.store(false);
	std::thread(Run).detach();
}

void Stop() {
	g_stop.store(true);
}

void OnPadRead() {
	if (g_entries.empty() || g_origin_claimed.load(std::memory_order_relaxed)) {
		return;
	}
	bool expected = false;
	if (g_origin_claimed.compare_exchange_strong(expected, true)) {
		g_origin_frame.store(Graphics::FrameTiming::PresentedFrames(), std::memory_order_relaxed);
		g_origin_ns.store(Clock::now().time_since_epoch().count(), std::memory_order_relaxed);
		g_origin_ready.store(true, std::memory_order_release);
		if (g_log) {
			LOGF("[pad-script] origin at first pad read, frame=%llu\n",
			     static_cast<unsigned long long>(g_origin_frame.load()));
		}
	}
}

} // namespace Libs::Controller::PadScript
