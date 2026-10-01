#include "libs/padScript.h"

#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/presentation/frameTiming.h"
#include "libs/controller.h"
#include "libs/padScriptParser.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
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

// Pad pose for a roll (about z) and pitch (about x) in degrees: gravity in the pad frame, in G,
// and the orientation quaternion (x, y, z, w) of Rz(roll) * Rx(pitch).
void TiltPose(double roll_deg, double pitch_deg, float* accel, float* orientation) {
	constexpr double DEG = 3.14159265358979323846 / 180.0;
	const double     r   = roll_deg * DEG;
	const double     p   = pitch_deg * DEG;
	accel[0]             = static_cast<float>(std::sin(r));
	accel[1]             = static_cast<float>(std::cos(r) * std::cos(p));
	accel[2]             = static_cast<float>(-std::cos(r) * std::sin(p));
	const double cr = std::cos(r * 0.5), sr = std::sin(r * 0.5);
	const double cp = std::cos(p * 0.5), sp = std::sin(p * 0.5);
	orientation[0]  = static_cast<float>(cr * sp);
	orientation[1]  = static_cast<float>(sr * sp);
	orientation[2]  = static_cast<float>(sr * cp);
	orientation[3]  = static_cast<float>(cr * cp);
}

void Run() {
	static constexpr Axis AXES[4] = {Axis::LeftX, Axis::LeftY, Axis::RightX, Axis::RightY};

	uint32_t applied_buttons = 0;
	int8_t   applied_sticks[4] {};
	bool     tilted = false;
	double   applied_roll = 0.0, applied_pitch = 0.0;

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
		bool     tilt = false;
		double   roll = 0.0, pitch = 0.0;
		for (const auto& entry: g_entries) {
			if (IsActive(entry, seconds, frames)) {
				buttons |= entry.buttons;
				if (entry.tilt) {
					const double now  = entry.start_unit == Unit::Frames ? frames : seconds;
					const double fade = entry.tilt_ramp > 0.0
					                        ? std::min(1.0, (now - entry.start) / entry.tilt_ramp)
					                        : 1.0;
					tilt  = true;
					roll  = entry.tilt_roll * fade;
					pitch = entry.tilt_pitch * fade;
				}
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

		// While a tilt entry is active the pose is refreshed every tick; on release it returns level.
		if (tilt || tilted) {
			if (tilt != tilted || roll != applied_roll || pitch != applied_pitch) {
				float accel[3];
				float orientation[4];
				TiltPose(roll, pitch, accel, orientation);
				SetMotionPose(HOST_INPUT_CONTROLLER_ID, accel, orientation);
				applied_roll  = roll;
				applied_pitch = pitch;
				changed       = changed || tilt != tilted;
			}
			tilted = tilt;
		}

		if (changed && g_log) {
			LOGF("[pad-script] t=%.3fs frame=%.0f buttons=0x%08x sticks=%d,%d,%d,%d tilt=%d(%.1f,%.1f)\n",
			     seconds, frames, buttons, sticks[0], sticks[1], sticks[2], sticks[3],
			     tilt ? 1 : 0, roll, pitch);
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
