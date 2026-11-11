#ifndef KYTY_COMMON_FP_ENV_H_
#define KYTY_COMMON_FP_ENV_H_

#include "common/common.h"

#include <cstdint>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

// Read by the fiber switch. 1 only while KYTY_GUEST_FPENV=1.
extern "C" uint8_t kyty_s536_guest_fp_env;

namespace Common {

struct FpEnv {
	uint32_t mxcsr;
	uint16_t control;
};

inline constexpr FpEnv GuestFpEnv {0x9fc0, 0x037f};

// Unset or any value other than "1" keeps the host FP environment.
inline bool GuestFpEnvEnabled() {
	const char* value = std::getenv("KYTY_GUEST_FPENV");
	const bool  on    = value != nullptr && value[0] == '1' && value[1] == '\0';
	kyty_s536_guest_fp_env = static_cast<uint8_t>(on);
	return on;
}

#if defined(__x86_64__) || defined(_M_X64)
inline FpEnv ReadFpEnv() {
	FpEnv env {_mm_getcsr(), 0};
	asm volatile("fnstcw %0" : "=m"(env.control));
	return env;
}

inline void WriteFpEnv(FpEnv env) {
	_mm_setcsr(env.mxcsr);
	asm volatile("fldcw %0" : : "m"(env.control) : "memory");
}
#else
inline FpEnv ReadFpEnv() {
	return {};
}

inline void WriteFpEnv(FpEnv /*env*/) {}
#endif

} // namespace Common

#endif /* KYTY_COMMON_FP_ENV_H_ */
