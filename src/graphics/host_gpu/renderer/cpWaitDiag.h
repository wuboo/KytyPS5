#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CPWAITDIAG_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CPWAITDIAG_H_

// S413 (DIAG v5, default off): why the command processor thread blocks in MasterSemaphore::Wait.
//   KYTY_DIAG_CP_WAIT=1   a wait of >= 1 ms on the CP thread logs one [cp-wait] line (flip, PM4 opcode, CP breadcrumb site at the call,
//                         cause, wait_us, lag = ticks the GPU is behind) and adds the time to the frames.csv column cp_wait_cause_<cause>_ns.
// The cause is a thread-local tag set by RAII scopes at the call sites that end in a wait (innermost scope wins, leaving restores the outer
// one, outside every scope it is `other`). This header is the pure part (tests/CpWaitDiagTests.cpp pins it). Nothing here feeds a shader
// or pipeline cache key.

#include "graphics/host_gpu/renderer/rfOpDiag.h"
#include "graphics/host_gpu/renderer/syncCensus.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace Libs::Graphics::CpWait {

// Order = the cp_wait_cause_*_ns columns of FrameTiming::Counter (frameTiming.cpp checks the names against ColumnName).
enum class Cause : uint8_t { RfGpu, Spec, Dcc, ReadMem, Eager, DeleteDrain, Drain, Stream, DescHeap, Other, Count };
constexpr uint32_t kCauseCount = static_cast<uint32_t>(Cause::Count);

constexpr const char* CauseName(Cause cause) {
	constexpr const char* names[] = {"rf_gpu", "spec", "dcc", "readmem", "eager", "delete_drain", "drain", "stream", "desc_heap", "other"};
	return static_cast<uint32_t>(cause) < kCauseCount ? names[static_cast<uint32_t>(cause)] : "?";
}
constexpr const char* ColumnName(uint32_t cause) {
	constexpr const char* names[] = {"cp_wait_cause_rf_gpu_ns",       "cp_wait_cause_spec_ns",     "cp_wait_cause_dcc_ns",
	                                 "cp_wait_cause_readmem_ns",      "cp_wait_cause_eager_ns",    "cp_wait_cause_delete_drain_ns",
	                                 "cp_wait_cause_drain_ns",        "cp_wait_cause_stream_ns",   "cp_wait_cause_desc_heap_ns",
	                                 "cp_wait_cause_other_ns"};
	return cause < kCauseCount ? names[cause] : "?";
}

inline bool Enabled() {
	static const bool on = SyncCensus::FlagOn("KYTY_DIAG_CP_WAIT");
	return on;
}

inline Cause& CurrentCauseRef() {
	static thread_local Cause cause = Cause::Other;
	return cause;
}
inline Cause CurrentCause() {
	return CurrentCauseRef();
}

class CauseScope {
public:
	explicit CauseScope(Cause cause): m_previous(CurrentCauseRef()) {
#if defined(KYTY_CPWAIT_MUTANT_NO_CAUSE)
		(void)cause;
#else
		CurrentCauseRef() = cause;
#endif
	}
	~CauseScope() { CurrentCauseRef() = m_previous; }
	CauseScope(const CauseScope&)            = delete;
	CauseScope& operator=(const CauseScope&) = delete;

private:
	Cause m_previous;
};

// Cause of an explicit ReadMemory by its SyncCensus site (the tag the caller set).
constexpr Cause CauseOfSite(SyncCensus::Site site) {
	switch (site) {
		case SyncCensus::Site::RfGpu: return Cause::RfGpu;
		case SyncCensus::Site::Spec: return Cause::Spec;
		case SyncCensus::Site::Dcc: return Cause::Dcc;
		default: return Cause::ReadMem;
	}
}

constexpr uint64_t kMinWaitUs = 1000;

struct Event {
	uint64_t    flip    = 0;
	uint32_t    opcode  = RfOp::kNoOpcode;
	std::string site    = "none";
	Cause       cause   = Cause::Other;
	uint64_t    wait_us = 0;
	uint64_t    lag     = 0;
};

inline std::string FormatLine(const Event& event) {
	char op[16];
	if (event.opcode == RfOp::kNoOpcode) {
		std::snprintf(op, sizeof(op), "none");
	} else {
		std::snprintf(op, sizeof(op), "0x%02x", event.opcode);
	}
	char text[512];
	std::snprintf(text, sizeof(text), "[cp-wait] flip=%llu op=%s site=%s cause=%s wait_us=%llu lag=%llu",
	              static_cast<unsigned long long>(event.flip), op, event.site.c_str(), CauseName(event.cause),
	              static_cast<unsigned long long>(event.wait_us), static_cast<unsigned long long>(event.lag));
	return text;
}

// Waits shorter than 1 ms are not logged; at most 400 lines plus every line of flips 6300-6310.
constexpr bool ShouldLog(uint64_t index, uint64_t flip, uint64_t wait_us) {
	return wait_us >= kMinWaitUs && RfOp::ShouldLog(index, flip);
}

} // namespace Libs::Graphics::CpWait

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_CPWAITDIAG_H_ */
