#ifndef KYTY_GRAPHICS_PRESENTATION_CPCOUNTERS_H_
#define KYTY_GRAPHICS_PRESENTATION_CPCOUNTERS_H_

// S132: pure helpers behind the CP/submit time counters, the dispatch classes and KYTY_DIAG_PROTO_SKIP.
// No Vulkan, no globals: the renderer only counts (and, with the diag switch, skips prototype draws).

#include <cstdint>

namespace Libs::Graphics::FrameTiming {

// Where a vkCmdDispatch came from, in the order of the dispatch_* columns of KYTY_FRAME_LOG.
enum class DispatchClass : uint32_t {
	Guest,                // the guest's own compute dispatches (direct and indirect)
	ProtoUnpack,          // per-vertex prototype: vertex fetch unpack pass
	ProtoCapture,         // per-vertex prototype: capture pass with the native (64 vertices per group) VS
	ProtoCaptureNonNative, // ... capture pass of a VS that cannot run natively (one group per vertex)
	Tiler,                // image tiling / detiling passes
	Fault,                // fault manager page-bitmap processing
	Other,                // anything not classified above
	Count,
};

// The capture pass has two flavors; tell them apart by the program the draw uses.
constexpr DispatchClass ProtoCaptureClass(bool native_capture) {
#if defined(KYTY_CPCOUNTERS_MUTANT_NATIVE_SWAP)
	return native_capture ? DispatchClass::ProtoCaptureNonNative : DispatchClass::ProtoCapture;
#else
	return native_capture ? DispatchClass::ProtoCapture : DispatchClass::ProtoCaptureNonNative;
#endif
}

// Nanoseconds from begin to end on one clock; a clock step back counts as zero, never as a huge value.
constexpr uint64_t ElapsedNs(int64_t begin_ns, int64_t end_ns) {
#if defined(KYTY_CPCOUNTERS_MUTANT_NO_CLAMP)
	return static_cast<uint64_t>(end_ns - begin_ns);
#else
	return end_ns > begin_ns ? static_cast<uint64_t>(end_ns - begin_ns) : 0u;
#endif
}

// KYTY_DIAG_PROTO_SKIP is on only for the value "1" (an empty or "0" variable keeps the path).
constexpr bool DiagFlagOn(const char* value) {
#if defined(KYTY_CPCOUNTERS_MUTANT_FLAG_ANY)
	return value != nullptr;
#else
	return value != nullptr && value[0] == '1' && value[1] == '\0';
#endif
}

enum class ProtoDrawAction : uint32_t {
	Skip,        // the classifier rejected the draw (topology/restart/index count): nothing runs, as before
	Execute,     // unpack + capture + replay
	DiagSkipped, // KYTY_DIAG_PROTO_SKIP=1: ablation, a supported draw that is not drawn
};

constexpr ProtoDrawAction ProtoDrawDecision(bool verdict_run, bool diag_skip) {
#if defined(KYTY_CPCOUNTERS_MUTANT_SKIP_IGNORED)
	(void)diag_skip;
	return verdict_run ? ProtoDrawAction::Execute : ProtoDrawAction::Skip;
#else
	if (!verdict_run) return ProtoDrawAction::Skip;
	return diag_skip ? ProtoDrawAction::DiagSkipped : ProtoDrawAction::Execute;
#endif
}

} // namespace Libs::Graphics::FrameTiming

#endif /* KYTY_GRAPHICS_PRESENTATION_CPCOUNTERS_H_ */
