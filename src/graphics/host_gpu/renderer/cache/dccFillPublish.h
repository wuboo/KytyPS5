#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCFILLPUBLISH_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_CACHE_DCCFILLPUBLISH_H_

#include "graphics/host_gpu/renderer/cache/gpuWriteLedger.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::DccFillPublish {

// KYTY_DCC_KNOWN=0|1|shadow (unset = 1). A guest range whose whole GPU-modified content is one remembered fill (the ledger says the newest
// GPU write to it was a FillBuffer) is handed to the guest by writing the pattern into its backing and dropping the GPU-modified
// state, instead of copyBuffer + Wait (a drain of the queue) on a write fault or a read. A range that is only partly known, or
// touched by a later GPU write, takes the old path. shadow = the old path runs and its result stands; the bytes it downloaded are
// compared with the prediction.

enum class Mode : uint32_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) {
#ifdef KYTY_DCCPUBLISH_MUTANT_DEFAULT_OFF
		return Mode::Off;
#else
		return Mode::On; // S403: on by default; KYTY_DCC_KNOWN=0 restores the drain
#endif
	}
	if (std::strcmp(text, "1") == 0) {
		return Mode::On;
	}
	if (std::strcmp(text, "shadow") == 0) {
		return Mode::Shadow;
	}
	return Mode::Off;
}

// KYTY_DCC_KNOWN, read once.
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_DCC_KNOWN"));
	return mode;
}

enum class Reason : uint32_t {
	Publish,            // the whole range is a known fill and nothing newer wrote it
	FallbackPartial,    // not (all) known: a fragment is unknown, was forgotten, evicted or filled by another value
	FallbackNewerWrite, // a later GPU write touched it, or a queued download of it has not landed
};

struct Decision {
	bool     publish = false;
	Reason   reason  = Reason::FallbackPartial;
	uint32_t value   = 0;
};

// q: ledger.QueryFill(range). pending_download: a queued (eager) download of the range has not reached the backing yet; it
// would overwrite the published pattern with older bytes.
inline Decision Decide(const FillQuery& q, bool pending_download) {
	Decision decision {};
	decision.value = q.value;
#ifdef KYTY_DCCPUBLISH_MUTANT_PARTIAL
	// Mutant: a fragment that is not known counts as known (the value of any fill that touches the range).
	if (q.seen && q.kind == FillKind::Partial) {
		decision.publish = true;
		decision.reason  = Reason::Publish;
		return decision;
	}
#endif
	if (q.kind == FillKind::NewerWrite) {
		decision.reason = Reason::FallbackNewerWrite;
		return decision;
	}
	if (q.kind != FillKind::Known) {
		decision.reason = Reason::FallbackPartial;
		return decision;
	}
	if (pending_download) {
		decision.reason = Reason::FallbackNewerWrite;
		return decision;
	}
	decision.publish = true;
	decision.reason  = Reason::Publish;
	return decision;
}

// S404 (diagnostics): why a piece took the old path, finer than Reason. Order = the dcc_fallback_<why>_n counters.
enum class Why : uint32_t { NoFill, Partial, NewerWrite, PendingDownload, Other, Count };
constexpr const char* WhyName(Why why) {
	switch (why) {
		case Why::NoFill: return "no_fill";
		case Why::Partial: return "partial";
		case Why::NewerWrite: return "newer_write";
		case Why::PendingDownload: return "pending_download";
		default: return "other";
	}
}
// Same inputs as Decide; for a piece Decide refused.
inline Why Explain(const FillQuery& q, bool pending_download) {
	switch (q.kind) {
		case FillKind::Partial: return q.seen ? Why::Partial : Why::NoFill;
		case FillKind::NewerWrite: return Why::NewerWrite;
#ifdef KYTY_DCCEXPLAIN_MUTANT_PENDING_AS_OTHER
		case FillKind::Known: return Why::Other;
#else
		case FillKind::Known: return pending_download ? Why::PendingDownload : Why::Other;
#endif
	}
	return Why::Other;
}

// The bytes of [address, address + size) when the dword-aligned fill `value` covers them.
inline std::vector<uint8_t> Pattern(uint32_t value, uint64_t address, uint64_t size) {
	std::vector<uint8_t> bytes(static_cast<size_t>(size));
	for (size_t i = 0; i < bytes.size(); ++i) {
		bytes[i] = static_cast<uint8_t>(value >> (8 * ((address + i) & 3u)));
	}
	return bytes;
}

// Publishes one piece: write the pattern into the guest backing, then end its GPU-modified state (dirty ranges, tracker, ledger).
template <class WriteBacking, class Unmark>
void Apply(uint64_t address, uint64_t size, uint32_t value, WriteBacking&& write_backing, Unmark&& unmark) {
	const auto bytes = Pattern(value, address, size);
	write_backing(address, bytes.data(), bytes.size());
#ifndef KYTY_DCCPUBLISH_MUTANT_NO_UNMARK
	unmark(address, size);
#else
	(void)unmark;
#endif
}

// Offset of the first byte of `actual` that differs from the fill pattern, or size when none does.
inline uint64_t FirstDifference(uint32_t value, uint64_t address, const uint8_t* actual, uint64_t size) {
	for (uint64_t i = 0; i < size; ++i) {
		if (actual[i] != static_cast<uint8_t>(value >> (8 * ((address + i) & 3u)))) {
			return i;
		}
	}
	return size;
}

// MaterializeColorClear publishes the expanded keys of a converted fast clear into the metadata backing (not a guest write).
inline constexpr uint8_t kMetadataKeyByte = 0xff;

// The DCC/CMASK clear code a fill value stands for: one code repeated in all four bytes. False = no fast clear.
inline bool CodeFromFill(uint32_t value, uint8_t* code) {
	*code = static_cast<uint8_t>(value & 0xffu);
	return value == static_cast<uint32_t>(*code) * 0x01010101u;
}

} // namespace Libs::Graphics::DccFillPublish

#endif
