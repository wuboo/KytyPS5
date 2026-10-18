#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_EOP_COALESCE_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_EOP_COALESCE_H

// S429 (KYTY_EOP_COALESCE=0|1|shadow): an end-of-pipe interrupt is a deferred priority operation that runs once the tick it was recorded in has
// completed, and all it does is KernelTriggerEvent(eq, event_id, GRAPHICS, udata = context_id) for every registration of the event. The label of
// the EOP packet is written on the CPU while the packet is recorded, so it is already visible when the interrupt runs. Two interrupts of the same
// (event_id, context_id) recorded in the same tick therefore run back to back with nothing in between that the guest could tell apart; the
// second one only adds a pending event the equeue cap (S420) folds into the first anyway. The tracker lets the caller defer the first one only.
//
// Pure bookkeeping, no GPU: tests/EopCoalesceTests.cpp pins it.

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::EopCoalesce {

enum class Mode : uint8_t { Off, On, Shadow };

// null/""/"0" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
[[nodiscard]] inline Mode ParseMode(const char* text) {
	if (text == nullptr || text[0] == '\0') {
		return Mode::Off;
	}
	if (std::strcmp(text, "1") == 0) {
		return Mode::On;
	}
	if (std::strcmp(text, "shadow") == 0) {
		return Mode::Shadow;
	}
	return Mode::Off;
}

// KYTY_EOP_COALESCE, read once.
[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_EOP_COALESCE"));
	return mode;
}

class Tracker {
public:
	// True when an interrupt of (event_id, context_id) was already deferred in this tick, i.e. the new one can be dropped.
	// Otherwise it is remembered and false is returned: the caller defers it.
	[[nodiscard]] bool AlreadyDeferred(uint64_t tick, int event_id, uint32_t context_id) {
#ifdef KYTY_EOPCOALESCE_MUTANT_DROP_CONTEXT
		context_id = 0;
#endif
#ifndef KYTY_EOPCOALESCE_MUTANT_COALESCE_ACROSS_TICKS
		if (!m_valid || tick != m_tick) {
			m_entries.clear();
			m_tick  = tick;
			m_valid = true;
		}
#endif
		for (const auto& entry: m_entries) {
			if (entry.event_id == event_id && entry.context_id == context_id) {
				return true;
			}
		}
		m_entries.push_back({event_id, context_id});
		return false;
	}

private:
	struct Entry {
		int      event_id;
		uint32_t context_id;
	};
	std::vector<Entry> m_entries;
	uint64_t           m_tick  = 0;
	bool               m_valid = false;
};

} // namespace Libs::Graphics::EopCoalesce

#endif
