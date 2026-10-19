#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESAVEGATE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINESAVEGATE_H_

// S437: when the driver pipeline cache (_PipelineCache/<TITLE>.bin, ~88 MB) needs to be written. Before, every created pipeline made it dirty, also
// when the driver took it from its own cache and added nothing (52-55 saves of the same 88.48 MB per run, each one two serializations under the
// driver's cache lock). Now:
//   - a pipeline whose creation feedback says APPLICATION_PIPELINE_CACHE_HIT does not make the cache dirty (counted as skipped);
//   - no feedback (the driver did not fill it in) counts as a miss: the cache may have grown;
//   - debounce: a dirty cache is written when no pipeline was created for kDebounce, or at the latest kMaxDelay after it became dirty.
// Pure logic, no Vulkan: tests/PipelineSaveGateTests.cpp. Mutants: KYTY_PSGATE_MUTANT_{HIT_DIRTIES, NO_DEBOUNCE, UNKNOWN_IS_HIT}.

#include <chrono>
#include <cstdint>

namespace Libs::Graphics {

enum class PipelineCacheUse : uint8_t { Miss, Hit, Unknown };

// Bits of VkPipelineCreationFeedback::flags (VK_PIPELINE_CREATION_FEEDBACK_*).
inline constexpr uint32_t kFeedbackValidBit    = 0x1;
inline constexpr uint32_t kFeedbackCacheHitBit = 0x2;

[[nodiscard]] inline PipelineCacheUse ClassifyFeedback(uint32_t flags) {
	if ((flags & kFeedbackValidBit) == 0) return PipelineCacheUse::Unknown;
#ifdef KYTY_PSGATE_MUTANT_UNKNOWN_IS_HIT
	return PipelineCacheUse::Hit;
#else
	return (flags & kFeedbackCacheHitBit) != 0 ? PipelineCacheUse::Hit : PipelineCacheUse::Miss;
#endif
}

class PipelineSaveGate {
public:
	using Clock = std::chrono::steady_clock;
	static constexpr auto kDebounce = std::chrono::seconds(2);
	static constexpr auto kMaxDelay = std::chrono::seconds(30);

	// A pipeline was created. Returns true when it made the cache dirty.
	bool NoteCreated(PipelineCacheUse use, Clock::time_point now) {
#ifndef KYTY_PSGATE_MUTANT_HIT_DIRTIES
		if (use == PipelineCacheUse::Hit) {
			m_skipped++;
			return false;
		}
#endif
		if (!m_dirty) m_dirty_since = now;
		m_dirty       = true;
		m_last_change = now;
		return true;
	}

	// The cache should be written now (call before NoteCreated of the next pipeline and at checkpoints).
	[[nodiscard]] bool Due(Clock::time_point now) const {
		if (!m_dirty) return false;
#ifdef KYTY_PSGATE_MUTANT_NO_DEBOUNCE
		return true;
#else
		return now - m_last_change >= kDebounce || now - m_dirty_since >= kMaxDelay;
#endif
	}

	// The write was handed over / failed.
	void Cleared() { m_dirty = false; }
	void Failed(Clock::time_point now) {
		if (!m_dirty) m_dirty_since = now;
		m_dirty       = true;
		m_last_change = now - kDebounce; // retry at the next chance
	}
	// The write was skipped because nothing changed (size equal to the last write).
	void SkippedWrite() { m_skipped++; }

	[[nodiscard]] bool     Dirty() const { return m_dirty; }
	[[nodiscard]] uint64_t Skipped() const { return m_skipped; }
	// Counter delta for FrameTiming: skipped since the last call.
	[[nodiscard]] uint64_t TakeSkippedDelta() {
		const uint64_t delta = m_skipped - m_reported;
		m_reported           = m_skipped;
		return delta;
	}

private:
	bool                m_dirty = false;
	Clock::time_point   m_dirty_since {};
	Clock::time_point   m_last_change {};
	uint64_t            m_skipped = 0, m_reported = 0;
};

} // namespace Libs::Graphics

#endif
