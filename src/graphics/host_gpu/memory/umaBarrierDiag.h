#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMABARRIERDIAG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMABARRIERDIAG_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

// DIAG (KYTY_DIAG_UMA_BARRIERS): with every buffer a window of the imported arena, a shader write
// and a later use of the same bytes in the same submit are ordered only by the Vulkan barriers the
// renderer records. MoltenVK's hazard tracking can hide a missing one (153: untracked = 0 flips).
// This finds the uses that follow a write of the same bytes in the same submit with no memory
// barrier recorded in between.
namespace Libs::Graphics::UmaBarrierDiag {

enum class Mode : uint8_t { Off = 0, Count = 1, Insert = 2 };

[[nodiscard]] inline Mode ModeFromEnv() {
	const char* value = std::getenv("KYTY_DIAG_UMA_BARRIERS");
	if (value == nullptr || value[0] == '\0' || value[0] == '0') {
		return Mode::Off;
	}
	return value[0] == '2' ? Mode::Insert : Mode::Count;
}

// Counts the generic memory barriers the renderer records (shader write / access / hazard
// barriers, buffer upload copies). Any thread.
inline std::atomic<uint64_t> g_barrier_epoch {0};
inline void                  NoteBarrier() { g_barrier_epoch.fetch_add(1, std::memory_order_relaxed); }
[[nodiscard]] inline uint64_t Epoch() { return g_barrier_epoch.load(std::memory_order_relaxed); }

enum class Verdict : uint8_t {
	None,    // no earlier writer of these bytes in this submit
	Covered, // there is one, and a barrier was recorded after it
	Missing, // there is one, and nothing was recorded after it
};

// Writers of the submit being recorded. GPU thread only.
class SameSubmitTracker {
public:
	static constexpr size_t MaxEntries = 4096;

	// Verdict for a use of [vaddr, vaddr + size) now, `epoch` = the barrier epoch at this moment.
	[[nodiscard]] Verdict Check(uint64_t tick, uint64_t vaddr, uint64_t size, uint64_t epoch) {
		NewTick(tick);
		const auto end   = vaddr + size;
		bool       found = false;
		for (const auto& entry: m_entries) {
			if (entry.start >= end || vaddr >= entry.end) {
				continue;
			}
			found = true;
#ifdef KYTY_UMA_MUTANT_BARRIER_EPOCH_IGNORED
			(void)epoch;
#else
			if (entry.epoch == epoch) {
				return Verdict::Missing;
			}
#endif
		}
		return found ? Verdict::Covered : Verdict::None;
	}

	void NoteWrite(uint64_t tick, uint64_t vaddr, uint64_t size, uint64_t epoch) {
		NewTick(tick);
		if (size == 0) {
			return;
		}
		if (m_entries.size() >= MaxEntries) {
			m_entries.erase(m_entries.begin(), m_entries.begin() + MaxEntries / 2);
			++m_dropped;
		}
		m_entries.push_back({vaddr, vaddr + size, epoch});
	}

	[[nodiscard]] size_t   Size() const { return m_entries.size(); }
	[[nodiscard]] uint64_t Dropped() const { return m_dropped; }

private:
	struct Entry {
		uint64_t start;
		uint64_t end;
		uint64_t epoch;
	};
	void NewTick(uint64_t tick) {
		if (tick != m_tick) {
			m_tick = tick;
			m_entries.clear();
		}
	}

	uint64_t           m_tick    = 0;
	uint64_t           m_dropped = 0;
	std::vector<Entry> m_entries;
};

} // namespace Libs::Graphics::UmaBarrierDiag

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_MEMORY_UMABARRIERDIAG_H_
