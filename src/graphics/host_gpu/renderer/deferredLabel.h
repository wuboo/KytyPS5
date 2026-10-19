#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>

// KYTY_OPT_ON=eop_on_completion: EVENT_WRITE_EOP / RELEASE_MEM labels reach guest memory when the GPU
// work recorded before them has finished, not when the packet is parsed. The write runs as a
// priority operation of the scheduler: it fires after the tick of the current submit completed and
// in FIFO order, so labels of one queue (and the interrupts queued right after them) keep their order.
namespace Libs::Graphics::Sync {

using GuestLabelWriter = void (*)(uint64_t address, const void* data, uint64_t size);

// Labels that were parsed but whose tick has not finished (KYTY_DIAG_EOP_EARLY=1). A guest-visible
// read of such an address is a read of a label that, with eop_on_completion off, is already set.
class PendingLabels {
public:
	void Add(uint64_t address) {
		std::lock_guard lock(m_mutex);
		m_pending[address]++;
	}
	void Remove(uint64_t address) {
		std::lock_guard lock(m_mutex);
		auto            it = m_pending.find(address);
		if (it != m_pending.end() && --it->second == 0) {
			m_pending.erase(it);
		}
	}
	// Returns true (and counts) when `address` has a label still waiting for its tick.
	bool NoteRead(uint64_t address) {
		std::lock_guard lock(m_mutex);
		if (m_pending.find(address) == m_pending.end()) {
			return false;
		}
		m_early_reads.fetch_add(1, std::memory_order_relaxed);
		return true;
	}
	[[nodiscard]] uint64_t EarlyReads() const { return m_early_reads.load(std::memory_order_relaxed); }

private:
	std::mutex                             m_mutex;
	std::unordered_map<uint64_t, uint32_t> m_pending;
	std::atomic<uint64_t>                  m_early_reads {0};
};

// `Scheduler` needs DeferPriorityOperation(callable); the real CommandScheduler and the test fake do.
template <class Scheduler>
void DeferGuestLabelWrite(Scheduler& scheduler, GuestLabelWriter writer, PendingLabels* pending,
                          uint64_t address, uint64_t value, uint32_t size) {
	if (pending != nullptr) {
		pending->Add(address);
	}
	scheduler.DeferPriorityOperation([writer, pending, address, value, size] {
		writer(address, &value, size);
		if (pending != nullptr) {
			pending->Remove(address);
		}
	});
}

} // namespace Libs::Graphics::Sync
