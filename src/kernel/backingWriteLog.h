#ifndef KERNEL_BACKINGWRITELOG_H
#define KERNEL_BACKINGWRITELOG_H

// S479 (KYTY_CP_PIPE): a log of the writes into the guest's backing (the publish of a GPU->host copy, an upload that goes through the backing). A pipeline worker reads the
// backing while the CP goes on; when a write touched bytes the worker read after the worker started, what the worker computed may be older than the draw: the CP must not use it.
// Each write logs its begin and its end (a sequence number each). A write endangers a reader that started at `seq0` when it ended after seq0 or has not ended yet. The ring keeps
// the newest writes; a query that finds it has wrapped past seq0 answers "yes" (cannot tell).
//
// Mutants (WILL_FAIL, tests/BackingWriteLogTests.cpp): KYTY_BWLOG_MUTANT_IGNORE_RUNNING (a write that has not ended is no danger), KYTY_BWLOG_MUTANT_NO_WRAP_GUARD (a wrapped ring
// answers from what is left), KYTY_BWLOG_MUTANT_ANY_RANGE (the pages are not compared), KYTY_BWLOG_MUTANT_NO_FAST_PATH_GUARD (the lock-free answer forgets the writes in progress).

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>

namespace Libs::LibKernel {

class BackingWriteLog {
public:
	static constexpr size_t kSize = 64; // about 16 writes per flip; a join comes within a millisecond of the start

	struct Token {
		size_t   index = 0;
		uint64_t begin = 0;
	};

	Token Begin(uint64_t address, uint64_t size) {
		const std::lock_guard lock(m_mutex);
		Token token;
		m_running.fetch_add(1, std::memory_order_acq_rel);
		token.index = m_next++ % kSize;
		token.begin = m_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
		m_ring[token.index] = {token.begin, 0, address, size};
		return token;
	}

	void End(const Token& token) {
		const std::lock_guard lock(m_mutex);
		auto&                 entry = m_ring[token.index];
		if (entry.begin == token.begin) entry.end = m_seq.fetch_add(1, std::memory_order_acq_rel) + 1;
		m_running.fetch_sub(1, std::memory_order_acq_rel);
	}

	[[nodiscard]] uint64_t Seq() const { return m_seq.load(std::memory_order_acquire); }

	// Did a write that ended after seq0 (or has not ended) touch one of the 4 KiB pages?
	[[nodiscard]] bool WrittenSince(uint64_t seq0, const uint64_t* pages, size_t count) {
#ifndef KYTY_BWLOG_MUTANT_NO_FAST_PATH_GUARD
		// Nothing began or ended since seq0 and nothing is running: the common case, no lock.
		if (m_seq.load(std::memory_order_acquire) == seq0 && m_running.load(std::memory_order_acquire) == 0) return false;
#else
		if (m_seq.load(std::memory_order_acquire) == seq0) return false;
#endif
		const std::lock_guard lock(m_mutex);
#ifndef KYTY_BWLOG_MUTANT_NO_WRAP_GUARD
		if (m_next > kSize) {
			uint64_t oldest = UINT64_MAX;
			for (const auto& entry: m_ring) {
				if (entry.begin != 0 && entry.begin < oldest) oldest = entry.begin;
			}
			if (oldest > seq0) return true;
		}
#endif
		for (const auto& entry: m_ring) {
			if (entry.begin == 0) continue;
#ifdef KYTY_BWLOG_MUTANT_IGNORE_RUNNING
			if (entry.end == 0 || entry.end <= seq0) continue;
#else
			if (entry.end != 0 && entry.end <= seq0) continue;
#endif
			for (size_t i = 0; i < count; i++) {
#ifdef KYTY_BWLOG_MUTANT_ANY_RANGE
				(void)pages;
				return true;
#else
				if (entry.addr < pages[i] + 4096 && entry.addr + entry.size > pages[i]) return true;
#endif
			}
		}
		return false;
	}

private:
	struct Entry {
		uint64_t begin = 0;
		uint64_t end   = 0;
		uint64_t addr  = 0;
		uint64_t size  = 0;
	};

	std::mutex               m_mutex;
	std::array<Entry, kSize> m_ring {};
	uint64_t                 m_next = 0; // entries written so far
	std::atomic<uint64_t>    m_seq {0};
	std::atomic<int>         m_running {0}; // writes that have begun and not ended
};

} // namespace Libs::LibKernel

#endif
