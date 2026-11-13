#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_W2DIAG_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_W2DIAG_H_

// S560 (KYTY_W2_DIAG=1, default 0 = nothing counted): the measurements the K7 W2 gates (deferred Vulkan recording on an emitter thread) need.
// - CommandBuffer::Handle() per call site (file:line of the caller, taken by default arguments): a W2 emitter must either take a site over into its
//   IR or drain before it, so the gate is the number of sites that cover the calls. Logged as cumulative counts every 100 guest flips
//   ([w2-diag] / [w2-diag-site] lines); w2_handle_n / w2_handle_other_n columns (CP thread / any other thread).
// - MasterSemaphore::Wait on the CP thread: every call, those whose tick was not known free at the entry (each needs the emitter to have emitted
//   that tick first), those that blocked in vkWaitSemaphores, by CpWait cause; waits that first submitted the open buffer (a full emitter drain).
// - Vulkan image / buffer destruction outside a tick-ordered deferred operation while the renderer's open command buffer has recorded commands
//   (w2_destroy_unsub_n): with recording deferred such an object may still be referenced by commands not yet emitted.
// With 0 the only cost is one predictable branch per Handle() / Wait / destructor.
//
// SiteTable is header-only (tests/W2DiagTests.cpp). Mutant NO_LINE (the key drops the line, so two sites of one file merge) must fail the test.

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace Libs::Graphics {
class CommandScheduler;
} // namespace Libs::Graphics

namespace Libs::Graphics::W2Diag {

inline bool ParseFlag(const char* value) {
	return value != nullptr && value[0] == '1' && value[1] == '\0';
}

inline bool Enabled() {
	static const bool on = ParseFlag(std::getenv("KYTY_W2_DIAG"));
	return on;
}

// Lock-free open addressing on (file pointer, line). One file name can have several pointers (one per translation unit that inlines the call):
// Snapshot() merges rows by the file's text.
template <size_t N = 2048>
class SiteTable {
	static_assert((N & (N - 1)) == 0);

public:
	static constexpr uint64_t kPtrMask = (uint64_t {1} << 48u) - 1u;

	static uint64_t Key(const char* file, uint32_t line) {
#if defined(KYTY_W2DIAG_MUTANT_NO_LINE)
		(void)line;
		return reinterpret_cast<uintptr_t>(file) & kPtrMask;
#else
		return (reinterpret_cast<uintptr_t>(file) & kPtrMask) | (static_cast<uint64_t>(line & 0xFFFFu) << 48u);
#endif
	}

	// False when the table is full (counted in Dropped()).
	bool Add(const char* file, uint32_t line, bool cp_thread, uint64_t n = 1) {
		const uint64_t key = Key(file, line);
		uint64_t       h   = key * 0x9E3779B97F4A7C15ull;
		size_t         i   = static_cast<size_t>(h >> 40u) & (N - 1);
		for (size_t probe = 0; probe < N; probe++, i = (i + 1) & (N - 1)) {
			auto&    entry = m_entries[i];
			uint64_t k     = entry.key.load(std::memory_order_acquire);
			if (k == 0) {
				uint64_t expected = 0;
				k = entry.key.compare_exchange_strong(expected, key, std::memory_order_acq_rel) ? key : expected;
			}
			if (k == key) {
				(cp_thread ? entry.cp : entry.other).fetch_add(n, std::memory_order_relaxed);
				return true;
			}
		}
		m_dropped.fetch_add(n, std::memory_order_relaxed);
		return false;
	}

	struct Row {
		std::string file;
		uint32_t    line  = 0;
		uint64_t    cp    = 0;
		uint64_t    other = 0;
	};

	// Rows merged by (file text from its last "src/" on, line), most calls first.
	std::vector<Row> Snapshot() const {
		std::vector<Row> rows;
		for (const auto& entry: m_entries) {
			const uint64_t key = entry.key.load(std::memory_order_acquire);
			if (key == 0) continue;
			const auto*    file = reinterpret_cast<const char*>(static_cast<uintptr_t>(key & kPtrMask));
			const uint32_t line = static_cast<uint32_t>(key >> 48u);
			const auto     name = ShortName(file);
			const uint64_t cp   = entry.cp.load(std::memory_order_relaxed);
			const uint64_t oth  = entry.other.load(std::memory_order_relaxed);
			auto it = std::find_if(rows.begin(), rows.end(), [&](const Row& r) { return r.line == line && r.file == name; });
			if (it == rows.end()) {
				rows.push_back({name, line, cp, oth});
			} else {
				it->cp += cp;
				it->other += oth;
			}
		}
		std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
			if (a.cp + a.other != b.cp + b.other) return a.cp + a.other > b.cp + b.other;
			if (a.file != b.file) return a.file < b.file;
			return a.line < b.line;
		});
		return rows;
	}

	uint64_t Dropped() const { return m_dropped.load(std::memory_order_relaxed); }

	static std::string ShortName(const char* file) {
		if (file == nullptr) return "?";
		std::string s(file);
		std::replace(s.begin(), s.end(), '\\', '/');
		const auto at = s.rfind("src/");
		return at == std::string::npos ? s : s.substr(at + 4);
	}

private:
	struct Entry {
		std::atomic<uint64_t> key {0};
		std::atomic<uint64_t> cp {0};
		std::atomic<uint64_t> other {0};
	};
	std::array<Entry, N>  m_entries {};
	std::atomic<uint64_t> m_dropped {0};
};

// Production side (w2Diag.cpp). All are no-ops unless Enabled(); callers test Enabled() first.
void NoteHandle(const char* file, uint32_t line) noexcept;
// MasterSemaphore::Wait: not_free = the tick was not known free at the entry; blocked = it waited in vkWaitSemaphores.
void NoteWait(bool not_free, bool blocked) noexcept;
// A wait that first submitted the open command buffer (FlushAndWait, Finish, Wait of the open tick).
void NoteSubmitWait() noexcept;
// A Vulkan image or buffer is being destroyed.
void NoteDestroy(const char* kind) noexcept;
// The renderer's scheduler (the presenter's has its own and is not watched).
void SetRendererScheduler(const CommandScheduler* scheduler) noexcept;

// S565 (KYTY_CP_EMIT_SITES=1, default 0; independent of KYTY_W2_DIAG): the CommandBuffer::Handle() sites that drained a non-empty KYTY_CP_EMIT
// stream (each costs a wait for the emitter in thread mode). Cumulative counts every 100 guest flips on stderr ([emit-drain-site] lines).
inline bool DrainSitesEnabled() {
	static const bool on = ParseFlag(std::getenv("KYTY_CP_EMIT_SITES"));
	return on;
}
void NoteDrainSite(const char* file, uint32_t line) noexcept;

} // namespace Libs::Graphics::W2Diag

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_W2DIAG_H_
