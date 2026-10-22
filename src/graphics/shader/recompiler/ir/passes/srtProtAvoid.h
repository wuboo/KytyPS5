#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTPROTAVOID_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTPROTAVOID_H_

// S486: KYTY_SRT_PROT_AVOID=0|1|shadow (default 0 = the walk of S445 bit for bit, none of this runs).
// The ordinary reads of an SRT walk (compiled walker, per-slot / batch entries, interpreter memcpy) load straight from the guest's mapping. A 4 KiB page that holds
// GPU-written bytes is protected, so a load anywhere in it faults and the fault handler waits for the page's download (the GPU, 30-80 ms behind the command
// processor in the landing scene). Landing, flips 4400-5300: all of those faults hit ONE page, the bytes the walk read were NOT GPU-dirty (stale == fresh, dirty=0,
// page_dirty=1): the page is shared with an unrelated GPU-written range (false sharing), the wait buys nothing.
// 1: the faults of a walk teach the table which pages the program reads (HotPages). While one of them is protected the program's walk skips the compiled code and
// reads its ordinary loads through the strict reader (backing when the bytes are current; GPU-dirty or in-flight bytes still drain, exactly as the fault would
// have). A page that has not been seen faulting is read as before (the fault stays the safety net). shadow: the same decision, but every ordinary load reads the
// backing first and then the guest's mapping (whose fault drains as before); the two must be equal (srt_pa_shadow_ne_n stays 0).
// Mutants (tests): KYTY_SRTPA_MUTANT_ALWAYS_AVOID, _NEVER_PROTECTED, _NOTE_WHEN_AVOIDING, _NO_SCOPE_CHECK, _SHARE_PROGRAMS, _NO_CAP.

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR::SrtProtAvoid {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode Parse(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

// Tests: a mode >= 0 replaces the environment's.
inline std::atomic<int>& Override() {
	static std::atomic<int> value {-1};
	return value;
}

inline Mode Current() {
	static const Mode mode = Parse(std::getenv("KYTY_SRT_PROT_AVOID"));
	const int         forced = Override().load(std::memory_order_relaxed);
	return forced >= 0 ? static_cast<Mode>(forced) : mode;
}

constexpr uint64_t kPageSize = 4096;
constexpr uint64_t PageOf(uint64_t address) {
	return address & ~(kPageSize - 1);
}

// The pages a program's walk was seen faulting on, per program hash.
class HotPages {
public:
	static constexpr size_t kPerProgram = 8;
	static constexpr size_t kPrograms   = 4096;

	// Returns true when the page is new for the program.
	bool Note(uint64_t program, uint64_t page) {
#if defined(KYTY_SRTPA_MUTANT_SHARE_PROGRAMS)
		program = 1;
#endif
		std::unique_lock lock(m_mutex);
#if !defined(KYTY_SRTPA_MUTANT_NO_CAP)
		if (m_pages.size() >= kPrograms && m_pages.find(program) == m_pages.end()) return false;
#endif
		auto& entry = m_pages[program];
		for (size_t i = 0; i < entry.count; ++i) {
			if (entry.pages[i] == page) return false;
		}
#if !defined(KYTY_SRTPA_MUTANT_NO_CAP)
		if (entry.count >= kPerProgram) return false;
#endif
		entry.pages[entry.count % kPerProgram] = page;
		++entry.count;
		return true;
	}

	// True when any page of the program satisfies `is_protected`.
	template <class F>
	bool AnyProtected(uint64_t program, F&& is_protected) const {
#if defined(KYTY_SRTPA_MUTANT_SHARE_PROGRAMS)
		program = 1;
#endif
		std::array<uint64_t, kPerProgram> pages {};
		size_t                            count = 0;
		{
			std::shared_lock lock(m_mutex);
			const auto       found = m_pages.find(program);
			if (found == m_pages.end()) return false;
			count = found->second.count < kPerProgram ? found->second.count : kPerProgram;
			for (size_t i = 0; i < count; ++i) pages[i] = found->second.pages[i];
		}
#if defined(KYTY_SRTPA_MUTANT_ALWAYS_AVOID)
		return count != 0;
#endif
		for (size_t i = 0; i < count; ++i) {
#if defined(KYTY_SRTPA_MUTANT_NEVER_PROTECTED)
			continue;
#endif
			if (is_protected(pages[i])) return true;
		}
		return false;
	}

	size_t Programs() const {
		std::shared_lock lock(m_mutex);
		return m_pages.size();
	}
	size_t PagesOf(uint64_t program) const {
		std::shared_lock lock(m_mutex);
		const auto       found = m_pages.find(program);
		return found == m_pages.end() ? 0 : (found->second.count < kPerProgram ? found->second.count : kPerProgram);
	}
	void Clear() {
		std::unique_lock lock(m_mutex);
		m_pages.clear();
	}

private:
	struct Entry {
		std::array<uint64_t, kPerProgram> pages {};
		size_t                            count = 0;
	};
	mutable std::shared_mutex             m_mutex;
	std::unordered_map<uint64_t, Entry>   m_pages;
};

inline HotPages& Global() {
	static HotPages pages;
	return pages;
}

// Is the 4 KiB page protected for a GPU-written range right now? Set by the pipeline cache (the CP thread asks; anything else gets false).
using PageProtectedHook = bool (*)(uint64_t page);
inline std::atomic<PageProtectedHook>& ProtectedHook() {
	static std::atomic<PageProtectedHook> hook {nullptr};
	return hook;
}

// The walk the thread is running: its program, and whether it avoids the faults.
struct Walk {
	uint64_t program = 0;
	bool     avoid   = false;
};
inline Walk& CurrentWalk() {
	static thread_local Walk walk;
	return walk;
}

class WalkScope {
public:
	WalkScope(uint64_t program, bool avoid): m_saved(CurrentWalk()) { CurrentWalk() = {program, avoid}; }
	~WalkScope() { CurrentWalk() = m_saved; }
	WalkScope(const WalkScope&)            = delete;
	WalkScope& operator=(const WalkScope&) = delete;

private:
	Walk m_saved;
};

inline bool Avoiding() {
	return CurrentWalk().avoid;
}

enum class Cell : uint8_t { WalkAvoided, ReadBacking, ReadFallback, ShadowNe, FaultNoted, Count };
inline std::array<std::atomic<uint64_t>, static_cast<size_t>(Cell::Count)>& Cells() {
	static std::array<std::atomic<uint64_t>, static_cast<size_t>(Cell::Count)> cells {};
	return cells;
}
inline void Note(Cell cell, uint64_t n = 1) {
	Cells()[static_cast<size_t>(cell)].fetch_add(n, std::memory_order_relaxed);
}
inline std::array<uint64_t, static_cast<size_t>(Cell::Count)> Drain() {
	std::array<uint64_t, static_cast<size_t>(Cell::Count)> out {};
	for (size_t i = 0; i < out.size(); ++i) out[i] = Cells()[i].exchange(0, std::memory_order_relaxed);
	return out;
}

// The decision of a walk: the program has a hot page that is protected now.
inline bool Decide(uint64_t program) {
	if (Current() == Mode::Off || program == 0) return false;
	const auto hook = ProtectedHook().load(std::memory_order_acquire);
	if (hook == nullptr) return false;
	return Global().AnyProtected(program, hook);
}

// A read fault of the command processor at `address` while a walk is running (and not already avoiding): the page is hot for the walk's program.
inline void NoteFault(uint64_t address) {
	if (Current() == Mode::Off) return;
	const auto& walk = CurrentWalk();
#if !defined(KYTY_SRTPA_MUTANT_NO_SCOPE_CHECK)
	if (walk.program == 0) return;
#endif
#if !defined(KYTY_SRTPA_MUTANT_NOTE_WHEN_AVOIDING)
	if (walk.avoid) return;
#endif
	if (Global().Note(walk.program, PageOf(address))) Note(Cell::FaultNoted);
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::SrtProtAvoid

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_SRTPROTAVOID_H_
