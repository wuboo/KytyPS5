#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_WALKNOFAULT_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_IR_PASSES_WALKNOFAULT_H_

// S594 (KYTY_WALK_NOFAULT=0|1, unset = 0): a compiled SRT walker load (`mov eax, [r10]`) on a page that is protected because the GPU
// wrote other bytes faults, and with KYTY_RB_CLEAN_WALK=1 the handler completes it from the backing (S507). In the garden that is
// ~1000-2600 faults per flip on the command processor thread, each a full exception round trip (kernel, dispatcher, handler, NtContinue).
// 1: the walker is compiled with a check before every guest load: when the load's 4 KiB page is in the hot page table (a page whose
// walker load the handler served from the backing before), it first asks the reader hook. The hook answers with exactly the clean-walk
// decision (bytes not GPU-dirty, not in flight, the tracker not busy: the backing's dword); anything else, and every load on a pipeline
// worker thread, goes to the ordinary `mov` at the same code offset, so the fault path (clean walk, K-rf patch / superset, drain) is
// unchanged. 0: the walker code is the one of 922ef725 byte for byte (no check is emitted).
// The switch is not codegen: the served dword is the one the fault handler would have served, SPIR-V does not change.
// Mutants (tests): KYTY_WALKNF_MUTANT_WORKER_SERVED (a worker's load is served), KYTY_WALKNF_MUTANT_IGNORE_REFUSAL (a refused read
// still skips the ordinary load). Test build KYTY_WALKNF_EVERY_LOAD: the table check is not emitted, every load asks the hook (the
// whole walker suite runs through the call path).

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::ShaderRecompiler::IR::WalkNoFault {

enum class Mode : uint8_t { Off, On };

inline Mode Parse(const char* text) {
	if (text != nullptr && std::strcmp(text, "1") == 0) return Mode::On;
	return Mode::Off;
}

inline Mode EnvMode() {
	static const Mode mode = Parse(std::getenv("KYTY_WALK_NOFAULT"));
	return mode;
}

// Tests: a value >= 0 replaces the environment's (read when a walker is compiled).
inline std::atomic<int>& Override() {
	static std::atomic<int> value {-1};
	return value;
}

inline Mode Current() {
	const int forced = Override().load(std::memory_order_relaxed);
	return forced >= 0 ? static_cast<Mode>(forced) : EnvMode();
}

constexpr uint32_t kPageShift = 12;
constexpr size_t   kSlots     = 256;
static_assert((kSlots & (kSlots - 1)) == 0);
// No 48-bit address has this page number, so an empty slot never matches.
constexpr uint64_t kEmpty = ~uint64_t {0};

constexpr uint64_t PageOf(uint64_t address) {
	return address >> kPageShift;
}
constexpr size_t SlotOf(uint64_t page) {
	return static_cast<size_t>(page & (kSlots - 1));
}

// Direct-mapped table of hot page numbers. The walker reads a slot with a plain 64-bit load, so a slot is a lock-free 64-bit atomic.
struct HotPages {
	std::array<std::atomic<uint64_t>, kSlots> pages;
	HotPages() {
		for (auto& page: pages) page.store(kEmpty, std::memory_order_relaxed);
	}
};
static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t));
static_assert(std::atomic<uint64_t>::is_always_lock_free);

inline HotPages& Hot() {
	static HotPages table;
	return table;
}

// A page whose walker load was served from the backing. Returns true when the slot changed.
inline bool NoteHot(uint64_t address) {
	const auto page = PageOf(address);
	auto&      slot = Hot().pages[SlotOf(page)];
	return slot.exchange(page, std::memory_order_relaxed) != page;
}

inline bool IsHot(uint64_t address) {
	const auto page = PageOf(address);
	return Hot().pages[SlotOf(page)].load(std::memory_order_relaxed) == page;
}

inline void ClearHot() {
	for (auto& page: Hot().pages) page.store(kEmpty, std::memory_order_relaxed);
}

// The answer of the reader: kServed | dword, or 0 (take the ordinary load).
constexpr uint64_t kServed = uint64_t {1} << 32;
using Reader               = uint64_t (*)(uint64_t address);

inline std::atomic<Reader>& ReaderHook() {
	static std::atomic<Reader> hook {nullptr};
	return hook;
}

constexpr uint64_t Served(uint32_t dword) {
	return kServed | dword;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR::WalkNoFault

#endif
