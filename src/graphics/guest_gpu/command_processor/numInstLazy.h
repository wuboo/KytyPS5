#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_NUMINSTLAZY_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_NUMINSTLAZY_H

// S577 (L1): KYTY_NUMINST_LAZY=0|1|shadow (default 0). An indirect draw leaves the address of its last record's instanceCount as the NUM_INSTANCES
// register (CommandProcessor::m_num_instances_addr). Only a later draw that inherits the count (DRAW_INDEX_2 / DRAW_INDEX_OFFSET_2 / DRAW_INDEX_AUTO with no
// instance count of its own) needs the value; dispatches and indirect draws never read it.
//   0       the old path: the CP front (KYTY_CP_FRONT_THREAD) resolves the address in the snapshot of the first Draw/Dispatch item after the indirect draw.
//           The record was usually written by the GPU, so that read faults and drains the GPU queue on the recorder (S575: ~1/flip in the garden).
//   1       the snapshot carries {address, epoch} instead of a value. The recorder reads the address when it executes the first item that inherits the
//           count (ExecNumInstances), at that item's place in the stream, and keeps the value for the epoch (every later item of the epoch gets the same
//           value, as the old path's one read). NUM_INSTANCES clears the address on the front as before. That is the semantics of the unpiped CP
//           (NumInstances() at the draw); it differs from 0 only when GPU work between the indirect draw and the consumer rewrote the record (the watch
//           below counts that case: numinst_taint_use_n).
//   shadow  the front reads as with 0 and the item uses that value; the recorder reads lazily too at the consumer and counts the difference
//           (numinst_ne_n, must be 0).
// Only the front-thread path (PipeFrontEnqueue) is lazy; the one-thread pipe (PipeEnqueue) and the unpiped CP stay as they are.
//
// Mutants (WILL_FAIL, tests/NumInstLazyTests.cpp): KYTY_NUMINST_MUTANT_STALE_EPOCH (the recorder keeps the value across indirect draws),
// KYTY_NUMINST_MUTANT_REREAD (every consumer reads again), KYTY_NUMINST_MUTANT_PACKET_KEEPS_ADDR (NUM_INSTANCES leaves the address pending),
// KYTY_NUMINST_MUTANT_NO_TAINT (a GPU write over the pending record is not noticed).

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace Libs::Graphics::NumInstLazy {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

// KYTY_NUMINST_LAZY (read once).
inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_NUMINST_LAZY"));
	return mode;
}

inline const char* ModeName(Mode mode) {
	return mode == Mode::On ? "1" : (mode == Mode::Shadow ? "shadow" : "0");
}

// The front's view of the register for the lazy path: the address the last indirect draw left (0: the register holds a value), and the epoch that
// names it (one per indirect draw). The old path keeps its own m_num_instances_addr, which its eager read clears.
struct FrontState {
	uint64_t pending_addr = 0;
	uint64_t epoch        = 0;
	uint64_t skip_epoch   = 0; // the last epoch counted as a read the old path would have done (numinst_lazy_skip_n)

	void OnIndirectDraw(uint64_t addr) {
		pending_addr = addr;
		epoch++;
	}
	void OnPacket() {
#ifndef KYTY_NUMINST_MUTANT_PACKET_KEEPS_ADDR
		pending_addr = 0;
#endif
	}
};

// What a Draw/Dispatch item carries (CpDrawState::num_instances, ni_addr, ni_epoch).
struct Snap {
	uint32_t value   = 1; // the count; with addr != 0 it is the eager value (shadow) or meaningless (1)
	uint64_t addr    = 0;
	uint64_t epoch   = 0;
	bool     skipped = false; // 1: the old path would have read here
};

// The front's snapshot. `eager` is the old resolution (CommandProcessor::NumInstances()); it is called in Off and Shadow only. `live_value` is the
// register as the last NUM_INSTANCES packet set it.
template <class Eager>
Snap TakeSnapshot(Mode mode, FrontState& front, uint32_t live_value, Eager&& eager) {
	Snap s;
	if (mode == Mode::Off) {
		s.value = eager();
		return s;
	}
	s.addr  = front.pending_addr;
	s.epoch = front.pending_addr != 0 ? front.epoch : 0;
	if (mode == Mode::Shadow) {
		s.value = eager();
		return s;
	}
	s.value = live_value;
	if (s.addr != 0 && front.skip_epoch != front.epoch) {
		front.skip_epoch = front.epoch;
		s.skipped        = true;
	}
	return s;
}

// The recorder's memo: the value of one epoch, read once at its first consumer.
struct Resolver {
	uint64_t epoch = 0; // 0: nothing resolved
	uint32_t value = 0;

	// `read(addr)` dereferences the guest record (it may fault and drain the GPU queue, as the unpiped path at the draw). *did_read: 1 when it ran.
	template <class Read>
	uint32_t Resolve(uint64_t addr, uint64_t item_epoch, Read&& read, bool* did_read) {
#if defined(KYTY_NUMINST_MUTANT_STALE_EPOCH)
		const bool hit = epoch != 0;
#elif defined(KYTY_NUMINST_MUTANT_REREAD)
		const bool hit = false;
#else
		const bool hit = epoch != 0 && epoch == item_epoch;
#endif
		if (hit) {
			*did_read = false;
			return value;
		}
		value     = read(addr);
		epoch     = item_epoch;
		*did_read = true;
		return value;
	}
};

// GPU writes recorded over the pending record between the indirect draw and the lazy read (BufferCache::ObtainBuffer with is_written). The old path
// read before any of them, the lazy one after: a write here is the only way the two can differ (CP writes that do not go through the buffer cache
// and guest CPU writes are not seen; shadow compares the values themselves). Armed by the recorder after the indirect draw, read by the recorder
// at the lazy read; the atomics only make a write noted from another queue's thread harmless.
struct Watch {
	std::atomic<uint64_t> addr {0};
	std::atomic<uint64_t> epoch {0};
	std::atomic<bool>     tainted {false};

	void Arm(uint64_t record_addr, uint64_t record_epoch) {
		tainted.store(false, std::memory_order_relaxed);
		epoch.store(record_epoch, std::memory_order_relaxed);
		addr.store(record_addr, std::memory_order_release);
	}
	void Disarm() { addr.store(0, std::memory_order_relaxed); }
	void NoteGpuWrite(uint64_t vaddr, uint64_t size) {
		const uint64_t a = addr.load(std::memory_order_relaxed);
		if (a == 0) return;
#ifndef KYTY_NUMINST_MUTANT_NO_TAINT
		if (vaddr < a + 4u && a < vaddr + size) tainted.store(true, std::memory_order_relaxed);
#else
		(void)vaddr;
		(void)size;
#endif
	}
	[[nodiscard]] bool TaintedFor(uint64_t record_epoch) const {
		return addr.load(std::memory_order_relaxed) != 0 && epoch.load(std::memory_order_relaxed) == record_epoch &&
		       tainted.load(std::memory_order_relaxed);
	}
};

inline Watch g_watch; // constant-initialised: no guard on the ObtainBuffer path

inline Watch& GlobalWatch() {
	return g_watch;
}

// BufferCache::ObtainBuffer (is_written).
inline void NoteGpuWrite(uint64_t vaddr, uint64_t size) {
	GlobalWatch().NoteGpuWrite(vaddr, size);
}

} // namespace Libs::Graphics::NumInstLazy

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_NUMINSTLAZY_H
