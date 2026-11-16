#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSPLIT_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSPLIT_H_

// S612 (KYTY_PASS_SPLIT=0|1|shadow|classify, default 0): stage E1 of the parallel resolution of the recorder (artifacts/S611/DESIGN-PARALLEL-RECORD.md
// §5). On the one recorder thread a draw is split into a phase R, the reservation (everything that mutates state shared with the next draws, in stream
// order) and a phase P (the work a resolver thread could take later: lookups and the emission of the draw from what R reserved). Nothing runs on
// another thread yet; this stage measures how much of a draw P is and what the seam costs.
//
// What moves (renderDraw.cpp ExecutePreparedDraw, the tail from the pipeline lookup on, RunTail below; off is the original order):
//   - R: the colour/depth attachment transitions (AcquireRenderTargets), the eager split, the transitions and GDS barriers of the bound images
//     (CommitBindingsTransit, the first half of CommitBindings) and every decision on the executor's deferred-write state (S511 m_pending_writes:
//     the flush before a draw that reads a pending range, the immediate barrier, the record of this draw's pending writes); the plan keeps them;
//   - P: the pipeline lookup, the vertex/index binds, the descriptor writes and push data (CommitBindingsBuild), the push constants, the dynamic
//     state, the barriers and pass ends the plan asks for, BeginRendering, the bind of the pipeline, the draw and the deferred barrier.
//   The emission differs from off only in that the vertex-buffer bind follows the binding transitions (commands that commute: a bind is state of the
//   command buffer, a barrier or a pass end does not touch it); the state each draw is recorded with is the same (tests/PassSplitTests.cpp).
//   A draw that takes the image heap or the GPU mesh path (S472) keeps the off order (serial: its heap commit or prepare compute needs the pipeline
//   or the ring slot before the transitions).
//
// Measurement (frames.csv pass_split_* columns, every mode but off; on / shadow time one draw in 8 and count it 8 times, classify times every
// draw): the draw's time is cut into sections by laps (rdtsc); each lap closes the
// current section and files its time as clean / soft / hard by the mutations noted on the thread since the section began (Note below: hard = a
// buffer or image created, an upload, a GPU write marked, a clear, a transition with a barrier, a submit, a GPU wait, a readback, a guest write;
// soft = a view created, a stream allocation, a descriptor set from the shared heap). pass_split_r_ns / pass_split_p_ns are the two phases;
// pass_split_p_hard_n counts the P sections with a hard mutation (must be 0: P must not mutate). With classify the R phase is cut further (head,
// job, rt, tex, find, img, buf, vbib, acq, cbr): its clean sections are R work that mutated nothing for this draw, the pool a later stage could also
// move to P when R can tell a hit cheaply (the upper bound of P).
// shadow (split + checks): the pipeline is also looked up at its off position and must be the one P finds; the deferred-write decisions are taken
// again at their off position (after the dynamic state) on a copy of the state R decided on and must match; differences count pass_split_ne_n.
//
// Mutants (WILL_FAIL, tests/PassSplitTests.cpp): KYTY_PASSSPLIT_MUTANT_DECIDE_EARLY (R decides before the binding transitions),
// KYTY_PASSSPLIT_MUTANT_NO_NOTE_END (P drops the pass end of a full pending-write list), KYTY_PASSSPLIT_MUTANT_BUILD_BEFORE_TRANSIT (the descriptor
// writes are built before the transitions), KYTY_PASSSPLIT_MUTANT_DRAW_BEFORE_DYNAMIC, KYTY_PASSSPLIT_MUTANT_NO_EMIT_BEFORE (P drops the planned
// pass end / barrier before the draw), KYTY_PASSSPLIT_MUTANT_SERIAL_IGNORED (an image-heap draw is split), KYTY_PASSSPLIT_MUTANT_SOFT_AS_CLEAN,
// KYTY_PASSSPLIT_MUTANT_P_AS_R (the P section is filed as R), KYTY_PASSSPLIT_MUTANT_NOTE_OUTSIDE (mutations outside a draw are counted),
// KYTY_PASSSPLIT_MUTANT_NESTED_SCOPE (a nested draw scope restarts the outer one).

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_M_X64) || defined(__x86_64__)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
#define KYTY_PASSSPLIT_TSC 1
#endif

namespace Libs::Graphics::PassSplit {

enum class Mode : uint8_t { Off = 0, On = 1, Shadow = 2, Classify = 3 };

inline Mode ParseMode(const char* value) {
	if (value == nullptr) return Mode::Off;
	if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) return Mode::On;
	if (std::strcmp(value, "shadow") == 0) return Mode::Shadow;
	if (std::strcmp(value, "classify") == 0) return Mode::Classify;
	return Mode::Off;
}

inline const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::On: return "on";
		case Mode::Shadow: return "shadow";
		case Mode::Classify: return "classify";
		default: return "off";
	}
}

// The first call (the first draw) logs `Environment: KYTY_PASS_SPLIT=<value> mode=<off|on|shadow|classify>`.
inline Mode EnvMode() {
	static const Mode mode = [] {
		const char* value = std::getenv("KYTY_PASS_SPLIT");
		const Mode  m     = ParseMode(value);
		std::fprintf(stderr, "Environment: KYTY_PASS_SPLIT=%s mode=%s\n", value != nullptr ? value : "", ModeName(m));
		std::fflush(stderr);
		return m;
	}();
	return mode;
}

// A draw that keeps the off order in a split mode (see the top).
inline bool NeedsSerial(bool mesh_gpu, bool image_heap) {
#if defined(KYTY_PASSSPLIT_MUTANT_SERIAL_IGNORED)
	(void)image_heap;
	return mesh_gpu;
#else
	return mesh_gpu || image_heap;
#endif
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Mutations of shared state (Note) and the sections of a draw (Lap).

enum class Mut : uint8_t {
	BufCreate,  // BufferCache::CreateBuffer
	Upload,     // a buffer upload (SynchronizeBuffer with copies, the stream path of ObtainBuffer)
	GpuWrite,   // a GPU write marked (ObtainBuffer is_written, a storage image)
	ImgCreate,  // an image inserted, registered, unregistered or deleted
	ImgUpload,  // an image initialised from guest memory
	ImgCopy,    // an image copied into another (overlap resolution)
	Clear,      // a colour clear materialised, an image cleared
	Transit,    // an image transition that records a barrier
	Submit,     // a command buffer ended (submit)
	Wait,       // a wait for the GPU
	Readback,   // a download of GPU data to the guest
	GuestWrite, // metadata bytes written into guest memory
	View,       // soft: an image view created
	Stream,     // soft: an allocation from the stream buffer
	Heap,       // soft: a descriptor set from the shared heap
	Count
};
inline constexpr size_t kMutCount = static_cast<size_t>(Mut::Count);
inline constexpr std::array<bool, kMutCount> kMutHard = {true, true, true, true, true, true, true, true, true, true, true, true, false, false, false};
inline constexpr std::array<const char*, kMutCount> kMutNames = {"buf_create", "upload", "gpu_write", "img_create", "img_upload",
                                                                  "img_copy",   "clear",  "transit",   "submit",     "wait",
                                                                  "readback",   "guest_write", "view", "stream",     "heap"};

enum class Sec : uint8_t { Head, Job, Rt, Tex, Find, Img, Buf, Vbib, Acq, Cbr, P, Tail, Count };
inline constexpr size_t kSecCount = static_cast<size_t>(Sec::Count);
inline constexpr std::array<const char*, kSecCount> kSecNames = {"head", "job", "rt", "tex", "find", "img", "buf", "vbib", "acq", "cbr", "p", "tail"};

enum Bucket : uint8_t { Clean = 0, Soft = 1, Hard = 2 };

// Clocks: rdtsc on x86 (ticks, converted at the flush), steady_clock ns elsewhere.
inline uint64_t Ticks() {
#if defined(KYTY_PASSSPLIT_TSC)
	return __rdtsc();
#else
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
#endif
}

// ns per tick, calibrated lazily against steady_clock without sleeping (as CpEmit::NsPerTick): 0 until 20 ms have passed since the first call.
inline double NsPerTick() {
#if defined(KYTY_PASSSPLIT_TSC)
	static std::atomic<uint64_t> t0 {0}, n0 {0};
	static std::atomic<double>   ratio {0.0};
	if (const double r = ratio.load(std::memory_order_relaxed); r != 0.0) return r;
	const uint64_t t = Ticks();
	const uint64_t n = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
	uint64_t       expected = 0;
	if (n0.compare_exchange_strong(expected, n)) {
		t0.store(t);
		return 0.0;
	}
	const uint64_t base_t = t0.load();
	if (n - expected < 20'000'000 || base_t == 0 || t <= base_t) return 0.0;
	const double r = static_cast<double>(n - expected) / static_cast<double>(t - base_t);
	ratio.store(r, std::memory_order_relaxed);
	return r;
#else
	return 1.0;
#endif
}

// The clock of the laps (a test replaces it).
inline uint64_t (*g_ticks)() = &Ticks;
// on / shadow time one draw in g_sample_period (each timed draw counts that many times: the columns estimate every draw at an eighth of the
// clock reads); classify times every draw.
inline uint32_t g_sample_period = 8;

struct Ledger {
	std::array<std::array<uint64_t, 3>, kSecCount> ticks {}; // [section][bucket]
	std::array<uint64_t, kMutCount>                mut {};   // mutations noted inside draws
	uint64_t draws = 0, p_draws = 0, serial = 0, nop = 0, laps = 0, p_soft = 0, p_hard = 0, shadow = 0, ne = 0;
};

struct Thread {
	uint8_t  level     = 0; // 0: nothing noted (off, or before the first draw of a split mode), 1: split, 2: classify
	bool     draw      = false; // a timed draw is open
	bool     reached_p = false;
	Sec      sec       = Sec::Head;
	uint32_t sample    = 0; // draws seen (the sampling of on / shadow)
	uint64_t weight    = 1; // what a timed draw counts for
	uint64_t t0 = 0, hard = 0, soft = 0, h0 = 0, s0 = 0;
	Ledger   led;
};
inline constinit thread_local Thread t_thread {};

inline void Note(Mut kind) {
	auto& t = t_thread;
	if (t.level == 0) return;
	if (kMutHard[static_cast<size_t>(kind)]) {
		t.hard++;
	} else {
		t.soft++;
	}
#if defined(KYTY_PASSSPLIT_MUTANT_NOTE_OUTSIDE)
	t.led.mut[static_cast<size_t>(kind)]++;
#else
	if (t.draw) t.led.mut[static_cast<size_t>(kind)] += t.weight;
#endif
}

inline void LapNow(Thread& t, Sec next) {
	const uint64_t now = g_ticks();
	const uint64_t dt  = now - t.t0;
#if defined(KYTY_PASSSPLIT_MUTANT_SOFT_AS_CLEAN)
	const Bucket b = t.hard != t.h0 ? Hard : Clean;
#else
	const Bucket b = t.hard != t.h0 ? Hard : (t.soft != t.s0 ? Soft : Clean);
#endif
#if defined(KYTY_PASSSPLIT_MUTANT_P_AS_R)
	const Sec filed = t.sec == Sec::P ? Sec::Cbr : t.sec;
#else
	const Sec filed = t.sec;
#endif
	t.led.ticks[static_cast<size_t>(filed)][b] += dt * t.weight;
	if (t.sec == Sec::P) {
		if (b == Hard) t.led.p_hard += t.weight;
		if (b == Soft) t.led.p_soft += t.weight;
	}
	t.sec = next;
	t.t0  = now;
	t.h0  = t.hard;
	t.s0  = t.soft;
	t.led.laps += t.weight;
}

// A lap of the split (P begins / ends): every split mode.
inline void Lap(Sec next) {
	auto& t = t_thread;
	if (t.draw) LapNow(t, next);
}
// A lap of the classification of R: classify only.
inline void LapC(Sec next) {
	auto& t = t_thread;
	if (t.draw && t.level >= 2) LapNow(t, next);
}
inline void EnterP() {
	auto& t = t_thread;
	if (!t.draw) return;
	LapNow(t, Sec::P);
	t.reached_p = true;
	t.led.p_draws += t.weight;
}
inline void NoteSerial() {
	auto& t = t_thread;
	if (t.draw) t.led.serial += t.weight;
}
inline void NoteShadow(bool ne) {
	auto& t = t_thread;
	t.led.shadow++;
	if (ne) t.led.ne++;
}

// One per draw (renderDraw.cpp DrawIndex / DrawAuto): opens the head section, closes the current one at the end. A nested scope does nothing.
// In on / shadow only every g_sample_period-th draw is timed (the others are split all the same, they only file nothing).
class DrawScope {
public:
	DrawScope() noexcept: DrawScope(EnvMode()) {}
	explicit DrawScope(Mode mode) noexcept {
		if (mode == Mode::Off) return;
		auto& t = t_thread;
#if !defined(KYTY_PASSSPLIT_MUTANT_NESTED_SCOPE)
		if (t.draw) return;
#endif
		t.level = mode == Mode::Classify ? 2 : 1;
		if (t.level == 1) {
			const uint32_t period = g_sample_period != 0 ? g_sample_period : 1;
			if (++t.sample % period != 0) return;
			t.weight = period;
		} else {
			t.weight = 1;
		}
		t.draw      = true;
		t.reached_p = false;
		t.sec       = Sec::Head;
		t.t0        = g_ticks();
		t.h0        = t.hard;
		t.s0        = t.soft;
		m_mine      = true;
	}
	~DrawScope() {
		if (!m_mine) return;
		auto& t = t_thread;
		LapNow(t, Sec::Head);
		t.led.laps -= t.weight; // the closing lap is not a lap between two sections
		t.draw = false;
		t.led.draws += t.weight;
		if (!t.reached_p) t.led.nop += t.weight;
	}
	DrawScope(const DrawScope&)            = delete;
	DrawScope& operator=(const DrawScope&) = delete;

private:
	bool m_mine = false;
};

// ---------------------------------------------------------------------------------------------------------------------------------------
// The frames.csv columns, in this order (frameTiming.h PassSplitDrawN.. ; the flush adds slot i to the i-th of them).

enum Slot : uint32_t {
	SlotDrawN, SlotPDrawN, SlotSerialN, SlotNopN, SlotLapN, SlotRNs, SlotPNs, SlotPSoftN, SlotPHardN, SlotShadowN, SlotNeN,
	SlotSecFirst, // kSecCount x {clean, soft, hard} ns
	SlotMutFirst = SlotSecFirst + static_cast<uint32_t>(kSecCount) * 3u,
	SlotCount    = SlotMutFirst + static_cast<uint32_t>(kMutCount)
};

// Moves the thread's ledger to sink(slot, value) (non-zero values only). Times need a calibrated clock: with ns_per_tick == 0 nothing moves yet.
template <class Sink>
void Flush(double ns_per_tick, Sink&& sink) {
	auto& led = t_thread.led;
	if (ns_per_tick == 0.0) return;
	const auto ns = [ns_per_tick](uint64_t ticks) { return static_cast<uint64_t>(static_cast<double>(ticks) * ns_per_tick); };
	const auto put = [&](uint32_t slot, uint64_t value) {
		if (value != 0) sink(slot, value);
	};
	put(SlotDrawN, led.draws);
	put(SlotPDrawN, led.p_draws);
	put(SlotSerialN, led.serial);
	put(SlotNopN, led.nop);
	put(SlotLapN, led.laps);
	uint64_t r = 0, p = 0;
	for (size_t s = 0; s < kSecCount; s++) {
		for (size_t b = 0; b < 3; b++) {
			const uint64_t v = ns(led.ticks[s][b]);
			put(SlotSecFirst + static_cast<uint32_t>(s * 3 + b), v);
			(s == static_cast<size_t>(Sec::P) ? p : r) += v;
		}
	}
	put(SlotRNs, r);
	put(SlotPNs, p);
	put(SlotPSoftN, led.p_soft);
	put(SlotPHardN, led.p_hard);
	put(SlotShadowN, led.shadow);
	put(SlotNeN, led.ne);
	for (size_t k = 0; k < kMutCount; k++) put(SlotMutFirst + static_cast<uint32_t>(k), led.mut[k]);
	led = {};
}

// Flush with the calibrated clock, on a thread that has seen a draw of a split mode (nothing to do otherwise).
template <class Sink>
void FlushOn(Sink&& sink) {
	if (t_thread.level == 0) return;
	Flush(NsPerTick(), sink);
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// The tail of a draw (the pipeline lookup on). Ops is called with one tag per step; Prologue returns false when nothing of the draw is recorded.

namespace Op {
struct Pipeline {};        // the pipeline lookup
struct AcquireTargets {};  // AcquireRenderTargets (attachment transitions; the pass state)
struct Prologue {};        // mesh task params, the eager split at a pass change, the mesh ring slot, the IR handle, the mesh prepare compute
struct CommitVertex {};    // bindVertexBuffers
struct BindingsWhole {};   // CommitBindings as one (off and serial)
struct BindingsTransit {}; // its transitions and GDS barriers, the layouts kept in the bindings
struct BindingsBuild {};   // its descriptor writes and push data
struct CommitIndex {};     // bindIndexBuffer
struct PushDraw {};        // the draw's push constants and user clip planes
struct Dynamic {};         // the dynamic state
struct DecideBefore {};    // the deferred-write decisions before the draw (plan: flush for a pending read, immediate barrier)
struct EmitBefore {};      // what DecideBefore planned: pass ends, the write hazard barrier
struct Draw {};            // BeginRendering, the pipeline bind, the occlusion query, the draw
struct AfterBarrier {};    // the barrier after an immediate write, or the deferred barrier
struct DecideAfter {};     // the record of this draw's pending writes (plan: the list is full)
struct AfterEnd {};        // the pass end of a full list
struct ShadowEarly {};     // shadow: the pipeline at its off position, a copy of the deferred-write state
struct ShadowCheck {};     // shadow: the decisions again at their off position
} // namespace Op

template <class Ops>
bool RunTail(Ops& ops, bool split, bool shadow) {
	if (!split) {
		ops(Op::Pipeline {});
		ops(Op::AcquireTargets {});
		if (!ops(Op::Prologue {})) return false;
		ops(Op::CommitVertex {});
		ops(Op::BindingsWhole {});
		ops(Op::CommitIndex {});
		ops(Op::PushDraw {});
		ops(Op::Dynamic {});
		ops(Op::DecideBefore {});
		ops(Op::EmitBefore {});
		ops(Op::Draw {});
		ops(Op::AfterBarrier {});
		ops(Op::DecideAfter {});
		ops(Op::AfterEnd {});
		return true;
	}
	// R
	if (shadow) ops(Op::ShadowEarly {});
	LapC(Sec::Acq);
	ops(Op::AcquireTargets {});
	if (!ops(Op::Prologue {})) return false;
	LapC(Sec::Cbr);
#if defined(KYTY_PASSSPLIT_MUTANT_DECIDE_EARLY)
	ops(Op::DecideBefore {});
	ops(Op::BindingsTransit {});
#elif defined(KYTY_PASSSPLIT_MUTANT_BUILD_BEFORE_TRANSIT)
	ops(Op::BindingsBuild {});
	ops(Op::BindingsTransit {});
	ops(Op::DecideBefore {});
#else
	ops(Op::BindingsTransit {});
	ops(Op::DecideBefore {});
#endif
	ops(Op::DecideAfter {});
	// P
	EnterP();
	ops(Op::Pipeline {});
	ops(Op::CommitVertex {});
#if !defined(KYTY_PASSSPLIT_MUTANT_BUILD_BEFORE_TRANSIT)
	ops(Op::BindingsBuild {});
#endif
	ops(Op::CommitIndex {});
	ops(Op::PushDraw {});
#if defined(KYTY_PASSSPLIT_MUTANT_DRAW_BEFORE_DYNAMIC)
	if (shadow) ops(Op::ShadowCheck {});
	ops(Op::EmitBefore {});
	ops(Op::Draw {});
	ops(Op::Dynamic {});
#else
	ops(Op::Dynamic {});
	if (shadow) ops(Op::ShadowCheck {});
#if !defined(KYTY_PASSSPLIT_MUTANT_NO_EMIT_BEFORE)
	ops(Op::EmitBefore {});
#endif
	ops(Op::Draw {});
#endif
	ops(Op::AfterBarrier {});
#if !defined(KYTY_PASSSPLIT_MUTANT_NO_NOTE_END)
	ops(Op::AfterEnd {});
#endif
	Lap(Sec::Tail);
	return true;
}

// The tags are dispatched to an overload set of lambdas.
template <class... F>
struct Overload: F... {
	using F::operator()...;
};
template <class... F>
Overload(F...) -> Overload<F...>;

} // namespace Libs::Graphics::PassSplit

#endif /* EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PASSSPLIT_H_ */
