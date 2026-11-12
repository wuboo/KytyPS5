#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTUREPREDECODE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTUREPREDECODE_H_

// S558 (KYTY_CP_PREDECODE=0|1|shadow, K7 W1a step 1): the pipeline worker that answered a draw's GfpJob also decodes the draw's T#s (the part of
// ResolveTexture before TextureCache::FindImage) from the job's own snapshot (job.out.vs_res / ps_res) into the job (GfpJob::pre). The recorder uses a
// stage of it only when JoinJob accepted the job and the stage it binds runs on that same snapshot (same module, same ResourceSnapshot object), and then
// only calls FindImage. Synchronisation: none new. The worker writes the job before its release store of state = Done; the recorder reads it after the
// acquire in JoinJob/WaitDone; the job lives in its FIFO item until the item has run.
// 0: nothing changes. 1: use. shadow: the worker decodes, the recorder resolves as before and compares (predecode_ne_n), the result used is the recorder's.
//
// Header only and templated on the slot and value types, so tests/CpPredecodeTests.cpp runs this protocol with fake jobs on two worker threads.
// Mutants (tests must fail): STALE_SNAPSHOT (the worker decodes the snapshot of its previous job), NO_JOIN (the recorder uses a job JoinJob rejected,
// matching the stage by its module only).

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace Libs::Graphics::Predecode {

enum class Mode : uint8_t { Off, On, Shadow };

inline Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_CP_PREDECODE"));
	return mode;
}

inline const char* ModeName(Mode mode) {
	return mode == Mode::On ? "1" : (mode == Mode::Shadow ? "shadow" : "0");
}

// Ok: the slot holds the description FindImage takes. Null: the T# is null (the binding is the null image). Decline: the recorder resolves it itself.
enum class Status : uint8_t { Ok, Null, Decline };

inline constexpr size_t kStages = 2; // 0 = vertex_info[0], 1 = pixel

// One stage of a job: written only by the worker that runs the job, read by the recorder after JoinJob.
template <class Slot>
struct Stage {
	const void*       module    = nullptr; // the program the worker decoded the images of
	const void*       resources = nullptr; // the snapshot it decoded (the job's own)
	std::vector<Slot> slots;               // one per program image (heap images: Decline)

	void Clear() {
		module    = nullptr;
		resources = nullptr;
		slots.clear();
	}
};

// Worker: decode the `count` images of a stage from `values` (the job's own snapshot) with decode(index, value, slot) -> Status.
template <class Slot, class Value, class Decode>
void RunStage(Stage<Slot>& stage, size_t stage_index, const void* module, const void* resources, const std::vector<Value>& values, size_t count,
              Decode&& decode) {
	stage.module    = module;
	stage.resources = resources;
	stage.slots.resize(count);
#ifdef KYTY_PREDECODE_MUTANT_STALE_SNAPSHOT
	// MUTANT: the values of the previous job this thread ran (a copy, so the mutant itself does not race).
	thread_local std::array<std::vector<Value>, kStages> previous;
	const std::vector<Value> stale = previous[stage_index % kStages];
	previous[stage_index % kStages] = values;
	for (size_t i = 0; i < count; i++) {
		const auto& source = i < stale.size() ? stale : values;
		stage.slots[i].status = i < source.size() ? decode(i, source[i], stage.slots[i]) : Status::Decline;
	}
#else
	(void)stage_index;
	for (size_t i = 0; i < count; i++) {
		stage.slots[i].status = i < values.size() ? decode(i, values[i], stage.slots[i]) : Status::Decline;
	}
#endif
}

// Recorder: the stage it may use for the stage it binds (module, resources: what the binding runs on; images: the program's image count), or null.
// joined: JoinJob accepted the job (its pages are clean and the worker's result is the draw's).
template <class Slot, size_t N>
const Stage<Slot>* Usable(const std::array<Stage<Slot>, N>& stages, bool joined, const void* module, const void* resources, size_t images) {
#ifndef KYTY_PREDECODE_MUTANT_NO_JOIN
	if (!joined) return nullptr;
#else
	(void)joined;
#endif
	for (const auto& stage: stages) {
		if (stage.module == nullptr || stage.module != module || stage.slots.size() != images) continue;
#ifndef KYTY_PREDECODE_MUTANT_NO_JOIN
		if (stage.resources != resources) continue;
#else
		(void)resources;
#endif
		return &stage;
	}
	return nullptr;
}

// S560 (KYTY_CP_PREDECODE_SLIM=0|1, default 0): with KYTY_CP_PREDECODE=1 the binding of a predecoded image is built in place in the stage's binding slot
// (FillBinding): one copy of the ~1.2 KB ImageDesc (slot -> binding) instead of three (slot -> local, local -> returned binding, returned -> slot).
inline bool SlimEnabled() {
	static const bool on = [] {
		const char* v = std::getenv("KYTY_CP_PREDECODE_SLIM");
		return v != nullptr && v[0] == '1' && v[1] == '\0';
	}();
	return on;
}

// The binding the recorder's returned-value path makes for a predecoded image, written into `out`, which holds the previous draw's binding of the
// slot: the description copied, find(out.desc) -> image id (null: not found; it may change the description, as FindImage and the view format do),
// layout `found` / `missing`, no image view, mip_views emptied (its capacity kept, as the swap in PrepareBindings keeps it). Every field the returned
// binding sets is set, so nothing of the previous draw stays. Mutant KEEP_STALE (the view and the mip views of the previous draw stay) must fail.
template <class Binding, class Desc, class Layout, class Find>
void FillBinding(Binding& out, const Desc& desc, Layout found, Layout missing, Find&& find) {
	out.desc = desc;
#ifndef KYTY_PREDECODE_SLIM_MUTANT_KEEP_STALE
	out.image_view = decltype(out.image_view) {};
	out.mip_views.clear();
#endif
	const auto id = find(out.desc);
	out.image_id  = id ? id : decltype(out.image_id) {};
	out.layout    = id ? found : missing;
}

// Per PrepareBindings call, added to the FrameTiming columns once (predecode_img_n / _decline_n / _nojob_n / _ne_n).
struct Counts {
	uint64_t img     = 0; // slots taken from the worker (shadow: compared)
	uint64_t decline = 0; // the worker declined (or the recorder could not take the slot: --lod-stats)
	uint64_t nojob   = 0; // images of a draw stage without a usable predecoded stage
	uint64_t ne      = 0; // shadow: the worker's slot differs from the recorder's resolve
};

} // namespace Libs::Graphics::Predecode

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_TEXTUREPREDECODE_H_
