#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDPREPMEMO_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDPREPMEMO_H_

// S469 KYTY_BIND_PREP_MEMO=0|1|shadow (default 0 = the code as it was): per-program memo of RenderExecutor::PrepareBindings.
// For one program (the compiled-shader object) it remembers what the last call produced - the TextureBinding of every image, the sampler handles
// and the shader-data dwords - together with the hash of everything those were computed from (the T#/S# words of the snapshot, the user-data words the
// program reads, the shader's own image resources) and the generation of the control file (force_tex / diag_tex change the resolve).
// A call whose inputs hash equals the remembered one replays the result instead of resolving every texture again; the images are not trusted blindly:
// each one is asked of the texture cache's own memo (TexMemo::Reuse: slot alive, registered, no rebind pending, no registration of its pages since the
// entry, then tick + LRU touch + MaterializeColorClear), so the replay is valid exactly as long as KYTY_TEX_RESOLVE_MEMO=1 would be.
// shadow: the full path runs, the replay is only predicted (commit=false) and compared with it (counter bind_prep_memo_ne_n).
// This header holds the keyed table and the input hash (no Vulkan), so tests/BindPrepMemoTests.cpp can pin them.
//
// Mutants (tests, WILL_FAIL): KYTY_BINDPREP_MUTANT_{IGNORES_INPUTS, IGNORES_CONTROL_GEN, HASH_SKIPS_SAMPLERS, HASH_SKIPS_USER_DATA}.

#include "graphics/host_gpu/renderer/pipeline/bindMemo.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>

namespace Libs::Graphics::BindPrepMemo {

enum class Mode : uint8_t { Off, On, Shadow };

// null/""/"0" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
[[nodiscard]] inline Mode ParseMode(const char* text) {
	if (text == nullptr || text[0] == '\0') return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

[[nodiscard]] inline const char* ModeName(Mode mode) {
	return mode == Mode::On ? "1" : mode == Mode::Shadow ? "shadow" : "0";
}

[[nodiscard]] inline Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_BIND_PREP_MEMO"));
	return mode;
}

// The hash of the inputs of one PrepareBindings call. Words go in as they are; the sections are separated so that a word moving between them changes the digest.
class InputsHash {
public:
	void Program(uint64_t shader_hash, uint32_t stage, uint32_t image_count, uint32_t sampler_count, uint32_t shader_data_dwords) {
		h_.Add(shader_hash);
		h_.Add(static_cast<uint64_t>(stage) << 32 | image_count);
		h_.Add(static_cast<uint64_t>(sampler_count) << 32 | shader_data_dwords);
	}
	// What of the program's ImageResource shapes the resolve, already packed by the caller (flags / enums / mip fields).
	void ImageResource(uint64_t packed_a, uint64_t packed_b) {
		h_.Add(packed_a);
		h_.Add(packed_b);
	}
	void Image(const uint32_t* dwords, uint32_t count) {
		h_.Add(0x100u | count);
		for (uint32_t i = 0; i < count; i++) h_.Add(dwords[i]);
	}
	// `flags`: the program's own sampler fields NativeSampler reads (depth_compare, force_point_filtering, integer_border).
	void Sampler(const uint32_t* dwords, uint32_t count, uint32_t flags) {
#if !defined(KYTY_BINDPREP_MUTANT_HASH_SKIPS_SAMPLERS)
		h_.Add(0x200u | count | static_cast<uint64_t>(flags) << 16);
		for (uint32_t i = 0; i < count; i++) h_.Add(dwords[i]);
#else
		(void)dwords;
		(void)count;
		(void)flags;
#endif
	}
	// `index`: the position of the register in the program's user-data list (the dwords land in the shader data at that position).
	void UserData(uint32_t index, uint32_t word) {
#if !defined(KYTY_BINDPREP_MUTANT_HASH_SKIPS_USER_DATA)
		h_.Add(static_cast<uint64_t>(index) << 32 | word);
#else
		(void)index;
		(void)word;
#endif
	}
	[[nodiscard]] uint64_t Digest() const { return h_.Digest(); }

private:
	BindMemo::Hash h_;
};

// One slot per program. `Payload` is what the caller replays (the prepared images, samplers, shader data).
template <class Payload>
class Table {
public:
	static constexpr size_t kMaxPrograms = 4096;

	// The payload of `owner` when it was stored with the same inputs under the same control-file generation; nullptr otherwise.
	[[nodiscard]] Payload* Find(const void* owner, uint64_t inputs, uint64_t control_gen) {
		const auto found = m_slots.find(owner);
		if (found == m_slots.end()) return nullptr;
		auto& slot = found->second;
#if !defined(KYTY_BINDPREP_MUTANT_IGNORES_INPUTS)
		if (slot.inputs != inputs) return nullptr;
#endif
#if !defined(KYTY_BINDPREP_MUTANT_IGNORES_CONTROL_GEN)
		if (slot.control_gen != control_gen) return nullptr;
#endif
		return &slot.payload;
	}
	void Store(const void* owner, uint64_t inputs, uint64_t control_gen, Payload&& payload) {
		if (m_slots.size() >= kMaxPrograms && m_slots.find(owner) == m_slots.end()) m_slots.clear();
		auto& slot       = m_slots[owner];
		slot.inputs      = inputs;
		slot.control_gen = control_gen;
		slot.payload     = std::move(payload);
	}
	void   Erase(const void* owner) { m_slots.erase(owner); }
	size_t Size() const { return m_slots.size(); }

private:
	struct Slot {
		uint64_t inputs      = 0;
		uint64_t control_gen = 0;
		Payload  payload;
	};
	std::unordered_map<const void*, Slot> m_slots;
};

} // namespace Libs::Graphics::BindPrepMemo

#endif // KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDPREPMEMO_H_
