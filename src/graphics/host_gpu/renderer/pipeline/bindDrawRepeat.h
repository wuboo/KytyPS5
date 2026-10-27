#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDDRAWREPEAT_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_BINDDRAWREPEAT_H_

// S493 KYTY_BIND_DRAW_REPEAT=0|1|shadow (default 0): when PrepareBindings is called again for the same compiled program with the same
// snapshot inputs (T#/S# words, user data, resource shapes) and the texture cache registration generation is unchanged, replay the last
// prepared images/samplers/shader-data without ResolveTexture, ReuseImage, BindImage, or MaterializeColorClear. Each stored image is
// rechecked (slot alive, registered, no rebind, no registration touch on its guest pages since it was stored). shadow compares replay vs full path.
// Mutants (tests, WILL_FAIL): KYTY_BINDREPEAT_MUTANT_{IGNORES_REG_GEN, IGNORES_INPUTS}.

#include "graphics/host_gpu/renderer/pipeline/bindMemo.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::BindDrawRepeat {

enum class Mode : uint8_t { Off, On, Shadow };

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
	static const Mode mode = ParseMode(std::getenv("KYTY_BIND_DRAW_REPEAT"));
	return mode;
}

class InputsHash {
public:
	void Program(uint64_t shader_hash, uint32_t stage, uint32_t image_count, uint32_t sampler_count, uint32_t shader_data_dwords) {
		h_.Add(shader_hash);
		h_.Add(static_cast<uint64_t>(stage) << 32 | image_count);
		h_.Add(static_cast<uint64_t>(sampler_count) << 32 | shader_data_dwords);
	}
	void ImageResource(uint64_t packed_a, uint64_t packed_b) {
		h_.Add(packed_a);
		h_.Add(packed_b);
	}
	void Image(const uint32_t* dwords, uint32_t count) {
		h_.Add(0x100u | count);
		for (uint32_t i = 0; i < count; i++) h_.Add(dwords[i]);
	}
	void Sampler(const uint32_t* dwords, uint32_t count, uint32_t flags) {
#if !defined(KYTY_BINDREPEAT_MUTANT_IGNORES_INPUTS)
		h_.Add(0x200u | count | static_cast<uint64_t>(flags) << 16);
		for (uint32_t i = 0; i < count; i++) h_.Add(dwords[i]);
#else
		(void)dwords;
		(void)count;
		(void)flags;
#endif
	}
	void UserData(uint32_t index, uint32_t word) {
#if !defined(KYTY_BINDREPEAT_MUTANT_IGNORES_INPUTS)
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

// One image row the replay must still validate against the texture cache.
template <class Id>
struct ImageStamp {
	Id       image_id {};
	uint64_t address = 0;
	uint64_t size    = 0;
	uint64_t reg_gen = 0;
	bool     storage = false;
};

template <class Id, class H>
[[nodiscard]] bool ImageStillValid(const ImageStamp<Id>& stamp, H& hooks) {
	if (!stamp.image_id) return true;
	const auto state = hooks.State(stamp.image_id);
	if (!state.exists || !state.registered || state.needs_rebind) return false;
#if !defined(KYTY_BINDREPEAT_MUTANT_IGNORES_REG_GEN)
	if (hooks.RegistrationsSince(stamp.address, stamp.size, stamp.reg_gen)) return false;
#endif
	return true;
}

template <class Entry>
class Table {
public:
	[[nodiscard]] Entry* Find(const void* program, uint64_t inputs, uint64_t reg_gen) {
		const Key key {program, inputs, reg_gen};
		const auto found = map_.find(key);
		return found == map_.end() ? nullptr : &found->second;
	}
	void Store(const void* program, uint64_t inputs, uint64_t reg_gen, Entry&& entry) {
		map_[{program, inputs, reg_gen}] = std::move(entry);
	}
	void Erase(const void* program) {
		for (auto it = map_.begin(); it != map_.end();) {
			if (it->first.program == program) it = map_.erase(it);
			else ++it;
		}
	}

private:
	struct Key {
		const void* program = nullptr;
		uint64_t    inputs  = 0;
		uint64_t    reg_gen = 0;
		bool operator==(const Key& o) const {
			return program == o.program && inputs == o.inputs && reg_gen == o.reg_gen;
		}
	};
	struct KeyHash {
		size_t operator()(const Key& k) const {
			return std::hash<const void*> {}(k.program) ^ static_cast<size_t>(k.inputs) * 1315423911u ^
			       static_cast<size_t>(k.reg_gen) * 2654435761u;
		}
	};
	std::unordered_map<Key, Entry, KeyHash> map_;
};

} // namespace Libs::Graphics::BindDrawRepeat

#endif
