#ifndef KYTY_GRAPHICS_SHADER_RECOMPILER_SRTREADER_H_
#define KYTY_GRAPHICS_SHADER_RECOMPILER_SRTREADER_H_

// KYTY_SRT_READER selects how ordinary SRT walks read guest memory.
// main (default): compiled walker when read_memory is empty, otherwise memcpy.
// nojit: RefreshFlatBuffer skips srt_code and interprets the same memcpy path.
// checked: PipelineCache installs ReadShaderGuestMemoryOrZero. A miss returns
// zeros and success, because MaterializeResources treats a false read as EXIT.

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>

namespace Libs::Graphics::SrtReader {

inline constexpr uint64_t kGuestMemoryMin = 0x40000;

enum class Mode { Main, NoJit, Checked };

inline Mode Parse(const char* text) {
	if (text != nullptr) {
		if (std::strcmp(text, "nojit") == 0) {
			return Mode::NoJit;
		}
		if (std::strcmp(text, "checked") == 0) {
			return Mode::Checked;
		}
	}
	return Mode::Main;
}

inline Mode Current() {
	static const Mode mode = Parse(std::getenv("KYTY_SRT_READER"));
	return mode;
}

inline const char* ModeName(Mode mode) {
	switch (mode) {
		case Mode::NoJit: return "nojit";
		case Mode::Checked: return "checked";
		case Mode::Main: return "main";
	}
	return "main";
}

inline const char* ModeName() {
	return ModeName(Current());
}

// Compiled walker is the main/checked default. checked also clears it by setting read_memory.
inline bool UseCompiledWalker() {
#if defined(KYTY_SRT_MUTANT_IGNORE_NOJIT)
	return true;
#else
	return Current() != Mode::NoJit;
#endif
}

// Install the ordinary reader for checked. Leaves main and nojit on the memcpy/JIT path.
inline void Apply(ShaderRecompiler::IR::SrtRuntime& runtime, ShaderRecompiler::IR::SrtMemoryReader reader) {
#if !defined(KYTY_SRT_MUTANT_NO_CHECKED)
	if (Current() == Mode::Checked && runtime.read_memory == nullptr) {
		runtime.read_memory = reader;
	}
#else
	(void)runtime;
	(void)reader;
#endif
}

// `read_backing` is TryReadBufferBacking, or a test stub. False, a null stub, or an address
// below guest memory becomes zeros and success. `gpu_dirty` is reported by the stub for DIAG.
using BackingRead = bool (*)(void* userdata, uint64_t address, void* data, uint64_t size, bool* gpu_dirty);

inline bool ReadOrZero(uint64_t address, std::span<uint32_t> values, BackingRead read_backing, void* userdata) {
	if (values.empty()) {
		return true;
	}
	const auto zeros = [&]() {
#if defined(KYTY_SRT_MUTANT_FAULT_UNMAPPED)
		return false;
#else
		std::fill(values.begin(), values.end(), 0u);
		return true;
#endif
	};
	if (address < kGuestMemoryMin) {
		return zeros();
	}
	bool gpu_dirty = false;
	if (read_backing == nullptr ||
	    !read_backing(userdata, address, values.data(), values.size_bytes(), &gpu_dirty)) {
		return zeros();
	}
	(void)gpu_dirty;
	return true;
}

} // namespace Libs::Graphics::SrtReader

#endif
