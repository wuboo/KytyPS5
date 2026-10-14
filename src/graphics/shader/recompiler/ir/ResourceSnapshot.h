#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_

#include "graphics/shader/recompiler/ir/SrtMemo.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

struct DescriptorValue {
	std::array<uint32_t, 8> dwords      = {};
	uint32_t                dword_count = 0;

	bool operator==(const DescriptorValue& other) const {
		return dword_count == other.dword_count && dwords == other.dwords;
	}
};

enum class UniformFillKind { None, Buffer, Image };

struct UniformFill {
	UniformFillKind          kind         = UniformFillKind::None;
	uint32_t                 resource     = 0;
	std::array<uint32_t, 3> group_stride {};
	uint32_t                 words        = 0;
	uint32_t                 value        = 0;
	uint32_t                 bounded_elements = 0;

	bool operator==(const UniformFill&) const = default;
};

// S379: why a bound image became a null T#. Empty unless KYTY_DIAG_NULLDESC=1.
// Not a specialization key.
struct ImageNullNote {
	uint8_t  reason = 0;
	uint64_t addr   = 0;
};

struct ResourceSnapshot {
	std::vector<DescriptorValue> buffers;
	std::vector<DescriptorValue> images;
	std::vector<DescriptorValue> samplers;
	std::vector<uint32_t>        flattened_srt;
	std::vector<uint32_t>        user_data;
	std::vector<std::pair<uint64_t, uint64_t>> specialization_reads;
	UniformFill                 uniform_fill;
	std::vector<ImageNullNote>  image_null_notes;
	// S324 (KYTY_SRT_FAST): the walk that produced this snapshot, and what the last MaterializeResources did with it.
	SrtMemo                     srt_memo;
	SrtMemoOutcome              srt_outcome  = SrtMemoOutcome::None;
	uint32_t                    srt_mismatch = UINT32_MAX; // Shadow: first differing flat slot (UINT32_MAX: a descriptor)
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_RESOURCESNAPSHOT_H_
