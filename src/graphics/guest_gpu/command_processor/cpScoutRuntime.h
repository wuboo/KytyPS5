#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSCOUT_RUNTIME_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSCOUT_RUNTIME_H

// S454: start the lookahead worker on a graphics DCB and take its GetGraphicsPrograms results.

#include "graphics/guest_gpu/command_processor/cpScout.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/shader.h"

#include <array>
#include <span>

namespace Libs::Graphics {

class CommandProcessor;
class CommandBuffer;

namespace CpScout {

struct Prepared {
	PipelineCache::GraphicsPrograms      programs;
	std::array<ShaderVertexInputInfo, 3> vertex_info {};
	ShaderPixelInputInfo                 pixel_info {};
};

enum class TakeKind : uint8_t { Miss, Hit, Shadow };

void Begin(CommandProcessor& cp, PipelineCache& cache, std::span<const uint32_t> commands);
void Shutdown();
TakeKind Take(CommandBuffer& buffer, Prepared& out);

} // namespace CpScout
} // namespace Libs::Graphics

#endif
