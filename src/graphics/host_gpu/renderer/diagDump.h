#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_

#include <cstdint>
#include <vector>

namespace Libs::Graphics::DiagDump {

// Control file key dump_rt=<ps>[:<flip>]: after every draw of a listed pixel shader in the matching
// flip the renderer reads every colour target and the depth target back (Store keeps the last
// draw's copy per target); FlipReport (called when the flip completes) writes them as
// $KYTY_FRAME_DUMP_DIR/<label>-<flip>-mrt<N>.ppm (-depth for depth) and logs one [dump-rt] line each.
struct Target {
	int                  mrt       = 0; // colour slot, -1 = depth
	int                  vk_format = 0; // numeric VkFormat of the copy
	uint32_t             width     = 0;
	uint32_t             height    = 0;
	std::vector<uint8_t> data;
};

bool Wants(uint64_t ps_hash);
void Store(uint64_t ps_hash, Target&& target);
void FlipReport(uint64_t flip);

} // namespace Libs::Graphics::DiagDump

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_
