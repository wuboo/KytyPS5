#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_

#include "common/slotVector.h"
#include "graphics/host_gpu/renderer/diagTex.h"

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

// Control file key diag_tex=<ps>:<slot>: the first resolve of that slot in a block logs the T# ([diag-tex]) and
// queues a request; the renderer reads the host image back after the draw (StoreTex) and FlipReport writes
// $KYTY_DIAG_DIR/<label>-<flip>-tex-<ps>-s<slot>.{ppm,raw} (mip 0, layer 0) with min/max/mean and the zero fraction.
struct TexRequest {
	Common::SlotId image;
	TexInfo        info;
};

bool                    WantsTex(uint64_t ps_hash, uint32_t slot); // true once per (block, ps, slot)
void                    AddTexRequest(TexRequest&& request);
std::vector<TexRequest> TakeTexRequests();
void                    StoreTex(const TexInfo& info, Target&& target); // empty data = could not be read back

bool Wants(uint64_t ps_hash);
void Store(uint64_t ps_hash, Target&& target);
void FlipReport(uint64_t flip);

} // namespace Libs::Graphics::DiagDump

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_
