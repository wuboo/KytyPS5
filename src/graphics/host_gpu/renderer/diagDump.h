#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_

#include "common/controlFile.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/renderer/bcDump.h"
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
	uint64_t             addr      = 0; // guest address of the target image (0 = unknown)
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

// diag_tex=<ps>:<slot>:all: ResolveTexture queues one TexDraw per sampled slot of a draw (WantsTexAll); the draw's end
// (DiagTexDrawDone, with the index count and the first render target's address) logs [diag-tex-d] and feeds the
// per-flip histogram that FlipReport prints as [diag-tex-h]. No readback.
struct TexDraw {
	TexInfo    info;
	bool       has_bc7 = false;
	Bc7Summary bc7;
};
bool WantsTexAll(uint64_t ps_hash, uint32_t slot);
void AddTexDraw(TexDraw&& draw);
void TexDrawsDone(uint64_t index_count, uint64_t rt_addr);

// dump_tex=<ps>:<slot>[@filter]: ResolveTexture queues one request per block (WantsDumpTex fires at the first draw whose T#
// fits the filter); after that draw the renderer decodes mip 0 from guest memory and from the host image and hands both to
// WriteDumpTex, which logs [dump-tex] lines and writes the PNGs to $KYTY_DIAG_DIR.
struct DumpTexRequest {
	Common::SlotId image;
	TexInfo        info;
};
struct DumpTexResult {
	TexInfo              info;
	uint32_t             blocks_w = 0, blocks_h = 0; // BC blocks of mip 0
	std::vector<uint8_t> guest, host;                // blocks_w * blocks_h * 16 bytes each, empty = not available
	std::vector<std::string> notes;                  // one [dump-tex] line each: layout, host image, tracker state
};
bool                        WantsDumpTex(uint64_t ps_hash, uint32_t slot, const Common::ControlFile::TexKey& key);
void                        AddDumpTexRequest(DumpTexRequest&& request);
std::vector<DumpTexRequest> TakeDumpTexRequests();
void                        WriteDumpTex(const DumpTexResult& result);

bool Wants(uint64_t ps_hash);
void Store(uint64_t ps_hash, Target&& target);
void FlipReport(uint64_t flip);

} // namespace Libs::Graphics::DiagDump

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DIAGDUMP_H_
