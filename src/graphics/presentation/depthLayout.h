#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_DEPTHLAYOUT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_DEPTHLAYOUT_H_

#include <array>
#include <atomic>
#include <cstdint>

// S313: which layout a depth attachment is kept in between draws. depth_attachment_layout() follows the write set of every draw
// (DepthAttachment / DepthReadOnly / DepthStencilAttachment / ...), so alternating draws change the layout and the barrier closes the
// open render pass (renderDraw.cpp, S311 transit-sites). With KYTY_DEPTH_GENERAL=1 an attachment that is not a feedback loop stays in
// GENERAL. Plain integers (VkImageLayout values), so the decision runs in a CPU test without Vulkan.
namespace Libs::Graphics::DepthLayout {

constexpr uint32_t kUndefined = 0, kGeneral = 1, kDepthStencilAttachment = 3, kDepthStencilReadOnly = 4, kShaderReadOnly = 5,
                   kTransferSrc = 6, kTransferDst = 7, kDepthReadOnlyStencilAttachment = 1000117000,
                   kDepthAttachmentStencilReadOnly = 1000117001, kDepthAttachment = 1000241000, kDepthReadOnly = 1000241001,
                   kStencilAttachment = 1000241002, kStencilReadOnly = 1000241003, kFeedbackLoop = 1000339000;

// natural   layout depth_attachment_layout() (plus the sampled-aspect fix-up) asks for
// feedback the draw samples an aspect it also writes: the old layout, GENERAL would not be enough on a host with the loop extension
constexpr uint32_t Choose(bool on, uint32_t natural, bool feedback) {
#if defined(KYTY_DEPTH_GENERAL_MUTANT_FEEDBACK)
	return on ? kGeneral : natural;
#else
	return on && !feedback ? kGeneral : natural;
#endif
}

// A request that would have been a layout change under the old rules and is not one now (the image already sits in `chosen`).
constexpr bool Skipped(bool on, uint32_t current, uint32_t natural, uint32_t chosen) {
#if defined(KYTY_DEPTH_GENERAL_MUTANT_SKIP_ANY)
	return on && current == chosen;
#else
	return on && current == chosen && current != natural;
#endif
}

// Slot of a layout in the pair table (0 = other).
constexpr uint32_t Slot(uint32_t layout) {
	switch (layout) {
		case kUndefined: return 1;
		case kGeneral: return 2;
		case kDepthStencilAttachment: return 3;
		case kDepthStencilReadOnly: return 4;
		case kShaderReadOnly: return 5;
		case kTransferSrc: return 6;
		case kTransferDst: return 7;
		case kDepthReadOnlyStencilAttachment: return 8;
		case kDepthAttachmentStencilReadOnly: return 9;
		case kDepthAttachment: return 10;
		case kDepthReadOnly: return 11;
		case kStencilAttachment: return 12;
		case kStencilReadOnly: return 13;
		case kFeedbackLoop: return 14;
		default: return 0;
	}
}
constexpr uint32_t kSlots = 15;

constexpr const char* SlotName(uint32_t slot) {
	constexpr const char* names[kSlots] = {"other", "undefined", "general", "depth_stencil_attachment", "depth_stencil_read_only",
	                                       "shader_read_only", "transfer_src", "transfer_dst", "depth_read_stencil_attachment",
	                                       "depth_attachment_stencil_read", "depth_attachment", "depth_read_only",
	                                       "stencil_attachment", "stencil_read_only", "feedback_loop"};
	return slot < kSlots ? names[slot] : "other";
}

struct Pairs {
	std::array<std::atomic<uint64_t>, kSlots * kSlots> n {};
};
inline Pairs& Table() {
	static Pairs pairs;
	return pairs;
}
// Counts only real changes (src != dst); returns whether it was one.
inline bool NotePair(uint32_t src, uint32_t dst) {
	if (src == dst) return false;
	Table().n[Slot(src) * kSlots + Slot(dst)].fetch_add(1, std::memory_order_relaxed);
	return true;
}

} // namespace Libs::Graphics::DepthLayout

#endif
