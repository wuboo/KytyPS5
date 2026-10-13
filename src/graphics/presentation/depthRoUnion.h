#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_DEPTHROUNION_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_DEPTHROUNION_H_

#include "graphics/presentation/depthLayout.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

// S316: a depth image in a read-only layout, attached to the open pass, is read through DS_R|DS_W (renderDraw, "store writes") and then
// DS_R|DS_W|ShaderRead (descriptors); GetBarriers compares the masks with ==, so every alternation is a barrier that closes the pass
// (S314: 20.4/flip). Nothing writes the image in a read-only layout, so once an access is visible it stays visible until the next write:
// KYTY_DEPTH_RO_UNION=1 skips the barrier when the new access is a subset of the union already visible (shadow = decide and count, still
// record the barrier). Plain integers (VkImageLayout / VkAccessFlags2 values), so the decision runs in a CPU test without Vulkan.
namespace Libs::Graphics::DepthRoUnion {

using namespace Libs::Graphics::DepthLayout;

// VkAccessFlagBits2 values.
constexpr uint64_t kShaderRead = 0x20, kShaderWrite = 0x40, kColorWrite = 0x100, kDsWrite = 0x400, kTransferWrite = 0x1000,
                   kHostWrite = 0x4000, kMemoryWrite = 0x10000, kShaderStorageWrite = 1ull << 34;

// Writes that make the image dirty. DS_W is a real write only in a writable layout; in a read-only layout it is a fiction of the
// attachment store mask.
constexpr uint64_t kWriteMask = kShaderWrite | kColorWrite | kTransferWrite | kHostWrite | kMemoryWrite | kShaderStorageWrite;

// Layouts of the read-only family (depth and/or stencil cannot be written by draws; the mixed ones keep one aspect attachable, which a
// sampler of the other aspect never sees).
constexpr bool IsReadOnlyLayout(uint32_t layout) {
	switch (layout) {
		case kDepthStencilReadOnly:
		case kDepthReadOnly:
		case kStencilReadOnly:
		case kDepthReadOnlyStencilAttachment:
		case kDepthAttachmentStencilReadOnly: return true;
#if defined(KYTY_DEPTH_RO_UNION_MUTANT_ALLOW_GENERAL)
		case kGeneral:
		case kFeedbackLoop: return true;
#endif
		default: return false;
	}
}

// Does a request with this layout/access write the image? (the flag dirty_since_barrier is set by every such request)
constexpr bool Writes(uint32_t layout, uint64_t access) {
	return (access & kWriteMask) != 0 || (!IsReadOnlyLayout(layout) && (access & kDsWrite) != 0);
}

enum class Action : uint32_t { Barrier, Elide };
enum class Reason : uint32_t { NotApplicable, Layout, Dirty, NotSubset, Elided, Count };

struct Request {
	uint32_t layout_old = 0;
	uint32_t layout_new = 0;
	uint64_t access_old = 0; // the union visible so far
	uint64_t access_new = 0;
	bool     pass_open  = false;
	bool     attached   = false; // the image is an attachment of the open pass
	bool     whole      = false; // the request covers the whole image and no per-subresource state exists
	bool     dirty      = false; // written since the last recorded barrier (including by this request)
};

struct Decision {
	Action action = Action::Barrier;
	Reason reason = Reason::NotApplicable;
};

constexpr Decision Decide(const Request& r) {
	if (!r.pass_open || !r.attached || !r.whole) return {Action::Barrier, Reason::NotApplicable};
	if (r.layout_old != r.layout_new || !IsReadOnlyLayout(r.layout_new)) return {Action::Barrier, Reason::Layout};
#if !defined(KYTY_DEPTH_RO_UNION_MUTANT_IGNORE_DIRTY)
	if (r.dirty) return {Action::Barrier, Reason::Dirty};
#endif
	if ((r.access_new & ~r.access_old) != 0) return {Action::Barrier, Reason::NotSubset};
	return {Action::Elide, Reason::Elided};
}

// The invariant the whole scheme stands on, evaluated independently of Decide: an elision of a dirty image or in a writable layout.
constexpr bool BadElide(const Request& r, Action action) {
	return action == Action::Elide && (r.dirty || !IsReadOnlyLayout(r.layout_new) || r.layout_old != r.layout_new);
}

// Visible union after a recorded barrier that Decide rejected only because the access was not a subset: the old accesses stay visible
// (nothing wrote the image since), the new one becomes visible.
constexpr uint64_t UnionAfter(const Decision& d, uint64_t access_old, uint64_t access_new) {
	return d.reason == Reason::NotSubset ? (access_old | access_new) : access_new;
}

// Dirty flag after the request: set by every write, cleared by a recorded whole-image barrier (the write of the request itself follows it).
constexpr bool DirtyAfter(bool dirty, bool barrier_whole, uint32_t layout_new, uint64_t access_new) {
	return ((barrier_whole ? false : dirty) || Writes(layout_new, access_new));
}

enum class Mode : uint32_t { Off, On, Shadow };

inline Mode ModeFromEnv(const char* value) {
	if (value == nullptr) return Mode::Off;
	if (std::strcmp(value, "1") == 0) return Mode::On;
	if (std::strcmp(value, "shadow") == 0) return Mode::Shadow;
	return Mode::Off;
}

inline Mode CurrentMode() {
	static const Mode mode = ModeFromEnv(std::getenv("KYTY_DEPTH_RO_UNION"));
	return mode;
}

} // namespace Libs::Graphics::DepthRoUnion

#endif
