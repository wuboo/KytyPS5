#include "graphics/presentation/depthRoUnion.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics::DepthRoUnion;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "DepthRoUnionTests: failed: %s\n", message);
		std::exit(1);
	}
}

constexpr uint64_t kDsRead = 0x200, kSampled = 0x20;
constexpr uint64_t kStore  = kDsRead | kDsWrite;          // renderDraw: attachment mask
constexpr uint64_t kSample = kDsRead | kDsWrite | kSampled; // descriptors: attachment mask + ShaderRead

Request Ro(uint64_t access_old, uint64_t access_new) {
	Request r;
	r.layout_old = r.layout_new = kDepthReadOnly;
	r.access_old = access_old;
	r.access_new = access_new;
	r.pass_open = r.attached = r.whole = true;
	return r;
}

// The S314 alternation: after the first barrier (…->kSample) the draws alternate kStore / kSample with no barrier needed.
void TestAlternation() {
	Check(Decide(Ro(kStore, kSample)).reason == Reason::NotSubset, "first sampled read is a new access: barrier");
	const auto back = Decide(Ro(kSample, kStore));
	Check(back.action == Action::Elide && back.reason == Reason::Elided, "back to the store mask: subset of the union, elided");
	Check(Decide(Ro(kSample, kSample)).action == Action::Elide, "same mask: elided");
	// the union after the rejected first request keeps the old bits
	Check(UnionAfter(Decide(Ro(kStore, kSample)), kStore, kSample) == kSample, "union after a not-subset barrier");
	Check(UnionAfter(Decide(Ro(kStore, kSample)), kStore, kSampled) == (kStore | kSampled), "union keeps old accesses");
	Check(UnionAfter(Decide(Ro(kStore, kStore)), kStore, kSample) == kSample, "other reasons: the new mask only");
}

void TestDirty() {
	auto r  = Ro(kSample, kStore);
	r.dirty = true;
	Check(Decide(r).reason == Reason::Dirty && Decide(r).action == Action::Barrier, "written since the last barrier: barrier");
	Check(Writes(kDepthReadOnly, kTransferWrite) && Writes(kDepthReadOnly, kShaderWrite) && Writes(kDepthReadOnly, kShaderStorageWrite) &&
	          Writes(kDepthReadOnly, kMemoryWrite) && Writes(kDepthReadOnly, kColorWrite) && Writes(kDepthReadOnly, kHostWrite),
	      "every write path counts: transfer (copy, upload, clear), shader, storage, memory, color, host");
	Check(!Writes(kDepthReadOnly, kStore), "DS_W in a read-only layout is a fiction, not a write");
	Check(Writes(kDepthStencilAttachment, kStore), "DS_W in a writable layout is a write");
	Check(!Writes(kDepthReadOnly, kSample), "sampling is not a write");
	// the flag lifecycle: a barrier clears it, then the request's own write sets it again
	Check(!DirtyAfter(true, true, kDepthReadOnly, kSample), "recorded whole-image barrier clears dirty");
	Check(DirtyAfter(true, false, kDepthReadOnly, kSample), "partial barrier does not clear it");
	Check(DirtyAfter(true, true, kDepthReadOnly, kTransferWrite), "the write of the request that follows the barrier sets it");
	Check(DirtyAfter(false, false, kDepthStencilAttachment, kStore), "attachment write in a writable layout sets it");
}

void TestLayout() {
	auto r      = Ro(kSample, kStore);
	r.layout_old = r.layout_new = kGeneral;
	Check(Decide(r).reason == Reason::Layout, "GENERAL is not read-only: barrier");
	r.layout_old = r.layout_new = kFeedbackLoop;
	Check(Decide(r).reason == Reason::Layout, "feedback loop: barrier");
	r.layout_old = r.layout_new = kDepthStencilAttachment;
	Check(Decide(r).reason == Reason::Layout, "writable attachment layout: barrier");
	r = Ro(kSample, kStore);
	r.layout_old = kDepthStencilAttachment;
	Check(Decide(r).reason == Reason::Layout, "layout change: barrier");
	for (uint32_t l: {kDepthStencilReadOnly, kDepthReadOnly, kStencilReadOnly, kDepthReadOnlyStencilAttachment, kDepthAttachmentStencilReadOnly}) {
		r.layout_old = r.layout_new = l;
		Check(Decide(r).action == Action::Elide, "read-only family layout: elided");
	}
}

void TestApplicability() {
	auto r   = Ro(kSample, kStore);
	r.whole  = false;
	Check(Decide(r).action == Action::Barrier && Decide(r).reason == Reason::NotApplicable, "partial range or per-subresource: barrier");
	r = Ro(kSample, kStore);
	r.attached = false;
	Check(Decide(r).reason == Reason::NotApplicable, "not an attachment: barrier");
	r = Ro(kSample, kStore);
	r.pass_open = false;
	Check(Decide(r).reason == Reason::NotApplicable, "no pass open: barrier");
}

void TestBadElide() {
	auto r  = Ro(kSample, kStore);
	r.dirty = true;
	Check(BadElide(r, Action::Elide), "elision of a dirty image is bad");
	Check(!BadElide(r, Action::Barrier), "a barrier is never bad");
	r = Ro(kSample, kStore);
	r.layout_old = r.layout_new = kGeneral;
	Check(BadElide(r, Action::Elide), "elision in GENERAL is bad (independent of Decide)");
	Check(!BadElide(Ro(kSample, kStore), Action::Elide), "clean read-only elision is fine");
	// Decide itself never produces a bad elision over the whole input grid
	for (uint32_t layout: {kGeneral, kDepthReadOnly, kDepthStencilAttachment, kFeedbackLoop, kDepthStencilReadOnly}) {
		for (int dirty = 0; dirty < 2; dirty++) {
			Request q = Ro(kSample, kStore);
			q.layout_old = q.layout_new = layout;
			q.dirty      = dirty != 0;
			Check(!BadElide(q, Decide(q).action), "Decide never elides a dirty image or a writable layout");
		}
	}
}

void TestMode() {
	Check(ModeFromEnv(nullptr) == Mode::Off && ModeFromEnv("0") == Mode::Off && ModeFromEnv("") == Mode::Off, "default off");
	Check(ModeFromEnv("1") == Mode::On && ModeFromEnv("shadow") == Mode::Shadow && ModeFromEnv("2") == Mode::Off, "modes");
}

} // namespace

int main() {
	TestAlternation();
	TestDirty();
	TestLayout();
	TestApplicability();
	TestBadElide();
	TestMode();
	std::puts("DepthRoUnionTests: ok");
	return 0;
}
