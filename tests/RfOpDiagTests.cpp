#include "graphics/host_gpu/renderer/rfOpDiag.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using namespace Libs::Graphics::RfOp;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "RfOpDiagTests: failed: %s\n", message);
		std::exit(1);
	}
}

bool Has(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

// A simulated read fault inside the handler of WAIT_REG_MEM: the line names the opcode and the packet, the slot is the opcode's column.
void TestFaultInsideHandler() {
	const uint32_t packet[6] = {0xc0053c00u, 0x13u, 0xdeadbeefu, 0x1u, 0xffu, 0x0au};
	BeginPacket(0x3C, packet, 6);
	Event event;
	event.flip        = 6301;
	event.packet      = CurrentPacket();
	event.addr        = 0x56e35b970ull;
	event.pc_text     = "Handler+0x20";
	event.writer_op   = 0x2C;
	event.stale.ok    = true;
	event.fresh.ok    = true;
	event.stale.data[0] = 0x11;
	event.fresh.data[0] = 0x22;
	const auto line = FormatLine(event);
	Check(Has(line, "[rf-op] flip=6301 op=0x3c "), "line names the opcode being handled");
	Check(Has(line, "pkt=c0053c00,00000013,deadbeef,00000001 "), "line carries the first four dwords of the packet");
	Check(Has(line, "addr=0x000000056e35b970 "), "fault address");
	Check(Has(line, "pc=Handler+0x20 "), "faulting pc");
	Check(Has(line, "writer_op=0x2c writer_hash=- "), "last writer opcode, hash unknown");
	Check(Has(line, "stale=11000000000000000000000000000000 fresh=22000000000000000000000000000000"), "16 bytes before and after the drain");
	Check(SlotOf(event.packet.opcode) == 0 && std::strcmp(ColumnName(SlotOf(event.packet.opcode)), "rf_op_wait_reg_mem_n") == 0,
	      "WAIT_REG_MEM counts in its own column");
	EndPacket();
	Check(CurrentPacket().opcode == kNoOpcode, "EndPacket clears the opcode");
	Event outside;
	outside.packet = CurrentPacket();
	Check(Has(FormatLine(outside), " op=none ") && SlotOf(outside.packet.opcode) == kSlotNone, "fault outside any handler is 'none'");
}

void TestShortPacketAndOther() {
	const uint32_t packet[2] = {0xc0001000u, 0x7u};
	BeginPacket(0x27, packet, 2);
	Check(CurrentPacket().dw[1] == 7 && CurrentPacket().dw[2] == 0 && CurrentPacket().dw[3] == 0, "dwords past the buffer end read as 0");
	Check(SlotOf(0x27) == kSlotOther && std::strcmp(ColumnName(kSlotOther), "rf_op_other_n") == 0, "untracked opcode counts in other");
	Check(std::strcmp(ColumnName(kSlotNone), "rf_op_none_n") == 0, "none column");
	for (uint32_t slot = 0; slot < kTrackedCount; slot++) {
		Check(SlotOf(kTrackedOpcodes[slot]) == slot, "tracked opcode maps to its slot");
	}
	EndPacket();
}

void TestPacketIsPerThread() {
	const uint32_t packet[1] = {1};
	BeginPacket(0x22, packet, 1);
	Check(&CurrentPacket() == &CurrentPacket(), "one packet per thread");
	EndPacket();
}

void TestLogGate() {
	Check(ShouldLog(0, 100) && ShouldLog(399, 100), "first 400 lines");
	Check(!ShouldLog(400, 100) && !ShouldLog(5000, 6299), "no more lines outside the window");
	Check(ShouldLog(5000, 6300) && ShouldLog(5000, 6310) && !ShouldLog(5000, 6311), "flips 6300-6310 always log");
}

void TestWriterOps() {
	WriterOps writers;
	Check(writers.Find(0x10000, 0x100) == kNoOpcode, "no writer known");
	writers.Note(0x10000, 0x1000, 0x27);
	writers.Note(0x11000, 0x2000, 0x15);
	Check(writers.Find(0x10010, 4) == 0x27, "writer of a page");
	Check(writers.Find(0x10800, 0x1000) == 0x15, "range over two pages: the newer writer");
	writers.Note(0x10000, 0x10, 0x2C);
	Check(writers.Find(0x10ff0, 4) == 0x2C, "a later write replaces the writer");
	Check(writers.Find(0x13000, 4) == kNoOpcode, "page nobody wrote");
}

void TestResolvePc() {
	const auto text = ResolvePc(reinterpret_cast<uint64_t>(&TestResolvePc) + 4);
	Check(Has(text, "+0x") && text != "?", "pc resolves to symbol or image plus an offset");
	Check(ResolvePc(0) == "?", "null pc");
}


std::string FakeResolve(uint64_t pc) {
	char text[32];
	std::snprintf(text, sizeof(text), "f%llx", static_cast<unsigned long long>(pc));
	return text;
}

// S413: the new fields of the line.
void TestLineSiteWaitDirtyBt() {
	Event event;
	event.flip       = 6302;
	event.site       = SiteLabel(Libs::Graphics::CpSampler::SiteId(Libs::Graphics::CpSampler::Site::PvIndexHash));
	event.wait_us    = 8123;
	event.eager      = true;
	event.dirty      = true;
	event.page_dirty = false;
	const uint64_t frames[3] = {0x10, 0x20, 0x30};
	event.bt_text            = FormatBacktrace(frames, 3, &FakeResolve);
	const auto line          = FormatLine(event);
	Check(Has(line, " site=pv.index_hash "), "line names the CP breadcrumb site");
	Check(Has(line, " wait_us=8123 eager=1 dirty=1 page_dirty=0 "), "line carries wait, eager, dirty, page_dirty");
	Check(Has(line, " bt=f10|f20|f30"), "line carries the backtrace");
	Check(FormatBacktrace(frames, 0, &FakeResolve) == "-", "no frames");
	Event plain;
	Check(Has(FormatLine(plain), " site=none wait_us=0 eager=0 dirty=0 page_dirty=0 bt=-"), "defaults");
}

// S413: the RBP walker on a synthetic stack (the array stands for the thread stack, indices count 8-byte words).
void TestWalkFrames() {
	alignas(16) uint64_t stack[32] = {};
	const auto addr = [&](int word) { return reinterpret_cast<uint64_t>(&stack[word]); };
	// Leaf at sp = word 2 (returns to 0xA1 pushed at the word 2), frame chain: rbp=word 6 -> word 12 -> word 20 -> 0.
	stack[2]  = 0xA1;
	stack[6]  = addr(12);
	stack[7]  = 0xB1;
	stack[12] = addr(20);
	stack[13] = 0xC1;
	stack[20] = 0;
	stack[21] = 0xD1;
	FaultContext ctx {0x1000, addr(2), addr(6)};
	uint64_t     out[kMaxFrames] = {};
	auto         n               = WalkFrames(ctx, addr(0), addr(32), out);
	Check(n == 5, "pc, word at rsp and three chain frames");
	Check(out[0] == 0x1000 && out[1] == 0xA1 && out[2] == 0xB1 && out[3] == 0xC1 && out[4] == 0xD1, "frames in order");
	n = WalkFrames(ctx, addr(0), addr(32), out, 3);
	Check(n == 3, "capped at max");
	// A frame pointer that does not grow stops the chain after the frames already collected.
	stack[12] = addr(6);
	n = WalkFrames(ctx, addr(0), addr(32), out);
	Check(n == 4 && out[3] == 0xC1, "non-growing rbp stops the walk");
	stack[12] = addr(12); // a frame that points at itself must not loop
	n = WalkFrames(ctx, addr(0), addr(32), out);
	Check(n == 4 && out[3] == 0xC1, "self-pointing rbp stops the walk");
	// A bad rbp (outside the stack, misaligned, below rsp) yields only pc and the word at rsp.
	for (const uint64_t bad: {uint64_t {0}, addr(0) - 8, addr(32), addr(6) + 1, addr(1)}) {
		FaultContext c {0x1000, addr(2), bad};
		Check(WalkFrames(c, addr(0), addr(32), out) == 2, "bad rbp stops at once");
	}
	FaultContext no_sp {0x1000, 0, 0};
	Check(WalkFrames(no_sp, addr(0), addr(32), out) == 1 && out[0] == 0x1000, "no stack pointer: pc only");
	FaultContext near_top {0x1000, addr(2), addr(31)};
	Check(WalkFrames(near_top, addr(0), addr(32), out) == 2, "frame whose two words do not fit is refused");
}

void TestCurrentStackBounds() {
#if defined(__APPLE__)
	uint64_t lo = 0, hi = 0;
	int      local = 0;
	Check(CurrentStackBounds(&lo, &hi) && lo < reinterpret_cast<uint64_t>(&local) && reinterpret_cast<uint64_t>(&local) < hi,
	      "a local variable lies inside the thread stack bounds");
#endif
}

} // namespace

int main() {
	TestFaultInsideHandler();
	TestShortPacketAndOther();
	TestPacketIsPerThread();
	TestLogGate();
	TestWriterOps();
	TestResolvePc();
	TestLineSiteWaitDirtyBt();
	TestWalkFrames();
	TestCurrentStackBounds();
	return 0;
}
