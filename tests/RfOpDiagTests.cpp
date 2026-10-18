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

} // namespace

int main() {
	TestFaultInsideHandler();
	TestShortPacketAndOther();
	TestPacketIsPerThread();
	TestLogGate();
	TestWriterOps();
	TestResolvePc();
	return 0;
}
