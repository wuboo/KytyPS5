#include "graphics/host_gpu/renderer/gfDiag.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using namespace Libs::Graphics::GuestFault;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "GfDiagTests: failed: %s\n", message);
		std::exit(1);
	}
}

bool Has(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

void TestInsertAndHit() {
	BackingReads reads;
	reads.Note(0x10000100, 64, Site::VertexAttribTable, 6620);
	reads.Note(0x20000000, 8192, Site::IndexBytes, 6621);
	const auto a = reads.Find(0x10000ff0); // same 4 KiB page as the first read, different bytes
	Check(a.found && a.site == Site::VertexAttribTable && a.flip == 6620, "a read marks its whole page");
	Check(AfterBacking(a, 6620), "same flip -> after_backing");
	Check(!AfterBacking(a, 6621), "later flip -> not after_backing");
	const auto b = reads.Find(0x20001800); // second page of the 8 KiB range
	Check(b.found && b.site == Site::IndexBytes, "a range covers every page it touches");
	Check(!reads.Find(0x20002000).found, "the page after the range is untouched");
	Check(!reads.Find(0x0fffff00).found, "the page before is untouched");
	Check(!AfterBacking(BackingReads::Hit {}, 0), "no hit is never after_backing");
}

void TestNewestWinsAndExpiry() {
	BackingReads reads;
	reads.Note(0x30000000, 16, Site::VertexBufferTable, 10);
	reads.Note(0x30000010, 16, Site::VertexAttribTable, 11);
	Check(reads.Find(0x30000000).site == Site::VertexAttribTable, "the newest overlapping read wins");
	for (uint64_t i = 0; i < BackingReads::kReads; i++) {
		reads.Note(0x40000000 + i * 0x10000, 16, Site::IndexBytes, 12);
	}
	Check(!reads.Find(0x30000000).found, "a read falls out after 256 newer ones");
	Check(reads.Find(0x40000000 + (BackingReads::kReads - 1) * 0x10000).found, "the newest 256 stay");
	reads.Note(0x50000000, 0, Site::IndexBytes, 13);
	Check(!reads.Find(0x50000000).found, "an empty range is not recorded");
	reads.Clear();
	Check(!reads.Find(0x40000000 + (BackingReads::kReads - 1) * 0x10000).found, "Clear forgets everything");
}

void TestFormatLine() {
	Event event;
	event.flip          = 6620;
	event.addr          = 0x56e35b970ull;
	event.wait_us       = 8123;
	event.eager         = true;
	event.dirty         = true;
	event.page_dirty    = true;
	event.after_backing = true;
	event.backing_site  = Site::VertexAttribTable;
	event.bt_text       = "a+0x1|b+0x2";
	const auto line = FormatLine(event);
	Check(Has(line, "[gf] flip=6620 addr=0x000000056e35b970 page=0x000000056e35b000 wait_us=8123 "), "head of the line");
	Check(Has(line, "eager=1 dirty=1 page_dirty=1 after_backing=1 backing_site=vertex_attrib_table bt=a+0x1|b+0x2"), "tail of the line");
	Event none;
	const auto line2 = FormatLine(none);
	Check(Has(line2, "after_backing=0 backing_site=- bt=-"), "defaults");
}

void TestWindowAndLimit() {
	uint64_t a = 1, b = 1;
	Check(ParseWindow("6600:6700", &a, &b) && a == 6600 && b == 6700, "window parses");
	Check(!ParseWindow("6600", &a, &b) && !ParseWindow("7:3", &a, &b) && !ParseWindow(nullptr, &a, &b) && !ParseWindow("1:2x", &a, &b),
	      "malformed windows are refused");
	Check(ShouldLog(399, false) && !ShouldLog(400, false), "400 lines outside the window");
	Check(ShouldLog(100000, true), "every line inside the window");
}

} // namespace

int main() {
	TestInsertAndHit();
	TestNewestWinsAndExpiry();
	TestFormatLine();
	TestWindowAndLimit();
	std::puts("GfDiagTests: ok");
	return 0;
}
