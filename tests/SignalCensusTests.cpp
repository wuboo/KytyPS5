// S276: signal census (common/signalCensus.h): signal classification, counters, main/other thread split, csv columns and the
// per-RIP histogram (top list, opcode capture, overflow). Pure logic, no signals raised.
#include "common/signalCensus.h"

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <thread>

namespace {

using namespace Common::SignalCensus;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "SignalCensusTests: failed: %s\n", message);
		std::exit(1);
	}
}

uint64_t Column(const char* name) {
	uint64_t values[kColumnCount];
	Snapshot(values);
	for (size_t column = 0; column < kColumnCount; column++) {
		if (std::strcmp(ColumnName(column), name) == 0) return values[column];
	}
	Check(false, "column exists");
	return 0;
}

void TestClassification() {
	Check(SigFromHostSignal(SIGSEGV) == Sig::Segv, "SIGSEGV");
	Check(SigFromHostSignal(SIGBUS) == Sig::Bus, "SIGBUS");
	Check(SigFromHostSignal(SIGILL) == Sig::Ill, "SIGILL");
	Check(SigFromHostSignal(SIGUSR1) == Sig::Usr1, "SIGUSR1");
	Check(SigFromHostSignal(SIGTERM) == Sig::Other, "anything else");
}

void TestCounters() {
	Reset();
	OnSignal(Sig::Bus);
	OnSignal(Sig::Bus);
	OnSignal(Sig::Ill);
	OnPath(Path::GpuFaultHandled);
	OnPath(Path::Sse4a);
	OnPath(Path::Sse4a);
	OnPath(Path::IllUnhandled);
	Check(Column("sig_bus") == 2 && Column("sig_ill") == 1 && Column("sig_segv") == 0 && Column("sig_usr1") == 0, "signal columns");
	Check(Column("path_gpu_fault_ok") == 1 && Column("path_sse4a") == 2 && Column("path_ill_unhandled") == 1 && Column("path_cpuext") == 0,
	      "path columns");
	Reset();
	Check(Column("sig_bus") == 0 && Column("path_sse4a") == 0, "reset");
}

void TestThreadSplit() {
	Reset();
	Check(!OnMainThread(), "no main thread registered: everything is 'other'");
	OnSignal(Sig::Segv);
	SetMainThread();
	Check(OnMainThread(), "registered thread");
	OnSignal(Sig::Segv);
	OnSignal(Sig::Usr1);
	std::thread([] { OnSignal(Sig::Segv); }).join();
	Check(Column("sig_main_thread") == 2, "two signals on the main thread");
	Check(Column("sig_other_threads") == 2, "one before registration + one on a worker thread");
	Check(Column("sig_segv") == 3 && Column("sig_usr1") == 1, "totals unaffected by the split");
}

void TestCsv() {
	Reset();
	OnSignal(Sig::Ill);
	OnPath(Path::Rsqrt);
	char* header_text = nullptr;
	char* row_text    = nullptr;
	size_t header_size = 0, row_size = 0;
	FILE* header = open_memstream(&header_text, &header_size);
	FILE* row    = open_memstream(&row_text, &row_size);
	WriteCsvHeader(header);
	WriteCsvRow(row);
	std::fclose(header);
	std::fclose(row);
	const std::string h(header_text), r(row_text);
	std::free(header_text);
	std::free(row_text);
	Check(h.rfind(",sig_segv,sig_bus,sig_ill,sig_usr1,sig_other,path_gpu_fault_ok", 0) == 0, "header starts with the signals, then paths");
	Check(h.find(",sig_main_thread,sig_other_threads") == h.size() - std::strlen(",sig_main_thread,sig_other_threads"),
	      "header ends with the thread split");
	size_t header_commas = 0, row_commas = 0;
	for (char c: h) header_commas += c == ',';
	for (char c: r) row_commas += c == ',';
	Check(header_commas == kColumnCount && row_commas == kColumnCount, "one value per column");
	Check(r.substr(0, 11) == ",0,0,1,0,0,", "row: sig_ill=1 in the third signal column");
}

void TestHistogram() {
	ResetHistogram();
	void* page = ::mmap(nullptr, 8192, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
	Check(page != MAP_FAILED, "map");
	auto* bytes = static_cast<uint8_t*>(page);
	std::memset(bytes + 4096, 0xee, 4096); // what a copy running past the page end would pick up
	for (int i = 0; i < 16; i++) bytes[i] = static_cast<uint8_t>(0xa0 + i);
	// Last 3 bytes of the first page: only those may be copied (the next page holds 0xee).
	bytes[4093] = 0x11;
	bytes[4094] = 0x22;
	bytes[4095] = 0x33;
	const auto base = reinterpret_cast<uint64_t>(page);

	RecordRip(base, Path::Sse4a);
	RecordRip(base, Path::Sse4a);
	RecordRip(base, Path::Sse4a);
	RecordRip(base + 4, Path::CpuExtensions);
	RecordRip(base + 4, Path::CpuExtensions);
	RecordRip(base + 8, Path::Rsqrt);
	RecordRip(base + 4093, Path::MonitorxMwaitx);
	RecordRip(0, Path::Sse4a); // ignored

	TopEntry top[16] {};
	Check(TopRips(top, 16) == 4, "four distinct RIPs (rip 0 ignored)");
	Check(top[0].rip == base && top[0].count == 3 && top[0].kind == Path::Sse4a, "most frequent first");
	Check(top[1].rip == base + 4 && top[1].count == 2 && top[1].kind == Path::CpuExtensions, "second");
	Check(top[2].count == 1 && top[3].count == 1, "ties last");
	uint8_t op[kOpcodeBytes];
	std::memcpy(op, &top[0].opcode, sizeof(op));
	Check(op[0] == 0xa0 && op[7] == 0xa7, "eight opcode bytes kept");
	for (size_t i = 0; i < 4; i++) {
		if (top[i].rip == base + 4093) {
			std::memcpy(op, &top[i].opcode, sizeof(op));
			Check(op[0] == 0x11 && op[1] == 0x22 && op[2] == 0x33 && op[3] == 0 && op[7] == 0, "copy stops at the page end (nothing from the next page)");
		}
	}
	TopEntry few[16] {};
	Check(TopRips(few, 2) == 2 && few[0].count == 3 && few[1].count == 2, "n < found keeps the best two");

	// More distinct RIPs than the table can probe: counted as overflow, not lost silently, and nothing crashes.
	ResetHistogram();
	for (uint64_t i = 1; i <= kHistogramSlots * 2; i++) RecordRip(base + (i & 7), Path::Rsqrt);
	TopEntry again[16] {};
	Check(TopRips(again, 16) == 8 && again[0].count > 0, "eight RIPs, many hits each");
	ResetHistogram();
	Check(TopRips(again, 16) == 0, "reset empties the table");
	::munmap(page, 8192);
}

void TestDump() {
	ResetHistogram();
	static std::string text;
	text.clear();
	alignas(8) static uint8_t code[16] = {0x66, 0x0f, 0x78, 0xc2, 0x28, 0x00};
	RecordRip(reinterpret_cast<uint64_t>(code), Path::Sse4a);
	RecordRip(reinterpret_cast<uint64_t>(code), Path::Sse4a);
	DumpHistogram(
	    "unit",
	    [](uint64_t rip, char* module, size_t module_size, uint64_t* offset) {
		    std::snprintf(module, module_size, "eboot.bin");
		    *offset = rip - reinterpret_cast<uint64_t>(code);
		    return true;
	    },
	    [](const char* line) { text += line; });
	Check(text.find("[sigcensus] unit: 1 rips") != std::string::npos, "header line");
	Check(text.find("#1 count=2 kind=path_sse4a at=eboot.bin+0x0 op=66 0f 78 c2 28 00 00 00") != std::string::npos, "entry line");
	ResetHistogram();
}

} // namespace

int main() {
	TestClassification();
	TestCounters();
	TestThreadSplit();
	TestCsv();
	TestHistogram();
	TestDump();
	std::puts("SignalCensusTests: ok");
	return 0;
}
