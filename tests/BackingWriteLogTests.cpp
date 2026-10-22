// S479: the log of backing writes (kernel/backingWriteLog.h): a worker that started at seq0 must be told when a write touched a page it read. Every check fails under one of the
// KYTY_BWLOG_MUTANT_* defines (CMake: backing_write_log_mutant_*, WILL_FAIL).

#include "kernel/backingWriteLog.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace {

using Libs::LibKernel::BackingWriteLog;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "BackingWriteLogTests: failed: %s\n", text);
		std::exit(1);
	}
}

} // namespace

int main() {
	const uint64_t page_a[] = {0x10000};
	const uint64_t page_b[] = {0x20000};
	{
		BackingWriteLog log;
		const auto      seq0 = log.Seq();
		Check(!log.WrittenSince(seq0, page_a, 1), "empty log: nothing written");
		// a write that began and ended before the reader started is no danger
		const auto old_write = log.Begin(0x10000, 4096);
		log.End(old_write);
		const auto seq1 = log.Seq();
		Check(!log.WrittenSince(seq1, page_a, 1), "a write that ended before the reader started");
		// a write that ends after the reader started is
		const auto write = log.Begin(0x10800, 16);
		log.End(write);
		Check(log.WrittenSince(seq1, page_a, 1), "a write after the start into the page");
		Check(!log.WrittenSince(seq1, page_b, 1), "a write after the start into another page");
	}
	{
		// a write that has not ended (the reader may see half of it)
		BackingWriteLog log;
		const auto      running = log.Begin(0x10000, 64);
		const auto      seq0    = log.Seq();
		Check(log.WrittenSince(seq0, page_a, 1), "a write in progress when the reader started");
		log.End(running);
		Check(log.WrittenSince(seq0, page_a, 1), "the write that was in progress ended after the reader started");
	}
	{
		// the ring wraps: older writes are forgotten, the log must not claim there were none
		BackingWriteLog log;
		const auto      seq0 = log.Seq();
		for (size_t i = 0; i < BackingWriteLog::kSize + 10; i++) {
			const auto token = log.Begin(0x90000 + i * 4096, 16); // never the pages asked about
			log.End(token);
		}
		Check(log.WrittenSince(seq0, page_a, 1), "a wrapped ring cannot tell: yes");
		Check(!log.WrittenSince(log.Seq(), page_a, 1), "a reader that started after everything is safe");
	}
	std::puts("BackingWriteLogTests: ok");
	return 0;
}
