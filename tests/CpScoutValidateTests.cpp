// S471: what the CP checks before it uses a result the scout worker computed (cpScoutValidate.h): every recorded read returns the recorded bytes,
// strict reads need a quiet page, the vertex tables are read again. Each mutant (WILL_FAIL) breaks one of those checks.

#include "graphics/guest_gpu/command_processor/cpScoutValidate.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <span>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CpScout;
using Libs::Graphics::ShaderRecompiler::IR::SrtMemo;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpScoutValidateTests: failed: %s\n", text);
		std::exit(1);
	}
}

using Memory = std::map<uint64_t, uint32_t>;

SrtMemo MakeMemo() {
	SrtMemo memo;
	memo.valid = true;
	const std::vector<uint32_t> a {1, 2};
	const std::vector<uint32_t> b {3};
	const std::vector<uint32_t> c {4};
	memo.Record(false, 0x1000, a, true);
	memo.Record(true, 0x2000, b, true);
	memo.Record(false, 0x3000, c, true);
	return memo;
}

Memory MakeMemory() {
	return {{0x1000, 1}, {0x1004, 2}, {0x2000, 3}, {0x3000, 4}};
}

bool Read(const Memory& memory, uint64_t address, std::span<uint32_t> out) {
	for (size_t i = 0; i < out.size(); ++i) {
		const auto it = memory.find(address + i * 4u);
		if (it == memory.end()) return false;
		out[i] = it->second;
	}
	return true;
}

Verdict Run(const SrtMemo& memo, const Memory& memory, bool busy, uint32_t* reads = nullptr) {
	return ReadSetVerdict(
	    memo, [&](uint64_t address, std::span<uint32_t> out) { return !busy && Read(memory, address, out); },
	    [&](uint64_t address, std::span<uint32_t> out) { return Read(memory, address, out); }, reads);
}

} // namespace

int main() {
	const auto memo = MakeMemo();
	{
		uint32_t reads = 0;
		Check(Run(memo, MakeMemory(), false, &reads) == Verdict::Ok, "unchanged reads validate");
		Check(reads == 3, "every read is replayed");
	}
	{
		auto memory = MakeMemory();
		memory[0x3000] = 5; // the last read, a strict one
		Check(Run(memo, memory, false) == Verdict::Changed, "a changed last read fails");
	}
	{
		auto memory = MakeMemory();
		memory[0x1004] = 9; // inside the first read
		Check(Run(memo, memory, false) == Verdict::Changed, "a changed word of the first read fails");
	}
	{
		auto memory = MakeMemory();
		memory[0x2000] = 7; // the ordinary read
		Check(Run(memo, memory, false) == Verdict::Changed, "a changed ordinary read fails");
	}
	Check(Run(memo, MakeMemory(), true) == Verdict::Busy, "a strict read of a busy page is not validated");
	{
		auto bad  = memo;
		bad.valid = false;
		Check(Run(bad, MakeMemory(), false) == Verdict::NoMemo, "no memo, no validation");
	}
	{
		SrtMemo failed;
		failed.valid = true;
		failed.Record(false, 0x1000, std::vector<uint32_t> {0}, false);
		Check(Run(failed, MakeMemory(), false) == Verdict::NoMemo, "a failed read cannot be replayed");
	}
	{
		ScoutTableLog::Log log;
		log.reads.push_back({0x5000, 2, 1, 0});
		log.words = {10, 11};
		Memory memory {{0x5000, 10}, {0x5004, 11}};
		const auto reader = [&](uint32_t, uint64_t address, uint32_t* dst, uint32_t dwords) { Read(memory, address, std::span(dst, dwords)); };
		Check(TablesUnchanged(log, reader), "unchanged tables validate");
		memory[0x5004] = 12;
		Check(!TablesUnchanged(log, reader), "a changed table fails");
		memory[0x5004] = 11;
		log.unsafe = true;
		Check(!TablesUnchanged(log, reader), "a table the log cannot replay fails");
	}
	std::puts("CpScoutValidateTests: ok");
	return 0;
}
