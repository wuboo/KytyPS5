#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSCOUTVALIDATE_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_CPSCOUTVALIDATE_H

// S471: what the CP checks before it uses a GetGraphicsPrograms result the scout worker computed. The worker's walk recorded every guest read it issued (SrtMemo:
// address, count, words, strict or ordinary) and every vertex-table read (ScoutTableLog). The result is a function of the registers and of those reads, so
// it is the CP's own result when the registers match and every read returns the bytes it returned on the worker.
//  strict   reads (specialization / condition reads): the CP's reader drains GPU-dirty pages and, with KYTY_SPEC_COND=superset, fails conditions on them; so the
//           check reads only quiet pages (strict returns false otherwise: Busy) and compares.
//  ordinary reads (the flat SRT walk): the CP reads them raw, so the check does too.
// Pure part; tests/CpScoutValidateTests.cpp pins it. Mutants (WILL_FAIL): KYTY_CPSCOUT_MUTANT_VALUES_NOT_COMPARED, _FIRST_READ_ONLY, _BUSY_OK, _TABLES_IGNORED.

#include "graphics/shader/recompiler/ir/SrtMemo.h"
#include "graphics/shader/scoutTableLog.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::CpScout {

enum class Verdict : uint8_t { Ok, NoMemo, Busy, Changed };

// strict(address, span) / ordinary(address, span) -> bool (false: could not read). `reads` (optional) gets the number of reads replayed.
template <class Strict, class Ordinary>
Verdict ReadSetVerdict(const ShaderRecompiler::IR::SrtMemo& memo, Strict&& strict, Ordinary&& ordinary, uint32_t* reads = nullptr) {
	if (!memo.valid) return Verdict::NoMemo;
	std::array<uint32_t, 32> buffer {};
	std::vector<uint32_t>    large;
	uint32_t                 replayed = 0;
	for (const auto& read: memo.reads) {
		if (!read.ok) return Verdict::NoMemo;
		std::span<uint32_t> out;
		if (read.count <= buffer.size()) {
			out = std::span(buffer).first(read.count);
		} else {
			large.resize(read.count);
			out = large;
		}
		const bool got = read.ordinary ? ordinary(read.address, out) : strict(read.address, out);
		++replayed;
#ifdef KYTY_CPSCOUT_MUTANT_BUSY_OK
		if (!got) continue;
#else
		if (!got) {
			if (reads != nullptr) *reads += replayed;
			return read.ordinary ? Verdict::Changed : Verdict::Busy;
		}
#endif
#ifndef KYTY_CPSCOUT_MUTANT_VALUES_NOT_COMPARED
		if (got && !std::equal(out.begin(), out.end(), memo.words.begin() + read.first)) {
			if (reads != nullptr) *reads += replayed;
			return Verdict::Changed;
		}
#endif
#ifdef KYTY_CPSCOUT_MUTANT_FIRST_READ_ONLY
		break;
#endif
	}
	if (reads != nullptr) *reads += replayed;
	return Verdict::Ok;
}

// read(site, address, dst, dwords): the CP's own table reader.
template <class Reader>
bool TablesUnchanged(const ScoutTableLog::Log& log, Reader&& read) {
#ifdef KYTY_CPSCOUT_MUTANT_TABLES_IGNORED
	(void)log;
	(void)read;
	return true;
#else
	if (log.unsafe) return false;
	std::vector<uint32_t> now;
	for (const auto& entry: log.reads) {
		now.assign(entry.dwords, 0u);
		read(entry.site, entry.address, now.data(), entry.dwords);
		if (!std::equal(now.begin(), now.end(), log.words.begin() + entry.first)) return false;
	}
	return true;
#endif
}

} // namespace Libs::Graphics::CpScout

#endif
