#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTMEMO_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTMEMO_H_

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {

// S324 KYTY_SRT_FAST: Off = the walk runs on every draw; On = a stage whose user data and every guest read of its
// previous walk are unchanged reuses the previous snapshot; Shadow = walk anyway, compare with what On would reuse.
enum class SrtFastMode : uint8_t { Off, On, Shadow };
enum class SrtMemoOutcome : uint8_t { None, Hit, Miss, Mismatch };

inline SrtFastMode ParseSrtFastMode(const char* text) {
	if (text == nullptr) return SrtFastMode::Off;
	if (std::strcmp(text, "1") == 0) return SrtFastMode::On;
	if (std::strcmp(text, "shadow") == 0) return SrtFastMode::Shadow;
	return SrtFastMode::Off;
}

using SrtMemoReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> values);

// What one walk observed: its inputs (key) and every guest read it issued, with the value or the failure.
// A later walk with the same key whose reads all return the same results would produce the same snapshot.
// NewestWriteTick does not do this job: CPU writes of the guest are not tracked.
struct SrtMemo {
	struct Read {
		uint64_t address  = 0;
		uint32_t first    = 0; // into words
		uint32_t count    = 0;
		bool     ok       = false;
		bool     ordinary = false; // issued through read_memory (else the specialization reader)
	};

	bool                  valid = false;
	const void*           plan  = nullptr;
	uint64_t              shader_base = 0;
	uint64_t              indirect_args = 0; // S425: compute indirect dispatch args address (the walk reads the counts there)
	bool                  min_lod_remap = false;
	bool                  has_ordinary = false;
	std::vector<uint32_t> user_data;
	std::vector<uint32_t> workgroup_counts;
	std::vector<Read>     reads;
	std::vector<uint32_t> words;

	void Clear() {
		valid = false;
		reads.clear();
		words.clear();
	}

	void Begin(const void* plan_id, std::span<const uint32_t> user, uint64_t base, std::span<const uint32_t> groups,
	           uint64_t indirect, bool remap, bool ordinary_reader) {
		Clear();
		plan             = plan_id;
		shader_base      = base;
		indirect_args    = indirect;
		min_lod_remap    = remap;
		has_ordinary     = ordinary_reader;
		user_data.assign(user.begin(), user.end());
		workgroup_counts.assign(groups.begin(), groups.end());
	}

	void Record(bool ordinary, uint64_t address, std::span<const uint32_t> values, bool ok) {
#ifdef KYTY_SRTMEMO_MUTANT_NO_FAILED
		if (!ok) return;
#endif
		Read read {.address = address, .first = static_cast<uint32_t>(words.size()),
		           .count = static_cast<uint32_t>(values.size()), .ok = ok, .ordinary = ordinary};
		if (ok) words.insert(words.end(), values.begin(), values.end());
		reads.push_back(read);
	}

	bool KeyMatches(const void* plan_id, std::span<const uint32_t> user, uint64_t base,
	                std::span<const uint32_t> groups, uint64_t indirect, bool remap, bool ordinary_reader) const {
#ifdef KYTY_SRTMEMO_MUTANT_IGNORES_PLAN
		(void)plan_id;
		const bool same_plan = true;
#else
		const bool same_plan = plan == plan_id;
#endif
#ifdef KYTY_SRTMEMO_MUTANT_IGNORES_INDIRECT
		(void)indirect;
		const bool same_indirect = true;
#else
		const bool same_indirect = indirect_args == indirect;
#endif
#ifdef KYTY_SRTMEMO_MUTANT_IGNORES_USERDATA
		const bool same_user = true;
#else
		const bool same_user = std::ranges::equal(user_data, user);
#endif
		return valid && same_plan && shader_base == base && same_indirect && min_lod_remap == remap &&
		       has_ordinary == ordinary_reader && same_user && std::ranges::equal(workgroup_counts, groups);
	}

	// Reads every logged address again; true when each returns what it returned during the walk.
	bool ReadsUnchanged(SrtMemoReader strict, SrtMemoReader ordinary, void* userdata) const {
		std::array<uint32_t, 32> small_buf; // not "small": windows.h defines it as a macro
		std::vector<uint32_t>    large;
		for (const auto& read: reads) {
			const auto reader = read.ordinary ? ordinary : strict;
			if (reader == nullptr) return false;
			std::span<uint32_t> out;
			if (read.count <= small_buf.size()) {
				out = std::span(small_buf).first(read.count);
			} else {
				large.resize(read.count);
				out = large;
			}
			if (reader(userdata, read.address, out) != read.ok) return false;
#ifndef KYTY_SRTMEMO_MUTANT_ADDR_ONLY
			if (read.ok && !std::ranges::equal(out, std::span(words).subspan(read.first, read.count))) return false;
#endif
		}
		return true;
	}
};

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SRTMEMO_H_ */
