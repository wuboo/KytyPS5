#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MATMEMOSHADOW_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_MATMEMOSHADOW_H_

// S406 (KYTY_DIAG_MATMEMO=1, DIAG, default off): shadow of a memo "after the walk" for shader resource materialization.
// Nothing here changes behavior or feeds a cache key. For every ProgramCache::Get that reaches MaterializeResources the shadow
// is told the INPUT of the part after the walk (the user_data words the plan or its programs use, the flattened SRT the walk
// produced, workgroup counts, the min-LOD remap flag, the push-data cursor) and the OUTPUT (snapshot buffers / images / samplers /
// used user_data / flattened SRT / uniform fill, the specialization and the index of the chosen permutation), each as a 64-bit hash.
// A per-SourceEntry MRU-4 list then says what a memo keyed by the input would have done:
//   in_hit   an entry with the same input exists (the memo would skip everything after the walk)
//   unsafe   ... and its output differs (MUST stay 0: the input would not determine the output)
//   out_hit  some entry has the same output (any input): the result-level hit
//   rebase   no entry has the same output, but one has the same output once the Base48 of the buffer descriptors is masked
//            (the memo could patch the bases of the snapshot and keep the specialization / permutation)
// tests/MatMemoShadowTests.cpp pins it.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::MatMemo {

inline uint64_t Mix(uint64_t hash, uint64_t value) {
	hash ^= value + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
	hash *= 0xff51afd7ed558ccdull;
	hash ^= hash >> 33;
	return hash;
}
inline uint64_t HashWords(uint64_t seed, std::span<const uint32_t> words) {
	uint64_t hash = Mix(seed, words.size());
	for (const auto word: words) hash = Mix(hash, word);
	return hash;
}

struct InputParts {
	std::span<const uint32_t> user_data;      // the runtime's user_data
	uint64_t                  used_mask = 0;  // bit i = user_data[i] is used by the plan or one of its programs
	std::span<const uint32_t> flat;           // flattened SRT the walk produced
	std::span<const uint32_t> workgroup_counts;
	bool                      min_lod_remap = false;
	uint32_t                  push_cursor   = 0;
};

inline uint64_t InputKey(const InputParts& parts) {
	uint64_t hash = 0x6d61746d656d6f01ull;
	for (size_t i = 0; i < parts.user_data.size() && i < 64; i++) {
		if (((parts.used_mask >> i) & 1u) != 0) hash = Mix(hash, (static_cast<uint64_t>(i) << 32) | parts.user_data[i]);
	}
#if !defined(KYTY_MATMEMO_MUTANT_IGNORE_FLAT)
	hash = HashWords(hash, parts.flat);
#endif
	hash = HashWords(hash, parts.workgroup_counts);
	hash = Mix(hash, parts.min_lod_remap ? 1 : 0);
	hash = Mix(hash, parts.push_cursor);
	return hash;
}

struct Output {
	uint64_t full   = 0;
	uint64_t masked = 0; // the same with the Base48 of buffer descriptors zeroed
};

struct Result {
	bool in_hit  = false;
	bool unsafe  = false;
	bool out_hit = false;
	bool rebase  = false;
};

// MRU-4 list of (input, output) pairs of one SourceEntry.
class Shadow {
public:
	static constexpr size_t kWays = 4;

	Result Note(uint64_t input, Output output) {
		Result result;
		for (size_t i = 0; i < m_count; i++) {
			if (m_entries[i].output.full == output.full) result.out_hit = true;
		}
		if (!result.out_hit) {
			for (size_t i = 0; i < m_count; i++) {
				if (m_entries[i].output.masked == output.masked) result.rebase = true;
			}
		}
		size_t found = m_count;
		for (size_t i = 0; i < m_count; i++) {
			if (m_entries[i].input == input) {
				found = i;
				break;
			}
		}
		if (found < m_count) {
			result.in_hit = true;
			result.unsafe = m_entries[found].output.full != output.full;
			// Most recently used first; the entry now holds the newest output for this input.
			std::rotate(m_entries.begin(), m_entries.begin() + static_cast<std::ptrdiff_t>(found),
			            m_entries.begin() + static_cast<std::ptrdiff_t>(found) + 1);
			m_entries[0].output = output;
		} else {
			if (m_count < kWays) m_count++;
			std::rotate(m_entries.begin(), m_entries.begin() + static_cast<std::ptrdiff_t>(m_count) - 1,
			            m_entries.begin() + static_cast<std::ptrdiff_t>(m_count));
			m_entries[0] = {input, output};
		}
		return result;
	}
	[[nodiscard]] size_t Size() const { return m_count; }

private:
	struct Entry {
		uint64_t input = 0;
		Output   output;
	};
	std::array<Entry, kWays> m_entries {};
	size_t                   m_count = 0;
};

// Per-program tally for the top-N log.
class Report {
public:
	struct Row {
		uint64_t hash   = 0;
		uint32_t stage  = 0; // 0 vs, 1 ps, 2 cs
		uint64_t n      = 0;
		uint64_t in_hit = 0;
		uint64_t unsafe = 0;
		[[nodiscard]] uint64_t Misses() const { return n - in_hit; }
	};
	void Add(uint64_t hash, uint32_t stage, const Result& result) {
		auto& row = m_rows[hash ^ (static_cast<uint64_t>(stage) << 62)];
		row.hash  = hash;
		row.stage = stage;
		row.n++;
		row.in_hit += result.in_hit ? 1 : 0;
		row.unsafe += result.unsafe ? 1 : 0;
	}
	[[nodiscard]] std::vector<Row> TopMisses(size_t count) const {
		std::vector<Row> rows;
		rows.reserve(m_rows.size());
		for (const auto& [key, row]: m_rows) rows.push_back(row);
		std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
			return a.Misses() != b.Misses() ? a.Misses() > b.Misses() : a.hash < b.hash;
		});
		if (rows.size() > count) rows.resize(count);
		return rows;
	}
	void Clear() { m_rows.clear(); }

private:
	std::unordered_map<uint64_t, Row> m_rows;
};

} // namespace Libs::Graphics::MatMemo

#endif
