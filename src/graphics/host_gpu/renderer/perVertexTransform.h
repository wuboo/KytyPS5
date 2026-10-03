#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

constexpr const char* kPerVertexCacheMagic       = "KYTYPV02";
constexpr uint32_t    kPerVertexTransformVersion = 2;

std::string GetPerVertexTransformSignature();

// Records [first, end) of the vertex buffer that per_vertex_unpack.comp can read for a draw
// (VertexIndex(): first_vertex + local for non-indexed draws, index + vertex_offset for indexed
// ones; records at or past num_records are dropped by the shader). Returns false when the range
// cannot be derived with certainty (unsupported index size, short index data, arithmetic that
// would overflow the shader's 32-bit math); the caller then has to use the whole buffer.
struct PerVertexRecordRange {
	uint64_t first = 0;
	uint64_t end   = 0; // first == end: the draw reads no vertex data
};
bool ComputePerVertexRecordRange(bool indexed, const void* index_data, uint64_t index_bytes,
                                 uint32_t index_element_size, uint32_t index_count,
                                 uint32_t first_vertex, int32_t vertex_offset,
                                 uint32_t num_records, PerVertexRecordRange& range);

// Memo of ComputePerVertexRecordRange for indexed draws. The range depends only on the index
// bytes (identified by their 64-bit content hash, which the capture cache key needs anyway) and
// the draw parameters below, so a mesh drawn again with unchanged indices skips the scan of every
// index. Entries compare all fields, so a hash-bucket collision is a miss, never a wrong range.
class PerVertexRangeMemo {
public:
	struct Key {
		uint64_t index_hash         = 0;
		uint64_t index_bytes        = 0;
		uint32_t index_element_size = 0;
		uint32_t index_count        = 0;
		int32_t  vertex_offset      = 0;
		uint32_t num_records        = 0;
		bool     operator==(const Key&) const = default;
	};
	// `found` is set to whether the key was known; the result is only valid then. A key whose
	// range is not derivable is memoized too (derivable == false), so the caller falls back to the
	// whole buffer without rescanning.
	struct Result {
		bool                 found      = false;
		bool                 derivable  = false;
		PerVertexRecordRange range;
	};
	[[nodiscard]] Result Find(const Key& key) const;
	void                 Insert(const Key& key, bool derivable, const PerVertexRecordRange& range);
	[[nodiscard]] size_t Size() const { return m_entries.size(); }

	static constexpr size_t kMaxEntries = 8192;

private:
	struct Entry {
		Key                  key;
		bool                 derivable = false;
		PerVertexRecordRange range;
	};
	[[nodiscard]] static uint64_t Bucket(const Key& key);
	std::unordered_map<uint64_t, Entry> m_entries;
};

struct PerVertexLayout {
	uint32_t                                   num_params         = 0;
	bool                                       has_clip           = false;
	uint32_t                                   clip_slot          = UINT32_MAX;
	uint32_t                                   record_stride_vec4 = 0;
	std::vector<std::pair<uint32_t, uint32_t>> param_locations_and_slots; // (location, slot)
	std::map<uint32_t, uint32_t>               location_to_slot;
};

bool DerivePerVertexLayout(const std::string& vs_source, const std::string& ps_source,
                           PerVertexLayout& layout, std::map<uint32_t, std::string>& vs_param_vars);
bool DerivePerVertexVertexLayout(const std::string& vs_source, PerVertexLayout& layout,
                                 std::map<uint32_t, std::string>& vs_param_vars);
std::string GenerateReplayVertexSpvasm(const PerVertexLayout& layout);
std::string LowerVertexToCompute(const std::string& source, const PerVertexLayout& layout,
                                 const std::map<uint32_t, std::string>& vs_param_vars);
std::string LowerFragmentToBufferReplay(const std::string& source, const PerVertexLayout& layout);

bool TryLoadTransformedShadersFromDisk(const std::filesystem::path& cache_dir, uint64_t vs_hash,
                                       uint64_t ps_hash, PerVertexLayout& layout,
                                       std::vector<uint32_t>& cap_words,
                                       std::vector<uint32_t>& frag_words,
                                       std::vector<uint32_t>& replay_words);

bool TryLoadTransformedShadersFromDisk(const std::string& title_id, uint64_t vs_hash,
                                       uint64_t ps_hash, PerVertexLayout& layout,
                                       std::vector<uint32_t>& cap_words,
                                       std::vector<uint32_t>& frag_words,
                                       std::vector<uint32_t>& replay_words);

bool SaveTransformedShadersToDisk(const std::filesystem::path& cache_dir, uint64_t vs_hash,
                                  uint64_t ps_hash, const PerVertexLayout& layout,
                                  std::span<const uint32_t> cap_words,
                                  std::span<const uint32_t> frag_words,
                                  std::span<const uint32_t> replay_words);

bool SaveTransformedShadersToDisk(const std::string& title_id, uint64_t vs_hash, uint64_t ps_hash,
                                  const PerVertexLayout&    layout,
                                  std::span<const uint32_t> cap_words,
                                  std::span<const uint32_t> frag_words,
                                  std::span<const uint32_t> replay_words);

} // namespace Libs::Graphics
