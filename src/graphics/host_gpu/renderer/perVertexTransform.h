#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

constexpr const char* kPerVertexCacheMagic       = "KYTYPV02";
constexpr uint32_t    kPerVertexTransformVersion = 2;

// `unique`: the signature of the KYTY_PV_UNIQUE replay shaders (S134), kept in separate cache files.
std::string GetPerVertexTransformSignature(bool unique = false);

// Records [first, end) of the vertex buffer that per_vertex_unpack.comp can read for a draw
// (VertexIndex(): first_vertex + local for non-indexed draws, index + vertex_offset for indexed
// ones; records at or past num_records are dropped by the shader). Returns false when the range
// cannot be derived with certainty (unsupported index size, short index data, arithmetic that
// would overflow the shader's 32-bit math); the caller then has to use the whole buffer.
struct PerVertexRecordRange {
	uint64_t first = 0;
	uint64_t end   = 0; // first == end: the draw reads no vertex data
	bool dropped = false; // some index (or first_vertex + local) fell outside [0, num_records)
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

enum class PvLayoutFastMode : uint8_t { Off, On, Shadow };

bool DerivePerVertexLayout(const std::string& vs_source, const std::string& ps_source,
                           PerVertexLayout& layout, std::map<uint32_t, std::string>& vs_param_vars);
bool DerivePerVertexVertexLayout(const std::string& vs_source, PerVertexLayout& layout,
                                 std::map<uint32_t, std::string>& vs_param_vars);
// S432 (KYTY_PV_LAYOUT_FAST): the same layout from the SPIR-V words of the vertex shader, without the text of the disassembly. Output variables are
// the OpVariable ... Output without an initializer, their location the first OpDecorate Location, their name the OpName (what the friendly-names
// disassembler prints), so the result equals DerivePerVertexVertexLayout(Disassemble(words)).
// S437: false with *ambiguous = true when two Output variables carry the same friendly name (the disassembler makes them distinct with a suffix,
// FriendlyName does not, so the std::set would merge them into one parameter and the record stride would be wrong): the caller must use the
// disassembly route for that shader.
bool DerivePerVertexVertexLayoutFromSpirv(std::span<const uint32_t> words, PerVertexLayout& layout,
                                          std::map<uint32_t, std::string>& vs_param_vars, bool* ambiguous = nullptr);
// S437 (KYTY_PV_LAYOUT_FAST=1): the layout from the words, or from the disassembly when the words are ambiguous (*fell_back = true).
// `disassemble` fills the friendly-names text of the same module; returns false on failure (then nothing is derived).
bool DerivePerVertexVertexLayoutFast(std::span<const uint32_t> words, const std::function<bool(std::string*)>& disassemble, PerVertexLayout& layout,
                                     std::map<uint32_t, std::string>& vs_param_vars, bool* fell_back);
// S437 KYTY_PV_SKIP_VALIDATE=0|1 (default 0, bench only): "1" skips spvtools Validate of our own per-vertex variants. null/""/"0"/anything else -> false.
// Independent of KYTY_PV_LAYOUT_FAST (before S437 LAYOUT_FAST=1 skipped Validate too).
bool ParsePvSkipValidate(const char* text);
// Whether Validate runs for a variant: !skip_env, whatever the layout mode.
bool PvShouldValidate(PvLayoutFastMode mode, const char* skip_validate_env);

// null/""/"0" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
PvLayoutFastMode ParsePvLayoutFastMode(const char* text);
// `unique` (S134): the replay reads captured[map[vertex index].x] instead of captured[vertex index];
// the map is binding 2 of the extra set (one 8-byte entry per corner).
std::string GenerateReplayVertexSpvasm(const PerVertexLayout& layout, bool unique = false);
std::string LowerVertexToCompute(const std::string& source, const PerVertexLayout& layout,
                                 const std::map<uint32_t, std::string>& vs_param_vars);
std::string LowerFragmentToBufferReplay(const std::string& source, const PerVertexLayout& layout,
                                        bool unique = false);

bool TryLoadTransformedShadersFromDisk(const std::filesystem::path& cache_dir, uint64_t vs_hash,
                                       uint64_t ps_hash, PerVertexLayout& layout,
                                       std::vector<uint32_t>& cap_words,
                                       std::vector<uint32_t>& frag_words,
                                       std::vector<uint32_t>& replay_words, bool unique = false);

bool TryLoadTransformedShadersFromDisk(const std::string& title_id, uint64_t vs_hash,
                                       uint64_t ps_hash, PerVertexLayout& layout,
                                       std::vector<uint32_t>& cap_words,
                                       std::vector<uint32_t>& frag_words,
                                       std::vector<uint32_t>& replay_words, bool unique = false);

bool SaveTransformedShadersToDisk(const std::filesystem::path& cache_dir, uint64_t vs_hash,
                                  uint64_t ps_hash, const PerVertexLayout& layout,
                                  std::span<const uint32_t> cap_words,
                                  std::span<const uint32_t> frag_words,
                                  std::span<const uint32_t> replay_words, bool unique = false);

bool SaveTransformedShadersToDisk(const std::string& title_id, uint64_t vs_hash, uint64_t ps_hash,
                                  const PerVertexLayout&    layout,
                                  std::span<const uint32_t> cap_words,
                                  std::span<const uint32_t> frag_words,
                                  std::span<const uint32_t> replay_words, bool unique = false);

} // namespace Libs::Graphics
