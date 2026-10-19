#ifndef KYTY_GRAPHICS_SHADER_PERMUTATIONRECORD_H_
#define KYTY_GRAPHICS_SHADER_PERMUTATIONRECORD_H_

// S434 (S443) KYTY_PROGRAM_DISK_CACHE, stage 1: what CompilePermutation produced, as it goes into a file of programDiskCache.h.
//
// A record holds the request it answers (resource specialization + push data start: the module-affecting input of the emit), the SPIR-V words,
// the CompiledShaderInfo the renderer keeps (stage metadata, resources, bindings) and the effect of the per-vertex hooks as data, not as a decision:
// the prototype record (capture words, flags) and the VS-store shader the hooks left behind for this program. A hit gives the renderer everything a
// compile would have: it creates the VkShaderModule from the words and imports the per-vertex records under the new program id.
//
// The serializer writes every field by name; the static_asserts in permutationRecord.cpp pin the size of each structure it covers, so a field added to
// one of them stops the build until it is written here (and kProgramDiskFormat is bumped).
//
// Mutants (tests/PermutationRecordTests.cpp, KYTY_PERMREC_MUTANT_*): IGNORE_SPEC_COMPARE, IGNORE_PUSH_START, DROP_PV_RECORD, STALE_BINDINGS,
// DROP_INFO_VECTORS, SKIP_REQUEST_CHECK.

#include "graphics/shader/programDiskCache.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::ProgramDisk {

// What RememberPerVertexPrototypeShader left for the program (the SPIR-V of the original is the permutation's own).
struct PvOriginalRecord {
	uint32_t              stage      = 0;
	bool                  per_vertex = false;
	std::vector<uint32_t> capture_words;
	bool operator==(const PvOriginalRecord&) const = default;
};

// What RememberPvVsStoreShader left for the program.
struct PvStoreRecord {
	uint32_t              stage           = 0;
	std::vector<uint32_t> words;
	uint32_t              mask            = 0;
	uint32_t              corners         = 0;
	uint64_t              hash            = 0;
	uint32_t              compact_spec_id = 0;
	bool operator==(const PvStoreRecord&) const = default;
};

struct PvRecord {
	std::optional<PvOriginalRecord> original;
	std::optional<PvStoreRecord>    store;
	bool operator==(const PvRecord&) const = default;
};

struct PermutationRecord {
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	uint32_t                                     push_data_start_dword = 0;
	std::vector<uint32_t>                        spirv;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	PvRecord                                     pv;
};

// The request alone (what the file name and the permutation key are made of).
[[nodiscard]] std::vector<uint8_t> SerializeRequest(const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                                    uint32_t push_data_start_dword);
[[nodiscard]] std::vector<uint8_t> SerializeRecord(const PermutationRecord& record);
[[nodiscard]] bool                 DeserializeRecord(std::span<const uint8_t> payload, PermutationRecord* record);

// Does the record answer this request? (compared field by field, not by hash)
[[nodiscard]] bool Answers(const PermutationRecord& record, const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                           uint32_t push_data_start_dword);

// Which part of two payloads differs first: "spirv word 12 (a=.. b=..)", "info", "pv.original", "pv.store", "request", "undecodable", "same".
[[nodiscard]] std::string DescribeDifference(std::span<const uint8_t> stored, std::span<const uint8_t> fresh);

// ---- the store on disk ------------------------------------------------------------------------------------------------------------------

struct Request {
	Key                   program;
	Key                   permutation;
	uint32_t              stage = 0;
	std::filesystem::path path;
};

[[nodiscard]] Request MakeRequest(const std::filesystem::path& dir, const Key& fingerprint, const ProgramIdentity& identity,
                                  const ShaderRecompiler::IR::ResourceSpecialization& specialization, uint32_t push_data_start_dword);

enum class Found : uint8_t {
	Miss,     // no file
	Rejected, // a file that is damaged, of another format or key, or does not decode, or answers another request
	Hit,
};

[[nodiscard]] Found Find(const Request& request, const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                         uint32_t push_data_start_dword, PermutationRecord* record);

// The bytes of the file for the record (the caller hands them to an AsyncWriter).
[[nodiscard]] std::vector<uint8_t> MakeFileBytes(const Request& request, const PermutationRecord& record);

enum class Verdict : uint8_t { NoFile, Equal, NotEqual, Rejected };

// verify: compares the fresh record with the stored file byte for byte; `difference` gets DescribeDifference for NotEqual.
[[nodiscard]] Verdict VerifyAgainst(const Request& request, const PermutationRecord& fresh, std::string* difference);

} // namespace Libs::Graphics::ProgramDisk

#endif /* KYTY_GRAPHICS_SHADER_PERMUTATIONRECORD_H_ */
