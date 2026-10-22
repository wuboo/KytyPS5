#ifndef KYTY_GRAPHICS_SHADER_PROGRAMDISKCACHE_H_
#define KYTY_GRAPHICS_SHADER_PROGRAMDISKCACHE_H_

// S434 KYTY_PROGRAM_DISK_CACHE=0|1|verify (default 0): the result of PipelineCache's CompilePermutation (SPIR-V, CompiledShaderInfo, per-vertex
// variants) kept on disk per (program, permutation), so a warm start skips emit / validate / module / PV variants (plan-hitch-2026-10-19, stage 1).
// This file is the format and the key (S442); permutationRecord.h is what goes into a file (S443).
//
// Key of a file = XXH3-128 over
//   fingerprint  KYTY_RECOMPILER_SIGNATURE (sources of the recompiler, no git revision), kProgramDiskFormat, every KYTY_* variable that is set and not
//                in kNonCodegenKytyEnv (knownEnv.h; unknown names are in), config (shader optimization type; with a graphics debug dump or a shader
//                log the cache is off), device (vendor, device, driver version, pipeline cache UUID, subgroup size, maintenance8, mesh shaders),
//   program      ProgramKey (stage, params.hash, user data count, code size, the whole static state) + user data base + wave size + XXH3-128 of the
//                bytes of the code and of the back code (params.hash has 64 bits),
//   permutation  the bytes the caller serializes for (specialization, push data start).
// A file holds magic, format, both keys and a trailing XXH3-64 of everything before it; a file that is short, long, damaged or of another key is
// rejected. Writes are atomic (tmp + rename) and run on a worker thread. The bytes of the guest code are never stored, only their hash.
//
// Mutants (tests/ProgramDiskCacheTests.cpp, KYTY_PROGDISK_MUTANT_*): KEY_IGNORES_ENV, KEY_IGNORES_SIGNATURE, KEY_DROPS_STATIC_STATE,
// ACCEPTS_CORRUPT, ACCEPT_TRUNCATED, NO_RENAME, WRITE_ON_CALLER. S476 (tests/ProgramDiskPrefetchTests.cpp): PROCESS_ENV_NULL, PREFETCH_UNUSED,
// PREFETCH_KEEPS_BYTES, PREFETCH_UNBOUNDED,
// LIST_IGNORES_FINGERPRINT.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace Libs::Graphics::ProgramDisk {

// Bump when the layout of a file (this header or permutationRecord.h) changes.
inline constexpr uint32_t kProgramDiskFormat = 1;

enum class Mode : uint8_t { Off, On, Verify };

// null / "" -> Off, "1" -> On, "verify" -> Verify, anything else -> Off.
[[nodiscard]] Mode ParseMode(const char* text);
[[nodiscard]] Mode EnvMode();

struct Key {
	uint64_t lo = 0;
	uint64_t hi = 0;
	bool operator==(const Key&) const = default;
	[[nodiscard]] std::string Hex() const; // 32 digits
};

class Hasher {
public:
	Hasher();
	~Hasher();
	Hasher(const Hasher&)            = delete;
	Hasher& operator=(const Hasher&) = delete;
	void Add(const void* data, size_t size);
	void AddU32(uint32_t value) { Add(&value, sizeof(value)); }
	void AddU64(uint64_t value) { Add(&value, sizeof(value)); }
	void AddKey(const Key& key) { AddU64(key.lo), AddU64(key.hi); }
	void AddString(std::string_view text); // length-prefixed
	[[nodiscard]] Key Finish();

private:
	void* m_state = nullptr;
};

[[nodiscard]] Key HashBytes(std::span<const uint32_t> words);

// ---- fingerprint ------------------------------------------------------------------------------------------------------------------------

// S476: the environment of this process as a null-terminated environ, the one MakeFingerprint must read (on Windows it used to be nullptr, so no KYTY_*
// variable was in the key there).
[[nodiscard]] const char* const* ProcessEnvironment();

[[nodiscard]] bool IsNonCodegenEnv(std::string_view name);
// "NAME=VALUE" of every KYTY_* entry of `env` (a null-terminated environ) that is not in kNonCodegenKytyEnv, sorted.
[[nodiscard]] std::vector<std::string> CodegenEnvironment(const char* const* env);

struct DeviceInfo {
	uint32_t vendor_id        = 0;
	uint32_t device_id        = 0;
	uint32_t driver_version   = 0;
	uint8_t  cache_uuid[16]   = {};
	uint32_t subgroup_size    = 0;
	bool     maintenance8     = false;
	bool     mesh_shader      = false;
};

struct ConfigInfo {
	int  shader_optimization_type = 0;
	bool debug_dump               = false; // emit_debug_info changes the SPIR-V
	bool shader_log               = false; // the IR dumps need a real compile
};

struct FingerprintInputs {
	std::string_view     recompiler_signature;
	const char* const*   environment = nullptr;
	DeviceInfo           device;
	ConfigInfo           config;
};

struct Fingerprint {
	Key         key;
	bool        enabled = true;
	std::string reason;      // why it is disabled
	std::string description; // for the log line: signature, number of env entries, device
};

[[nodiscard]] Fingerprint MakeFingerprint(const FingerprintInputs& inputs);

// ---- keys --------------------------------------------------------------------------------------------------------------------------------

struct ProgramIdentity {
	uint32_t                  stage           = 0;
	uint64_t                  params_hash     = 0;
	uint32_t                  user_data_count = 0;
	uint32_t                  code_size       = 0;
	std::span<const uint32_t> static_state;
	uint32_t                  user_data_base  = 0;
	uint32_t                  wave_size       = 0;
	std::span<const uint32_t> code;
	std::span<const uint32_t> back_code;
};

[[nodiscard]] Key MakeProgramKey(const Key& fingerprint, const ProgramIdentity& identity);
[[nodiscard]] Key MakePermutationKey(const Key& program_key, std::span<const uint8_t> permutation_input);

// ---- file --------------------------------------------------------------------------------------------------------------------------------

[[nodiscard]] std::vector<uint8_t> SerializeFile(const Key& program, const Key& permutation, std::span<const uint8_t> payload);
[[nodiscard]] bool DeserializeFile(std::span<const uint8_t> data, const Key& program, const Key& permutation, std::vector<uint8_t>* payload);

// <dir>/prog_v1/<stage>_<program16>_<permutation16>.kpp (the full keys are inside the file).
[[nodiscard]] std::filesystem::path FilePath(const std::filesystem::path& dir, uint32_t stage, const Key& program, const Key& permutation);

// nullopt: no file, unreadable or rejected (`rejected` is set for the last case).
[[nodiscard]] std::optional<std::vector<uint8_t>> Load(const std::filesystem::path& path, const Key& program, const Key& permutation,
                                                       bool* rejected = nullptr);
// Synchronous atomic write: tmp file in the same directory, then rename.
[[nodiscard]] bool Save(const std::filesystem::path& path, std::span<const uint8_t> file_bytes);

// S476 (KYTY_PROGRAM_DISK_PREFETCH=0|1, default 0): a worker thread reads prog_v1 files into memory ahead of the first lookup, so the CP thread finds
// the bytes instead of opening the file (the 35-200 ms frames of a warm run were prog_disk_load_ns). Which files: the list of the previous runs of the same
// fingerprint (prog_used_v1.lst, names in order of first use) - the directory also holds files of every other fingerprint, reading it all took 7 s and
// 270 MB in the bench. It reads nothing from the guest: it knows no key, only file names. Load() takes the bytes from the installed Prefetch when it has
// them and reads the file otherwise (never waits for the worker). The bytes still go through the same checks (checksum, both keys, size), so a stale or
// damaged file is refused as before. Memory: bytes read and not yet taken stay under max_bytes (the worker stops reading at the limit).
class Prefetch {
public:
	Prefetch(std::vector<std::filesystem::path> paths, uint64_t max_bytes); // read in this order
	~Prefetch();
	Prefetch(const Prefetch&)            = delete;
	Prefetch& operator=(const Prefetch&) = delete;

	// The bytes of `path` when the worker has read them (and removes them from the store), nullopt otherwise.
	[[nodiscard]] std::optional<std::vector<uint8_t>> Take(const std::filesystem::path& path);
	void WaitDone(); // the worker has seen every file (or hit the limit); for tests

	struct Stats {
		uint64_t files   = 0; // files read by the worker
		uint64_t bytes   = 0; // bytes read by the worker
		uint64_t read_ns = 0; // time the worker spent reading
		uint64_t hit     = 0; // Take answered from the store
		uint64_t miss    = 0; // Take found nothing (not read yet, not listed, or already taken)
	};
	[[nodiscard]] Stats GetStats() const;

private:
	struct State;
	State* m_state;
};

[[nodiscard]] bool PrefetchEnvEnabled(); // KYTY_PROGRAM_DISK_PREFETCH=1
// prog_used_v1.lst: first line = hex of the fingerprint, then one file name (<stage>_<program16>_<permutation16>.kpp) per line, in order of first use.
// ParseUsedList returns no names for another fingerprint and skips every line that is not such a name (no path separators, no "..").
[[nodiscard]] std::string              SerializeUsedList(const Key& fingerprint, std::span<const std::string> names);
[[nodiscard]] std::vector<std::string> ParseUsedList(std::string_view text, const Key& fingerprint);
[[nodiscard]] std::filesystem::path    UsedListPath(const std::filesystem::path& dir);
// Process-wide read-ahead used by Load(); nullptr = none. The caller keeps the object alive until it has been reset to nullptr.
void SetPrefetch(Prefetch* prefetch);
// Nanoseconds the last Load() of the calling thread spent getting the bytes (file read or store lookup), before the checks.
[[nodiscard]] uint64_t LastLoadReadNs();

// One worker thread that writes files (CP only hands the buffer over). Flush() waits for everything queued; the destructor flushes.
class AsyncWriter {
public:
	AsyncWriter();
	~AsyncWriter();
	AsyncWriter(const AsyncWriter&)            = delete;
	AsyncWriter& operator=(const AsyncWriter&) = delete;

	// false: the queue is full (kMaxQueued), the file is dropped.
	bool Enqueue(std::filesystem::path path, std::vector<uint8_t> file_bytes);
	void Flush();

	[[nodiscard]] uint64_t Written() const;
	[[nodiscard]] uint64_t Failed() const;
	[[nodiscard]] uint64_t Bytes() const;
	[[nodiscard]] uint64_t Dropped() const;
	[[nodiscard]] std::thread::id LastWriter() const; // thread of the last write

	static constexpr size_t kMaxQueued = 4096;

	// Test hook, set before the first Enqueue.
	std::function<void(const std::filesystem::path& final_path)> before_rename;

private:
	struct State;
	State* m_state;
};

} // namespace Libs::Graphics::ProgramDisk

#endif /* KYTY_GRAPHICS_SHADER_PROGRAMDISKCACHE_H_ */
