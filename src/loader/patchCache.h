#ifndef KYTY_LOADER_PATCH_CACHE_H_
#define KYTY_LOADER_PATCH_CACHE_H_

// S432 KYTY_PATCH_CACHE=0|1|verify: the result of patching the executable segments of a module (AMD CPU patcher, red-zone patcher, 256-bit store
// split) kept on disk, so a start does not decode the whole eboot with Zydis again (~11 s of the start of a Rosetta run).
//
// What is cached is the effect, not the decision: the byte ranges of the segments the patchers changed (original and patched bytes), the
// trampoline area they filled, the used size of the patcher's trampoline generator, the wide-store trap table and the counters of the log lines.
// The key is a hash of every input of the patchers: the bytes of the executable segments, the function starts, the addresses (trampolines hold
// absolute and rel32 targets), the flags, the host features and PatchVersion. Bump PatchVersion whenever the patchers change their output.
// A hit is applied only if the live bytes under every patched range still equal the stored original bytes; otherwise it is dropped and the
// module is patched from scratch. verify patches normally and compares the result with the stored entry byte for byte.
// The cache directory belongs to the variant (like _ShaderCache); it holds game code and never goes to git or a RESULT.
//
// Mutants (tests/PatchCacheTests.cpp): KYTY_PATCHCACHE_MUTANT_{CACHE_IGNORES_KEY, APPLY_SKIPS_TRAMPOLINE, APPLY_NO_ORIGINAL_CHECK,
// CAPTURE_DROPS_LAST_RANGE, SERIALIZE_DROPS_TRAPS, CHECKSUM_IGNORED, SIGNATURE_IGNORED}.

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Loader::PatchCache {

// Changing what the patchers write (any change in guestInstructionPatcher.cpp, x64InstructionEmulator.cpp, redZonePatcher) requires a new value.
inline constexpr uint32_t PatchVersion = 1;

enum class Mode : uint8_t { Off, On, Verify };

// null/"" -> On (the default), "0" -> Off, "verify" -> Verify, anything else -> Off.
[[nodiscard]] Mode ParseMode(const char* text);
[[nodiscard]] Mode EnvMode();

struct Key {
	uint64_t lo = 0;
	uint64_t hi = 0;
	bool operator==(const Key&) const = default;
	[[nodiscard]] std::string Hex() const;
};

class KeyBuilder {
public:
	KeyBuilder();
	~KeyBuilder();
	KeyBuilder(const KeyBuilder&)            = delete;
	KeyBuilder& operator=(const KeyBuilder&) = delete;
	void Add(const void* data, size_t size);
	void AddU64(uint64_t value);
	// S437: length and bytes of the build signature (see BuildSignature).
	void AddSignature(std::string_view signature);
	[[nodiscard]] Key Finish();

private:
	void* m_state = nullptr;
};

// S437: KYTY_PATCHER_SIGNATURE of this build (SHA256 of the patcher sources and the Zydis / xbyak pins, src/patcher_signature.cmake), "unknown" when
// the build does not have one. It is part of every key, so an entry written by a build with other patchers is never applied: PatchVersion is no
// longer the only guard. A build without a signature must not use the cache (the caller checks).
[[nodiscard]] std::string_view BuildSignature();
// Adds the signature of this build, or `signature` when given (tests).
void AddBuildSignature(KeyBuilder& key);
void AddBuildSignature(KeyBuilder& key, std::string_view signature);

struct Range {
	uint64_t             addr = 0;
	std::vector<uint8_t> original;
	std::vector<uint8_t> patched;
	bool                 operator==(const Range&) const = default;
};

struct Entry {
	std::vector<Range>                         ranges;
	uint64_t                                   trampoline_addr = 0;
	std::vector<uint8_t>                       trampoline;
	uint64_t                                   patcher_bytes = 0; // used size of the patcher's trampoline generator
	std::vector<std::pair<uint64_t, uint64_t>> wide_traps;        // (site, trampoline) of the 256-bit store split
	std::vector<uint64_t>                      stats;             // counters for the log lines, opaque here
	bool                                       operator==(const Entry&) const = default;
};

// A copy of the executable segments taken before the patchers run.
class Snapshot {
public:
	Snapshot(std::span<const std::pair<uint64_t, uint64_t>> segments);
	// The ranges that differ now, plus [trampoline_addr, trampoline_addr + trampoline_used).
	[[nodiscard]] Entry Capture(uint64_t trampoline_addr, uint64_t trampoline_used) const;

private:
	struct Segment {
		uint64_t             addr = 0;
		std::vector<uint8_t> bytes;
	};
	std::vector<Segment> m_segments;
};

// True if every patched range of the entry still holds its original bytes in memory.
[[nodiscard]] bool OriginalsMatch(const Entry& entry);
// Writes the patched ranges and the trampolines. Returns false and writes nothing when OriginalsMatch fails.
[[nodiscard]] bool Apply(const Entry& entry);

[[nodiscard]] std::vector<uint8_t> Serialize(const Key& key, const Entry& entry);
[[nodiscard]] bool                 Deserialize(std::span<const uint8_t> data, const Key& key, Entry* entry);

[[nodiscard]] bool Load(const std::filesystem::path& dir, const Key& key, Entry* entry);
[[nodiscard]] bool Save(const std::filesystem::path& dir, const Key& key, const Entry& entry);

} // namespace Loader::PatchCache

#endif /* KYTY_LOADER_PATCH_CACHE_H_ */
