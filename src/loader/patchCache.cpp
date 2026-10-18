#include "loader/patchCache.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <xxhash.h>

namespace Loader::PatchCache {

namespace {

constexpr char     Magic[8]  = {'K', 'P', 'C', 'H', '0', '0', '0', '1'};
constexpr uint64_t MergeGap  = 16; // changed runs closer than this share one range
constexpr uint64_t MaxCount  = 1ull << 28;

void PutU64(std::vector<uint8_t>& out, uint64_t value) {
	const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
	out.insert(out.end(), bytes, bytes + sizeof(value));
}

void PutBytes(std::vector<uint8_t>& out, const std::vector<uint8_t>& bytes) {
	PutU64(out, bytes.size());
	out.insert(out.end(), bytes.begin(), bytes.end());
}

class Reader {
public:
	explicit Reader(std::span<const uint8_t> data): m_data(data) {}
	bool U64(uint64_t* value) {
		if (m_data.size() - m_pos < sizeof(*value)) return false;
		std::memcpy(value, m_data.data() + m_pos, sizeof(*value));
		m_pos += sizeof(*value);
		return true;
	}
	bool Bytes(std::vector<uint8_t>* bytes) {
		uint64_t size = 0;
		if (!U64(&size) || size > m_data.size() - m_pos) return false;
		bytes->assign(m_data.begin() + static_cast<ptrdiff_t>(m_pos), m_data.begin() + static_cast<ptrdiff_t>(m_pos + size));
		m_pos += size;
		return true;
	}
	bool Raw(void* out, size_t size) {
		if (m_data.size() - m_pos < size) return false;
		std::memcpy(out, m_data.data() + m_pos, size);
		m_pos += size;
		return true;
	}
	size_t Pos() const { return m_pos; }

private:
	std::span<const uint8_t> m_data;
	size_t                   m_pos = 0;
};

} // namespace

Mode ParseMode(const char* text) {
	if (text == nullptr || text[0] == '\0') return Mode::On;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "verify") == 0) return Mode::Verify;
	return Mode::Off;
}

Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_PATCH_CACHE"));
	return mode;
}

std::string Key::Hex() const {
	char text[40];
	std::snprintf(text, sizeof(text), "%016llx%016llx", static_cast<unsigned long long>(hi), static_cast<unsigned long long>(lo));
	return text;
}

KeyBuilder::KeyBuilder() {
	auto* state = XXH3_createState();
	XXH3_128bits_reset(state);
	m_state = state;
}

KeyBuilder::~KeyBuilder() {
	XXH3_freeState(static_cast<XXH3_state_t*>(m_state));
}

void KeyBuilder::Add(const void* data, size_t size) {
#ifdef KYTY_PATCHCACHE_MUTANT_CACHE_IGNORES_KEY
	if (size > 64) return; // the bytes of the segments no longer reach the key
#endif
	XXH3_128bits_update(static_cast<XXH3_state_t*>(m_state), data, size);
}

void KeyBuilder::AddU64(uint64_t value) {
	Add(&value, sizeof(value));
}

Key KeyBuilder::Finish() {
	const auto hash = XXH3_128bits_digest(static_cast<XXH3_state_t*>(m_state));
	return {hash.low64, hash.high64};
}

Snapshot::Snapshot(std::span<const std::pair<uint64_t, uint64_t>> segments) {
	m_segments.reserve(segments.size());
	for (const auto& [addr, size]: segments) {
		Segment segment;
		segment.addr = addr;
		segment.bytes.assign(reinterpret_cast<const uint8_t*>(addr), reinterpret_cast<const uint8_t*>(addr) + size);
		m_segments.push_back(std::move(segment));
	}
}

Entry Snapshot::Capture(uint64_t trampoline_addr, uint64_t trampoline_used) const {
	Entry entry;
	for (const auto& segment: m_segments) {
		const auto*    live = reinterpret_cast<const uint8_t*>(segment.addr);
		const auto*    old  = segment.bytes.data();
		const uint64_t size = segment.bytes.size();
		uint64_t       i    = 0;
		while (i < size) {
			// Skip equal bytes a word at a time.
			while (i + 8 <= size && std::memcmp(live + i, old + i, 8) == 0) i += 8;
			while (i < size && live[i] == old[i]) ++i;
			if (i >= size) break;
			uint64_t begin = i;
			uint64_t end   = i + 1; // one past the last changed byte seen
			for (i = end; i < size && i - end < MergeGap; ++i) {
				if (live[i] != old[i]) end = i + 1;
			}
			Range range;
			range.addr = segment.addr + begin;
			range.original.assign(old + begin, old + end);
			range.patched.assign(live + begin, live + end);
			entry.ranges.push_back(std::move(range));
			i = end;
		}
	}
#ifdef KYTY_PATCHCACHE_MUTANT_CAPTURE_DROPS_LAST_RANGE
	if (!entry.ranges.empty()) entry.ranges.pop_back();
#endif
	entry.trampoline_addr = trampoline_addr;
	const auto* tramp     = reinterpret_cast<const uint8_t*>(trampoline_addr);
	entry.trampoline.assign(tramp, tramp + trampoline_used);
	return entry;
}

bool OriginalsMatch(const Entry& entry) {
#ifdef KYTY_PATCHCACHE_MUTANT_APPLY_NO_ORIGINAL_CHECK
	return true;
#else
	return std::ranges::all_of(entry.ranges, [](const Range& range) {
		return std::memcmp(reinterpret_cast<const void*>(range.addr), range.original.data(), range.original.size()) == 0;
	});
#endif
}

bool Apply(const Entry& entry) {
	if (!OriginalsMatch(entry)) return false;
	for (const auto& range: entry.ranges) {
		std::memcpy(reinterpret_cast<void*>(range.addr), range.patched.data(), range.patched.size());
	}
#ifndef KYTY_PATCHCACHE_MUTANT_APPLY_SKIPS_TRAMPOLINE
	if (!entry.trampoline.empty()) {
		std::memcpy(reinterpret_cast<void*>(entry.trampoline_addr), entry.trampoline.data(), entry.trampoline.size());
	}
#endif
	return true;
}

std::vector<uint8_t> Serialize(const Key& key, const Entry& entry) {
	std::vector<uint8_t> out(Magic, Magic + sizeof(Magic));
	PutU64(out, PatchVersion);
	PutU64(out, key.lo);
	PutU64(out, key.hi);
	PutU64(out, entry.ranges.size());
	for (const auto& range: entry.ranges) {
		PutU64(out, range.addr);
		PutBytes(out, range.original);
		PutBytes(out, range.patched);
	}
	PutU64(out, entry.trampoline_addr);
	PutBytes(out, entry.trampoline);
	PutU64(out, entry.patcher_bytes);
#ifdef KYTY_PATCHCACHE_MUTANT_SERIALIZE_DROPS_TRAPS
	PutU64(out, 0);
#else
	PutU64(out, entry.wide_traps.size());
	for (const auto& [site, trampoline]: entry.wide_traps) {
		PutU64(out, site);
		PutU64(out, trampoline);
	}
#endif
	PutU64(out, entry.stats.size());
	for (const uint64_t value: entry.stats) PutU64(out, value);
	PutU64(out, XXH3_64bits(out.data(), out.size()));
	return out;
}

bool Deserialize(std::span<const uint8_t> data, const Key& key, Entry* entry) {
	if (data.size() < sizeof(Magic) + 8) return false;
#ifndef KYTY_PATCHCACHE_MUTANT_CHECKSUM_IGNORED
	uint64_t stored = 0;
	std::memcpy(&stored, data.data() + data.size() - 8, 8);
	if (stored != XXH3_64bits(data.data(), data.size() - 8)) return false;
#endif
	Reader reader(data.first(data.size() - 8));
	char   magic[sizeof(Magic)];
	if (!reader.Raw(magic, sizeof(magic)) || std::memcmp(magic, Magic, sizeof(Magic)) != 0) return false;
	uint64_t version = 0, lo = 0, hi = 0, count = 0;
	if (!reader.U64(&version) || version != PatchVersion || !reader.U64(&lo) || !reader.U64(&hi) || lo != key.lo || hi != key.hi ||
	    !reader.U64(&count) || count > MaxCount) {
		return false;
	}
	Entry out;
	out.ranges.resize(count);
	for (auto& range: out.ranges) {
		if (!reader.U64(&range.addr) || !reader.Bytes(&range.original) || !reader.Bytes(&range.patched) ||
		    range.original.size() != range.patched.size()) {
			return false;
		}
	}
	if (!reader.U64(&out.trampoline_addr) || !reader.Bytes(&out.trampoline) || !reader.U64(&out.patcher_bytes) || !reader.U64(&count) ||
	    count > MaxCount) {
		return false;
	}
	out.wide_traps.resize(count);
	for (auto& [site, trampoline]: out.wide_traps) {
		if (!reader.U64(&site) || !reader.U64(&trampoline)) return false;
	}
	if (!reader.U64(&count) || count > MaxCount) return false;
	out.stats.resize(count);
	for (auto& value: out.stats) {
		if (!reader.U64(&value)) return false;
	}
	if (reader.Pos() != data.size() - 8) return false;
	*entry = std::move(out);
	return true;
}

bool Load(const std::filesystem::path& dir, const Key& key, Entry* entry) {
	std::ifstream file(dir / (key.Hex() + ".kpc"), std::ios::binary | std::ios::ate);
	if (!file) return false;
	const auto size = file.tellg();
	if (size <= 0) return false;
	std::vector<uint8_t> data(static_cast<size_t>(size));
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(data.data()), size)) return false;
	return Deserialize(data, key, entry);
}

bool Save(const std::filesystem::path& dir, const Key& key, const Entry& entry) {
	std::error_code error;
	std::filesystem::create_directories(dir, error);
	const auto data = Serialize(key, entry);
	const auto path = dir / (key.Hex() + ".kpc");
	auto       tmp  = path;
	tmp += ".tmp";
	{
		std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
		if (!file || !file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()))) return false;
	}
	std::filesystem::rename(tmp, path, error);
	return !error;
}

} // namespace Loader::PatchCache
