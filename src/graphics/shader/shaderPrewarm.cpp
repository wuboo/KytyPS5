#include "graphics/shader/shaderPrewarm.h"

#include "graphics/shader/shaderCompileWorkers.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <xxhash.h>

namespace Libs::Graphics::ShaderPrewarm {

Mode ParseMode(const char* text) {
	if (text == nullptr) return Mode::Off;
	const std::string_view value(text);
	if (value == "1") return Mode::On;
	if (value == "shadow") return Mode::Shadow;
	return Mode::Off;
}

Mode EnvMode() {
	static const Mode mode = [] {
		if (const char* text = std::getenv("KYTY_SHADER_PREWARM"); text != nullptr && *text != 0) {
			const Mode parsed = ParseMode(text);
			if (parsed != Mode::Off) return parsed;
		}
		if (ShaderCompileWorkers::ManifestEnabled()) return Mode::On;
		return Mode::Off;
	}();
	return mode;
}

// ---- manifest ----------------------------------------------------------------------------------------------------------------------------

namespace {

constexpr char kMagic[8] = {'K', 'P', 'W', 'M', '0', '0', '0', '1'};
// Bounds that keep a damaged count from allocating the machine.
constexpr uint32_t kMaxEntries    = 1u << 20;
constexpr uint32_t kMaxStaticWords = 1u << 16;
constexpr uint32_t kMaxInfoBytes   = 1u << 20;

class Writer {
public:
	void Bytes(const void* data, size_t size) {
		const auto* p = static_cast<const uint8_t*>(data);
		m_bytes.insert(m_bytes.end(), p, p + size);
	}
	void U32(uint32_t v) { Bytes(&v, sizeof(v)); }
	void U64(uint64_t v) { Bytes(&v, sizeof(v)); }
	std::vector<uint8_t> m_bytes;
};

class Reader {
public:
	explicit Reader(std::span<const uint8_t> data): m_data(data) {}
	bool Bytes(void* out, size_t size) {
		if (m_data.size() - m_at < size) return false;
		std::memcpy(out, m_data.data() + m_at, size);
		m_at += size;
		return true;
	}
	bool U32(uint32_t& v) { return Bytes(&v, sizeof(v)); }
	bool U64(uint64_t& v) { return Bytes(&v, sizeof(v)); }
	[[nodiscard]] size_t Left() const { return m_data.size() - m_at; }

private:
	std::span<const uint8_t> m_data;
	size_t                   m_at = 0;
};

} // namespace

ProgramDisk::Key EntryKey(const ManifestEntry& entry) {
	ProgramDisk::Hasher hasher;
	hasher.AddU32(entry.stage);
	hasher.AddU64(entry.code_hash);
	hasher.AddU32(entry.user_data_count);
	hasher.AddU32(entry.code_size_words);
	hasher.AddU32(entry.user_data_base);
	hasher.AddU32(entry.wave_size);
	hasher.AddU32(entry.maintenance8 ? 1u : 0u);
	hasher.AddU32(static_cast<uint32_t>(entry.static_state.size()));
	hasher.Add(entry.static_state.data(), entry.static_state.size() * sizeof(uint32_t));
	return hasher.Finish();
}

std::vector<uint8_t> SerializeManifest(const ProgramDisk::Key& fingerprint, const ManifestLayout& layout, std::span<const ManifestEntry> entries) {
	Writer out;
	out.Bytes(kMagic, sizeof(kMagic));
	out.U32(kManifestFormat);
	out.U64(fingerprint.lo);
	out.U64(fingerprint.hi);
	out.U32(layout.vertex_info_size);
	out.U32(layout.pixel_info_size);
	out.U32(layout.compute_info_size);
	out.U32(static_cast<uint32_t>(entries.size()));
	for (const auto& entry: entries) {
		out.U32(entry.stage);
		out.U64(entry.code_hash);
		out.U32(entry.user_data_count);
		out.U32(entry.code_size_words);
		out.U32(entry.user_data_base);
		out.U32(entry.wave_size);
		out.U32(entry.maintenance8 ? 1u : 0u);
		out.U32(static_cast<uint32_t>(entry.static_state.size()));
		out.Bytes(entry.static_state.data(), entry.static_state.size() * sizeof(uint32_t));
		out.U32(static_cast<uint32_t>(entry.input_info.size()));
		out.Bytes(entry.input_info.data(), entry.input_info.size());
	}
	out.U64(XXH3_64bits(out.m_bytes.data(), out.m_bytes.size()));
	return std::move(out.m_bytes);
}

bool ParseManifest(std::span<const uint8_t> data, const ProgramDisk::Key& fingerprint, const ManifestLayout& layout, std::vector<ManifestEntry>* entries) {
	if (data.size() < sizeof(kMagic) + sizeof(uint64_t)) return false;
#ifndef KYTY_PREWARM_MUTANT_MANIFEST_ACCEPTS_CORRUPT
	uint64_t stored = 0;
	std::memcpy(&stored, data.data() + data.size() - sizeof(stored), sizeof(stored));
	if (stored != XXH3_64bits(data.data(), data.size() - sizeof(stored))) return false;
#endif
	Reader      in(data.first(data.size() - sizeof(uint64_t)));
	char        magic[sizeof(kMagic)];
	uint32_t    format = 0, count = 0;
	ProgramDisk::Key key;
	ManifestLayout   stored_layout;
	if (!in.Bytes(magic, sizeof(magic)) || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 || !in.U32(format) || format != kManifestFormat) return false;
	if (!in.U64(key.lo) || !in.U64(key.hi) || !(key == fingerprint)) return false;
	if (!in.U32(stored_layout.vertex_info_size) || !in.U32(stored_layout.pixel_info_size) || !in.U32(stored_layout.compute_info_size) ||
	    !(stored_layout == layout)) {
		return false;
	}
	if (!in.U32(count) || count > kMaxEntries) return false;
	std::vector<ManifestEntry> parsed;
	parsed.reserve(std::min<uint32_t>(count, 4096u));
	for (uint32_t i = 0; i < count; i++) {
		ManifestEntry entry;
		uint32_t      maintenance8 = 0, static_words = 0, info_bytes = 0;
		if (!in.U32(entry.stage) || !in.U64(entry.code_hash) || !in.U32(entry.user_data_count) || !in.U32(entry.code_size_words) ||
		    !in.U32(entry.user_data_base) || !in.U32(entry.wave_size) || !in.U32(maintenance8) || maintenance8 > 1 || !in.U32(static_words) ||
		    static_words > kMaxStaticWords || in.Left() < static_cast<size_t>(static_words) * sizeof(uint32_t)) {
			return false;
		}
		entry.maintenance8 = maintenance8 != 0;
		entry.static_state.resize(static_words);
		if (!in.Bytes(entry.static_state.data(), static_cast<size_t>(static_words) * sizeof(uint32_t))) return false;
		if (!in.U32(info_bytes) || info_bytes > kMaxInfoBytes || in.Left() < info_bytes) return false;
		entry.input_info.resize(info_bytes);
		if (!in.Bytes(entry.input_info.data(), info_bytes)) return false;
		parsed.push_back(std::move(entry));
	}
	if (in.Left() != 0) return false; // bytes after the last entry under a valid checksum
	if (entries != nullptr) *entries = std::move(parsed);
	return true;
}

bool ManifestBuilder::Add(ManifestEntry entry) {
	const auto key = EntryKey(entry);
	if (!m_index.try_emplace(key, m_entries.size()).second) return false;
	m_entries.push_back(std::move(entry));
	return true;
}

void ManifestBuilder::Load(std::vector<ManifestEntry> entries) {
	for (auto& entry: entries) Add(std::move(entry));
}

std::filesystem::path ManifestPath(const std::filesystem::path& title_dir) {
	return title_dir / "prewarm_v1.lst";
}

// ---- shadow --------------------------------------------------------------------------------------------------------------------------------

std::string CompareSpirv(std::span<const uint32_t> own, std::span<const uint32_t> prewarmed) {
#ifdef KYTY_PREWARM_MUTANT_SHADOW_IGNORES_DIFF
	(void)own;
	(void)prewarmed;
	return {};
#else
	if (own.size() != prewarmed.size()) {
		return "spirv size " + std::to_string(own.size()) + " vs " + std::to_string(prewarmed.size()) + " words";
	}
	for (size_t i = 0; i < own.size(); i++) {
		if (own[i] != prewarmed[i]) return "spirv word " + std::to_string(i) + ": " + std::to_string(own[i]) + " vs " + std::to_string(prewarmed[i]);
	}
	return {};
#endif
}

// ---- mapped hook ---------------------------------------------------------------------------------------------------------------------------

namespace {

struct Pending {
	uint64_t                                    hash;
	std::shared_ptr<const std::vector<uint32_t>> code;
};

struct HookState {
	std::mutex            mutex;
	MappedFn              fn;
	std::vector<Pending>  pending;
	std::set<uint64_t>    seen;
};

HookState& Hook() {
	static HookState* state = new HookState(); // never destroyed: ShaderMapUserData may run during exit
	return *state;
}

} // namespace

void OnMapped(uint64_t code_hash, const uint32_t* code, size_t words) {
	if (EnvMode() == Mode::Off || code == nullptr || words == 0) return;
	auto& hook = Hook();
	std::scoped_lock lock(hook.mutex);
	if (!hook.seen.insert(code_hash).second) return;
	auto copy = std::make_shared<const std::vector<uint32_t>>(code, code + words);
	if (hook.fn) {
		hook.fn(code_hash, std::move(copy));
	} else {
		hook.pending.push_back({code_hash, std::move(copy)});
	}
}

void SetMappedHook(MappedFn fn) {
	auto& hook = Hook();
	std::scoped_lock lock(hook.mutex);
	hook.fn = std::move(fn);
	if (!hook.fn) return;
	auto pending = std::move(hook.pending);
	hook.pending.clear();
	for (auto& item: pending) hook.fn(item.hash, std::move(item.code));
}

size_t PendingCodes() {
	auto& hook = Hook();
	std::scoped_lock lock(hook.mutex);
	return hook.pending.size();
}

} // namespace Libs::Graphics::ShaderPrewarm
