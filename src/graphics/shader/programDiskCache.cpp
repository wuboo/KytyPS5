#include "graphics/shader/programDiskCache.h"

#include "common/knownEnv.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <xxhash.h>

#if defined(__APPLE__)
#include <crt_externs.h>
#elif !defined(_WIN32)
extern char** environ;
#endif

namespace Libs::Graphics::ProgramDisk {

namespace {

constexpr char Magic[8] = {'K', 'P', 'R', 'G', '0', '0', '0', '1'};
// magic + format + program key + permutation key + payload size, then the payload, then the checksum.
constexpr size_t HeaderSize = sizeof(Magic) + 8 + 16 + 16 + 8;

void PutU64(std::vector<uint8_t>& out, uint64_t value) {
	const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
	out.insert(out.end(), bytes, bytes + sizeof(value));
}

uint64_t GetU64(const uint8_t* at) {
	uint64_t value = 0;
	std::memcpy(&value, at, sizeof(value));
	return value;
}

} // namespace

Mode ParseMode(const char* text) {
	if (text == nullptr || text[0] == '\0') return Mode::Off;
	if (std::strcmp(text, "1") == 0) return Mode::On;
	if (std::strcmp(text, "verify") == 0) return Mode::Verify;
	return Mode::Off;
}

Mode EnvMode() {
	static const Mode mode = ParseMode(std::getenv("KYTY_PROGRAM_DISK_CACHE"));
	return mode;
}

std::string Key::Hex() const {
	char text[40];
	std::snprintf(text, sizeof(text), "%016llx%016llx", static_cast<unsigned long long>(hi), static_cast<unsigned long long>(lo));
	return text;
}

Hasher::Hasher() {
	auto* state = XXH3_createState();
	XXH3_128bits_reset(state);
	m_state = state;
}

Hasher::~Hasher() {
	XXH3_freeState(static_cast<XXH3_state_t*>(m_state));
}

void Hasher::Add(const void* data, size_t size) {
	XXH3_128bits_update(static_cast<XXH3_state_t*>(m_state), data, size);
}

void Hasher::AddString(std::string_view text) {
	AddU64(text.size());
	Add(text.data(), text.size());
}

Key Hasher::Finish() {
	const auto hash = XXH3_128bits_digest(static_cast<XXH3_state_t*>(m_state));
	return {hash.low64, hash.high64};
}

Key HashBytes(std::span<const uint32_t> words) {
	Hasher hasher;
	hasher.AddU64(words.size());
	hasher.Add(words.data(), words.size_bytes());
	return hasher.Finish();
}

// ---- fingerprint ------------------------------------------------------------------------------------------------------------------------

const char* const* ProcessEnvironment() {
#ifdef KYTY_PROGDISK_MUTANT_PROCESS_ENV_NULL
	return nullptr;
#elif defined(__APPLE__)
	return *_NSGetEnviron();
#elif defined(_WIN32)
	return _environ;
#else
	return environ;
#endif
}

bool IsNonCodegenEnv(std::string_view name) {
	return std::ranges::find(Common::kNonCodegenKytyEnv, name) != std::end(Common::kNonCodegenKytyEnv);
}

std::vector<std::string> CodegenEnvironment(const char* const* env) {
	std::vector<std::string> result;
	if (env == nullptr) return result;
	for (const char* const* entry = env; *entry != nullptr; ++entry) {
		const std::string_view text(*entry);
		if (!text.starts_with("KYTY_")) continue;
		const auto equals = text.find('=');
		const auto name   = text.substr(0, equals);
		if (IsNonCodegenEnv(name)) continue;
		result.emplace_back(text);
	}
	std::ranges::sort(result);
	return result;
}

Fingerprint MakeFingerprint(const FingerprintInputs& inputs) {
	Fingerprint result;
	if (inputs.config.debug_dump) {
		result.enabled = false;
		result.reason  = "graphics debug dump is on (the SPIR-V carries debug info)";
	} else if (inputs.config.shader_log) {
		result.enabled = false;
		result.reason  = "shader log is on (the IR dumps need a real compile)";
	}
	Hasher hasher;
	hasher.AddU32(kProgramDiskFormat);
#ifndef KYTY_PROGDISK_MUTANT_KEY_IGNORES_SIGNATURE
	hasher.AddString(inputs.recompiler_signature);
#endif
	const auto env = CodegenEnvironment(inputs.environment);
#ifndef KYTY_PROGDISK_MUTANT_KEY_IGNORES_ENV
	hasher.AddU64(env.size());
	for (const auto& entry: env) hasher.AddString(entry);
#endif
	hasher.AddU32(static_cast<uint32_t>(inputs.config.shader_optimization_type));
	const auto& device = inputs.device;
	hasher.AddU32(device.vendor_id);
	hasher.AddU32(device.device_id);
	hasher.AddU32(device.driver_version);
	hasher.Add(device.cache_uuid, sizeof(device.cache_uuid));
	hasher.AddU32(device.subgroup_size);
	hasher.AddU32(device.maintenance8 ? 1u : 0u);
	hasher.AddU32(device.mesh_shader ? 1u : 0u);
	result.key = hasher.Finish();
	char text[256];
	std::snprintf(text, sizeof(text), "fingerprint=%s format=%u env_entries=%zu opt=%d vendor=%04x device=%04x driver=%08x subgroup=%u m8=%d mesh=%d",
	              result.key.Hex().c_str(), kProgramDiskFormat, env.size(), inputs.config.shader_optimization_type, device.vendor_id,
	              device.device_id, device.driver_version, device.subgroup_size, device.maintenance8 ? 1 : 0, device.mesh_shader ? 1 : 0);
	result.description = text;
	return result;
}

// ---- keys --------------------------------------------------------------------------------------------------------------------------------

Key MakeProgramKey(const Key& fingerprint, const ProgramIdentity& identity) {
	Hasher hasher;
	hasher.AddKey(fingerprint);
	hasher.AddU32(identity.stage);
	hasher.AddU64(identity.params_hash);
	hasher.AddU32(identity.user_data_count);
	hasher.AddU32(identity.code_size);
#ifndef KYTY_PROGDISK_MUTANT_KEY_DROPS_STATIC_STATE
	hasher.AddU64(identity.static_state.size());
	hasher.Add(identity.static_state.data(), identity.static_state.size_bytes());
#endif
	hasher.AddU32(identity.user_data_base);
	hasher.AddU32(identity.wave_size);
	hasher.AddKey(HashBytes(identity.code));
	hasher.AddKey(HashBytes(identity.back_code));
	return hasher.Finish();
}

Key MakePermutationKey(const Key& program_key, std::span<const uint8_t> permutation_input) {
	Hasher hasher;
	hasher.AddKey(program_key);
	hasher.AddU64(permutation_input.size());
	hasher.Add(permutation_input.data(), permutation_input.size());
	return hasher.Finish();
}

// ---- file --------------------------------------------------------------------------------------------------------------------------------

std::vector<uint8_t> SerializeFile(const Key& program, const Key& permutation, std::span<const uint8_t> payload) {
	std::vector<uint8_t> out(Magic, Magic + sizeof(Magic));
	out.reserve(HeaderSize + payload.size() + 8);
	PutU64(out, kProgramDiskFormat);
	PutU64(out, program.lo);
	PutU64(out, program.hi);
	PutU64(out, permutation.lo);
	PutU64(out, permutation.hi);
	PutU64(out, payload.size());
	out.insert(out.end(), payload.begin(), payload.end());
	PutU64(out, XXH3_64bits(out.data(), out.size()));
	return out;
}

bool DeserializeFile(std::span<const uint8_t> data, const Key& program, const Key& permutation, std::vector<uint8_t>* payload) {
	if (data.size() < HeaderSize + 8) return false;
#ifndef KYTY_PROGDISK_MUTANT_ACCEPTS_CORRUPT
	if (GetU64(data.data() + data.size() - 8) != XXH3_64bits(data.data(), data.size() - 8)) return false;
#endif
	if (std::memcmp(data.data(), Magic, sizeof(Magic)) != 0) return false;
	const uint8_t* at = data.data() + sizeof(Magic);
	if (GetU64(at) != kProgramDiskFormat) return false;
	if (GetU64(at + 8) != program.lo || GetU64(at + 16) != program.hi) return false;
	if (GetU64(at + 24) != permutation.lo || GetU64(at + 32) != permutation.hi) return false;
	const uint64_t size = GetU64(at + 40);
#ifdef KYTY_PROGDISK_MUTANT_ACCEPT_TRUNCATED
	const bool size_ok = size <= data.size() - HeaderSize;
#else
	const bool size_ok = size == data.size() - HeaderSize - 8;
#endif
	if (!size_ok) return false;
	payload->assign(data.begin() + HeaderSize, data.begin() + HeaderSize + static_cast<ptrdiff_t>(size));
	return true;
}

std::filesystem::path FilePath(const std::filesystem::path& dir, uint32_t stage, const Key& program, const Key& permutation) {
	char name[80];
	std::snprintf(name, sizeof(name), "%u_%016llx_%016llx.kpp", stage, static_cast<unsigned long long>(program.lo),
	              static_cast<unsigned long long>(permutation.lo));
	return dir / "prog_v1" / name;
}

namespace {
std::atomic<Prefetch*> s_prefetch {nullptr};
thread_local uint64_t  t_last_read_ns = 0;

std::optional<std::vector<uint8_t>> ReadWholeFile(const std::filesystem::path& path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) return std::nullopt;
	const auto size = file.tellg();
	if (size <= 0) return std::vector<uint8_t> {};
	std::vector<uint8_t> data(static_cast<size_t>(size));
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(data.data()), size)) return std::nullopt;
	return data;
}
} // namespace

uint64_t LastLoadReadNs() {
	return t_last_read_ns;
}

void SetPrefetch(Prefetch* prefetch) {
	s_prefetch.store(prefetch, std::memory_order_release);
}

std::optional<std::vector<uint8_t>> Load(const std::filesystem::path& path, const Key& program, const Key& permutation, bool* rejected) {
	if (rejected != nullptr) *rejected = false;
	std::optional<std::vector<uint8_t>> data;
	const auto                          read_begin = std::chrono::steady_clock::now();
#ifndef KYTY_PROGDISK_MUTANT_PREFETCH_UNUSED
	if (auto* prefetch = s_prefetch.load(std::memory_order_acquire)) data = prefetch->Take(path);
#endif
	if (!data) data = ReadWholeFile(path);
	t_last_read_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - read_begin).count());
	if (!data) return std::nullopt;
	if (data->empty()) {
		if (rejected != nullptr) *rejected = true;
		return std::nullopt;
	}
	std::vector<uint8_t> payload;
	if (!DeserializeFile(*data, program, permutation, &payload)) {
		if (rejected != nullptr) *rejected = true;
		return std::nullopt;
	}
	return payload;
}

// ---- prefetch ----------------------------------------------------------------------------------------------------------------------------

bool PrefetchEnvEnabled() {
	static const bool enabled = [] {
		const char* text = std::getenv("KYTY_PROGRAM_DISK_PREFETCH");
		return text != nullptr && std::strcmp(text, "1") == 0;
	}();
	return enabled;
}

struct Prefetch::State {
	std::mutex                                                mutex;
	std::condition_variable                                   wake;
	std::unordered_map<std::string, std::vector<uint8_t>>     store;
	uint64_t                                                  resident = 0;
	uint64_t                                                  max_bytes = 0;
	bool                                                      stop = false;
	bool                                                      done = false;
	std::atomic<uint64_t>                                     files {0};
	std::atomic<uint64_t>                                     bytes {0};
	std::atomic<uint64_t>                                     read_ns {0};
	std::atomic<uint64_t>                                     hit {0};
	std::atomic<uint64_t>                                     miss {0};
	std::thread                                               worker;
};

Prefetch::Prefetch(std::filesystem::path dir, uint64_t max_bytes): m_state(new State) {
	State* state     = m_state;
	state->max_bytes = max_bytes;
	state->worker    = std::thread([state, dir = std::move(dir)] {
		std::vector<std::filesystem::path> paths;
		{
			std::error_code error;
			for (std::filesystem::directory_iterator it(dir, error), end; !error && it != end; it.increment(error)) {
				if (it->path().extension() == ".kpp") paths.push_back(it->path());
			}
		}
		for (const auto& path: paths) {
			const auto key = path.generic_string();
			{
				std::unique_lock lock(state->mutex);
				if (state->stop) break;
				std::error_code error;
				const auto      size = std::filesystem::file_size(path, error);
				if (error) continue;
#ifndef KYTY_PROGDISK_MUTANT_PREFETCH_UNBOUNDED
				if (state->resident + size > state->max_bytes) break;
#endif
			}
			const auto begin = std::chrono::steady_clock::now();
			auto       data  = ReadWholeFile(path);
			const auto ns    = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count();
			state->read_ns.fetch_add(static_cast<uint64_t>(ns));
			if (!data || data->empty()) continue;
			state->files.fetch_add(1);
			state->bytes.fetch_add(data->size());
			std::scoped_lock lock(state->mutex);
			if (state->stop) break;
			state->resident += data->size();
			state->store[key] = std::move(*data);
		}
		std::scoped_lock lock(state->mutex);
		state->done = true;
		state->wake.notify_all();
	});
}

Prefetch::~Prefetch() {
	{
		std::scoped_lock lock(m_state->mutex);
		m_state->stop = true;
	}
	m_state->wake.notify_all();
	m_state->worker.join();
	delete m_state;
}

std::optional<std::vector<uint8_t>> Prefetch::Take(const std::filesystem::path& path) {
	std::scoped_lock lock(m_state->mutex);
	const auto       it = m_state->store.find(path.generic_string());
	if (it == m_state->store.end()) {
		m_state->miss.fetch_add(1);
		return std::nullopt;
	}
	std::vector<uint8_t> data = std::move(it->second);
#ifndef KYTY_PROGDISK_MUTANT_PREFETCH_KEEPS_BYTES
	m_state->resident -= data.size();
	m_state->store.erase(it);
#endif
	m_state->hit.fetch_add(1);
	return data;
}

void Prefetch::WaitDone() {
	std::unique_lock lock(m_state->mutex);
	m_state->wake.wait(lock, [&] { return m_state->done; });
}

Prefetch::Stats Prefetch::GetStats() const {
	return {m_state->files.load(), m_state->bytes.load(), m_state->read_ns.load(), m_state->hit.load(), m_state->miss.load()};
}

namespace {

std::atomic<uint64_t> s_tmp_counter {0};

bool SaveWithHook(const std::filesystem::path& path, std::span<const uint8_t> bytes,
                  const std::function<void(const std::filesystem::path&)>& before_rename) {
	std::error_code error;
	std::filesystem::create_directories(path.parent_path(), error);
	auto tmp = path;
	tmp += ".tmp." + std::to_string(s_tmp_counter.fetch_add(1));
#ifdef KYTY_PROGDISK_MUTANT_NO_RENAME
	const auto& target = path; // written in place: a reader can see a half written file
#else
	const auto& target = tmp;
#endif
	{
		std::ofstream file(target, std::ios::binary | std::ios::trunc);
		if (!file) return false;
		const auto half = bytes.size() / 2;
		file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(half));
		if (before_rename) before_rename(path);
		file.write(reinterpret_cast<const char*>(bytes.data() + half), static_cast<std::streamsize>(bytes.size() - half));
		if (!file) {
			file.close();
			std::filesystem::remove(target, error);
			return false;
		}
	}
#ifdef KYTY_PROGDISK_MUTANT_NO_RENAME
	return true;
#else
	std::filesystem::rename(tmp, path, error);
	if (error) {
		std::filesystem::remove(tmp, error);
		return false;
	}
	return true;
#endif
}

} // namespace

bool Save(const std::filesystem::path& path, std::span<const uint8_t> file_bytes) {
	return SaveWithHook(path, file_bytes, {});
}

struct AsyncWriter::State {
	std::mutex                                                  mutex;
	std::condition_variable                                     wake;
	std::condition_variable                                     idle;
	std::deque<std::pair<std::filesystem::path, std::vector<uint8_t>>> queue;
	bool                                                        busy = false;
	bool                                                        stop = false;
	std::thread::id                                             last_writer;
	std::atomic<uint64_t>                                       written {0};
	std::atomic<uint64_t>                                       failed {0};
	std::atomic<uint64_t>                                       bytes {0};
	std::atomic<uint64_t>                                       dropped {0};
	std::thread                                                 worker;
};

AsyncWriter::AsyncWriter(): m_state(new State) {
	State* state = m_state;
	state->worker = std::thread([this, state] {
		std::unique_lock lock(state->mutex);
		for (;;) {
			state->wake.wait(lock, [&] { return state->stop || !state->queue.empty(); });
			if (state->queue.empty()) {
				if (state->stop) return;
				continue;
			}
			auto item = std::move(state->queue.front());
			state->queue.pop_front();
			state->busy = true;
			lock.unlock();
			const bool ok = SaveWithHook(item.first, item.second, before_rename);
			lock.lock();
			state->last_writer = std::this_thread::get_id();
			if (ok) {
				state->written.fetch_add(1);
				state->bytes.fetch_add(item.second.size());
			} else {
				state->failed.fetch_add(1);
			}
			state->busy = false;
			if (state->queue.empty()) state->idle.notify_all();
		}
	});
}

AsyncWriter::~AsyncWriter() {
	Flush();
	{
		std::scoped_lock lock(m_state->mutex);
		m_state->stop = true;
	}
	m_state->wake.notify_all();
	m_state->worker.join();
	delete m_state;
}

bool AsyncWriter::Enqueue(std::filesystem::path path, std::vector<uint8_t> file_bytes) {
#ifdef KYTY_PROGDISK_MUTANT_WRITE_ON_CALLER
	const bool ok = SaveWithHook(path, file_bytes, before_rename);
	std::scoped_lock lock(m_state->mutex);
	m_state->last_writer = std::this_thread::get_id();
	if (ok) {
		m_state->written.fetch_add(1);
		m_state->bytes.fetch_add(file_bytes.size());
	} else {
		m_state->failed.fetch_add(1);
	}
	return true;
#else
	{
		std::scoped_lock lock(m_state->mutex);
		if (m_state->queue.size() >= kMaxQueued) {
			m_state->dropped.fetch_add(1);
			return false;
		}
		m_state->queue.emplace_back(std::move(path), std::move(file_bytes));
	}
	m_state->wake.notify_one();
	return true;
#endif
}

void AsyncWriter::Flush() {
	std::unique_lock lock(m_state->mutex);
	m_state->idle.wait(lock, [&] { return m_state->queue.empty() && !m_state->busy; });
}

uint64_t AsyncWriter::Written() const { return m_state->written.load(); }
uint64_t AsyncWriter::Failed() const { return m_state->failed.load(); }
uint64_t AsyncWriter::Bytes() const { return m_state->bytes.load(); }
uint64_t AsyncWriter::Dropped() const { return m_state->dropped.load(); }

std::thread::id AsyncWriter::LastWriter() const {
	std::scoped_lock lock(m_state->mutex);
	return m_state->last_writer;
}

} // namespace Libs::Graphics::ProgramDisk
