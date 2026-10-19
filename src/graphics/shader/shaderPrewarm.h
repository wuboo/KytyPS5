#ifndef KYTY_GRAPHICS_SHADER_SHADERPREWARM_H_
#define KYTY_GRAPHICS_SHADER_SHADERPREWARM_H_

// S435 (S446) KYTY_SHADER_PREWARM=0|1|shadow (default 0): the part of the first use of a shader that does not depend on guest memory at the
// draw (PrepareShaderSource, TranslateProgram, ExtractResourcePlan) is done on a worker thread as soon as the game registers the shader
// (ShaderMapUserData), so the command processor finds it finished at the first lookup.
//
// Nothing can be translated from the code alone: the translation reads the stage input info (vertex fetch layout, pixel interpolators, workgroup
// shape) that is known only at a draw. So the first run of a title writes a manifest (the static state and the raw input info of every program that
// was built) and the next runs build from it. Without a manifest the mode does nothing.
//
// Pieces here have no Vulkan or recompiler dependency:
//   Mailbox<Payload>  the hand-over between the worker and the command processor (CP). One build per key, ever: a key is Queued -> Building -> Done ->
//                     Taken. CP Claim() of a Queued key cancels it (CP builds itself, the worker skips it); of a Building key waits (bounded) for the
//                     worker; of a Done key takes the result; of an unknown or finished key returns nothing and CP builds itself.
//   Manifest          the file prewarm_v1.lst under _ShaderCache/<TITLE>/: magic, format, fingerprint, struct sizes, entries, trailing XXH3-64.
//   Mapped hook       ShaderMapUserData -> OnMapped(); the codes that are seen before the hook is installed are kept and replayed.
//
// Mutants (tests/ShaderPrewarmTests.cpp, KYTY_PREWARM_MUTANT_*): DOUBLE_COMPILE, NO_WAIT_ON_BUILDING, FORGET_TAKEN, WAIT_UNBOUNDED,
// MANIFEST_ACCEPTS_CORRUPT, SHADOW_IGNORES_DIFF, NO_READY_CAP.

#include "graphics/shader/programDiskCache.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderPrewarm {

enum class Mode : uint8_t { Off, On, Shadow };

// null / "" -> Off, "1" -> On, "shadow" -> Shadow, anything else -> Off.
[[nodiscard]] Mode ParseMode(const char* text);
[[nodiscard]] Mode EnvMode();

struct KeyHash {
	size_t operator()(const ProgramDisk::Key& key) const { return static_cast<size_t>(key.lo ^ (key.hi * 0x9e3779b97f4a7c15ull)); }
};

struct MailboxStats {
	uint64_t queued    = 0; // jobs accepted
	uint64_t done      = 0; // jobs the worker finished (with a result or a failure)
	uint64_t failed    = 0; // jobs whose work returned nothing or threw
	uint64_t hit       = 0; // CP took a finished result without waiting
	uint64_t waited    = 0; // CP took a result after waiting for a Building key
	uint64_t wait_ns   = 0; // time CP waited
	uint64_t cancelled = 0; // CP claimed a Queued key: it builds itself
	uint64_t timeout   = 0; // CP gave up waiting for a Building key (the key may be built twice)
	uint64_t dedup     = 0; // Submit refused: the key is already known
	uint32_t ready      = 0; // finished results nobody has taken yet (they hold the translation of a whole program)
	uint32_t ready_peak = 0;
};

template <typename Payload>
class Mailbox {
public:
	// max_ready: the workers stop taking jobs while that many finished results wait to be taken (memory); 0 = no limit.
	explicit Mailbox(unsigned workers = 1, std::chrono::milliseconds max_wait = std::chrono::milliseconds(30000), std::function<void()> worker_setup = {},
	                 uint32_t max_ready = 0)
	    : m_max_wait(max_wait), m_setup(std::move(worker_setup)), m_max_ready(max_ready) {
		for (unsigned i = 0; i < workers; i++) m_threads.emplace_back([this] { Run(); });
	}
	~Mailbox() {
		{
			std::scoped_lock lock(m_mutex);
			m_stop = true;
		}
		m_wake.notify_all();
		for (auto& thread: m_threads) thread.join();
	}
	Mailbox(const Mailbox&)            = delete;
	Mailbox& operator=(const Mailbox&) = delete;

	using Work = std::function<std::optional<Payload>()>;

	// false: the key is known already (queued, building, done or taken): never a second build.
	bool Submit(const ProgramDisk::Key& key, Work work) {
		{
			std::scoped_lock lock(m_mutex);
			if (m_stop || !m_slots.try_emplace(key, Slot {.work = std::move(work)}).second) {
				m_dedup++;
				return false;
			}
			m_queue.push_back(key);
			m_queued++;
		}
		m_wake.notify_one();
		return true;
	}

	struct Claimed {
		std::optional<Payload> payload; // empty: CP builds the program itself
		bool                   waited = false;
		uint64_t               wait_ns = 0;
	};

	Claimed Claim(const ProgramDisk::Key& key) {
		Claimed result;
		std::unique_lock lock(m_mutex);
		auto             found = m_slots.find(key);
		if (found == m_slots.end()) return result;
		auto& slot = found->second;
		switch (slot.state) {
			case State::Queued:
#ifndef KYTY_PREWARM_MUTANT_DOUBLE_COMPILE // the mutant leaves the job queued: CP builds and the worker still runs it
				slot.state = State::Cancelled;
				slot.work  = nullptr;
#endif
				m_cancelled++;
				return result;
			case State::Building: {
#ifdef KYTY_PREWARM_MUTANT_NO_WAIT_ON_BUILDING
				return result; // CP builds while the worker still does
#else
				const auto begin = std::chrono::steady_clock::now();
#ifdef KYTY_PREWARM_MUTANT_WAIT_UNBOUNDED
				m_done.wait(lock, [&] { return slot.state != State::Building; });
				const bool finished = true;
#else
				const bool finished = m_done.wait_for(lock, m_max_wait, [&] { return slot.state != State::Building; });
#endif
				result.waited  = true;
				result.wait_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
				m_wait_ns += result.wait_ns;
				if (!finished) {
					slot.state = State::Abandoned; // the worker drops its result when it finishes
					m_timeout++;
					return result;
				}
				if (slot.state != State::Done) return result;
				result.payload = std::move(slot.payload);
				MarkTaken(found);
				m_waited++;
				return result;
#endif
			}
			case State::Done:
				result.payload = std::move(slot.payload);
				MarkTaken(found);
				m_hit++;
				return result;
			case State::Failed:
			case State::Taken:
			case State::Cancelled:
			case State::Abandoned: return result;
		}
		return result;
	}

	// Test aid: wait until the queue is empty and nothing is Building.
	void Drain() {
		std::unique_lock lock(m_mutex);
		m_done.wait(lock, [&] { return m_queue.empty() && m_building == 0; });
	}

	[[nodiscard]] MailboxStats Stats() const {
		std::scoped_lock lock(m_mutex);
		MailboxStats     stats;
		stats.queued    = m_queued;
		stats.done      = m_finished;
		stats.failed    = m_failed;
		stats.hit       = m_hit;
		stats.waited    = m_waited;
		stats.wait_ns   = m_wait_ns;
		stats.cancelled = m_cancelled;
		stats.timeout   = m_timeout;
		stats.dedup     = m_dedup;
		stats.ready      = m_ready;
		stats.ready_peak = m_ready_peak;
		return stats;
	}
	[[nodiscard]] std::vector<std::thread::id> WorkerIds() const {
		std::vector<std::thread::id> ids;
		for (const auto& thread: m_threads) ids.push_back(thread.get_id());
		return ids;
	}

private:
	enum class State : uint8_t { Queued, Building, Done, Failed, Taken, Cancelled, Abandoned };
	struct Slot {
		State                  state = State::Queued;
		Work                   work;
		std::optional<Payload> payload;
	};

	void MarkTaken(typename std::unordered_map<ProgramDisk::Key, Slot, KeyHash>::iterator found) {
		m_ready--;
		m_wake.notify_all(); // a worker held back by the limit may go on
#ifdef KYTY_PREWARM_MUTANT_FORGET_TAKEN
		m_slots.erase(found); // the key is unknown again: a later Submit builds it a second time
#else
		found->second.payload.reset();
		found->second.state = State::Taken;
#endif
	}

	void Run() {
		if (m_setup) m_setup();
		for (;;) {
			ProgramDisk::Key key;
			Work             work;
			{
				std::unique_lock lock(m_mutex);
#ifdef KYTY_PREWARM_MUTANT_NO_READY_CAP
				m_wake.wait(lock, [&] { return m_stop || !m_queue.empty(); });
#else
				m_wake.wait(lock, [&] { return m_stop || (!m_queue.empty() && (m_max_ready == 0 || m_ready < m_max_ready)); });
#endif
				if (m_stop) return;
				key = m_queue.front();
				m_queue.pop_front();
				auto& slot = m_slots.at(key);
				if (slot.state != State::Queued) { // CP took the key
					m_done.notify_all();
					continue;
				}
				slot.state = State::Building;
				work       = std::move(slot.work);
				m_building++;
			}
			std::optional<Payload> payload;
			payload = work(); // the build code does not throw (the emulator is built without exceptions): a failure is an empty result
			{
				std::scoped_lock lock(m_mutex);
				auto&            slot = m_slots.at(key);
				m_building--;
				m_finished++;
				if (!payload) m_failed++;
				if (slot.state == State::Abandoned) {
					// CP gave up and builds on its own: the result is dropped.
				} else if (payload) {
					slot.payload = std::move(payload);
					slot.state   = State::Done;
					m_ready++;
					m_ready_peak = std::max(m_ready_peak, m_ready);
				} else {
					slot.state = State::Failed;
				}
			}
			m_done.notify_all();
		}
	}

	mutable std::mutex                                        m_mutex;
	std::condition_variable                                   m_wake;
	std::condition_variable                                   m_done;
	std::unordered_map<ProgramDisk::Key, Slot, KeyHash>       m_slots;
	std::deque<ProgramDisk::Key>                              m_queue;
	std::vector<std::thread>                                  m_threads;
	std::chrono::milliseconds                                 m_max_wait;
	std::function<void()>                                     m_setup;
	uint32_t                                                  m_max_ready = 0;
	uint32_t                                                  m_ready = 0, m_ready_peak = 0;
	bool                                                      m_stop = false;
	uint32_t                                                  m_building = 0;
	uint64_t                                                  m_queued = 0, m_finished = 0, m_failed = 0, m_hit = 0, m_waited = 0, m_wait_ns = 0;
	uint64_t                                                  m_cancelled = 0, m_timeout = 0, m_dedup = 0;
};

// ---- manifest ----------------------------------------------------------------------------------------------------------------------------

inline constexpr uint32_t kManifestFormat = 1;

struct ManifestEntry {
	uint32_t              stage           = 0;
	uint64_t              code_hash       = 0; // params.hash == the hash ShaderMapUserData computes (programs with a back code are not recorded)
	uint32_t              user_data_count = 0;
	uint32_t              code_size_words = 0;
	uint32_t              user_data_base  = 0;
	uint32_t              wave_size       = 0;
	bool                  maintenance8    = false;
	std::vector<uint32_t> static_state;
	std::vector<uint8_t>  input_info; // the raw bytes of the stage input info, its runtime pointers cleared

	bool operator==(const ManifestEntry&) const = default;
};

struct ManifestLayout {
	uint32_t vertex_info_size  = 0;
	uint32_t pixel_info_size   = 0;
	uint32_t compute_info_size = 0;
	bool     operator==(const ManifestLayout&) const = default;
};

// The key of the mailbox / of the manifest: stage, code hash, user data count, code size, user data base, wave size, maintenance8, static state.
[[nodiscard]] ProgramDisk::Key EntryKey(const ManifestEntry& entry);

[[nodiscard]] std::vector<uint8_t> SerializeManifest(const ProgramDisk::Key& fingerprint, const ManifestLayout& layout, std::span<const ManifestEntry> entries);
// false: short, long, damaged, other format / fingerprint / layout.
[[nodiscard]] bool ParseManifest(std::span<const uint8_t> data, const ProgramDisk::Key& fingerprint, const ManifestLayout& layout, std::vector<ManifestEntry>* entries);

// The set of programs a run built, kept sorted by key so the file does not depend on the order of the draws.
class ManifestBuilder {
public:
	// true: a new entry.
	bool Add(ManifestEntry entry);
	[[nodiscard]] size_t Size() const { return m_entries.size(); }
	[[nodiscard]] const std::vector<ManifestEntry>& Entries() const { return m_entries; }
	void Load(std::vector<ManifestEntry> entries); // the entries of the previous run stay (the manifest only grows)
private:
	std::vector<ManifestEntry>                                  m_entries;
	std::unordered_map<ProgramDisk::Key, size_t, KeyHash>       m_index;
};

[[nodiscard]] std::filesystem::path ManifestPath(const std::filesystem::path& title_dir);

// ---- shadow --------------------------------------------------------------------------------------------------------------------------------

// First difference between the SPIR-V of the command processor's own build and of the one built from the pre-warmed translation ("" = equal).
[[nodiscard]] std::string CompareSpirv(std::span<const uint32_t> own, std::span<const uint32_t> prewarmed);

// ---- mapped hook ---------------------------------------------------------------------------------------------------------------------------

using MappedFn = std::function<void(uint64_t code_hash, std::shared_ptr<const std::vector<uint32_t>> code)>;

// Called by ShaderMapUserData with the bytes of the code (copied here: the worker never reads guest memory). With no hook yet the code is kept
// and replayed by SetMappedHook; the same hash is handed over once. With KYTY_SHADER_PREWARM off it returns at once.
void OnMapped(uint64_t code_hash, const uint32_t* code, size_t words);
// Install (or, with an empty function, remove) the hook. Returns after any call in progress ended.
void SetMappedHook(MappedFn fn);
[[nodiscard]] size_t PendingCodes();

} // namespace Libs::Graphics::ShaderPrewarm

#endif /* KYTY_GRAPHICS_SHADER_SHADERPREWARM_H_ */
