// S435 (S446): KYTY_SHADER_PREWARM. The mailbox must build every key once and only once whoever gets there first (CP cancels a queued job, waits for a
// building one with a bound, takes a finished one, never lets a key be submitted again), the worker must be another thread than the caller, the
// manifest must come back equal and be refused when short, long, damaged or of another fingerprint / layout, the shadow compare must see any
// difference of the SPIR-V and the mapped hook must copy the code and hand each hash over once.
// Mutants: see src/graphics/shader/shaderPrewarm.h. Run with KYTY_SHADER_PREWARM=1 (the "off" argument runs the checks for the unset variable).

#include "graphics/shader/shaderPrewarm.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <map>
#include <string>
#include <thread>
#include <vector>
#include <xxhash.h>

using namespace Libs::Graphics::ShaderPrewarm;
using namespace std::chrono_literals;
using Libs::Graphics::ProgramDisk::Key;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

Key K(uint64_t n) {
	return Key {n, ~n};
}

void TestMode() {
	Check(ParseMode(nullptr) == Mode::Off, "null -> off");
	Check(ParseMode("") == Mode::Off, "empty -> off");
	Check(ParseMode("0") == Mode::Off, "0 -> off");
	Check(ParseMode("1") == Mode::On, "1 -> on");
	Check(ParseMode("shadow") == Mode::Shadow, "shadow -> shadow");
	Check(ParseMode("verify") == Mode::Off, "other -> off");
}

// A job that finished before CP looked: CP takes it, it ran on a worker thread, a second take and a second submit get nothing.
void TestDoneIsTakenOnce() {
	Mailbox<int>    box(1);
	std::thread::id ran_on;
	Check(box.Submit(K(1), [&]() -> std::optional<int> {
		ran_on = std::this_thread::get_id();
		return 42;
	}), "submit accepted");
	box.Drain();
	Check(ran_on != std::thread::id() && ran_on != std::this_thread::get_id(), "the job runs on the worker, not on the caller");
	const auto workers = box.WorkerIds();
	Check(workers.size() == 1 && workers[0] == ran_on, "the worker is the one the mailbox started");
	auto claimed = box.Claim(K(1));
	Check(claimed.payload && *claimed.payload == 42, "CP takes the finished result");
	Check(!claimed.waited, "a finished job is not waited for");
	Check(!box.Claim(K(1)).payload, "a result is taken once");
	Check(!box.Submit(K(1), []() -> std::optional<int> { return 7; }), "a taken key is never submitted again");
	box.Drain();
	Check(!box.Claim(K(1)).payload, "the refused submit built nothing");
	Check(!box.Claim(K(99)).payload, "an unknown key gives nothing (CP builds)");
	const auto stats = box.Stats();
	Check(stats.queued == 1 && stats.done == 1 && stats.hit == 1 && stats.waited == 0 && stats.dedup == 1, "counters of the finished case");
}

// CP arrives while the worker builds: it waits and takes the result; nothing is built a second time.
void TestWaitsForBuilding() {
	Mailbox<int>        box(1);
	std::promise<void>  started, release;
	auto                release_future = release.get_future().share();
	std::atomic<int>    worker_builds {0};
	box.Submit(K(5), [&]() -> std::optional<int> {
		worker_builds++;
		started.set_value();
		release_future.wait();
		return 5;
	});
	started.get_future().wait();
	std::thread releaser([&] {
		std::this_thread::sleep_for(60ms);
		release.set_value();
	});
	int  cp_builds = 0;
	auto claimed   = box.Claim(K(5));
	if (!claimed.payload) cp_builds++;
	releaser.join();
	box.Drain();
	Check(claimed.payload && *claimed.payload == 5, "CP gets the result of the job that was building");
	Check(claimed.waited && claimed.wait_ns >= 30'000'000ull, "CP waited for it");
	Check(cp_builds == 0 && worker_builds.load() == 1, "one build of the key, not two");
	const auto stats = box.Stats();
	Check(stats.waited == 1 && stats.wait_ns >= 30'000'000ull, "wait counters");
}

// CP arrives first: the queued job is cancelled, CP builds, the worker never runs it.
void TestCancelsQueued() {
	Mailbox<int>       box(1);
	std::promise<void> started, release;
	auto               release_future = release.get_future().share();
	std::atomic<int>   runs_a {0}, runs_b {0};
	box.Submit(K(10), [&]() -> std::optional<int> {
		runs_a++;
		started.set_value();
		release_future.wait();
		return 1;
	});
	started.get_future().wait(); // the only worker is busy: the next job stays queued
	box.Submit(K(11), [&]() -> std::optional<int> {
		runs_b++;
		return 2;
	});
	int  cp_builds = 0;
	auto claimed   = box.Claim(K(11));
	if (!claimed.payload) cp_builds++;
	release.set_value();
	box.Drain();
	Check(!claimed.payload && cp_builds == 1, "CP builds a key that was only queued");
	Check(runs_a.load() == 1 && runs_b.load() == 0, "the worker skipped the cancelled job: the key is built once");
	Check(!box.Claim(K(11)).payload, "a cancelled key gives nothing later either");
	Check(!box.Submit(K(11), []() -> std::optional<int> { return 3; }), "a cancelled key is not submitted again");
	Check(box.Stats().cancelled == 1, "cancel counter");
}

// The wait is bounded: after the limit CP builds on its own and the late result is dropped.
void TestWaitIsBounded() {
	Mailbox<int>       box(1, 25ms);
	std::promise<void> started, release;
	auto               release_future = release.get_future().share();
	box.Submit(K(20), [&]() -> std::optional<int> {
		started.set_value();
		release_future.wait();
		return 9;
	});
	started.get_future().wait();
	std::thread releaser([&] {
		std::this_thread::sleep_for(400ms);
		release.set_value();
	});
	const auto begin   = std::chrono::steady_clock::now();
	auto       claimed = box.Claim(K(20));
	const auto took    = std::chrono::steady_clock::now() - begin;
	Check(!claimed.payload, "after the limit CP gets nothing and builds itself");
	Check(took < 300ms, "the wait ended at the limit, not when the worker finished");
	Check(box.Stats().timeout == 1, "timeout counter");
	releaser.join();
	box.Drain();
	Check(!box.Claim(K(20)).payload, "the late result of an abandoned job is dropped");
}

void TestFailedWork() {
	Mailbox<int> box(1);
	box.Submit(K(30), []() -> std::optional<int> { return std::nullopt; });
	box.Submit(K(31), []() -> std::optional<int> { return std::nullopt; });
	box.Submit(K(32), []() -> std::optional<int> { return 1; });
	box.Drain();
	Check(!box.Claim(K(30)).payload && !box.Claim(K(31)).payload, "a failed job gives CP nothing");
	Check(box.Claim(K(32)).payload.has_value(), "the worker survived the failures");
	Check(box.Stats().failed == 2, "failed counter");
}

// Many keys, two workers and a CP that claims in a different order: every key is built exactly once, by the worker or by CP.
void TestEveryKeyBuiltOnce() {
	constexpr int          kKeys = 300;
	Mailbox<int>           box(2);
	std::mutex             mutex;
	std::map<uint64_t, int> builds;
	const auto             build = [&](uint64_t n) {
        std::scoped_lock lock(mutex);
        builds[n]++;
        std::this_thread::yield();
	};
	for (int i = 0; i < kKeys; i++) {
		box.Submit(K(1000 + i), [&, i]() -> std::optional<int> {
			build(1000 + i);
			return i;
		});
	}
	// One CP (the command processor is one thread), against two workers.
	for (int i = 0; i < kKeys; i++) {
		const int n       = (i * 7) % kKeys;
		auto      claimed = box.Claim(K(1000 + n));
		if (claimed.payload) {
			if (*claimed.payload != n) {
				std::scoped_lock lock(mutex);
				builds[999999]++; // a wrong payload
			}
		} else {
			build(1000 + n);
		}
	}
	box.Drain();
	int wrong = 0, not_once = 0, missing = 0;
	for (int n = 0; n < kKeys; n++) {
		const auto found = builds.find(1000 + n);
		if (found == builds.end()) missing++;
		else if (found->second != 1) not_once++;
	}
	if (builds.count(999999)) wrong = builds[999999];
	Check(wrong == 0, "the payload belongs to its key");
	Check(not_once == 0, "no key is built twice");
	Check(missing == 0, "no key is left unbuilt");
}

// ---- manifest ----------------------------------------------------------------------------------------------------------------------------

ManifestEntry Entry(uint32_t seed) {
	ManifestEntry entry;
	entry.stage           = seed % 3;
	entry.code_hash       = 0x1111222233334444ull + seed;
	entry.user_data_count = 12 + seed;
	entry.code_size_words = 100 + seed;
	entry.user_data_base  = seed % 2 ? 8 : 0;
	entry.wave_size       = seed % 2 ? 32 : 64;
	entry.maintenance8    = seed % 2 != 0;
	for (uint32_t i = 0; i < 5 + seed; i++) entry.static_state.push_back(0xa0000000u + seed * 100 + i);
	for (uint32_t i = 0; i < 40 + seed; i++) entry.input_info.push_back(static_cast<uint8_t>(i * 7 + seed));
	return entry;
}

std::vector<uint8_t> WithChecksum(std::vector<uint8_t> body) {
	const uint64_t sum = XXH3_64bits(body.data(), body.size());
	const auto*    p   = reinterpret_cast<const uint8_t*>(&sum);
	body.insert(body.end(), p, p + sizeof(sum));
	return body;
}

void TestManifest() {
	const Key            fingerprint {0x1234, 0x5678};
	const ManifestLayout layout {1000, 400, 200};
	std::vector<ManifestEntry> entries {Entry(0), Entry(1), Entry(2)};
	const auto           bytes = SerializeManifest(fingerprint, layout, entries);
	std::vector<ManifestEntry> back;
	Check(ParseManifest(bytes, fingerprint, layout, &back), "round trip parses");
	Check(back == entries, "round trip gives the same entries (every field)");
	Check(ParseManifest(SerializeManifest(fingerprint, layout, {}), fingerprint, layout, &back) && back.empty(), "empty manifest");

	// every field of an entry moves its key
	const auto base = EntryKey(Entry(0));
	{
		auto e = Entry(0); e.stage++;           Check(!(EntryKey(e) == base), "key moves with the stage");
		e = Entry(0); e.code_hash++;            Check(!(EntryKey(e) == base), "key moves with the code hash");
		e = Entry(0); e.user_data_count++;      Check(!(EntryKey(e) == base), "key moves with the user data count");
		e = Entry(0); e.code_size_words++;      Check(!(EntryKey(e) == base), "key moves with the code size");
		e = Entry(0); e.user_data_base++;       Check(!(EntryKey(e) == base), "key moves with the user data base");
		e = Entry(0); e.wave_size++;            Check(!(EntryKey(e) == base), "key moves with the wave size");
		e = Entry(0); e.maintenance8 = !e.maintenance8; Check(!(EntryKey(e) == base), "key moves with maintenance8");
		e = Entry(0); e.static_state.back()++;  Check(!(EntryKey(e) == base), "key moves with the static state");
		e = Entry(0); e.static_state.push_back(0); Check(!(EntryKey(e) == base), "key moves with the static state length");
		e = Entry(0); e.input_info[3]++;        Check(EntryKey(e) == base, "the input info bytes are not in the key (the static state stands for them)");
	}

	// refused: other fingerprint, other layout, every proper prefix, every flipped byte, bytes after the last entry under a valid sum
	Check(!ParseManifest(bytes, Key {0x1234, 0x5679}, layout, nullptr), "other fingerprint refused");
	Check(!ParseManifest(bytes, fingerprint, ManifestLayout {1001, 400, 200}, nullptr), "other vertex info size refused");
	Check(!ParseManifest(bytes, fingerprint, ManifestLayout {1000, 401, 200}, nullptr), "other pixel info size refused");
	Check(!ParseManifest(bytes, fingerprint, ManifestLayout {1000, 400, 201}, nullptr), "other compute info size refused");
	bool prefixes_refused = true;
	for (size_t n = 0; n < bytes.size(); n++) {
		prefixes_refused = prefixes_refused && !ParseManifest(std::span(bytes).first(n), fingerprint, layout, nullptr);
	}
	Check(prefixes_refused, "every proper prefix is refused");
	bool flips_refused = true;
	for (size_t n = 0; n < bytes.size(); n += 3) {
		auto damaged = bytes;
		damaged[n] ^= 0x40;
		flips_refused = flips_refused && !ParseManifest(damaged, fingerprint, layout, nullptr);
	}
	Check(flips_refused, "a damaged byte is refused (the checksum)");
	auto longer = std::vector<uint8_t>(bytes.begin(), bytes.end() - sizeof(uint64_t));
	longer.push_back(0xee);
	Check(!ParseManifest(WithChecksum(longer), fingerprint, layout, nullptr), "bytes after the last entry under a valid checksum are refused");
	// a count larger than the data under a valid sum
	auto big = std::vector<uint8_t>(bytes.begin(), bytes.end() - sizeof(uint64_t));
	const uint32_t huge = 0x7fffffffu;
	std::memcpy(big.data() + 8 + 4 + 16 + 12, &huge, sizeof(huge));
	Check(!ParseManifest(WithChecksum(big), fingerprint, layout, nullptr), "an absurd entry count is refused");
	// a count that is off by one under a valid sum
	auto off = std::vector<uint8_t>(bytes.begin(), bytes.end() - sizeof(uint64_t));
	const uint32_t four = 4;
	std::memcpy(off.data() + 8 + 4 + 16 + 12, &four, sizeof(four));
	Check(!ParseManifest(WithChecksum(off), fingerprint, layout, nullptr), "a count larger than the entries is refused");
}

void TestManifestBuilder() {
	ManifestBuilder builder;
	Check(builder.Add(Entry(0)), "first entry is new");
	Check(builder.Add(Entry(1)), "second entry is new");
	Check(!builder.Add(Entry(0)), "the same key is not added twice");
	auto same_key = Entry(0);
	same_key.input_info[0] ^= 0xff; // the info bytes carry padding noise: same key, same entry
	Check(!builder.Add(same_key), "the same key with other info bytes is not added");
	Check(builder.Size() == 2, "two entries");
	ManifestBuilder next;
	next.Load(builder.Entries());
	Check(!next.Add(Entry(1)) && next.Add(Entry(2)) && next.Size() == 3, "the entries of the previous run stay, new ones are added");
}

void TestShadowCompare() {
	const std::vector<uint32_t> a {1, 2, 3, 4};
	auto                        b = a;
	Check(CompareSpirv(a, b).empty(), "equal words are equal");
	b[2] = 9;
	Check(!CompareSpirv(a, b).empty(), "a different word is seen");
	Check(CompareSpirv(a, b).find("word 2") != std::string::npos, "the report names the word");
	b = a;
	b.push_back(5);
	Check(!CompareSpirv(a, b).empty() && !CompareSpirv(b, a).empty(), "a different length is seen");
}

// ---- mapped hook ---------------------------------------------------------------------------------------------------------------------------

void TestMappedHookOff() {
	Check(EnvMode() == Mode::Off, "the off run has the variable unset");
	const uint32_t code[4] = {1, 2, 3, 4};
	OnMapped(0x9000, code, 4);
	Check(PendingCodes() == 0, "with the variable unset the code is not kept");
	int calls = 0;
	SetMappedHook([&](uint64_t, std::shared_ptr<const std::vector<uint32_t>>) { calls++; });
	OnMapped(0x9001, code, 4);
	Check(calls == 0, "with the variable unset the hook is not called");
	SetMappedHook({});
}

void TestMappedHook() {
	Check(EnvMode() == Mode::On, "the run has KYTY_SHADER_PREWARM=1");
	std::vector<uint32_t> code {10, 20, 30, 40};
	OnMapped(0xa000, code.data(), code.size());
	OnMapped(0xa000, code.data(), code.size()); // the game registers the same shader again
	OnMapped(0xa001, code.data(), 2);
	Check(PendingCodes() == 2, "before the hook the codes are kept, a hash once");
	code[0] = 99; // the guest overwrites its memory: what was kept is a copy
	std::map<uint64_t, std::vector<uint32_t>> seen;
	std::vector<uint64_t>                     order;
	SetMappedHook([&](uint64_t hash, std::shared_ptr<const std::vector<uint32_t>> bytes) {
		seen[hash] = *bytes;
		order.push_back(hash);
	});
	Check(PendingCodes() == 0, "the replay empties the kept codes");
	Check(seen.size() == 2 && seen[0xa000] == std::vector<uint32_t>({10, 20, 30, 40}), "the replay hands over the copy made at the mapping");
	Check(seen[0xa001] == std::vector<uint32_t>({10, 20}), "the size is the size of the mapping");
	std::vector<uint32_t> later {7, 8};
	OnMapped(0xa002, later.data(), later.size());
	later[0] = 0;
	Check(seen.count(0xa002) == 1 && seen[0xa002] == std::vector<uint32_t>({7, 8}), "after the hook a mapping is handed over at once, copied");
	OnMapped(0xa002, later.data(), later.size());
	Check(order.size() == 3, "a hash is handed over once");
	SetMappedHook({});
	OnMapped(0xa003, later.data(), later.size());
	Check(PendingCodes() == 1 && order.size() == 3, "with the hook removed the code is kept again");
	OnMapped(0xa003, nullptr, 0);
	OnMapped(0xa004, later.data(), 0);
	Check(PendingCodes() == 1, "an empty code is ignored");
}

} // namespace

int main(int argc, char** argv) {
	const bool off = argc > 1 && std::strcmp(argv[1], "off") == 0;
	TestMode();
	if (off) {
		TestMappedHookOff();
	} else {
		TestMappedHook();
		TestDoneIsTakenOnce();
		TestWaitsForBuilding();
		TestCancelsQueued();
		TestWaitIsBounded();
		TestFailedWork();
		TestEveryKeyBuiltOnce();
		TestManifest();
		TestManifestBuilder();
		TestShadowCompare();
	}
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::puts("ShaderPrewarmTests: OK");
	return 0;
}
