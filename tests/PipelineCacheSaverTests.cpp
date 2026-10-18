// S433 (S440): KYTY_PIPELINE_SAVE_ASYNC. The driver pipeline cache save runs on a worker thread, one file write at a time, newer requests replace a
// waiting older one, the file is replaced atomically (tmp + rename), nothing is lost on Flush/destruction. Mutants: see PipelineCacheSaver.cpp
// (KYTY_PCSAVER_MUTANT_{SAVE_ON_CALLER, NO_RENAME, NO_COALESCE, DROP_FLUSH}).

#include "graphics/host_gpu/renderer/pipeline/pipelineCacheSaver.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

using Libs::Graphics::PipelineCacheSaver;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

std::string ReadAll(const std::filesystem::path& path) {
	std::ifstream in(path, std::ios::binary);
	return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

PipelineCacheSaver::Build Constant(std::string header, std::string payload, std::thread::id* ran_on = nullptr) {
	return [=](std::string& h, std::vector<uint8_t>& p) {
		if (ran_on != nullptr) {
			*ran_on = std::this_thread::get_id();
		}
		h = header;
		p.assign(payload.begin(), payload.end());
		return true;
	};
}

struct Latch {
	std::mutex              m;
	std::condition_variable cv;
	bool                    open = false;
	void Wait() {
		std::unique_lock lock(m);
		cv.wait_for(lock, std::chrono::seconds(2), [this] { return open; }); // bounded: a mutant that runs the job on the caller must fail, not hang
	}
	void Open() {
		{
			std::lock_guard lock(m);
			open = true;
		}
		cv.notify_all();
	}
};

size_t CountTemps(const std::filesystem::path& dir) {
	size_t n = 0;
	for (const auto& entry : std::filesystem::directory_iterator(dir)) {
		if (entry.path().filename().string().find(".tmp.") != std::string::npos) {
			n++;
		}
	}
	return n;
}

void TestAsyncRunsOffCaller(const std::filesystem::path& dir) {
	std::atomic<uint64_t> done_n {0}, done_bytes {0};
	PipelineCacheSaver    saver(true, [&](uint64_t, uint64_t bytes, bool ok) {
        if (ok) {
            done_n++;
            done_bytes += bytes;
        }
    });
	std::thread::id       ran_on;
	const auto            path = dir / "a.bin";
	saver.Request(path, Constant("HDR", "payload-1", &ran_on));
	saver.Flush();
	Check(ran_on != std::thread::id {}, "build was not run");
	Check(ran_on != std::this_thread::get_id(), "async build ran on the calling thread");
	Check(ran_on == saver.WorkerId(), "async build did not run on the worker");
	Check(ReadAll(path) == "HDRpayload-1", "file content");
	Check(done_n == 1 && done_bytes == 12, "done callback counters");
	Check(CountTemps(dir) == 0, "temporary file left behind");
}

void TestSyncRunsOnCaller(const std::filesystem::path& dir) {
	PipelineCacheSaver saver(false, {});
	std::thread::id    ran_on;
	saver.Request(dir / "s.bin", Constant("H", "p", &ran_on));
	Check(ran_on == std::this_thread::get_id(), "sync build must run on the calling thread");
	Check(ReadAll(dir / "s.bin") == "Hp", "sync file content");
}

void TestCoalescing(const std::filesystem::path& dir) {
	Latch              gate;
	std::atomic<bool>  first_started {false};
	PipelineCacheSaver saver(true, {});
	const auto         path = dir / "c.bin";
	saver.Request(path, [&](std::string& h, std::vector<uint8_t>& p) {
		first_started = true;
		gate.Wait(); // the first job is held on the worker
		h = "H";
		p = {'1'};
		return true;
	});
	while (!first_started) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	for (char c = '2'; c <= '6'; c++) {
		saver.Request(path, Constant("H", std::string(1, c)));
	}
	gate.Open();
	saver.Flush();
	const auto stats = saver.GetStats();
	Check(stats.requests == 6, "requests");
	Check(stats.saves == 2, "coalescing: first job + only the newest waiting one are written");
	Check(stats.coalesced == 4, "coalescing: four waiting jobs replaced");
	Check(ReadAll(path) == "H6", "the newest request wins");
	Check(stats.max_concurrent == 1, "never more than one write at a time");
}

void TestAtomicReplace(const std::filesystem::path& dir) {
	const auto path = dir / "t.bin";
	{
		std::ofstream out(path, std::ios::binary);
		out << "OLD-COMPLETE";
	}
	std::string seen_at_hook;
	bool        tmp_is_other = false;
	PipelineCacheSaver saver(true, {}, [&](const std::filesystem::path& tmp) {
		seen_at_hook = ReadAll(path); // the destination while the new file is half written
		tmp_is_other = tmp != path;
	});
	saver.Request(path, Constant("NEWHDR", std::string(4096, 'x')));
	saver.Flush();
	Check(seen_at_hook == "OLD-COMPLETE", "destination changed before the rename (not atomic)");
	Check(tmp_is_other, "the file was written in place");
	Check(ReadAll(path) == "NEWHDR" + std::string(4096, 'x'), "new content after rename");
	Check(CountTemps(dir) == 0, "temporary file left behind after rename");
}

void TestFailureKeepsOldFile(const std::filesystem::path& dir) {
	const auto path = dir / "f.bin";
	{
		std::ofstream out(path, std::ios::binary);
		out << "KEEP";
	}
	std::atomic<int> failures {0};
	PipelineCacheSaver saver(true, [&](uint64_t, uint64_t, bool ok) { failures += ok ? 0 : 1; });
	saver.Request(path, [](std::string&, std::vector<uint8_t>&) { return false; });
	saver.Flush();
	Check(failures == 1, "failure reported");
	Check(ReadAll(path) == "KEEP", "a failed build must not touch the old file");
	Check(saver.GetStats().failed == 1, "failed counter");
}

void TestFlushOnDestruction(const std::filesystem::path& dir) {
	const auto path = dir / "d.bin";
	{
		Latch              gate;
		PipelineCacheSaver saver(true, {});
		std::atomic<bool>  started {false};
		saver.Request(path, [&](std::string& h, std::vector<uint8_t>& p) {
			started = true;
			std::this_thread::sleep_for(std::chrono::milliseconds(150));
			h = "END";
			p = {'!'};
			return true;
		});
		while (!started) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		saver.Request(path, Constant("END", "?")); // waiting when the destructor runs
	}
	Check(ReadAll(path) == "END?", "destruction must flush the running and the waiting job");
}

void TestParse() {
	Check(!PipelineCacheSaver::ParseAsync(nullptr), "default is off");
	Check(!PipelineCacheSaver::ParseAsync("0"), "0 is off");
	Check(PipelineCacheSaver::ParseAsync("1"), "1 is on");
	Check(!PipelineCacheSaver::ParseAsync("yes"), "other values are off");
}

} // namespace

int main() {
	const auto dir = std::filesystem::temp_directory_path() / ("kyty_pcsaver_" + std::to_string(::getpid()));
	std::filesystem::create_directories(dir);

	TestAsyncRunsOffCaller(dir);
	TestSyncRunsOnCaller(dir);
	TestCoalescing(dir);
	TestAtomicReplace(dir);
	TestFailureKeepsOldFile(dir);
	TestFlushOnDestruction(dir);
	TestParse();

	std::error_code error;
	std::filesystem::remove_all(dir, error);
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::printf("pipeline_cache_saver: OK\n");
	return 0;
}
