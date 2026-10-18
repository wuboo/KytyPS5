#include "graphics/host_gpu/renderer/pipeline/pipelineCacheSaver.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <optional>
#include <system_error>

#ifndef _WIN32
#include <unistd.h>
#endif

namespace Libs::Graphics {

namespace {
struct Job {
	std::filesystem::path path;
	PipelineCacheSaver::Build build;
};

std::atomic<uint64_t> g_tmp_counter {0};

uint64_t NowNs() {
	return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
	                                 std::chrono::steady_clock::now().time_since_epoch())
	                                 .count());
}
} // namespace

struct PipelineCacheSaver::Impl {
	PipelineCacheSaver::Done                          done;
	std::function<void(const std::filesystem::path&)> before_rename;

	std::mutex              mutex;
	std::condition_variable cv;
	std::optional<Job>      waiting;
	bool                    running = false;
	bool                    stop    = false;
	std::thread             worker;
	std::atomic<uint64_t>   active {0};
	Stats                   stats;

	// Returns the number of bytes written, or nullopt on failure. Runs on the caller (sync) or the worker.
	std::optional<uint64_t> Write(const Job& job) {
		std::string          header;
		std::vector<uint8_t> payload;
		if (!job.build(header, payload)) {
			return std::nullopt;
		}
		std::error_code error;
		if (!job.path.parent_path().empty()) {
			std::filesystem::create_directories(job.path.parent_path(), error);
			if (error) {
				return std::nullopt;
			}
		}
#ifdef KYTY_PCSAVER_MUTANT_NO_RENAME
		const auto temp_path = job.path;
#else
		const auto temp_path = std::filesystem::path(
		    job.path.string() + ".tmp." + std::to_string(g_tmp_counter.fetch_add(1)));
#endif
		std::FILE* file = std::fopen(temp_path.string().c_str(), "wb");
		if (file == nullptr) {
			return std::nullopt;
		}
		bool ok = std::fwrite(header.data(), 1, header.size(), file) == header.size();
		if (before_rename) {
			// Let the hook see a half-written state: header out, payload not yet.
			std::fflush(file);
			before_rename(temp_path);
		}
		ok = ok && std::fwrite(payload.data(), 1, payload.size(), file) == payload.size();
		ok = ok && std::fflush(file) == 0;
#ifndef _WIN32
		ok = ok && ::fsync(::fileno(file)) == 0;
#endif
		ok = (std::fclose(file) == 0) && ok;
		if (ok) {
#ifndef KYTY_PCSAVER_MUTANT_NO_RENAME
			std::filesystem::rename(temp_path, job.path, error);
			ok = !error;
#endif
		}
		if (!ok) {
			std::filesystem::remove(temp_path, error);
			return std::nullopt;
		}
		return static_cast<uint64_t>(header.size() + payload.size());
	}

	void Run(const Job& job) {
		const uint64_t concurrent = active.fetch_add(1) + 1;
		const uint64_t begin      = NowNs();
		const auto     bytes      = Write(job);
		const uint64_t ns         = NowNs() - begin;
		active.fetch_sub(1);
		{
			std::lock_guard lock(mutex);
			if (concurrent > stats.max_concurrent) {
				stats.max_concurrent = concurrent;
			}
			if (bytes) {
				stats.saves++;
				stats.bytes += *bytes;
			} else {
				stats.failed++;
			}
		}
		if (done) {
			done(ns, bytes.value_or(0), bytes.has_value());
		}
	}

	void WorkerLoop() {
		std::unique_lock lock(mutex);
		for (;;) {
			cv.wait(lock, [this] { return stop || waiting.has_value(); });
			if (!waiting.has_value()) {
				return; // stop and nothing left
			}
			Job job = std::move(*waiting);
			waiting.reset();
			running = true;
			lock.unlock();
			Run(job);
			lock.lock();
			running = false;
			cv.notify_all();
		}
	}
};

PipelineCacheSaver::PipelineCacheSaver(bool async, Done done,
                                       std::function<void(const std::filesystem::path&)> before_rename)
    : m_impl(new Impl), m_async(async) {
	m_impl->done          = std::move(done);
	m_impl->before_rename = std::move(before_rename);
}

PipelineCacheSaver::~PipelineCacheSaver() {
#ifndef KYTY_PCSAVER_MUTANT_DROP_FLUSH
	Flush();
#endif
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->stop = true;
		m_impl->waiting.reset();
	}
	m_impl->cv.notify_all();
	if (m_impl->worker.joinable()) {
		m_impl->worker.join();
	}
	delete m_impl;
}

void PipelineCacheSaver::Request(const std::filesystem::path& path, Build build) {
#ifdef KYTY_PCSAVER_MUTANT_SAVE_ON_CALLER
	const bool async = false;
#else
	const bool async = m_async;
#endif
	if (!async) {
		{
			std::lock_guard lock(m_impl->mutex);
			m_impl->stats.requests++;
		}
		m_impl->Run(Job {path, std::move(build)});
		return;
	}
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->stats.requests++;
		if (!m_impl->worker.joinable()) {
			m_impl->worker = std::thread([impl = m_impl] { impl->WorkerLoop(); });
		}
#ifdef KYTY_PCSAVER_MUTANT_NO_COALESCE
		// Mutant: the oldest waiting job stays, newer requests are dropped.
		if (m_impl->waiting.has_value()) {
			return;
		}
#else
		if (m_impl->waiting.has_value()) {
			m_impl->stats.coalesced++;
		}
#endif
		m_impl->waiting = Job {path, std::move(build)};
	}
	m_impl->cv.notify_all();
}

void PipelineCacheSaver::Flush() {
	std::unique_lock lock(m_impl->mutex);
	m_impl->cv.wait(lock, [this] { return !m_impl->running && !m_impl->waiting.has_value(); });
}

PipelineCacheSaver::Stats PipelineCacheSaver::GetStats() const {
	std::lock_guard lock(m_impl->mutex);
	return m_impl->stats;
}

std::thread::id PipelineCacheSaver::WorkerId() const {
	std::lock_guard lock(m_impl->mutex);
	return m_impl->worker.get_id();
}

bool PipelineCacheSaver::ParseAsync(const char* text) {
	return text != nullptr && std::strcmp(text, "1") == 0;
}

} // namespace Libs::Graphics
