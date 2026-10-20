#include "graphics/shader/shaderCompileWorkers.h"

#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace Libs::Graphics::ShaderCompileWorkers {

namespace {

unsigned ParseWorkers() {
#ifdef KYTY_COMPILE_WORKERS_MUTANT_SYNC
	return 0;
#endif
	if (const char* text = std::getenv("KYTY_COMPILE_WORKERS"); text != nullptr && *text != 0) {
		const unsigned n = static_cast<unsigned>(std::strtoul(text, nullptr, 10));
		if (n > 0 && n <= 32) return n;
	}
	return 0;
}

bool ParseManifest() {
	if (const char* text = std::getenv("KYTY_SHADER_MANIFEST"); text != nullptr) return std::strcmp(text, "1") == 0;
	return false;
}

} // namespace

unsigned WorkerCount() {
	static const unsigned count = ParseWorkers();
	return count;
}

bool ManifestEnabled() {
	static const bool enabled = ParseManifest();
	return enabled;
}

unsigned PrewarmThreadCount() {
	const unsigned workers = WorkerCount();
	return workers > 0 ? workers : 1u;
}

struct Pool::Impl {
	explicit Impl(unsigned workers) {
		for (unsigned i = 0; i < workers; i++) threads.emplace_back([this] { RunLoop(); });
	}
	~Impl() {
		{
			std::scoped_lock lock(mutex);
			stop = true;
		}
		wake.notify_all();
		for (auto& thread: threads) thread.join();
	}

	void Enqueue(std::function<void()> job) {
		{
			std::scoped_lock lock(mutex);
			queue.push_back(std::move(job));
		}
		wake.notify_one();
	}

	void RunLoop() {
		for (;;) {
			std::function<void()> job;
			{
				std::unique_lock lock(mutex);
				wake.wait(lock, [&] { return stop || !queue.empty(); });
				if (stop && queue.empty()) return;
				job = std::move(queue.front());
				queue.pop_front();
			}
			job();
		}
	}

	std::mutex                        mutex;
	std::condition_variable           wake;
	std::deque<std::function<void()>> queue;
	std::vector<std::thread>          threads;
	bool                              stop = false;
};

Pool::Pool(): m_impl(WorkerCount() > 0 ? new Impl(WorkerCount()) : nullptr) {}

Pool::~Pool() {
	delete m_impl;
}

Pool& Pool::Instance() {
	static Pool pool;
	return pool;
}

void Pool::Enqueue(std::function<void()> job) {
	if (m_impl == nullptr) {
		job();
		return;
	}
	m_impl->Enqueue(std::move(job));
}

} // namespace Libs::Graphics::ShaderCompileWorkers
