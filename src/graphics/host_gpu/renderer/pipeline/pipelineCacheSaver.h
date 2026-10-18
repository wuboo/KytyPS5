#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECACHESAVER_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINECACHESAVER_H_

// S433 (S440) KYTY_PIPELINE_SAVE_ASYNC=0|1: the driver pipeline cache (_PipelineCache/<TITLE>.bin, ~88 MB) was fetched (vkGetPipelineCacheData
// twice), hashed and written on the command processor thread, at most every 500 ms after a new pipeline - that is, in the middle of scene
// changes. This class moves the whole job to a worker thread.
//
// Semantics: Request() hands over a job (a callable that produces header + payload, the destination path). With async off the job runs on the
// caller, exactly as before. With async on, at most one job runs at a time, one more may wait; a newer waiting job replaces the older one
// (coalescing: the cache grows monotonically, so the newest request covers the older). The file is written to <path>.tmp.* and renamed over
// <path>, so a reader (or a crash) sees the old file or the complete new one. Flush() and the destructor wait for the running and the waiting job.

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class PipelineCacheSaver {
public:
	// Fills header and payload of the file; returns false when there is nothing valid to write.
	using Build = std::function<bool(std::string& header, std::vector<uint8_t>& payload)>;
	// Called after every job (on the thread that ran it): job time in ns, bytes written, success.
	using Done = std::function<void(uint64_t ns, uint64_t bytes, bool ok)>;

	struct Stats {
		uint64_t requests  = 0;
		uint64_t saves     = 0; // jobs that wrote a file
		uint64_t failed    = 0;
		uint64_t coalesced = 0; // waiting jobs replaced by a newer one
		uint64_t bytes     = 0;
		uint64_t max_concurrent = 0; // jobs running at the same time (must be 1)
	};

	// before_rename: test hook, called with the temporary path after it is complete and before the rename.
	PipelineCacheSaver(bool async, Done done, std::function<void(const std::filesystem::path&)> before_rename = {});
	~PipelineCacheSaver();
	PipelineCacheSaver(const PipelineCacheSaver&)            = delete;
	PipelineCacheSaver& operator=(const PipelineCacheSaver&) = delete;

	void Request(const std::filesystem::path& path, Build build);
	void Flush();

	[[nodiscard]] bool  Async() const { return m_async; }
	[[nodiscard]] Stats GetStats() const;
	// True when the calling thread is the worker (tests).
	[[nodiscard]] std::thread::id WorkerId() const;

	// "0"/null -> false, "1" -> true.
	[[nodiscard]] static bool ParseAsync(const char* text);

private:
	struct Impl;
	Impl* m_impl = nullptr;
	bool  m_async;
};

} // namespace Libs::Graphics

#endif
