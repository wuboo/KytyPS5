// S455: KYTY_COMPILE_WORKERS thread pool and KYTY_SHADER_MANIFEST prewarm enable. Mutant: KYTY_COMPILE_WORKERS_MUTANT_SYNC forces sync path.

#include "graphics/shader/shaderCompileWorkers.h"
#include "graphics/shader/shaderPrewarm.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>

using namespace Libs::Graphics::ShaderCompileWorkers;
using namespace Libs::Graphics::ShaderPrewarm;

namespace {

int g_failures = 0;

void Check(bool condition, const char* message) {
	if (!condition) {
		std::fprintf(stderr, "FAIL: %s\n", message);
		g_failures++;
	}
}

void TestManifestEnablesPrewarm() {
#ifdef KYTY_COMPILE_WORKERS_MUTANT_SYNC
	Check(WorkerCount() == 0, "mutant sync: workers off");
	Check(PrewarmThreadCount() == 4, "mutant must fail: prewarm threads ignore env");
#else
	Check(WorkerCount() == 4, "KYTY_COMPILE_WORKERS=4");
	Check(ManifestEnabled(), "KYTY_SHADER_MANIFEST=1");
	Check(EnvMode() == Mode::On, "manifest alone enables prewarm on");
	Check(PrewarmThreadCount() == 4, "prewarm threads follow workers");
#endif
}

void TestPoolRunsOffCallerThread() {
	std::atomic<std::thread::id> ran_on {};
	const auto                   self = std::this_thread::get_id();
	Pool::Instance().Run([&] {
		ran_on = std::this_thread::get_id();
		return 42;
	});
#ifdef KYTY_COMPILE_WORKERS_MUTANT_SYNC
	Check(ran_on == self, "mutant sync: job stays on caller");
#else
	Check(ran_on != std::thread::id() && ran_on != self, "pool job runs on a worker");
#endif
	Check(Pool::Instance().Run([] { return 7; }) == 7, "pool returns the result");
}

} // namespace

int main(int argc, char** argv) {
	(void)argc;
	(void)argv;
	TestManifestEnablesPrewarm();
	TestPoolRunsOffCallerThread();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d failure(s)\n", g_failures);
		return 1;
	}
	std::fprintf(stderr, "OK\n");
	return 0;
}
