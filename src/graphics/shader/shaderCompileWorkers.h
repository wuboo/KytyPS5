#ifndef KYTY_GRAPHICS_SHADER_SHADERCOMPILEWORKERS_H_
#define KYTY_GRAPHICS_SHADER_SHADERCOMPILEWORKERS_H_

// S455 (E1): KYTY_COMPILE_WORKERS=<n> (default 0 = legacy). When n>0, shader prewarm uses n worker threads and the heavy
// translate / SPIR-V emit steps on a command-processor miss may run on the pool (CP waits for the result).
// KYTY_SHADER_MANIFEST=1: enable manifest load/save like KYTY_SHADER_PREWARM=1 when prewarm is unset (see shaderPrewarm.cpp).

#include <cstdint>
#include <functional>
#include <future>
#include <type_traits>

namespace Libs::Graphics::ShaderCompileWorkers {

[[nodiscard]] unsigned WorkerCount();
[[nodiscard]] bool     ManifestEnabled();
[[nodiscard]] unsigned PrewarmThreadCount();

class Pool {
public:
	[[nodiscard]] static Pool& Instance();

	template <typename Fn>
	[[nodiscard]] std::invoke_result_t<Fn> Run(Fn&& fn) {
		using Result = std::invoke_result_t<Fn>;
		if (WorkerCount() == 0) return fn();
		std::packaged_task<Result()> task(std::forward<Fn>(fn));
		auto                         future = task.get_future();
		Enqueue([&task]() { task(); });
		return future.get();
	}

private:
	Pool();
	~Pool();
	Pool(const Pool&)            = delete;
	Pool& operator=(const Pool&) = delete;

	void Enqueue(std::function<void()> job);

	struct Impl;
	Impl* m_impl;
};

} // namespace Libs::Graphics::ShaderCompileWorkers

#endif /* KYTY_GRAPHICS_SHADER_SHADERCOMPILEWORKERS_H_ */
