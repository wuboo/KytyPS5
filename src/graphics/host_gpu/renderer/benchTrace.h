#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_BENCHTRACE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_BENCHTRACE_H_

#include <cstdint>

namespace Libs::Graphics::BenchTrace {

// Bench diagnostic: remembers the last GPU operations with the scheduler tick that carries
// them, so a stuck tick can be matched to the shaders it runs.
void Record(uint64_t tick, const char* kind, uint64_t hash0, uint64_t hash1);
void Dump(uint64_t tick);
// True when KYTY_BENCH_SKIP_CS lists this compute shader hash (hex, comma separated).
bool SkipCompute(uint64_t shader_hash);
// The main scheduler's tick being recorded (set by CommandScheduler::BeginNext).
void     SetTick(uint64_t tick);
uint64_t Tick();

} // namespace Libs::Graphics::BenchTrace

// Records the source line of a GPU command about to be recorded into the current tick.
#define KYTY_BENCH_TRACE_SITE()                                                                    \
	::Libs::Graphics::BenchTrace::Record(::Libs::Graphics::BenchTrace::Tick(), __FILE__, __LINE__, 0)

namespace Libs::Graphics::BenchTrace {

} // namespace Libs::Graphics::BenchTrace

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_BENCHTRACE_H_ */
