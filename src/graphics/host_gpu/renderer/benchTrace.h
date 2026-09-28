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
// True when KYTY_BENCH_SKIP_DRAW lists the draw's vertex or pixel shader hash (hex, comma sep.).
bool SkipDraw(uint64_t vs_hash, uint64_t ps_hash);
// True when KYTY_BENCH_SKIP_BATCH lists the draw's vertex or pixel shader hash (hex, comma
// sep.): keeps the draw itself (unlike SkipDraw), only excludes it from prototype_batch.
bool SkipBatch(uint64_t vs_hash, uint64_t ps_hash);
// The main scheduler's tick being recorded (set by CommandScheduler::BeginNext).
void     SetTick(uint64_t tick);
uint64_t Tick();

// GPU-thread "current operation" (the draw/dispatch/copy being prepared), and a ring of GPU
// writes to guest memory tagged with it, so a readback can name what wrote the range.
void SetCurrent(const char* kind, uint64_t hash0, uint64_t hash1);
void RecordGpuWrite(uint64_t vaddr, uint64_t size);
// Logs the newest recorded writer overlapping [vaddr, vaddr + size) (rate-limited per writer).
void LogGpuWriter(const char* reason, uint64_t vaddr, uint64_t size);

} // namespace Libs::Graphics::BenchTrace

// Records the source line of a GPU command about to be recorded into the current tick.
#define KYTY_BENCH_TRACE_SITE()                                                                    \
	::Libs::Graphics::BenchTrace::Record(::Libs::Graphics::BenchTrace::Tick(), __FILE__, __LINE__, 0)

namespace Libs::Graphics::BenchTrace {

} // namespace Libs::Graphics::BenchTrace

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_BENCHTRACE_H_ */
