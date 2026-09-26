#ifndef KYTY_GRAPHICS_HOST_GPU_RENDERER_BENCHTRACE_H_
#define KYTY_GRAPHICS_HOST_GPU_RENDERER_BENCHTRACE_H_

#include <cstdint>

namespace Libs::Graphics::BenchTrace {

// Bench diagnostic: remembers the last GPU operations with the scheduler tick that carries
// them, so a stuck tick can be matched to the shaders it runs.
void Record(uint64_t tick, const char* kind, uint64_t hash0, uint64_t hash1);
void Dump(uint64_t tick);

} // namespace Libs::Graphics::BenchTrace

#endif /* KYTY_GRAPHICS_HOST_GPU_RENDERER_BENCHTRACE_H_ */
