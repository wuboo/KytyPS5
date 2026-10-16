#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NullDescriptorPolicyTests: failed: %s\n", message);
		std::abort();
	}
}

} // namespace

static_assert(Libs::Graphics::UseNativeNullDescriptor(true, true), "a supported feature must use the native null descriptor");
static_assert(!Libs::Graphics::UseNativeNullDescriptor(true, false), "an unsupported feature must use the stand-in image");

int main() {
	using Libs::Graphics::UseNativeNullDescriptor;

	Check(UseNativeNullDescriptor(true, true), "robustness2 with nullDescriptor binds native null descriptors");
	Check(!UseNativeNullDescriptor(true, false), "robustness2 without nullDescriptor (MoltenVK) binds the stand-in image");
	Check(!UseNativeNullDescriptor(false, false), "no robustness2 extension binds the stand-in image");
	Check(!UseNativeNullDescriptor(false, true), "a feature bit without the extension enabled must not be trusted");

	std::printf("NullDescriptorPolicyTests: ok\n");
	return 0;
}
