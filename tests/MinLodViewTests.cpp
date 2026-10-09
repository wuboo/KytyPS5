#include "graphics/host_gpu/renderer/image/minLodView.h"

#include <cstdio>
#include <cstdlib>

namespace {

using namespace Libs::Graphics;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "MinLodViewTests: failed: %s\n", message);
		std::exit(1);
	}
}

bool Is(const MinLodViewRange& r, uint32_t base, uint32_t count, uint32_t min_lod) {
	return r.base_level == base && r.level_count == count && r.min_lod == min_lod;
}

// S126: MinLod (U4.8, relative to the view base) -> view base level.
void TestMapping() {
	Check(Is(MapMinLodToBaseLevel(0, 10, 0), 0, 10, 0), "MinLod 0 must leave the view alone");
	Check(Is(MapMinLodToBaseLevel(3, 5, 0), 3, 5, 0), "MinLod 0 with a base must leave the view alone");
	Check(Is(MapMinLodToBaseLevel(0, 10, 256), 1, 9, 0), "MinLod 256 = one level up");
	Check(Is(MapMinLodToBaseLevel(0, 10, 1024), 4, 6, 0), "MinLod 1024 = four levels up");
	Check(Is(MapMinLodToBaseLevel(0, 10, 1100), 5, 5, 0), "MinLod 1100 (4.29) rounds up to 5");
	Check(Is(MapMinLodToBaseLevel(0, 10, 1), 1, 9, 0), "any fraction rounds up");
	Check(Is(MapMinLodToBaseLevel(2, 6, 512), 4, 4, 0), "shift is added to the view base");
	Check(Is(MapMinLodToBaseLevel(0, 4, 256 * 9), 3, 1, 0), "shift stops at the last level");
	Check(Is(MapMinLodToBaseLevel(0, 1, 256), 0, 1, 0), "single-level view keeps its level");
	Check(Is(MapMinLodToBaseLevel(0, 0, 256), 0, 0, 256), "empty view is left for the caller to reject");
}

void TestDropped() {
	Check(MinLodDropped(true, false, false, false, 256), "Apple without remap drops the clamp");
	Check(!MinLodDropped(true, true, false, false, 256), "remapped clamp is not dropped");
	Check(!MinLodDropped(false, false, false, false, 256), "native hosts keep the clamp");
	Check(!MinLodDropped(true, false, true, false, 256), "Dynamic mips clamp in the shader");
	Check(!MinLodDropped(true, false, false, true, 256), "storage views have no clamp");
	Check(!MinLodDropped(true, false, false, false, 0), "MinLod 0 is nothing to drop");
}

} // namespace

int main() {
	TestMapping();
	TestDropped();
	std::puts("MinLodViewTests: ok");
	return 0;
}
