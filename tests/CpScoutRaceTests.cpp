// S478: the scout worker must not write members of the CommandProcessor (index type / base / size, instance count, indirect bases). WriteCpMember is the gate every such setter
// goes through. Mutant RACE_WRITES: the gate lets the worker write (WILL_FAIL).

#include "graphics/guest_gpu/command_processor/cpScout.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>

namespace {

using namespace Libs::Graphics::CpScout;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpScoutRaceTests: failed: %s\n", text);
		std::exit(1);
	}
}

// the shape of the bug: a CP that draws with the index type the last SET wrote, and a lookahead that has already run to the next draw's SET
struct FakeCp {
	uint32_t index_type = 0; // 16 bit
	uint32_t instances  = 1;
	void     SetIndexType(uint32_t v, bool on_worker, bool fix) { WriteCpMember(index_type, v & 3u, on_worker, fix); }
	void     SetInstances(uint32_t v, bool on_worker, bool fix) { WriteCpMember(instances, v, on_worker, fix); }
};

} // namespace

int main() {
	// the CP's own thread always writes
	{
		FakeCp cp;
		cp.SetIndexType(1, false, true);
		cp.SetInstances(7, false, true);
		Check(cp.index_type == 1 && cp.instances == 7, "the CP thread writes its own members");
	}
	// the worker, fix on: the CP's values stay, however far ahead the lookahead is
	{
		FakeCp cp;
		cp.SetIndexType(0, false, true);
		cp.SetInstances(1, false, true);
		std::thread worker([&] {
			cp.SetIndexType(1, true, true); // the next draw is 32 bit
			cp.SetInstances(500, true, true);
		});
		worker.join();
		Check(cp.index_type == 0, "worker: the CP's index type is untouched");
		Check(cp.instances == 1, "worker: the CP's instance count is untouched");
	}
	// the worker, fix off (KYTY_SCOUT_RACE_FIX=0): the old behaviour, which this test documents
	{
		FakeCp cp;
		cp.SetIndexType(0, false, true);
		cp.SetIndexType(1, true, false);
		Check(cp.index_type == 1, "worker, fix off: the write goes through (legacy)");
	}
	// the gate reports what it did
	{
		uint64_t member = 5;
		Check(!WriteCpMember(member, uint64_t {9}, true, true) && member == 5, "gate refuses the worker");
		Check(WriteCpMember(member, uint64_t {9}, false, true) && member == 9, "gate lets the CP through");
	}
	std::puts("CpScoutRaceTests: ok");
	return 0;
}
