#include "graphics/host_gpu/renderer/eagerSubmit.h"

#include <cstdio>
#include <cstdlib>

// S326: policy of the eager submit (ShouldEagerSplit). Built four times: normally and with KYTY_EAGER_MUTANT_PASS_OPEN / _BUSY / _NO_AGE.
// The mutant builds must FAIL (ctest WILL_FAIL).

namespace {

using Libs::Graphics::Sync::EagerQuery;
using Libs::Graphics::Sync::ShouldEagerSplit;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "EagerSubmitTests: failed: %s\n", message);
		std::exit(1); // not abort(): ctest WILL_FAIL does not invert signals
	}
}

EagerQuery Ready() {
	EagerQuery q;
	q.enabled    = true;
	q.has_work   = true;
	q.gpu_idle   = true;
	q.pass_open  = false;
	q.age_us     = 500;
	q.min_age_us = 300;
	return q;
}

} // namespace

int main() {
	Check(ShouldEagerSplit(Ready()), "idle GPU, closed pass, work, old enough: split");
	{
		auto q = Ready();
		q.enabled = false;
		Check(!ShouldEagerSplit(q), "disabled never splits");
	}
	{
		auto q = Ready();
		q.has_work = false;
		Check(!ShouldEagerSplit(q), "empty buffer never splits");
	}
	{
		auto q = Ready();
		q.pass_open = true;
		Check(!ShouldEagerSplit(q), "open pass: no split");
	}
	{
		auto q = Ready();
		q.gpu_idle = false;
		Check(!ShouldEagerSplit(q), "busy GPU: no split");
	}
	{
		auto q = Ready();
		q.age_us = 299;
		Check(!ShouldEagerSplit(q), "younger than the threshold: no split");
		q.age_us = 300;
		Check(ShouldEagerSplit(q), "exactly the threshold: split");
	}
	std::printf("EagerSubmitTests: ok\n");
	return 0;
}
