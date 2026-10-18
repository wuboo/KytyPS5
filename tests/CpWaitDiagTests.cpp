#include "graphics/host_gpu/renderer/cpWaitDiag.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using namespace Libs::Graphics::CpWait;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "CpWaitDiagTests: failed: %s\n", message);
		std::exit(1);
	}
}

bool Has(const std::string& text, const char* part) {
	return text.find(part) != std::string::npos;
}

// The tag is the innermost scope, leaving restores the outer one, and outside every scope it is `other`.
void TestCauseScopeNesting() {
	Check(CurrentCause() == Cause::Other, "no scope: other");
	{
		CauseScope outer(Cause::RfGpu);
		Check(CurrentCause() == Cause::RfGpu, "outer scope sets the cause");
		{
			CauseScope inner(Cause::Eager);
			Check(CurrentCause() == Cause::Eager, "inner scope wins");
			{
				CauseScope innermost(Cause::Stream);
				Check(CurrentCause() == Cause::Stream, "three deep");
			}
			Check(CurrentCause() == Cause::Eager, "leaving restores the inner tag");
		}
		Check(CurrentCause() == Cause::RfGpu, "leaving restores the outer tag");
	}
	Check(CurrentCause() == Cause::Other, "outside every scope: other again");
}

void TestCauseOfSite() {
	using Libs::Graphics::SyncCensus::Site;
	Check(CauseOfSite(Site::RfGpu) == Cause::RfGpu && CauseOfSite(Site::Spec) == Cause::Spec && CauseOfSite(Site::Dcc) == Cause::Dcc,
	      "rf_gpu, spec, dcc keep their names");
	Check(CauseOfSite(Site::Other) == Cause::ReadMem && CauseOfSite(Site::WriteFault) == Cause::ReadMem, "other explicit readbacks");
	{
		CauseScope scope(CauseOfSite(Site::Spec));
		Check(CurrentCause() == Cause::Spec, "ReadMemory-style scope from a SyncCensus site");
	}
}

void TestLineAndGate() {
	Event event;
	event.flip    = 6305;
	event.opcode  = 0x25;
	event.site    = "pv.index_hash";
	event.cause   = Cause::Eager;
	event.wait_us = 9240;
	event.lag     = 3;
	Check(FormatLine(event) == "[cp-wait] flip=6305 op=0x25 site=pv.index_hash cause=eager wait_us=9240 lag=3", "line format");
	Event none;
	Check(Has(FormatLine(none), " op=none site=none cause=other "), "defaults");
	Check(!ShouldLog(0, 100, 999) && ShouldLog(0, 100, 1000), "only waits of 1 ms and more");
	Check(ShouldLog(399, 100, 5000) && !ShouldLog(400, 100, 5000) && ShouldLog(5000, 6305, 5000), "400 lines plus flips 6300-6310");
}

void TestNames() {
	for (uint32_t cause = 0; cause < kCauseCount; cause++) {
		const std::string column = ColumnName(cause);
		Check(column == std::string("cp_wait_cause_") + CauseName(static_cast<Cause>(cause)) + "_ns", "column follows the cause name");
	}
}

} // namespace

int main() {
	TestCauseScopeNesting();
	TestCauseOfSite();
	TestLineAndGate();
	TestNames();
	return 0;
}
