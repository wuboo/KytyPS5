#include "graphics/shader/nullDescDiag.h"

#include <cstdio>
#include <cstdlib>

namespace {

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "NullDescDiagTests: failed: %s\n", message);
		std::exit(1);
	}
}

} // namespace

int main() {
	using Libs::Graphics::NullDesc::Classify;
	using Libs::Graphics::NullDesc::FlipWindow;
	using Libs::Graphics::NullDesc::InWindow;
	using Libs::Graphics::NullDesc::ParseFlipWindow;
	using Libs::Graphics::NullDesc::Reason;
	using Libs::Graphics::NullDesc::TextureBase40;
	using Libs::Graphics::NullDesc::TopList;

	const auto def = ParseFlipWindow(nullptr);
	Check(def.lo == 6590 && def.hi == 6600, "default window is 6590-6600");
	Check(ParseFlipWindow("").lo == 6590 && ParseFlipWindow("").hi == 6600, "empty window stays default");
	const auto span = ParseFlipWindow("100-110");
	Check(span.lo == 100 && span.hi == 110, "explicit A-B window");
	const auto one = ParseFlipWindow("5");
	Check(one.lo == 5 && one.hi == 5, "single frame window");
	const auto bad = ParseFlipWindow("nope");
	Check(bad.lo == 6590 && bad.hi == 6600, "garbage window stays default");
	const auto backwards = ParseFlipWindow("9-1");
	Check(backwards.lo == 6590 && backwards.hi == 6600, "reversed window stays default");
	Check(InWindow(span, 100) && InWindow(span, 110) && !InWindow(span, 99) && !InWindow(span, 111),
	      "window bounds are inclusive");

	Check(Classify(true, false, 3, 0) == Reason::NullTable, "nulltable classified as empty");
	Check(Classify(true, false, 3, 9) == Reason::NullTable, "null table wins over unresolved");
	Check(Classify(false, false, 2, 0) == Reason::NullTable, "null table wins over an invalid T#");
	Check(Classify(true, false, 0, 4) == Reason::Unresolved, "unresolved address of a null T#");
	Check(Classify(false, false, 0, 0) == Reason::FilteredValid, "invalid non-null T# is filtered_valid");
	Check(Classify(true, false, 0, 0) == Reason::ZeroMem, "game null T# is zero_mem");
	Check(Classify(false, true, 0, 0, false, true) == Reason::FilteredDim, "dimension reject keeps the T#");
	Check(Classify(true, false, 0, 0, true) == Reason::Inactive, "inactive source is not zero_mem");
	Check(Classify(false, true, 1, 1) == Reason::None, "a valid T# is not a stand-in");

	uint32_t words[2] = {0, 0};
	Check(TextureBase40(words) == 0, "zero dwords are a null base");
	words[0] = 0x11;
	words[1] = 0x22;
	Check(TextureBase40(words) == (((0x11ull | (0x22ull << 32u)) & 0xffffffffffull) << 8u),
	      "base is the low 40 bits shifted by 8");
	Check(TextureBase40(nullptr) == 0, "missing dwords are a null base");

	TopList top;
	top.Add(0xaaa, 1, 0x1000, Reason::EmptyT);
	top.Add(0xbbb, 2, 0x2000, Reason::NullTable);
	top.Add(0xbbb, 2, 0x2000, Reason::NullTable);
	top.Add(0xccc, 3, 0x3000, Reason::Other);
	top.Add(0xccc, 3, 0x3000, Reason::Other);
	top.Add(0xccc, 3, 0x3000, Reason::Other);
	const auto ranked = top.Top(2);
	Check(ranked.size() == 2, "top limit");
	Check(ranked[0].shader == 0xccc && ranked[0].n == 3 && ranked[0].reason == Reason::Other, "highest count first");
	Check(ranked[1].shader == 0xbbb && ranked[1].slot == 2 && ranked[1].addr == 0x2000 && ranked[1].n == 2,
	      "second count keeps slot and address");

	std::printf("NullDescDiagTests: ok\n");
	return 0;
}
