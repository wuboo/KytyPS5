#include "graphics/guest_gpu/lodStats.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using namespace Libs::Graphics::LodStats;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "LodStatsTests: failed: %s\n", message);
		std::exit(1);
	}
}

uint32_t Dword(const std::vector<uint8_t>& b, uint32_t offset) {
	uint32_t v = 0;
	std::memcpy(&v, b.data() + offset, 4);
	return v;
}

uint32_t Counter(const std::vector<uint8_t>& b, uint32_t slot, uint32_t which) {
	return Dword(b, kFirstCounter + slot * kCounterBytes + which * 4u);
}

void TestMode() {
	Check(ParseMode(nullptr) == Mode::Real, "unset = real (default)");
	Check(ParseMode("zero") == Mode::Zero, "zero");
	Check(ParseMode("full") == Mode::Full, "full");
	Check(ParseMode("real") == Mode::Real, "real");
	Check(ParseMode("fullx") == Mode::Zero, "only exact names");
	Check(ParseMode("") == Mode::Zero, "empty = zero");
}

void TestZero() {
	std::vector<uint8_t> b(kReportBytes, 0xAA);
	Check(!WriteReport(Mode::Zero, b.data(), static_cast<uint32_t>(b.size()), nullptr), "zero writes the stub only");
	Check(Dword(b, 0) == 1, "stub label = 1");
	Check(Dword(b, 4) == 0 && Counter(b, 255, 1) == 0, "rest zeroed");
}

void TestFull() {
	std::vector<uint8_t> b(kReportBytes + 128, 0xAA);
	Check(WriteReport(Mode::Full, b.data(), static_cast<uint32_t>(b.size()), nullptr), "full report written");
	Check(Dword(b, 0) == kFirstCounter, "header points at the first counter");
	Check(Counter(b, 7, 0) == (0x10000u | (7u << 24)), "dword0 = samples | slot");
	Check(Counter(b, 255, 0) >> 24 == 255u, "last slot id");
	Check(Counter(b, 7, 1) == 0x10000u, "dword1: finest mip 0");
	Check(Dword(b, kReportBytes) == 0, "bytes past the counters are zero");
}

void TestReal() {
	std::vector<uint32_t> uses(kSlotCount, 0);
	uses[3] = 2;
	uses[9] = 5000; // saturates
	std::vector<uint8_t> b(kReportBytes);
	Check(WriteReport(Mode::Real, b.data(), static_cast<uint32_t>(b.size()), uses.data()), "real report written");
	Check(Counter(b, 3, 0) == (0x20000u | (3u << 24)), "sampled slot counted");
	Check(Counter(b, 3, 1) == 0x20000u, "sampled slot asks for mip 0");
	Check(Counter(b, 4, 1) == (kNoMipSampled << 24), "unsampled slot reports 15");
	Check(Counter(b, 4, 0) == (4u << 24), "unsampled slot keeps its id");
	Check(Counter(b, 9, 1) == kMaxSamples, "samples saturate at 24 bits");
	Check(!WriteReport(Mode::Real, b.data(), kReportBytes - 1, uses.data()), "short buffer falls back to the stub");
	Check(Dword(b, 0) == 1, "short buffer stub label");
}

void TestNull() {
	Check(!WriteReport(Mode::Full, nullptr, 100, nullptr), "null destination");
	uint8_t tiny[2] = {7, 7};
	WriteReport(Mode::Zero, tiny, 2, nullptr);
	Check(tiny[0] == 0 && tiny[1] == 0, "buffer < 4 bytes is zeroed without a label");
}

} // namespace

int main() {
	TestMode();
	TestZero();
	TestFull();
	TestReal();
	TestNull();
	std::puts("LodStatsTests: ok");
	return 0;
}
