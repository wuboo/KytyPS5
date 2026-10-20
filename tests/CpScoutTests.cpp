// S454 (B2): KYTY_CP_SCOUT mini-parser. A PM4 stream of SET_SH_REG + DRAW records the SH words and the
// guest dwords the user-data pointers address; Matches() re-reads those dwords. Mutant SKIP_VALIDATE
// claims a hit after the guest word changes (WILL_FAIL).

#include "graphics/guest_gpu/command_processor/cpScout.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::CpScout;

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "CpScoutTests: failed: %s\n", text);
		std::exit(1);
	}
}

bool ReadHost(uint64_t address, uint32_t& word) {
	if (address < 0x40000ull) return false;
	std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
	return true;
}

std::vector<uint32_t> PacketSetSh(uint32_t offset, std::initializer_list<uint32_t> values) {
	std::vector<uint32_t> packet;
	packet.push_back(KYTY_PM4(static_cast<uint32_t>(values.size()) + 2u, Pm4::IT_SET_SH_REG, 0u));
	packet.push_back(offset);
	packet.insert(packet.end(), values);
	return packet;
}

std::vector<uint32_t> PacketDrawIndex2() {
	return {KYTY_PM4(6, Pm4::IT_DRAW_INDEX_2, 0u), 0, 0, 0, 0, 0};
}

std::vector<uint32_t> PacketWaitRegMem() {
	return {KYTY_PM4(6, Pm4::IT_WAIT_REG_MEM, 0u), 0, 0, 0, 0, 0};
}

std::vector<uint32_t> PacketIndirectBuffer(const uint32_t* ib, uint32_t num_dw) {
	const auto addr = reinterpret_cast<uint64_t>(ib);
	return {KYTY_PM4(4, Pm4::IT_INDIRECT_BUFFER, 0u), static_cast<uint32_t>(addr),
	        static_cast<uint32_t>(addr >> 32u), num_dw | 0x0f200000u};
}

void Append(std::vector<uint32_t>& dst, const std::vector<uint32_t>& src) {
	dst.insert(dst.end(), src.begin(), src.end());
}

} // namespace

int main() {
	Check(Parse(nullptr) == Mode::Off, "default off");
	Check(Parse("0") == Mode::Off, "0 is off");
	Check(Parse("1") == Mode::On, "1 is on");
	Check(Parse("shadow") == Mode::Shadow, "shadow");
	Check(Parse("nope") == Mode::Off, "unknown is off");

	uint32_t flag = 0x11111111u;
	const auto addr = reinterpret_cast<uint64_t>(&flag);
	const uint32_t low  = static_cast<uint32_t>(addr) | 3u;
	const uint32_t high = static_cast<uint32_t>(addr >> 32u) | 0xabcd0000u;

	std::vector<uint32_t> dcb;
	Append(dcb, PacketSetSh(Pm4::SPI_SHADER_PGM_LO_PS, {0x1000u, 0x2u}));
	Append(dcb, PacketSetSh(Pm4::SPI_SHADER_USER_DATA_PS_0, {low, high}));
	Append(dcb, PacketSetSh(Pm4::SPI_SHADER_PGM_LO_GS, {0x2000u}));
	Append(dcb, PacketDrawIndex2());
	Append(dcb, PacketSetSh(Pm4::SPI_SHADER_USER_DATA_PS_0, {low, high}));
	Append(dcb, PacketDrawIndex2());

	auto walked = Walk(dcb, ReadHost);
	Check(walked.draws.size() == 2, "two draws in one DCB");
	Check(walked.draws[0].sh[Pm4::SPI_SHADER_PGM_LO_PS] == 0x1000u, "PS PGM_LO recorded");
	Check(walked.draws[0].sh[Pm4::SPI_SHADER_PGM_HI_PS] == 0x2u, "PS PGM_HI recorded");
	Check(walked.draws[0].sh[Pm4::SPI_SHADER_PGM_LO_GS] == 0x2000u, "GS PGM_LO recorded");
	Check(walked.draws[0].sh[Pm4::SPI_SHADER_USER_DATA_PS_0] == low, "PS user0 recorded");
	Check(!walked.draws[0].reads.empty(), "user-data pointer produced a guest read");
	Check(walked.draws[0].reads[0].word == 0x11111111u, "guest dword captured");
	Check(Matches(walked.draws[0], ReadHost), "unchanged guest still matches");

	flag = 0x22222222u;
	Check(!Matches(walked.draws[0], ReadHost), "changed guest dword is a miss");

	flag = 0x11111111u;
	std::vector<uint32_t> ib;
	Append(ib, PacketSetSh(Pm4::SPI_SHADER_PGM_LO_PS, {0xabcdu}));
	Append(ib, PacketDrawIndex2());
	std::vector<uint32_t> outer;
	Append(outer, PacketIndirectBuffer(ib.data(), static_cast<uint32_t>(ib.size())));
	auto nested = Walk(outer, ReadHost);
	Check(nested.draws.size() == 1, "IB followed to a draw");
	Check(nested.draws[0].sh[Pm4::SPI_SHADER_PGM_LO_PS] == 0xabcdu, "IB SET_SH_REG applied");

	std::vector<uint32_t> stopped;
	Append(stopped, PacketSetSh(Pm4::SPI_SHADER_PGM_LO_PS, {0x5u}));
	Append(stopped, PacketDrawIndex2());
	Append(stopped, PacketWaitRegMem());
	Append(stopped, PacketSetSh(Pm4::SPI_SHADER_PGM_LO_PS, {0x6u}));
	Append(stopped, PacketDrawIndex2());
	auto cut = Walk(stopped, ReadHost);
	Check(cut.stopped, "WAIT_REG_MEM stops the scout");
	Check(cut.draws.size() == 1, "draw after WAIT_REG_MEM is not recorded");
	Check(cut.draws[0].sh[Pm4::SPI_SHADER_PGM_LO_PS] == 0x5u, "draw before the wait kept");

	std::puts("CpScoutTests: OK");
	return 0;
}
