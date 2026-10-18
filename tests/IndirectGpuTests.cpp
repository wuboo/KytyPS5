#include "graphics/host_gpu/renderer/indirectGpu.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using namespace Libs::Graphics::IndirectGpu;

void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "IndirectGpuTests: failed: %s\n", message);
		std::exit(1);
	}
}

Candidate Good() {
	Candidate c;
	c.indexed           = true;
	c.draw_count        = 1;
	c.index_buffer_size = 600;
	c.index_element     = 2;
	c.first_instance_ok = true;
	return c;
}

void TestParseMode() {
	Check(ParseMode(nullptr) == Mode::Off, "unset = off");
	Check(ParseMode("0") == Mode::Off, "0 = off");
	Check(ParseMode("1") == Mode::On, "1 = on");
	Check(ParseMode("shadow") == Mode::Shadow, "shadow");
	Check(ParseMode("2") == Mode::Off && ParseMode("") == Mode::Off && ParseMode("on") == Mode::Off && ParseMode("Shadow") == Mode::Off,
	      "an unknown value is off");
}

void TestQualify() {
	Check(Qualify(Good()) == Reason::Eligible, "indexed, one draw, no count, no patches, known index size");
	auto c = Good();
	c.indexed = false;
	Check(Qualify(c) == Reason::NotIndexed, "non-indexed stays on the CPU");
	c = Good();
	c.has_count_addr = true;
	Check(Qualify(c) == Reason::CountAddr, "a count buffer: no");
	c = Good();
	c.draw_count = 2;
	Check(Qualify(c) == Reason::DrawCount, "more than one draw: no");
	c.draw_count = 0;
	Check(Qualify(c) == Reason::DrawCount, "no draw: no");
	c = Good();
	c.registers_patched = true;
	Check(Qualify(c) == Reason::RegisterPatch, "a patched register: no");
	c = Good();
	c.index_buffer_size = 0;
	Check(Qualify(c) == Reason::NoIndexSize, "unknown index buffer size: no");
	c = Good();
	c.index_element = 1;
	Check(Qualify(c) == Reason::Index8, "8-bit indices are widened on the CPU: no");
	c = Good();
	c.first_instance_ok = false;
	Check(Qualify(c) == Reason::NoFirstInstance, "no drawIndirectFirstInstance: no");
}

void TestQualifyHost() {
	HostPath h;
	Check(QualifyHost(h) == Reason::Eligible, "a plain draw");
	h = {};
	h.pv_store_possible = true;
	Check(QualifyHost(h) == Reason::PerVertexStore, "the VS-store path reads index_count: no");
	h = {};
	h.proto_possible = true;
	Check(QualifyHost(h) == Reason::PerVertexProto, "the per-vertex prototype reads index_count: no");
	h = {};
	h.mesh = true;
	Check(QualifyHost(h) == Reason::Mesh, "mesh groups come from the counts: no");
	h = {};
	h.copy_depth = true;
	Check(QualifyHost(h) == Reason::CopyDepthToColor, "copy depth to color: no");
	h = {};
	h.quad_list_legacy = true;
	Check(QualifyHost(h) == Reason::PrimitiveType, "quad list legacy loops over index_count: no");
	h = {};
	h.restart = true;
	Check(QualifyHost(h) == Reason::PrimitiveRestart, "restart scans the indices: no");
	h = {};
	h.native_fetch = true;
	Check(QualifyHost(h) == Reason::NativeFetch, "native fetch uses patched SGPRs: no");
	h = {};
	h.pv_store_possible = true;
	h.native_fetch      = true;
	Check(QualifyHost(h) == Reason::PerVertexStore, "the first reason wins");
}

void TestNames() {
	Check(std::string(ReasonName(Reason::Eligible)) == "eligible" && std::string(ReasonName(Reason::PerVertexStore)) == "pv_store" &&
	          std::string(ReasonName(Reason::CountAddr)) == "count_addr",
	      "reason names");
}

void TestCpuFallback() {
	IndexedArgs args;
	args.index_count_per_instance = 900;
	args.instance_count           = 3;
	args.start_index_location     = 10;
	args.base_vertex_location     = static_cast<uint32_t>(-5);
	args.start_instance_location  = 7;
	const auto draw               = ResolveOnCpu(args, 0x10000, 2, 600);
	Check(draw.index_count == 600, "the CPU path clamps the index count to INDEX_BUFFER_SIZE");
	Check(draw.index_addr == 0x10000 + 20, "the index address is the base plus the start index");
	Check(draw.instance_count == 3 && draw.base_vertex == -5 && draw.first_instance == 7, "the rest comes from the args");
	args.index_count_per_instance = 100;
	Check(ResolveOnCpu(args, 0x10000, 4, 600).index_count == 100, "a count inside the buffer stays");
	Check(ResolveOnCpu(args, 0x10000, 4, 600).index_addr == 0x10000 + 40, "4-byte indices");
	Check(ClampRisk(601, 600) && !ClampRisk(600, 600) && !ClampRisk(5, 0), "clamp risk");
	Check(IndexRangeBytes(600, 2) == 1200, "bound range");
}

} // namespace

int main() {
	TestParseMode();
	TestQualify();
	TestQualifyHost();
	TestNames();
	TestCpuFallback();
	return 0;
}
