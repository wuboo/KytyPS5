#include "graphics/shader/nullDescDiag.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/srtTreeReject.h"
#include "graphics/shader/recompiler/srtReader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <span>
#include <utility>
#include <vector>
#include <xbyak/xbyak.h>

namespace {

void Check(bool value, const char* text) {
	if (!value) {
		std::fprintf(stderr, "SrtReaderTests: failed: %s\n", text);
		std::exit(1);
	}
}

using namespace Libs::Graphics::ShaderRecompiler::IR;

Block& AddValueBlock(Program& program) {
	auto block = std::make_unique<Block>();
	auto* result = block.get();
	program.blocks.push_back(result);
	result->id = static_cast<uint32_t>(program.blocks.size() - 1u);
	program.block_storage.push_back(std::move(block));
	return *result;
}

Program SrtProgram(uint64_t address) {
	Program program;
	program.stage = Libs::Graphics::ShaderType::Compute;
	program.srt_plan_complete = true;
	program.resource_tracking_complete = true;
	auto& value_block = AddValueBlock(program);

	MemoryInfo memory;
	memory.kind = ResourceKind::ScalarAddress;
	memory.planning_only = true;
	program.memory_info.push_back(memory);
	const auto low = Value(static_cast<uint32_t>(address));
	const auto high = Value(static_cast<uint32_t>(address >> 32u));
	auto& handle = value_block.AppendNewInst(ValueOpcode::GetAddressResource, {low, high});
	auto& raw = value_block.AppendNewInst(ValueOpcode::LoadAddressU32,
	                                      {Value(&handle), Value(0u), Value(0u), Value(true)});
	raw.SetFlags(MemoryFlags {.index = 0, .pc = 0x40});
	program.srt_reads.push_back({Value(&raw), 0});
	return program;
}

uint32_t Refresh(ResourcePlan& plan, const SrtRuntime& runtime) {
	SrtWalker walker(plan, runtime);
	std::vector<uint32_t> flat;
	Check(walker.RefreshFlatBuffer(flat) && flat.size() == 1u, "SRT refresh failed");
	return flat[0];
}

bool RejectBacking(void*, uint64_t, void*, uint64_t, bool*) {
	return false;
}

bool OrZeroReader(void* userdata, uint64_t address, std::span<uint32_t> words) {
	return Libs::Graphics::SrtReader::ReadOrZero(address, words, RejectBacking, userdata);
}

void TestUnmappedReadsZero() {
	std::array<uint32_t, 1> low {0xffffffffu};
	int calls = 0;
	const auto counting = [](void* data, uint64_t, void*, uint64_t, bool*) {
		++*static_cast<int*>(data);
		return false;
	};
	Check(Libs::Graphics::SrtReader::ReadOrZero(0x1000u, low, counting, &calls) && low[0] == 0u && calls == 0,
	      "address below guest memory did not return zero without touching backing");
	std::array<uint32_t, 1> window {0xffffffffu};
	Check(Libs::Graphics::SrtReader::ReadOrZero(0xfe0040000ull, window, RejectBacking, nullptr) && window[0] == 0u,
	      "unmapped driver-window read did not return zero");

	auto low_plan = ExtractResourcePlan(SrtProgram(0x1000u));
	SrtRuntime low_runtime {.read_memory = OrZeroReader};
	Check(Refresh(low_plan, low_runtime) == 0u, "null table read through the zero reader failed the walk");
	auto window_plan = ExtractResourcePlan(SrtProgram(0xfe0040000ull));
	Check(Refresh(window_plan, low_runtime) == 0u, "unmapped read through the zero reader failed the walk");
}

void TestHostGuardLetsReaderRun() {
	int calls = 0;
	const auto reader = [](void* data, uint64_t, std::span<uint32_t> words) {
		++*static_cast<int*>(data);
		words[0] = 0x99u;
		return true;
	};
	SrtRuntime runtime {.read_memory = reader, .userdata = &calls};
	std::vector<std::pair<uint64_t, uint64_t>> ranges;
	SrtReadCapture capture(runtime, ranges);
	const auto observed = capture.ObservedRuntime();
	Check(!SrtReadCapture::ReadsHostMemory(observed), "a real reader was treated as a host memcpy");
	auto plan = ExtractResourcePlan(SrtProgram(0x1000u));
	Check(Refresh(plan, observed) == 0x99u && calls > 0,
	      "null-table guard swallowed a reader below guest memory");

	const auto specialization = [](void*, uint64_t, std::span<uint32_t>) { return false; };
	SrtRuntime call_runtime {.read_memory = reader, .read_specialization_memory = specialization};
	const auto clean = CleanRuntime(call_runtime);
	Check(clean.read_memory == specialization,
	      "call-source refresh no longer walks through the specialization reader");
}

void TestCheckedSeesGpuWhereMainSeesStale() {
	uint32_t stale = 0x11111111u;
	struct Ctx {
		uint32_t* host = nullptr;
		int calls = 0;
	} ctx {&stale, 0};
	const auto reader = [](void* data, uint64_t address, std::span<uint32_t> words) {
		auto& ctx = *static_cast<Ctx*>(data);
		++ctx.calls;
		if (address != reinterpret_cast<uint64_t>(ctx.host) || words.size() != 1u) {
			return false;
		}
		words[0] = 0x47505531u;
		return true;
	};
	auto plan = ExtractResourcePlan(SrtProgram(reinterpret_cast<uint64_t>(&stale)));
	Check(plan.srt_code == nullptr, "constant SRT address compiled a walker; the test would not isolate memcpy");
	SrtRuntime runtime {.userdata = &ctx};
	Libs::Graphics::SrtReader::Apply(runtime, reader);
	const auto word = Refresh(plan, runtime);
	const auto mode = Libs::Graphics::SrtReader::Current();
	if (mode == Libs::Graphics::SrtReader::Mode::Checked) {
		Check(word == 0x47505531u && ctx.calls > 0, "checked reader did not see the GPU value");
	} else {
		Check(word == stale && ctx.calls == 0, "main/nojit read did not keep the stale host word");
	}
}

void TestDriverWindowAndDirty() {
	Check(Libs::Graphics::NullDesc::Enabled(), "KYTY_DIAG_NULLDESC=1 is required");
	const auto reads = Libs::Graphics::NullDesc::SrtDriverWindowReads().load();
	const auto nonzero = Libs::Graphics::NullDesc::SrtDriverWindowNonzero().load();
	const auto dirty = Libs::Graphics::NullDesc::SrtGpuDirtyReads().load();
	std::array<uint32_t, 1> word {0};
	const auto backing = [](void*, uint64_t, void* data, uint64_t size, bool* gpu_dirty) {
		*gpu_dirty = true;
		std::fill_n(static_cast<uint8_t*>(data), size, 0);
		if (size >= sizeof(uint32_t)) {
			*static_cast<uint32_t*>(data) = 1u;
		}
		return true;
	};
	Check(Libs::Graphics::SrtReader::ReadOrZero(0xfe0040000ull, word, backing, nullptr) && word[0] == 1u,
	      "checked reader dropped a nonzero driver-window word");
	Check(Libs::Graphics::NullDesc::SrtDriverWindowReads().load() == reads + 1u &&
	          Libs::Graphics::NullDesc::SrtDriverWindowNonzero().load() == nonzero + 1u &&
	          Libs::Graphics::NullDesc::SrtGpuDirtyReads().load() == dirty + 1u,
	      "driver-window or gpu-dirty counter did not move");
}

void TestFilteredDimensionKeepsAddress() {
	Check(Libs::Graphics::NullDesc::Enabled(), "KYTY_DIAG_NULLDESC=1 is required");
	std::array<uint32_t, 8> image {};
	image[0] = 0x11u;
	image[1] = static_cast<uint32_t>(Libs::Graphics::Prospero::BufferFormat::k8UNorm) << 20u;
	image[3] = Libs::Graphics::DstSel(4, 5, 6, 7) |
	           (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor3D) << 28u);
	const auto address = reinterpret_cast<uint64_t>(image.data());
	Program program;
	program.stage = Libs::Graphics::ShaderType::Compute;
	program.srt_plan_complete = true;
	program.resource_tracking_complete = true;
	AddValueBlock(program);
	DescriptorSource table;
	table.dword_count = 4;
	table.dwords[0] = Value(static_cast<uint32_t>(address));
	table.dwords[1] = Value(static_cast<uint32_t>(address >> 32u));
	table.dwords[2] = Value(32u);
	table.dwords[3] = Value(1u << 12u);
	program.descriptor_sources.push_back(table);
	DescriptorSource root;
	root.dword_count = 8;
	root.dwords.fill(Value(0u));
	auto& indirect = root.indirect_descriptor.emplace();
	indirect.table_source = 0;
	indirect.table_stride = 32;
	indirect.table_maximum = 0;
	program.descriptor_sources.push_back(root);
	program.info.images.push_back({
	    .source = 1,
	    .resource_class = ImageResourceClass::Sampled,
	    .numeric_class = Libs::Graphics::Prospero::TextureNumericClass::Float,
	    .dimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension::Dim2D,
	});
	auto plan = ExtractResourcePlan(program);
	struct Heap {
		uint64_t address = 0;
		const uint32_t* words = nullptr;
	} heap {address, image.data()};
	const auto read = [](void* data, uint64_t read_address, std::span<uint32_t> words) {
		const auto& heap = *static_cast<Heap*>(data);
		if (read_address != heap.address || words.size() > 8u) {
			return false;
		}
		std::copy_n(heap.words, words.size(), words.begin());
		return true;
	};
	SrtRuntime runtime {.userdata = &heap, .read_specialization_memory = read};
	ResourceSnapshot snapshot;
	ResourceSpecialization specialization;
	Check(MaterializeResources(plan, runtime, snapshot, specialization),
	      "dimension-filtered image heap did not materialize");
	Check(!snapshot.image_null_notes.empty(), "dimension filter left no null note");
	const auto& note = snapshot.image_null_notes[0];
	const auto base = 0x11ull << 8u;
	Check(note.reason == static_cast<uint8_t>(Libs::Graphics::NullDesc::Reason::FilteredDim) && note.addr == base,
	      "dimension filter did not keep the pre-filter T# address");
	Check(snapshot.images[0].dwords[0] == 0u, "dimension filter did not zero the rejected T#");
}

// S441 DIAG: a plan whose SRT address is an immediate (no pointer instruction) is refused by the tree JIT with a recorded reason.
void TestTreeRejectReason() {
	uint32_t host = 0;
	const auto plan = ExtractResourcePlan(SrtProgram(reinterpret_cast<uint64_t>(&host)));
	Check(plan.srt_code == nullptr, "an immediate SRT address compiled a tree walker");
	Check(plan.tree_reject == static_cast<uint8_t>(Libs::Graphics::ShaderRecompiler::IR::SrtTreeReject::Reason::HandleWords),
	      "the tree JIT refusal was not recorded as handle_words");
}

void TestCompiledWalkerSwitch() {
	uint32_t host = 0xABCu;
	auto plan = ExtractResourcePlan(SrtProgram(reinterpret_cast<uint64_t>(&host)));
	auto code = std::make_unique<Xbyak::CodeGenerator>(256);
	code->mov(code->dword[code->rsi], 0xDEADBEEF);
	code->mov(code->eax, 1);
	code->ret();
	plan.srt_code = std::move(code);
	SrtRuntime runtime;
	const auto word = Refresh(plan, runtime);
	const auto mode = Libs::Graphics::SrtReader::Current();
	if (mode == Libs::Graphics::SrtReader::Mode::NoJit) {
		Check(word == 0xABCu, "nojit still returned the compiled walker sentinel");
	} else if (mode == Libs::Graphics::SrtReader::Mode::Main) {
		Check(word == 0xDEADBEEFu, "main did not run the compiled walker");
	}
}

} // namespace

int main() {
	using Libs::Graphics::SrtReader::Current;
	using Libs::Graphics::SrtReader::Mode;
	TestUnmappedReadsZero();
	TestHostGuardLetsReaderRun();
	TestCheckedSeesGpuWhereMainSeesStale();
	if (Current() == Mode::Main || Current() == Mode::NoJit) {
		TestCompiledWalkerSwitch();
	}
	TestTreeRejectReason();
	TestDriverWindowAndDirty();
	TestFilteredDimensionKeepsAddress();
	std::printf("SrtReaderTests: ok mode=%s\n", Libs::Graphics::SrtReader::ModeName());
	return 0;
}

#include "graphics/shader/recompiler/ir/Block.cpp"
#include "graphics/shader/recompiler/ir/Program.cpp"
#include "graphics/shader/recompiler/ir/Type.cpp"
#include "graphics/shader/recompiler/ir/Value.cpp"
#include "graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
