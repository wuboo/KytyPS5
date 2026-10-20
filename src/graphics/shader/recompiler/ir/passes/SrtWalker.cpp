#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/recompiler/ir/passes/specShadow.h"
#include "graphics/shader/recompiler/ir/passes/srtJitCover.h"
#include "graphics/shader/recompiler/ir/passes/srtSlotBatch.h"
#include "graphics/shader/recompiler/ir/passes/srtTreeReject.h"
#include "graphics/shader/recompiler/ir/passes/srtWalkPaths.h"
#include "graphics/presentation/cpPcSampler.h"
#include "graphics/presentation/frameTiming.h"
#include "graphics/shader/recompiler/ir/SrtCfMemo.h"
#include <optional>

#include "common/abi.h"
#include "graphics/shader/nullDescDiag.h"
#include "graphics/shader/recompiler/srtReader.h"
#include "common/assert.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <xbyak/xbyak.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics::ShaderRecompiler::IR {

namespace {

// Guest memory starts at HOST_SYSTEM_MANAGED_MIN (memoryAddressSpace.inc), so a table pointer or an
// ordinary read below it comes from a null table pointer. The walker reaches such reads when it
// follows both sides of a branch it cannot decide, so read zero there instead of faulting the host.
constexpr uint64_t GUEST_MEMORY_MIN = 0x40000;

// S451 (KYTY_PROGS_MEMO): the memo that records the reachability loop in progress on this thread, or null.
thread_local SrtCfMemo* t_cf_rec = nullptr;

} // namespace

SrtRuntime SrtReadCapture::ObservedRuntime() {
	auto runtime = m_source;
	runtime.userdata = this;
	runtime.read_specialization_memory = m_source.read_specialization_memory != nullptr
	                                         ? ReadStrict : nullptr;
	runtime.read_memory = ReadOrdinary;
	return runtime;
}

bool SrtReadCapture::ReadStrict(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (!capture.m_source.read_specialization_memory(capture.m_source.userdata, address, values)) {
		return false;
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

std::vector<std::pair<uint64_t, uint64_t>>*
SrtReadCapture::DirectRanges(const SrtRuntime& runtime) {
	if (runtime.read_memory != ReadOrdinary) return nullptr;
	auto& capture = *static_cast<SrtReadCapture*>(runtime.userdata);
	return capture.m_source.read_memory == nullptr ? &capture.m_ranges : nullptr;
}

bool SrtReadCapture::ReadOrdinary(void* userdata, uint64_t address, std::span<uint32_t> values) {
	auto& capture = *static_cast<SrtReadCapture*>(userdata);
	if (capture.m_source.read_memory != nullptr) {
		if (!capture.m_source.read_memory(capture.m_source.userdata, address, values)) return false;
	} else if (address < GUEST_MEMORY_MIN) {
		NullDesc::NoteUnresolved(address);
		std::ranges::fill(values, 0u);
		NullDesc::NoteSrtRead(address, values, false);
		return true;
	} else {
		std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
		NullDesc::NoteSrtRead(address, values, false);
	}
	capture.m_ranges.emplace_back(address, values.size_bytes());
	return true;
}

bool SrtReadCapture::ReadsHostMemory(const SrtRuntime& runtime) {
	return runtime.read_memory == ReadOrdinary &&
	       static_cast<const SrtReadCapture*>(runtime.userdata)->m_source.read_memory == nullptr;
}

SrtRuntime CleanRuntime(SrtRuntime runtime) {
	runtime.read_memory = runtime.read_specialization_memory != nullptr
	                          ? runtime.read_specialization_memory
	                          : +[](void*, uint64_t, std::span<uint32_t>) { return false; };
	return runtime;
}

namespace {

constexpr uint64_t AddressMask = 0x0000ffffffffffffull;

bool AddSignedAddress(uint64_t base, int64_t offset, uint64_t& result) {
	if (base > AddressMask) {
		return false;
	}
	if (offset < 0) {
		const auto magnitude = uint64_t {0} - static_cast<uint64_t>(offset);
		if (magnitude > base) {
			return false;
		}
		result = base - magnitude;
		return true;
	}
	const auto magnitude = static_cast<uint64_t>(offset);
	if (magnitude > AddressMask - base) {
		return false;
	}
	result = base + magnitude;
	return true;
}

bool IsRawRead(const ResourcePlan& values, const Inst& inst) {
	const auto op = inst.GetOpcode();
	if (op != ValueOpcode::LoadAddressU32 && op != ValueOpcode::ReadConstBuffer &&
	    op != ValueOpcode::LoadBufferU32) {
		return false;
	}
	const auto index = inst.Flags<MemoryFlags>().index;
	if (index >= values.memory_info.size()) {
		return false;
	}
	const auto& memory = values.memory_info[index];
	if (op != ValueOpcode::LoadBufferU32) {
		return (op == ValueOpcode::LoadAddressU32 && memory.kind == ResourceKind::ScalarAddress) ||
		       (op == ValueOpcode::ReadConstBuffer && memory.kind == ResourceKind::ScalarBuffer);
	}
	if (inst.NumArgs() != 5u || memory.kind != ResourceKind::Buffer || memory.typed ||
	    memory.formatted || memory.coherent || memory.data_bits != 32u ||
	    memory.data_dwords != 1u || memory.idxen || memory.offen || memory.offset != 0u ||
	    inst.Arg(4).GetType() != Type::U1) {
		return false;
	}
	for (uint32_t arg = 1; arg <= 3; ++arg) {
		if (inst.Arg(arg).Resolve() != Value(0u)) return false;
	}
	return true;
}

bool IsDescriptorHandle(ValueOpcode opcode) {
	switch (opcode) {
		case ValueOpcode::GetBufferResource:
		case ValueOpcode::GetAddressResource:
		case ValueOpcode::GetImageResource:
		case ValueOpcode::GetSamplerResource: return true;
		default: return false;
	}
}

bool IsRuntimeSelect(ValueOpcode op) {
	return op == ValueOpcode::SelectU1 || op == ValueOpcode::SelectU32 ||
	       op == ValueOpcode::SelectF32;
}

bool IsRuntimeUniformOp(ValueOpcode op) {
	switch (op) {
		case ValueOpcode::ConditionRef:
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32:
		case ValueOpcode::ConvertU32F32:
		case ValueOpcode::ConvertF32U32:
		case ValueOpcode::CompositeConstructU64:
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeConstructU32x2:
		case ValueOpcode::CompositeExtractU32x2:
		case ValueOpcode::BitFieldInsert:
		case ValueOpcode::BitFieldUExtract:
		case ValueOpcode::BitFieldSExtract:
		case ValueOpcode::IAdd32:
		case ValueOpcode::IAdd64:
		case ValueOpcode::IAddCarry32:
		case ValueOpcode::ISub32:
		case ValueOpcode::ISub64:
		case ValueOpcode::IMul32:
		case ValueOpcode::IMul64:
		case ValueOpcode::UMulHi:
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
		case ValueOpcode::ShiftLeftLogical32:
		case ValueOpcode::ShiftLeftLogical64:
		case ValueOpcode::ShiftRightLogical32:
		case ValueOpcode::ShiftRightLogical64:
		case ValueOpcode::ShiftRightArithmetic32:
		case ValueOpcode::ShiftRightArithmetic64:
		case ValueOpcode::BitwiseAnd32:
		case ValueOpcode::BitwiseAnd64:
		case ValueOpcode::BitwiseOr32:
		case ValueOpcode::BitwiseXor32:
		case ValueOpcode::BitwiseNot32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectF32:
		case ValueOpcode::ULessThan32:
		case ValueOpcode::ULessThanEqual32:
		case ValueOpcode::IEqual32:
		case ValueOpcode::UGreaterThan32:
		case ValueOpcode::SGreaterThanEqual32:
		case ValueOpcode::INotEqual32:
		case ValueOpcode::LogicalOr:
		case ValueOpcode::LogicalAnd:
		case ValueOpcode::LogicalXor:
		case ValueOpcode::LogicalNot:
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
		case ValueOpcode::FPIsNan32:
		case ValueOpcode::FPMul32:
		case ValueOpcode::FPRecipIFlag32:
		case ValueOpcode::FPTrunc32: return true;
		default: return false;
	}
}

class RuntimeValidator {
public:
	explicit RuntimeValidator(const ResourcePlan& program, RuntimeValueType type,
	                          bool wave_uniform = false)
	    : m_program(program), m_type(type), m_wave_uniform(wave_uniform) {}

	bool Run(Value value) { return Validate(value); }

private:
	bool ValidateArguments(const Inst& inst, bool require_uniform) {
		for (size_t index = 0; index < inst.NumArgs(); index++) {
			if (!Validate(inst.Arg(index), require_uniform)) return false;
		}
		return true;
	}

	bool Validate(Value value, bool require_uniform = true) {
		value = value.Resolve();
		// Host floating-point evaluation does not model shader rounding/denormal modes.
		if (m_type != RuntimeValueType::Any &&
		    TypesOverlap(value.GetType(), Type::F16 | Type::F32 | Type::F32x2)) {
			return false;
		}
		const auto* inst = value.TryInstruction();
		if (inst == nullptr) {
			if (!require_uniform) return true;
			switch (value.GetType()) {
				case Type::U1:
				case Type::U8:
				case Type::U16:
				case Type::U32:
				case Type::U64:
				case Type::F32: return true;
				default: return false;
			}
		}
		if (require_uniform && !m_active_mask.IsEmpty() && value == m_active_mask) return true;
		// Integer-only dependency checks do not depend on the active EXEC mask.
		if (!require_uniform && m_validated_dependencies.contains(inst)) return true;
		if (!m_visiting.insert(inst).second) {
			return !require_uniform;
		}
		const auto finish = [&](bool valid) {
			m_visiting.erase(inst);
			if (valid && !require_uniform) m_validated_dependencies.insert(inst);
			return valid;
		};
		const auto op = inst->GetOpcode();
		if (m_type == RuntimeValueType::ImmutableInteger &&
		    (op == ValueOpcode::ReadConst || BufferAccessOf(op) != BufferAccess::None ||
		     AddressOpcodeInfoOf(op).access != AddressAccess::None ||
		     ImageOpcodeInfoOf(op).access != ImageAccess::None ||
		     SharedAccessOf(op) != SharedAccess::None)) {
			return finish(false);
		}
		if (op == ValueOpcode::ReadConst) {
			const auto slot = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (inst->NumArgs() != 2 || inst->Arg(0).Resolve().TryInstruction() == nullptr ||
			    inst->Arg(0).Resolve().TryInstruction()->GetOpcode() !=
			        ValueOpcode::GetSrtResource ||
			    !slot.IsImmediate() || slot.GetType() != Type::U32 ||
			    slot.U32() >= m_program.srt_reads.size()) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any) {
				const auto active_mask = m_active_mask;
				m_active_mask          = {};
				const bool valid       = Validate(m_program.srt_reads[slot.U32()].value);
				m_active_mask          = active_mask;
				if (!valid) return finish(false);
			}
		}
		if (!require_uniform) return finish(ValidateArguments(*inst, false));
		if (!m_active_mask.IsEmpty() && IsRuntimeSelect(op) && inst->NumArgs() == 3 &&
		    inst->Arg(0).Resolve() == m_active_mask) {
			// Empty EXEC reads lane zero, so ignored operands still require integer types.
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(2), false)) {
				return finish(false);
			}
			return finish(Validate(inst->Arg(1)));
		}
		if (op == ValueOpcode::UndefU1 || op == ValueOpcode::UndefU8 ||
		    op == ValueOpcode::UndefU16 || op == ValueOpcode::UndefU32 ||
		    op == ValueOpcode::UndefU64 || op == ValueOpcode::Void) {
			return finish(false);
		}
		if (op == ValueOpcode::GetUserData) {
			if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != Type::ScalarReg) {
				return finish(false);
			}
			const auto reg = RegIndex(inst->Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_program.user_data_count) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::GetShaderBase) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (m_wave_uniform && op == ValueOpcode::GetBuiltin) {
			const auto axis = inst->NumArgs() == 2u ? inst->Arg(1).Resolve() : Value {};
			return finish(m_program.stage == ShaderType::Compute && inst->NumArgs() == 2u &&
			              inst->Arg(0) == Value(static_cast<uint32_t>(StageInputKind::WorkgroupId)) &&
			              axis.IsImmediate() && axis.GetType() == Type::U32 && axis.U32() < 3u);
		}
		if (op == ValueOpcode::Phi) {
			if (m_type != RuntimeValueType::Any && !ValidateArguments(*inst, false)) {
				return finish(false);
			}
			const auto invariant = ResolveInvariantPhi(m_program, value);
			if (invariant.IsEmpty()) {
				return finish(false);
			}
			return finish(Validate(invariant));
		}
		if (op == ValueOpcode::ReadFirstLane) {
			if (inst->NumArgs() != 2 || inst->Arg(0).GetType() != Type::U32 ||
			    inst->Arg(1).GetType() != Type::U1) {
				return finish(false);
			}
			if (m_type != RuntimeValueType::Any && !Validate(inst->Arg(1), false)) {
				return finish(false);
			}
			const auto active_mask = m_active_mask;
			m_active_mask          = inst->Arg(1).Resolve();
			const bool valid       = Validate(inst->Arg(0));
			m_active_mask          = active_mask;
			return finish(valid);
		}
		if (op == ValueOpcode::GetSrtResource) {
			if (inst->NumArgs() != 0) {
				return finish(false);
			}
			return finish(true);
		}
		if (op == ValueOpcode::LoadAddressU32 || op == ValueOpcode::ReadConstBuffer ||
		    op == ValueOpcode::LoadBufferU32) {
			const auto  expected = op == ValueOpcode::LoadAddressU32
			                           ? ValueOpcode::GetAddressResource
			                           : ValueOpcode::GetBufferResource;
			const auto* handle = inst->NumArgs() != 0 ? inst->Arg(0).ResolveInstruction() : nullptr;
			if (!IsRawRead(m_program, *inst) || handle == nullptr ||
			    handle->GetOpcode() != expected) {
				return finish(false);
			}
			if (op == ValueOpcode::LoadBufferU32) {
				const auto guard = inst->Arg(4).Resolve();
				return finish(Validate(inst->Arg(0)) &&
				              ((!m_active_mask.IsEmpty() && guard == m_active_mask) || Validate(guard)));
			}
		} else if (op == ValueOpcode::CompositeExtractU64) {
			const auto index = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (!index.IsImmediate() || index.GetType() != Type::U32 || index.U32() >= 2u) {
				return finish(false);
			}
		} else if (op == ValueOpcode::CompositeExtractU32x2) {
			const auto* source = inst->NumArgs() == 2 ? inst->Arg(0).ResolveInstruction() : nullptr;
			const auto  index  = inst->NumArgs() == 2 ? inst->Arg(1).Resolve() : Value {};
			if (source == nullptr || !index.IsImmediate() || index.GetType() != Type::U32 ||
			    index.U32() >= 2u ||
			    (source->GetOpcode() != ValueOpcode::CompositeConstructU32x2 &&
			     source->GetOpcode() != ValueOpcode::IAddCarry32)) {
				return finish(false);
			}
		}
		if (IsDescriptorHandle(op)) {
			size_t expected = 4u;
			if (op == ValueOpcode::GetImageResource) {
				expected = 8u;
			} else if (op == ValueOpcode::GetAddressResource) {
				expected = 2u;
			}
			if (inst->NumArgs() != expected) {
				return finish(false);
			}
		} else if (op != ValueOpcode::ReadConst && op != ValueOpcode::ReadConstBuffer &&
		           op != ValueOpcode::LoadAddressU32 && !IsRuntimeUniformOp(op)) {
			return finish(false);
		}
		return finish(ValidateArguments(*inst, true));
	}

	const ResourcePlan&             m_program;
	RuntimeValueType                m_type;
	bool                            m_wave_uniform;
	Value                           m_active_mask;
	std::unordered_set<const Inst*> m_visiting;
	std::unordered_set<const Inst*> m_validated_dependencies;
};

bool IsUserDataWord(const ResourcePlan& program, const Inst& inst) {
	return inst.GetOpcode() == ValueOpcode::GetUserData && inst.NumArgs() == 1 &&
	       inst.Arg(0).GetType() == Type::ScalarReg &&
	       RegIndex(inst.Arg(0).ScalarRegister()) >= program.user_data_base;
}

// A scalar-address read (S442, KYTY_SRT_JIT_COVER: or a constant-buffer read) with a constant offset whose pointer words (the four words of a V#) are
// user data or such reads. Clean reads keep the strict reader of the clean evaluator.
bool IsCompiledRead(const ResourcePlan& program, const Inst& inst, uint32_t depth = 0) {
	const auto op = inst.GetOpcode();
	const bool cb = op == ValueOpcode::ReadConstBuffer && SrtJitCover::Current() != SrtJitCover::Mode::Off;
	if (depth > 16 || (op != ValueOpcode::LoadAddressU32 && !cb) || inst.NumArgs() < 2 ||
	    !IsRawRead(program, inst) || inst.Flags<SrtReadFlags>().clean != 0u)
		return false;
	const auto  offset = inst.Arg(1).Resolve();
	const auto* handle = inst.Arg(0).Resolve().TryInstruction();
	const size_t words = cb ? 4 : 2;
	if (!offset.IsImmediate() || offset.GetType() != Type::U32 || handle == nullptr ||
	    handle->GetOpcode() != (cb ? ValueOpcode::GetBufferResource : ValueOpcode::GetAddressResource) || handle->NumArgs() != words)
		return false;
	// The interpreter fails the walk on a negative immediate of a constant-buffer read; keep that path.
	if (cb && static_cast<int32_t>(program.memory_info[inst.Flags<SrtReadFlags>().index].offset) < 0) return false;
	for (size_t word = 0; word < words; ++word) {
		const auto* source = handle->Arg(word).Resolve().TryInstruction();
		if (source == nullptr ||
		    (!IsUserDataWord(program, *source) && !IsCompiledRead(program, *source, depth + 1)))
			return false;
	}
	return true;
}

// S442: the tree of a compiled read contains a constant-buffer read.
bool UsesConstBuffer(const Inst& inst) {
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer) return true;
	const auto* handle = inst.Arg(0).Resolve().TryInstruction();
	for (size_t word = 0; handle != nullptr && word < handle->NumArgs(); ++word) {
		const auto* source = handle->Arg(word).Resolve().TryInstruction();
		if (source != nullptr && (source->GetOpcode() == ValueOpcode::LoadAddressU32 || source->GetOpcode() == ValueOpcode::ReadConstBuffer) &&
		    UsesConstBuffer(*source))
			return true;
	}
	return false;
}

// S441 DIAG: why IsCompiledRead refuses a slot (same conditions, same order).
std::string DiagSlotReject(const ResourcePlan& program, const Inst* inst, uint32_t depth = 0) {
	if (inst == nullptr) return "not_an_instruction";
	if (depth > 16) return "too_deep";
	if (inst->GetOpcode() != ValueOpcode::LoadAddressU32 || inst->NumArgs() < 2) return "opcode:" + std::string(ValueOpcodeName(inst->GetOpcode()));
	if (!IsRawRead(program, *inst)) return "not_raw_read";
	if (inst->Flags<SrtReadFlags>().clean != 0u) return "clean_read";
	const auto  offset = inst->Arg(1).Resolve();
	const auto* handle = inst->Arg(0).Resolve().TryInstruction();
	if (!offset.IsImmediate() || offset.GetType() != Type::U32) return "offset_not_imm";
	if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource || handle->NumArgs() != 2) return "handle_shape";
	for (size_t word = 0; word < 2; ++word) {
		const auto* source = handle->Arg(word).Resolve().TryInstruction();
		if (source == nullptr) return "immediate_pointer_word";
		if (IsUserDataWord(program, *source)) continue;
		if (IsCompiledRead(program, *source, depth + 1)) continue;
		return "pointer_word:" + DiagSlotReject(program, source, depth + 1);
	}
	return "compilable";
}

// S445: the compute path of one memoized read (after a memo miss): the DWORD ends in eax, recorded in the address cursor (r11) and memoized as
// EvaluateRawRead does; `source_value(source)` leaves the DWORD of a pointer word that is not user data in eax (a call of its routine, or its inlined
// code), `finish()` ends the path (ret, or a jump past the inlined code). Shared by the per-slot routines and the S445 batch entries.
template <class SourceValue, class Finish>
void EmitComputePath(Xbyak::CodeGenerator& code, const ResourcePlan& program, const Inst* inst, Xbyak::Label& failed, const SourceValue& source_value, const Finish& finish) {
	using Entry       = ResourcePlan::EvaluationContext::Entry;
	const auto memo   = inst->EvaluationIndex(program.evaluation_value_count) * sizeof(Entry);
	const auto* handle = inst->Arg(0).Resolve().TryInstruction();
	if (inst->GetOpcode() == ValueOpcode::ReadConstBuffer) {
		// S442: ReadConstBuffer as EvaluateRawRead does it: V# words low, high (stride in bits 16..29), records, word 3 (evaluated, unused); the DWORD
		// is at base + byte offset when byte offset + 4 <= size, else 0. A null table falls back to the interpreter (it notes it).
		for (size_t word = 0; word < 4; ++word) {
			const auto* source = handle->Arg(word).Resolve().TryInstruction();
			if (IsUserDataWord(program, *source)) {
				const auto index = RegIndex(source->Arg(0).ScalarRegister()) - program.user_data_base;
				code.cmp(code.r8, index);
				code.jbe(failed, Xbyak::CodeGenerator::T_NEAR);
				code.mov(code.eax, code.dword[code.rdi + index * sizeof(uint32_t)]);
			} else {
				source_value(*source);
			}
			code.push(code.rax);
		}
		// [rsp] word 3, [rsp + 8] records, [rsp + 16] high, [rsp + 24] low
		const auto& cb_memory = program.memory_info[inst->Flags<SrtReadFlags>().index];
		uint64_t    byte_offset = (static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(cb_memory.offset))) & ~uint64_t {3}) +
		                       (inst->Arg(1).Resolve().U32() & ~3u);
#ifdef KYTY_SRTCOVER_MUTANT_CB_WRONG_OFFSET
		byte_offset += 4;
#endif
		Xbyak::Label no_stride;
		Xbyak::Label outside;
		Xbyak::Label store;
		code.mov(code.eax, code.dword[code.rsp + 8]);
		code.mov(code.r10d, code.dword[code.rsp + 16]);
		code.shr(code.r10d, 16);
		code.and_(code.r10d, 0x3fff);
		code.jz(no_stride);
		code.imul(code.rax, code.r10);
		code.L(no_stride);
		code.mov(code.r10, byte_offset);
		code.cmp(code.rax, code.r10);
		code.jb(outside, Xbyak::CodeGenerator::T_NEAR);
		code.sub(code.rax, code.r10);
		code.cmp(code.rax, static_cast<uint32_t>(sizeof(uint32_t)));
		code.jb(outside, Xbyak::CodeGenerator::T_NEAR);
		code.mov(code.eax, code.dword[code.rsp + 24]);
		code.mov(code.r10d, code.dword[code.rsp + 16]);
		code.shl(code.r10, 32);
		code.or_(code.rax, code.r10);
		code.mov(code.r10, AddressMask & ~uint64_t {3});
		code.and_(code.rax, code.r10);
		code.mov(code.r10, GUEST_MEMORY_MIN);
		code.cmp(code.rax, code.r10);
		code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
		code.mov(code.r10, byte_offset);
		code.add(code.rax, code.r10);
		code.mov(code.qword[code.r11], code.rax);
		code.add(code.r11, sizeof(uint64_t));
		code.mov(code.eax, code.dword[code.rax]);
		code.jmp(store);
		code.L(outside);
		code.xor_(code.eax, code.eax);
		code.L(store);
		code.mov(code.qword[code.rdx + memo + offsetof(Entry, value)], code.rax);
		code.mov(code.qword[code.rdx + memo + offsetof(Entry, generation)], code.rcx);
		code.add(code.rsp, 4 * sizeof(uint64_t));
		finish();
		return;
	}
	for (size_t word = 0; word < 2; ++word) {
		const auto* source = handle->Arg(word).Resolve().TryInstruction();
		if (IsUserDataWord(program, *source)) {
			const auto index =
			    RegIndex(source->Arg(0).ScalarRegister()) - program.user_data_base;
			code.cmp(code.r8, index);
			code.jbe(failed, Xbyak::CodeGenerator::T_NEAR);
			code.mov(code.eax, code.dword[code.rdi + index * sizeof(uint32_t)]);
		} else {
			source_value(*source);
		}
		if (word == 0) code.push(code.rax);
	}
	code.mov(code.r10d, code.eax);
	code.pop(code.rax);
	code.shl(code.r10, 32);
	code.or_(code.rax, code.r10);
	code.mov(code.r10, AddressMask & ~uint64_t {3});
	code.and_(code.rax, code.r10);
	// A table below guest memory (a null pointer) falls back to the interpreted read, which
	// reads zero there instead of faulting the host.
	code.mov(code.r10, GUEST_MEMORY_MIN);
	code.cmp(code.rax, code.r10);
	code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
	const auto& memory = program.memory_info[inst->Flags<SrtReadFlags>().index];
	const auto  relative =
	    (static_cast<int64_t>(static_cast<int32_t>(memory.offset)) & ~int64_t {3}) +
	    static_cast<int64_t>(inst->Arg(1).Resolve().U32() & ~3u);
	if (relative < 0) {
		code.mov(code.r10, -relative);
		code.cmp(code.rax, code.r10);
		code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
		code.sub(code.rax, code.r10);
	} else if (relative > 0) {
		code.mov(code.r10, AddressMask - static_cast<uint64_t>(relative));
		code.cmp(code.rax, code.r10);
		code.ja(failed, Xbyak::CodeGenerator::T_NEAR);
		code.mov(code.r10, relative);
		code.add(code.rax, code.r10);
	}
	code.mov(code.qword[code.r11], code.rax);
	code.add(code.r11, sizeof(uint64_t));
	code.mov(code.eax, code.dword[code.rax]);
	code.mov(code.qword[code.rdx + memo + offsetof(Entry, value)], code.rax);
	code.mov(code.qword[code.rdx + memo + offsetof(Entry, generation)], code.rcx);
	finish();
}

// S445 (KYTY_SRT_SLOT_BATCH=1|shadow): per control-flow block, the compiled slots that follow each other in block.srt_reads form a run; every run of two or
// more is one entry into generated code (rdi user data, rsi flat buffer, rdx memo, rcx generation, r8 user data count, r9 address cursor, as the per-slot
// entries). The pointer chain of a slot is inlined: a memo check, then the compute path of EmitComputePath; a node that an earlier slot of the run already
// ensured is read from the memo without a check (the memo holds it for this generation on every path that got there). Returns 0 when every slot is
// filled, else the 1-based position of the slot that failed (its partial reads stay in the cursor, as with the per-slot entry).
void CompileSrtBatch(ResourcePlan& program, const std::vector<const Inst*>& slots) {
	using Entry = ResourcePlan::EvaluationContext::Entry;
	if (program.control_flow.empty()) return;
	const auto compiled = [&](uint32_t slot) { return slot < slots.size() && slots[slot] != nullptr; };
	program.srt_block_runs.assign(program.control_flow.size(), {});
	bool any = false;
	for (size_t block = 0; block < program.control_flow.size(); ++block) {
		const auto& list = program.control_flow[block].srt_reads;
		for (size_t i = 0; i < list.size();) {
			if (!compiled(list[i])) {
				++i;
				continue;
			}
			size_t j = i;
#ifdef KYTY_SRTBATCH_MUTANT_IGNORE_CLEAN
			while (j < list.size()) ++j; // an uncompiled (clean) slot does not end the run: it is dropped from it
#else
			while (j < list.size() && compiled(list[j])) ++j;
#endif
			ResourcePlan::SrtBatchRun run;
			run.first = static_cast<uint32_t>(i);
			run.count = static_cast<uint32_t>(j - i);
			run.entry = UINT32_MAX;
			for (size_t k = i; k < j; ++k)
				if (compiled(list[k]) && list[k] < program.srt_read_cb.size() && program.srt_read_cb[list[k]] != 0u) ++run.cb;
			program.srt_block_runs[block].push_back(run);
			any = any || run.count >= SrtSlotBatch::kMinRun;
			i = j;
		}
	}
	if (!any) {
		program.srt_block_runs.clear();
		return;
	}
	program.srt_batch_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	auto&  code            = *program.srt_batch_code;
	size_t max_addresses   = 0;
	for (size_t block = 0; block < program.control_flow.size(); ++block) {
		const auto& list = program.control_flow[block].srt_reads;
		for (auto& run: program.srt_block_runs[block]) {
			if (run.count < SrtSlotBatch::kMinRun) continue;
			run.entry = static_cast<uint32_t>(code.getSize());
			code.push(code.rbp);
			code.mov(code.rbp, code.rsp);
			code.mov(code.r11, code.qword[code.r9]);
			std::vector<Xbyak::Label> fails(run.count);
			std::vector<const Inst*>  ensured; // nodes that are in the memo for this generation at this point of the run
			size_t                    emitted = 0;
			const auto                ensure  = [&](auto&& self, const Inst* inst, Xbyak::Label& fail) -> void {
				const auto memo = inst->EvaluationIndex(program.evaluation_value_count) * sizeof(Entry);
#ifndef KYTY_SRTBATCH_MUTANT_SKIP_MEMO_CHECK
				if (std::find(ensured.begin(), ensured.end(), inst) != ensured.end()) {
					code.mov(code.eax, code.dword[code.rdx + memo + offsetof(Entry, value)]);
					return;
				}
				Xbyak::Label done;
				const auto   mark = ensured.size();
				Xbyak::Label compute;
				code.cmp(code.rcx, code.qword[code.rdx + memo + offsetof(Entry, generation)]);
				code.jne(compute, Xbyak::CodeGenerator::T_NEAR);
				code.mov(code.eax, code.dword[code.rdx + memo + offsetof(Entry, value)]);
				code.jmp(done, Xbyak::CodeGenerator::T_NEAR);
				code.L(compute);
#else
				Xbyak::Label done;
				const auto   mark = ensured.size();
#endif
				++emitted;
				EmitComputePath(code, program, inst, fail, [&](const Inst& source) { self(self, &source, fail); }, [&] { code.jmp(done, Xbyak::CodeGenerator::T_NEAR); });
				code.L(done);
				// What the compute path ensured holds only on that path (the memo-hit path did not run it).
				ensured.resize(mark);
#ifndef KYTY_SRTBATCH_MUTANT_SKIP_MEMO_CHECK
				ensured.push_back(inst);
#endif
			};
			for (uint32_t step = 0; step < run.count; ++step) {
#ifdef KYTY_SRTBATCH_MUTANT_ORDER_SWAP
				const uint32_t k = run.count - 1u - step; // the slots of the run in reverse order: the captured addresses change order
#else
				const uint32_t k = step;
#endif
				const uint32_t slot = list[run.first + k];
				if (!compiled(slot)) continue; // only with the IGNORE_CLEAN mutant
				ensure(ensure, slots[slot], fails[k]);
				code.mov(code.dword[code.rsi + program.srt_reads[slot].flat_offset * sizeof(uint32_t)], code.eax);
			}
			code.mov(code.qword[code.r9], code.r11);
			code.xor_(code.eax, code.eax);
			code.pop(code.rbp);
			code.ret();
			for (uint32_t k = 0; k < run.count; ++k) {
				code.L(fails[k]);
				code.mov(code.qword[code.r9], code.r11);
#ifdef KYTY_SRTBATCH_MUTANT_NO_FAIL_POS
				code.mov(code.eax, 1u);
#else
				code.mov(code.eax, k + 1u);
#endif
				code.mov(code.rsp, code.rbp);
				code.pop(code.rbp);
				code.ret();
			}
			max_addresses = std::max(max_addresses, emitted);
		}
	}
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
	if (CpSampler::Pc::Enabled()) CpSampler::Pc::RegisterJit(code.getCode(), code.getSize(), CpSampler::Pc::JitKind::Slots, program.shader_hash); // S439 DIAG
	if (program.srt_read_addresses.size() < max_addresses) program.srt_read_addresses.resize(max_addresses);
}

// Compiles the SRT slots of plans the whole-plan walker rejects (block conditions, captured
// reads): one entry per slot, one memoized routine per read. Other slots stay interpreted.
void CompileSrtReads(ResourcePlan& program) {
	if (!program.resource_tracking_complete || !program.srt_plan_complete ||
	    program.srt_reads.empty())
		return;
	std::vector<const Inst*> slots(program.srt_reads.size(), nullptr);
	bool                     any = false;
	const bool               diag = SrtTreeReject::Active().load(std::memory_order_relaxed);
	if (diag) program.srt_slot_reject.assign(slots.size(), std::string());
	for (uint32_t slot = 0; slot < slots.size(); ++slot) {
		const auto& read = program.srt_reads[slot];
		const auto* inst = read.value.Resolve().TryInstruction();
		if (inst != nullptr && read.flat_offset < program.srt_reads.size() &&
		    IsCompiledRead(program, *inst)) {
			slots[slot] = inst;
			any         = true;
		} else if (diag) {
			program.srt_slot_reject[slot] = read.flat_offset >= program.srt_reads.size() ? "flat_oob" : DiagSlotReject(program, inst);
		}
	}
	if (!any) return;

	using Entry           = ResourcePlan::EvaluationContext::Entry;
	program.srt_read_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	auto& code            = *program.srt_read_code;
	// rdi user data, rsi flat buffer, rdx memo, rcx generation, r8 user data count,
	// r9 address cursor (kept in r11). rbp holds the entry frame for the shared failure exit.
	Xbyak::Label                                  failed;
	std::unordered_map<const Inst*, Xbyak::Label> routines;
	std::vector<const Inst*>                      pending;
	const auto                                    routine = [&](const Inst* inst) -> Xbyak::Label& {
		const auto [found, inserted] = routines.try_emplace(inst);
		if (inserted) pending.push_back(inst);
		return found->second;
	};
	program.srt_read_entries.assign(slots.size(), UINT32_MAX);
	program.srt_read_cb.assign(slots.size(), 0u);
	for (uint32_t slot = 0; slot < slots.size(); ++slot) {
		if (slots[slot] == nullptr) continue;
		program.srt_read_cb[slot] = UsesConstBuffer(*slots[slot]) ? 1u : 0u;
		program.srt_read_entries[slot] = static_cast<uint32_t>(code.getSize());
		code.push(code.rbp);
		code.mov(code.rbp, code.rsp);
		code.mov(code.r11, code.qword[code.r9]);
		code.call(routine(slots[slot]));
		code.mov(code.dword[code.rsi + program.srt_reads[slot].flat_offset * sizeof(uint32_t)],
		         code.eax);
		code.mov(code.qword[code.r9], code.r11);
		code.mov(code.eax, 1);
		code.pop(code.rbp);
		code.ret();
	}
	code.L(failed);
	code.mov(code.qword[code.r9], code.r11);
	code.xor_(code.eax, code.eax);
	code.mov(code.rsp, code.rbp);
	code.pop(code.rbp);
	code.ret();
	// Each routine returns its DWORD in eax: from the memo, or read, recorded and memoized as
	// EvaluateRawRead does. Pointer words come from user data or nested routines.
	while (!pending.empty()) {
		const auto* inst = pending.back();
		pending.pop_back();
		code.L(routines.at(inst));
		const auto   memo = inst->EvaluationIndex(program.evaluation_value_count) * sizeof(Entry);
		Xbyak::Label compute;
		code.cmp(code.rcx, code.qword[code.rdx + memo + offsetof(Entry, generation)]);
		code.jne(compute, Xbyak::CodeGenerator::T_NEAR);
		code.mov(code.eax, code.dword[code.rdx + memo + offsetof(Entry, value)]);
		code.ret();
		code.L(compute);
		EmitComputePath(code, program, inst, failed, [&](const Inst& source) { code.call(routine(&source)); }, [&] { code.ret(); });
	}
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
	if (CpSampler::Pc::Enabled()) CpSampler::Pc::RegisterJit(code.getCode(), code.getSize(), CpSampler::Pc::JitKind::Slots, program.shader_hash); // S439 DIAG
	program.srt_read_addresses.resize(routines.size());
	if (SrtSlotBatch::Current() != SrtSlotBatch::Mode::Off) CompileSrtBatch(program, slots);
}

void CompileTreeWalker(ResourcePlan& program) {
	using Reason = SrtTreeReject::Reason;
	// S441 DIAG: the first reason the whole-tree JIT is refused (a byte store; the detail string only for the census).
	const auto reject = [&](Reason reason, std::string_view detail = "") {
#ifndef KYTY_TREEREJECT_MUTANT_NO_RECORD
		program.tree_reject = static_cast<uint8_t>(reason);
		if (SrtTreeReject::Active().load(std::memory_order_relaxed)) program.tree_reject_detail = std::string(detail);
#else
		(void)reason;
		(void)detail;
#endif
	};
	if (!program.resource_tracking_complete) return reject(Reason::NotTracked);
	if (!program.srt_plan_complete) return reject(Reason::PlanIncomplete);
	if (program.capture_specialization_reads) return reject(Reason::CaptureSpecReads);
	if (!program.control_flow.empty()) return reject(Reason::ControlFlow);
	if (program.srt_reads.empty()) return reject(Reason::NoReads);

	using Reads = std::map<int32_t, const SrtRead*>;
	std::map<uint32_t, Reads> roots;
	std::unordered_map<const Inst*, Reads> pointer_uses;
	std::unordered_map<const Inst*, int32_t> offsets;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.Resolve().TryInstruction();
		if (inst == nullptr || inst->GetOpcode() != ValueOpcode::LoadAddressU32 || inst->NumArgs() < 2)
			return reject(Reason::SlotOpcode, inst == nullptr ? "not_an_instruction" : ValueOpcodeName(inst->GetOpcode()));
		if (!IsRawRead(program, *inst)) return reject(Reason::SlotNotRaw);
		if (read.flat_offset >= program.srt_reads.size()) return reject(Reason::SlotFlatOob);
		const auto offset = inst->Arg(1).Resolve();
		if (!offset.IsImmediate() || offset.GetType() != Type::U32) return reject(Reason::OffsetNotImm);
		const auto& memory = program.memory_info[inst->Flags<SrtReadFlags>().index];
		const auto relative =
		    (static_cast<int64_t>(static_cast<int32_t>(memory.offset)) & ~int64_t {3}) +
		    static_cast<int64_t>(offset.U32() & ~3u);
		if (relative < INT32_MIN || relative > INT32_MAX ||
		    !offsets.emplace(inst, static_cast<int32_t>(relative)).second) return reject(Reason::OffsetRange);
	}
	uint32_t user_data_count = 0;
	for (const auto& read: program.srt_reads) {
		const auto* inst = read.value.ResolveInstruction();
		const auto* handle = inst->Arg(0).Resolve().TryInstruction();
		if (handle == nullptr || handle->GetOpcode() != ValueOpcode::GetAddressResource ||
		    handle->NumArgs() != 2) return reject(Reason::HandleShape, handle == nullptr ? "not_an_instruction" : ValueOpcodeName(handle->GetOpcode()));
		const auto* low = handle->Arg(0).Resolve().TryInstruction();
		const auto* high = handle->Arg(1).Resolve().TryInstruction();
		if (low == nullptr || high == nullptr) return reject(Reason::HandleWords, "immediate_pointer_word");
		Reads* uses = nullptr;
		if (low->GetOpcode() == ValueOpcode::GetUserData &&
		    high->GetOpcode() == ValueOpcode::GetUserData && low->NumArgs() == 1 &&
		    high->NumArgs() == 1 && low->Arg(0).GetType() == Type::ScalarReg &&
		    high->Arg(0).GetType() == Type::ScalarReg) {
			const auto reg = RegIndex(low->Arg(0).ScalarRegister());
			if (reg < program.user_data_base ||
			    RegIndex(high->Arg(0).ScalarRegister()) != reg + 1u) return reject(Reason::UserDataRegs);
			const auto base = reg - program.user_data_base;
			user_data_count = std::max(user_data_count, base + 2u);
			uses = &roots[base];
		} else {
			const auto lo = offsets.find(low);
			const auto hi = offsets.find(high);
			if (lo == offsets.end() || hi == offsets.end() ||
			    static_cast<int64_t>(lo->second) + 4 != hi->second ||
			    !EquivalentValue(program, low->Arg(0), high->Arg(0)))
				return reject(Reason::PointerOffsets, ValueOpcodeName(low->GetOpcode()));
			uses = &pointer_uses[low];
		}
		if (!uses->emplace(offsets.at(inst), &read).second) return reject(Reason::DuplicateOffset);
	}
	std::unordered_set<const Inst*> visited;
	const auto reachable = [&](auto&& self, const Reads& reads) -> bool {
		for (const auto& [offset, read]: reads) {
			const auto* inst = read->value.ResolveInstruction();
			const auto child = pointer_uses.find(inst);
			if (child != pointer_uses.end() &&
			    (!visited.insert(inst).second || !self(self, child->second))) return false;
		}
		return true;
	};
	for (const auto& [reg, reads]: roots) {
		if (!reachable(reachable, reads)) return reject(Reason::PointerCycle);
	}
	if (visited.size() != pointer_uses.size()) return reject(Reason::PointerUnreached);

	program.srt_code = std::make_unique<Xbyak::CodeGenerator>(4096, Xbyak::AutoGrow);
	auto& code = *program.srt_code;
	Xbyak::Label failed, finished;
	code.push(code.rbp);
	code.mov(code.rbp, code.rsp);
	code.cmp(code.r8, user_data_count);
	code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
	code.mov(code.r9, AddressMask & ~uint64_t {3});
	// Follow shadPS4's VisitPointer: copy one table level in offset order, then
	// descend through its child pointers. Seed the descriptor evaluator's memo.
	const auto visit = [&](auto&& self, int32_t pointer_offset, const Reads& reads) -> void {
		code.push(code.rdi);
		code.mov(code.rdi, code.qword[code.rdi + pointer_offset]);
		code.and_(code.rdi, code.r9);
		const auto first = reads.begin()->first;
		const auto last = reads.rbegin()->first;
		// Tables below guest memory and underflowing reads fall back to the interpreted walk.
		code.mov(code.r10, GUEST_MEMORY_MIN + static_cast<uint64_t>(-std::min<int64_t>(first, 0)));
		code.cmp(code.rdi, code.r10);
		code.jb(failed, Xbyak::CodeGenerator::T_NEAR);
		if (last > 0) {
			code.mov(code.r10, AddressMask - last);
			code.cmp(code.rdi, code.r10);
			code.ja(failed, Xbyak::CodeGenerator::T_NEAR);
		}
		for (const auto& [offset, read]: reads) {
			const auto index = read->value.ResolveInstruction()->EvaluationIndex(program.evaluation_value_count);
			using Entry = ResourcePlan::EvaluationContext::Entry;
			const auto slot = code.rdx + index * sizeof(Entry);
			code.mov(code.r10d, code.dword[code.rdi + offset]);
			code.mov(code.dword[code.rsi + read->flat_offset * sizeof(uint32_t)], code.r10d);
			code.mov(code.qword[slot + offsetof(Entry, value)], code.r10);
			code.mov(code.qword[slot + offsetof(Entry, generation)], code.rcx);
		}
		for (const auto& [offset, read]: reads) {
			const auto child = pointer_uses.find(read->value.ResolveInstruction());
			if (child == pointer_uses.end()) continue;
			self(self, offset, child->second);
		}
		code.pop(code.rdi);
	};
	for (const auto& [reg, reads]: roots) {
		visit(visit, reg * sizeof(uint32_t), reads);
	}
	code.mov(code.eax, 1);
	code.jmp(finished, Xbyak::CodeGenerator::T_NEAR);
	code.L(failed);
	code.xor_(code.eax, code.eax);
	code.L(finished);
	code.mov(code.rsp, code.rbp);
	code.pop(code.rbp);
	code.ret();
	code.readyRE();
	EXIT_IF(Xbyak::GetError() != 0);
	if (CpSampler::Pc::Enabled()) CpSampler::Pc::RegisterJit(code.getCode(), code.getSize(), CpSampler::Pc::JitKind::Tree, program.shader_hash); // S439 DIAG
}

} // namespace

void CompileSrtWalker(ResourcePlan& program) {
	CompileTreeWalker(program);
	if (program.srt_code == nullptr) CompileSrtReads(program);
}

SrtWalker::SrtWalker(const ResourcePlan& program, const SrtRuntime& runtime,
                     SrtWalker* clean_evaluator)
    : m_program(program), m_runtime(runtime), m_clean_evaluator(clean_evaluator),
      m_session(this),
      m_read_policy(clean_evaluator != nullptr ? 2u : 0u), m_context(AcquireContext()) {}

SrtWalker::SrtWalker(SrtWalker& parent, Value active_mask, bool clean,
                     SrtWalker* clean_evaluator)
    : m_program(parent.m_program), m_runtime(clean ? CleanRuntime(parent.m_runtime) : parent.m_runtime),
      m_clean_evaluator(clean_evaluator), m_active_mask(active_mask.Resolve()),
      m_reachable_blocks(parent.m_reachable_blocks), m_session(parent.m_session),
      m_read_policy((parent.m_read_policy & 1u) | (clean ? 1u : 0u) |
                    (clean_evaluator != nullptr ? 2u : 0u)), m_context(AcquireContext()) {}

SrtWalker::~SrtWalker() {
	if (m_session != this) return;
	for (auto& context: m_program.evaluation_contexts)
		if (context.session == this) context.session = nullptr;
}

bool SrtWalker::Evaluate(Value value, uint32_t& result) {
	uint64_t wide = 0;
	if (!EvaluateWide(value, wide)) {
		return false;
	}
	result = static_cast<uint32_t>(wide);
	return true;
}

ResourcePlan::EvaluationContext& SrtWalker::AcquireContext() {
	ResourcePlan::EvaluationContext* unused = nullptr;
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session == m_session && context.active_mask == m_active_mask &&
		    context.read_policy == m_read_policy) return context;
		if (context.session == nullptr && unused == nullptr) unused = &context;
	}
	auto& context = unused != nullptr ? *unused : m_program.evaluation_contexts.emplace_back();
	context.session = m_session;
	context.active_mask = m_active_mask;
	context.read_policy = m_read_policy;
	context.generation += 4;
	return context;
}

void SrtWalker::ResetUnknowns() {
	for (auto& context: m_program.evaluation_contexts) {
		if (context.session != m_session) continue;
		for (auto& entry: context.values)
			if (entry.generation == (context.generation | 2u)) entry.generation = 0;
	}
}

float SrtWalker::Float32(uint64_t bits) {
	return std::bit_cast<float>(static_cast<uint32_t>(bits));
}

bool SrtWalker::EvaluateWide(Value value, uint64_t& result) {
	value = value.Resolve();
	if (value.IsImmediate()) {
		switch (value.GetType()) {
			case Type::U1: result = value.U1(); return true;
			case Type::U8: result = value.U8(); return true;
			case Type::U16: result = value.U16(); return true;
			case Type::U32: result = value.U32(); return true;
			case Type::U64: result = value.U64(); return true;
			case Type::F32: result = std::bit_cast<uint32_t>(value.F32Value()); return true;
			default: return false;
		}
	}
	if (!m_active_mask.IsEmpty() && value == m_active_mask) {
		result = 1u;
		return true;
	}
	auto* inst = value.TryInstruction();
	if (inst == nullptr) {
		return false;
	}
	if (!m_active_mask.IsEmpty() && IsRuntimeSelect(inst->GetOpcode()) &&
	    inst->NumArgs() == 3 && inst->Arg(0).Resolve() == m_active_mask) {
		return EvaluateWide(inst->Arg(1), result);
	}
	const auto index = inst->EvaluationIndex(m_program.evaluation_value_count);
	if (index >= m_context.values.size()) {
		m_context.values.resize(m_program.evaluation_value_count);
	}
	if (m_context.values[index].generation == m_context.generation) {
		result = m_context.values[index].value;
		return true;
	}
	// Visiting and completed-unknown entries both stop repeated evaluation in this
	// context. Only completed unknowns are retried when reachability becomes tighter.
	if (m_context.values[index].generation == (m_context.generation | 1u) ||
	    m_context.values[index].generation == (m_context.generation | 2u)) {
		return false;
	}
	m_context.values[index].generation = m_context.generation | 1u;
	uint64_t out = 0;
	const bool evaluated = EvaluateInst(*inst, out);
	// Recursive evaluation may grow the dense memo vector.
	auto& memo = m_context.values[index];
	if (!evaluated) {
		memo.generation = m_context.generation | 2u;
		return false;
	}
	memo.value      = out;
	memo.generation = m_context.generation;
	result = out;
	return true;
}

bool SrtWalker::Arg(const Inst& inst, size_t index, uint64_t& result) {
	return EvaluateWide(inst.Arg(index), result);
}

bool SrtWalker::EvaluatePhi(const Inst& inst, uint64_t& result) {
	if (const auto control = m_program.control_phi_predecessors.find(&inst);
	    control != m_program.control_phi_predecessors.end()) {
		bool found = false;
		for (size_t index = 0; index < inst.NumArgs(); ++index) {
			if (m_reachable_blocks != nullptr &&
			    !m_reachable_blocks->at(control->second[index])) continue;
			uint64_t incoming = 0;
			if (!Arg(inst, index, incoming) || (found && incoming != result)) return false;
			result = incoming;
			found = true;
		}
		return found;
	}
	const auto value = ResolveInvariantPhi(m_program, Value(const_cast<Inst*>(&inst)));
	return !value.IsEmpty() && EvaluateWide(value, result);
}

bool SrtWalker::EvaluateExtract(const Inst& inst, uint64_t& result) {
	const auto index = inst.Arg(1).Resolve();
	if (!index.IsImmediate() || index.GetType() != Type::U32) {
		return false;
	}
	const auto component = index.U32();
	if (component >= 2u) {
		return false;
	}
	if (inst.GetOpcode() == ValueOpcode::CompositeExtractU64) {
		uint64_t packed = 0;
		if (!Arg(inst, 0, packed)) {
			return false;
		}
		result = static_cast<uint32_t>(packed >> (component * 32u));
		return true;
	}
	const auto* source = inst.Arg(0).ResolveInstruction();
	if (source == nullptr) {
		return false;
	}
	if (source->GetOpcode() == ValueOpcode::CompositeConstructU32x2) {
		return EvaluateWide(source->Arg(component), result);
	}
	if (source->GetOpcode() == ValueOpcode::IAddCarry32) {
		uint64_t lhs = 0;
		uint64_t rhs = 0;
		if (!Arg(*source, 0, lhs) || !Arg(*source, 1, rhs)) {
			return false;
		}
		const auto sum =
		    static_cast<uint64_t>(static_cast<uint32_t>(lhs)) + static_cast<uint32_t>(rhs);
		result =
		    component == 0u ? static_cast<uint32_t>(sum) : static_cast<uint32_t>(sum >> 32u);
		return true;
	}
	return false;
}

bool SrtWalker::EvaluateRawRead(const Inst& inst, uint64_t& result) {
	const auto flags = inst.Flags<SrtReadFlags>();
	if (flags.index >= m_program.memory_info.size()) {
		return false;
	}
	const auto& mem    = m_program.memory_info[flags.index];
	const bool vector = inst.GetOpcode() == ValueOpcode::LoadBufferU32;
	const auto guard = vector ? inst.Arg(4).Resolve() : Value {};
	if (vector && (guard.IsImmediate() || guard != m_active_mask)) {
		uint64_t enabled = 0;
		if (!Arg(inst, 4, enabled)) return false;
		if (enabled == 0u) {
			result = 0u;
			return true;
		}
	}
	const auto* handle = inst.Arg(0).ResolveInstruction();
	if (handle == nullptr) {
		return false;
	}
	uint64_t low    = 0;
	uint64_t high   = 0;
	uint64_t offset = 0;
	if (!Arg(*handle, 0, low) || !Arg(*handle, 1, high) || !Arg(inst, 1, offset)) {
		return false;
	}
	const auto base      = ((high << 32u) | static_cast<uint32_t>(low)) & AddressMask;
	const auto immediate = static_cast<int64_t>(static_cast<int32_t>(mem.offset));
	uint64_t   address   = 0;
	if (inst.GetOpcode() == ValueOpcode::ReadConstBuffer || vector) {
		uint64_t records = 0;
		uint64_t word3   = 0;
		if (handle->NumArgs() != 4u || !Arg(*handle, 2, records) || !Arg(*handle, 3, word3)) {
			return false;
		}
		if (immediate < 0) {
			return false;
		}
		const auto byte_offset =
		    (static_cast<uint64_t>(immediate) & ~uint64_t {3}) + (static_cast<uint32_t>(offset) & ~3u);
		const auto stride  = (static_cast<uint32_t>(high) >> 16u) & 0x3fffu;
		if (vector) {
			// Only the uniform, unswizzled structured DWORD address is evaluated on the host.
			if ((high & (1u << 31u)) != 0u || (word3 & ((1u << 23u) | 0xf0000000u)) != 0u)
				return false;
			if (stride == 0u || records == 0u || ((word3 >> 12u) & 0x7fu) == 0u) {
				result = 0u;
				return true;
			}
		}
		const auto size = stride == 0u
		                      ? static_cast<uint64_t>(static_cast<uint32_t>(records))
		                      : static_cast<uint64_t>(stride) * static_cast<uint32_t>(records);
		if (byte_offset > size || size - byte_offset < sizeof(uint32_t)) {
			result = 0u;
			return true;
		}
		address = (base & ~uint64_t {3}) + byte_offset;
	} else {
		const auto relative = (immediate & ~int64_t {3}) +
		                      static_cast<int64_t>(static_cast<uint32_t>(offset) & ~3u);
		if (!AddSignedAddress(base & ~uint64_t {3}, relative, address)) {
			return false;
		}
	}
	uint32_t word = 0;
	const auto reader = vector ? m_runtime.read_specialization_memory : m_runtime.read_memory;
	// A host-memory read through a table below guest memory reads zero, whatever its offset.
	if (!vector && base < GUEST_MEMORY_MIN &&
	    (reader == nullptr || SrtReadCapture::ReadsHostMemory(m_runtime))) {
		NullDesc::NoteNullTable(base);
		result = 0;
		return true;
	}
	if (reader != nullptr) {
		const bool read_ok = reader(m_runtime.userdata, address, {&word, 1});
		if (t_cf_rec != nullptr) t_cf_rec->NoteRead(reader, address, word, read_ok); // S451
		if (!read_ok) {
			return false;
		}
	} else {
		if (vector) return false;
		if (address >= GUEST_MEMORY_MIN) {
			std::memcpy(&word, reinterpret_cast<const void*>(address), sizeof(word));
		} else {
			NullDesc::NoteUnresolved(address);
		}
		if (t_cf_rec != nullptr) t_cf_rec->NoteRead(nullptr, address, word, true); // S451
		NullDesc::NoteSrtRead(address, std::span<const uint32_t>(&word, 1), false);
	}
	result = word;
	return true;
}

// S451 DIAG: calls of EvaluateInst / of ReadFirstLane on this thread; the control-flow loop reports the difference.
static thread_local uint64_t t_eval_inst_calls = 0;
static thread_local uint64_t t_rfl_calls       = 0;

bool SrtWalker::EvaluateInst(const Inst& inst, uint64_t& result) {
	++t_eval_inst_calls;
	uint64_t   a       = 0;
	uint64_t   b       = 0;
	uint64_t   c       = 0;
	const auto binary  = [&]() { return Arg(inst, 0, a) && Arg(inst, 1, b); };
	const auto ternary = [&]() {
		return Arg(inst, 0, a) && Arg(inst, 1, b) && Arg(inst, 2, c);
	};
	switch (inst.GetOpcode()) {
		case ValueOpcode::GetUserData: {
			const auto reg = RegIndex(inst.Arg(0).ScalarRegister());
			if (reg < m_program.user_data_base ||
			    reg - m_program.user_data_base >= m_runtime.user_data.size()) {
				return false;
			}
			result = m_runtime.user_data[reg - m_program.user_data_base];
			if (t_cf_rec != nullptr) t_cf_rec->NoteUserData(reg - m_program.user_data_base, m_runtime.user_data[reg - m_program.user_data_base]); // S451
			return true;
		}
		case ValueOpcode::GetShaderBase:
			result = m_runtime.shader_base;
			if (t_cf_rec != nullptr) t_cf_rec->NoteShaderBase(m_runtime.shader_base); // S451
			return true;
		case ValueOpcode::Phi: return EvaluatePhi(inst, result);
		case ValueOpcode::ReadFirstLane: {
			++t_rfl_calls;
			if (!m_active_mask.IsEmpty() && inst.Arg(1).Resolve() == m_active_mask) {
				return EvaluateWide(inst.Arg(0), result);
			}
			SrtWalker clean_active(*this, inst.Arg(1), true);
			SrtWalker active(*this, inst.Arg(1), false, &clean_active);
			return active.EvaluateWide(inst.Arg(0), result);
		}
		case ValueOpcode::BitCastU32F32:
		case ValueOpcode::BitCastF32U32: return Arg(inst, 0, result);
		case ValueOpcode::CompositeExtractU64:
		case ValueOpcode::CompositeExtractU32x2: return EvaluateExtract(inst, result);
		case ValueOpcode::CompositeConstructU64:
			if (!binary()) {
				return false;
			}
			result = static_cast<uint32_t>(a) |
			         (static_cast<uint64_t>(static_cast<uint32_t>(b)) << 32u);
			return true;
		case ValueOpcode::LoadAddressU32:
		case ValueOpcode::ReadConstBuffer:
		case ValueOpcode::LoadBufferU32:
			if (IsRawRead(m_program, inst)) {
				if (m_clean_evaluator != nullptr &&
				    (inst.Flags<SrtReadFlags>().clean != 0u ||
				     (inst.GetOpcode() == ValueOpcode::LoadBufferU32 &&
				      m_clean_evaluator->m_active_mask == m_active_mask))) {
					return m_clean_evaluator->EvaluateWide(Value(const_cast<Inst*>(&inst)), result);
				}
				return EvaluateRawRead(inst, result);
			}
			break;
		case ValueOpcode::IAdd32:
			if (binary()) {
				result = static_cast<uint32_t>(a + b);
				return true;
			}
			return false;
		case ValueOpcode::IAdd64:
			if (binary()) {
				result = a + b;
				return true;
			}
			return false;
		case ValueOpcode::ISub32:
			if (binary()) {
				result = static_cast<uint32_t>(a - b);
				return true;
			}
			return false;
		case ValueOpcode::ISub64:
			if (binary()) {
				result = a - b;
				return true;
			}
			return false;
		case ValueOpcode::IMul32:
			if (binary()) {
				result = static_cast<uint32_t>(a * b);
				return true;
			}
			return false;
		case ValueOpcode::IMul64:
			if (binary()) {
				result = a * b;
				return true;
			}
			return false;
		case ValueOpcode::UMulHi:
			if (binary()) {
				result = (static_cast<uint64_t>(static_cast<uint32_t>(a)) * static_cast<uint32_t>(b)) >> 32u;
				return true;
			}
			return false;
		case ValueOpcode::UMin32:
		case ValueOpcode::UMax32:
			if (binary()) {
				result = inst.GetOpcode() == ValueOpcode::UMin32
				    ? std::min(static_cast<uint32_t>(a), static_cast<uint32_t>(b))
				    : std::max(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::ConvertF32U32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(static_cast<float>(static_cast<uint32_t>(a)));
				return true;
			}
			return false;
		case ValueOpcode::ConvertU32F32:
			if (Arg(inst, 0, a)) {
				const auto value = Float32(a);
				if (!std::isfinite(value) || value < 0.0f ||
				    static_cast<double>(value) > UINT32_MAX) {
					return false;
				}
				result = static_cast<uint32_t>(value);
				return true;
			}
			return false;
		case ValueOpcode::FPMul32:
			if (binary()) {
				result = std::bit_cast<uint32_t>(Float32(a) * Float32(b));
				return true;
			}
			return false;
		case ValueOpcode::FPTrunc32:
			if (Arg(inst, 0, a)) {
				result = std::bit_cast<uint32_t>(std::trunc(Float32(a)));
				return true;
			}
			return false;
		case ValueOpcode::FPRecipIFlag32:
			if (Arg(inst, 0, a)) {
				const auto exponent = (a >> 23u) & 0xffu;
				// Normal positive powers of two have exact normal reciprocals in every FP mode.
				if ((a & 0x807fffffu) != 0u || exponent == 0u || exponent >= 254u) return false;
				result = (254u - exponent) << 23u;
				return true;
			}
			return false;
		case ValueOpcode::FPIsNan32:
			if (Arg(inst, 0, a)) {
				result = std::isnan(Float32(a));
				return true;
			}
			return false;
		case ValueOpcode::FPOrdLessThanEqual32:
		case ValueOpcode::FPOrdGreaterThanEqual32:
			if (binary()) {
				const auto operand = [&](uint64_t bits) {
					if (inst.Flags<FPCompareFlags>().flush_input_denorms &&
					    (bits & 0x7fffffffu) < 0x00800000u) {
						bits &= 0x80000000u;
					}
					return Float32(bits);
				};
				result = inst.GetOpcode() == ValueOpcode::FPOrdLessThanEqual32
				             ? operand(a) <= operand(b)
				             : operand(a) >= operand(b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd32:
			if (binary()) {
				result = static_cast<uint32_t>(a & b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseAnd64:
			if (binary()) {
				result = a & b;
				return true;
			}
			return false;
		case ValueOpcode::BitwiseOr32:
			if (binary()) {
				result = static_cast<uint32_t>(a | b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseXor32:
			if (binary()) {
				result = static_cast<uint32_t>(a ^ b);
				return true;
			}
			return false;
		case ValueOpcode::BitwiseNot32:
			if (Arg(inst, 0, a)) {
				result = ~static_cast<uint32_t>(a);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) << (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftLeftLogical64:
			if (binary()) {
				result = a << (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical32:
			if (binary()) {
				result = static_cast<uint32_t>(a) >> (b & 31u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightLogical64:
			if (binary()) {
				result = a >> (b & 63u);
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic32:
			if (binary()) {
				result = static_cast<uint32_t>(
				    std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >> (b & 31u));
				return true;
			}
			return false;
		case ValueOpcode::ShiftRightArithmetic64:
			if (binary()) {
				result = static_cast<uint64_t>(std::bit_cast<int64_t>(a) >> (b & 63u));
				return true;
			}
			return false;
		case ValueOpcode::BitFieldUExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				const auto mask = width == 32u  ? UINT32_MAX
				                  : width == 0u ? 0u
				                                : (uint32_t {1} << width) - 1u;
				result = width == 0u ? 0u : (static_cast<uint32_t>(a) >> offset) & mask;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldSExtract:
			if (ternary()) {
				const auto offset = static_cast<uint32_t>(b);
				const auto width  = static_cast<uint32_t>(c);
				if (offset > 32u || width > 32u - offset) {
					return false;
				}
				if (width == 0u) {
					result = 0;
					return true;
				}
				const auto mask = width == 32u ? UINT32_MAX : (uint32_t {1} << width) - 1u;
				auto       bits = (static_cast<uint32_t>(a) >> offset) & mask;
				if (width < 32u && (bits & (uint32_t {1} << (width - 1u))) != 0u) {
					bits |= ~mask;
				}
				result = bits;
				return true;
			}
			return false;
		case ValueOpcode::BitFieldInsert: {
			uint64_t d = 0;
			if (!ternary() || !Arg(inst, 3, d)) {
				return false;
			}
			const auto offset = static_cast<uint32_t>(c);
			const auto width  = static_cast<uint32_t>(d);
			if (offset > 32u || width > 32u - offset) {
				return false;
			}
			if (width == 0u) {
				result = static_cast<uint32_t>(a);
				return true;
			}
			const auto mask =
			    width == 32u ? UINT32_MAX : ((uint32_t {1} << width) - 1u) << offset;
			result = (static_cast<uint32_t>(a) & ~mask) |
			         ((static_cast<uint32_t>(b) << offset) & mask);
			return true;
		}
		case ValueOpcode::SelectU32:
		case ValueOpcode::SelectU1:
		case ValueOpcode::SelectF32: {
			auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
			if (predicate.EvaluateWide(inst.Arg(0), a)) {
				return Arg(inst, a != 0u ? 1u : 2u, result);
			}
			return false;
		}
		case ValueOpcode::IEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) == static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::INotEqual32:
			if (binary()) {
				result = static_cast<uint32_t>(a) != static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThan32:
			// No U32 is below zero, including an unknown loop-carried value.
			if (!Arg(inst, 1, b)) return false;
			if (static_cast<uint32_t>(b) == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 0, a)) {
				result = static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::ULessThanEqual32:
			// A zero lower bound accepts every U32, including a varying lane value.
			if (!Arg(inst, 0, a)) return false;
			if (a == 0u) {
				result = 1u;
				return true;
			}
			if (Arg(inst, 1, b)) {
				result = static_cast<uint32_t>(a) <= static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::UGreaterThan32:
			if (binary()) {
				result = static_cast<uint32_t>(a) > static_cast<uint32_t>(b);
				return true;
			}
			return false;
		case ValueOpcode::SGreaterThanEqual32:
			if (binary()) {
				result = std::bit_cast<int32_t>(static_cast<uint32_t>(a)) >=
				         std::bit_cast<int32_t>(static_cast<uint32_t>(b));
				return true;
			}
			return false;
		case ValueOpcode::LogicalAnd: {
			const bool left = Arg(inst, 0, a);
			if (left && a == 0u) {
				result = 0u;
				return true;
			}
			if (Arg(inst, 1, b) && (b == 0u || left)) {
				result = b != 0u;
				return true;
			}
			if (!left) {
				// Only lanes satisfying the left EXEC factor can make the conjunction
				// true. Use a separate memo so this assumption cannot escape its RHS.
				SrtWalker active(m_clean_evaluator != nullptr ? *m_clean_evaluator : *this,
				                 inst.Arg(0));
				if (active.EvaluateWide(inst.Arg(1), b) && b == 0u) {
					result = 0u;
					return true;
				}
			}
			return false;
		}
		case ValueOpcode::LogicalOr: {
			const bool left = Arg(inst, 0, a);
			if (left && a != 0u) {
				result = 1u;
				return true;
			}
			if (!Arg(inst, 1, b) || (b == 0u && !left)) return false;
			result = b != 0u;
			return true;
		}
		case ValueOpcode::LogicalXor:
			if (binary()) {
				result = (a != 0u) != (b != 0u);
				return true;
			}
			return false;
		case ValueOpcode::ConditionRef: return Arg(inst, 0, result);
		case ValueOpcode::LogicalNot:
			if (Arg(inst, 0, a)) {
				result = a == 0u;
				return true;
			}
			return false;
		case ValueOpcode::UndefU1:
		case ValueOpcode::UndefU8:
		case ValueOpcode::UndefU16:
		case ValueOpcode::UndefU32:
		case ValueOpcode::UndefU64: return false;
		default: break;
	}
	return false;
}
bool SrtWalker::EvaluateDescriptor(uint32_t source, DescriptorValue& result) {
	if (source >= m_program.descriptor_sources.size()) {
		return false;
	}
	const auto& descriptor = m_program.descriptor_sources[source];
	result = {};
	result.dword_count = descriptor.dword_count;
	for (uint32_t index = 0; index < descriptor.dword_count; ++index) {
		if (!Evaluate(descriptor.dwords[index], result.dwords[index])) {
			return false;
		}
	}
	return true;
}

// S441 DIAG: a walk that did not get the tree JIT, with its reason.
static void NoteTreeRejectFor(uint64_t plan, SrtTreeReject::Reason reason, const std::string& detail) {
	if (SrtTreeReject::Active().load(std::memory_order_relaxed)) SrtTreeReject::Global().Note(plan, reason, detail);
}

// S442: a plan that reads nothing from guest memory and has no block conditions. Its walk evaluates no value at all, so the flat buffer is empty and
// no descriptor source is switched off, whatever the user data and the guest memory hold.
static bool IsNoReadsPlan(const ResourcePlan& program) {
	return program.srt_plan_complete && program.srt_reads.empty() && program.control_flow.empty() &&
	       !program.capture_specialization_reads;
}

bool SrtWalker::RefreshFlatBuffer(std::vector<uint32_t>& flat) {
	const auto mode = SrtJitCover::Current();
	if (mode == SrtJitCover::Mode::Off || !IsNoReadsPlan(m_program)) return RefreshFlatBufferWalk(flat);
	std::vector<uint32_t> planned;
#ifdef KYTY_SRTCOVER_MUTANT_NOREADS_IGNORES_USERDATA
	if (!m_runtime.user_data.empty()) planned.push_back(m_runtime.user_data[0]);
#endif
	if (mode == SrtJitCover::Mode::On) {
		SrtJitCover::Note(SrtJitCover::Cell::NoReads);
		flat = std::move(planned);
		m_program.active_sources.clear();
		return true;
	}
	// shadow: the walk is the result; the plan's answer must equal it bit for bit.
	const bool walked = RefreshFlatBufferWalk(flat);
	SrtJitCover::Note(SrtJitCover::Cell::NoReads);
	if (!walked || flat != planned || !m_program.active_sources.empty()) {
		SrtJitCover::Note(SrtJitCover::Cell::Ne);
		if (SrtJitCover::ShouldLogNe()) {
			char line[160];
			std::snprintf(line, sizeof(line), "[srt-cover-ne] noreads plan=0x%016llx walked=%d flat=%zu planned=%zu\n",
			              static_cast<unsigned long long>(m_program.shader_hash), walked ? 1 : 0, flat.size(), planned.size());
			SrtJitCover::LogSink()(line);
		}
	}
	return walked;
}

bool SrtWalker::RefreshFlatBufferWalk(std::vector<uint32_t>& flat) {
	const auto NoteTreeReject = [&](SrtTreeReject::Reason reason, const std::string& detail) { NoteTreeRejectFor(m_program.shader_hash, reason, detail); };
	if (!m_program.srt_plan_complete) return false;
	// nojit keeps the interpreted memcpy walk so a bad compiled walker can be separated from the read itself.
	if (m_program.srt_code && m_runtime.read_memory == nullptr && SrtReader::UseCompiledWalker()) {
		flat.resize(m_program.srt_reads.size());
		m_program.active_sources.clear();
		m_context.values.resize(m_program.evaluation_value_count);
		using Walker = bool KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*,
		    ResourcePlan::EvaluationContext::Entry*, uint64_t, size_t);
		if (CpSampler::Pc::WalkActive().load(std::memory_order_relaxed)) CpSampler::Pc::Tally().jit_tree_calls++; // S439 DIAG
		std::optional<FrameTiming::ScopedCounterNs> tree_timer; // S451: graphics only
		if (m_program.stage != ShaderType::Compute) tree_timer.emplace(FrameTiming::Counter::GfpWalkTreeNs);
		if (m_program.srt_code->getCode<Walker>()(m_runtime.user_data.data(), flat.data(),
		                                          m_context.values.data(), m_context.generation,
		                                          m_runtime.user_data.size())) {
			SrtWalkPaths::Note(SrtWalkPaths::Path::Tree);
			return true;
		}
		SrtWalkPaths::Note(SrtWalkPaths::Path::TreeFail);
		NoteTreeReject(SrtTreeReject::Reason::TreeReturnedFalse, "");
	} else if (m_program.srt_code && m_runtime.read_memory != nullptr) {
		SrtWalkPaths::Note(SrtWalkPaths::Path::Capture);
		NoteTreeReject(SrtTreeReject::Reason::CaptureRuntime, "");
	} else if (m_program.srt_code) {
		NoteTreeReject(SrtTreeReject::Reason::NojitEnv, "");
	} else {
		NoteTreeReject(static_cast<SrtTreeReject::Reason>(m_program.tree_reject), m_program.tree_reject_detail);
	}
	SrtWalkPaths::Note(SrtWalkPaths::Path::InterpWalk);
	const bool pc_diag = CpSampler::Pc::WalkActive().load(std::memory_order_relaxed); // S439 DIAG
	// Compiled slots read mapped memory directly, as the ordinary reader of a capture does.
	auto*      captured = SrtReadCapture::DirectRanges(m_runtime);
	const bool compiled = m_program.srt_read_code && m_active_mask.IsEmpty() &&
	                      (m_runtime.read_memory == nullptr || captured != nullptr);
	if (compiled) m_context.values.resize(m_program.evaluation_value_count);
	const auto cover = SrtJitCover::Current(); // S442
	// S442 shadow: the constant-buffer JIT of a slot runs on a scratch memo and flat buffer; the interpreter's value is the result and must equal it.
	std::vector<ResourcePlan::EvaluationContext::Entry> cover_memo;
	std::vector<uint32_t>                               cover_flat;
	// S445 shadow: where the per-slot path records the addresses it read and the first slot whose compiled read failed (null / UINT32_MAX: not recording).
	std::vector<uint64_t>* shadow_addresses = nullptr;
	uint32_t               shadow_failed    = UINT32_MAX;
	size_t                 shadow_cut       = 0; // addresses recorded up to and including the failing slot
	const auto refresh = [&](uint32_t slot, bool allow_jit = true) {
		if (slot >= m_program.srt_reads.size()) return false;
		bool     cover_compared = false;
		uint32_t cover_word     = 0;
		if (allow_jit && compiled && m_program.srt_read_entries[slot] != UINT32_MAX) {
			using Read = bool KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*,
			                                    ResourcePlan::EvaluationContext::Entry*, uint64_t,
			                                    size_t, uint64_t**);
			const bool cover_cb = cover != SrtJitCover::Mode::Off && slot < m_program.srt_read_cb.size() && m_program.srt_read_cb[slot] != 0u;
			if (cover_cb && cover == SrtJitCover::Mode::Shadow) {
				// shadow: scratch memo and flat buffer, nothing captured; the interpreter below is the result.
				cover_memo.assign(m_program.evaluation_value_count, {});
				cover_flat.assign(flat.size(), 0u);
				auto* const addresses = m_program.srt_read_addresses.data();
				auto*       cursor    = addresses;
				cover_compared = reinterpret_cast<Read>(m_program.srt_read_code->getCode<uint8_t*>() + m_program.srt_read_entries[slot])(
				    m_runtime.user_data.data(), cover_flat.data(), cover_memo.data(), m_context.generation, m_runtime.user_data.size(), &cursor);
				if (cover_compared) cover_word = cover_flat[m_program.srt_reads[slot].flat_offset];
			} else {
				auto* const       addresses = m_program.srt_read_addresses.data();
				auto*             cursor    = addresses;
				const bool ok = reinterpret_cast<Read>(m_program.srt_read_code->getCode<uint8_t*>() +
				                                       m_program.srt_read_entries[slot])(
				    m_runtime.user_data.data(), flat.data(), m_context.values.data(),
				    m_context.generation, m_runtime.user_data.size(), &cursor);
				if (pc_diag) { // S439 DIAG: the JIT slot reads are the 32-bit loads listed in the address cursor
					CpSampler::Pc::Tally().jit_slot_calls++;
					CpSampler::Pc::NoteReads(addresses, static_cast<size_t>(cursor - addresses), sizeof(uint32_t));
				}
				if (captured != nullptr) {
					for (const auto* address = addresses; address != cursor; ++address) {
						captured->emplace_back(*address, sizeof(uint32_t));
					}
				}
				if (shadow_addresses != nullptr) {
					shadow_addresses->insert(shadow_addresses->end(), addresses, cursor);
					if (!ok && shadow_failed == UINT32_MAX) {
						shadow_failed = slot;
						shadow_cut    = shadow_addresses->size();
					}
				}
				// A failed read is evaluated again below, with the walker's own result.
				if (ok) {
					SrtWalkPaths::Note(SrtWalkPaths::Path::SlotJit);
					if (cover != SrtJitCover::Mode::Off) { // S442
						if (cover_cb) SrtJitCover::Note(SrtJitCover::Cell::ConstBuffer);
						if (m_program.capture_specialization_reads) SrtJitCover::Note(SrtJitCover::Cell::Capture);
					}
					return true;
				}
			}
		}
		SrtWalkPaths::Note(SrtWalkPaths::Path::InterpSlots);
		if (SrtTreeReject::Active().load(std::memory_order_relaxed)) { // S441 DIAG
			const char* why = "no_slot_code";
			if (m_program.srt_read_code) {
				if (slot < m_program.srt_read_entries.size() && m_program.srt_read_entries[slot] == UINT32_MAX)
					why = slot < m_program.srt_slot_reject.size() ? m_program.srt_slot_reject[slot].c_str() : "not_compiled";
				else
					why = compiled ? "jit_read_failed" : "slot_jit_off_at_runtime";
			}
			SrtTreeReject::Global().Note(m_program.shader_hash, SrtTreeReject::Reason::SlotInterp, why);
		}
		if (pc_diag) CpSampler::Pc::Tally().interp_calls++; // S439 DIAG
		const auto& read = m_program.srt_reads[slot];
		const auto* inst = read.value.ResolveInstruction();
		const bool clean = inst != nullptr && inst->Flags<SrtReadFlags>().clean != 0u;
		if (clean && (m_clean_evaluator == nullptr || m_runtime.read_specialization_memory == nullptr))
			return false;
		auto& evaluator = clean ? *m_clean_evaluator : *this;
		SpecShadow::PurposeScope purpose(clean ? SpecShadow::Purpose::Slot : SpecShadow::CurrentPurpose()); // S406
		const bool evaluated = read.flat_offset < flat.size() && evaluator.Evaluate(read.value, flat[read.flat_offset]);
		if (cover_compared && evaluated) { // S442 shadow
			SrtJitCover::Note(SrtJitCover::Cell::ConstBuffer);
			if (cover_word != flat[read.flat_offset]) {
				SrtJitCover::Note(SrtJitCover::Cell::Ne);
				if (SrtJitCover::ShouldLogNe()) {
					char line[160];
					std::snprintf(line, sizeof(line), "[srt-cover-ne] cb plan=0x%016llx slot=%u jit=0x%08x walk=0x%08x\n",
					              static_cast<unsigned long long>(m_program.shader_hash), slot, cover_word, flat[read.flat_offset]);
					SrtJitCover::LogSink()(line);
				}
			}
		}
		return evaluated;
	};
	// S445 (KYTY_SRT_SLOT_BATCH): the runs of compiled slots of a visited block. Not with the S442 constant-buffer shadow (its per-slot path compares scratch results).
	const auto batch_mode = (compiled && !m_program.srt_block_runs.empty() && cover != SrtJitCover::Mode::Shadow) ? SrtSlotBatch::Current() : SrtSlotBatch::Mode::Off;
	using BatchEntry      = uint32_t KYTY_SYSV_ABI (*)(const uint32_t*, uint32_t*, ResourcePlan::EvaluationContext::Entry*, uint64_t, size_t, uint64_t**);
	std::vector<ResourcePlan::EvaluationContext::Entry> batch_memo; // shadow scratch
	std::vector<uint32_t>                               batch_flat;
	std::vector<uint64_t>                               batch_addresses;
	std::vector<uint64_t>                               batch_reference;
	// Fills one run with one entry on `memo` / `out` / `addresses`; returns 0 or the 1-based position of the slot that failed, and the cursor.
	const auto enter_batch = [&](const ResourcePlan::SrtBatchRun& run, ResourcePlan::EvaluationContext::Entry* memo, uint32_t* out, uint64_t* addresses, uint64_t*& cursor) {
		cursor = addresses;
		return reinterpret_cast<BatchEntry>(m_program.srt_batch_code->getCode<uint8_t*>() + run.entry)(
		    m_runtime.user_data.data(), out, memo, m_context.generation, m_runtime.user_data.size(), &cursor);
	};
	// Counts of a finished run that filled `done` slots (the failing one is not among them).
	const auto note_batch = [&](const ResourcePlan::SrtBatchRun& run, const std::vector<uint32_t>& list, uint32_t done, bool failed) {
		SrtSlotBatch::Note(SrtSlotBatch::Cell::Runs);
		SrtSlotBatch::Note(SrtSlotBatch::Cell::Slots, done);
		if (failed) SrtSlotBatch::Note(SrtSlotBatch::Cell::Fail);
		SrtWalkPaths::Note(SrtWalkPaths::Path::SlotJit, done);
		if (cover != SrtJitCover::Mode::Off) {
			uint64_t cb = run.cb;
			if (failed) {
				cb = 0;
				for (uint32_t k = 0; k < done; ++k)
					if (list[run.first + k] < m_program.srt_read_cb.size() && m_program.srt_read_cb[list[run.first + k]] != 0u) ++cb;
			}
			if (cb != 0u) SrtJitCover::Note(SrtJitCover::Cell::ConstBuffer, cb);
			if (m_program.capture_specialization_reads && done != 0u) SrtJitCover::Note(SrtJitCover::Cell::Capture, done);
		}
	};
	// KYTY_SRT_SLOT_BATCH=1: one entry for the run. A slot that fails goes to the interpreter (no second try of its JIT), the rest of the run through the per-slot entries.
	const auto refresh_run = [&](const ResourceBlock& block, const ResourcePlan::SrtBatchRun& run) {
		auto* const addresses = m_program.srt_read_addresses.data();
		uint64_t*   cursor    = nullptr;
		const auto  result    = enter_batch(run, m_context.values.data(), flat.data(), addresses, cursor);
		if (pc_diag) { // S439 DIAG
			CpSampler::Pc::Tally().jit_slot_calls += run.count;
			CpSampler::Pc::NoteReads(addresses, static_cast<size_t>(cursor - addresses), sizeof(uint32_t));
		}
		if (captured != nullptr) {
			for (const auto* address = addresses; address != cursor; ++address) captured->emplace_back(*address, sizeof(uint32_t));
		}
		const uint32_t done = result == 0u ? run.count : result - 1u;
		note_batch(run, block.srt_reads, done, result != 0u);
		if (result == 0u) return true;
		if (!refresh(block.srt_reads[run.first + done], false)) return false;
		for (uint32_t k = done + 1u; k < run.count; ++k)
			if (!refresh(block.srt_reads[run.first + k])) return false;
		return true;
	};
	// KYTY_SRT_SLOT_BATCH=shadow: the run on a scratch copy of the memo, then the per-slot path (the result) with its addresses recorded; the two must agree.
	const auto shadow_run = [&](const ResourceBlock& block, const ResourcePlan::SrtBatchRun& run) {
		const auto& list = block.srt_reads;
#ifdef KYTY_SRTBATCH_MUTANT_SHADOW_REAL_MEMO
		auto* const memo = m_context.values.data();
#else
		batch_memo.assign(m_context.values.begin(), m_context.values.end());
		auto* const memo = batch_memo.data();
#endif
		batch_flat.assign(flat.size(), 0u);
		batch_addresses.assign(m_program.srt_read_addresses.size(), 0u);
		uint64_t*  cursor = nullptr;
		const auto result = enter_batch(run, memo, batch_flat.data(), batch_addresses.data(), cursor);
		const uint32_t done = result == 0u ? run.count : result - 1u;
		const std::vector<uint64_t> shadow_seen(batch_addresses.data(), cursor);
		note_batch(run, list, done, result != 0u);
		batch_reference.clear();
		shadow_addresses = &batch_reference;
		shadow_failed    = UINT32_MAX;
		shadow_cut       = 0;
		bool ok          = true;
		for (uint32_t k = 0; k < run.count && ok; ++k) ok = refresh(list[run.first + k]);
		shadow_addresses = nullptr;
		const char* why  = nullptr;
		uint32_t    at   = 0;
		const uint32_t expect_failed = result == 0u ? UINT32_MAX : list[run.first + result - 1u];
		if (ok) {
			if (shadow_failed != expect_failed) why = "fail";
			for (uint32_t k = 0; k < done && why == nullptr; ++k) {
				const auto offset = m_program.srt_reads[list[run.first + k]].flat_offset;
				if (flat[offset] != batch_flat[offset]) {
					why = "flat";
					at  = k;
				}
			}
			// The batch stops at its failing slot: compare with the per-slot reads up to the same slot.
			if (shadow_failed != UINT32_MAX) batch_reference.resize(shadow_cut);
			if (why == nullptr && shadow_seen != batch_reference) why = "addr";
		}
		if (why != nullptr) {
			SrtSlotBatch::Note(SrtSlotBatch::Cell::Mismatch);
			if (SrtSlotBatch::ShouldLog()) {
				char line[200];
				std::snprintf(line, sizeof(line), "[srt-batch-shadow] hash=0x%016llx first=%u count=%u slot=%u why=%s batch_reads=%zu slot_reads=%zu\n",
				              static_cast<unsigned long long>(m_program.shader_hash), run.first, run.count, list[run.first + at], why, shadow_seen.size(), batch_reference.size());
				SrtSlotBatch::LogSink()(line);
			}
		}
		return ok;
	};
	// The slots of a visited block: runs of two or more through refresh_run / shadow_run, everything else per slot.
	const auto refresh_block = [&](uint32_t index, const ResourceBlock& block) {
		if (batch_mode == SrtSlotBatch::Mode::Off || index >= m_program.srt_block_runs.size() || m_program.srt_block_runs[index].empty()) {
			for (const auto slot: block.srt_reads)
				if (!refresh(slot)) return false;
			return true;
		}
		const auto& runs = m_program.srt_block_runs[index];
		size_t      next = 0;
		for (uint32_t position = 0; position < block.srt_reads.size();) {
			if (next < runs.size() && runs[next].first == position) {
				const auto& run = runs[next++];
				if (run.entry != UINT32_MAX && run.count >= SrtSlotBatch::kMinRun) {
					if (batch_mode == SrtSlotBatch::Mode::On ? !refresh_run(block, run) : !shadow_run(block, run)) return false;
					position += run.count;
					continue;
				}
				SrtSlotBatch::Note(SrtSlotBatch::Cell::Single, run.count);
			}
			if (!refresh(block.srt_reads[position])) return false;
			++position;
		}
		return true;
	};
	auto& active = m_program.active_sources;
	if (m_program.control_flow.empty()) {
		active.clear();
		flat.resize(m_program.srt_reads.size());
		std::optional<FrameTiming::ScopedCounterNs> slots_timer; // S451
		if (m_program.stage != ShaderType::Compute) slots_timer.emplace(FrameTiming::Counter::GfpWalkSlotsNs);
		for (uint32_t slot = 0; slot < m_program.srt_reads.size(); ++slot) {
			if (!refresh(slot)) return false;
		}
		return true;
	}
	std::optional<FrameTiming::ScopedCounterNs> cf_timer; // S451
	const bool gfp_cf = m_program.stage != ShaderType::Compute;
	const uint64_t cf_eval0 = t_eval_inst_calls, cf_rfl0 = t_rfl_calls;
	uint64_t       cf_iter = 0, cf_cond = 0;
	if (gfp_cf) cf_timer.emplace(FrameTiming::Counter::GfpWalkCfNs);
	auto& visited = m_program.visited_blocks;
	auto& reachable = m_program.reachable_blocks;
	auto& pending = m_program.pending_blocks;
	// S451 (KYTY_PROGS_MEMO): the loop is a function of the user data words and shader base it reads and of the guest words it reads (srtCfMemo.h).
	// The replay goes through the runtime's own readers, so a capturing runtime (SrtReadCapture) records the replayed reads as the loop's would have been.
	// Not for plans with retained call-source reads (the plan is edited in place), under the NullDesc diag, or for a nested active mask.
	CfMemoMode cf_mode = CfMemoMode::Off;
	if (gfp_cf) {
		cf_mode = CfMemoModeFromEnv();
		if (cf_mode != CfMemoMode::Off &&
		    (!m_active_mask.IsEmpty() || !m_program.source_reads.empty() || NullDesc::Enabled())) {
			FrameTiming::Add(FrameTiming::Counter::GfpMemoSkipN);
			cf_mode = CfMemoMode::Off;
		}
	}
	auto&                cf_set  = m_program.cf_memo;
	const SrtCfMemo*     cf_way  = nullptr;
	SrtCfMemo*           cf_rec  = nullptr;
	std::vector<uint8_t> cf_expect;
	if (cf_mode != CfMemoMode::Off) {
		{
			SpecShadow::PurposeScope purpose(SpecShadow::Purpose::Cond); // the replayed reads are the ones the loop issues under this purpose
			cf_way = cf_set.Find(m_runtime.user_data, m_runtime.shader_base, m_runtime.read_memory, m_runtime.read_specialization_memory, m_runtime.userdata);
			if (cf_way != nullptr && cf_way->visited.size() != m_program.control_flow.size()) cf_way = nullptr;
		}
		FrameTiming::Add(cf_way != nullptr ? FrameTiming::Counter::GfpMemoHitN : FrameTiming::Counter::GfpMemoMissN);
	}
	const bool cf_hit  = cf_way != nullptr;
	const bool cf_skip = cf_hit && cf_mode == CfMemoMode::On;
	if (cf_skip) {
		visited            = cf_way->visited;
		m_reachable_blocks = &visited;
		if (m_clean_evaluator != nullptr) m_clean_evaluator->m_reachable_blocks = &visited;
	} else {
		if (cf_mode != CfMemoMode::Off) {
			if (cf_hit) cf_expect = cf_way->visited; // shadow: what the memo would have answered
			cf_rec = &cf_set.Victim(m_runtime.user_data, m_runtime.shader_base, m_runtime.read_memory, m_runtime.read_specialization_memory);
			cf_rec->Begin(static_cast<uint32_t>(m_runtime.user_data.size()), m_runtime.read_memory, m_runtime.read_specialization_memory);
			t_cf_rec = cf_rec;
		}
		visited.assign(m_program.control_flow.size(), 1u);
		ResetUnknowns();
		m_reachable_blocks = &visited;
		if (m_clean_evaluator != nullptr) {
			m_clean_evaluator->ResetUnknowns();
			m_clean_evaluator->m_reachable_blocks = &visited;
		}
		for (;;) {
			++cf_iter;
			reachable.assign(m_program.control_flow.size(), 0u);
			pending.clear();
			pending.push_back(0u);
			while (!pending.empty()) {
				const auto index = pending.back();
				pending.pop_back();
				if (reachable.at(index)) continue;
				reachable[index] = 1u;
				const auto& block = m_program.control_flow[index];
				uint32_t condition = 0;
				auto& predicate = m_clean_evaluator != nullptr ? *m_clean_evaluator : *this;
				SpecShadow::PurposeScope purpose(SpecShadow::Purpose::Cond); // S406
				if (!block.condition.IsEmpty()) ++cf_cond;
				if (!block.condition.IsEmpty() &&
				    (m_runtime.read_specialization_memory != nullptr || !m_program.capture_specialization_reads) &&
				    predicate.Evaluate(block.condition, condition)) {
					pending.push_back(block.successors[condition != 0u ? 0u : 1u]);
				} else {
					pending.insert(pending.end(), block.successors.begin(), block.successors.end());
				}
			}
			if (reachable == visited) break;
			visited.swap(reachable);
			if (m_program.control_phi_predecessors.empty()) break;
			// Only unknown values need another evaluation: successful Phi facts remain
			// true as their possible predecessors shrink. Keep uniform-read memos stable.
			ResetUnknowns();
			if (m_clean_evaluator != nullptr) m_clean_evaluator->ResetUnknowns();
		}
		if (cf_mode != CfMemoMode::Off) {
			t_cf_rec = nullptr;
			cf_rec->End(visited);
			if (cf_hit && cf_expect != visited) FrameTiming::Add(FrameTiming::Counter::GfpMemoNeN); // shadow: the memo's answer differs (must be 0)
		}
	}
	cf_timer.reset();
	if (gfp_cf) {
		FrameTiming::Add(FrameTiming::Counter::GfpCfWalkN);
		FrameTiming::Add(FrameTiming::Counter::GfpCfIterN, cf_iter);
		FrameTiming::Add(FrameTiming::Counter::GfpCfBlocksN, m_program.control_flow.size());
		FrameTiming::Add(FrameTiming::Counter::GfpCfCondN, cf_cond);
		FrameTiming::Add(FrameTiming::Counter::GfpCfEvalN, t_eval_inst_calls - cf_eval0);
		FrameTiming::Add(FrameTiming::Counter::GfpCfRflN, t_rfl_calls - cf_rfl0);
	}
	std::optional<FrameTiming::ScopedCounterNs> slots_timer; // S451
	if (m_program.stage != ShaderType::Compute) slots_timer.emplace(FrameTiming::Counter::GfpWalkSlotsNs);
	flat.assign(m_program.srt_reads.size(), 0u);
	active.assign(m_program.descriptor_sources.size(), 1u);
	for (const auto& block: m_program.control_flow)
		for (const auto source: block.sources) active.at(source) = 0u;
	for (uint32_t index = 0; index < visited.size(); ++index) {
#ifdef KYTY_SRTBATCH_MUTANT_ALL_BLOCKS
		if (!visited[index] && batch_mode == SrtSlotBatch::Mode::On && index < m_program.srt_block_runs.size()) { // the batch also fills the slots of unvisited blocks
			const auto& hidden = m_program.control_flow[index];
			for (const auto& run: m_program.srt_block_runs[index])
				if (run.entry != UINT32_MAX && !refresh_run(hidden, run)) return false;
		}
#endif
		if (!visited[index]) continue;
		const auto& block = m_program.control_flow[index];
		for (const auto source: block.sources) active[source] = 1u;
		if (!refresh_block(index, block)) return false;
	}
	return true;
}

bool ValidateRuntimeValue(const ResourcePlan& program, Value value, RuntimeValueType type) {
	return RuntimeValidator(program, type).Run(value);
}

bool ValidateWaveUniformValue(const ResourcePlan& program, Value value) {
	return RuntimeValidator(program, RuntimeValueType::Any, true).Run(value);
}


} // namespace Libs::Graphics::ShaderRecompiler::IR
