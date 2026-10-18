#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

// S149 (KYTY_PV_VSSTORE): the guest VS stores its PerVertex parameters to cap[record * stride +
// slot]; the guest PS reads the three corners of its primitive back. The record arithmetic mirrors
// graphics/host_gpu/renderer/perVertexVsStore.h (PvVsStoreVertexRecord / PvVsStoreCornerRecord).

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

namespace {

bool IsVertex(const EmitterState& state) {
	return state.pv_store->role == PvStoreInfo::Role::Vertex;
}

uint32_t U(EmitterState& state, spv::Op op, uint32_t lhs, uint32_t rhs) {
	return Binary(state, op, TypeU32(state), lhs, rhs);
}

uint32_t C(EmitterState& state, uint32_t value) {
	return ConstantU32(state, value);
}

bool Guarded(const EmitterState& state) {
#if defined(KYTY_PVVS_MUTANT_GUARD_ALWAYS)
	return state.pv_store != nullptr;
#else
	return state.pv_store != nullptr && state.pv_store->guard;
#endif
}

uint32_t Push(EmitterState& state, uint32_t dword) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
	                          state.push_constant_variable, C(state, 1), C(state, dword));
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

uint32_t LoadSigned(EmitterState& state, uint32_t variable) {
	const auto value = state.builder.AllocateId();
	const auto bits  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeI32(state), value, variable);
	state.builder.AddFunction(spv::OpBitcast, TypeU32(state), bits, value);
	return bits;
}

// base + (instance - first_instance) * span + (vertex - first)
uint32_t Record(EmitterState& state, uint32_t instance, uint32_t vertex) {
	const auto relative_instance = U(state, spv::OpISub, instance, Push(state, kPvPushFirstInstance));
	const auto instance_part     = U(state, spv::OpIMul, relative_instance, Push(state, kPvPushSpan));
	const auto relative_vertex   = U(state, spv::OpISub, vertex, Push(state, kPvPushFirst));
	return U(state, spv::OpIAdd, U(state, spv::OpIAdd, Push(state, kPvPushBase), instance_part),
	         relative_vertex);
}

// record - base >= records (unsigned: a vertex below `first` wraps); mirrors PvVsStoreRecordOutOfRange.
uint32_t RecordOutOfRange(EmitterState& state, uint32_t record) {
	const auto relative = U(state, spv::OpISub, record, Push(state, kPvPushBase));
	return Binary(state, spv::OpUGreaterThanEqual, TypeBool(state), relative, Push(state, kPvPushRecords));
}

// S409: the PS location mask as a specialization constant, and OpSpecConstantOp arithmetic on it. The default literal is patched
// per (VS, PS) pair (PvCompactPatchMask), the module is otherwise the same for every PS.
uint32_t SpecOp(EmitterState& state, uint32_t type, spv::Op op, uint32_t lhs, uint32_t rhs) {
	return state.builder.Constant(spv::OpSpecConstantOp, type, static_cast<uint32_t>(op), lhs, rhs);
}

uint32_t CompactMask(EmitterState& state) {
	if (state.pv_compact_mask == 0) {
		state.pv_compact_mask = state.builder.Constant(spv::OpSpecConstant, TypeU32(state), 0x7fffffffu);
		state.builder.AddAnnotation(spv::OpDecorate, state.pv_compact_mask, spv::DecorationSpecId, 0);
		state.pv_compact_rank.push_back(C(state, 0));
		if (state.pv_store->compact_mask_id != nullptr) *state.pv_store->compact_mask_id = state.pv_compact_mask;
	}
	return state.pv_compact_mask;
}

// Bit `location` of the mask as 0/1 (uint) and the number of mask bits below `location` (the compact slot), both spec constants.
uint32_t CompactBit(EmitterState& state, uint32_t location) {
	const auto shifted = SpecOp(state, TypeU32(state), spv::OpShiftRightLogical, CompactMask(state), C(state, location));
	return SpecOp(state, TypeU32(state), spv::OpBitwiseAnd, shifted, C(state, 1));
}

uint32_t CompactSlotId(EmitterState& state, uint32_t location) {
	CompactMask(state);
#if defined(KYTY_PVVS_MUTANT_COMPACT_SLOT_LOCATION)
	return C(state, location);
#endif
	while (state.pv_compact_rank.size() <= location) {
		const auto below = static_cast<uint32_t>(state.pv_compact_rank.size()) - 1;
		state.pv_compact_rank.push_back(SpecOp(state, TypeU32(state), spv::OpIAdd, state.pv_compact_rank.back(), CompactBit(state, below)));
	}
	return state.pv_compact_rank[location];
}

uint32_t CompactStored(EmitterState& state, uint32_t location) {
	return state.builder.Constant(spv::OpSpecConstantOp, TypeBool(state), static_cast<uint32_t>(spv::OpINotEqual),
	                              CompactBit(state, location), C(state, 0));
}

uint32_t CapComponentPointer(EmitterState& state, uint32_t record, uint32_t slot_id,
                             uint32_t component) {
	// S150: stride 0 = the stride comes with the draw (the two variants are emitted independently).
#if defined(KYTY_PVVS_MUTANT_STRIDE_DWORD)
	constexpr uint32_t kStrideDword = kPvPushSpan;
#else
	constexpr uint32_t kStrideDword = kPvPushStride;
#endif
	const auto stride = state.pv_store->record_stride_vec4 != 0
	                        ? C(state, state.pv_store->record_stride_vec4)
	                        : Push(state, kStrideDword);
	const auto element = U(state, spv::OpIAdd, U(state, spv::OpIMul, record, stride), slot_id);
	const auto pointer = state.builder.AllocateId();
	if (component == UINT32_MAX) {
		state.builder.AddFunction(
		    spv::OpAccessChain,
		    TypePointer(state, spv::StorageClassStorageBuffer, TypeU32Vector(state, 4)), pointer,
		    state.pv_cap_buffer, C(state, 0), element);
	} else {
		state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassStorageBuffer, TypeU32(state)),
		                          pointer, state.pv_cap_buffer, C(state, 0), element,
		                          C(state, component));
	}
	return pointer;
}

uint32_t SlotOf(const EmitterState& state, uint32_t host_location) {
	if (state.pv_store->record_stride_vec4 == 0) {
		// Dynamic stride: slot == host location (location 31 carries the instance index).
		EXIT_IF(host_location >= kPvVsStoreInstanceLocation);
		if (state.pv_store->location_mask != nullptr) *state.pv_store->location_mask |= 1u << host_location;
		return host_location;
	}
	const auto slot = state.pv_store->parameter_slots.find(host_location);
	EXIT_IF(slot == state.pv_store->parameter_slots.end() ||
	        slot->second >= state.pv_store->record_stride_vec4);
	return slot->second;
}

// Index value of element `element` (16- or 32-bit buffer, chosen by the push flag).
uint32_t FetchIndex(EmitterState& state, uint32_t element) {
	const auto index32 = Binary(state, spv::OpINotEqual, TypeBool(state), Push(state, kPvPushIndex32),
	                            C(state, 0));
	const auto word = Select(state, TypeU32(state), index32, element,
	                         U(state, spv::OpShiftRightLogical, element, C(state, 1)));
	const auto length = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), length, state.pv_index_buffer, 0);
	const auto clamped = EmitGlsl<GLSLstd450UMin, IR::Type::U32>(
	    state, word, U(state, spv::OpISub, length, C(state, 1)));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassStorageBuffer, TypeU32(state)), pointer,
	                          state.pv_index_buffer, C(state, 0), clamped);
	const auto raw = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), raw, pointer);
	const auto half_shift =
	    U(state, spv::OpShiftLeftLogical, U(state, spv::OpBitwiseAnd, element, C(state, 1)), C(state, 4));
	const auto shifted = U(state, spv::OpShiftRightLogical, raw,
	                       Select(state, TypeU32(state), index32, C(state, 0), half_shift));
	return U(state, spv::OpBitwiseAnd, shifted,
	         Select(state, TypeU32(state), index32, C(state, 0xffffffffu), C(state, 0xffffu)));
}

} // namespace

void DefinePvStore(EmitterState& state) {
	const bool vertex = IsVertex(state);
	const auto uvec4  = TypeU32Vector(state, 4);
	const auto array  = state.builder.DecoratedType(
        spv::OpTypeRuntimeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 16}}}, uvec4);
	const auto block = state.builder.DecoratedType(spv::OpTypeStruct,
	                                               {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
	                                                {spv::OpDecorate, {spv::DecorationBlock}}},
	                                               array);
	state.pv_cap_buffer = state.builder.DefineGlobalVariable(
	    TypePointer(state, spv::StorageClassStorageBuffer, block), spv::StorageClassStorageBuffer);
	state.builder.AddName(state.pv_cap_buffer, "pv_cap");
	state.builder.AddAnnotation(spv::OpDecorate, state.pv_cap_buffer, spv::DecorationDescriptorSet, 1);
	state.builder.AddAnnotation(spv::OpDecorate, state.pv_cap_buffer, spv::DecorationBinding, 0);
	if (!vertex) state.builder.AddAnnotation(spv::OpDecorate, state.pv_cap_buffer, spv::DecorationNonWritable);
	if (state.pv_store->coherent) {
		state.builder.AddAnnotation(spv::OpDecorate, state.pv_cap_buffer, spv::DecorationCoherent);
	}
	if (!vertex) {
		const auto words = state.builder.DecoratedType(
		    spv::OpTypeRuntimeArray, {{spv::OpDecorate, {spv::DecorationArrayStride, 4}}},
		    TypeU32(state));
		const auto index_block = state.builder.DecoratedType(
		    spv::OpTypeStruct,
		    {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
		     {spv::OpDecorate, {spv::DecorationBlock}}},
		    words);
		state.pv_index_buffer = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassStorageBuffer, index_block),
		    spv::StorageClassStorageBuffer);
		state.builder.AddName(state.pv_index_buffer, "pv_indices");
		state.builder.AddAnnotation(spv::OpDecorate, state.pv_index_buffer,
		                            spv::DecorationDescriptorSet, 1);
		state.builder.AddAnnotation(spv::OpDecorate, state.pv_index_buffer, spv::DecorationBinding, 1);
		state.builder.AddAnnotation(spv::OpDecorate, state.pv_index_buffer, spv::DecorationNonWritable);
		// PrimitiveId in a fragment shader needs the Geometry capability.
		state.builder.RequireCapability(spv::CapabilityGeometry);
	}
	// The instance index travels VS -> PS as a flat varying (the PS has no instance builtin).
	state.pv_instance_variable = DefineInterfaceVariable(
	    state, TypeU32(state), vertex ? spv::StorageClassOutput : spv::StorageClassInput,
	    "pv_instance");
	state.builder.AddAnnotation(spv::OpDecorate, state.pv_instance_variable, spv::DecorationFlat);
	state.builder.AddAnnotation(spv::OpDecorate, state.pv_instance_variable, spv::DecorationLocation,
	                            kPvVsStoreInstanceLocation);
	if (!Guarded(state)) return;
	if (vertex) {
		const auto counter = state.builder.DecoratedType(spv::OpTypeStruct,
		                                                 {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
		                                                  {spv::OpDecorate, {spv::DecorationBlock}}},
		                                                 TypeU32(state));
		state.pv_guard_counter = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassStorageBuffer, counter), spv::StorageClassStorageBuffer);
		state.builder.AddName(state.pv_guard_counter, "pv_guard");
		state.builder.AddAnnotation(spv::OpDecorate, state.pv_guard_counter, spv::DecorationDescriptorSet, 1);
		state.builder.AddAnnotation(spv::OpDecorate, state.pv_guard_counter, spv::DecorationBinding, kPvGuardBinding);
	} else {
		state.pv_oob_flag = state.builder.DefineGlobalVariable(TypePointer(state, spv::StorageClassPrivate, TypeU32(state)),
		                                                       spv::StorageClassPrivate);
		state.builder.AddName(state.pv_oob_flag, "pv_oob");
	}
}

void InitializePvStore(EmitterState& state) {
	if (state.pv_store == nullptr) return;
	if (!IsVertex(state)) {
		if (Guarded(state)) state.builder.AddFunction(spv::OpStore, state.pv_oob_flag, C(state, 0));
		return;
	}
	const auto vertex   = LoadSigned(state, InputVariableForKind(state, IR::StageInputKind::VertexIndex));
	const auto instance = LoadSigned(state, InputVariableForKind(state, IR::StageInputKind::InstanceIndex));
	state.builder.AddFunction(spv::OpStore, state.pv_instance_variable, instance);
	state.pv_vertex_record = Record(state, instance, vertex);
	if (!Guarded(state)) return;
	// S388: a record outside the draw's range is counted (device-wide atomic, one dword) and never stored.
	const auto outside = RecordOutOfRange(state, state.pv_vertex_record);
	state.pv_guard_ok  = Unary(state, spv::OpLogicalNot, TypeBool(state), outside);
	EmitIfCondition(state, outside, [&] {
		const auto pointer = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAccessChain, TypePointer(state, spv::StorageClassStorageBuffer, TypeU32(state)),
		                          pointer, state.pv_guard_counter, C(state, 0));
		const auto old = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpAtomicIAdd, TypeU32(state), old, pointer, C(state, spv::ScopeDevice), C(state, 0),
		                          C(state, 1));
	});
}

void EmitPvStoreParameter(EmitterState& state, uint32_t host_location, uint32_t value) {
	EXIT_IF(state.pv_store == nullptr || !IsVertex(state) || state.pv_vertex_record == 0);
	const bool compact = state.pv_store->compact_spec;
	// S409: SlotOf still reports the location to location_mask; the slot itself is the rank of the location in the PS mask.
	const auto slot_id = [&] {
		const auto host_slot = SlotOf(state, host_location);
		return compact ? CompactSlotId(state, host_location) : C(state, host_slot);
	};
	const auto store = [&] {
		const auto bits = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpBitcast, TypeU32Vector(state, 4), bits, value);
		state.builder.AddFunction(
		    spv::OpStore,
		    CapComponentPointer(state, state.pv_vertex_record, slot_id(), UINT32_MAX),
		    bits);
		if (state.pv_store->coherent) {
			state.builder.AddFunction(spv::OpMemoryBarrier, C(state, spv::ScopeDevice),
			                          C(state, spv::MemorySemanticsAcquireReleaseMask |
			                                       spv::MemorySemanticsUniformMemoryMask));
		}
	};
	const auto guarded_store = [&] {
#if defined(KYTY_PVVS_MUTANT_GUARD_NO_SKIP)
		store();
#else
		if (Guarded(state)) {
			EmitIfCondition(state, state.pv_guard_ok, store);
		} else {
			store();
		}
#endif
	};
	if (compact) {
		EXIT_IF(host_location >= kPvVsStoreInstanceLocation);
#if defined(KYTY_PVVS_MUTANT_COMPACT_NO_BRANCH)
		guarded_store();
#else
		EmitIfCondition(state, CompactStored(state, host_location), guarded_store);
#endif
	} else {
		guarded_store();
	}
}

uint32_t EmitPvLoadCorner(EmitterState& state, uint32_t host_location, uint32_t corner,
                          uint32_t chan) {
	EXIT_IF(state.pv_store == nullptr || IsVertex(state) || corner >= kPvVsStoreCorners);
	if (state.pv_store->corner_mask != nullptr) *state.pv_store->corner_mask |= 1u << corner;
	const auto primitive = LoadSigned(state, InputVariableForKind(state, IR::StageInputKind::PrimitiveId));
	const auto instance  = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), instance, state.pv_instance_variable);
	// element = first_index + primitive * 3 + corner
	const auto element = U(state, spv::OpIAdd,
	                       U(state, spv::OpIAdd, Push(state, kPvPushFirstIndex),
	                         U(state, spv::OpIMul, primitive, C(state, kPvVsStoreCorners))),
	                       C(state, PvVsStoreCornerElement(corner)));
	// gl_VertexIndex of that corner in the VS: index + vertex_offset
	const auto vertex = U(state, spv::OpIAdd, FetchIndex(state, element), Push(state, kPvPushVertexOffset));
	auto record = Record(state, instance, vertex);
	if (Guarded(state)) {
		// S388: remember a read outside the draw's range (the PS paints magenta) and read the first record instead.
		const auto outside = RecordOutOfRange(state, record);
		const auto seen    = state.builder.AllocateId();
		state.builder.AddFunction(spv::OpLoad, TypeU32(state), seen, state.pv_oob_flag);
		const auto merged = U(state, spv::OpBitwiseOr, seen, Select(state, TypeU32(state), outside, C(state, 1), C(state, 0)));
		state.builder.AddFunction(spv::OpStore, state.pv_oob_flag, merged);
		record = Select(state, TypeU32(state), outside, Push(state, kPvPushBase), record);
	}
	const auto pointer = CapComponentPointer(state, record, C(state, SlotOf(state, host_location)), chan);
	const auto bits    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), bits, pointer);
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpBitcast, TypeF32(state), value, bits);
	return value;
}

// S388: the value of a color export, or magenta when this PS read a record outside the draw's range.
uint32_t PvGuardColor(EmitterState& state, uint32_t value) {
	if (!Guarded(state) || IsVertex(state)) return value;
#if defined(KYTY_PVVS_MUTANT_GUARD_NO_MAGENTA)
	return value;
#else
	const auto seen = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), seen, state.pv_oob_flag);
	const auto outside = Binary(state, spv::OpINotEqual, TypeBool(state), seen, C(state, 0));
	const auto mask    = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeBoolVector(state, 4), mask, outside, outside, outside, outside);
	const auto one     = ConstantF32Value(state, 1.0f);
	const auto zero    = ConstantF32Value(state, 0.0f);
	const auto magenta = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeConstruct, TypeF32Vector(state, 4), magenta, one, zero, one, one);
	return Select(state, TypeF32Vector(state, 4), mask, magenta, value);
#endif
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
