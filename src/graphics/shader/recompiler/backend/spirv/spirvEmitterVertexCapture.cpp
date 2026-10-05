#include "graphics/shader/recompiler/backend/spirv/spirvEmitterInternal.h"

namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter {

namespace {

// Dword indices inside the fused push block (UnpackPushConstants layout).
constexpr uint32_t kFusedIndexCount   = 1;
constexpr uint32_t kFusedFirstVertex  = 2;
constexpr uint32_t kFusedVertexOffset = 3;
constexpr uint32_t kFusedFirstInst    = 4;
constexpr uint32_t kFusedStride       = 5;
constexpr uint32_t kFusedNumRecords   = 6;
constexpr uint32_t kFusedFlags        = 7;
constexpr uint32_t kFusedAttrs        = 8;

bool Fused(const EmitterState& state) {
	return state.vertex_capture != nullptr && state.vertex_capture->fused;
}

uint32_t FusedPush(EmitterState& state, uint32_t dword) {
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypePushConstantElementPointer(state), pointer,
	                          state.push_constant_variable, ConstantU32(state, 1),
	                          ConstantU32(state, dword));
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

uint32_t U(EmitterState& state, spv::Op op, uint32_t lhs, uint32_t rhs) {
	return Binary(state, op, TypeU32(state), lhs, rhs);
}

uint32_t C(EmitterState& state, uint32_t value) {
	return ConstantU32(state, value);
}

uint32_t Cmp(EmitterState& state, spv::Op op, uint32_t lhs, uint32_t rhs) {
	return Binary(state, op, TypeBool(state), lhs, rhs);
}

uint32_t Min(EmitterState& state, uint32_t lhs, uint32_t rhs) {
	return EmitGlsl<GLSLstd450UMin, IR::Type::U32>(state, lhs, rhs);
}

// Bounds-clamped dword load from a fused raw buffer (0 = vertices, 2 = indices).
uint32_t LoadWord(EmitterState& state, uint32_t binding, uint32_t index) {
	const auto length = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), length,
	                          state.capture_buffers[binding], 0);
	const auto clamped = Min(state, index, U(state, spv::OpISub, length, C(state, 1)));
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.capture_buffers[binding], C(state, 0), clamped);
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

// Mirrors per_vertex_unpack.comp ReadBits(): up to 32 bits at an arbitrary bit position.
uint32_t ReadBits(EmitterState& state, uint32_t total_bit, uint32_t width) {
	const auto word  = U(state, spv::OpShiftRightLogical, total_bit, C(state, 5));
	const auto shift = U(state, spv::OpBitwiseAnd, total_bit, C(state, 31));
	const auto low   = U(state, spv::OpShiftRightLogical, LoadWord(state, 0, word), shift);
	const auto high_shift = U(state, spv::OpBitwiseAnd, U(state, spv::OpISub, C(state, 32), shift),
	                          C(state, 31));
	const auto high = U(state, spv::OpShiftLeftLogical,
	                    LoadWord(state, 0, U(state, spv::OpIAdd, word, C(state, 1))), high_shift);
	const auto crosses = Cmp(state, spv::OpUGreaterThan, U(state, spv::OpIAdd, shift, width),
	                         C(state, 32));
	const auto value   = Select(state, TypeU32(state), crosses,
	                            U(state, spv::OpBitwiseOr, low, high), low);
	const auto mask    = U(state, spv::OpISub,
	                       U(state, spv::OpShiftLeftLogical, C(state, 1),
	                         U(state, spv::OpBitwiseAnd, width, C(state, 31))),
	                       C(state, 1));
	return Select(state, TypeU32(state), Cmp(state, spv::OpIEqual, width, C(state, 32)), value,
	              U(state, spv::OpBitwiseAnd, value, mask));
}

// Mirrors per_vertex_unpack.comp VertexIndex(): the guest vertex id of draw-local vertex `local`,
// 0xffffffff when out of range.
uint32_t FusedVertexIndex(EmitterState& state, uint32_t local) {
	const auto flags       = FusedPush(state, kFusedFlags);
	const auto num_records = FusedPush(state, kFusedNumRecords);
	const auto indexed     = Cmp(state, spv::OpINotEqual,
	                             U(state, spv::OpBitwiseAnd,
	                               U(state, spv::OpShiftRightLogical, flags, C(state, 16)), C(state, 1)),
	                             C(state, 0));
	const auto type        = U(state, spv::OpBitwiseAnd,
	                           U(state, spv::OpShiftRightLogical, flags, C(state, 17)), C(state, 3));
	const auto byte_offset = U(state, spv::OpBitwiseAnd, flags, C(state, 0xffff));
	const auto is16        = Cmp(state, spv::OpIEqual, type, C(state, 1));
	const auto is32        = Cmp(state, spv::OpIEqual, type, C(state, 2));
	const auto size_shift  = Select(state, TypeU32(state), is16, C(state, 1),
	                                Select(state, TypeU32(state), is32, C(state, 2), C(state, 0)));
	const auto address     = U(state, spv::OpIAdd, byte_offset,
	                           U(state, spv::OpShiftLeftLogical, local, size_shift));
	const auto word        = LoadWord(state, 2, U(state, spv::OpShiftRightLogical, address, C(state, 2)));
	const auto bit_shift   = Select(
        state, TypeU32(state), is16,
        U(state, spv::OpShiftLeftLogical, U(state, spv::OpBitwiseAnd, address, C(state, 2)),
          C(state, 3)),
        Select(state, TypeU32(state), is32, C(state, 0),
               U(state, spv::OpShiftLeftLogical, U(state, spv::OpBitwiseAnd, address, C(state, 3)),
                 C(state, 3))));
	const auto mask        = Select(state, TypeU32(state), is16, C(state, 0xffff),
	                                Select(state, TypeU32(state), is32, C(state, 0xffffffffu),
	                                       C(state, 0xff)));
	const auto raw         = U(state, spv::OpBitwiseAnd,
	                           U(state, spv::OpShiftRightLogical, word, bit_shift), mask);
	const auto adjusted    = U(state, spv::OpIAdd, raw, FusedPush(state, kFusedVertexOffset));
	const auto adjusted_ok = Cmp(state, spv::OpULessThan, adjusted, num_records);
	// adjusted >= 0 as signed: the sign bit is clear.
	const auto non_negative = Cmp(state, spv::OpIEqual,
	                              U(state, spv::OpShiftRightLogical, adjusted, C(state, 31)),
	                              C(state, 0));
	const auto indexed_ok   = Binary(state, spv::OpLogicalAnd, TypeBool(state), non_negative,
	                                 adjusted_ok);
	const auto indexed_id   = Select(state, TypeU32(state), indexed_ok, adjusted, C(state, 0xffffffffu));
	const auto linear       = U(state, spv::OpIAdd, FusedPush(state, kFusedFirstVertex), local);
	const auto linear_id    = Select(state, TypeU32(state),
	                                 Cmp(state, spv::OpULessThan, linear, num_records), linear,
	                                 C(state, 0xffffffffu));
	return Select(state, TypeU32(state), indexed, indexed_id, linear_id);
}

// Mirrors per_vertex_unpack.comp DecodeAttribute() for a single component; returns raw bits.
uint32_t FusedAttribute(EmitterState& state, uint32_t location, uint32_t component) {
	const auto half       = state.lane_half;
	const auto meta       = FusedPush(state, kFusedAttrs + location * 2u);
	const auto bit_counts = FusedPush(state, kFusedAttrs + location * 2u + 1u);
	const auto offset     = U(state, spv::OpBitwiseAnd, meta, C(state, 0xffff));
	const auto count      = U(state, spv::OpBitwiseAnd,
	                          U(state, spv::OpShiftRightLogical, meta, C(state, 17)), C(state, 7));
	const auto kind       = U(state, spv::OpBitwiseAnd,
	                          U(state, spv::OpShiftRightLogical, meta, C(state, 20)), C(state, 7));
	const auto byte_base  = U(state, spv::OpIAdd,
	                          U(state, spv::OpIMul, state.capture_vertex[half],
	                            FusedPush(state, kFusedStride)),
	                          offset);
	const auto field      = [&](uint32_t index) {
        return U(state, spv::OpBitwiseAnd,
                 U(state, spv::OpShiftRightLogical, bit_counts, C(state, index * 8u)), C(state, 0xff));
	};
	uint32_t bit_offset = C(state, 0);
	for (uint32_t i = 0; i < component; i++) bit_offset = U(state, spv::OpIAdd, bit_offset, field(i));
	const auto bits   = field(component);
	const auto is     = [&](uint32_t value) { return Cmp(state, spv::OpIEqual, kind, C(state, value)); };
	const auto sixteen = Binary(state, spv::OpLogicalOr, TypeBool(state), is(2), is(3));
	const auto width  = Select(state, TypeU32(state), is(1), C(state, 32),
	                           Select(state, TypeU32(state), sixteen, C(state, 16), bits));
	const auto total  = U(state, spv::OpIAdd, U(state, spv::OpShiftLeftLogical, byte_base, C(state, 3)),
	                      bit_offset);
	const auto raw    = ReadBits(state, total, width);

	// Format kind 2: signed 16-bit normalized.
	const auto signed16 = Unary(state, spv::OpBitcast, TypeI32(state),
	                            U(state, spv::OpShiftLeftLogical, raw, C(state, 16)));
	const auto sext     = Binary(state, spv::OpShiftRightArithmetic, TypeI32(state), signed16,
	                             C(state, 16));
	const auto snorm    = EmitGlsl<GLSLstd450FMax, IR::Type::F32>(
        state,
        Binary(state, spv::OpFDiv, TypeF32(state),
               Unary(state, spv::OpConvertSToF, TypeF32(state), sext),
               ConstantF32Value(state, 32767.0f)),
        ConstantF32Value(state, -1.0f));
	// Format kind 3: half.
	const auto half2 = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpExtInst, TypeF32Vector(state, 2), half2, GlslStd450(state),
	                          GLSLstd450UnpackHalf2x16, raw);
	const auto half_value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpCompositeExtract, TypeF32(state), half_value, half2, 0);
	// Format kinds 4/5: unsigned / sign-extended integers.
	const auto sign = U(state, spv::OpShiftLeftLogical, C(state, 1),
	                    U(state, spv::OpBitwiseAnd, U(state, spv::OpISub, bits, C(state, 1)),
	                      C(state, 31)));
	const auto sint = U(state, spv::OpISub, U(state, spv::OpBitwiseXor, raw, sign), sign);
	// Format kind 0: unsigned normalized.
	const auto max_value = U(state, spv::OpISub,
	                         U(state, spv::OpShiftLeftLogical, C(state, 1),
	                           U(state, spv::OpBitwiseAnd, bits, C(state, 31))),
	                         C(state, 1));
	const auto unorm     = Binary(state, spv::OpFDiv, TypeF32(state),
	                              Unary(state, spv::OpConvertUToF, TypeF32(state), raw),
	                              Unary(state, spv::OpConvertUToF, TypeF32(state), max_value));
	const auto as_bits   = [&](uint32_t value) { return Unary(state, spv::OpBitcast, TypeU32(state), value); };
	auto       decoded   = as_bits(unorm);
	decoded = Select(state, TypeU32(state), is(5), sint, decoded);
	decoded = Select(state, TypeU32(state), is(4), raw, decoded);
	decoded = Select(state, TypeU32(state), is(3), as_bits(half_value), decoded);
	decoded = Select(state, TypeU32(state), is(2), as_bits(snorm), decoded);
	decoded = Select(state, TypeU32(state), is(1), raw, decoded);

	const auto fallback = component == 3
	                          ? Select(state, TypeU32(state),
	                                   Binary(state, spv::OpLogicalOr, TypeBool(state), is(4), is(5)),
	                                   C(state, 1), C(state, 0x3f800000u))
	                          : C(state, 0);
	const auto present  = Binary(state, spv::OpLogicalAnd, TypeBool(state),
	                             state.capture_vertex_ok[half],
	                             Cmp(state, spv::OpUGreaterThan, count, C(state, component)));
	return Select(state, TypeU32(state), present, decoded, fallback);
}

} // namespace

void DefineVertexCapture(EmitterState& state) {
	const auto& capture = *state.vertex_capture;
	EXIT_IF(capture.record_stride_vec4 == 0 ||
	        capture.num_attributes > ShaderVertexInputInfo::RES_MAX);
	EXIT_IF(capture.clip_slot != UINT32_MAX && capture.clip_slot >= capture.record_stride_vec4);
	for (const auto& [location, slot]: capture.parameter_slots) {
		EXIT_IF(slot == 0 || slot >= capture.record_stride_vec4 || slot == capture.clip_slot);
	}
	for (const auto& input: state.program.info.inputs) {
		EXIT_IF(input.kind != IR::StageInputKind::VertexIndex &&
		        input.kind != IR::StageInputKind::InstanceIndex &&
		        input.kind != IR::StageInputKind::Parameter);
		EXIT_IF(input.kind == IR::StageInputKind::Parameter &&
		        input.location >= capture.num_attributes);
	}
	for (uint32_t binding = 0; binding < 3; binding++) {
		const auto components = Fused(state) && binding != 1 ? 1u : binding == 2 ? 2u : 4u;
		const auto element =
		    binding == 1 ? TypeF32Vector(state, 4) : TypeU32Vector(state, components);
		const auto array = state.builder.DecoratedType(
		    spv::OpTypeRuntimeArray,
		    {{spv::OpDecorate, {spv::DecorationArrayStride, components * 4}}}, element);
		const auto block =
		    state.builder.DecoratedType(spv::OpTypeStruct,
		                                {{spv::OpMemberDecorate, {0, spv::DecorationOffset, 0}},
		                                 {spv::OpDecorate, {spv::DecorationBlock}}},
		                                array);
		const auto variable = state.builder.DefineGlobalVariable(
		    TypePointer(state, spv::StorageClassStorageBuffer, block),
		    spv::StorageClassStorageBuffer);
		state.capture_buffers[binding] = variable;
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationDescriptorSet, 1);
		state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationBinding, binding);
		if (binding != 1)
			state.builder.AddAnnotation(spv::OpDecorate, variable, spv::DecorationNonWritable);
	}
}

void InitializeVertexCapture(EmitterState& state) {
	if (state.vertex_capture == nullptr) return;
	const auto count = Fused(state) ? FusedPush(state, 0) : state.builder.AllocateId();
	if (!Fused(state)) {
		state.builder.AddFunction(spv::OpArrayLength, TypeU32(state), count,
		                          state.capture_buffers[2], 0);
	}
	const auto group = EmitInputComponentU32(state, IR::StageInputKind::WorkgroupId, 0);
	const auto base  = EmitBinaryU32(state, spv::OpIMul, group, ConstantU32(state, 64));
	const auto low   = EmitAddU32(state, base, EmitLocalInvocationIndex(state));
	for (uint32_t half = 0; half < state.lane_count; half++) {
		state.capture_index[half] =
		    half == 0 ? low : EmitAddU32(state, low, ConstantU32(state, 32));
		state.capture_valid[half] =
		    Binary(state, spv::OpULessThan, TypeBool(state), state.capture_index[half], count);
		if (!Fused(state)) continue;
		// Same invocation -> {vertex, instance} split as per_vertex_unpack.comp.
		const auto index_count = EmitGlsl<GLSLstd450UMax, IR::Type::U32>(
		    state, FusedPush(state, kFusedIndexCount), C(state, 1));
		const auto invocation = state.capture_index[half];
		const auto vertex_id =
		    FusedVertexIndex(state, U(state, spv::OpUMod, invocation, index_count));
		state.capture_vertex_ok[half] = Cmp(state, spv::OpINotEqual, vertex_id, C(state, 0xffffffffu));
		state.capture_vertex[half]    = Select(state, TypeU32(state), state.capture_vertex_ok[half],
		                                       vertex_id, C(state, 0));
		state.capture_instance[half]  = U(state, spv::OpIAdd, FusedPush(state, kFusedFirstInst),
		                                  U(state, spv::OpUDiv, invocation, index_count));
	}
}

uint32_t MaskCaptureExecution(EmitterState& state, uint32_t predicate, uint32_t half) {
	return state.vertex_capture == nullptr ? predicate
	                                       : Binary(state, spv::OpLogicalAnd, TypeBool(state),
	                                                predicate, state.capture_valid[half]);
}

uint32_t CaptureInput(EmitterState& state, bool attribute, uint32_t location, uint32_t component) {
	if (Fused(state)) {
		if (attribute) return FusedAttribute(state, location, component);
		return component == 0 ? state.capture_vertex[state.lane_half]
		                      : state.capture_instance[state.lane_half];
	}
	// Inactive tail lanes still participate in subgroup operations, but never read past the draw.
	const auto index = Select(state, TypeU32(state), state.capture_valid[state.lane_half],
	                          state.capture_index[state.lane_half], ConstantU32(state, 0));
	const auto record =
	    attribute
	        ? EmitAddU32(state,
	                     EmitBinaryU32(state, spv::OpIMul, index,
	                                   ConstantU32(state, state.vertex_capture->num_attributes)),
	                     ConstantU32(state, location))
	        : index;
	const auto pointer = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpAccessChain, TypeStorageBufferElementPointer(state), pointer,
	                          state.capture_buffers[attribute ? 0 : 2], ConstantU32(state, 0),
	                          record, ConstantU32(state, component));
	const auto value = state.builder.AllocateId();
	state.builder.AddFunction(spv::OpLoad, TypeU32(state), value, pointer);
	return value;
}

uint32_t CaptureOutputPointer(EmitterState& state, uint32_t slot, uint32_t component) {
	const auto index =
	    EmitAddU32(state,
	               EmitBinaryU32(state, spv::OpIMul, state.capture_index[state.lane_half],
	                             ConstantU32(state, state.vertex_capture->record_stride_vec4)),
	               ConstantU32(state, slot));
	const auto pointer = state.builder.AllocateId();
	if (component == UINT32_MAX) {
		state.builder.AddFunction(
		    spv::OpAccessChain,
		    TypePointer(state, spv::StorageClassStorageBuffer, TypeF32Vector(state, 4)), pointer,
		    state.capture_buffers[1], ConstantU32(state, 0), index);
	} else {
		state.builder.AddFunction(
		    spv::OpAccessChain, TypePointer(state, spv::StorageClassStorageBuffer, TypeF32(state)),
		    pointer, state.capture_buffers[1], ConstantU32(state, 0), index,
		    ConstantU32(state, component));
	}
	return pointer;
}

} // namespace Libs::Graphics::ShaderRecompiler::Spirv::Emitter
