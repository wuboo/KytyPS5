void TestRayTracingDispatchDetection() {
  namespace D = ShaderRecompiler::Decoder;
  const auto check_decoded = [](std::span<const uint32_t> code, bool expected,
                                const char *message) {
    D::Program decoded;
    D::DecodeProgram(code, decoded);
    Check(decoded.has_bvh == expected, message);
    if (expected) {
      // BVH intersections are translated (every ray misses), so the decoder keeps going
      // to the end of the program instead of stopping at the first BVH instruction.
      Check(decoded.instructions.back().opcode == D::Opcode::S_ENDPGM,
            "decoder stopped at the first BVH instruction");
      const bool well_formed =
          std::none_of(decoded.instructions.begin(), decoded.instructions.end(),
                       [](const D::Instruction &inst) {
                         return inst.opcode == D::Opcode::UNSUPPORTED;
                       });
      if (well_formed) {
        const auto result = ShaderRecompiler::TranslateProgram(
            code, MakeCompileOptions(ShaderType::Compute));
        Check(!result.skip_dispatch && !result.program.blocks.empty(),
              "BVH compute shader was skipped instead of translated");
      }
    }
  };
  const auto check = [&](std::initializer_list<uint32_t> code, bool expected,
                         const char *message) {
    check_decoded(std::span<const uint32_t>{code.begin(), code.size()}, expected, message);
  };

  check({0xf1989f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "captured contiguous BVH instruction did not skip dispatch");
  check({0xf1989f07u, 0x00071001u, 0x3d3e3c15u, 0x38095851u, 0x00003a39u,
         EncodeSopp(0x01)}, true,
        "captured NSA BVH instruction did not skip dispatch");
  check({0xf19c9f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "RDNA2 BVH64 instruction did not skip dispatch");
  check({0xbe8e210eu, 0xf1989f01u, 0x00081014u, EncodeSopp(0x01)}, true,
        "unsupported SWAPPC before BVH prevented dispatch detection");
  // An unknown instruction family after a BVH instruction is decoded like in any other
  // shader (the decoder no longer stops at the BVH instruction), so it is not a case here.

  for (uint32_t opcode : {0xe6u, 0xe7u}) {
    for (uint32_t nsa = 0; nsa < 4; ++nsa) {
      for (uint32_t slc = 0; slc < 2; ++slc) {
        std::array<uint32_t, 6> code{
            EncodeMimg0(opcode, 0xf, true) | (nsa << 1u) | (slc << 25u),
            EncodeMimg1(0, 0, 0, 0)};
        code[2 + nsa] = EncodeSopp(0x01);
        check_decoded(std::span(code).first(3 + nsa), true,
                      "BVH cache flags or NSA length prevented dispatch detection");
      }
    }
  }
  for (uint32_t opcode : {0x66u, 0x67u, 0xe4u, 0xe5u, 0xe8u, 0xe9u}) {
    check({EncodeMimg0(opcode, 0xf), EncodeMimg1(0, 0, 0, 0), EncodeSopp(0x01)},
          false, "non-BVH image opcode was incorrectly skipped");
  }

  check({EncodeSMovB32(0, 255), 0xf1989f01u, EncodeSopp(0x01)}, false,
        "scalar literal was mistaken for a BVH instruction");
  check({EncodeMimg0(0x20, 0xf) | 6u, EncodeMimg1(0, 0, 0, 0),
         0xf1989f01u, 0xf19c9f01u, 0u, EncodeSopp(0x01)}, false,
        "NSA address words were mistaken for BVH instructions");
  check({EncodeMimg0(0x20, 0xf), 0xf1989f01u, EncodeSopp(0x01)}, false,
        "MIMG operand word was mistaken for a BVH instruction");
  check({EncodeSopp(0x01), 0xf1989f01u, 0x00081014u}, false,
        "data after the shader end was mistaken for a BVH instruction");
  check({EncodeMimg0(0x20, 0xf), EncodeMimg1(0, 0, 0, 0), EncodeSopp(0x01)},
        false, "ordinary image shader was incorrectly skipped");

  // An end instruction on one branch does not end another branch's code.
  check({EncodeSopp(0x04, 1), EncodeSopp(0x01), 0xf1989f01u, 0x00081014u,
         EncodeSopp(0x01)}, true,
        "BVH at a branch target after an early end was not detected");

  const std::array bvh_code{0xf1989f01u, 0x00081014u, 0xf19c9f01u, 0x00081014u,
                            EncodeSopp(0x01)};
  const std::array ordinary_code{EncodeSMovB32(0, 255), 0xf1989f01u, EncodeSopp(0x01)};
  D::Program reused;
  D::DecodeProgram(bvh_code, reused);
  Check(reused.has_bvh && reused.instructions.size() == 3 &&
            reused.instructions.back().opcode == D::Opcode::S_ENDPGM,
        "decoder did not decode both BVH instructions and the end");
  D::DecodeProgram(ordinary_code, reused);
  Check(!reused.has_bvh, "decoder retained stale BVH metadata");
  for (const auto stage : {ShaderType::Compute, ShaderType::Vertex, ShaderType::Pixel}) {
    const auto options = MakeCompileOptions(stage);
    Check(!ShaderRecompiler::TranslateProgram(ordinary_code, options).skip_dispatch,
          "BVH-shaped literal marked an ordinary shader for skipping");
    const auto compiled = RecompileForTest(ordinary_code, options);
    Check(!compiled.spirv.empty(), "ordinary shader did not compile");
    CheckSpirvBinaryValidates(compiled.spirv);
  }
}
