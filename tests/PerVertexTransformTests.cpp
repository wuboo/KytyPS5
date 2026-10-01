#include "graphics/host_gpu/renderer/perVertexTransform.h"

#include <cstdio>
#include <cstdlib>

int main() {
    // Exercise line anchors, regex iteration over adjacent declarations, and
    // searches away from the start, including an unterminated final line.
    for (const std::string newline : {"\n", "\r\n"}) {
        const std::string vs = "OpDecorate %param0 Location 0" + newline +
            "OpDecorate %param3 Location 3" + newline +
            "%param0 = OpVariable %ptr Output" + newline +
            "%param3 = OpVariable %ptr Output" + newline +
            "%gl_ClipDistance = OpVariable %clip_ptr Output";
        const std::string ps = "OpDecorate %in0 PerVertexKHR" + newline +
            "OpDecorate %in3 PerVertexKHR" + newline +
            "OpDecorate %in0 Location 0" + newline +
            "OpDecorate %in3 Location 3";
        Libs::Graphics::PerVertexLayout layout;
        std::map<uint32_t, std::string> params;
        const std::map<uint32_t, std::string> expected_params = {{0, "%param0"}, {3, "%param3"}};
        const std::map<uint32_t, uint32_t> expected_slots = {{0, 1}, {3, 2}};
        if (!Libs::Graphics::DerivePerVertexLayout(vs, ps, layout, params) ||
            params != expected_params || layout.location_to_slot != expected_slots ||
            layout.num_params != 2 || !layout.has_clip || layout.clip_slot != 3 ||
            layout.record_stride_vec4 != 4) {
            std::fprintf(stderr, "FAIL: multiline PerVertex layout derivation\n");
            return EXIT_FAILURE;
        }
        const std::string occupied_primitive_location =
            vs + newline + "OpDecorate %param31 Location 31" + newline +
            "%param31 = OpVariable %ptr Output";
        if (Libs::Graphics::DerivePerVertexLayout(occupied_primitive_location, ps, layout, params)) {
            std::fprintf(stderr, "FAIL: reserved primitive location 31 accepted\n");
            return EXIT_FAILURE;
        }

        const std::string missing_location = ps + newline + "OpDecorate %missing PerVertexKHR";
        if (Libs::Graphics::DerivePerVertexLayout(vs, missing_location, layout, params)) {
            std::fprintf(stderr, "FAIL: missing location on final line was accepted\n");
            return EXIT_FAILURE;
        }
    }
    std::puts("PASS: multiline PerVertex parsing (LF/CRLF, adjacent and final lines)");

    std::string wide_vs;
    std::string wide_ps;
    for (int i = 0; i < 13; ++i) {
        wide_vs += "OpDecorate %p" + std::to_string(i) + " Location " + std::to_string(i) + "\n";
        wide_vs += "%p" + std::to_string(i) + " = OpVariable %ptr Output\n";
        wide_ps += "OpDecorate %in" + std::to_string(i) + " PerVertexKHR\n";
        wide_ps += "OpDecorate %in" + std::to_string(i) + " Location " + std::to_string(i) + "\n";
    }
    Libs::Graphics::PerVertexLayout wide_layout;
    std::map<uint32_t, std::string>  wide_params;
    if (!Libs::Graphics::DerivePerVertexLayout(wide_vs, wide_ps, wide_layout, wide_params) ||
        wide_layout.num_params != 13 || wide_layout.location_to_slot.size() != 13 ||
        wide_layout.location_to_slot.at(0) != 1 || wide_layout.location_to_slot.at(12) != 13 ||
        wide_layout.record_stride_vec4 != 14) {
        std::fprintf(stderr, "FAIL: layout with 13 parameters was rejected or mis-slotted\n");
        return EXIT_FAILURE;
    }
    std::puts("PASS: 13 vertex parameters (above the old 12-attribute unpack cap)");
    return EXIT_SUCCESS;
}
