#include "graphics/host_gpu/renderer/perVertexTransform.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

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
    {
        using Libs::Graphics::ComputePerVertexRecordRange;
        using Libs::Graphics::PerVertexRecordRange;
        PerVertexRecordRange r;
        const uint16_t idx16[] = {7, 3, 9, 3, 12, 5};
        const uint8_t  idx8[]  = {200, 4, 10};
        const uint32_t idx32[] = {1u, 0x7fffffffu, 6u};
        bool           ok      = true;
        // Non-indexed: first_vertex + local, clipped to num_records.
        ok = ok && ComputePerVertexRecordRange(false, nullptr, 0, 0, 30, 10, 0, 100, r) &&
             r.first == 10 && r.end == 40;
        ok = ok && ComputePerVertexRecordRange(false, nullptr, 0, 0, 30, 90, 0, 100, r) &&
             r.first == 90 && r.end == 100;
        ok = ok && ComputePerVertexRecordRange(false, nullptr, 0, 0, 30, 100, 0, 100, r) &&
             r.first == r.end;
        // Indexed 16-bit, with vertex_offset; indices outside [0, num_records) are ignored.
        ok = ok && ComputePerVertexRecordRange(true, idx16, sizeof(idx16), 2, 6, 0, 0, 100, r) &&
             r.first == 3 && r.end == 13;
        ok = ok && ComputePerVertexRecordRange(true, idx16, sizeof(idx16), 2, 6, 0, -4, 100, r) &&
             r.first == 1 && r.end == 9;
        ok = ok && ComputePerVertexRecordRange(true, idx16, sizeof(idx16), 2, 6, 0, 0, 10, r) &&
             r.first == 3 && r.end == 10;
        ok = ok && ComputePerVertexRecordRange(true, idx16, sizeof(idx16), 2, 6, 0, 1000, 100, r) &&
             r.first == r.end;
        // Only the first index_count indices count.
        ok = ok && ComputePerVertexRecordRange(true, idx16, sizeof(idx16), 2, 2, 0, 0, 100, r) &&
             r.first == 3 && r.end == 8;
        // 8-bit and 32-bit indices.
        ok = ok && ComputePerVertexRecordRange(true, idx8, sizeof(idx8), 1, 3, 0, 0, 256, r) &&
             r.first == 4 && r.end == 201;
        ok = ok && ComputePerVertexRecordRange(true, idx32, sizeof(idx32), 4, 3, 0, 0, 100, r) &&
             r.first == 1 && r.end == 7;
        // 32-bit wrap of first_vertex + local in the shader: not derivable, must use the whole buffer.
        ok = ok && !ComputePerVertexRecordRange(false, nullptr, 0, 0, 30, 0xfffffff0u, 0, 100, r);
        ok = ok && ComputePerVertexRecordRange(false, nullptr, 0, 0, 16, 0xfffffff0u, 0, 100, r) &&
             r.first == r.end;
        // Not derivable with certainty: short index data, bad element size, 32-bit wrap.
        ok = ok && !ComputePerVertexRecordRange(true, idx16, sizeof(idx16), 2, 7, 0, 0, 100, r);
        ok = ok && !ComputePerVertexRecordRange(true, idx16, sizeof(idx16), 3, 2, 0, 0, 100, r);
        ok = ok && !ComputePerVertexRecordRange(true, nullptr, 0, 2, 0, 0, 0, 100, r);
        ok = ok && !ComputePerVertexRecordRange(true, idx32, sizeof(idx32), 4, 3, 0, 0x7fffffff, 100, r);
        // Differential check: every record the shader's VertexIndex() can return must lie in the range.
        uint32_t seed = 12345u;
        auto     rnd  = [&seed]() { seed = seed * 1664525u + 1013904223u; return seed >> 8; };
        for (int iter = 0; ok && iter < 20000; ++iter) {
            const uint32_t num_records = 1 + rnd() % 64;
            const uint32_t count       = 1 + rnd() % 12;
            const int32_t  voff        = static_cast<int32_t>(rnd() % 160) - 80;
            uint32_t       first       = rnd() % 80;
            if (iter % 7 == 0) first = 0xffffffffu - rnd() % 16; // near the 32-bit wrap
            std::vector<uint32_t> idx(count);
            for (auto& v : idx) v = (iter % 5 == 0) ? 0xffffffffu - rnd() % 8 : rnd() % 80;
            const bool indexed = (iter & 1) != 0;
            PerVertexRecordRange rr;
            if (!ComputePerVertexRecordRange(indexed, idx.data(), count * 4, 4, count, first, voff,
                                             num_records, rr)) {
                continue; // caller hashes the whole buffer
            }
            for (uint32_t local = 0; local < count; ++local) {
                uint64_t rec = UINT64_MAX; // none
                if (!indexed) {
                    const uint32_t index = first + local; // 32-bit wrap, as in the shader
                    if (index < num_records) rec = index;
                } else {
                    const int32_t adjusted = static_cast<int32_t>(
                        static_cast<uint32_t>(static_cast<int32_t>(idx[local])) + static_cast<uint32_t>(voff));
                    if (adjusted >= 0 && static_cast<uint32_t>(adjusted) < num_records) rec = adjusted;
                }
                if (rec != UINT64_MAX && (rec < rr.first || rec >= rr.end)) ok = false;
            }
        }
        if (!ok) {
            std::fprintf(stderr, "FAIL: ComputePerVertexRecordRange\n");
            return EXIT_FAILURE;
        }
        std::puts("PASS: ComputePerVertexRecordRange");
    }
    std::puts("PASS: 13 vertex parameters (above the old 12-attribute unpack cap)");
    return EXIT_SUCCESS;
}
