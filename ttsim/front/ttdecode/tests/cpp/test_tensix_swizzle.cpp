#include "decode/tensix.hpp"
#include "decode/utils.hpp"
#include "isa/isa.hpp"
#include <gtest/gtest.h>
#include <string>

static std::string get_assembly_path(const std::string &rel) {
    std::string current_file(__FILE__);
    size_t pos = current_file.find_last_of("/\\");
    std::string dir = (pos != std::string::npos) ? current_file.substr(0, pos) : ".";
    return dir + "/" + rel;
}

TEST(Tensix, SwizzledVsUnswizzledSame) {
    using namespace ttdecode::decode;

    const std::string wh = get_assembly_path("../../../../config/llk/instruction_sets/ttwh/assembly.yaml");
    const std::string bh = get_assembly_path("../../../../config/llk/instruction_sets/ttbh/assembly.yaml");
    const std::string qs = get_assembly_path("../../../../config/llk/instruction_sets/ttqs/assembly.yaml");

    const struct { ttdecode::isa::instruction_kind kind; std::string path; } cases[] = {
        { ttdecode::isa::instruction_kind::ttwh, wh },
        { ttdecode::isa::instruction_kind::ttbh, bh },
        { ttdecode::isa::instruction_kind::ttqs, qs },
    };

    for (const auto &c : cases) {
        // Use an opcode that exists in the YAML (0xA0 used in other tests)
        std::uint32_t word = 0xA0u << 24;
        auto di_sw = tensix_decode(ttdecode::decode::swizzle(word), c.kind, c.path, /*is_swizzled=*/true);
        auto di_unsw = tensix_decode(word, c.kind, c.path, /*is_swizzled=*/false);

        EXPECT_EQ(di_sw.opcode, di_unsw.opcode);
        EXPECT_EQ(di_sw.mnemonic, di_unsw.mnemonic);
        // operands may be empty depending on YAML, but when present, all mapping should match
        if (di_sw.operands.has_value() && di_unsw.operands.has_value()) {
            EXPECT_EQ(di_sw.operands->all, di_unsw.operands->all);
            EXPECT_EQ(di_sw.operands->attributes, di_unsw.operands->attributes);
        }
    }
}
