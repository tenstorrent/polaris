#include "decode/tensix.hpp"
#include "isa/isa.hpp"
#include <gtest/gtest.h>
#include <string>

static std::string get_assembly_path(const std::string &rel) {
    std::string current_file(__FILE__);
    size_t pos = current_file.find_last_of("/\\");
    std::string dir = (pos != std::string::npos) ? current_file.substr(0, pos) : ".";
    return dir + "/" + rel;
}

static std::uint32_t make_tensix_swizzled_with_opcode(std::uint8_t op) {
    std::uint32_t x = 0u;
    for (int i = 0; i < 8; ++i) {
        int src = (i + 26) & 31;
        if ((op >> i) & 1) x |= (1u << src);
    }
    return x;
}

TEST(DecodeAll, Tensix_AllInstructions_MnemonicAndSwizzleParity) {
    using ttdecode::isa::instruction_kind;

    const std::string wh = get_assembly_path("../../../../config/llk/instruction_sets/ttwh/assembly.yaml");
    const std::string bh = get_assembly_path("../../../../config/llk/instruction_sets/ttbh/assembly.yaml");
    const std::string qs = get_assembly_path("../../../../config/llk/instruction_sets/ttqs/assembly.yaml");

    const struct { ttdecode::isa::instruction_kind kind; std::string path; } cases[] = {
        { instruction_kind::ttwh, wh },
        { instruction_kind::ttbh, bh },
        { instruction_kind::ttqs, qs },
    };

    for (const auto &c : cases) {
        auto iset = ttdecode::isa::get_instruction_set(c.path, c.kind);
        for (const auto &kv : iset) {
            const std::string &mnemonic = kv.first;
            const std::uint8_t opcode = kv.second.opcode;

            std::uint32_t sw = make_tensix_swizzled_with_opcode(opcode);
            ttdecode::decode::decoded_instruction d1 = ttdecode::decode::tensix_decode(sw, c.kind, c.path, true);
            ttdecode::decode::decoded_instruction d2 = ttdecode::decode::tensix_decode(ttdecode::decode::swizzle(sw), c.kind, c.path, false);

            ASSERT_TRUE(d1.mnemonic.has_value());
            EXPECT_EQ(*d1.mnemonic, mnemonic);
            ASSERT_TRUE(d1.opcode.has_value());
            EXPECT_EQ(*d1.opcode, static_cast<std::uint32_t>(opcode));

            EXPECT_EQ(d1.opcode, d2.opcode);
            EXPECT_EQ(d1.mnemonic, d2.mnemonic);
        }
    }
}

