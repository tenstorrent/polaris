#include "decode/rv32.hpp"
#include "decode/tensix.hpp"
#include <gtest/gtest.h>
#include <string>

static std::string get_assembly_path(const std::string &rel) {
    std::string current_file(__FILE__);
    size_t pos = current_file.find_last_of("/\\");
    std::string dir = (pos != std::string::npos) ? current_file.substr(0, pos) : ".";
    return dir + "/" + rel;
}

TEST(Decode, RV32_LUI) {
    using namespace ttdecode::decode;
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/rv32/assembly.yaml");

    // Build a simple LUI: rd=2, uimm=0x12345
    uint32_t uimm = 0x12345u & 0xFFFFFu;
    uint32_t rd = 2u;
    uint32_t opcode = 0x37u; // LUI
    uint32_t word = (uimm << 12) | (rd << 7) | opcode;

    auto di = rv32_decode(word, file, /*is_swizzled=*/true);
    ASSERT_TRUE(di.mnemonic.has_value());
    EXPECT_EQ(*di.mnemonic, std::string("LUI"));
    ASSERT_TRUE(di.opcode.has_value());
    EXPECT_EQ(*di.opcode, opcode);
}

// Helper: craft a swizzled tensix word whose opcode (after rotl32 by 6) is 'op'
static uint32_t make_tensix_swizzled_with_opcode(uint8_t op) {
    uint32_t x = 0u;
    // After rotl by 6, low 8 bits come from original bits [26..31,0,1]
    // Map op bit i -> x bit (i+26) mod 32 for i=0..7
    for (int i = 0; i < 8; ++i) {
        int src = (i + 26) & 31;
        if ((op >> i) & 1) {
            x |= (1u << src);
        }
    }
    return x;
}

TEST(Decode, Tensix_OpcodeMatch_WH) {
    using namespace ttdecode::decode;
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/ttwh/assembly.yaml");

    // ATGETM in ttwh has op_binary 0xA0
    uint32_t sw = make_tensix_swizzled_with_opcode(0xA0u);
    auto di = tensix_decode(sw, ttdecode::isa::instruction_kind::ttwh, file, /*is_swizzled=*/true);

    ASSERT_TRUE(di.opcode.has_value());
    EXPECT_EQ(*di.opcode, 0xA0u);
    // Expect a valid mnemonic for this opcode (ATGETM)
    ASSERT_TRUE(di.mnemonic.has_value());
}

TEST(Decode, Tensix_OpcodeMatch_BH) {
    using namespace ttdecode::decode;
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/ttbh/assembly.yaml");

    uint32_t sw = make_tensix_swizzled_with_opcode(0xA0u);
    auto di = tensix_decode(sw, ttdecode::isa::instruction_kind::ttbh, file, true);

    ASSERT_TRUE(di.opcode.has_value());
    EXPECT_EQ(*di.opcode, 0xA0u);
    ASSERT_TRUE(di.mnemonic.has_value());
}

TEST(Decode, Tensix_OpcodeMatch_QS) {
    using namespace ttdecode::decode;
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/ttqs/assembly.yaml");

    uint32_t sw = make_tensix_swizzled_with_opcode(0xA0u);
    auto di = tensix_decode(sw, ttdecode::isa::instruction_kind::ttqs, file, true);

    ASSERT_TRUE(di.opcode.has_value());
    EXPECT_EQ(*di.opcode, 0xA0u);
    ASSERT_TRUE(di.mnemonic.has_value());
}
