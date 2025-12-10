#include "isa/isa.hpp"
#include <gtest/gtest.h>
#include <string>

static std::string get_assembly_path(const std::string &rel) {
    std::string current_file(__FILE__);
    size_t pos = current_file.find_last_of("/\\");
    std::string dir = (pos != std::string::npos) ? current_file.substr(0, pos) : ".";
    return dir + "/" + rel;
}

TEST(IsaYaml, ParsesRv32Basic) {
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/rv32/assembly.yaml");
    auto set = ttdecode::isa::get_instruction_set(file, 0, 32);

    // Sanity: a few known mnemonics must exist
    ASSERT_TRUE(set.find("LUI") != set.end());
    ASSERT_TRUE(set.find("JALR") != set.end());
    ASSERT_TRUE(set.find("ADD") != set.end());

    // Check opcode for LUI (0b0110111 == 55)
    EXPECT_EQ(set["LUI"].opcode, static_cast<std::uint8_t>(0b0110111));

    // JALR has funct3 at start 12 with value 0 and size 3
    {
        const auto &ins = set["JALR"];
        auto it = ins.encodings.find(12);
        ASSERT_TRUE(it != ins.encodings.end());
        EXPECT_EQ(it->second.size, 3);
        EXPECT_EQ(it->second.value, 0u);
    }

    // ADD has funct7 at 25 (0), funct3 at 12 (0), and rs2 at 20
    {
        const auto &ins = set["ADD"];
        auto f7 = ins.encodings.find(25);
        auto f3 = ins.encodings.find(12);
        ASSERT_TRUE(f7 != ins.encodings.end());
        ASSERT_TRUE(f3 != ins.encodings.end());
        EXPECT_EQ(f7->second.value, 0u);
        EXPECT_EQ(f3->second.value, 0u);

        auto rs2 = ins.arguments.find(20);
        ASSERT_TRUE(rs2 != ins.arguments.end());
        EXPECT_EQ(rs2->second.size, 5);
    }
}

TEST(IsaYaml, ParsesWHBasic) {
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/ttwh/assembly.yaml");
    auto set = ttdecode::isa::get_instruction_set(file, 24, 32);

    // Sanity: a few known mnemonics must exist
    ASSERT_TRUE(set.find("SFPNOP") != set.end());
    ASSERT_TRUE(set.find("MOP") != set.end());
    ASSERT_TRUE(set.find("MVMUL") != set.end());

    EXPECT_EQ(set["ATGETM"].opcode, static_cast<std::uint8_t>(0xa0));

    {
        const auto &ins = set["SEMGET"];
        auto it = ins.arguments.find(2);
        ASSERT_TRUE(it != ins.arguments.end());
        EXPECT_EQ(it->second.size, 22);
    }

    {
        const auto &ins = set["SFPCAST"];
        auto a1 = ins.arguments.find(0);
        auto a2 = ins.arguments.find(4);
        auto a3 = ins.arguments.find(8);
        ASSERT_TRUE(a1 != ins.arguments.end());
        ASSERT_TRUE(a2 != ins.arguments.end());
        ASSERT_TRUE(a3 != ins.arguments.end());
        EXPECT_EQ(a1->second.name, "instr_mod1");
        EXPECT_EQ(a1->second.size, 4);
        EXPECT_EQ(a2->second.name, "lreg_dest");
        EXPECT_EQ(a2->second.size, 4);
        EXPECT_EQ(a3->second.name, "lreg_src_c");
        EXPECT_EQ(a3->second.size, 16);
    }
}

TEST(IsaYaml, ParsesBHBasic) {
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/ttbh/assembly.yaml");
    auto set = ttdecode::isa::get_instruction_set(file, 24, 32);

    ASSERT_TRUE(set.find("ATGETM") != set.end());
    ASSERT_TRUE(set.find("MVMUL") != set.end());
    ASSERT_TRUE(set.find("NOP") != set.end());

    EXPECT_EQ(set["ATGETM"].opcode, static_cast<std::uint8_t>(0xa0));
}

TEST(IsaYaml, ParsesQSBasic) {
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/ttqs/assembly.yaml");
    auto set = ttdecode::isa::get_instruction_set(file, 24, 32);

    ASSERT_TRUE(set.find("ATGETM") != set.end());
    ASSERT_TRUE(set.find("MVMUL") != set.end());
    ASSERT_TRUE(set.find("NOP") != set.end());

    EXPECT_EQ(set["ATGETM"].opcode, static_cast<std::uint8_t>(0xa0));
}
