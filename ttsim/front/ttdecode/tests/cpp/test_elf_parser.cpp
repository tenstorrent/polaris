#include <gtest/gtest.h>
#include "elf/elf_parsers.hpp"

TEST(ElfParsers, ConstructEmpty) {
    std::vector<std::string> files;
    ttdecode::elf::parsers ps(files);
    EXPECT_EQ(ps.size(), static_cast<std::size_t>(0));
    EXPECT_TRUE(ps.instruction_kinds_match_for_all_elfs());
    const auto merged = ps.get_instruction_kinds();
    EXPECT_TRUE(merged.empty());
    const auto common = ps.get_instruction_kinds("common");
    EXPECT_TRUE(common.empty());
}
