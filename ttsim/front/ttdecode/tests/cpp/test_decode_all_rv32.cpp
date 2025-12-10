#include "decode/rv32.hpp"
#include "isa/isa.hpp"
#include <gtest/gtest.h>
#include <string>
#include <map>

static std::string get_assembly_path(const std::string &rel) {
    std::string current_file(__FILE__);
    size_t pos = current_file.find_last_of("/\\");
    std::string dir = (pos != std::string::npos) ? current_file.substr(0, pos) : ".";
    return dir + "/" + rel;
}

static std::uint32_t set_bits(std::uint32_t word, std::uint32_t start, std::uint32_t size, std::uint32_t value) {
    const std::uint32_t mask = ((size == 32 ? 0xFFFFFFFFu : ((1u << size) - 1u)) << start);
    word &= ~mask;
    word |= ((value & (size == 32 ? 0xFFFFFFFFu : ((1u << size) - 1u))) << start);
    return word;
}

TEST(DecodeAll, RV32_AllInstructionsMatchMnemonic) {
    const std::string file = get_assembly_path("../../../../config/llk/instruction_sets/rv32/assembly.yaml");
    auto iset = ttdecode::isa::get_instruction_set(file, ttdecode::isa::instruction_kind::rv32);

    for (const auto &kv : iset) {
        const std::string &mnemonic = kv.first;
        const ttdecode::isa::instruction &ins = kv.second;

        // Skip RV64-specific instructions that should not be decoded by RV32 decoder
        if (mnemonic == "SLLI.R1" || mnemonic == "SRLI.R1" || mnemonic == "SRAI.R1") {
            continue;
        }

        std::uint32_t word = 0u;
        word = set_bits(word, 0u, 7u, static_cast<std::uint32_t>(ins.opcode));
        for (const std::pair<const std::uint8_t, ttdecode::isa::encoding> &e_kv : ins.encodings) {
            const ttdecode::isa::encoding &enc = e_kv.second;
            word = set_bits(word, enc.start, enc.size, enc.value);
        }

        ttdecode::decode::decoded_instruction di = ttdecode::decode::rv32_decode(word, iset, true);
        ASSERT_TRUE(di.mnemonic.has_value());
        EXPECT_EQ(*di.mnemonic, mnemonic);
        ASSERT_TRUE(di.opcode.has_value());
        EXPECT_EQ(*di.opcode, static_cast<std::uint32_t>(ins.opcode));
    }
}
