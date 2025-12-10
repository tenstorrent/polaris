#pragma once

#include "constants/constants.hpp"
#include "decode/decoded_instruction.hpp"
#include "decode/utils.hpp"
#include "isa/isa.hpp"
#include "isa/yaml_parser.hpp"
#include <string>

namespace ttdecode {
namespace decode {

// Tensix helpers (ttwh/ttbh/ttqs share the same core format)
inline std::uint32_t tensix_get_opcode(std::uint32_t instruction, bool is_swizzled = true) {
    if (is_swizzled) {
        // Rotate left by 6 then mask 8 bits
        return (rotl32(instruction, 6u) & 0xFFu);
    }
    return tensix_get_opcode(swizzle(instruction), true);
}

bool tensix_is_valid(const std::uint32_t instruction, bool is_swizzled = true);

// Decode using instruction set
decoded_instruction
tensix_decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const std::map<std::string, ttdecode::isa::instruction>& iset,
    const bool is_swizzled = true);

// Decode using pre-parsed YAML::Node
decoded_instruction
tensix_decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const YAML::Node& iset,
    const bool is_swizzled = true);

// Convenience: load from file path
decoded_instruction
tensix_decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const std::string& yaml_file,
    const bool is_swizzled = true);

} // namespace decode
} // namespace ttdecode

