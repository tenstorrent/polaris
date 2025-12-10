#pragma once

#include "decode/decoded_instruction.hpp"
#include "decode/utils.hpp"
#include "isa/yaml_parser.hpp"
#include <string>

namespace ttdecode {
namespace decode {

// RV32 specific helpers
bool rv32_is_valid(std::uint32_t word, bool is_swizzled = true);

// Build grouped operands from flat argument map (mirrors Python get_operands)
operands rv32_get_operands(const std::map<std::string, int>& arguments);

decoded_instruction rv32_decode(const std::uint32_t word,
    const std::map<std::string, ttdecode::isa::instruction>& iset,
    const bool is_swizzled = true);

// Decode using a pre-parsed YAML::Node
decoded_instruction rv32_decode(
    const std::uint32_t word,
    const YAML::Node& iset,
    const bool is_swizzled = true);

// Convenience overload: load from file path
decoded_instruction rv32_decode(
    const std::uint32_t word,
    const std::string& yaml_file,
    const bool is_swizzled = true);

} // namespace decode
} // namespace ttdecode
