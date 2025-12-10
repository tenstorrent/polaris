#pragma once

#include "decode/tensix.hpp"

namespace ttdecode {
namespace decode {

inline decoded_instruction ttbh_decode(const std::uint32_t instruction,
    const std::map<std::string, ttdecode::isa::instruction>& iset,
    const bool is_swizzled = true) {
    return tensix_decode(
        instruction, 
        ttdecode::isa::instruction_kind::ttbh, 
        iset, 
        is_swizzled);
}

inline decoded_instruction ttbh_decode(const std::uint32_t instruction,
    const YAML::Node& iset,
    const bool is_swizzled = true) {
    return tensix_decode(
        instruction, 
        ttdecode::isa::instruction_kind::ttbh, 
        iset, 
        is_swizzled);
}

inline decoded_instruction ttbh_decode(const std::uint32_t instruction,
    const std::string& yaml_file,
    const const bool is_swizzled = true) {
    return tensix_decode(
        instruction, 
        ttdecode::isa::instruction_kind::ttbh, 
        yaml_file, 
        is_swizzled);
}

} // namespace decode
} // namespace ttdecode

