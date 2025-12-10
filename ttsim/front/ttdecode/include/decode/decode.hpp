#pragma once

#include "decode/decoded_instruction.hpp"
#include "decode/rv32.hpp"
#include "decode/tensix.hpp"
#include "isa/isa.hpp"

namespace ttdecode {
namespace decode {

ttdecode::isa::instruction_kind
get_instruction_kind(const std::uint32_t instruction,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled = true);

ttdecode::isa::instruction_kind
get_instruction_kind(const std::uint32_t instruction,
    const std::set<ttdecode::isa::instruction_kind> kinds,
    const bool is_swizzled);

decoded_instruction
decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const ttdecode::isa::instruction_set& set,
    const bool is_swizzled = true);

decoded_instruction
decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled = true);

decoded_instruction
decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled = true);

decoded_instruction
decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const bool is_swizzled = true);

decoded_instruction
decode(const std::uint32_t instruction,
    const bool is_swizzled = true);

std::vector<decoded_instruction>
decode(const std::vector<std::uint32_t>& instructions,
    const std::set<ttdecode::isa::instruction_kind> kinds,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled = true);

std::vector<decoded_instruction>
decode(const std::vector<std::uint32_t>& instructions,
    const std::set<ttdecode::isa::instruction_kind> kinds,
    const ttdecode::isa::instruction_sets& sets,
    const std::vector<bool>& swizzle_flags);

}   // namespace decode
}  // namespace ttdecode