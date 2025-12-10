#include "decode/decode.hpp"

namespace ttdecode {
namespace decode {

ttdecode::isa::instruction_kind
get_instruction_kind(const std::uint32_t instruction,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled)
{
    // Assumes there's only one valid tensix kind per instruction

    for (const auto& [kind, set] : sets) {
        if (ttdecode::isa::is_tensix(kind) && tensix_is_valid(instruction, is_swizzled)) {
            return kind;
        } else if ((ttdecode::isa::instruction_kind::rv32 == kind) && rv32_is_valid(instruction, is_swizzled)) {
            return kind;
        }
    }

    throw std::invalid_argument("Instruction does not match any known instruction set.");
}

ttdecode::isa::instruction_kind
get_instruction_kind(const std::uint32_t instruction,
    const std::set<ttdecode::isa::instruction_kind> kinds,
    const bool is_swizzled)
{
    // Assumes there's only one valid tensix kind per instruction

    for (const auto& kind : kinds) {
        if (ttdecode::isa::is_tensix(kind) && tensix_is_valid(instruction, is_swizzled)) {
            return kind;
        } else if ((ttdecode::isa::instruction_kind::rv32 == kind) && rv32_is_valid(instruction, is_swizzled)) {
            return kind;
        }
    }

    throw std::invalid_argument("Instruction does not match any known instruction set.");
}

decoded_instruction
decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const ttdecode::isa::instruction_set& set,
    const bool is_swizzled)
{
    if (ttdecode::isa::is_tensix(kind)) {
        return tensix_decode(instruction, kind, set, is_swizzled);
    } else {
        return rv32_decode(instruction, set, is_swizzled);
    }
}

decoded_instruction
decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled)
{
    auto it = sets.find(kind);
    if (it == sets.end()) {
        throw std::invalid_argument("Instruction set for the specified kind not found.");
    }
    return decode(instruction, kind, it->second, is_swizzled);
}

decoded_instruction
decode(const std::uint32_t instruction,
    const std::set<ttdecode::isa::instruction_kind> kinds,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled)
{
    const auto kind = get_instruction_kind(instruction, kinds, is_swizzled);
    return decode(instruction, kind, sets, is_swizzled);
}

decoded_instruction
decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled)
{
    const auto  kind = get_instruction_kind(instruction, sets, is_swizzled);
    const auto& set = sets.at(kind);
    return decode(instruction, kind, set, is_swizzled);
}

decoded_instruction
decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const bool is_swizzled)
{
    return decode(instruction, kind, ttdecode::isa::get_instruction_set(kind), is_swizzled);
}

decoded_instruction
decode(const std::uint32_t instruction,
    const bool is_swizzled)
{
    if (rv32_is_valid(instruction, is_swizzled)) {
        return decode(
            instruction,
            ttdecode::isa::instruction_kind::rv32,
            ttdecode::isa::get_instruction_set(ttdecode::isa::instruction_kind::rv32),
            is_swizzled);
    }

    throw std::invalid_argument("- error: could not match instruction to unique instruction set.");
}

std::vector<decoded_instruction>
decode(const std::vector<std::uint32_t>& instructions,
    const std::set<ttdecode::isa::instruction_kind> kinds,
    const ttdecode::isa::instruction_sets& sets,
    const bool is_swizzled)
{
    std::vector<decoded_instruction> results;
    results.reserve(instructions.size());

    for (std::size_t i = 0; i < instructions.size(); ++i) {
        results.push_back(decode(instructions[i], kinds, sets, is_swizzled));
    }

    return results;
}

std::vector<decoded_instruction>
decode(const std::vector<std::uint32_t>& instructions,
    const std::set<ttdecode::isa::instruction_kind> kinds,
    const ttdecode::isa::instruction_sets& sets,
    const std::vector<bool>& is_swizzled)
{
    if (instructions.size() != is_swizzled.size()) {
        throw std::invalid_argument("- error: instructions and swizzle flags size mismatch");
    }

    std::vector<decoded_instruction> results;
    results.reserve(instructions.size());

    for (std::size_t i = 0; i < instructions.size(); ++i) {
        results.push_back(decode(instructions[i], kinds, sets, is_swizzled[i]));
    }

    return results;
}

} // namespace decode
}  // namespace ttdecode