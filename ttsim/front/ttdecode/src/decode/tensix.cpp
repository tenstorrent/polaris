#include "decode/tensix.hpp"
#include "isa/isa.hpp"
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

namespace ttdecode {
namespace decode {

bool tensix_is_valid(std::uint32_t instruction, bool is_swizzled) {
    if (is_swizzled) {
        return (instruction & 0x3u) != 0x3u;
    }
    return tensix_is_valid(swizzle(instruction), true);
}

decoded_instruction tensix_decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const std::map<std::string, ttdecode::isa::instruction>& iset,
    const bool is_swizzled) {

    decoded_instruction out;
    out.word = is_swizzled ? instruction : swizzle(instruction);

    if (!tensix_is_valid(out.word, true)) {
        return out;
    }

    const std::uint32_t opcode = tensix_get_opcode(out.word, true);

    const ttdecode::isa::instruction* chosen = nullptr;
    std::string chosen_name;
    for (const auto& kv : iset) {
        if (kv.second.opcode == opcode) { chosen = &kv.second; chosen_name = kv.first; break; }
    }

    if (!chosen) {
        return out;
    }
    std::uint32_t unswizzled_word = unswizzle(out.word);
    std::map<std::string, int> args;
    std::map<std::string, std::vector<std::string>> decoded_labels;
    for (const auto& a_kv : chosen->arguments) {
        const ttdecode::isa::argument& a = a_kv.second;
        const std::uint32_t v = extract_bits(unswizzled_word, a.start, a.size);
        args[a.name] = static_cast<int>(v);
        if (a.fcov.has_value()) {
            const ttdecode::isa::argument::fcov_spec& spec = a.fcov.value();
            std::vector<std::string> labels;
            if (spec.kind == ttdecode::isa::argument::fcov_kind::bins) {
                if (spec.bin_eval == ttdecode::isa::argument::fcov_spec::bin_eval_kind::bitwise) {
                    for (const auto& b : spec.bins) {
                        if ((v & b.value) != 0u) { labels.push_back(b.name); }
                    }
                } else {
                    for (const auto& b : spec.bins) {
                        if (v == b.value) { labels.push_back(b.name); break; }
                    }
                }
            } else if (spec.kind == ttdecode::isa::argument::fcov_kind::bin_interval) {
                for (const auto& iv : spec.intervals) {
                    if (v >= iv.low && v <= iv.high) { labels.push_back(iv.name); }
                }
            } else if (spec.kind == ttdecode::isa::argument::fcov_kind::boolean) {
                labels.push_back(v ? std::string("true") : std::string("false"));
            }
            if (!labels.empty()) { decoded_labels.emplace(a.name, std::move(labels)); }
        }
    }

    out.kind = kind;
    out.opcode = opcode;
    out.mnemonic = chosen_name;
    operands ops;
    ops.all = args;
    ops.attributes = args;
    ops.decoded_values = std::move(decoded_labels);

    out.operands = ops;
    return out;
}

decoded_instruction tensix_decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const YAML::Node& iset,
    const bool is_swizzled) {
    // Build instruction set from YAML and delegate to map-based decoder
    auto set = ttdecode::isa::get_instruction_set(iset,
        ttdecode::isa::opcode_start_bit(kind),
        ttdecode::constants::NUM_BITS_PER_INSTRUCTION);
    return tensix_decode(instruction, kind, set, is_swizzled);
}

decoded_instruction tensix_decode(const std::uint32_t instruction,
    const ttdecode::isa::instruction_kind kind,
    const std::string& yaml_file,
    const bool is_swizzled) {
    auto iset = ttdecode::isa::get_instruction_set(yaml_file, kind);
    return tensix_decode(instruction, kind, iset, is_swizzled);
}



} // namespace decode
} // namespace ttdecode
