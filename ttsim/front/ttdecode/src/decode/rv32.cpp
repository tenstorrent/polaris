#include "decode/rv32.hpp"
#include "isa/isa.hpp"

#include <algorithm>
#include <stdexcept>

namespace ttdecode {
namespace decode {

bool rv32_is_valid(std::uint32_t word, const bool is_swizzled) {
    if (is_swizzled) {
        return (word & 0x3u) == 0x3u;
    }
    return rv32_is_valid(swizzle(word), true);
}

// Helper to collect values by key prefix in sorted key order
static inline std::vector<int> collect_values_by_prefix(const std::map<std::string, int>& m, const std::string& prefix, std::vector<std::string>& out_keys) {
    std::vector<int> values;
    for (const auto& kv : m) {
        const std::string& k = kv.first;
        if (k.rfind(prefix, 0) == 0u) {
            values.push_back(kv.second);
            out_keys.push_back(k);
        }
    }
    return values;
}

operands rv32_get_operands(const std::map<std::string, int>& arguments) {
    operands op;
    op.all = arguments;

    std::vector<std::string> ignore_keys;

    // sources: integers rs*, floats frs*
    {
        std::vector<std::string> keys;
        op.sources.integers = collect_values_by_prefix(arguments, std::string("rs"), keys);
        ignore_keys.insert(ignore_keys.end(), keys.begin(), keys.end());
    }
    {
        std::vector<std::string> keys;
        op.sources.floats = collect_values_by_prefix(arguments, std::string("frs"), keys);
        ignore_keys.insert(ignore_keys.end(), keys.begin(), keys.end());
    }

    // destinations: integers rd*, floats frd*
    {
        std::vector<std::string> keys;
        op.destinations.integers = collect_values_by_prefix(arguments, std::string("rd"), keys);
        ignore_keys.insert(ignore_keys.end(), keys.begin(), keys.end());
    }
    {
        std::vector<std::string> keys;
        op.destinations.floats = collect_values_by_prefix(arguments, std::string("frd"), keys);
        ignore_keys.insert(ignore_keys.end(), keys.begin(), keys.end());
    }

    // immediates: imm*
    {
        std::vector<std::string> keys;
        op.immediates = collect_values_by_prefix(arguments, std::string("imm"), keys);
        ignore_keys.insert(ignore_keys.end(), keys.begin(), keys.end());
    }

    // attributes = remaining keys
    for (const auto& kv : arguments) {
        if (std::find(ignore_keys.begin(), ignore_keys.end(), kv.first) == ignore_keys.end()) {
            op.attributes[kv.first] = kv.second;
        }
    }

    return op;
}

decoded_instruction rv32_decode(uint32_t instruction, const std::string& yaml_file, const bool is_swizzled) {
    YAML::Node iset = ttdecode::isa::parse_instruction_set_file(yaml_file);
    return rv32_decode(instruction, iset, is_swizzled);
}

decoded_instruction rv32_decode(std::uint32_t word, const YAML::Node& iset, const bool is_swizzled) {
    // Use isa::get_instruction_set to parse YAML once into a structured map
    auto set = ttdecode::isa::get_instruction_set(iset, ttdecode::isa::instruction_kind::rv32);

    return rv32_decode(word, set, is_swizzled);
}

decoded_instruction rv32_decode(const std::uint32_t word,
    const std::map<std::string, ttdecode::isa::instruction>& iset,
    const bool is_swizzled) {
    decoded_instruction out;
    out.word = is_swizzled ? word : swizzle(word);

    if (!rv32_is_valid(out.word, true)) {
        return out;
    }

    // 7-bit opcode is in bits [6:0]
    const std::uint32_t instr_opcode = (out.word & 0x7Fu);

    // Collect pointers to matching instructions, excluding RV64-specific ones
    // Sort by specificity (more encodings = more specific = checked first)
    std::vector<const std::pair<const std::string, ttdecode::isa::instruction>*> candidates;
    candidates.reserve(16);

    for (const auto& kv : iset) {
        // Skip RV64-specific instructions in RV32 decoder
        if (kv.first == "SLLI.R1" || kv.first == "SRLI.R1" || kv.first == "SRAI.R1") {
            continue;
        }
        if (kv.second.opcode == instr_opcode) {
            candidates.push_back(&kv);
        }
    }

    // Sort by number of encodings (descending) - more specific instructions first
    std::sort(candidates.begin(), candidates.end(),
        [](const auto* a, const auto* b) {
            return a->second.encodings.size() > b->second.encodings.size();
        });

    // Walk candidates and check encodings
    for (const auto* kv_ptr : candidates) {
        const auto& mnemonic = kv_ptr->first;
        const auto& ins = kv_ptr->second;

        bool encodings_match = true;
        for (const auto& e_kv : ins.encodings) {
            const auto& enc = e_kv.second;
            const std::uint32_t mask = extract_bits(out.word, enc.start, enc.size);
            if (mask != enc.value) { encodings_match = false; break; }
        }
        if (!encodings_match) continue;

        // Extract operands (keep immediate combiners similar to 7/ implementation)
        std::map<std::string, int> args;

        auto find_arg = [&](const std::string& name) -> std::optional<ttdecode::isa::argument> {
            for (const auto& a_kv : ins.arguments) {
                if (a_kv.second.name == name) return a_kv.second;
            }
            return std::nullopt;
        };

        for (const auto& a_kv : ins.arguments) {
            const auto& arg = a_kv.second;
            const std::string& name = arg.name;
            const std::uint32_t start_bit = arg.start;
            const std::uint32_t size      = arg.size;

            if (name == "imm[11:0]") {
                const std::uint32_t v = extract_bits(out.word, start_bit, size);
                args["imm"] = sign_extend(v, size - 1);
            } else if (name == "imm[4:0]") {
                auto o = find_arg("imm[11:5]");
                if (!o) throw std::runtime_error("imm[11:5] not found for imm[4:0]");
                const std::uint32_t a1 = extract_bits(out.word, start_bit, size);
                const std::uint32_t a2 = extract_bits(out.word, o->start, o->size);
                const std::uint32_t v  = (a2 << size) | a1;
                args["imm"] = sign_extend(v, size + o->size - 1);
            } else if (name == "imm[11:5]") {
                // handled with imm[4:0]
            } else if (name == "imm[4:1|11]") {
                auto o = find_arg("imm[12|10:5]");
                if (!o) throw std::runtime_error("imm[12|10:5] not found for imm[4:1|11]");
                const std::uint32_t imm10_5 = extract_bits(out.word, o->start, 6);
                const std::uint32_t imm12   = extract_bits(out.word, o->start + 6, 1);
                const std::uint32_t imm11   = extract_bits(out.word, start_bit, 1);
                const std::uint32_t imm4_1  = extract_bits(out.word, start_bit + 1, 4);
                const std::uint32_t v       = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1);
                args["imm"] = sign_extend(v, 12);
            } else if (name == "imm[12|10:5]") {
                // handled with imm[4:1|11]
            } else if (name == "imm[31:12]") {
                const std::uint32_t v = extract_bits(out.word, start_bit, size);
                args["imm"] = sign_extend(v, size - 1);
            } else if (name == "imm[20|10:1|11|19:12]") {
                const std::uint32_t imm20    = extract_bits(out.word, 31, 1);
                const std::uint32_t imm10_1  = extract_bits(out.word, 21, 11);
                const std::uint32_t imm11    = extract_bits(out.word, 20, 1);
                const std::uint32_t imm19_12 = extract_bits(out.word, 12, 8);
                const std::uint32_t v        = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1);
                args["imm"] = sign_extend(v, 20);
            } else if (name == "shamt" || name == "uimm" || name == "uimm[31:12]") {
                const std::uint32_t v = extract_bits(out.word, start_bit, size);
                args["imm"] = static_cast<int>(v);
            } else {
                const std::uint32_t v = extract_bits(out.word, start_bit, size);
                args[name] = static_cast<int>(v);
            }
        }

        out.kind = ttdecode::isa::instruction_kind::rv32;
        out.opcode = instr_opcode;
        out.mnemonic = mnemonic;
        out.operands = rv32_get_operands(args);
        return out;
    }

    return out;
}

} // namespace decode
} // namespace ttdecode
