#include "isa/isa.hpp"
#include "isa/defaults.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ttdecode {
namespace isa {

namespace {

// Parse a string into unsigned integer.
// Supports decimal, 0x (hex), 0b (binary), and 0o (octal).
// Accepts optional leading/trailing spaces.
static std::uint32_t parse_string_to_uint(const std::string &s) {
    // trim spaces
    size_t i = 0, j = s.size();
    while (i < j && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    while (j > i && std::isspace(static_cast<unsigned char>(s[j - 1]))) --j;

    if (i >= j) {
        throw std::invalid_argument("Empty string");
    }

    // Check for prefixes
    if (j - i >= 2 && s[i] == '0') {
        char prefix = std::tolower(s[i + 1]);

        if (prefix == 'b') {
            // Binary: 0b prefix
            std::uint32_t v = 0;
            for (size_t k = i + 2; k < j; ++k) {
                char c = s[k];
                if (c == '0' || c == '1') {
                    v = (v << 1) | static_cast<std::uint32_t>(c - '0');
                } else if (std::isspace(static_cast<unsigned char>(c))) {
                    continue;
                } else {
                    throw std::invalid_argument("Invalid binary literal: " + s);
                }
            }
            return v;
        } else if (prefix == 'x') {
            // Hexadecimal: 0x prefix
            std::uint32_t v = 0;
            for (size_t k = i + 2; k < j; ++k) {
                char c = std::tolower(s[k]);
                if (std::isdigit(c)) {
                    v = (v << 4) | static_cast<std::uint32_t>(c - '0');
                } else if (c >= 'a' && c <= 'f') {
                    v = (v << 4) | static_cast<std::uint32_t>(c - 'a' + 10);
                } else if (std::isspace(static_cast<unsigned char>(c))) {
                    continue;
                } else {
                    throw std::invalid_argument("Invalid hexadecimal literal: " + s);
                }
            }
            return v;
        } else if (prefix == 'o') {
            // Octal: 0o prefix
            std::uint32_t v = 0;
            for (size_t k = i + 2; k < j; ++k) {
                char c = s[k];
                if (c >= '0' && c <= '7') {
                    v = (v << 3) | static_cast<std::uint32_t>(c - '0');
                } else if (std::isspace(static_cast<unsigned char>(c))) {
                    continue;
                } else {
                    throw std::invalid_argument("Invalid octal literal: " + s);
                }
            }
            return v;
        }
    }

    // Decimal (fallback)
    std::uint32_t v = 0;
    for (size_t k = i; k < j; ++k) {
        char c = s[k];
        if (std::isdigit(c)) {
            std::uint32_t digit = static_cast<std::uint32_t>(c - '0');
            // Check for overflow
            if (v > (UINT32_MAX - digit) / 10) {
                throw std::invalid_argument("Decimal overflow: " + s);
            }
            v = v * 10 + digit;
        } else if (std::isspace(static_cast<unsigned char>(c))) {
            continue;
        } else {
            throw std::invalid_argument("Invalid decimal literal: " + s);
        }
    }
    return v;
}


static argument parse_argument(const YAML::Node &n) {
    argument a{};
    a.name = n["name"].as<std::string>();
    a.start = static_cast<std::uint8_t>(n["start_bit"].as<int>());
    if (n["size"]) {
        a.size = static_cast<std::uint8_t>(n["size"].as<int>());
    }
    // Parse optional fcov specification attached to this argument
    if (n["fcov_point_bins"]) {
        const YAML::Node &bins_node = n["fcov_point_bins"]["bins"];
        if (bins_node && bins_node.IsSequence() && bins_node.size() > 0) {
            argument::fcov_spec spec{};
            spec.kind = argument::fcov_kind::bins;
            spec.bins.reserve(bins_node.size());
            for (std::size_t i = 0; i < bins_node.size(); ++i) {
                const YAML::Node &b = bins_node[i];
                argument::fcov_bin fb{};
                fb.name = b["name"].as<std::string>();

                // Handle both "value" and "wildcard_value" keys
                std::string val_s;
                if (b["value"]) {
                    val_s = b["value"].as<std::string>();
                } else if (b["wildcard_value"]) {
                    val_s = b["wildcard_value"].as<std::string>();

                    // Trim whitespace first
                    size_t start = 0, end = val_s.size();
                    while (start < end && std::isspace(static_cast<unsigned char>(val_s[start]))) ++start;
                    while (end > start && std::isspace(static_cast<unsigned char>(val_s[end - 1]))) --end;

                    if (start >= end) {
                        throw std::runtime_error("wildcard_value is empty or contains only whitespace");
                    }

                    // Check if it has a valid prefix (0b, 0x, 0o, etc.)
                    if (end - start >= 2 && val_s[start] == '0') {
                        char prefix = std::tolower(val_s[start + 1]);
                        bool has_valid_prefix = (prefix == 'b' || prefix == 'x' || prefix == 'o');

                        if (has_valid_prefix) {
                            // Replace 'x'/'X' wildcards with '0' starting after the prefix
                            for (size_t i = start + 2; i < end; ++i) {
                                if (val_s[i] == 'x' || val_s[i] == 'X') {
                                    val_s[i] = '0';
                                }
                            }

                            // Verify no wildcards remain after the prefix
                            for (size_t i = start + 2; i < end; ++i) {
                                if (val_s[i] == 'x' || val_s[i] == 'X') {
                                    throw std::runtime_error("wildcard_value contains unresolved wildcard character after replacement: " + val_s);
                                }
                            }
                        } else {
                            // Has '0' prefix but not a recognized format - check for wildcards anyway
                            bool has_wildcards = false;
                            for (size_t i = start; i < end; ++i) {
                                if (val_s[i] == 'x' || val_s[i] == 'X') {
                                    has_wildcards = true;
                                    break;
                                }
                            }
                            if (has_wildcards) {
                                throw std::runtime_error("wildcard_value contains wildcards but has unrecognized prefix (expected 0b, 0x, or 0o): " + val_s);
                            }
                        }
                    } else {
                        // No prefix - check if wildcards are present (shouldn't be for decimal)
                        for (size_t i = start; i < end; ++i) {
                            if (val_s[i] == 'x' || val_s[i] == 'X') {
                                throw std::runtime_error("wildcard_value contains wildcards in decimal format (prefix required): " + val_s);
                            }
                        }
                    }
                } else {
                    throw std::runtime_error("fcov_point_bins entry missing both 'value' and 'wildcard_value' keys");
                }

                fb.value = parse_string_to_uint(val_s);
                spec.bins.push_back(fb);
            }
            // Determine evaluation kind for bins once during parse
            auto is_power_of_two = [](std::uint32_t v) -> bool { return v != 0u && (v & (v - 1u)) == 0u; };
            bool all_powers = true;
            for (const auto &b : spec.bins) {
                if (!is_power_of_two(b.value)) { all_powers = false; break; }
            }
            spec.bin_eval = all_powers ? argument::fcov_spec::bin_eval_kind::bitwise
                                       : argument::fcov_spec::bin_eval_kind::equality;
            a.fcov = spec;
        }
    } else if (n["fcov_point_bin_interval"]) {
        const YAML::Node &bins_node = n["fcov_point_bin_interval"]["bins"];
        if (bins_node && bins_node.IsSequence() && bins_node.size() > 0) {
            argument::fcov_spec spec{};
            spec.kind = argument::fcov_kind::bin_interval;
            for (std::size_t i = 0; i < bins_node.size(); ++i) {
                const YAML::Node &b = bins_node[i];
                argument::fcov_interval fi{};
                fi.name = b["name"].as<std::string>();
                const YAML::Node &interval = b["interval"];
                if (interval && interval.IsSequence() && interval.size() == 2) {
                    const std::string lo_s = interval[0].as<std::string>();
                    const std::string hi_s = interval[1].as<std::string>();
                    fi.low = parse_string_to_uint(lo_s);
                    fi.high = parse_string_to_uint(hi_s);
                    spec.intervals.push_back(fi);
                }
            }
            spec.bin_eval = argument::fcov_spec::bin_eval_kind::unknown;
            if (!spec.intervals.empty()) {
                a.fcov = spec;
            }
        }
    } else if (n["fcov_point_bool"]) {
        argument::fcov_spec spec{};
        spec.kind = argument::fcov_kind::boolean;
        spec.bin_eval = argument::fcov_spec::bin_eval_kind::unknown;
        a.fcov = spec;
    }
    return a;
}

static encoding parse_encoding(const YAML::Node &n) {
    encoding e{};
    e.name = n["name"].as<std::string>();
    e.start = static_cast<std::uint8_t>(n["start_bit"].as<int>());
    e.size = static_cast<std::uint8_t>(n["size"].as<int>());
    e.value = parse_string_to_uint(n["opcode"].as<std::string>());

    return e;
}

} // namespace

std::string
to_string(const instruction_kind kind) {
    switch (kind) {
        case instruction_kind::rv32: return "rv32";
        case instruction_kind::ttwh: return "ttwh";
        case instruction_kind::ttbh: return "ttbh";
        case instruction_kind::ttqs: return "ttqs";
        default: {
            std::stringstream msg;
            msg<<"- error: no string defined for given instruction_kind, int value: "<<static_cast<std::size_t>(kind)<<std::endl;
            throw std::invalid_argument(msg.str());
        };
    }
}

instruction_kind
to_instruction_kind(const std::string& kind)
{
    if (kind == "rv32") return instruction_kind::rv32;
    if (kind == "ttwh") return instruction_kind::ttwh;
    if (kind == "ttbh") return instruction_kind::ttbh;
    if (kind == "ttqs") return instruction_kind::ttqs;

    std::stringstream msg;
    msg << "- error: unknown instruction_kind string: '" << kind << "'" << std::endl;
    throw std::invalid_argument(msg.str());
}

std::set<instruction_kind>
tensix_instruction_kinds() {
    return {
        instruction_kind::ttwh,
        instruction_kind::ttbh,
        instruction_kind::ttqs
    };
}

std::string
to_string(const argument::fcov_kind kind) {
    switch (kind) {
        case argument::fcov_kind::none : return "none";
        case argument::fcov_kind::bins : return "bins";
        case argument::fcov_kind::bin_interval : return "bin_interval";
        case argument::fcov_kind::boolean : return "boolean";
        default : {
            throw std::invalid_argument("- error: no string defined for given argument::fcov_kind");
        }
    }
}

std::string
to_string(const instruction_set &iset) {
    std::stringstream msg;
    msg << "- instruction set:\n";
    for (const auto& [mnemominc, instruction] : iset) {
        msg << "  - "<<mnemominc<<"\n";
        msg << "    - opcode: " << std::hex << static_cast<std::size_t>(instruction.opcode) << std::dec << "\n";
        msg << "    - encondings: " << std::endl;
        for (const auto& [s, e] : instruction.encodings) {
            msg << "      - "<<e.name<<", start bit: "<<static_cast<std::size_t>(e.start)<<", size: "<<static_cast<std::size_t>(e.size)<<", value: "<<static_cast<std::size_t>(e.value)<<std::endl;
        }
        msg << "    - arguments: " << std::endl;
        for (const auto& [s, a] : instruction.arguments) {
            msg << "      - "<<a.name<<", start bit: "<<static_cast<std::size_t>(a.start)<<", size: "<<static_cast<std::size_t>(a.size)<<std::endl;
            if (a.fcov.has_value()) {
                const auto& fcov = a.fcov.value();
                msg << "        - fcov kind: "<<to_string(fcov.kind)<<std::endl;
                msg << "          - bins: "<<std::endl;
                for (const auto& b : fcov.bins) {
                    msg << "            - name: "<<b.name<<", value: "<<b.value<<std::endl;
                }
                msg << "          - bin interval: "<<std::endl;
                for (const auto& b : fcov.intervals) {
                    msg << "            - name: "<<b.name<<", low: "<<b.low<<", high: "<<b.high<<std::endl;
                }
            }
        }
    }

    return msg.str();
}

std::string
get_default_instruction_set_file_path(const instruction_kind kind) {
    const auto& m = ttdecode::isa::global_defaults().instruction_set_file_paths();
    const auto it = m.find(kind);
    if (it == m.end()) throw std::runtime_error("invalid instruction kind, can not return the default instruction set file path");
    return it->second;
}

bool
is_tensix(const instruction_kind kind) {
    for (const auto& tk : tensix_instruction_kinds()) {
        if (kind == tk) {
            return true;
        }
    }
    return false;
}

std::uint32_t
opcode_start_bit(const instruction_kind kind) {
    if (is_tensix(kind)) {
        return ttdecode::constants::TENSIX_OPCODE_START_BIT;
    } else if (kind == instruction_kind::rv32) {
        return ttdecode::constants::RISCV_OPCODE_START_BIT;
    } else {
        throw std::invalid_argument("- error: opcode start bit not defined for given instruction kind");
    }
}

// Function to check, calculate and update sizes of each argument/encoding, opcode in the given instruction
void validate_and_update_instruction_sizes(instruction& instr, const std::uint8_t opcode_start, const std::uint8_t max_length) {
    // Step 1: Check that opcode doesn't exceed its size
    if (opcode_start >= max_length) {
        std::stringstream msg;
        msg << "- error: opcode start bit (" << static_cast<int>(opcode_start)
            << ") exceeds maximum instruction length (" << static_cast<int>(max_length) << ")";
        throw std::runtime_error(msg.str());
    }

    // Step 2: Create a map of all start bits across opcode, arguments and encodings
    std::map<std::uint8_t, std::string> start_bit_fields; // start_bit -> field_type:field_name

    // Add opcode
    start_bit_fields[opcode_start] = "opcode:opcode";

    // Add encodings
    for (const auto& enc_pair : instr.encodings) {
        const encoding& enc = enc_pair.second;
        auto result = start_bit_fields.emplace(enc.start, "encoding:" + enc.name);
        if (!result.second) {
            std::stringstream msg;
            msg << "- error: duplicate start bit " << static_cast<int>(enc.start)
                << " found between " << result.first->second
                << " and encoding:" << enc.name;
            throw std::runtime_error(msg.str());
        }
    }

    // Add arguments
    for (const auto& arg_pair : instr.arguments) {
        const argument& arg = arg_pair.second;
        auto result = start_bit_fields.emplace(arg.start, "argument:" + arg.name);
        if (!result.second) {
            std::stringstream msg;
            msg << "- error: duplicate start bit " << static_cast<int>(arg.start)
                << " found between " << result.first->second
                << " and argument:" << arg.name;
            throw std::runtime_error(msg.str());
        }
    }

    // Step 3: Sort the start bits in ascending order (map is already sorted)
    // Step 4: Check that there are no repeat start bits (verified above with explicit error handling)

    // Step 5: For each start bit, process the field
    auto it = start_bit_fields.begin();
    while (it != start_bit_fields.end()) {
        std::uint8_t current_start = it->first;
        std::string field_info = it->second;

        // Get the next start bit
        auto next_it = std::next(it);
        std::uint8_t next_start = (next_it != start_bit_fields.end()) ? next_it->first : max_length;

        // Calculate the available size
        std::uint8_t available_size = next_start - current_start;

        // Parse field type and name
        size_t colon_pos = field_info.find(':');
        std::string field_type = field_info.substr(0, colon_pos);
        std::string field_name = field_info.substr(colon_pos + 1);

        if (field_type == "opcode") {
            // For opcode, assume it needs at least 1 bit if no explicit size given
            // Most opcodes are typically 8 bits, but this depends on architecture
            if (available_size < 1) {
                std::stringstream msg;
                msg << "- error: opcode at bit " << static_cast<int>(current_start)
                    << " has insufficient space (available: " << static_cast<int>(available_size) << " bits)";
                throw std::runtime_error(msg.str());
            }
        } else if (field_type == "encoding") {
            // Find the encoding and check/update its size
            auto enc_it = instr.encodings.find(current_start);
            if (enc_it != instr.encodings.end()) {
                encoding& enc = enc_it->second;

                // Check if it has a size, if not calculate it
                if (enc.size == 0) {
                    enc.size = available_size;
                } else {
                    // Check that existing size doesn't exceed available space
                    if (enc.size > available_size) {
                        std::stringstream msg;
                        msg << "- error: encoding '" << enc.name << "' at bit "
                            << static_cast<int>(current_start) << " has size "
                            << static_cast<int>(enc.size) << " but only "
                            << static_cast<int>(available_size) << " bits available";
                        throw std::runtime_error(msg.str());
                    }
                }

                // Check if value fits in the size
                std::uint32_t max_value = (1u << enc.size) - 1;
                if (enc.value > max_value) {
                    std::stringstream msg;
                    msg << "- error: encoding '" << enc.name << "' value "
                        << enc.value << " exceeds maximum value " << max_value
                        << " for " << static_cast<int>(enc.size) << " bits";
                    throw std::runtime_error(msg.str());
                }
            }
        } else if (field_type == "argument") {
            // Find the argument and check/update its size
            auto arg_it = instr.arguments.find(current_start);
            if (arg_it != instr.arguments.end()) {
                argument& arg = arg_it->second;

                // Check if it has a size, if not calculate it
                if (arg.size == 0) {
                    arg.size = available_size;
                } else {
                    // Check that existing size doesn't exceed available space
                    if (arg.size > available_size) {
                        std::stringstream msg;
                        msg << "- error: argument '" << arg.name << "' at bit "
                            << static_cast<int>(current_start) << " has size "
                            << static_cast<int>(arg.size) << " but only "
                            << static_cast<int>(available_size) << " bits available";
                        throw std::runtime_error(msg.str());
                    }
                }
            }
        }

        ++it;
    }
}

std::map<std::string, instruction>
get_instruction_set(const YAML::Node &root,
    const std::uint8_t opcode_start,
    const std::uint8_t max_length) {
    if (!root || !root.IsMap()) {
        return {};
    }

    std::map<std::string, instruction> out;
    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string mnemonic = it->first.as<std::string>("");
        // Check if mnemonic already exists
        if (out.find(mnemonic) != out.end()) {
            std::stringstream msg;
            msg << "- error: duplicate mnemonic '" << mnemonic << "' found in instruction set" << std::endl;
            throw std::runtime_error(msg.str());
        }

        const YAML::Node def = it->second;

        instruction ins{};
        ins.mnemonic = mnemonic;

        if (def["opcode"]) {
            const std::string op_s = def["opcode"].as<std::string>();
            ins.opcode = static_cast<std::uint8_t>(parse_string_to_uint(op_s) & 0xFFu);
        } else if (def["op_binary"]) {
            const std::string op_s = def["op_binary"].as<std::string>();
            ins.opcode = static_cast<std::uint8_t>(parse_string_to_uint(op_s) & 0xFFu);
        } else {
            std::stringstream msg;
            msg<<"- error: could not read opcode associated with mnemonic "<<mnemonic<<std::endl;
            throw std::runtime_error(msg.str());
        }

        if (def["encodings"] && def["encodings"].IsSequence()) {
            for (const auto &e : def["encodings"]) {
                encoding enc = parse_encoding(e);
                ins.encodings[enc.start] = enc;
            }
        }

        if (def["arguments"] && def["arguments"].IsSequence()) {
            for (const auto &a : def["arguments"]) {
                argument arg = parse_argument(a);
                ins.arguments[arg.start] = arg;
            }
        }

        validate_and_update_instruction_sizes(ins, opcode_start, max_length);

        out[mnemonic] = std::move(ins);
    }
    return out;
}

std::map<std::string, instruction>
get_instruction_set(const std::string &yaml_file_path,
    const std::uint8_t opcode_start,
    const std::uint8_t max_length) {
    // YAML::Node root = YAML::LoadFile(yaml_file_path);
    auto root = parse_instruction_set_file(yaml_file_path);
    return get_instruction_set(root, opcode_start, max_length);
}

std::map<std::string, instruction>
get_instruction_set(const YAML::Node &yaml_node,
    const instruction_kind kind) {
    return get_instruction_set(yaml_node,
        opcode_start_bit(kind),
        ttdecode::constants::NUM_BITS_PER_INSTRUCTION);
}

std::map<std::string, instruction>
get_instruction_set(const std::string &yaml_file_path,
    const instruction_kind kind)
{
    auto root = parse_instruction_set_file(yaml_file_path);
    return get_instruction_set(root, kind);
}

std::map<std::string, instruction>
get_instruction_set(const instruction_kind kind)
{
    return get_instruction_set(
        get_default_instruction_set_file_path(kind),
        opcode_start_bit(kind),
        ttdecode::constants::NUM_BITS_PER_INSTRUCTION);
}

instruction_sets
get_instruction_sets(const std::map<instruction_kind, YAML::Node> &kinds_yaml_nodes)
{
    instruction_sets sets;
    for (const auto& [kind, node] : kinds_yaml_nodes) {
        sets[kind] = get_instruction_set(node, kind);
    }
    return sets;
}

instruction_sets
get_instruction_sets(const std::map<instruction_kind, std::string> &kinds_file_paths)
{
    instruction_sets sets;
    for (const auto& [kind, file_path] : kinds_file_paths) {
        sets[kind] = get_instruction_set(file_path, kind);
    }
    return sets;
}

instruction_sets
get_instruction_sets(const std::set<instruction_kind> kinds) {
    instruction_sets sets;
    for (const auto& kind : kinds) {
        sets[kind] = get_instruction_set(kind);
    }
    return sets;
}

instruction_sets
get_instruction_sets_incl_rv32(std::map<instruction_kind, std::string> kinds_file_paths)
{
    kinds_file_paths.try_emplace(instruction_kind::rv32, get_default_instruction_set_file_path(instruction_kind::rv32));
    return get_instruction_sets(std::move(kinds_file_paths));
}

} // namespace isa
} // namespace ttdecode
