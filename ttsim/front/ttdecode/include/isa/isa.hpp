#pragma once

#include "constants/constants.hpp"
#include "isa/yaml_parser.hpp"
#include "yaml-cpp/yaml.h"
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <optional>
#include <vector>

namespace ttdecode {
namespace isa {

enum class instruction_kind {
    rv32,
    ttwh,
    ttbh,
    ttqs,
};

std::string
to_string(instruction_kind kind);

instruction_kind
to_instruction_kind(const std::string& kind);

// Describes a generic argument/field bit slice
struct argument {
    std::string name;
    std::uint8_t start;
    std::uint8_t size;
    // Optional functional coverage specification attached to this argument
    enum class fcov_kind {
        none,
        bins,
        bin_interval,
        boolean
    };
    struct fcov_bin {
        std::string name;
        std::uint32_t value;
    };
    struct fcov_interval {
        std::string name;
        std::uint32_t low;
        std::uint32_t high;
    };
    struct fcov_spec {
        fcov_kind kind = fcov_kind::none;
        std::vector<fcov_bin> bins;
        std::vector<fcov_interval> intervals;
        enum class bin_eval_kind { unknown, bitwise, equality };
        bin_eval_kind bin_eval = bin_eval_kind::unknown;
    };
    std::optional<fcov_spec> fcov;
};

// Describes a fixed-value encoding slice (e.g., funct3/funct7)
struct encoding {
    std::string name;
    std::uint8_t start;
    std::uint8_t size;
    std::uint32_t value; // parsed from binary literal like 0b010
};

struct instruction {
    std::string mnemonic;
    std::uint8_t opcode; // base opcode: 7-bit for RISCV, 8 bit for tensix
    std::map<std::uint8_t, encoding> encodings; // key by start bit for stable ordering
    std::map<std::uint8_t, argument> arguments; // key by start bit
};

using instruction_set = std::map<std::string, instruction>;
using instruction_sets = std::map<instruction_kind, std::map<std::string, instruction>>;

std::string to_string(const argument::fcov_kind kind);
std::string to_string(const instruction_set &iset);

std::set<instruction_kind>
tensix_instruction_kinds();

std::string
get_default_instruction_set_file_path(const instruction_kind kind);

bool
is_tensix(const instruction_kind kind);

std::uint32_t
opcode_start_bit(const instruction_kind kind);

void
validate_and_update_instruction_sizes(instruction &instr,
    const std::uint8_t opcode_start,
    const std::uint8_t max_length = ttdecode::constants::NUM_BITS_PER_INSTRUCTION);

std::map<std::string, instruction>
get_instruction_set(const YAML::Node &node,
    const std::uint8_t opcode_start,
    const std::uint8_t max_length = ttdecode::constants::NUM_BITS_PER_INSTRUCTION);

std::map<std::string, instruction>
get_instruction_set(const std::string &yaml_file_path,
    const std::uint8_t opcode_start,
    const std::uint8_t max_length = ttdecode::constants::NUM_BITS_PER_INSTRUCTION);

std::map<std::string, instruction>
get_instruction_set(const YAML::Node &node,
    const instruction_kind kind);

std::map<std::string, instruction>
get_instruction_set(const std::string &yaml_file_path,
    const instruction_kind kind);

std::map<std::string, instruction>
get_instruction_set(const instruction_kind kind);

instruction_sets
get_instruction_sets(const std::map<instruction_kind, YAML::Node> &kinds_yaml_nodes);

instruction_sets
get_instruction_sets(const std::map<instruction_kind, std::string> &kinds_file_paths);

instruction_sets
get_instruction_sets(const std::set<instruction_kind> kinds);

instruction_sets
get_instruction_sets_incl_rv32(std::map<instruction_kind, std::string> kinds_file_paths);

} // namespace isa
} // namespace ttdecode
