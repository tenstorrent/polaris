#pragma once

#include "isa/isa.hpp"
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace ttdecode {
namespace decode {

struct registers {
    std::vector<int> integers;
    std::vector<int> floats;

    bool set_integers(const std::vector<int>& vals);
    bool set_integers(const int v);
    bool set_floats(const std::vector<int>& vals);
    bool set_floats(const int v);
    bool empty() const;
};

struct operands {
    // Raw name -> value mapping extracted from the instruction
    std::map<std::string, int> all;
    // Attributes (architecture specific). Optional and kept minimal for now.
    std::map<std::string, int> attributes;
    // Grouped sources and destinations
    registers sources;
    registers destinations;
    // Collected immediates
    std::vector<int> immediates;
    // Decoded argument labels derived from YAML fcov information
    std::map<std::string, std::vector<std::string>> decoded_values;

    void set_all(const std::map<std::string, int>& arg_all, const std::string& mode = "q");
    void set_sources(const registers& r, const std::string& mode = "q");
    void set_destinations(const registers& r, const std::string& mode = "q");

    void set_integer_sources(const std::vector<int>& vals, const std::string& mode = "q");
    void set_integer_sources(const int v, const std::string& mode = "q");
    void set_float_sources(const std::vector<int>& vals, const std::string& mode = "q");
    void set_float_sources(const int v, const std::string& mode = "q");

    void set_integer_destinations(const std::vector<int>& vals, const std::string& mode = "q");
    void set_integer_destinations(const int v, const std::string& mode = "q");
    void set_float_destinations(const std::vector<int>& vals, const std::string& mode = "q");
    void set_float_destinations(const int v, const std::string& mode = "q");

    void set_immediates(const std::vector<int>& vals, const std::string& mode = "q");
    void set_immediates(const int v, const std::string& mode = "q");

    void set_attributes(const std::map<std::string, int>& attrs, const std::string& mode = "q");
    bool empty() const;
};

struct decoded_instruction {
    std::uint32_t word = 0;
    std::optional<std::uint32_t> program_counter;
    std::optional<ttdecode::isa::instruction_kind> kind;
    std::optional<std::uint32_t> opcode;
    std::optional<std::string> mnemonic;
    std::optional<struct operands> operands;

    void set_word(const std::uint32_t w, const std::string& mode = "q");
    void set_program_counter(const std::uint32_t pc, const std::string& mode = "q");
    void set_kind(const ttdecode::isa::instruction_kind k, const std::string& mode = "q");
    void set_opcode(const std::uint32_t op, const std::string& mode = "q");
    void set_mnemonic(const std::string& mnem, const std::string& mode = "q");
    void set_operands(const struct operands& opnds, const std::string& mode = "q");
    std::optional<std::uint32_t> get_program_counter() const;

    // Convenience methods that initialize operands if needed
    void set_all(const std::map<std::string, int>& arg_all, const std::string& mode = "q");
    void set_sources(const registers& r, const std::string& mode = "q");
    void set_destinations(const registers& r, const std::string& mode = "q");
    void set_integer_sources(const std::vector<int>& vals, const std::string& mode = "q");
    void set_integer_sources(const int v, const std::string& mode = "q");
    void set_float_sources(const std::vector<int>& vals, const std::string& mode = "q");
    void set_float_sources(const int v, const std::string& mode = "q");
    void set_integer_destinations(const std::vector<int>& vals, const std::string& mode = "q");
    void set_integer_destinations(const int v, const std::string& mode = "q");
    void set_float_destinations(const std::vector<int>& vals, const std::string& mode = "q");
    void set_float_destinations(const int v, const std::string& mode = "q");
    void set_immediates(const std::vector<int>& vals, const std::string& mode = "q");
    void set_immediates(const int v, const std::string& mode = "q");
    void set_attributes(const std::map<std::string, int>& attrs, const std::string& mode = "q");

    std::string to_string() const;
};

using decoded_instructions = std::vector<decoded_instruction>;

std::ostream&
operator << (std::ostream& os, const operands& op);

std::ostream&
operator << (std::ostream& os, const decoded_instruction& di);

} // namespace decode
} // namespace ttdecode
