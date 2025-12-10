#pragma once

#include "decode/decode.hpp"
#include "decode/decoded_instruction.hpp"
#include "isa/isa.hpp"
#include <concepts>
#include <cstdint>
#include <ranges>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace ttdecode {
namespace elf {

// Public-facing struct for a section
struct section {
    std::string   name;
    std::uint64_t type;
    std::uint64_t flags;
    std::uint64_t addr;
    std::uint64_t offset;
    std::uint64_t size;
};

bool
operator == (const section& a, const section& b);

bool
operator != (const section& a, const section& b);

// Public-facing struct for a function symbol
struct function_symbol {
    std::string   name;
    std::uint64_t value; // Virtual address
    std::uint64_t size;
    std::uint16_t section_index;
};

bool
operator < (const function_symbol &a, const function_symbol &b);

// Public-facing struct for RISC-V attributes
struct riscv_attribute {
    std::uint8_t tag;
    std::string name;
    std::string value; // For string attributes
    std::uint64_t numeric_value; // For numeric attributes
    bool is_numeric;
};

// Public-facing struct for RISC-V vendor information
struct riscv_attributes {
    std::string vendor_name;
    std::vector<riscv_attribute> attributes;

    const riscv_attribute&
    get_riscv_attribute_with_tag(const std::uint8_t tag) const;

    bool
    has_riscv_attribute_with_tag(const std::uint8_t tag) const;
};

// RISC-V attribute tags (commonly used ones)
enum class riscv_attr_tag : std::uint8_t {
    ARCH = 5,              // Architecture string
    PRIV_SPEC = 8,         // Privileged spec version
    PRIV_SPEC_MINOR = 10,  // Privileged spec minor version
    UNALIGNED_ACCESS = 6,  // Unaligned access support
    STACK_ALIGN = 4        // Stack alignment
};

class parser {
public:
    // Construct from a file path
    explicit parser(const std::string& file_path);
    // Construct from an in-memory buffer
    explicit parser(std::vector<uint8_t> data);

    // Basic validation and metadata
    bool is_valid() const;
    bool is_64_bit() const;
    std::string get_class() const;
    std::string get_data() const; // Endianness
    std::string get_type() const;

    // Accessors for parsed data
    const section& get_section(const std::string& sec_name) const;
    const section& get_section(const std::size_t section_idx) const;
    const std::vector<section>& get_sections() const;
    std::uint16_t get_section_index(const section& sec) const;
    function_symbol get_function(const std::string& func_name) const;
    std::vector<function_symbol> get_functions() const;
    std::vector<function_symbol> get_functions(const section &sec) const;

    // API to get the instruction stream for a function
    std::vector<uint8_t> get_bytes(const function_symbol& func) const;

    // API to get the raw bytes from a section
    std::vector<uint8_t> get_bytes(const section& sec) const;

    // API to get RISC-V attributes from .riscv.attributes section
    riscv_attributes get_riscv_attributes() const;

    std::string riscv_attribute_to_string() const;

    std::set<ttdecode::isa::instruction_kind>
    get_instruction_kinds() const;

    template <typename Sym_t>
    std::vector<function_symbol> get_functions_internal() const;

    // one function
    ttdecode::decode::decoded_instructions
    decode(const function_symbol &fun_sym, const ttdecode::isa::instruction_sets &sets) const;

    ttdecode::decode::decoded_instructions
    decode(const function_symbol &fun_sym,
        const std::map<ttdecode::isa::instruction_kind, std::string> &instruction_sets) const;

    ttdecode::decode::decoded_instructions
    decode(const function_symbol &fun_sym) const;

    // vector of functions
    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode(const std::vector<function_symbol> &functions, const ttdecode::isa::instruction_sets &sets) const;

    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode(const std::vector<function_symbol> &functions,
        const std::map<ttdecode::isa::instruction_kind, std::string> &instruction_sets) const;

    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode(const std::vector<function_symbol> &functions) const;

    // section
    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode(const section &sec, const ttdecode::isa::instruction_sets &sets) const;

    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode(const section &sec, const std::map<ttdecode::isa::instruction_kind, std::string> &kinds_file_paths) const;

    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode(const section &sec) const;

    // all functions
    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode(const ttdecode::isa::instruction_sets &sets) const;

    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode(const std::map<ttdecode::isa::instruction_kind, std::string> &kinds_file_paths) const;

    std::map<function_symbol, ttdecode::decode::decoded_instructions>
    decode() const;

    const std::string&
    file_path() const;

private:
    // Private implementation details
    void parse();
    // Templated parsing logic to handle 32-bit and 64-bit
    template <typename Ehdr_t, typename Shdr_t, typename Sym_t>
    void parse_internal();

    // Helper method to parse RISC-V attributes from section data
    riscv_attributes parse_riscv_attributes_section(const std::vector<uint8_t>& section_data) const;

    std::vector<uint8_t> m_data;
    bool m_is_valid = false;
    bool m_is_64_bit = false;
    bool m_is_little_endian = false;
    std::uint16_t m_type;
    std::string m_file_path;
    std::vector<section> m_sections;
};

ttdecode::decode::decoded_instructions
decode_function(const std::string &function_name, const std::string &elf_file_path);

std::map<function_symbol, ttdecode::decode::decoded_instructions>
decode_section(const std::string &section_name, const std::string &elf_file_path);

} // namespace elf
} // namespace ttdecode
