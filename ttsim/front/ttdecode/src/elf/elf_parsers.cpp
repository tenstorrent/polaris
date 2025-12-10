#include "elf/elf_parsers.hpp"

namespace ttdecode {
namespace elf {

// In the constructor implementation (.cpp file)
parsers::parsers(const std::vector<std::string>& file_names)
    : m_parsers()  // Start with empty vector
{
    m_parsers.reserve(file_names.size());  // Pre-allocate for efficiency
    for (const auto& filename : file_names) {
        m_parsers.emplace_back(filename);  // Construct parser in-place
    }
}

parser&
parsers::at(const std::size_t idx) {
    if (idx >= m_parsers.size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_parsers.at(idx);
}

const parser&
parsers::at(const std::size_t idx) const {
    if (idx >= m_parsers.size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_parsers.at(idx);
}

parser&
parsers::operator [] (const std::size_t idx) {
    if (idx >= m_parsers.size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_parsers[idx];
}

const parser&
parsers::operator [] (const std::size_t idx) const {
    if (idx >= m_parsers.size()) {
        throw std::out_of_range("Index out of range");
    }
    return m_parsers[idx];
}

std::size_t parsers::size() const {
    return m_parsers.size();
}

std::set<ttdecode::isa::instruction_kind>
parsers::get_instruction_kinds(const std::string& mode) const {
    if ("merged" == mode) {
        std::set<ttdecode::isa::instruction_kind> combined_kinds;

        for (const auto& m_parser : m_parsers) {
            for (const auto &kind : m_parser.get_instruction_kinds()) {
                combined_kinds.insert(kind);
            }
        }

        return combined_kinds;
    }
    if ("common" == mode) {
        std::set<ttdecode::isa::instruction_kind> common_kinds;
        if (m_parsers.empty()) {
            return common_kinds;
        }
        common_kinds = m_parsers[0].get_instruction_kinds();
        for (std::size_t i = 1; i < m_parsers.size(); ++i) {
            std::set<ttdecode::isa::instruction_kind> next_common;
            const auto kinds_i = m_parsers[i].get_instruction_kinds();
            for (const auto &k : common_kinds) {
                if (kinds_i.count(k)) {
                    next_common.insert(k);
                }
            }
            common_kinds.swap(next_common);
            if (common_kinds.empty()) {
                break;
            }
        }
        return common_kinds;
    }

    throw std::invalid_argument("- error: function 'parsers::get_instruction_kinds(std::string)' supports modes 'merged' and 'common' only.");
}

bool
parsers::instruction_kinds_match_for_all_elfs() const {
    if (m_parsers.size()) {
        const auto kinds = m_parsers[0].get_instruction_kinds();
        for (std::size_t i = 1; i < m_parsers.size(); ++i) {
            if (kinds != m_parsers[i].get_instruction_kinds()) {
                return false;
            }
        }
    }

    return true;
}

std::string
parsers::get_tensix_architecture() const {
    // Verify all ELF files have consistent instruction kinds
    if (not this->instruction_kinds_match_for_all_elfs()) {
        throw std::runtime_error("get_tensix_architecture: Instruction kinds do not match across all ELF files. "
                                 "All ELF files must contain the same set of instruction kinds.");
    }

    // Get merged set of instruction kinds from all parsers
    const auto instruction_kinds = this->get_instruction_kinds("merged");

    // Expected: exactly one Tensix ISA variant and one RISC-V ISA
    if (instruction_kinds.size() != 2) {
        throw std::runtime_error("get_tensix_architecture: Expected exactly 2 instruction kinds (1 Tensix + 1 RISC-V), "
                                 "but found " + std::to_string(instruction_kinds.size()) + " kinds.");
    }

    // Count Tensix and RISC-V instruction kinds
    std::uint8_t num_tensix = 0;
    std::uint8_t num_riscv = 0;

    for (const auto& kind : instruction_kinds) {
        if (ttdecode::isa::is_tensix(kind)) {
            num_tensix++;
        }
        else {
            num_riscv++;
        }
    }

    // Validate exactly one Tensix architecture variant
    if (num_tensix != 1) {
        throw std::runtime_error("get_tensix_architecture: Expected exactly 1 Tensix instruction kind, "
                                 "but found " + std::to_string(num_tensix) + ". "
                                 "Cannot determine unique Tensix architecture.");
    }

    // Validate exactly one RISC-V architecture
    if (num_riscv != 1) {
        throw std::runtime_error("get_tensix_architecture: Expected exactly 1 RISC-V instruction kind, "
                                 "but found " + std::to_string(num_riscv) + ".");
    }

    // Find and return the Tensix architecture string (e.g., "ttwh", "ttbh", "ttqs")
    for (const auto& kind : instruction_kinds) {
        if (ttdecode::isa::is_tensix(kind)) {
            return ttdecode::isa::to_string(kind);
        }
    }

    // Should never reach here due to validation above
    throw std::logic_error("get_tensix_architecture: Internal error - no Tensix instruction kind found after validation.");
}

} // namespace elf
} // namespace ttdecode
