#pragma once

#include "elf/elf_parser.hpp"
#include <set>
#include <vector>

namespace ttdecode {
namespace elf {

class parsers {
public:
    explicit parsers(const std::vector<std::string>& file_names);

    parser&
    at(const std::size_t idx);

    const parser&
    at(const std::size_t idx) const;

    parser&
    operator [] (const std::size_t idx);

    const parser&
    operator [] (const std::size_t idx) const;

    std::size_t size() const;

    std::set<ttdecode::isa::instruction_kind>
    get_instruction_kinds(const std::string& mode = "merged") const;

    bool
    instruction_kinds_match_for_all_elfs() const;

    std::string
    get_tensix_architecture() const;

private:
    std::vector<parser> m_parsers;
};

} // namespace elf
} // namespace ttdecode
