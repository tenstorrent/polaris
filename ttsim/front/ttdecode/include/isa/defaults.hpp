#pragma once
#include "isa/isa.hpp"
#include <map>
#include <set>
#include <string>

namespace ttdecode {
namespace isa {

class defaults {
public:
    defaults();

    const std::map<ttdecode::isa::instruction_kind, std::string>& instruction_set_file_paths() const;
    const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>& riscv_attributes_instruction_kinds() const;

    void update_instruction_set_path(const std::map<ttdecode::isa::instruction_kind, std::string>& other);
    void update_instruction_set_path(const ttdecode::isa::instruction_kind kind, const std::string& path);

    void append_riscv_attribute(const std::set<ttdecode::isa::instruction_kind>& instruction_kinds, const std::set<std::string>& riscv_attributes);
    void append_riscv_attribute(const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>& riscv_attributes_instruction_kinds);
    void append_riscv_attribute(const std::set<ttdecode::isa::instruction_kind>& instruction_kinds, const std::string& riscv_attribute);

    void remove_riscv_attribute(const std::set<ttdecode::isa::instruction_kind>& instruction_kinds, const std::set<std::string>& riscv_attributes);
    void remove_riscv_attribute(const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>& riscv_attributes_instruction_kinds);
    void remove_riscv_attribute(const std::set<ttdecode::isa::instruction_kind>& instruction_kinds, const std::string& riscv_attribute);

    void reset_riscv_attributes_instruction_kinds();
    void reset_instruction_set_file_paths();
    void reset();

private:
    std::map<ttdecode::isa::instruction_kind, std::string> m_instruction_set_file_paths;
    std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>> m_riscv_attributes_instruction_kinds;
};

defaults& global_defaults();

} // namespace isa
} // namespace ttdecode
