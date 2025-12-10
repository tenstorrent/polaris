#include "isa/defaults.hpp"
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>

namespace ttdecode {
namespace isa {

defaults::defaults() {
    this->reset();
}

const std::map<ttdecode::isa::instruction_kind, std::string>& defaults::instruction_set_file_paths() const {
    return m_instruction_set_file_paths;
}

const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>& defaults::riscv_attributes_instruction_kinds() const {
    return m_riscv_attributes_instruction_kinds;
}

void defaults::update_instruction_set_path(const std::map<ttdecode::isa::instruction_kind, std::string>& other) {
    for (const auto& kv : other) {
        m_instruction_set_file_paths[kv.first] = kv.second;
    }
}

void defaults::update_instruction_set_path(const ttdecode::isa::instruction_kind kind, const std::string& path) {
    m_instruction_set_file_paths[kind] = path;
}

void defaults::append_riscv_attribute(const std::set<ttdecode::isa::instruction_kind>& instruction_kinds, const std::set<std::string>& riscv_attributes) {
    auto& dest = m_riscv_attributes_instruction_kinds[instruction_kinds];
    for (const auto& s : riscv_attributes) dest.insert(s);
}

void defaults::append_riscv_attribute(const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>& riscv_attributes_instruction_kinds) {
    for (const auto& kv : riscv_attributes_instruction_kinds) {
        append_riscv_attribute(kv.first, kv.second);
    }
}

void defaults::append_riscv_attribute(const std::set<ttdecode::isa::instruction_kind>& instruction_kinds, const std::string& riscv_attribute) {
    m_riscv_attributes_instruction_kinds[instruction_kinds].insert(riscv_attribute);
}

void defaults::remove_riscv_attribute(const std::set<ttdecode::isa::instruction_kind>& instruction_kinds, const std::set<std::string>& riscv_attributes) {
    auto it = m_riscv_attributes_instruction_kinds.find(instruction_kinds);
    if (it == m_riscv_attributes_instruction_kinds.end()) return;
    for (const auto& s : riscv_attributes) it->second.erase(s);
    if (it->second.empty()) m_riscv_attributes_instruction_kinds.erase(it);
}

void defaults::remove_riscv_attribute(const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>& riscv_attributes_instruction_kinds) {
    for (const auto& kv : riscv_attributes_instruction_kinds) {
        remove_riscv_attribute(kv.first, kv.second);
    }
}

void defaults::remove_riscv_attribute(const std::set<ttdecode::isa::instruction_kind>& instruction_kinds, const std::string& riscv_attribute) {
    remove_riscv_attribute(instruction_kinds, std::set<std::string>{riscv_attribute});
}

void defaults::reset_instruction_set_file_paths() {
    m_instruction_set_file_paths.clear();
    
    // Get the directory of this source file
    std::filesystem::path source_dir = std::filesystem::path(__FILE__).parent_path();
    
    // Construct paths relative to the source file location
    m_instruction_set_file_paths[ttdecode::isa::instruction_kind::rv32] = (source_dir / "../../../../config/llk/instruction_sets/rv32/assembly.yaml").lexically_normal().string();
    m_instruction_set_file_paths[ttdecode::isa::instruction_kind::ttwh] = (source_dir / "../../../../config/llk/instruction_sets/ttwh/assembly.yaml").lexically_normal().string();
    m_instruction_set_file_paths[ttdecode::isa::instruction_kind::ttbh] = (source_dir / "../../../../config/llk/instruction_sets/ttbh/assembly.yaml").lexically_normal().string();
    m_instruction_set_file_paths[ttdecode::isa::instruction_kind::ttqs] = (source_dir / "../../../../config/llk/instruction_sets/ttqs/assembly.yaml").lexically_normal().string();
}

void defaults::reset_riscv_attributes_instruction_kinds() {
    m_riscv_attributes_instruction_kinds.clear();
    m_riscv_attributes_instruction_kinds[std::set<ttdecode::isa::instruction_kind>{ttdecode::isa::instruction_kind::rv32, ttdecode::isa::instruction_kind::ttwh}] = std::set<std::string>{"riscv_rv32i2p0_m2p0_xttwh1p0", "riscv_rv32i2p0_m2p0_zmmul1p0_xttwh1p0"};
    m_riscv_attributes_instruction_kinds[std::set<ttdecode::isa::instruction_kind>{ttdecode::isa::instruction_kind::rv32, ttdecode::isa::instruction_kind::ttbh}] = std::set<std::string>{"riscv_rv32i2p0_m2p0_xttbh1p0"};
    m_riscv_attributes_instruction_kinds[std::set<ttdecode::isa::instruction_kind>{ttdecode::isa::instruction_kind::rv32, ttdecode::isa::instruction_kind::ttqs}] = std::set<std::string>{"riscv_rv32i2p0_m2p0_a2p0_f2p0_v1p0_zfh0p1_zvamo1p0_zvlsseg1p0"};
}

void defaults::reset() {
    reset_instruction_set_file_paths();
    reset_riscv_attributes_instruction_kinds();
}

defaults& global_defaults() {
    static defaults d;
    return d;
}

} // namespace isa
} // namespace ttdecode
