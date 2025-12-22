#include "elf/elf_parser.hpp"
#include "isa/defaults.hpp"
#include <algorithm> // For std::copy
#include <fstream>
#include <gelf.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

// --- Private Helper Functions ---

// Generic helper to read a struct from the data buffer at a given offset.
// This prevents repeated reinterpret_casts and improves readability.
template <typename T>
const T* get_struct_at(const std::vector<uint8_t>& data, uint64_t offset) {
    if (offset + sizeof(T) > data.size()) {
        return nullptr;
    }
    return reinterpret_cast<const T*>(&data[offset]);
}

// Handles endianness conversion for multi-byte fields.
// It's a no-op if the file's endianness matches the host's.
template <typename T>
T
swap_if_required(T value, bool should_swap) {
    if (!should_swap) {
        return value;
    }
    T result;
    char* src = reinterpret_cast<char*>(&value);
    char* dst = reinterpret_cast<char*>(&result);
    for (size_t i = 0; i < sizeof(T); ++i) {
        dst[i] = src[sizeof(T) - 1 - i];
    }
    return result;
}

namespace ttdecode {
namespace elf {

bool
operator < (const function_symbol &a, const function_symbol &b) {
    return std::tie(a.value, a.size, a.section_index, a.name) <
           std::tie(b.value, b.size, b.section_index, b.name);
}

bool
operator == (const section& a, const section& b) {
    return a.name   == b.name &&
           a.type   == b.type &&
           a.flags  == b.flags &&
           a.addr   == b.addr &&
           a.offset == b.offset &&
           a.size   == b.size;
}

bool
operator != (const section& a, const section& b) {
    return !(a == b);
}

const riscv_attribute&
riscv_attributes::get_riscv_attribute_with_tag(const std::uint8_t tag) const {
    for (const auto& attr : this->attributes) {
        if (tag == attr.tag) {
            return attr;
        }
    }

    throw std::invalid_argument("- error: could not find riscv attribute with tag");
}

bool
riscv_attributes::has_riscv_attribute_with_tag(const std::uint8_t tag) const {
    for (const auto& attr : this->attributes) {
        if (tag == attr.tag) {
            return true;
        }
    }
    return false;
}

// --- Constructor Implementations ---

parser::parser(const std::string& file_path)
    : m_file_path(file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    m_data.resize(size);
    if (!file.read(reinterpret_cast<char*>(m_data.data()), size)) {
        throw std::runtime_error("Failed to read file: " + file_path);
    }

    parse();
}

parser::parser(std::vector<uint8_t> data) : m_data(std::move(data)) {
    parse();
}


// --- Public API Implementations ---

bool parser::is_valid() const { return m_is_valid; }
bool parser::is_64_bit() const { return m_is_64_bit; }

const section&
parser::get_section(const std::string& section_name) const {
    for (const auto& sec : m_sections) {
        if (sec.name == section_name) {
            return sec;
        }
    }

    std::stringstream msg;
    msg << "- error: section not found\n";
    msg << "- Requested section: '" << section_name << "'\n";
    msg << "- ELF file: " << this->file_path() << "\n";
    msg << "- Available sections (" << m_sections.size() << "):" << "\n";
    for (const auto& sec : m_sections) {
        msg << "  - name='" << sec.name << "'"
            << ", type=" << sec.type
            << ", addr=0x" << std::hex << sec.addr << std::dec
            << ", offset=" << sec.offset
            << ", size=" << sec.size << "\n";
    }
    msg << "- Hint: verify the section exists and spelling/case matches.\n";
    msg << "- If the ELF is stripped, expected sections may be missing.";

    throw std::invalid_argument(msg.str());
}

std::uint16_t
parser::get_section_index(const section& sec) const {
    for (std::uint16_t i = 0; i < m_sections.size(); ++i) {
        if (sec == m_sections[i]) {
            return i;
        }
    }

    std::stringstream msg;
    msg << "- error: section index not found\n";
    msg << "- ELF file: " << this->file_path() << "\n";
    msg << "- Target section details:\n";
    msg << "  name='" << sec.name << "'"
        << ", type=" << sec.type
        << ", addr=0x" << std::hex << sec.addr << std::dec
        << ", offset=" << sec.offset
        << ", size=" << sec.size << "\n";
    msg << "- Available sections (" << m_sections.size() << "):\n";
    for (std::size_t i = 0; i < m_sections.size(); ++i) {
        const auto& s = m_sections[i];
        msg << "  [" << i << "] name='" << s.name << "'"
            << ", type=" << s.type
            << ", addr=0x" << std::hex << s.addr << std::dec
            << ", offset=" << s.offset
            << ", size=" << s.size << "\n";
    }
    msg << "- Hint: ensure you are comparing the same parsed section object.\n";
    msg << "  If sections were reconstructed, prefer lookup by name via get_section(name).";

    throw std::invalid_argument(msg.str());
}

const section&
parser::get_section(const std::size_t section_idx) const {
    if (section_idx < m_sections.size()) {
        return m_sections[section_idx];
    }

    std::stringstream msg;
    msg << "- error: section index out of range\n";
    msg << "- ELF file: " << this->file_path() << "\n";
    msg << "- Requested index: " << section_idx << "\n";
    msg << "- Number of available sections: " << m_sections.size();
    throw std::out_of_range(msg.str());
}

const std::vector<section>& parser::get_sections() const { return m_sections; }

std::string parser::get_class() const {
    return m_is_valid? (m_is_64_bit? "ELF64" : "ELF32") : "Invalid";
}

std::string parser::get_data() const {
    if (!m_is_valid) return "Invalid";
    // This check determines the host endianness to compare against the file's.
    uint32_t i = 1;
    bool host_is_little = *reinterpret_cast<char*>(&i) == 1;
    if (m_is_little_endian == host_is_little) {
        return m_is_little_endian? "Little Endian (Host)" : "Big Endian (Host)";
    }
    return m_is_little_endian? "Little Endian" : "Big Endian";
}

std::string parser::get_type() const {
    if (!m_is_valid) return "Invalid";
    switch (m_type) {
        case ET_NONE: return "No file type";
        case ET_REL:  return "Relocatable file";
        case ET_EXEC: return "Executable file";
        case ET_DYN:  return "Shared object file";
        case ET_CORE: return "Core file";
        default:      return "Unknown type";
    }
}

const std::string&
parser::file_path() const {
    return m_file_path;
}


// --- Core Parsing Logic ---

void parser::parse() {
    if (m_data.size() < EI_NIDENT) {
        return; // Not enough data for the identification array
    }

    // 1. Validate Magic Number
    const unsigned char* ident = m_data.data();
    if (ident[EI_MAG0]!= ELFMAG0 || ident[EI_MAG1]!= ELFMAG1 || ident[EI_MAG2]!= ELFMAG2 || ident[EI_MAG3]!= ELFMAG3) {
        return; // Invalid magic number
    }

    // 2. Determine Class (32/64-bit) and Endianness
    if (ident[EI_CLASS] == ELFCLASS64) {
        m_is_64_bit = true;
    } else if (ident[EI_CLASS] == ELFCLASS32) {
        m_is_64_bit = false;
    } else {
        return; // Unknown class
    }

    if (ident[EI_DATA] == ELFDATA2LSB) {
        m_is_little_endian = true;
    } else if (ident[EI_DATA] == ELFDATA2MSB) {
        m_is_little_endian = false;
    } else {
        return; // Unknown data format
    }

    m_is_valid = true;

    // 3. Dispatch to the appropriate templated parser
    if (m_is_64_bit) {
        parse_internal<Elf64_Ehdr, Elf64_Shdr, Elf64_Sym>();
    } else {
        parse_internal<Elf32_Ehdr, Elf32_Shdr, Elf32_Sym>();
    }
}

template <typename Ehdr_t, typename Shdr_t, typename Sym_t>
void parser::parse_internal() {
    // Determine if byte swapping is needed
    uint32_t i = 1;
    bool host_is_little = *reinterpret_cast<char*>(&i) == 1;
    bool should_swap = (m_is_little_endian!= host_is_little);

    // 1. Map and read the ELF Header
    const Ehdr_t* ehdr = get_struct_at<Ehdr_t>(m_data, 0);
    if (!ehdr) {
        m_is_valid = false;
        return;
    }

    m_type = swap_if_required(ehdr->e_type, should_swap);
    uint64_t shoff = swap_if_required(ehdr->e_shoff, should_swap);
    uint16_t shentsize = swap_if_required(ehdr->e_shentsize, should_swap);
    uint16_t shnum = swap_if_required(ehdr->e_shnum, should_swap);
    uint16_t shstrndx = swap_if_required(ehdr->e_shstrndx, should_swap);

    // Handle extended section numbering [1]
    if (shnum == 0 && shoff!= 0) {
        const Shdr_t* first_shdr = get_struct_at<Shdr_t>(m_data, shoff);
        if (first_shdr) {
            shnum = swap_if_required(first_shdr->sh_size, should_swap);
        }
    }

    // 2. Locate the Section Name String Table (.shstrtab)
    if (shstrndx >= shnum) {
        // Invalid index for the section name string table
        return;
    }
    const Shdr_t* shstr_shdr = get_struct_at<Shdr_t>(m_data, shoff + shstrndx * shentsize);
    if (!shstr_shdr) return;

    uint64_t shstr_offset = swap_if_required(shstr_shdr->sh_offset, should_swap);
    const char* shstrtab = reinterpret_cast<const char*>(&m_data[shstr_offset]);

    // 3. Iterate through all section headers and populate the sections vector
    for (uint16_t i = 0; i < shnum; ++i) {
        const Shdr_t* shdr = get_struct_at<Shdr_t>(m_data, shoff + i * shentsize);
        if (!shdr) continue;

        section sec;
        uint32_t name_offset = swap_if_required(shdr->sh_name, should_swap);
        sec.name = std::string(shstrtab + name_offset);
        sec.type = swap_if_required(shdr->sh_type, should_swap);
        sec.flags = swap_if_required(shdr->sh_flags, should_swap);
        sec.addr = swap_if_required(shdr->sh_addr, should_swap);
        sec.offset = swap_if_required(shdr->sh_offset, should_swap);
        sec.size = swap_if_required(shdr->sh_size, should_swap);

        m_sections.push_back(sec);
    }
}

// --- On-Demand Symbol Parsing ---

template <typename Sym_t>
std::vector<function_symbol> parser::get_functions_internal() const {
    std::vector<function_symbol> functions;
    std::uint32_t one = 1;
    bool host_is_little = *reinterpret_cast<char*>(&one) == 1;
    bool should_swap = (m_is_little_endian!= host_is_little);

    // Define the appropriate ELF header type based on Sym_t
    using Ehdr_t = std::conditional_t<std::is_same_v<Sym_t, Elf64_Sym>, Elf64_Ehdr, Elf32_Ehdr>;
    using Shdr_t = std::conditional_t<std::is_same_v<Sym_t, Elf64_Sym>, Elf64_Shdr, Elf32_Shdr>;

    for (uint16_t i = 0; i < m_sections.size(); ++i) {
        const auto& sec = m_sections[i];
        // Find symbol tables (.symtab or.dynsym) [1]
        if (sec.type == SHT_SYMTAB || sec.type == SHT_DYNSYM) {
            // The section header for a symbol table points to its string table via sh_link
            const Ehdr_t* ehdr = get_struct_at<Ehdr_t>(m_data, 0);
            if (!ehdr) continue;
            const Shdr_t* symtab_shdr = get_struct_at<Shdr_t>(m_data, swap_if_required(ehdr->e_shoff, should_swap) + i * swap_if_required(ehdr->e_shentsize, should_swap));
            uint32_t strtab_idx = swap_if_required(symtab_shdr->sh_link, should_swap);

            if (strtab_idx >= m_sections.size()) continue;
            const auto& strtab_sec = m_sections[strtab_idx];
            const char* strtab = reinterpret_cast<const char*>(&m_data[strtab_sec.offset]);

            const Sym_t* symbols = reinterpret_cast<const Sym_t*>(&m_data[sec.offset]);
            size_t num_symbols = sec.size / sizeof(Sym_t);

            for (size_t j = 0; j < num_symbols; ++j) {
                const Sym_t* sym = &symbols[j];
                // Check if the symbol type is a function [2, 3]
                if (ELF64_ST_TYPE(sym->st_info) == STT_FUNC) {
                    function_symbol func;
                    uint32_t name_offset = swap_if_required(sym->st_name, should_swap);
                    func.name = std::string(strtab + name_offset);
                    func.value = swap_if_required(sym->st_value, should_swap);
                    func.size = swap_if_required(sym->st_size, should_swap);
                    func.section_index = swap_if_required(sym->st_shndx, should_swap);
                    functions.push_back(func);
                }
            }
        }
    }
    return functions;
}

function_symbol
parser::get_function(const std::string& function_name) const {
    const auto all_funcs = this->get_functions();
    for (const auto& func : all_funcs) {
        if (func.name == function_name) {
            return func;
        }
    }

    std::stringstream msg;
    msg << "- error: function symbol not found\n";
    msg << "- Requested function: '" << function_name << "'\n";
    msg << "- ELF file: " << this->file_path() << "\n";
    msg << "- Total functions detected: " << all_funcs.size() << "\n";
    msg << "- Available function symbols (name, section, size, value):\n";
    for (const auto& f : all_funcs) {
        std::string sec_name = (f.section_index < m_sections.size()) ? m_sections[f.section_index].name : std::string("<invalid>");
        msg << "  - '" << f.name << "'"
            << ", section='" << sec_name << "'"
            << ", size=" << f.size
            << ", value=0x" << std::hex << f.value << std::dec << "\n";
    }
    msg << "- Hint: ensure symbols are not stripped and the function name matches exactly.\n";

    throw std::invalid_argument(msg.str());
}

std::vector<function_symbol> parser::get_functions() const {
    if (!m_is_valid) return {};
    if (m_is_64_bit) {
        return get_functions_internal<Elf64_Sym>();
    } else {
        return get_functions_internal<Elf32_Sym>();
    }
}

std::vector<function_symbol> parser::get_functions(const section& sec) const {
    std::vector<function_symbol> all_funcs = get_functions();
    std::vector<function_symbol> filtered_funcs;
    for (const auto& func : all_funcs) {
        if (func.section_index < m_sections.size() && m_sections[func.section_index] == sec) {
            filtered_funcs.push_back(func);
        }
    }
    return filtered_funcs;
}

std::vector<uint8_t> parser::get_bytes(const function_symbol& func) const {
    if (!m_is_valid || func.section_index >= m_sections.size() || func.size == 0) {
        return {};
    }

    const section& sec = m_sections[func.section_index];

    // A function's address (st_value) is a virtual address. We need to convert it
    // to a file offset to read the raw bytes.
    // File Offset = Section File Offset + (Function Virtual Address - Section Virtual Address)
    uint64_t offset_in_section = func.value - sec.addr;
    uint64_t file_offset = sec.offset + offset_in_section;

    if (file_offset + func.size > m_data.size()) {
        // The calculated range is out of bounds of the file data.
        return {};
    }

    std::vector<uint8_t> instructions(func.size);
    std::copy(m_data.begin() + file_offset, m_data.begin() + file_offset + func.size, instructions.begin());

    return instructions;
}

std::vector<uint8_t> parser::get_bytes(const section& sec) const {
    if (!m_is_valid || sec.size == 0) {
        return {};
    }

    // Check if the section's file offset and size are within bounds
    if (sec.offset + sec.size > m_data.size()) {
        return {};
    }

    std::vector<uint8_t> section_data(sec.size);
    std::copy(m_data.begin() + sec.offset, m_data.begin() + sec.offset + sec.size, section_data.begin());

    return section_data;
}

riscv_attributes parser::get_riscv_attributes() const {
    if (!m_is_valid) {
        return {};
    }

    // Find the .riscv.attributes section
    for (const auto& sec : m_sections) {
        if (sec.name == ".riscv.attributes") {
            // Extract the section data using the get_bytes method
            std::vector<uint8_t> section_data = get_bytes(sec);
            if (section_data.empty()) {
                return {}; // Section data is invalid or empty
            }

            return parse_riscv_attributes_section(section_data);
        }
    }

    return riscv_attributes(); // No .riscv.attributes section found
}

riscv_attributes parser::parse_riscv_attributes_section(const std::vector<uint8_t>& section_data) const {
    riscv_attributes attributes;

    if (section_data.empty()) {
        return attributes;
    }

    size_t offset = 0;

    // RISC-V attributes format:
    // - Format version (1 byte): usually 'A' (0x41)
    // - Section length (4 bytes): length of the entire section
    // - Vendor name (null-terminated string)
    // - Subsection tag (1 byte): usually 1 for File Attributes
    // - Subsection length (4 bytes)
    // - Attributes (tag-value pairs)

    if (offset >= section_data.size()) return attributes;

    // Check format version
    uint8_t format_version = section_data[offset++];
    if (format_version != 'A') {
        return attributes; // Unexpected format
    }

    if (offset + 4 > section_data.size()) return attributes;

    // Read section length (little-endian)
    uint32_t section_length = 0;
    if (m_is_little_endian) {
        section_length = *reinterpret_cast<const uint32_t*>(&section_data[offset]);
    } else {
        section_length = swap_if_required(*reinterpret_cast<const uint32_t*>(&section_data[offset]), true);
    }
    offset += 4;

    // Read vendor name
    while (offset < section_data.size() && section_data[offset] != 0) {
        attributes.vendor_name += static_cast<char>(section_data[offset++]);
    }
    if (offset < section_data.size()) offset++; // Skip null terminator

    // Process subsections
    while (offset < section_data.size()) {
        if (offset >= section_data.size()) break;

        uint8_t subsection_tag = section_data[offset++];

        if (offset + 4 > section_data.size()) break;

        uint32_t subsection_length = 0;
        if (m_is_little_endian) {
            subsection_length = *reinterpret_cast<const uint32_t*>(&section_data[offset]);
        } else {
            subsection_length = swap_if_required(*reinterpret_cast<const uint32_t*>(&section_data[offset]), true);
        }
        offset += 4;

        size_t subsection_end = offset + subsection_length - 5; // -5 for tag and length

        // Parse attributes within this subsection
        while (offset < subsection_end && offset < section_data.size()) {
            uint8_t attr_tag = section_data[offset++];

            riscv_attribute attr;
            attr.tag = attr_tag;

            // Set attribute name based on common RISC-V attribute tags
            switch (static_cast<riscv_attr_tag>(attr_tag)) {
                case riscv_attr_tag::ARCH:
                    attr.name = "ARCH";
                    break;
                case riscv_attr_tag::PRIV_SPEC:
                    attr.name = "PRIV_SPEC";
                    break;
                case riscv_attr_tag::PRIV_SPEC_MINOR:
                    attr.name = "PRIV_SPEC_MINOR";
                    break;
                case riscv_attr_tag::UNALIGNED_ACCESS:
                    attr.name = "UNALIGNED_ACCESS";
                    break;
                case riscv_attr_tag::STACK_ALIGN:
                    attr.name = "STACK_ALIGN";
                    break;
                default:
                    attr.name = "UNKNOWN_" + std::to_string(attr_tag);
                    break;
            }

            // Determine if this is a string or numeric attribute
            // Common numeric attributes: PRIV_SPEC, PRIV_SPEC_MINOR, UNALIGNED_ACCESS, STACK_ALIGN
            // Common string attributes: ARCH
            if (attr_tag == static_cast<uint8_t>(riscv_attr_tag::ARCH)) {
                // String attribute - read null-terminated string
                attr.is_numeric = false;
                std::string str_value;
                while (offset < section_data.size() && section_data[offset] != 0) {
                    str_value += static_cast<char>(section_data[offset++]);
                }
                if (offset < section_data.size()) offset++; // Skip null terminator
                attr.value = str_value;
                attr.numeric_value = 0;
            } else {
                // Numeric attribute - read ULEB128 encoded value
                attr.is_numeric = true;
                attr.numeric_value = 0;
                uint64_t shift = 0;
                uint8_t byte;
                do {
                    if (offset >= section_data.size()) break;
                    byte = section_data[offset++];
                    attr.numeric_value |= static_cast<uint64_t>(byte & 0x7F) << shift;
                    shift += 7;
                } while ((byte & 0x80) != 0 && shift < 64);
                attr.value = std::to_string(attr.numeric_value);
            }

            attributes.attributes.push_back(attr);
        }

        offset = subsection_end;
    }

    return attributes;
}

std::string
parser::riscv_attribute_to_string() const
{
    const auto riscv_attrs = this->get_riscv_attributes();
    if (not riscv_attrs.has_riscv_attribute_with_tag(static_cast<std::uint8_t>(riscv_attr_tag::ARCH))) {
        std::stringstream msg;
        msg << "- error: missing required RISC-V ARCH attribute\n";
        msg << "- ELF file: " << this->file_path() << "\n";
        msg << "- Details: .riscv.attributes section present? "
            << (riscv_attrs.attributes.empty() ? "no or empty" : "yes") << "\n";
        msg << "- Hint: ensure the ELF contains .riscv.attributes with ARCH string (e.g., rv32i2p0_m2p0_...).";
        throw std::invalid_argument(msg.str());
    }
    const auto& arch = riscv_attrs.get_riscv_attribute_with_tag(static_cast<std::uint8_t>(riscv_attr_tag::ARCH));
    if (arch.is_numeric) {
        std::stringstream msg;
        msg << "- error: ARCH attribute is numeric but string expected\n";
        msg << "- ELF file: " << this->file_path() << "\n";
        msg << "- Parsed numeric ARCH value: " << arch.numeric_value << "\n";
        msg << "- Hint: ARCH must be a string like 'rv32i2p0_m2p0_...'.";
        throw std::invalid_argument(msg.str());
    }
    std::stringstream msg;
    msg<<riscv_attrs.vendor_name<<"_"<<arch.value;
    return msg.str();
}

std::set<ttdecode::isa::instruction_kind>
parser::get_instruction_kinds() const {
    const auto riscv_attribute_str = this->riscv_attribute_to_string();
    const auto& mapping = ttdecode::isa::global_defaults().riscv_attributes_instruction_kinds();
    for (const auto& kv : mapping) {
        const auto& kinds = kv.first;
        const auto& attrs = kv.second;
        if (attrs.count(riscv_attribute_str)) return kinds;
    }
    std::stringstream msg;
    msg << "- error: could not determine instruction kinds from the given RISC-V attributes.\n";
    msg << "- ELF file: " << this->file_path() << "\n";
    msg << "- riscv attribute (vendor_name_arch_string): '" << riscv_attribute_str << "'\n";
    msg << "- Searched for matches in:\n";
    for (const auto& kv : mapping) {
        msg << "  - instruction kinds: {";
        for (const auto& kind : kv.first) msg << " " << ttdecode::isa::to_string(kind);
        msg << " }, arch strings: [";
        for (const auto& s : kv.second) msg << " '" << s << "'";
        msg << " ]\n";
    }
    msg << "- Update ttdecode::isa::global_defaults() to add required mappings.\n";
    throw std::invalid_argument(msg.str());
}

ttdecode::decode::decoded_instructions
parser::decode(const function_symbol &fun_sym, const ttdecode::isa::instruction_sets &sets) const
{
    const auto kinds = this->get_instruction_kinds();
    for (const auto kind : this->get_instruction_kinds()) {
        if (not sets.count(kind)) {
            std::stringstream msg;
            msg << "- error: instruction kind not found in provided set.\n";
            msg << "- ELF file: " << this->file_path() << "\n";
            msg << "- Missing instruction kind: " << ttdecode::isa::to_string(kind) << "\n";
            msg << "- Provided instruction kinds:\n";
            for (const auto& k : sets) {
                msg << "  - " << ttdecode::isa::to_string(k.first) << "\n";
            }
            msg << "- All instruction kinds detected in ELF:\n";
            for (const auto& k : this->get_instruction_kinds()) {
                msg << "  - " << ttdecode::isa::to_string(k) << "\n";
            }
            msg << "- Please verify that your instruction set configuration matches the ELF file's requirements.\n";
            throw std::invalid_argument(msg.str());
        }
    }
    const bool is_swizzled = true;
    auto instructions = ttdecode::decode::decode(
        ttdecode::decode::bytes_to_uint32_vector(this->get_bytes(fun_sym), m_is_little_endian, false),
        kinds,
        sets,
        is_swizzled);

    // Add program counters to each decoded instruction, starting at the function's virtual address.
    // instructions are 32-bit wide; increment PC by 4 for each instruction.
    {
        std::uint64_t pc = fun_sym.value;
        for (std::size_t i = 0; i < instructions.size(); ++i) {
            instructions[i].program_counter = static_cast<std::uint32_t>(pc);
            pc += ttdecode::constants::NUM_BYTES_PER_INSTRUCTION;
        }
    }

    return instructions;
}

ttdecode::decode::decoded_instructions
parser::decode(const function_symbol &fun_sym,
    const std::map<ttdecode::isa::instruction_kind, std::string> &kinds_file_paths) const
{
    const auto sets = ttdecode::isa::get_instruction_sets_incl_rv32(kinds_file_paths);
    return this->decode(fun_sym, sets);
}

ttdecode::decode::decoded_instructions
parser::decode(const function_symbol &fun_sym) const
{
    return this->decode(
        fun_sym,
        ttdecode::isa::get_instruction_sets(this->get_instruction_kinds()));
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode(const std::vector<function_symbol> &fun_syms, const ttdecode::isa::instruction_sets &sets) const
{
    std::map<function_symbol, ttdecode::decode::decoded_instructions> decoded;
    for (const auto &fun_sym : fun_syms) {
        decoded[fun_sym] = this->decode(fun_sym, sets);
    }
    return decoded;
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode(const std::vector<function_symbol> &fun_syms, const std::map<ttdecode::isa::instruction_kind, std::string> &kinds_file_paths) const
{
    const auto sets = ttdecode::isa::get_instruction_sets_incl_rv32(kinds_file_paths);
    return this->decode(fun_syms, sets);
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode(const std::vector<function_symbol> &fun_syms) const
{
    const auto kinds = this->get_instruction_kinds();
    const auto sets = ttdecode::isa::get_instruction_sets(kinds);
    return this->decode(fun_syms, sets);
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode(const section &sec, const ttdecode::isa::instruction_sets &sets) const
{
    return this->decode(this->get_functions(sec), sets);
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode(const section &sec, const std::map<ttdecode::isa::instruction_kind, std::string> &kinds_file_paths) const
{
    return this->decode(this->get_functions(sec), kinds_file_paths);
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode(const section &sec) const
{
    return this->decode(this->get_functions(sec));
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode(const ttdecode::isa::instruction_sets &sets) const
{
    return this->decode(this->get_functions(), sets);
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode(const std::map<ttdecode::isa::instruction_kind, std::string> &kinds_file_paths) const
{
    return this->decode(this->get_functions(), kinds_file_paths);
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
parser::decode() const
{
    return this->decode(this->get_functions());
}

} // namespace elf
} // namespace ttdecode

// Define free helper functions declared in the header.
// These provide simple, convenient one-shot decode utilities.
namespace ttdecode {
namespace elf {

ttdecode::decode::decoded_instructions
decode_function(const std::string &function_name, const std::string &elf_file_path)
{
    parser p(elf_file_path);
    const auto fun_sym = p.get_function(function_name);
    return p.decode(fun_sym);
}

std::map<function_symbol, ttdecode::decode::decoded_instructions>
decode_section(const std::string &section_name, const std::string &elf_file_path)
{
    parser p(elf_file_path);
    const auto &sec = p.get_section(section_name);
    return p.decode(sec);
}

} // namespace elf
} // namespace ttdecode
