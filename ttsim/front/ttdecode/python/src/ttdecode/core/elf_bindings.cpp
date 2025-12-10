#include "elf/elf_parser.hpp"
#include "elf/elf_parsers.hpp"
#include <iomanip>
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <sstream>

namespace nb = nanobind;
using namespace nb::literals;

void bind_elf(nb::module_& m) {
    // section
    nb::class_<ttdecode::elf::section>(m, "section")
        .def_ro("name", &ttdecode::elf::section::name)
        .def_ro("type", &ttdecode::elf::section::type)
        .def_ro("flags", &ttdecode::elf::section::flags)
        .def_ro("addr", &ttdecode::elf::section::addr)
        .def_ro("offset", &ttdecode::elf::section::offset)
        .def_ro("size", &ttdecode::elf::section::size)
        .def("__repr__", [](const ttdecode::elf::section &s) -> std::string {
            std::ostringstream ss;
            ss << "<section name='" << s.name << "' size=" << s.size << ">";
            return ss.str();
        });

    // function_symbol
    nb::class_<ttdecode::elf::function_symbol>(m, "function_symbol")
        .def_ro("name", &ttdecode::elf::function_symbol::name)
        .def_ro("value", &ttdecode::elf::function_symbol::value)
        .def_ro("size", &ttdecode::elf::function_symbol::size)
        .def_ro("section_index", &ttdecode::elf::function_symbol::section_index)
        .def("__repr__", [](const ttdecode::elf::function_symbol& f) -> std::string {
            std::ostringstream ss;
            ss << "<function_symbol name='" << f.name << "' addr=0x"
               << std::hex << std::setw(16) << std::setfill('0') << f.value
               << ">";
            return ss.str();
        });

    // riscv_attr_tag
    nb::enum_<ttdecode::elf::riscv_attr_tag>(m, "riscv_attr_tag")
        .value("ARCH", ttdecode::elf::riscv_attr_tag::ARCH)
        .value("PRIV_SPEC", ttdecode::elf::riscv_attr_tag::PRIV_SPEC)
        .value("PRIV_SPEC_MINOR", ttdecode::elf::riscv_attr_tag::PRIV_SPEC_MINOR)
        .value("UNALIGNED_ACCESS", ttdecode::elf::riscv_attr_tag::UNALIGNED_ACCESS)
        .value("STACK_ALIGN", ttdecode::elf::riscv_attr_tag::STACK_ALIGN);

    // riscv_attribute
    nb::class_<ttdecode::elf::riscv_attribute>(m, "riscv_attribute")
        .def_ro("tag", &ttdecode::elf::riscv_attribute::tag)
        .def_ro("name", &ttdecode::elf::riscv_attribute::name)
        .def_ro("value", &ttdecode::elf::riscv_attribute::value)
        .def_ro("numeric_value", &ttdecode::elf::riscv_attribute::numeric_value)
        .def_ro("is_numeric", &ttdecode::elf::riscv_attribute::is_numeric);

    // riscv_attributes
    nb::class_<ttdecode::elf::riscv_attributes>(m, "riscv_attributes")
        .def_ro("vendor_name", &ttdecode::elf::riscv_attributes::vendor_name)
        .def_ro("attributes", &ttdecode::elf::riscv_attributes::attributes)
        .def("get_riscv_attribute_with_tag",
             &ttdecode::elf::riscv_attributes::get_riscv_attribute_with_tag,
             nb::rv_policy::reference_internal,
             "tag"_a)
        .def("has_riscv_attribute_with_tag",
             &ttdecode::elf::riscv_attributes::has_riscv_attribute_with_tag,
             "tag"_a);

    // parser
    nb::class_<ttdecode::elf::parser>(m, "parser")
        .def(nb::init<const std::string &>(), "path"_a)
        .def(nb::init<std::vector<uint8_t>>(), "data"_a)
        .def("is_valid", &ttdecode::elf::parser::is_valid)
        .def("is_64_bit", &ttdecode::elf::parser::is_64_bit)
        .def("get_type", &ttdecode::elf::parser::get_type)
        .def("get_class", &ttdecode::elf::parser::get_class)
        .def("get_data", &ttdecode::elf::parser::get_data)
        .def("file_path", &ttdecode::elf::parser::file_path)
        .def("get_sections", &ttdecode::elf::parser::get_sections,
             nb::rv_policy::reference_internal)
        .def("get_section",
             nb::overload_cast<const std::string&>(&ttdecode::elf::parser::get_section, nb::const_),
             nb::rv_policy::reference_internal,
             "name"_a)
        .def("get_section",
             nb::overload_cast<const std::size_t>(&ttdecode::elf::parser::get_section, nb::const_),
             nb::rv_policy::reference_internal,
             "index"_a)
        .def("get_section_index", &ttdecode::elf::parser::get_section_index, "sec"_a)
        .def("get_function", &ttdecode::elf::parser::get_function, "name"_a)
        .def("get_functions",
             nb::overload_cast<>(&ttdecode::elf::parser::get_functions, nb::const_))
        .def("get_functions",
             nb::overload_cast<const ttdecode::elf::section&>(&ttdecode::elf::parser::get_functions, nb::const_),
             "sec"_a)
        .def("get_bytes",
             [](const ttdecode::elf::parser &self, const ttdecode::elf::function_symbol &func) {
                 std::vector<uint8_t> v = self.get_bytes(func);
                 return nb::bytes(reinterpret_cast<const char*>(v.data()), v.size());
             },
             "function_symbol"_a)
        .def("get_bytes",
             [](const ttdecode::elf::parser &self, const ttdecode::elf::section &sec) {
                 std::vector<uint8_t> v = self.get_bytes(sec);
                 return nb::bytes(reinterpret_cast<const char*>(v.data()), v.size());
             },
             "sec"_a)
        .def("get_riscv_attributes", &ttdecode::elf::parser::get_riscv_attributes)
        .def("riscv_attribute_to_string", &ttdecode::elf::parser::riscv_attribute_to_string)
        .def("get_instruction_kinds", &ttdecode::elf::parser::get_instruction_kinds)
        // decode: function_symbol overloads
        .def("decode",
             nb::overload_cast<const ttdecode::elf::function_symbol&, const ttdecode::isa::instruction_sets&>(&ttdecode::elf::parser::decode, nb::const_),
             "fun_sym"_a, "sets"_a)
        .def("decode",
             nb::overload_cast<const ttdecode::elf::function_symbol&, const std::map<ttdecode::isa::instruction_kind, std::string>&>(&ttdecode::elf::parser::decode, nb::const_),
             "fun_sym"_a, "kinds_file_paths"_a)
        .def("decode",
             nb::overload_cast<const ttdecode::elf::function_symbol&>(&ttdecode::elf::parser::decode, nb::const_),
             "fun_sym"_a)
        // decode: vector<function_symbol> overloads
        .def("decode",
             nb::overload_cast<const std::vector<ttdecode::elf::function_symbol>&, const ttdecode::isa::instruction_sets&>(&ttdecode::elf::parser::decode, nb::const_),
             "functions"_a, "sets"_a)
        .def("decode",
             nb::overload_cast<const std::vector<ttdecode::elf::function_symbol>&, const std::map<ttdecode::isa::instruction_kind, std::string>&>(&ttdecode::elf::parser::decode, nb::const_),
             "functions"_a, "kinds_file_paths"_a)
        .def("decode",
             nb::overload_cast<const std::vector<ttdecode::elf::function_symbol>&>(&ttdecode::elf::parser::decode, nb::const_),
             "functions"_a)
        // decode: section overloads
        .def("decode",
             nb::overload_cast<const ttdecode::elf::section&, const ttdecode::isa::instruction_sets&>(&ttdecode::elf::parser::decode, nb::const_),
             "sec"_a, "sets"_a)
        .def("decode",
             nb::overload_cast<const ttdecode::elf::section&, const std::map<ttdecode::isa::instruction_kind, std::string>&>(&ttdecode::elf::parser::decode, nb::const_),
             "sec"_a, "kinds_file_paths"_a)
        .def("decode",
             nb::overload_cast<const ttdecode::elf::section&>(&ttdecode::elf::parser::decode, nb::const_),
             "sec"_a)
        // decode: all functions overloads
        .def("decode",
             nb::overload_cast<const ttdecode::isa::instruction_sets&>(&ttdecode::elf::parser::decode, nb::const_),
             "sets"_a)
        .def("decode",
             nb::overload_cast<const std::map<ttdecode::isa::instruction_kind, std::string>&>(&ttdecode::elf::parser::decode, nb::const_),
             "kinds_file_paths"_a)
        .def("decode",
             nb::overload_cast<>(&ttdecode::elf::parser::decode, nb::const_));

    // parsers
    nb::class_<ttdecode::elf::parsers>(m, "parsers")
        .def(nb::init<const std::vector<std::string>&>(), "file_names"_a)
        .def("size", &ttdecode::elf::parsers::size)
        .def("get_instruction_kinds", &ttdecode::elf::parsers::get_instruction_kinds, "mode"_a = "merged")
        .def("instruction_kinds_match_for_all_elfs", &ttdecode::elf::parsers::instruction_kinds_match_for_all_elfs)
        .def("get_tensix_architecture", &ttdecode::elf::parsers::get_tensix_architecture)
        .def("at",
             nb::overload_cast<const std::size_t>(&ttdecode::elf::parsers::at, nb::const_),
             nb::rv_policy::reference_internal,
             "index"_a)
        .def("at",
             nb::overload_cast<const std::size_t>(&ttdecode::elf::parsers::at),
             nb::rv_policy::reference_internal,
             "index"_a)
        .def("__len__", &ttdecode::elf::parsers::size)
        .def("__getitem__",
             [](ttdecode::elf::parsers &self, const std::size_t idx) -> ttdecode::elf::parser& {
                 return self.at(idx);
             },
             nb::rv_policy::reference_internal,
             "index"_a);

    // Free helpers
    m.def("decode_function", &ttdecode::elf::decode_function, "function_name"_a, "elf_file_path"_a);
    m.def("decode_section", &ttdecode::elf::decode_section, "section_name"_a, "elf_file_path"_a);
}
