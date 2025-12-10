#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/set.h>

#include "isa/isa.hpp"
#include "decode/rv32.hpp"
#include "decode/tensix.hpp"
#include "decode/decode.hpp"
#include "decode/utils.hpp"

namespace nb = nanobind;
using namespace nb::literals;

static nb::dict operands_to_dict(const ttdecode::decode::operands& ops) {
    nb::dict d;
    for (const auto& kv : ops.all) {
        d[kv.first.c_str()] = kv.second;
    }
    return d;
}

void bind_decode(nb::module_& m) {
    // registers
    nb::class_<ttdecode::decode::registers>(m, "registers")
        .def(nb::init<>())
        .def_rw("integers", &ttdecode::decode::registers::integers)
        .def_rw("floats", &ttdecode::decode::registers::floats)
        .def("set_integers", nb::overload_cast<const std::vector<int>&>(&ttdecode::decode::registers::set_integers))
        .def("set_integers", nb::overload_cast<const int>(&ttdecode::decode::registers::set_integers))
        .def("set_floats", nb::overload_cast<const std::vector<int>&>(&ttdecode::decode::registers::set_floats))
        .def("set_floats", nb::overload_cast<const int>(&ttdecode::decode::registers::set_floats))
        .def("empty", &ttdecode::decode::registers::empty)
        .def("__copy__", [](const ttdecode::decode::registers& self) { return ttdecode::decode::registers{self}; })
        .def("__deepcopy__", [](const ttdecode::decode::registers& self, nb::object) { return ttdecode::decode::registers{self}; });

    // operands
    nb::class_<ttdecode::decode::operands>(m, "operands")
        .def(nb::init<>())
        .def_rw("all", &ttdecode::decode::operands::all)
        .def_rw("attributes", &ttdecode::decode::operands::attributes)
        .def_rw("sources", &ttdecode::decode::operands::sources)
        .def_rw("destinations", &ttdecode::decode::operands::destinations)
        .def_rw("immediates", &ttdecode::decode::operands::immediates)
        .def_rw("decoded_values", &ttdecode::decode::operands::decoded_values)
        .def("set_all", &ttdecode::decode::operands::set_all, "all"_a, "mode"_a = std::string("q"))
        .def("set_sources", &ttdecode::decode::operands::set_sources, "sources"_a, "mode"_a = std::string("q"))
        .def("set_destinations", &ttdecode::decode::operands::set_destinations, "destinations"_a, "mode"_a = std::string("q"))
        .def("set_integer_sources", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::operands::set_integer_sources), "vals"_a, "mode"_a = std::string("q"))
        .def("set_integer_sources", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::operands::set_integer_sources), "v"_a, "mode"_a = std::string("q"))
        .def("set_float_sources", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::operands::set_float_sources), "vals"_a, "mode"_a = std::string("q"))
        .def("set_float_sources", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::operands::set_float_sources), "v"_a, "mode"_a = std::string("q"))
        .def("set_integer_destinations", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::operands::set_integer_destinations), "vals"_a, "mode"_a = std::string("q"))
        .def("set_integer_destinations", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::operands::set_integer_destinations), "v"_a, "mode"_a = std::string("q"))
        .def("set_float_destinations", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::operands::set_float_destinations), "vals"_a, "mode"_a = std::string("q"))
        .def("set_float_destinations", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::operands::set_float_destinations), "v"_a, "mode"_a = std::string("q"))
        .def("set_immediates", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::operands::set_immediates), "vals"_a, "mode"_a = std::string("q"))
        .def("set_immediates", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::operands::set_immediates), "v"_a, "mode"_a = std::string("q"))
        .def("set_attributes", &ttdecode::decode::operands::set_attributes, "attrs"_a, "mode"_a = std::string("q"))
        .def("empty", &ttdecode::decode::operands::empty)
        .def("__copy__", [](const ttdecode::decode::operands& self) { return ttdecode::decode::operands{self}; })
        .def("__deepcopy__", [](const ttdecode::decode::operands& self, nb::object) { return ttdecode::decode::operands{self}; });

    // decoded_instruction
    nb::class_<ttdecode::decode::decoded_instruction>(m, "decoded_instruction")
        .def(nb::init<>())
        .def_rw("word", &ttdecode::decode::decoded_instruction::word)
        .def("set_word", &ttdecode::decode::decoded_instruction::set_word, "word"_a, "mode"_a = std::string("q"))
        .def_prop_rw("program_counter",
            [](const ttdecode::decode::decoded_instruction& d) -> nb::object {
                if (d.program_counter.has_value()) return nb::cast(d.program_counter.value());
                return nb::none();
            },
            [](ttdecode::decode::decoded_instruction& d, nb::object obj) {
                if (obj.is_none()) { d.program_counter.reset(); return; }
                d.program_counter = nb::cast<std::uint32_t>(obj);
            })
        .def("set_program_counter", &ttdecode::decode::decoded_instruction::set_program_counter, "pc"_a, "mode"_a = std::string("q"))
        .def_prop_rw("kind",
            [](const ttdecode::decode::decoded_instruction& d) -> nb::object {
                if (d.kind.has_value()) return nb::cast(d.kind.value());
                return nb::none();
            },
            [](ttdecode::decode::decoded_instruction& d, nb::object obj) {
                if (obj.is_none()) { d.kind.reset(); return; }
                d.kind = nb::cast<ttdecode::isa::instruction_kind>(obj);
            })
        .def("set_kind", &ttdecode::decode::decoded_instruction::set_kind, "kind"_a, "mode"_a = std::string("q"))
        .def_prop_rw("opcode",
            [](const ttdecode::decode::decoded_instruction& d) -> nb::object {
                if (d.opcode.has_value()) return nb::cast(d.opcode.value());
                return nb::none();
            },
            [](ttdecode::decode::decoded_instruction& d, nb::object obj) {
                if (obj.is_none()) { d.opcode.reset(); return; }
                d.opcode = nb::cast<std::uint32_t>(obj);
            })
        .def("set_opcode", &ttdecode::decode::decoded_instruction::set_opcode, "opcode"_a, "mode"_a = std::string("q"))
        .def_prop_rw("mnemonic",
            [](const ttdecode::decode::decoded_instruction& d) -> nb::object {
                if (d.mnemonic.has_value()) return nb::cast(d.mnemonic.value());
                return nb::none();
            },
            [](ttdecode::decode::decoded_instruction& d, nb::object obj) {
                if (obj.is_none()) { d.mnemonic.reset(); return; }
                d.mnemonic = nb::cast<std::string>(obj);
            })
        .def("set_mnemonic", &ttdecode::decode::decoded_instruction::set_mnemonic, "mnemonic"_a, "mode"_a = std::string("q"))
        .def_prop_rw("operands",
            [](const ttdecode::decode::decoded_instruction& d) -> nb::object {
                if (d.operands.has_value()) return nb::cast(d.operands.value());
                return nb::none();
            },
            [](ttdecode::decode::decoded_instruction& d, nb::object obj) {
                if (obj.is_none()) { d.operands.reset(); return; }
                d.operands = nb::cast<ttdecode::decode::operands>(obj);
            })
        .def("set_operands", &ttdecode::decode::decoded_instruction::set_operands, "operands"_a, "mode"_a = std::string("q"))
        .def("operands_dict", [](const ttdecode::decode::decoded_instruction& d) {
            if (!d.operands.has_value()) return nb::dict();
            return operands_to_dict(d.operands.value());
        })
        // Convenience methods that initialize operands if needed
        .def("set_all", &ttdecode::decode::decoded_instruction::set_all, "all"_a, "mode"_a = std::string("q"))
        .def("set_sources", &ttdecode::decode::decoded_instruction::set_sources, "sources"_a, "mode"_a = std::string("q"))
        .def("set_destinations", &ttdecode::decode::decoded_instruction::set_destinations, "destinations"_a, "mode"_a = std::string("q"))
        .def("set_integer_sources", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::decoded_instruction::set_integer_sources), "vals"_a, "mode"_a = std::string("q"))
        .def("set_integer_sources", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::decoded_instruction::set_integer_sources), "v"_a, "mode"_a = std::string("q"))
        .def("set_float_sources", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::decoded_instruction::set_float_sources), "vals"_a, "mode"_a = std::string("q"))
        .def("set_float_sources", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::decoded_instruction::set_float_sources), "v"_a, "mode"_a = std::string("q"))
        .def("set_integer_destinations", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::decoded_instruction::set_integer_destinations), "vals"_a, "mode"_a = std::string("q"))
        .def("set_integer_destinations", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::decoded_instruction::set_integer_destinations), "v"_a, "mode"_a = std::string("q"))
        .def("set_float_destinations", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::decoded_instruction::set_float_destinations), "vals"_a, "mode"_a = std::string("q"))
        .def("set_float_destinations", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::decoded_instruction::set_float_destinations), "v"_a, "mode"_a = std::string("q"))
        .def("set_immediates", nb::overload_cast<const std::vector<int>&, const std::string&>(&ttdecode::decode::decoded_instruction::set_immediates), "vals"_a, "mode"_a = std::string("q"))
        .def("set_immediates", nb::overload_cast<const int, const std::string&>(&ttdecode::decode::decoded_instruction::set_immediates), "v"_a, "mode"_a = std::string("q"))
        .def("set_attributes", &ttdecode::decode::decoded_instruction::set_attributes, "attrs"_a, "mode"_a = std::string("q"))
        .def("to_string", &ttdecode::decode::decoded_instruction::to_string)
        .def("__str__", &ttdecode::decode::decoded_instruction::to_string)
        .def("__repr__", &ttdecode::decode::decoded_instruction::to_string)
        .def("__copy__", [](const ttdecode::decode::decoded_instruction& self) { return ttdecode::decode::decoded_instruction{self}; })
        .def("__deepcopy__", [](const ttdecode::decode::decoded_instruction& self, nb::object) { return ttdecode::decode::decoded_instruction{self}; });

    // Validity helpers
    m.def("rv32_is_valid", &ttdecode::decode::rv32_is_valid, "word"_a, "is_swizzled"_a = true);
    m.def("tensix_is_valid", &ttdecode::decode::tensix_is_valid, "word"_a, "is_swizzled"_a = true);

    // Specific decoders (YAML path overloads)
    m.def("rv32_decode",
        [](std::uint32_t word, const std::string& yaml_file, bool is_swizzled) {
            return ttdecode::decode::rv32_decode(word, yaml_file, is_swizzled);
        },
        "word"_a, "yaml_file"_a, "is_swizzled"_a = true);

    m.def("tensix_decode",
        [](std::uint32_t word, ttdecode::isa::instruction_kind kind, const std::string& yaml_file, bool is_swizzled) {
            return ttdecode::decode::tensix_decode(word, kind, yaml_file, is_swizzled);
        },
        "word"_a, "kind"_a, "yaml_file"_a, "is_swizzled"_a = true);

    // Generic helpers from decode.hpp
    m.def("get_instruction_kind",
        nb::overload_cast<const std::uint32_t, const ttdecode::isa::instruction_sets&, const bool>(&ttdecode::decode::get_instruction_kind),
        "word"_a, "sets"_a, "is_swizzled"_a = true);

    m.def("get_instruction_kind",
        nb::overload_cast<const std::uint32_t, const std::set<ttdecode::isa::instruction_kind>, const bool>(&ttdecode::decode::get_instruction_kind),
        "word"_a, "kinds"_a, "is_swizzled"_a = true);

    // decode overloads (single)
    m.def("decode",
        nb::overload_cast<const std::uint32_t, const ttdecode::isa::instruction_kind, const ttdecode::isa::instruction_set&, const bool>(&ttdecode::decode::decode),
        "word"_a, "kind"_a, "set"_a, "is_swizzled"_a = true);
    m.def("decode",
        nb::overload_cast<const std::uint32_t, const ttdecode::isa::instruction_kind, const ttdecode::isa::instruction_sets&, const bool>(&ttdecode::decode::decode),
        "word"_a, "kind"_a, "sets"_a, "is_swizzled"_a = true);
    m.def("decode",
        nb::overload_cast<const std::uint32_t, const ttdecode::isa::instruction_sets&, const bool>(&ttdecode::decode::decode),
        "word"_a, "sets"_a, "is_swizzled"_a = true);
    m.def("decode",
        nb::overload_cast<const std::uint32_t, const ttdecode::isa::instruction_kind, const bool>(&ttdecode::decode::decode),
        "word"_a, "kind"_a, "is_swizzled"_a = true);
    m.def("decode",
        nb::overload_cast<const std::uint32_t, const bool>(&ttdecode::decode::decode),
        "word"_a, "is_swizzled"_a = true);

    // decode overloads (batch)
    m.def("decode_batch",
        nb::overload_cast<const std::vector<std::uint32_t>&, const std::set<ttdecode::isa::instruction_kind>, const ttdecode::isa::instruction_sets&, const bool>(&ttdecode::decode::decode),
        "words"_a, "kinds"_a, "sets"_a, "is_swizzled"_a = true);
    m.def("decode_batch",
        nb::overload_cast<const std::vector<std::uint32_t>&, const std::set<ttdecode::isa::instruction_kind>, const ttdecode::isa::instruction_sets&, const std::vector<bool>&>(&ttdecode::decode::decode),
        "words"_a, "kinds"_a, "sets"_a, "swizzle_flags"_a);

    // Utility functions
    m.def("rotl32", &ttdecode::decode::rotl32, "value"_a, "shift"_a,
        "Rotate left (circular shift) a 32-bit value by the specified number of bits");
    m.def("rotr32", &ttdecode::decode::rotr32, "value"_a, "shift"_a,
        "Rotate right (circular shift) a 32-bit value by the specified number of bits");
    m.def("swizzle", &ttdecode::decode::swizzle, "instruction"_a,
        "Apply swizzle transformation (left rotate by 2 bits)");
    m.def("unswizzle", &ttdecode::decode::unswizzle, "instruction"_a,
        "Reverse swizzle transformation (right rotate by 2 bits)");
}
