#include "isa/yaml_parser.hpp"
#include "isa/isa.hpp"
#include "isa/defaults.hpp"
#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/set.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals; // for _a shorthand

// Forward declarations
nb::object yaml_node_to_py_object(const YAML::Node& node);
YAML::Node py_object_to_yaml_node(nb::handle obj);

nb::object yaml_node_to_py_object(const YAML::Node& node) {
    switch (node.Type()) {
        case YAML::NodeType::Map: {
            nb::dict dict;
            for (auto it = node.begin(); it!= node.end(); ++it) {
                std::string key = it->first.as<std::string>();
                dict[key.c_str()] = yaml_node_to_py_object(it->second);
            }
            return dict;
        }
        case YAML::NodeType::Sequence: {
            nb::list list;
            for (const auto& item : node) {
                list.append(yaml_node_to_py_object(item));
            }
            return list;
        }
        case YAML::NodeType::Scalar: {
            return nb::str(node.as<std::string>().c_str());
        }
        case YAML::NodeType::Null: {
            return nb::none();
        }
        default: {
            return nb::none();
        }
    }
}

YAML::Node py_object_to_yaml_node(nb::handle obj) {
    if (nb::isinstance<nb::dict>(obj)) {
        YAML::Node node(YAML::NodeType::Map);
        nb::dict d = nb::cast<nb::dict>(obj);
        for (auto item : d) {
            std::string key = nb::cast<std::string>(item.first);
            YAML::Node val = py_object_to_yaml_node(item.second);
            node[key] = val;
        }
        return node;
    }
    if (nb::isinstance<nb::list>(obj)) {
        YAML::Node node(YAML::NodeType::Sequence);
        nb::list l = nb::cast<nb::list>(obj);
        for (auto v : l) {
            node.push_back(py_object_to_yaml_node(v));
        }
        return node;
    }
    if (obj.is_none()) {
        return YAML::Node();
    }
    try { return YAML::Node(nb::cast<std::string>(obj)); } catch (...) {}
    try { return YAML::Node(std::to_string(nb::cast<long long>(obj))); } catch (...) {}
    try { return YAML::Node(nb::cast<bool>(obj) ? "true" : "false"); } catch (...) {}
    return YAML::Node(nb::cast<std::string>(nb::str(obj)));
}

void bind_isa(nb::module_& m) {
    // Exceptions
    nb::exception<ttdecode::isa::YamlParsingException>(m, "YamlParsingError", PyExc_RuntimeError);

    // Enum: instruction_kind
    nb::enum_<ttdecode::isa::instruction_kind>(m, "instruction_kind")
        .value("rv32", ttdecode::isa::instruction_kind::rv32)
        .value("ttwh", ttdecode::isa::instruction_kind::ttwh)
        .value("ttbh", ttdecode::isa::instruction_kind::ttbh)
        .value("ttqs", ttdecode::isa::instruction_kind::ttqs);

    // argument nested enums and structs
    nb::enum_<ttdecode::isa::argument::fcov_kind>(m, "fcov_kind")
        .value("none", ttdecode::isa::argument::fcov_kind::none)
        .value("bins", ttdecode::isa::argument::fcov_kind::bins)
        .value("bin_interval", ttdecode::isa::argument::fcov_kind::bin_interval)
        .value("boolean", ttdecode::isa::argument::fcov_kind::boolean);

    nb::enum_<ttdecode::isa::argument::fcov_spec::bin_eval_kind>(m, "bin_eval_kind")
        .value("unknown", ttdecode::isa::argument::fcov_spec::bin_eval_kind::unknown)
        .value("bitwise", ttdecode::isa::argument::fcov_spec::bin_eval_kind::bitwise)
        .value("equality", ttdecode::isa::argument::fcov_spec::bin_eval_kind::equality);

    nb::class_<ttdecode::isa::argument::fcov_bin>(m, "fcov_bin")
        .def(nb::init<>())
        .def_rw("name", &ttdecode::isa::argument::fcov_bin::name)
        .def_rw("value", &ttdecode::isa::argument::fcov_bin::value);

    nb::class_<ttdecode::isa::argument::fcov_interval>(m, "fcov_interval")
        .def(nb::init<>())
        .def_rw("name", &ttdecode::isa::argument::fcov_interval::name)
        .def_rw("low", &ttdecode::isa::argument::fcov_interval::low)
        .def_rw("high", &ttdecode::isa::argument::fcov_interval::high);

    nb::class_<ttdecode::isa::argument::fcov_spec>(m, "fcov_spec")
        .def(nb::init<>())
        .def_rw("kind", &ttdecode::isa::argument::fcov_spec::kind)
        .def_rw("bins", &ttdecode::isa::argument::fcov_spec::bins)
        .def_rw("intervals", &ttdecode::isa::argument::fcov_spec::intervals)
        .def_rw("bin_eval", &ttdecode::isa::argument::fcov_spec::bin_eval);

    // core structs
    nb::class_<ttdecode::isa::argument>(m, "argument")
        .def(nb::init<>())
        .def_rw("name", &ttdecode::isa::argument::name)
        .def_rw("start", &ttdecode::isa::argument::start)
        .def_rw("size", &ttdecode::isa::argument::size)
        .def_prop_rw("fcov",
            [](ttdecode::isa::argument &a) -> nb::object {
                if (a.fcov.has_value()) return nb::cast(a.fcov.value());
                return nb::none();
            },
            [](ttdecode::isa::argument &a, nb::handle obj) {
                if (obj.is_none()) { a.fcov.reset(); return; }
                a.fcov = nb::cast<ttdecode::isa::argument::fcov_spec>(obj);
            }
        );

    nb::class_<ttdecode::isa::encoding>(m, "encoding")
        .def(nb::init<>())
        .def_rw("name", &ttdecode::isa::encoding::name)
        .def_rw("start", &ttdecode::isa::encoding::start)
        .def_rw("size", &ttdecode::isa::encoding::size)
        .def_rw("value", &ttdecode::isa::encoding::value);

    nb::class_<ttdecode::isa::instruction>(m, "instruction")
        .def(nb::init<>())
        .def_rw("mnemonic", &ttdecode::isa::instruction::mnemonic)
        .def_rw("opcode", &ttdecode::isa::instruction::opcode)
        .def_rw("encodings", &ttdecode::isa::instruction::encodings)
        .def_rw("arguments", &ttdecode::isa::instruction::arguments);

    // Free functions
    m.def("is_tensix", &ttdecode::isa::is_tensix, "kind"_a,
          "Return True if kind is a Tensix variant");
    m.def("opcode_start_bit", &ttdecode::isa::opcode_start_bit, "kind"_a,
          "Return opcode start bit for kind");
    m.def("get_default_instruction_set_file_path",
          &ttdecode::isa::get_default_instruction_set_file_path,
          "kind"_a,
          "Return default assembly.yaml path for kind");

    m.def("to_string",
          nb::overload_cast<const ttdecode::isa::instruction_kind>(&ttdecode::isa::to_string),
          "kind"_a);
    m.def("to_string",
          nb::overload_cast<const ttdecode::isa::argument::fcov_kind>(&ttdecode::isa::to_string),
          "fcov_kind"_a);

    m.def("to_instruction_kind", &ttdecode::isa::to_instruction_kind,
          "Return instruction_kind");

    m.def("tensix_instruction_kinds", &ttdecode::isa::tensix_instruction_kinds,
          "Return set of Tensix instruction kinds");

    m.def("validate_and_update_instruction_sizes",
          &ttdecode::isa::validate_and_update_instruction_sizes,
          "instr"_a, "opcode_start"_a,
          "max_length"_a = ttdecode::constants::NUM_BITS_PER_INSTRUCTION);

    // parser class (snake_case name in Python)
    nb::class_<ttdecode::isa::parser> parser_class(m, "parser");
    parser_class.doc() = "YAML parser (simple file loader).";
    parser_class
      .def(nb::init<>())
      .def(nb::init<const std::string&>())
      .def("file_path", &ttdecode::isa::parser::file_path,
           "Get the currently configured file path")
      .def("set_file_path", &ttdecode::isa::parser::set_file_path,
           "file_path"_a,
           "Set the file path for the parser")
      .def("parse",
           [](ttdecode::isa::parser &self) {
               YAML::Node root_node = self.parse();
               return yaml_node_to_py_object(root_node);
           },
           "Parse the configured YAML file and return a Python dict")
      .def("parse",
           [](ttdecode::isa::parser &self, const std::string &file_path) {
               YAML::Node root_node = self.parse(file_path);
               return yaml_node_to_py_object(root_node);
           },
           "file_path"_a,
           "Parse a YAML file and return a Python dict");

    // Module-level helper: parse_instruction_set_file -> dict
    m.def("parse_instruction_set_file",
          [](const std::string& file_path) {
              YAML::Node root_node = ttdecode::isa::parse_instruction_set_file(file_path);
              return yaml_node_to_py_object(root_node);
          },
          "file_path"_a,
          "Parse a YAML file and return a Python dict");

    // get_instruction_set overloads (prefer string path over Python object)
    m.def("get_instruction_set",
          nb::overload_cast<const std::string&, const std::uint8_t, const std::uint8_t>(&ttdecode::isa::get_instruction_set),
          "file_path"_a, "opcode_start"_a,
          "max_length"_a = ttdecode::constants::NUM_BITS_PER_INSTRUCTION);

    m.def("get_instruction_set",
          [](nb::dict node_obj, std::uint8_t opcode_start, std::uint8_t max_length) {
              YAML::Node node = py_object_to_yaml_node(node_obj);
              return ttdecode::isa::get_instruction_set(node, opcode_start, max_length);
          },
          "node"_a, "opcode_start"_a,
          "max_length"_a = ttdecode::constants::NUM_BITS_PER_INSTRUCTION,
          "Build instruction set from a YAML-like Python mapping");

    m.def("get_instruction_set",
          nb::overload_cast<const std::string&, const ttdecode::isa::instruction_kind>(&ttdecode::isa::get_instruction_set),
          "file_path"_a, "kind"_a);

    m.def("get_instruction_set",
          [](nb::dict node_obj, ttdecode::isa::instruction_kind kind) {
              YAML::Node node = py_object_to_yaml_node(node_obj);
              return ttdecode::isa::get_instruction_set(node, kind);
          },
          "node"_a, "kind"_a);

    m.def("get_instruction_set",
          nb::overload_cast<const ttdecode::isa::instruction_kind>(&ttdecode::isa::get_instruction_set),
          "kind"_a);

    // get_instruction_sets overloads
    // Smart dispatcher based on dict value types
    m.def("get_instruction_sets",
          [](nb::dict d) {
              // Check if values are strings (file paths) or dicts/objects (YAML nodes)
              if (d.size() > 0) {
                  auto first_item = *d.begin();
                  auto first_val = first_item.second;
                  if (nb::isinstance<nb::str>(first_val)) {
                      // String values: map to file paths
                      std::map<ttdecode::isa::instruction_kind, std::string> paths;
                      for (auto item : d) {
                          auto k = nb::cast<ttdecode::isa::instruction_kind>(item.first);
                          auto v = nb::cast<std::string>(item.second);
                          paths.emplace(k, v);
                      }
                      return ttdecode::isa::get_instruction_sets(paths);
                  }
              }
              // Dict/object values: map to YAML nodes
              std::map<ttdecode::isa::instruction_kind, YAML::Node> m_nodes;
              for (auto item : d) {
                  auto k = nb::cast<ttdecode::isa::instruction_kind>(item.first);
                  YAML::Node v = py_object_to_yaml_node(item.second);
                  m_nodes.emplace(k, v);
              }
              return ttdecode::isa::get_instruction_sets(m_nodes);
          },
          "kinds_yaml_nodes_or_paths"_a);

    m.def("get_instruction_sets",
          nb::overload_cast<const std::set<ttdecode::isa::instruction_kind>>(&ttdecode::isa::get_instruction_sets),
          "kinds"_a);

    m.def("get_instruction_sets_incl_rv32", &ttdecode::isa::get_instruction_sets_incl_rv32, "kinds_file_paths"_a);

    // defaults class and global access
    nb::class_<ttdecode::isa::defaults>(m, "defaults")
        .def(nb::init<>())
        .def("instruction_set_file_paths", &ttdecode::isa::defaults::instruction_set_file_paths)
        .def("riscv_attributes_instruction_kinds",
             [](const ttdecode::isa::defaults& self) {
                 nb::dict out;
                 const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>& m = self.riscv_attributes_instruction_kinds();
                 nb::object frozenset_ctor = nb::module_::import_("builtins").attr("frozenset");
                 for (const std::pair<const std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>& kv : m) {
                     nb::list items;
                     for (const ttdecode::isa::instruction_kind kind : kv.first) {
                         items.append(nb::cast(kind));
                     }
                     nb::object key = frozenset_ctor(items);
                     out[key] = nb::cast(kv.second);
                 }
                 return out;
             })
        // std::string version before std::set<std::string> overaload.
        .def("update_instruction_set_path", nb::overload_cast<const std::map<ttdecode::isa::instruction_kind, std::string>&>(&ttdecode::isa::defaults::update_instruction_set_path), "other"_a)
        .def("update_instruction_set_path", nb::overload_cast<const ttdecode::isa::instruction_kind, const std::string&>(&ttdecode::isa::defaults::update_instruction_set_path), "kind"_a, "path"_a)
        .def("append_riscv_attribute", nb::overload_cast<const std::set<ttdecode::isa::instruction_kind>&, const std::string&>(&ttdecode::isa::defaults::append_riscv_attribute), "instruction_kinds"_a, "riscv_attribute"_a)
        .def("append_riscv_attribute", nb::overload_cast<const std::set<ttdecode::isa::instruction_kind>&, const std::set<std::string>&>(&ttdecode::isa::defaults::append_riscv_attribute), "instruction_kinds"_a, "riscv_attributes"_a)
        .def("append_riscv_attribute", nb::overload_cast<const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>&>(&ttdecode::isa::defaults::append_riscv_attribute), "mapping"_a)
        .def("remove_riscv_attribute", nb::overload_cast<const std::set<ttdecode::isa::instruction_kind>&, const std::string&>(&ttdecode::isa::defaults::remove_riscv_attribute), "instruction_kinds"_a, "riscv_attribute"_a)
        .def("remove_riscv_attribute", nb::overload_cast<const std::set<ttdecode::isa::instruction_kind>&, const std::set<std::string>&>(&ttdecode::isa::defaults::remove_riscv_attribute), "instruction_kinds"_a, "riscv_attributes"_a)
        .def("remove_riscv_attribute", nb::overload_cast<const std::map<std::set<ttdecode::isa::instruction_kind>, std::set<std::string>>&>(&ttdecode::isa::defaults::remove_riscv_attribute), "mapping"_a)
        .def("reset_instruction_set_file_paths", &ttdecode::isa::defaults::reset_instruction_set_file_paths)
        .def("reset_riscv_attributes_instruction_kinds", &ttdecode::isa::defaults::reset_riscv_attributes_instruction_kinds)
        .def("reset", &ttdecode::isa::defaults::reset);

    m.def("global_defaults", []() -> ttdecode::isa::defaults& { return ttdecode::isa::global_defaults(); }, nb::rv_policy::reference);
}
