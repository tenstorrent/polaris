#include "isa/yaml_parser.hpp"
#include <iostream>
#include <string>

namespace ttdecode {
namespace isa {
    
parser::parser() {}

parser::parser(const std::string& file_path) : m_file_path(file_path) {}

parser::~parser() {}

YAML::Node parser::parse() {
    if (m_file_path.empty()) {
        throw YamlParsingException("Error: file path not set for parser");
    }
    try {
        // YAML::LoadFile is the primary function for parsing a file from disk.
        // It throws exceptions on failure.
        YAML::Node config = YAML::LoadFile(m_file_path);
        return config;
    } catch (const YAML::BadFile& e) {
        // Handle the case where the file does not exist or cannot be read.
        throw YamlParsingException("Error: Could not open or read file at '" + m_file_path + "'. Original error: " + e.what());
    } catch (const YAML::ParserException& e) {
        // Handle malformed YAML content.
        throw YamlParsingException("Error: Failed to parse YAML file at '" + m_file_path + "'. The file may be malformed. Original error: " + e.what());
    } catch (const YAML::Exception& e) {
        // A general catch-all for any other yaml-cpp specific exceptions.
        throw YamlParsingException("An unexpected YAML parsing error occurred with file '" + m_file_path + "'. Original error: " + e.what());
    }
}

YAML::Node parser::parse(const std::string& file_path) {
    this->set_file_path(file_path);
    return this->parse();
}

const std::string& parser::file_path() const {
    return m_file_path;
}

void parser::set_file_path(const std::string& file_path) {
    m_file_path = file_path;
}

YAML::Node
parse_instruction_set_file(const std::string& file_path) {
    return parser().parse(file_path);
}

} // namespace isa
} // namespace ttdecode
