#pragma once

#include "yaml-cpp/yaml.h"
#include <stdexcept>
#include <string>

namespace ttdecode {
namespace isa {

// Custom exception for clear error reporting
class YamlParsingException : public std::runtime_error {
public:
    explicit YamlParsingException(const std::string& what) : std::runtime_error(what) {}
};

class parser {
public:
    parser();
    parser(const std::string& file_path);
    ~parser();

    YAML::Node parse();

    const std::string&
    file_path() const;

    void
    set_file_path(const std::string& file_path);

    /**
     * @brief Parses a YAML file from the given path.
     * @param file_path The path to the YAML file.
     * @return A YAML::Node object representing the root of the parsed document.
     * @throws YamlParsingException if the file cannot be found or is malformed.
     */
    YAML::Node parse(const std::string& file_path);
private:
    std::string m_file_path;
};

YAML::Node
parse_instruction_set_file(const std::string& file_path);

} // namespace isa
} // namespace ttdecode
