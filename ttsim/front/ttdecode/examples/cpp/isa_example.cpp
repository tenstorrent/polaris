// Example: read an instruction-set assembly.yaml and print it nicely.
#include "isa/isa.hpp"

#include <iomanip>
#include <iostream>
#include <map>
#include <string>

static std::string to_binary(std::uint32_t v, std::uint8_t width) {
    if (width == 0) return std::string("0b");
    std::string s;
    s.reserve(static_cast<std::size_t>(width) + 2);
    s.append("0b");
    for (int i = static_cast<int>(width) - 1; i >= 0; --i) {
        s.push_back(((v >> i) & 1u) ? '1' : '0');
    }
    return s;
}

int main(int argc, char** argv) {
    const std::string file = (argc >= 2) ? std::string(argv[1]) : std::string("assembly.yaml");
    const std::uint8_t opcode_start = (argc >= 3) ? std::stoi(std::string(argv[2])) : 0;
    const std::uint8_t max_length = 32;

    try {
        std::map<std::string, ttdecode::isa::instruction> set = ttdecode::isa::get_instruction_set(file, opcode_start, max_length);

        std::cout << "Instructions: " << set.size() << "\n";

        for (const auto& kv : set) {
            const ttdecode::isa::instruction& ins = kv.second;
            std::cout << "- " << ins.mnemonic << "\n";
            std::cout << "  opcode: 0x" << std::hex << std::setw(2) << std::setfill('0')
                      << static_cast<unsigned>(ins.opcode) << std::dec << "\n";

            // Encodings printed in ascending start bit order
            if (!ins.encodings.empty()) {
                std::cout << "  encodings:" << "\n";
                for (const auto& e : ins.encodings) {
                    const ttdecode::isa::encoding& enc = e.second;
                    std::cout << "    - name: " << enc.name << "\n";
                    std::cout << "      start: " << static_cast<unsigned>(enc.start) << "\n";
                    std::cout << "      size: " << static_cast<unsigned>(enc.size) << "\n";
                    std::cout << "      value: " << to_binary(enc.value, enc.size) << "\n";
                }
            }

            // Arguments printed in ascending start bit order
            if (!ins.arguments.empty()) {
                std::cout << "  arguments:" << "\n";
                for (const auto& a : ins.arguments) {
                    const ttdecode::isa::argument& arg = a.second;
                    std::cout << "    - name: " << arg.name << "\n";
                    std::cout << "      start: " << static_cast<unsigned>(arg.start) << "\n";
                    std::cout << "      size: " << static_cast<unsigned>(arg.size) << "\n";
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
