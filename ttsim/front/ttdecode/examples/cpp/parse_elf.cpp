#include "elf/elf_parser.hpp"
#include <iomanip>
#include <iostream>
#include <sstream>

int
main(int argc, char **argv) {
    if (argc!= 2) {
        std::cerr << "Usage: " << argv << " <elf_file_path>" << std::endl;
        return 1;
    }

    try {
        ttdecode::elf::parser parser(argv[1]);
        if (!parser.is_valid()) {
            std::cerr << "Error: Not a valid ELF file." << std::endl;
            return 1;
        }

        std::cout << "--- ELF Header Info ---" << std::endl;
        std::cout << "Type: " << parser.get_type() << std::endl;
        std::cout << "Class: " << parser.get_class() << std::endl;
        std::cout << "Data: " << parser.get_data() << std::endl;

        std::cout << "\n--- Functions in .text section ---" << std::endl;
        auto functions = parser.get_functions(parser.get_section(".text"));
        for (const auto& func : functions) {
            std::cout << "Name: " << std::left << std::setw(30) << func.name
                      << " Address: 0x" << std::hex << func.value
                      << " Size: " << std::dec << func.size << " bytes" << std::endl;
        }

        std::cout << "\n--- RISC-V Attributes ---" << std::endl;
        auto riscv_attributes = parser.get_riscv_attributes();
        std::cout<<"- vendor name: "<<riscv_attributes.vendor_name<<std::endl;
        for (std::size_t a = 0; a < riscv_attributes.attributes.size(); ++a) {
            const auto& attr = riscv_attributes.attributes[a];
            std::stringstream msg;
            msg<<"- attribute: "<<a<<std::endl;
            msg<<"  - tag: "<<riscv_attributes.attributes[a].tag<<std::endl;
            msg<<"  - tag (int): "<<static_cast<int>(riscv_attributes.attributes[a].tag)<<std::endl;
            msg<<"  - name: "<<riscv_attributes.attributes[a].name<<std::endl;
            msg<<"  - is_numeric: "<<riscv_attributes.attributes[a].is_numeric<<std::endl;
            if (riscv_attributes.attributes[a].is_numeric) {
                msg<<"  - numeric_value: "<<riscv_attributes.attributes[a].numeric_value<<std::endl;
            }
            else {
                msg<<"  - value: "<<riscv_attributes.attributes[a].value<<std::endl;
                const auto& attr = riscv_attributes.attributes[a];
                if (attr.name == "ARCH") {
                    msg << "    Architecture breakdown:" << std::endl;
                    msg << "    - Base: " << (attr.value.substr(0, 4)) << std::endl;
                    if (attr.value.find("_i") != std::string::npos) {
                        msg << "    - Integer base ISA" << std::endl;
                    }
                    if (attr.value.find("_m") != std::string::npos) {
                        msg << "    - Multiply/Divide extension" << std::endl;
                    }
                    if (attr.value.find("_a") != std::string::npos) {
                        msg << "    - Atomic operations extension" << std::endl;
                    }
                    if (attr.value.find("_f") != std::string::npos) {
                        msg << "    - Single-precision floating-point" << std::endl;
                    }
                    if (attr.value.find("_d") != std::string::npos) {
                        msg << "    - Double-precision floating-point" << std::endl;
                    }
                    if (attr.value.find("_v") != std::string::npos) {
                        msg << "    - Vector extension" << std::endl;
                    }
                    if (attr.value.find("_zfh") != std::string::npos) {
                        msg << "    - Half-precision floating-point extension" << std::endl;
                    }
                    if (attr.value.find("_zvamo") != std::string::npos) {
                        msg << "    - Vector atomic memory operations" << std::endl;
                    }
                    if (attr.value.find("_zvlsseg") != std::string::npos) {
                        msg << "    - Vector load/store segment instructions" << std::endl;
                    }
                }
            }
            std::cout<<msg.str()<<std::endl;
        }

        {
            std::stringstream msg;
            for (const auto& [fun_sym, dis] : parser.decode(parser.get_section(".text"))) {
                msg<<"- function name: "<<fun_sym.name<<std::endl;
                for (std::size_t i = 0; i < dis.size(); ++i) {
                    msg<<"  - "<<dis[i]<<std::endl;
                }
            }
            std::cout<<msg.str()<<std::endl;
        }

        {
            std::stringstream msg;
            for (const auto& [fun_sym, dis] : parser.decode()) {
                msg<<"- function name: "<<fun_sym.name<<std::endl;
                for (std::size_t i = 0; i < dis.size(); ++i) {
                    msg<<"  - "<<dis[i]<<std::endl;
                }
            }
            std::cout<<msg.str()<<std::endl;
        }

        {
            std::cout<<"- riscv attribute: "<<parser.riscv_attribute_to_string()<<std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "An error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}