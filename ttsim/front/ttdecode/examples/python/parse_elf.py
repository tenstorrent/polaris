#!/usr/bin/env python

import ttdecode
import os
import sys

def main(argv):
    if len(argv) < 2:
        print("- please provide the name of the elf file")
        exit()
    
    elf_file = argv[1]
    assert os.path.isfile(elf_file), f"- error: could not find {elf_file}"
    elf_parser = ttdecode.elf.parser(elf_file)
    functions_instructions = elf_parser.decode({ttdecode.isa.instruction_kind.ttqs : "../../config/llk/instruction_sets/ttqs/assembly.sep23.yaml"})
    for func_sym, instructions in functions_instructions.items():
        print("- function name: ", func_sym.name)
        for instr in instructions:
            print(f"  - {instr}")

if __name__ == "__main__":
    main(sys.argv)
