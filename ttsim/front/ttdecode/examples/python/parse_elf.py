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
    functions_instructions = elf_parser.decode(elf_parser.get_section(".text"))
