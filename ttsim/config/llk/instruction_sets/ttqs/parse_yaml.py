#!/usr/bin/env python3
"""
Parse YAML instruction set file and print instructions with their arguments.
Format: Mnemonic argument1[end_bit:start_bit] argument2[end_bit:start_bit] ...
Arguments are sorted by start_bit in descending order.
"""

import yaml
import sys
from pathlib import Path


def parse_and_print_instructions(yaml_file):
    """Parse YAML file and print instructions with their arguments."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
    
    if not data:
        print("Error: Empty or invalid YAML file")
        return
    
    # allowedMnemonics = [
    #         'SFPABS', 'SFPAND', 'SFPARECIP', 'SFPCOMPC', 'SFPDIVP2',
    #         'SFPENCC', 'SFPEXEXP', 'SFPEXMAN', 'SFPGT', 'SFPIADD',
    #         'SFPLE', 'SFPLZ', 'SFPMOV', 'SFPNOT', 'SFPOR', 'SFPPOPC',
    #         'SFPPUSHC', 'SFPSETCC', 'SFPSETEXP', 'SFPSETMAN',
    #         'SFPSETSGN', 'SFPSHFT', 'SFPTRANSP', 'SFPXOR']

    allowedMnemonics = ['SFPADD', 'SFPMAD', 'SFPMUL', 'SFPMUL24']
    
    # Iterate through each instruction
    for mnemonic, instr_data in data.items():
        if mnemonic not in allowedMnemonics:
            continue
        
        if not isinstance(instr_data, dict):
            continue
        
        # Get arguments list
        arguments = instr_data.get('arguments', [])
        
        if not arguments:
            # Print instruction with no arguments
            print(f"{mnemonic}")
            continue
        
        # Sort arguments by start_bit in descending order
        sorted_args = sorted(arguments, key=lambda x: x.get('start_bit', 0), reverse=True)
        
        # Build argument strings
        arg_strings = []
        for arg in sorted_args:
            name = arg.get('name', 'unknown')
            start_bit = arg.get('start_bit', 0)
            size = arg.get('size', 1)
            end_bit = start_bit + size - 1
            
            arg_strings.append(f"{name}[{end_bit}:{start_bit}]")
        
        # Print instruction with arguments
        print(f"{mnemonic} {' '.join(arg_strings)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        # Default to the assembly.nov17.yaml file
        yaml_file = Path(__file__).parent / "assembly.nov17.yaml"
    
    if not Path(yaml_file).exists():
        print(f"Error: File '{yaml_file}' not found")
        sys.exit(1)
    
    parse_and_print_instructions(yaml_file)
