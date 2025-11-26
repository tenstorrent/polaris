#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import types
import typing

from elftools.elf.elffile import ELFFile

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.instructions as instructions

def _get_attributes_section(elf: ELFFile) -> bytes:
    """Get the .riscv.attributes section data from ELF file."""
    sec_name = ".riscv.attributes"
    attr_sec = elf.get_section_by_name(sec_name)
    if not attr_sec:
        msg = f"- error: could not find section {sec_name} in given elf file"
        if hasattr(elf, 'stream') and hasattr(elf.stream, 'name'):
            msg += f" {elf.stream.name}"
        raise Exception(msg)
    return attr_sec.data()

def _validate_format_version(data: bytes) -> None:
    """Validate the RISC-V attributes format version."""
    if len(data) == 0 or data[0] != 0x41:  # ASCII 'A'
        raise ValueError("Unknown or missing attributes format version")

def _parse_vendor_section_info(data: bytes, start_idx: int) -> tuple[int, int, str]:
    """Parse vendor section length and name.

    Returns:
        tuple: (vendor_section_end, next_idx, vendor_name)
    """
    if start_idx + 4 > len(data):
        raise ValueError("Insufficient data for vendor section length")

    vendor_section_length = int.from_bytes(data[start_idx:start_idx+4], byteorder='little')
    vendor_section_start_after_length = start_idx + 4

    # Calculate vendor section end (length includes the length field itself but not the format version)
    vendor_section_end = vendor_section_start_after_length + vendor_section_length - 4
    if vendor_section_end > len(data):
        raise ValueError("Invalid vendor section length")

    # Read vendor name (null-terminated string within vendor section)
    idx = vendor_section_start_after_length
    vendor_name_chars = []
    while idx < vendor_section_end and data[idx] != 0:
        if 31 < data[idx] < 127:
            vendor_name_chars.append(chr(data[idx]))
        idx += 1
    if idx < vendor_section_end:
        idx += 1  # Skip null terminator

    vendor_name = ''.join(vendor_name_chars)
    return vendor_section_end, idx, vendor_name

def _skip_uleb128(data: bytes, start_idx: int, end_boundary: int) -> int:
    """Skip a ULEB128 encoded number and return the new index."""
    idx = start_idx
    while idx < len(data) and idx < end_boundary:
        byte = data[idx]
        idx += 1
        if (byte & 0x80) == 0:  # Last byte of ULEB128
            break
    return idx

def _parse_architecture_string(data: bytes, start_idx: int, end_boundary: int) -> tuple[int, str]:
    """Parse a null-terminated architecture string.

    Returns:
        tuple: (next_idx, arch_string)
    """
    idx = start_idx
    arch_chars = []
    while idx < end_boundary and idx < len(data) and data[idx] != 0:
        if 31 < data[idx] < 127:
            arch_chars.append(chr(data[idx]))
        idx += 1
    if idx < end_boundary and idx < len(data):
        idx += 1  # Skip null terminator
    return idx, ''.join(arch_chars)

def _parse_file_attributes_subsection(data: bytes, start_idx: int, subsection_end: int, vendor_section_end: int) -> tuple[int, str | None]:
    """Parse a File Attributes subsection and extract architecture string.

    Returns:
        tuple: (next_idx, arch_string_or_none)
    """
    idx = start_idx
    arch_string = None

    while idx < vendor_section_end and idx < subsection_end:
        if idx >= len(data):
            break

        attr_tag = data[idx]
        idx += 1

        if attr_tag == 5:  # ARCH attribute (tag 5 is architecture string)
            if arch_string is not None:
                raise Exception(f"- error: multiple ARCH attributes found, previous: {arch_string}")
            idx, arch_string = _parse_architecture_string(data, idx, vendor_section_end)
        else:
            # Skip numeric attributes (ULEB128 encoded)
            idx = _skip_uleb128(data, idx, min(vendor_section_end, subsection_end))

    return idx, arch_string

def _parse_subsections(data: bytes, start_idx: int, vendor_section_end: int) -> str | None:
    """Parse all subsections within the vendor section to find architecture string.

    Returns:
        The architecture string if found, None otherwise.
    """
    idx = start_idx
    arch_string = None

    while idx < vendor_section_end:
        if idx >= len(data):
            break

        # Read subsection tag
        subsection_tag = data[idx]
        idx += 1

        # Read subsection length
        if (idx + 4) > vendor_section_end:
            break
        subsection_length = int.from_bytes(data[idx:idx+4], byteorder='little')
        idx += 4

        # Calculate subsection end
        # subsection_length includes tag (1 byte) + length field (4 bytes) + data
        subsection_end = idx + (subsection_length - 5)
        if subsection_end > vendor_section_end:
            raise ValueError("Invalid subsection length")

        # Process subsection based on tag
        if subsection_tag == 1:  # File Attributes subsection
            idx, found_arch = _parse_file_attributes_subsection(data, idx, subsection_end, vendor_section_end)
            if found_arch is not None:
                if arch_string is not None:
                    raise Exception(f"- error: multiple ARCH attributes found, previous: {arch_string}, new: {found_arch}")
                arch_string = found_arch
        else:
            # Unknown subsection type, skip it
            print(f"- WARNING: skipping unknown subsection type {subsection_tag}")

        # Ensure we advance to the end of the subsection
        if idx < subsection_end:
            idx = subsection_end

    # Check if we decoded the complete section
    if idx < vendor_section_end:
        print(f"- WARNING: complete .riscv.attributes section not decoded")

    return arch_string

def _format_riscv_attribute_string(vendor_name: str, arch_string: str) -> str:
    """Format the final RISC-V attribute string."""
    separator = '_' if (vendor_name and not vendor_name.endswith('_')) else ''
    return f"{vendor_name}{separator}{arch_string}"

def get_riscv_attribute_from_elf_object(elf: ELFFile) -> str:
    """Parse the .riscv.attributes section and extract the architecture string."""
    # Get section data
    data = _get_attributes_section(elf)

    # Validate format version
    _validate_format_version(data)

    # Parse vendor section information
    vendor_section_end, subsection_start_idx, vendor_name = _parse_vendor_section_info(data, 1)

    # Parse subsections to find architecture string
    arch_string = _parse_subsections(data, subsection_start_idx, vendor_section_end)

    if arch_string is None:
        raise Exception("- error: could not find RISC-V architecture attribute")

    return _format_riscv_attribute_string(vendor_name, arch_string)

def get_riscv_attribute(elf: ELFFile | str) -> str:
    if isinstance (elf, str):
        with open(elf, 'rb') as file:
            elf = ELFFile(file)
            return get_riscv_attribute_from_elf_object(elf)

    elif isinstance (elf, ELFFile):
        return get_riscv_attribute_from_elf_object(elf)

    else:
        raise Exception(f"- error: no method defined for elf of type {type(elf)}")

def get_instruction_kinds_from_riscv_attribute(riscv_attribute: str) -> set[decoded_instruction.instruction_kind]:
    return decoded_instruction.get_instruction_kinds_from_riscv_attribute(riscv_attribute)

def get_instruction_kinds_from_elf_object(elf: ELFFile) -> set[decoded_instruction.instruction_kind]:
    return get_instruction_kinds_from_riscv_attribute(get_riscv_attribute_from_elf_object(elf))

def get_instruction_kinds(elf: str | ELFFile) -> set[decoded_instruction.instruction_kind]:
    if isinstance (elf, str):
        with open(elf, 'rb') as file:
            elf = ELFFile(file)
            return get_instruction_kinds_from_elf_object(elf)

    elif isinstance (elf, ELFFile):
        return get_instruction_kinds_from_elf_object(elf)

    else:
        raise Exception(f"- error: no method defined for elf of type {type(elf)}")

def get_instruction_modules_from_elf_object(elf: str | ELFFile) -> dict[decoded_instruction.instruction_kind, types.ModuleType]:
    attr = get_riscv_attribute(elf)
    if isinstance(attr, str):
        return instructions.get_default_modules_from_riscv_attribute(attr)
    elif attr is None:
        raise AssertionError("- error: could not find riscv attribute for given elf file.")
    else:
        raise AssertionError(f"- error: no method defined to obtain instruction modules from riscv attribute of type {type(attr)}")

def get_instruction_modules(elf: str | ELFFile) -> dict[decoded_instruction.instruction_kind, types.ModuleType]:
    if isinstance (elf, str):
        with open(elf, 'rb') as file:
            elf = ELFFile(file)
            return get_instruction_modules_from_elf_object(elf)

    elif isinstance (elf, ELFFile):
        return get_instruction_modules_from_elf_object(elf)

    else:
        raise Exception(f"- error: no method defined for elf of type {type(elf)}")

def get_default_instruction_sets_from_riscv_attribute(riscv_attribute: str) -> dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]:
    return instructions.get_default_instruction_sets_from_riscv_attribute(riscv_attribute)

def get_default_instruction_sets_from_elf_object(elf: ELFFile) -> dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]:
    riscv_attribute = get_riscv_attribute(elf)
    return get_default_instruction_sets_from_riscv_attribute(riscv_attribute)

def get_default_instruction_sets(elf: str | ELFFile) -> dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]:
    if isinstance (elf, str):
        with open(elf, 'rb') as file:
            elf = ELFFile(file)
            return get_default_instruction_sets_from_elf_object(elf)

    elif isinstance (elf, ELFFile):
        return get_default_instruction_sets_from_elf_object(elf)

    else:
        raise Exception(f"- error: no method defined for elf of type {type(elf)}")

def get_all_function_ranges_from_elf_object(elf: ELFFile) -> None | list[tuple[str, int, int]]:
    function_ranges: list[tuple[str, int, int]] = []

    # Access symbol table (.symtab) for function symbols
    symtab = elf.get_section_by_name('.symtab')
    if symtab is None:
        print("No symbol table found.")
        return None

    # Loop through symbols in the symbol table to get functions
    for symbol in symtab.iter_symbols():
        if symbol['st_info']['type'] == 'STT_FUNC':
            name = symbol.name
            address = symbol['st_value']
            size = symbol['st_size']
            function_ranges.append((name, address, address + size))

    return sorted(function_ranges) if len(function_ranges) else None

def get_all_function_ranges(elf: str | ELFFile) -> None | list[tuple[str, int, int]]:
    if isinstance(elf, str):
        with open(elf, 'rb') as f:
            elf = ELFFile(f)
            return get_all_function_ranges_from_elf_object(elf)

    elif isinstance(elf, ELFFile):
        return get_all_function_ranges_from_elf_object(elf)

    else:
        raise Exception(f"- error: expected argument to be either elf file name (including path) or ELFFile object. Type of given argument: {type(elf)}")

def get_all_function_names_from_elf_object(elf: ELFFile) -> list[str]:
    # Access symbol table (.symtab) for function symbols
    symtab = elf.get_section_by_name('.symtab')
    if symtab is None:
        msg = "- WARNING: no symbol table found in given elf file"
        if hasattr(elf, 'stream') and hasattr(elf.stream, 'name'):
                msg += f" {elf.stream.name}"
        print(msg)
        return []

    function_names: list[str] = []

    # Loop through symbols in the symbol table to get functions
    for symbol in symtab.iter_symbols():
        if symbol['st_info']['type'] == 'STT_FUNC':
            function_names.append(symbol.name)

    return sorted(function_names)

def get_all_function_names(elf: str | ELFFile) -> list[str]:
    if isinstance(elf, str):
        with open(elf, 'rb') as f:
            elf = ELFFile(f)
            return get_all_function_names_from_elf_object(elf)

    elif isinstance(elf, ELFFile):
        return get_all_function_names_from_elf_object(elf)

    else:
        raise Exception(f"- error: expected argument to be either elf file name (including path) or ELFFile object. Type of given argument: {type(elf)}")

def get_function_range_from_elf_object(function_name: str, elf: ELFFile) -> None | tuple[str, int, int]:
    symtab = elf.get_section_by_name('.symtab')
    if symtab is None:
        print("No symbol table found.")
        return None

    # Loop through symbols in the symbol table to get functions
    for symbol in symtab.iter_symbols():
        if symbol['st_info']['type'] == 'STT_FUNC':
            if symbol.name == function_name:
                name    = symbol.name
                address = symbol['st_value']
                size    = symbol['st_size']
                return (name, address, address + size)

    return None

def get_function_range(function_name: str, elf: str | ELFFile) -> None | tuple[str, int, int]:
    if isinstance(elf, str):
        with open(elf, 'rb') as f:
            elf = ELFFile(f)
            return get_function_range_from_elf_object(function_name, elf)

    elif isinstance(elf, ELFFile):
        return get_function_range_from_elf_object(function_name, elf)

    else:
        raise Exception(f"- error: expected argument to be either elf file name (including path) or ELFFile object. Type of given argument: {type(elf)}")

def get_instruction_bytes_of_function_from_elf_object(function_name: str, elf: ELFFile) -> bytes:
    def get_section_id (function_name: str, elf: ELFFile) -> None | int:
        for symbol in elf.get_section_by_name('.symtab').iter_symbols():
            if symbol.name == function_name:
                if symbol['st_shndx'] and symbol['st_shndx'] < elf.num_sections():
                    return symbol['st_shndx']

        return None

    # def get_section_name (function_name: str, elf: ELFFile) -> None | int:
    #     section_id = get_section_id(function_name, elf)
    #     if section_id:
    #         return elf.get_section(section_id).name

    #     return None

    name_start_end = get_function_range_from_elf_object(function_name, elf)

    if name_start_end is None:
        raise Exception(f"- error: function {function_name} not present in given elf file")

    if name_start_end[0] != function_name:
        raise AssertionError(f"- error: function name mismatch. Expected {function_name}, received {name_start_end[0]}")

    start_addr = name_start_end[1]
    end_addr   = name_start_end[2]
    section_id = get_section_id(function_name, elf)

    if section_id is None:
        raise Exception(f"- error: could not find section id to which function {function_name} belongs to.")

    sec = elf.get_section(section_id)

    if sec is None:
        raise Exception(f"- error: given elf file does not contain the section with index {section_id}")

    sec_data = sec.data()
    sec_start_addr = sec['sh_addr']

    start_from = (start_addr - sec_start_addr)
    end_at = (end_addr - sec_start_addr)
    if (start_from < 0) or end_at > len(sec_data):
        raise Exception(f"- error: function exceeds section bounds. function name: {function_name}")

    # func_data = sec_data[(start_addr - sec_start_addr) : (end_addr - sec_start_addr)]
    func_data = sec_data[start_from : end_at]

    return func_data

def get_instruction_bytes_of_function(function_name: str, elf: str | ELFFile) -> bytes:
    if isinstance (elf, str):
        with open(elf, 'rb') as file:
            elf = ELFFile(file)
            return get_instruction_bytes_of_function_from_elf_object(function_name, elf)

    elif isinstance(elf, ELFFile):
        return get_instruction_bytes_of_function_from_elf_object(function_name, elf)

    else:
        raise Exception(f"- error: no method defined to get instruction from elf of type {type(elf)}")

def contains_function(function_name: str, elf: str | ELFFile) -> bool:
    function_names = get_all_function_names(elf)
    if function_names is None:
        return False
    else:
        return function_name in function_names

def add_program_counter(start_address: int, end_address: int, instruction_list: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction]) -> None:
    num_instrs = int((end_address - start_address) / instructions.decoded_instruction.get_num_bytes_per_instruction())

    if num_instrs != len(instruction_list):
        raise Exception(f"- error: number of instructions mismatch. len(instructions): {len(instruction_list)}. number of instructions: {num_instrs}")

    pc = start_address
    for idx in range(num_instrs):
        instruction_list[idx].set_program_counter(pc)
        pc += instructions.decoded_instruction.get_num_bytes_per_instruction()

def decode_function_from_elf_object(
    function_name: str,
    elf: ELFFile,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> list[decoded_instruction.decoded_instruction]:
    if not contains_function(function_name, elf):
        err_msg = f"- error: could not find function {function_name} in given elf file"
        if hasattr(elf, 'stream') and hasattr(elf.stream, 'name'):
                err_msg += f" {elf.stream.name}"
        raise Exception(err_msg)

    fn_range = get_function_range(function_name, elf)
    if fn_range:
        if fn_range[0] != function_name:
            raise AssertionError(f"- error: function name mismatch. expected: {function_name}, received: {fn_range[0]}")
        start_addr = fn_range[1]
        end_addr = fn_range[2]

        func_data = get_instruction_bytes_of_function_from_elf_object(function_name, elf)
        kinds = get_instruction_kinds_from_elf_object(elf)
        m_modules: dict[decoded_instruction.instruction_kind, types.ModuleType] = dict()
        m_sets: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]] = dict()
        for k in kinds:
            m_modules[k] = instructions.get_module(k, modules)
            m_sets[k] = instructions.get_instruction_set(k, instruction_sets = sets)

        if isinstance(sets, str) or decoded_instruction.is_instruction_set_dict_instance(sets):
            assert 1 == len(kinds), f"- error: expected only one instruction kind as given instruction set is of kind str or instruction_set_dict, but received multiple kinds. kinds: {kinds}"
            assert 1 == len(m_modules), f"- error: expected only one module as given instruction set is of kind str or instruction_set_dict, but received multiple kinds. kinds: {kinds}"
        instruction_list = instructions.decode_instructions(func_data, kinds, m_modules, m_sets, are_swizzled = True)
        add_program_counter(start_addr, end_addr, instruction_list)

        return instruction_list

    else:
        err_msg = f"- error: could not obtain function range for function {function_name} in given elf file"
        if hasattr(elf, 'stream') and hasattr(elf.stream, 'name'):
            err_msg += f" {elf.stream.name}"
        raise Exception(err_msg)

def decode_function(
    func_name: str,
    elf: str | ELFFile,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> list[decoded_instruction.decoded_instruction]:
    # decode_function:
    #   - func_name   : str
    #   - elf         : str (file_name), ELFFile
    #   - modules     : None, dict of {instruction_kind : module}
    #   - sets        : None, dict of {instruction_kind : None | str | instruction_set_dict}

    # elf is file
    if isinstance (elf, str):
        with open(elf, 'rb') as file:
            elf = ELFFile(file)
            return decode_function_from_elf_object(
                func_name,
                elf,
                modules,
                sets)

    elif isinstance(elf, ELFFile):
        return decode_function_from_elf_object(
            func_name,
            elf,
            modules,
            sets)

    else:
        raise Exception(f"- error: no method defined to read elf of type {type(elf)}")

def decode_functions_from_elf_object(
    function_names: list[str] | tuple[str] | set[str],
    elf: ELFFile,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[str, list[decoded_instruction.decoded_instruction]]:
    kinds = get_instruction_kinds_from_elf_object(elf)
    modules = instructions.get_modules(kinds = kinds, modules = modules)
    instruction_sets = typing.cast(dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]], instructions.get_instruction_sets(kinds, modules, sets))
    decoded_instrs: dict[str, list[decoded_instruction.decoded_instruction]] = dict()
    for func_name in sorted(set(function_names)):
        decoded_instrs[func_name] = decode_function(func_name, elf, modules = modules, sets = instruction_sets)

    return decoded_instrs

def decode_functions(
    function_names: list[str] | tuple[str] | set[str],
    elf: str | ELFFile,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[str, list[decoded_instruction.decoded_instruction]]:
    # elf is file
    if isinstance (elf, str):
        with open(elf, 'rb') as file:
            elf = ELFFile(file)
            return decode_functions_from_elf_object(
                function_names,
                elf,
                modules,
                sets)

    elif isinstance(elf, ELFFile):
        return decode_functions_from_elf_object (
            function_names,
            elf,
            modules,
            sets)

    else:
        raise Exception(f"- error: no method defined to decode functions from elf of type {type(elf)}")

def decode_all_functions_from_elf_object(
    elf: ELFFile,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[str, list[decoded_instruction.decoded_instruction]]:
    return decode_functions_from_elf_object(get_all_function_names_from_elf_object(elf), elf, modules, sets)

def decode_all_functions(
    elf: str | ELFFile,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[str, list[decoded_instruction.decoded_instruction]]:
    # elf is file
    if isinstance (elf, str):
        with open(elf, 'rb') as file:
            elf = ELFFile(file)

            return decode_all_functions_from_elf_object(
                elf,
                modules,
                sets)

    elif isinstance(elf, ELFFile):
        return decode_all_functions_from_elf_object(
            elf,
            modules,
            sets)

    else:
        raise Exception(f"- error: no method defined to decode all functions from elf of type {type(elf)}")

@typing.overload
def get_instruction_profile(
    file_name: str,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: typing.Literal[True] = True) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] | dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]]: ...

@typing.overload
def get_instruction_profile(
    file_name: str,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: typing.Literal[False] = False) -> tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]] | dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]]: ...

@typing.overload
def get_instruction_profile(
    file_name: str,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: bool = False) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] | tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]] | dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]] | dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]]: ...

@typing.overload
def get_instruction_profile(
    file_name: list[str] | tuple[str] | set[str],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: typing.Literal[True] = True) -> dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]]: ...

@typing.overload
def get_instruction_profile(
    file_name: list[str] | tuple[str] | set[str],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: typing.Literal[False] = False) -> dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]]: ...

@typing.overload
def get_instruction_profile(
    file_name: list[str] | tuple[str] | set[str],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: bool = False) -> dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]] | dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]]: ...

def get_instruction_profile(
    file_name: str | list[str] | tuple[str] | set[str],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: bool = False) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] | tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]] | dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]] | dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]]:

    def get_elf_files_from_dir(path: str):
        elf_files: list[str] = list()
        for pwd, _, files in os.walk(path):
            for file in files:
                if file.endswith(".elf"):
                    elf_files.append(os.path.join(pwd, file))

        return elf_files

    def get_elf_files_from_list(path: list[str] | set[str] | tuple[str]):
        elf_files: list[str] = list()
        for ele in set(path):
            if os.path.isfile(ele):
                elf_files.append(ele)
            elif os.path.isdir(ele):
                elf_files.extend(get_elf_files_from_dir(ele))
            else:
                raise Exception(f"- error: given path element is neither a file, nor a directory. path element: {ele}")

        return elf_files

    if isinstance(file_name, str):
        # if it is a file
        if os.path.isfile(file_name):
            return instructions.get_statistics(decode_all_functions(file_name, modules, sets), modules, flatten_dict)

        # if it is directory
        elif os.path.isdir(file_name):
            return get_instruction_profile(get_elf_files_from_dir(file_name), modules, sets, flatten_dict)

        else:
            raise Exception(f"- error: could not find elf file(s) associated with {file_name}")

    elif isinstance(file_name, (list, tuple, set)):
        elf_files = get_elf_files_from_list(file_name)
        if flatten_dict:
            flat_ips: dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]] = dict()
            for file in elf_files:
                flat_ips[file] = get_instruction_profile(file, modules, sets, flatten_dict)

            return flat_ips
        else:
            ips: dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]] = dict()
            for file in elf_files:
                ips[file] = get_instruction_profile(file, modules, sets, flatten_dict)

            return ips

    else:
        raise Exception(f"- error: no method defined to get instruction profile of file names of data type {type(file_name)}")

# @typing.overload
# def print_instruction_profile(
#     instruction_profile: tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] |
#     tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]] |
#     dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]] |
#     dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]],
#     modules: None,
#     sets: None,
#     print_instructions: None,
#     print_statistics: None,
#     flatten_dict: None,
#     preamble: str = "",
#     print_offset: int = 2) -> None: ...

# @typing.overload
# def print_instruction_profile(
#     instruction_profile: str | set[str],
#     modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
#     sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
#     print_instructions: bool = False,
#     print_statistics: bool = True,
#     flatten_dict: bool = True,
#     preamble: str = "",
#     print_offset: int = 2) -> None: ...

def print_instruction_profile(
    instruction_profile:
        tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] |
        tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]] |
        dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]] |
        dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]] | str | set[str],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    print_instructions: None | bool = False,
    print_statistics: None | bool = True,
    flatten_dict: None | bool = True,
    preamble: str = "",
    print_offset: int = 2) -> None:
    def is_histogram_dict(arg: typing.Any) -> bool:
        if not isinstance(arg, dict):
            return False

        if not all(isinstance(key, decoded_instruction.instruction_kind) for key in arg.keys()):
            return False

        if not all(isinstance(val, dict) for val in arg.values()):
            return False

        if not all(isinstance(key, str) for val in arg.values() for key in val.keys()):
            return False

        if not all(isinstance(vval, list) for val in arg.values() for vval in val.values()):
            return False

        for val in arg.values():
            for vval in val.values():
                if not all(isinstance(ele, (int, float)) for ele in vval):
                    return False

        return True

    def is_kind_histogram_dict(arg: typing.Any) -> bool:
        if not isinstance(arg, dict):
            return False

        if not all(isinstance(key, decoded_instruction.instruction_kind) for key in arg.keys()):
            return False

        if not all(isinstance(val, dict) for val in arg.values()):
            return False

        if not all(isinstance(key, str) for val in arg.values() for key in val.keys()):
            return False

        if not all(isinstance(vval, int) for val in arg.values() for vval in val.values()):
            return False

        return True

    def is_tuple_type1(instruction_profile: typing.Any) -> typing.TypeGuard[tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]]:
        if not isinstance(instruction_profile, tuple):
            return False

        if 2 != len(instruction_profile):
            return False

        if not is_histogram_dict(instruction_profile[0]):
            return False

        if not is_kind_histogram_dict(instruction_profile[1]):
            return False

        return True

    def is_tuple_type2(instruction_profile: typing.Any) -> typing.TypeGuard[tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]]:
        if not isinstance(instruction_profile, tuple):
            return False

        if 2 != len(instruction_profile):
            return False

        if not all(isinstance(ele, dict) for ele in instruction_profile):
            return False

        if not all(isinstance(key, str) for ele in instruction_profile for key in ele.keys()):
            return False

        if not all(is_histogram_dict(val) for val in instruction_profile[0].values()):
            return False

        if not all(is_kind_histogram_dict(val) for val in instruction_profile[1].values()):
            return False

        return True

    def is_dict_type1(instruction_profile: typing.Any) -> typing.TypeGuard[dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]]]:
        if not isinstance(instruction_profile, dict):
            return False

        if not all(isinstance(key, str) for key in instruction_profile.keys()):
            return False

        if not all(is_tuple_type1(val) for val in instruction_profile.values()):
            return False

        return True

    def is_dict_type2(instruction_profile: typing.Any) -> typing.TypeGuard[dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]]]:
        if not isinstance(instruction_profile, dict):
            return False

        if not all(isinstance(key, str) for key in instruction_profile.keys()):
            return False

        if not all(is_tuple_type2(val) for val in instruction_profile.values()):
            return False

        return True

    def print_instruction_profile_tuple(
        instruction_profile: tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] | tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]],
        preamble: str,
        print_offset: int) -> None:
        if preamble:
            print(preamble)
        print(f"{' ' * print_offset}- Instruction histogram")
        instructions.print_instruction_histogram(instruction_profile[0], print_offset = print_offset + 2)
        print(f"{' ' * print_offset}- Instruction kind histogram")
        instructions.print_instruction_kind_histogram(instruction_profile[1], print_offset = print_offset + 2)

    def print_instruction_profile_dict(
        instruction_profile: dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]] | dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]],
        preamble: str,
        print_offset: int) -> None:
        #TODO: print preemble
        for key, profile in instruction_profile.items():
            print_instruction_profile_tuple(profile, key, print_offset)

    # def print_instruction_profile_tuple_dict(
    #     instruction_profile: tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] |
    #     tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]] |
    #     dict[str, tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]] |
    #     dict[str, tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]],
    #     preamble: str,
    #     print_offset: int) -> None:
    #     if isinstance(instruction_profile, tuple):
    #         if preamble:
    #             print(preamble)
    #         print(f"{' ' * print_offset}- Instruction histogram")
    #         instructions.print_instruction_histogram(instruction_profile[0], print_offset = print_offset + 2)
    #         print(f"{' ' * print_offset}- Instruction kind histogram")
    #         instructions.print_instruction_kind_histogram(instruction_profile[1], print_offset = print_offset + 2)
    #     elif isinstance(instruction_profile, dict):
    #         for key, profile in instruction_profile.items():
    #             print_instruction_profile_tuple_dict(profile, key, print_offset)
    #     else:
    #         raise Exception(f"- error: no method defined to print instruction profile for instruction profile of type {type(instruction_profile)}")

    def print_instruction_profile_of_file(
        elf_file_name: str,
        modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType],
        sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]],
        print_instructions: bool,
        print_statistics: bool,
        flatten_dict: bool,
        preamble: str,
        print_offset: int) -> None:
        instrs = decode_all_functions(elf_file_name, modules, sets)

        if print_instructions:
            instructions.print_instructions(instrs)

        if print_statistics:
            print_instruction_profile_tuple(instructions.get_statistics(instructions = instrs, modules = modules, flatten_dict = flatten_dict), preamble, print_offset)

    def print_instruction_profile_of_files(path: str | set[str],
        modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType],
        sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]],
        print_instructions: bool,
        print_statistics: bool,
        flatten_dict: bool,
        preamble: str,
        print_offset: int) -> None:
        if isinstance(path, str):
            if os.path.isfile(path):
                print_instruction_profile_of_file(path, modules, sets, print_instructions, print_statistics, flatten_dict, preamble, print_offset)
            elif os.path.isdir(path):
                elf_files: list[str] = list()
                dirs = os.walk(path)
                for pwd, _, files in dirs:
                    for file_name in files:
                        if file_name.endswith(".elf"):
                            file_name_incl_path = os.path.join(pwd, file_name)
                            # print(f"reading file: {file_name_incl_path}")
                            elf_files.append(file_name_incl_path)
                            # print_instruction_profile_of_file(file_name_incl_path, modules, sets, print_instructions, print_statistics, flatten_dict, preamble, print_offset)
                for elf_file in sorted(elf_files):
                    print(f"reading file: {elf_file}")
                    print_instruction_profile_of_file(file_name_incl_path, modules, sets, print_instructions, print_statistics, flatten_dict, preamble, print_offset)

            else:
                raise Exception(f"- error: given string is neither a file name, nor a directory. path: {path}")
        elif isinstance(path, set):
            for ele in path:
                print_instruction_profile_of_files(ele, modules, sets, print_instructions, print_statistics, flatten_dict, preamble, print_offset)
        else:
            raise Exception(f"- error: no method defined to print instruction profile for path of type {type(path)}")

    if is_tuple_type1(instruction_profile) or is_tuple_type2(instruction_profile):
        print_instruction_profile_tuple(instruction_profile, preamble, print_offset)
    elif is_dict_type1(instruction_profile) or is_dict_type2(instruction_profile):
        print_instruction_profile_dict(instruction_profile, preamble, print_offset)
    elif isinstance(instruction_profile, (str, set)):
        if isinstance(instruction_profile, set):
            assert all(isinstance(ele, str) for ele in instruction_profile), f"- error: expected each element of set instruction_profile to be str"
        assert isinstance(print_instructions, bool), f"- error: expected print_instructions to be bool, received {type(print_instructions)}"
        assert isinstance(print_statistics,   bool), f"- error: expected print_statistics to be bool, received {type(print_statistics)}"
        assert isinstance(flatten_dict,       bool), f"- error: expected flatten_dict to be bool, received {type(flatten_dict)}"
        print_instruction_profile_of_files(instruction_profile, modules, sets, print_instructions, print_statistics, flatten_dict, preamble, print_offset)
    else:
        raise TypeError(f"- error: no method defined to print instruction profile of type: {type(instruction_profile)}")

    # if isinstance(instruction_profile, (tuple, dict)):
    #     print_instruction_profile_tuple_dict(instruction_profile, preamble, print_offset)
    # elif isinstance(instruction_profile, (str, set)):
    #     assert isinstance(print_instructions, bool), f"- error: expected print_instructions to be bool, received {type(print_instructions)}"
    #     assert isinstance(print_statistics,   bool), f"- error: expected print_statistics to be bool, received {type(print_statistics)}"
    #     assert isinstance(flatten_dict,       bool), f"- error: expected flatten_dict to be bool, received {type(flatten_dict)}"
    #     print_instruction_profile_of_files(instruction_profile, modules, sets, print_instructions, print_statistics, flatten_dict, preamble, print_offset)
    # else:
    #     raise TypeError(f"- error: no method defined to print instruction profile of type: {type(instruction_profile)}")

def any_unknown_instructions_in_elf(
    path: str,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    print_instructions: bool = False,
    print_statistics: bool = True) -> bool | list[str]:
    def any_unknown_instructions(kind_hist: dict[decoded_instruction.instruction_kind, dict[str, int]]) -> bool:
        for _, hist in kind_hist.items():
            if hist["no_opcode"] or hist["no_mnemonic"]:
                return True

        return False

    def contains_unknown_instruction(elf: str,
        modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType],
        sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]],
        print_instructions: bool,
        print_statistics: bool) -> bool:
        instrs = decode_all_functions(elf, modules, sets)
        instr_hist, kind_hist = instructions.get_statistics(instrs, modules, flatten_dict = True)
        if any_unknown_instructions(kind_hist):
            print(f"file {elf} contains unkwown instructions, print_instructions: {print_instructions}, print_statistics: {print_statistics}")
            if print_instructions:
                instructions.print_instructions(instrs, print_offset=2)

            if print_statistics:
                instructions.print_instruction_histogram(instr_hist, print_offset=2)
                instructions.print_instruction_kind_histogram(kind_hist, print_offset=2)

            return True

        return False

    if isinstance(path, str):
        if os.path.isfile(path):
            return contains_unknown_instruction(path, modules, sets, print_instructions = print_instructions, print_statistics = print_statistics)
        elif os.path.isdir(path):
            dirs = os.walk(path)
            unknowns: list[str] = list()
            for pwd, _, files in dirs:
                for file_name in files:
                    if file_name.endswith(".elf"):
                        file_name_incl_path = os.path.join(pwd, file_name)
                        if any_unknown_instructions_in_elf(file_name_incl_path, modules = modules, sets = sets, print_instructions = print_instructions, print_statistics = print_statistics):
                            unknowns.append(file_name_incl_path)
            return sorted(unknowns)
        else:
            raise Exception(f"- error: given path is neither a file nor a directory. path: {path}")
    else:
        raise Exception(f"- error: no method defined to find unknown instructions (if any) in path of type {type(path)}")

@typing.overload
def get_coverage(
    path: str,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: typing.Literal[True] = True) -> dict[decoded_instruction.instruction_kind, float]: ...

@typing.overload
def get_coverage(
    path: str,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: typing.Literal[False] = False) -> dict[str, dict[decoded_instruction.instruction_kind, float]] | dict[str, dict[str, dict[decoded_instruction.instruction_kind, float]]]: ...

@typing.overload
def get_coverage(
    path: str,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: bool = False) -> dict[decoded_instruction.instruction_kind, float] | dict[str, dict[decoded_instruction.instruction_kind, float]] | dict[str, dict[str, dict[decoded_instruction.instruction_kind, float]]]: ...

def get_coverage(
    path: str,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: bool = False) -> dict[decoded_instruction.instruction_kind, float] | dict[str, dict[decoded_instruction.instruction_kind, float]] | dict[str, dict[str, dict[decoded_instruction.instruction_kind, float]]]:

    if os.path.isdir(path):
        if flatten_dict:
            coverage: dict[decoded_instruction.instruction_kind, float] = dict()
            kinds_mnemonics: dict[decoded_instruction.instruction_kind, set[str]] = dict()
            for pwd, _, files in os.walk(path):
                for file_name in files:
                    if file_name.endswith(".elf"):
                        file_name_incl_path = os.path.join(pwd, file_name)
                        with open(file_name_incl_path, 'rb') as f:
                            elf     = ELFFile(f)
                            kinds   = get_instruction_kinds_from_elf_object(elf)
                            m_modules = instructions.get_modules(kinds, modules)
                            m_sets    = sets
                            m_sets    = typing.cast(dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]], instructions.get_instruction_sets(kinds, m_modules, m_sets))
                            instrs  = decode_all_functions_from_elf_object(elf, m_modules, m_sets)
                            instr_hist, _ = instructions.get_statistics(instrs, m_modules, flatten_dict)
                            for instr_kind, instr_counter in instr_hist.items():
                                if instr_kind not in kinds_mnemonics.keys():
                                    kinds_mnemonics[instr_kind] = set()

                                kinds_mnemonics[instr_kind].update(set(instr_counter.keys()))

            for k, v in kinds_mnemonics.items():
                kinds = set([k])
                m_modules = instructions.get_modules(kinds, modules)
                m_instruction_sets = instructions.get_instruction_sets(kinds, m_modules, sets)
                coverage.update({k : len(v) / len(m_instruction_sets[k])})

            if len(kinds_mnemonics) != len(coverage):
                raise Exception(f"- error: kinds_mnemonics.keys(): {kinds_mnemonics.keys()}, coverage = {coverage}")

            return coverage
        else:
            cov: dict[str, dict[str, dict[decoded_instruction.instruction_kind, float]]] = dict()
            for pwd, _, files in os.walk(path):
                for file_name in files:
                    if file_name.endswith(".elf"):
                        file_name_incl_path = os.path.join(pwd, file_name)
                        m_modules = modules
                        m_sets = sets
                        cov[file_name_incl_path] = instructions.get_coverage(decode_all_functions(file_name_incl_path, m_modules, m_sets), m_modules, m_sets, flatten_dict = flatten_dict)

            return cov
    else:
        return instructions.get_coverage(decode_all_functions(path, modules, sets), modules, sets, flatten_dict = flatten_dict)

# todo: decode_section
if "__main__" == __name__:
    if len(sys.argv) < 2:
        raise Exception("- Please provide elf file/directory containing elf file(s)")

    elf_files = []
    for pwd, _, file_names in os.walk(sys.argv[1]):
        for file_name in file_names:
            if file_name.endswith(".elf"):
                elf_files.append(os.path.join(pwd, file_name))

    elf_files = sorted(elf_files)
    print("number of elf files: ", len(elf_files))
    for elf_file in elf_files:
        print(f"file: {elf_file}")
        print_instruction_profile(elf_file, print_instructions = True, print_statistics = False)


    # print_instructions_from_elf(sys.argv[1], print_instructions = False, print_statistics = True)

    # ip = get_instruction_profile(sys.argv[1], flatten_dict = False)
    # print_instruction_profile(ip)

    # ip = get_instruction_profile(sys.argv[1], flatten_dict = True)
    # print_instruction_profile(ip)

    # cov = get_coverage(sys.argv[1], flatten_dict = True)
    # print(cov)

    # any_unknown_instructions_in_elf(sys.argv[1])