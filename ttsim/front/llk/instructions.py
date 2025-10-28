#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import collections
import importlib
import types
import typing

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.help.utils as help_utils
from ttsim.front.llk.decoded_instruction import instruction_kind as instruction_kind
from ttsim.front.llk.decoded_instruction import swizzle_instruction as swizzle_instruction
from ttsim.front.llk.decoded_instruction import unswizzle_instruction as unswizzle_instruction


def get_instruction_kinds(
    kinds: None | decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind] | set[decoded_instruction.instruction_kind],
    modules: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType],
    instruction_sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]]) -> set[decoded_instruction.instruction_kind]:
    def get_kinds_from_modules_and_instruction_sets(
        modules: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType],
        instruction_sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]]) -> set[decoded_instruction.instruction_kind]:
        kinds: set[decoded_instruction.instruction_kind] = set()
        if isinstance(modules, types.ModuleType) and hasattr(modules, 'instruction_kind'):
            kinds.add(modules.instruction_kind())
        elif decoded_instruction.is_instruction_kinds_modules_dict_instance(modules):
            kinds = kinds.union(modules.keys())

        if isinstance(instruction_sets, decoded_instruction.instruction_kind):
            kinds.add(instruction_sets)
        elif decoded_instruction.is_kinds_instruction_sets_dict_instance(instruction_sets):
            kinds = kinds.union(instruction_sets.keys())

        return kinds

    all_kinds: set[decoded_instruction.instruction_kind] = set()
    if isinstance(kinds, decoded_instruction.instruction_kind):
        all_kinds.add(kinds)
    elif kinds and all(isinstance(kind, decoded_instruction.instruction_kind) for kind in kinds):
        all_kinds = all_kinds.union(kinds)

    all_kinds = all_kinds.union(get_kinds_from_modules_and_instruction_sets(modules, instruction_sets))

    return all_kinds

def get_modules(kinds: None | decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] | set[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind] = None,
    modules: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    default_module_path: str = "ttsim.front.llk") -> dict[decoded_instruction.instruction_kind, types.ModuleType]:
    m_kinds = sorted(get_instruction_kinds(kinds = kinds, modules = modules, instruction_sets = None))
    m_kinds_modules: dict[decoded_instruction.instruction_kind, types.ModuleType] = dict()
    if decoded_instruction.is_instruction_kinds_modules_dict_instance(modules):
        for kind in m_kinds:
            if kind in modules.keys():
                if hasattr(modules[kind], 'instruction_kind') and modules[kind].instruction_kind() != kind:
                    raise Exception(f"- error: instruction kind mismatch. the module {modules[kind]} with instruction kind: {modules[kind].instruction_kind()} is associated with instruction_kind: {kind}")

                m_kinds_modules[kind] = modules[kind]

    elif isinstance(modules, types.ModuleType):
        if not hasattr(modules, 'instruction_kind'):
            raise Exception(f"- error: module {modules} does not have instruction_kind as attribute/function")
        m_kinds_modules[modules.instruction_kind()] = modules

    for kind in m_kinds:
        if kind not in m_kinds_modules.keys():
            m_kinds_modules[kind] = importlib.import_module(f"{default_module_path}.{kind}")

    return m_kinds_modules

def get_module(kind: decoded_instruction.instruction_kind,
    modules: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None) -> types.ModuleType:
    if isinstance(kind, decoded_instruction.instruction_kind):
        m_kinds_modules = get_modules(kind, modules)
        return m_kinds_modules[kind]
    else:
        raise Exception(f"- error: no method defined to get module from instruction kind of type {kind}")

def get_default_modules(kinds: decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] | set[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind]) -> dict[decoded_instruction.instruction_kind, types.ModuleType]:
    return get_modules(kinds)

def get_default_module(kind: decoded_instruction.instruction_kind) -> types.ModuleType:
    if isinstance(kind, decoded_instruction.instruction_kind):
        return get_default_modules(kind)[kind]
    else:
        raise Exception(f"- error: no method defined to get module from instruction kind of type {type(kind)}")

def get_default_modules_from_riscv_attribute(riscv_attribute: str) -> dict[decoded_instruction.instruction_kind, types.ModuleType]:
    return get_default_modules(decoded_instruction.get_instruction_kinds_from_riscv_attribute(riscv_attribute))

def get_instruction_sets(
    kinds: None | decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] | set[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind] = None,
    modules: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]:
    m_kinds = get_instruction_kinds(kinds, modules, instruction_sets)
    m_modules = get_modules(kinds = m_kinds, modules = modules)
    m_kinds_instruction_sets: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]] = dict()

    if isinstance(instruction_sets, str) or decoded_instruction.is_instruction_set_dict_instance(instruction_sets):
        assert 1 == len(m_kinds), f"- error: expected only one instruction kind as argument instruction_sets is of type str or instruction_set_dict. m_kinds: {m_kinds}"
        assert 1 == len(m_modules), f"- error: expected only one module as argument instruction_sets is of type str or instruction_set_dict. m_modules: {m_modules}"
        for kind in m_kinds:
            m_kinds_instruction_sets[kind] = m_modules[kind].get_instruction_set(instruction_sets)
    elif decoded_instruction.is_kinds_instruction_sets_dict_instance(instruction_sets):
        for kind in m_kinds:
            if kind in instruction_sets.keys():
                m_kinds_instruction_sets[kind] = m_modules[kind].get_instruction_set(instruction_sets[kind])

    for kind in m_kinds:
        if kind not in m_kinds_instruction_sets.keys():
            m_kinds_instruction_sets[kind] = m_modules[kind].get_instruction_set()

    return m_kinds_instruction_sets

def get_instruction_set(kind: decoded_instruction.instruction_kind,
    modules: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[str, typing.Any]:
    if not isinstance(kind, decoded_instruction.instruction_kind):
        raise Exception(f"- error: please provide correct instruction kind. accepted instruction kinds: {decoded_instruction.get_instruction_kinds()}")

    m_kinds_instruction_sets = get_instruction_sets(kind, modules, instruction_sets)
    return m_kinds_instruction_sets[kind]

def get_default_instruction_sets(kinds: decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] | set[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind]) -> dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]:
    return get_instruction_sets(kinds, modules = None, instruction_sets = None)

def get_default_instruction_set(kind: decoded_instruction.instruction_kind) -> dict[str, typing.Any]:
    if isinstance(kind, decoded_instruction.instruction_kind):
        m_kinds_instruction_sets = get_instruction_sets(kind, modules = None, instruction_sets = None)
        return m_kinds_instruction_sets[kind]
    else:
        raise Exception(f"- error: no method defined to get default instruction set from instruction_kind of type {type(kind)}, kind: {kind}")

def get_default_instruction_sets_from_riscv_attribute(riscv_attribute: str) -> dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]:
    return get_default_instruction_sets(decoded_instruction.get_instruction_kinds_from_riscv_attribute(riscv_attribute))

def get_valid_instruction_kinds(instruction: int,
    kinds: None | decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] | set[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind] = decoded_instruction.get_instruction_kinds(),
    modules: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    is_swizzled: bool = True,
    raise_exception: bool = False) -> set[decoded_instruction.instruction_kind]:
    m_kinds = get_instruction_kinds(kinds, modules, instruction_sets = None)
    m_modules = get_modules(kinds, modules)

    valid_kinds: set[decoded_instruction.instruction_kind] = set()
    for kind in m_kinds:
        if m_modules[kind].is_valid_instruction(instruction, is_swizzled):
            valid_kinds.add(kind)

    if 0 == len(valid_kinds) and raise_exception:
        raise Exception(f"- error: could not determine kind for given instruction: {hex(instruction)}, is_swizzled: {is_swizzled}")

    return valid_kinds

def get_valid_instruction_kind(instruction: int,
    kinds: None | decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] | set[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind] = decoded_instruction.get_instruction_kinds(),
    modules: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    is_swizzled: bool = True,
    raise_exception: bool = False) -> decoded_instruction.instruction_kind | set[decoded_instruction.instruction_kind]:
    valid_kinds = get_valid_instruction_kinds(instruction, kinds, modules, is_swizzled, raise_exception)

    if 0 == len(valid_kinds) and raise_exception:
        raise Exception(f"- error: could not determine kind for given instruction: {hex(instruction)}, is_swizzled: {is_swizzled}")

    if 1 == len(valid_kinds):
        for kind in valid_kinds:
            return kind

    return valid_kinds

@typing.overload
def get_instruction_kinds_from_decoded_instructions(
    decoded_instructions: decoded_instruction.decoded_instruction | list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...],
    flatten_dict: bool = False) -> set[decoded_instruction.instruction_kind]: ...

@typing.overload
def get_instruction_kinds_from_decoded_instructions(
    decoded_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    flatten_dict: typing.Literal[True] = True) -> set[decoded_instruction.instruction_kind]: ...

@typing.overload
def get_instruction_kinds_from_decoded_instructions(
    decoded_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    flatten_dict: typing.Literal[False] = False) -> dict[str, set[decoded_instruction.instruction_kind]]: ...

@typing.overload
def get_instruction_kinds_from_decoded_instructions(
    decoded_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    flatten_dict: bool = False) -> set[decoded_instruction.instruction_kind] | dict[str, set[decoded_instruction.instruction_kind]]: ...

def get_instruction_kinds_from_decoded_instructions(
    decoded_instructions: decoded_instruction.decoded_instruction | list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...] | dict[str, list[decoded_instruction.decoded_instruction]],
    flatten_dict: bool = False) -> set[decoded_instruction.instruction_kind] | dict[str, set[decoded_instruction.instruction_kind]]:
    def get_kinds_from_list(dis: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...]) -> set[decoded_instruction.instruction_kind]:
        kinds: set[decoded_instruction.instruction_kind] = set()
        for di in dis: # di: decoded instruction
            kinds.add(di.kind)

        return kinds

    def flatten_dict_and_get_kinds(funcs_dis: dict[str, list[decoded_instruction.decoded_instruction]]) -> set[decoded_instruction.instruction_kind]:
        kinds: set[decoded_instruction.instruction_kind] = set()
        for _, dis in funcs_dis.items():
            kinds = kinds.union(get_kinds_from_list(dis))

        return kinds

    def get_kinds_from_dict(funcs_dis: dict[str, list[decoded_instruction.decoded_instruction]]) -> dict[str, set[decoded_instruction.instruction_kind]]:
        kinds: dict[str, set[decoded_instruction.instruction_kind]] = dict()
        for func_name, dis in funcs_dis.items():
            kinds[func_name] = get_kinds_from_list(dis)

        return kinds

    if isinstance(decoded_instructions, decoded_instruction.decoded_instruction):
        if hasattr(decoded_instructions, 'kind'):
            return set([decoded_instructions.kind])
        else:
            return set()
    if isinstance(decoded_instructions, (list, tuple)):
        return get_kinds_from_list(decoded_instructions)
    elif isinstance(decoded_instructions, dict):
        if flatten_dict:
            return flatten_dict_and_get_kinds(decoded_instructions)
        else:
            return get_kinds_from_dict(decoded_instructions)

@typing.overload
def decode_instruction(instruction: int,
    kind: decoded_instruction.instruction_kind,
    module: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = ...,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = ...,
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

@typing.overload
def decode_instruction(instruction: int,
    *,
    module: types.ModuleType,
    instruction_set: None | str | dict[str, typing.Any] = None,
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

@typing.overload
def decode_instruction(instruction: int,
    *,
    instruction_set: decoded_instruction.instruction_kind,
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

@typing.overload
def decode_instruction(instruction: int,
    *,
    instruction_set: str | dict[str, typing.Any],
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction | list[decoded_instruction.decoded_instruction]: ... # list will never be returned as it will throw an error first

@typing.overload
def decode_instruction(instruction: int,
    *,
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction | list[decoded_instruction.decoded_instruction]: ...

def decode_instruction(instruction: int,
    kind: None | decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] = None,
    module: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    is_swizzled: bool = True) -> decoded_instruction.decoded_instruction | list[decoded_instruction.decoded_instruction]:
    m_kinds: set[decoded_instruction.instruction_kind] = set()
    if not isinstance(instruction, int):
        raise Exception(f"- error: expected instruction word to be a {decoded_instruction.get_num_bits_per_instruction()} bit integer")

    if isinstance(kind, decoded_instruction.instruction_kind):
        m_kinds.add(kind)
    elif isinstance(kind, (set, tuple, list)):
        m_kinds = m_kinds.union([ele for ele in set(kind) if isinstance(ele, decoded_instruction.instruction_kind)])

    if not m_kinds:
        m_kinds = m_kinds.union(get_instruction_kinds(kinds = None, modules = module, instruction_sets = instruction_set))

    if not m_kinds:
        m_kinds = m_kinds.union(get_valid_instruction_kinds(instruction, is_swizzled = is_swizzled))

    if not m_kinds:
        raise Exception(f"- error: could not determine instruction kind for instruction decode for given instruction {hex(instruction)}, is_swizzled: {is_swizzled}")

    m_modules = get_modules(kinds = m_kinds, modules = module)
    m_instruction_sets = get_instruction_sets(kinds = m_kinds, modules = m_modules, instruction_sets = instruction_set)

    dec_ins: list[decoded_instruction.decoded_instruction] = list()
    for k in m_kinds:
        dec_ins.append(m_modules[k].decode_instruction(instruction, m_instruction_sets[k], is_swizzled))

    if 0 == len(dec_ins):
        raise Exception(f"- error: could not decode given instruction {hex(instruction)}")
    elif 1 == len(dec_ins):
        return dec_ins[0]
    else:
        return dec_ins

def decode_instructions(
    data_stream: bytes,
    kinds: None | list[decoded_instruction.instruction_kind] | set[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind] = None,
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    are_swizzled: bool = True) -> list[decoded_instruction.decoded_instruction]:

    kinds            = get_instruction_kinds(kinds, modules, sets)
    modules          = get_modules(kinds, modules)
    instruction_sets = get_instruction_sets(kinds, modules, sets)

    instructions: list[decoded_instruction.decoded_instruction] = []

    num_bytes_per_instruction: int = decoded_instruction.get_num_bytes_per_instruction()

    for i in range(0, int(int(len(data_stream)/num_bytes_per_instruction) * num_bytes_per_instruction), num_bytes_per_instruction):
        instruction: int = int.from_bytes(data_stream[i:(i+num_bytes_per_instruction)], byteorder='little')
        decoded_instr = None
        for kind in kinds:
            if modules[kind].is_valid_instruction(instruction):
                decoded_instr = modules[kind].decode_instruction(instruction, instruction_sets[kind], are_swizzled)
                break

        if decoded_instr is None:
            decoded_instr = decoded_instruction.decoded_instruction()
            decoded_instr.set_word(instruction)

        instructions.append(decoded_instr)

    return instructions

def get_statistics_from_list(
    instructions: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]:

    m_kinds = get_instruction_kinds_from_decoded_instructions(instructions)
    m_modules = get_modules(m_kinds, modules)

    kinds_num_instrs: dict[decoded_instruction.instruction_kind, dict[str, int]] = dict()

    for kind in m_kinds:
        kinds_num_instrs.update({kind : dict()})

    for kind, instrs_dict in kinds_num_instrs.items():
        instrs_dict.update({"decoded"     : 0})
        instrs_dict.update({"no_opcode"   : 0})
        instrs_dict.update({"no_mnemonic" : 0})

    kind_instrs_list: dict[decoded_instruction.instruction_kind, list[str]] = dict()
    for kind in m_kinds:
        kind_instrs_list.update({kind : list()})

    for instruction in instructions:
        module = m_modules[instruction.kind]
        if not hasattr(instruction, 'opcode'):
            kinds_num_instrs[instruction.kind]["no_opcode"] += 1
            kind_instrs_list[instruction.kind].append("0b{:08b}".format(module.get_opcode(instruction.word)) + " not in ISA")

        elif not hasattr(instruction, 'mnemonic'):
            kinds_num_instrs[instruction.kind]["no_mnemonic"] += 1
            kind_instrs_list[instruction.kind].append("0b{:08b}".format(instruction.opcode) + " undecoded")
        else:
            kinds_num_instrs[instruction.kind]["decoded"] += 1
            kind_instrs_list[instruction.kind].append(instruction.mnemonic)

    kinds_instrs: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
    lsums: dict[decoded_instruction.instruction_kind, int] = dict()
    for kind, instrs in kind_instrs_list.items():
        counters = collections.Counter(instrs)
        kinds_instrs[kind] = dict()
        for key, value in counters.items():
            kinds_instrs[kind][key] = [int(0), float(0.0), float(0.0)]
            kinds_instrs[kind][key][0] = value

        lsums[kind] = sum(counters.values())

    gsum = sum(lsums.values())

    for kind in kinds_instrs.keys():
        for instr in kinds_instrs[kind].keys():
            kinds_instrs[kind][instr][1] = kinds_instrs[kind][instr][0] / lsums[kind]
            kinds_instrs[kind][instr][2] = kinds_instrs[kind][instr][0] / gsum

    return dict(sorted(kinds_instrs.items())), dict(sorted(kinds_num_instrs.items()))

def flatten_dict_and_get_statistics(function_names_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]:
    kinds = get_instruction_kinds_from_decoded_instructions(function_names_instructions, flatten_dict = True)
    modules = get_modules(kinds, modules)

    instructions: list[decoded_instruction.decoded_instruction] = list()
    for func_instrs in function_names_instructions.values():
        instructions.extend(func_instrs)

    return get_statistics_from_list(instructions, modules)

@typing.overload
def get_statistics_from_dict(
    function_names_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: typing.Literal[True] = True) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]: ...

@typing.overload
def get_statistics_from_dict(
    function_names_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: typing.Literal[False] = False) -> tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]: ...

@typing.overload
def get_statistics_from_dict(
    function_names_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: bool = False) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] | tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]: ...

def get_statistics_from_dict(
    function_names_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: bool = False) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] | tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]:

    kinds = get_instruction_kinds_from_decoded_instructions(function_names_instructions, flatten_dict = True)
    modules = get_modules(kinds, modules)

    if flatten_dict:
        return flatten_dict_and_get_statistics(function_names_instructions, modules)
    else:
        kinds_instrs: dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]] = dict()
        kinds_num_instrs: dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]] = dict()

        for function_name, instructions in function_names_instructions.items():
            func_kinds_instrs, func_kinds_num_instrs = get_statistics_from_list(instructions, modules)

            kinds_instrs.update({function_name : func_kinds_instrs})
            kinds_num_instrs.update({function_name : func_kinds_num_instrs})

        return kinds_instrs, kinds_num_instrs

@typing.overload
def get_statistics(
    instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: typing.Literal[False] = False) -> tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]: ...

@typing.overload
def get_statistics(
    instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: typing.Literal[True] = True) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]: ...

@typing.overload
def get_statistics(
    instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: bool = ...) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] | tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]: ...

@typing.overload
def get_statistics(
    instructions: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: bool = False) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]]: ...

def get_statistics(
    instructions: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...] | dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    flatten_dict: bool = False) -> tuple[dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], dict[decoded_instruction.instruction_kind, dict[str, int]]] | tuple[dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]]]:
    if isinstance(instructions, (list, tuple)):
        return get_statistics_from_list(instructions, modules)

    elif isinstance(instructions, dict):
        if flatten_dict:
            return flatten_dict_and_get_statistics(instructions, modules)
        else:
            return get_statistics_from_dict(instructions, modules, flatten_dict)

    else:
        raise Exception(f"- error: no method defined to get statistics from instructions of type {type(instructions)}")

def print_instruction(
    instruction: decoded_instruction.decoded_instruction | dict[typing.Any, decoded_instruction.decoded_instruction],
    module: None | types.ModuleType = None,
    instruction_set: None | dict[str, typing.Any] = None,
    end: str = '\n',
    print_offset: int = 2) -> None:
    if isinstance(instruction, dict):
        for value in instruction.values():
            print_instruction (value, module, instruction_set, end, print_offset)
    elif isinstance(instruction, decoded_instruction.decoded_instruction):
        if module is None:
            module = get_default_module(instruction.kind)

        module.print_instruction(instruction, get_instruction_set(instruction.kind), end, print_offset)
    else:
        raise Exception(f"- error: no method defined to print instruction of type {type(instruction)}")

def print_instructions(
    instructions: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction] | dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    end: str = '\n',
    print_offset: int = 2) -> None:
    def print_instructions_from_list (
        instructions: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction],
        modules: dict[decoded_instruction.instruction_kind, types.ModuleType],
        sets: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]],
        end: str,
        print_offset: int) -> None:
        for instruction in instructions:
            modules[instruction.kind].print_instruction(instruction, sets[instruction.kind], end, print_offset)

    def print_instructions_from_dict (
        functions_instructions: dict[str, list[decoded_instruction.decoded_instruction]],
        modules: dict[decoded_instruction.instruction_kind, types.ModuleType],
        sets: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]],
        end: str,
        print_offset: int) -> None:
        for function_name in sorted(functions_instructions.keys()):
            print(f"{print_offset * ' '}instructions from function: {function_name}")
            print_instructions_from_list(functions_instructions[function_name], modules, sets, end, print_offset + 2)

    m_kinds = get_instruction_kinds_from_decoded_instructions(instructions, flatten_dict=True)
    m_modules = get_modules(m_kinds, modules)
    m_sets = get_instruction_sets(m_kinds, m_modules, sets)

    if isinstance(instructions, (list, tuple)):
        return print_instructions_from_list(instructions, m_modules, m_sets, end, print_offset)

    elif isinstance(instructions, dict):
        return print_instructions_from_dict(instructions, m_modules, m_sets, end, print_offset)

    else:
        raise Exception(f"- print_instructions not defined for instructions type {type(instructions)}")

def instruction_histogram_to_str(
    histogram: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] | dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]],
    msg: str = "",
    preamble: str = "",
    print_offset: int = 2) -> str:
    def instruction_histogram_when_key_is_kind_to_str_list(
        histogram: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]],
        msg_list: list[str],
        preamble: str,
        print_offset: int):
        gmax_len: int = 0
        for arch, instrs in histogram.items():
            max_len = len(max(instrs.keys(), key=len)) if instrs.keys() else 0
            gmax_len = gmax_len if gmax_len > max_len else max_len

        if len(preamble):
            msg_list[0] += f"{' ' * print_offset}{preamble}\n"

        num_instructions: dict[decoded_instruction.instruction_kind, list[int]] = dict() # {kind : [num_instrs, num_unique_instrs, num_no_opcode_instrs, num_unique_no_opcode_instrs, num_no_mnemonic]}

        for kind, instrs in histogram.items():
            m_dict = sorted(instrs.items(), key = lambda x : x[1][0], reverse=True) #x is pair. x[0] is key, x[1] is value.

            msg_list[0] += f"{' ' * print_offset}- Instructions for kind {kind}\n"
            num_instrs                    = 0
            num_no_opcode_instrs          = 0
            num_unique_no_opcode_instrs   = 0
            num_no_mnemonic_instrs        = 0
            num_unique_no_mnemonic_instrs = 0
            for mnemonic, num_occ in m_dict:
                msg_list[0] += f"{' ' * print_offset}  - instruction: {mnemonic}, {' ' * (gmax_len - len(mnemonic))}number of occurrences: {num_occ[0]:4d}, % (within kind): {(num_occ[1] * 100.):>5.2f}, % (overall): {(num_occ[2] * 100.):>5.2f}\n"
                num_instrs += int(num_occ[0])
                if " undecoded" in mnemonic:
                    num_unique_no_mnemonic_instrs += 1
                    num_no_mnemonic_instrs += int(num_occ[0])
                if " not in ISA" in mnemonic:
                    num_unique_no_opcode_instrs += 1
                    num_no_opcode_instrs += int(num_occ[0])

            num_instructions.update({kind : [num_instrs, len(m_dict), num_no_opcode_instrs, num_unique_no_opcode_instrs, num_no_mnemonic_instrs, num_unique_no_mnemonic_instrs]})

            msg_list[0] += f"{' ' * print_offset}  - Instruction profile for kind {kind}\n"
            msg_list[0] += f"{' ' * print_offset}    - number of instructions:             {num_instrs:5d}, unique: {len(m_dict):4d}\n"
            msg_list[0] += f"{' ' * print_offset}    - number of no_opcode instructions:   {num_no_opcode_instrs:5d}, unique: {num_unique_no_opcode_instrs:4d}\n"
            msg_list[0] += f"{' ' * print_offset}    - number of no_mnemonic instructions: {num_no_mnemonic_instrs:5d}, unique: {num_unique_no_mnemonic_instrs:4d}\n"

        msg_list[0] += f"{' ' * print_offset}- Instruction profile:\n"
        msg_list[0] += f"{' ' * print_offset}  - number of instructions:               {sum([values[0] for values in num_instructions.values()]):5d}, unique: {sum([values[1] for values in num_instructions.values()]):4d}\n"
        msg_list[0] += f"{' ' * print_offset}  - number of no_opcode instructions:     {sum([values[2] for values in num_instructions.values()]):5d}, unique: {sum([values[3] for values in num_instructions.values()]):4d}\n"
        msg_list[0] += f"{' ' * print_offset}  - number of no_mnemonic instructions:   {sum([values[4] for values in num_instructions.values()]):5d}, unique: {sum([values[5] for values in num_instructions.values()]):4d}\n"

    def instruction_histogram_when_key_is_str_to_str_list(
        histogram: dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]],
        msg_list: list[str],
        preamble: str = "",
        print_offset: int = 2) -> None:
        for key, values in histogram.items():
            if isinstance(values, dict):
                instruction_histogram_when_key_is_kind_to_str_list(values, msg_list, preamble = str(key), print_offset = print_offset)
            else:
                raise Exception(f"- error: no method defined to print values of data type {type(values)}")

    msg_list: list[str] = [""]
    if all(isinstance(ele, decoded_instruction.instruction_kind) for ele in histogram.keys()) and all(isinstance(ele, dict) for ele in histogram.values()) and histogram:
        instruction_histogram_when_key_is_kind_to_str_list(typing.cast(dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]], histogram), msg_list, preamble, print_offset)
    elif all(isinstance(ele, str) for ele in histogram.keys()) and all(isinstance(ele, dict) for ele in histogram.values()) and histogram:
        instruction_histogram_when_key_is_str_to_str_list(typing.cast(dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]], histogram), msg_list, preamble, print_offset)
    else:
        raise Exception(f"- error: no method defined to print values of data type {type(histogram)}")

    msg += msg_list[0]
    return msg

def print_instruction_histogram(
    histogram: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] | dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]],
    preamble: str = "",
    print_offset: int = 2) -> None:
    msg: str = ""
    msg = instruction_histogram_to_str(histogram, msg, preamble, print_offset)
    print(msg.rstrip())

def instruction_kind_histogram_to_str(
    histogram: dict[decoded_instruction.instruction_kind, dict[str, int]] | dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]],
    msg: str = "",
    preamble: str = "",
    print_offset: int = 2) -> str:
    def instruction_kind_histogram_when_key_is_kind_to_str_list(
        histogram: dict[decoded_instruction.instruction_kind, dict[str, int]],
        msg_list: list[str],
        preamble: str,
        print_offset: int) -> None:
        if all(isinstance(ele, decoded_instruction.instruction_kind) for ele in histogram.keys()):
            if len(preamble):
                msg_list[0] += f"{' ' * print_offset}{preamble}\n"

            m_sum: int = 0
            for num_instrs in histogram.values():
                m_sum += sum(num_instrs.values())

            instr_types: list[str] = [f"{key}" for key in histogram.keys()]
            instr_types.append("")

            for values in histogram.values():
                instr_types += [key for key in values.keys()]

            max_len = len(max(instr_types, key = lambda x : len(x)))

            msg_list[0] += f"{' '*print_offset}- Number of instructions: {' '*(max_len - len('') + 4 + (1 if len('') == 0 else 0))}{m_sum:5d}\n"
            # +1 is because we need to add space.

            for arch, num_instrs in histogram.items():
                msg_list[0] += f"{' '*print_offset}  - Number of {arch} instructions: {' '*(max_len - len(f'{arch}') + 2)}{sum(num_instrs.values()):5d}\n"

                for instr_type, occ in num_instrs.items():
                    msg_list[0] += f"{' '*print_offset}    - Number of {instr_type} instructions: {' '*(max_len - len(instr_type))}{occ:5d}\n"

    def instruction_kind_histogram_when_key_is_str_to_str_list(
        histogram: dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]],
        msg_list: list[str],
        preamble: str,
        print_offset: int) -> None:
        if all(isinstance(key, str) for key in histogram.keys()) and all(isinstance(value, dict) for value in histogram.values()) and histogram:
            msg_list[0] += preamble
            for key, values in histogram.items():
                if all(isinstance(key1, decoded_instruction.instruction_kind) for key1 in values.keys()):
                    instruction_kind_histogram_when_key_is_kind_to_str_list(
                        typing.cast(dict[decoded_instruction.instruction_kind, dict[str, int]], values),
                        msg_list,
                        preamble = str(key),
                        print_offset = print_offset)
        else:
            raise Exception(f"- error: no method defined to print values of data type {type(histogram)}")

    msg_list: list[str] = [""]
    if all(isinstance(key, decoded_instruction.instruction_kind) for key in histogram.keys()) and all(isinstance(value, dict) for value in histogram.values()) and histogram:
        instruction_kind_histogram_when_key_is_kind_to_str_list(typing.cast(dict[decoded_instruction.instruction_kind, dict[str, int]], histogram), msg_list, preamble, print_offset)
    elif all(isinstance(key, str) for key in histogram.keys()) and all(isinstance(value, dict) for value in histogram.values()) and histogram:
        instruction_kind_histogram_when_key_is_str_to_str_list(typing.cast(dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]], histogram), msg_list, preamble, print_offset)
    else:
        raise Exception(f"- error: no method defined to obtain instruction histogram from data of type {type(histogram)}")

    msg += msg_list[0]
    return msg

def print_instruction_kind_histogram(
    histogram: dict[decoded_instruction.instruction_kind, dict[str, int]] | dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]],
    preamble: str = "",
    print_offset: int = 2) -> None:
    msg: str = ''
    msg = instruction_kind_histogram_to_str(histogram, msg, preamble, print_offset)
    print(msg.rstrip())

@typing.overload
def get_coverage(
    instructions: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: bool = False) -> dict[decoded_instruction.instruction_kind, float]: ...

@typing.overload
def get_coverage(
    instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: typing.Literal[True] = True) ->  dict[decoded_instruction.instruction_kind, float]: ...

@typing.overload
def get_coverage(
    instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: typing.Literal[False] = False) -> dict[str, dict[decoded_instruction.instruction_kind, float]]: ...

@typing.overload
def get_coverage(
    instructions: dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: bool = False) -> dict[decoded_instruction.instruction_kind, float] | dict[str, dict[decoded_instruction.instruction_kind, float]]: ...

def get_coverage(
    instructions: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction] | dict[str, list[decoded_instruction.decoded_instruction]],
    modules: None | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    flatten_dict: bool = False) -> dict[decoded_instruction.instruction_kind, float] | dict[str, dict[decoded_instruction.instruction_kind, float]]:
    def get_coverage_from_instruction_list (
        instructions: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction],
        modules: dict[decoded_instruction.instruction_kind, types.ModuleType],
        sets: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]) -> dict[decoded_instruction.instruction_kind, float]:
        instruction_histogram, _ = get_statistics(instructions, modules, flatten_dict=False)
        # False because we are not sending dict.
        coverage: dict[decoded_instruction.instruction_kind, float] = dict()
        if all(isinstance(key, decoded_instruction.instruction_kind) for key in instruction_histogram.keys()) and instruction_histogram:
            for kind, instrs in instruction_histogram.items():
                coverage.update({kind : len(instrs) / len(sets[kind])})
            # todo: remove not in ISA instructions.
        else:
            raise Exception(f"- error: key type mismatch in instruction_histogram. Not all keys are type {type(decoded_instruction.instruction_kind)}")

        return coverage

    def flatten_instruction_dict_and_get_coverage(
        instructions: dict[str, list[decoded_instruction.decoded_instruction]],
        modules: dict[decoded_instruction.instruction_kind, types.ModuleType],
        sets: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]) -> dict[decoded_instruction.instruction_kind, float]:
        all_instrs: list[decoded_instruction.decoded_instruction] = []
        for key, value in instructions.items():
            all_instrs += value

        return get_coverage_from_instruction_list(all_instrs, modules, sets)

    def get_coverage_from_instruction_dict (
        instructions: dict[str, list[decoded_instruction.decoded_instruction]],
        modules: dict[decoded_instruction.instruction_kind, types.ModuleType],
        sets: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]) -> dict[str, dict[decoded_instruction.instruction_kind, float]]:
        coverage: dict[str, dict[decoded_instruction.instruction_kind, float]] = dict()
        for func, instrs in instructions.items():
            coverage.update({func : get_coverage_from_instruction_list(instrs, modules, sets)})

        return coverage

    kinds = get_instruction_kinds_from_decoded_instructions(instructions, flatten_dict=True)
    modules = get_modules(kinds, modules)
    instruction_sets = get_instruction_sets(kinds, modules, sets)

    if isinstance(instructions, (list, tuple)):
        return get_coverage_from_instruction_list(instructions, modules, instruction_sets)

    elif isinstance(instructions, dict):
        if flatten_dict:
            return flatten_instruction_dict_and_get_coverage(instructions, modules, instruction_sets)
        else:
            return get_coverage_from_instruction_dict(instructions, modules, instruction_sets)
    else:
        raise Exception(f"- error: no method defined to get coverage from instructions of type {type(instructions)}")

@typing.overload
def instruction_to_str(instruction: decoded_instruction.decoded_instruction,
    module: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10,
    flatten_dict: bool = False) -> str: ...

@typing.overload
def instruction_to_str(instruction: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction],
    module: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10,
    flatten_dict: bool = False) -> list[str]: ...

@typing.overload
def instruction_to_str(instruction: dict[typing.Any, list[decoded_instruction.decoded_instruction]],
    module: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10,
    flatten_dict: typing.Literal[True] = True) -> list[str]: ...

@typing.overload
def instruction_to_str(instruction: dict[typing.Any, list[decoded_instruction.decoded_instruction]],
    module: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10,
    flatten_dict: typing.Literal[False] = False) -> dict[typing.Any, list[str]]: ...

@typing.overload
def instruction_to_str(instruction: dict[typing.Any, list[decoded_instruction.decoded_instruction]],
    module: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10,
    flatten_dict: bool = False) -> list[str] | dict[typing.Any, list[str]]: ...

def instruction_to_str(instruction: decoded_instruction.decoded_instruction | list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction] | dict[typing.Any, list[decoded_instruction.decoded_instruction]],
    module: None | types.ModuleType | dict[decoded_instruction.instruction_kind, types.ModuleType] = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10,
    flatten_dict = False) -> str | list[str] | dict[typing.Any, list[str]]:
    def instruction_without_kind_to_str(instruction):
        if decoded_instruction.get_num_bits_per_instruction() % 4:
            raise Exception(f"- error: decoded_instruction.get_num_bits_per_instruction() % 4 = {decoded_instruction.get_num_bits_per_instruction() % 4}")

        msg: str = ""
        if hasattr(instruction, 'program_counter'):
            msg += help_utils.from_int_to_hex_str(instruction.program_counter, int(decoded_instruction.get_num_bits_per_instruction()/4))
        msg += help_utils.from_int_to_hex_str(instruction.word, int(decoded_instruction.get_num_bits_per_instruction()/4))

        return msg.rstrip()

    kinds = get_instruction_kinds_from_decoded_instructions(instruction, flatten_dict = True)
    modules = get_modules(kinds, module)
    instruction_sets = typing.cast(dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]], get_instruction_sets(kinds, modules, instruction_set))

    if isinstance(instruction, decoded_instruction.decoded_instruction):
        if not hasattr(instruction, 'kind'):
            return instruction_without_kind_to_str(instruction)
        else:
            return modules[instruction.kind].instruction_to_str(instruction, instruction_set = instruction_sets, base = base)
    elif isinstance(instruction, (list, tuple)):
        if not all(isinstance(ele, decoded_instruction.decoded_instruction) for ele in instruction):
            raise Exception("- error: expected all elements of instruction list to be of type decoded_instruction.decoded_instruction, some are not")

        return [instruction_to_str(ele, modules, instruction_sets, base, flatten_dict) for ele in instruction]
    elif isinstance(instruction, dict):
        if not all(isinstance(ele, decoded_instruction.decoded_instruction) for instruction_list in instruction.values() for ele in instruction_list):
            raise Exception(f"- error: expected all elements of instruction dict values to be of type decoded_instruction.decoded_instruction, some are not. instruction.values(): {instruction.values()}")
        if flatten_dict:
            msgs0: list[str] = list()
            for instruction_list in instruction.values():
                for ele in instruction_list:
                    msgs0.append(instruction_to_str(ele, modules, instruction_sets, base, flatten_dict))

            return msgs0
        else:
            msgs: dict[typing.Any, list[str]] = dict()
            for key, instruction_list in instruction.items():
                msgs[key] = [instruction_to_str(ele, modules, instruction_sets, base, flatten_dict) for ele in instruction_list]
            return msgs

if "__main__" == __name__:
    pass
