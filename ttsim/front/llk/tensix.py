#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import collections.abc
import copy
import typing

import collections
import yaml

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.help.utils as help_utils
from ttsim.front.llk.decoded_instruction import swizzle_instruction as swizzle_instruction
from ttsim.front.llk.decoded_instruction import unswizzle_instruction as unswizzle_instruction


def to_int(t_str: str) -> int:
    if t_str.startswith(('0x', '-0x')):
        return int(t_str, base = 16)
    elif t_str.startswith(('0o', '-0o')):
        return int(t_str, base = 8)
    elif t_str.startswith(('0b', '-0b')):
        return int(t_str, base = 2)
    else:
        return int(t_str) # base 10

def num_bits_opcode() -> int:
    return 8

def add_size(instructions: dict[str, typing.Any]) -> dict[str, typing.Any]:
    # for each instruction
    #   calculate the number of arguments
    #   if 0, then nothing to do.
    #   sort arguments with start_bit
    #   for each argument:
    #     calculate the size based on start bit of next argument.
    #     # size bit from bit_length of max(values) from fcov is not included in this version.
    #     calculate min/max from values from fcov.
    #     add min, max, size to argument.

    for mnemonic, info in instructions.items():

        if not isinstance(info, dict):
            msg = f"- error: mnemonic: {mnemonic}\n"
            msg += f"- expected value/info type to be dict, received {type(info)}\n"
            msg += "- info:\n"
            msg += f"{info}"
            raise Exception(msg)

        if "arguments" not in info.keys():
            continue
    
        if isinstance(info["arguments"], list):
            if 0 == len(info["arguments"]):
                raise Exception(f"- {mnemonic} has a argument list with 0 length")

            args:       list[dict[str, typing.Any]] = sorted(info["arguments"], key = lambda x : x["start_bit"])
            start_bits: list[int]                   = [arg['start_bit'] for arg in args]
            start_bits.append(decoded_instruction.get_num_bits_per_instruction() - num_bits_opcode()) # opcode length for TT instructions.

            for idx, arg in enumerate(args):
                size: int = start_bits[idx + 1] - start_bits[idx]
                if size <= 0:
                    raise Exception(f"- error: size of argument {arg['name']} from instruction {mnemonic} is {size}")

                if "size" in arg.keys():
                    if arg["size"] > size:
                        msg  = f"- error: size of operand {arg['name']} from instruction {mnemonic} is greater than maximum possible.\n"
                        msg += f"  maximum possible size: {size}\n"
                        msg += f"  assigned size:         {arg['size']}"
                        raise Exception(msg)
                else:
                    arg.update({"size" : size})

            info["arguments"] = copy.deepcopy(args)

            for idx in range(len(args)):
                if ((info["arguments"][idx]["start_bit"] + info["arguments"][idx]["size"] - 1) >= start_bits[idx + 1]):
                    msg  = f"- error: size mismatch for one of the operands of instruction {mnemonic}" + "\n"
                    msg += f"  - operand:                             {info['arguments'][idx]['name']}" + "\n"
                    msg += f"    - start_bit:                         {info['arguments'][idx]['start_bit']}" + "\n"
                    msg += f"    - size:                              {info['arguments'][idx]['size']}" + "\n"
                    msg += f"    - start bit for next operand/opcode: {start_bits[idx + 1]}"
                    msg += "\n"
                    msg += f"- YAML dump of the instruction {mnemonic}:\n"
                    msg += yaml.dump(info)

                    raise Exception(msg)

    return instructions

def get_default_instruction_set_from_kind(kind: decoded_instruction.instruction_kind) -> dict[str, typing.Any]:
    return add_size(decoded_instruction.get_default_instruction_set(kind))

def get_instruction_set_from_file_name(file_name: str) -> dict[str, typing.Any]:
    return add_size(decoded_instruction.get_instruction_set_from_file_name(file_name))

def get_instruction_set(
    kind: None | decoded_instruction.instruction_kind = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[str, typing.Any]:
    if kind is None:
        if isinstance(instruction_set, decoded_instruction.instruction_kind):
            return get_default_instruction_set_from_kind(instruction_set)
        elif isinstance(instruction_set, str):
            return get_instruction_set_from_file_name(instruction_set)
        elif decoded_instruction.is_instruction_set_dict_instance(instruction_set):
            return instruction_set
        else:
            raise Exception(f"- error: could not get instruction set from given kind and instruction_set arguments. kind = {kind}, instruction_set_type = {type(instruction_set)}")
    elif isinstance(kind, decoded_instruction.instruction_kind):
        assert kind.is_tensix(), f"- error: given instruction kind {kind} is not a valid tensix instruction kind"
        if instruction_set is None:
            return get_default_instruction_set_from_kind(kind)
        elif isinstance(instruction_set, decoded_instruction.instruction_kind):
            assert kind == instruction_set, f"- error: expected kind and instruction set kind to be identical. given arguments: kind {kind}, instruction_set: {instruction_set}"
            return get_instruction_set(kind = kind)
        elif isinstance(instruction_set, str):
            return get_instruction_set_from_file_name(instruction_set)
        elif decoded_instruction.is_instruction_set_dict_instance(instruction_set):
            return instruction_set
        elif decoded_instruction.is_kinds_instruction_sets_dict_instance(instruction_set):
            assert kind in instruction_set.keys(), f"- error: given kind {kind} not included in instruction set keys: {instruction_set.keys()}"
            return get_instruction_set(kind = kind, instruction_set = instruction_set[kind])
        else:
            raise Exception(f"- error: could not get instruction set from given kind and instruction_set arguments. kind = {kind}, instruction_set_type = {type(instruction_set)}")
    else:
        raise Exception(f"- error: no method defined to get instruction set of from instruction kind of type {type(kind)}")

def get_instruction_sets(
    kinds: None | decoded_instruction.instruction_kind | list[decoded_instruction.instruction_kind] | tuple[decoded_instruction.instruction_kind, ...] | set[decoded_instruction.instruction_kind] = None,
    instruction_sets: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[decoded_instruction.instruction_kind, dict[str, typing.Any]]:
    all_kinds: set[decoded_instruction.instruction_kind] = set()
    if isinstance(kinds, decoded_instruction.instruction_kind) and kinds.is_tensix():
        all_kinds.add(kinds)
    elif isinstance(kinds, (list, tuple, set)):
        all_kinds = all_kinds.union([k for k in kinds if isinstance(k, decoded_instruction.instruction_kind) and k.is_tensix()])

    if isinstance(instruction_sets, decoded_instruction.instruction_kind) and instruction_sets.is_tensix():
        all_kinds.add(instruction_sets)
    elif decoded_instruction.is_kinds_instruction_sets_dict_instance(instruction_sets):
        all_kinds = all_kinds.union([k for k in instruction_sets.keys() if isinstance(k, decoded_instruction.instruction_kind) and k.is_tensix()])

    kinds_instruction_sets: dict[decoded_instruction.instruction_kind, dict[str, typing.Any]] = dict()

    if isinstance(instruction_sets, str) or decoded_instruction.is_instruction_set_dict_instance(instruction_sets):
        assert 1 == len(all_kinds), "- error: expected len(all_kinds) to be 1 as instruction_sets is of type either str or instruction_set_dict"
        kind = sorted(all_kinds)[0]
        kinds_instruction_sets[kind] = get_instruction_set(kind = kind, instruction_set = instruction_sets)
    elif decoded_instruction.is_kinds_instruction_sets_dict_instance(instruction_sets):
        for kind in instruction_sets.keys():
            kinds_instruction_sets[kind] = get_instruction_set(kind, instruction_sets[kind])

    for kind in all_kinds:
        if kind not in kinds_instruction_sets.keys():
            kinds_instruction_sets[kind] = get_instruction_set(kind)

    return kinds_instruction_sets

def get_execution_engines_and_instructions(instruction_set: decoded_instruction.instruction_kind | str | dict[str, typing.Any]) -> dict[str, list[str]]:

    if decoded_instruction.is_instruction_set_dict_instance(instruction_set):
        execution_engines: dict[str, list[str]] = dict()

        # engines = set()
        # for value in instruction_set.values():
        #     engines.add(value['ex_resource'])
        # engines = sorted(list(engines))

        # mypy will not allow reassignment, so doing all in 1 line.
        engines = sorted(set([value['ex_resource'] for value in instruction_set.values()]))

        for engine in engines:
            execution_engines.update({engine : list()})

        for mnemonic, value in instruction_set.items():
            execution_engines[value['ex_resource']].append(mnemonic)

        for engine in execution_engines.keys():
            execution_engines[engine] = sorted(execution_engines[engine])

        return execution_engines

    elif isinstance(instruction_set, decoded_instruction.instruction_kind):
        if not instruction_set.is_tensix():
            raise Exception(f"- error: execution engine not defined for instructions of kind {instruction_set}")

        instruction_set = get_default_instruction_set_from_kind(instruction_set)
        return get_execution_engines_and_instructions(instruction_set)

    elif isinstance(instruction_set, str):
        instruction_set = get_instruction_set_from_file_name(instruction_set)
        return get_execution_engines_and_instructions(instruction_set)

    else:
        msg = f"- error: no method defined to get execution engines of instruction_set of type {type(instruction_set)}"
        raise Exception(msg)

def get_opcode(instruction: int, is_swizzled = True) -> int:
    if is_swizzled:
        return decoded_instruction.left_circular_shift(instruction, 6, decoded_instruction.get_num_bits_per_instruction()) & 0b1111_1111
    else:
        return get_opcode(decoded_instruction.swizzle_instruction(instruction), is_swizzled=True)
    # todo: 0b1111_1111 should come from get_num_bits_per_opcode

@typing.overload
def is_valid_instruction(instruction: int, is_swizzled: bool = True) -> bool: ...

@typing.overload
def is_valid_instruction(instruction: decoded_instruction.decoded_instruction, is_swizzled: bool = True) -> bool: ...

@typing.overload
def is_valid_instruction(instruction: collections.abc.Sequence[int | decoded_instruction.decoded_instruction], is_swizzled: bool | collections.abc.Sequence[bool] = True) -> list[bool]: ...

# TODO: Add type overloading
def is_valid_instruction(instruction: int | decoded_instruction.decoded_instruction | collections.abc.Sequence[int | decoded_instruction.decoded_instruction], is_swizzled: bool | collections.abc.Sequence[bool] = True) -> bool | list[bool]:
    if isinstance(instruction, int):
        if is_swizzled:
            return (instruction & 0b11) != 0b11
        else:
            return is_valid_instruction(decoded_instruction.swizzle_instruction(instruction), True)

    elif isinstance(instruction, decoded_instruction.decoded_instruction):
        return instruction.kind.is_tensix()

    elif isinstance(instruction, (list, tuple)):
        if isinstance(is_swizzled, (list, tuple)):
            if len(instruction) != len(is_swizzled):
                raise Exception("- error: swizzled flag is not defined for each instruction.")

            is_valid: list[bool] = []
            for idx, ele in enumerate(instruction):
                if isinstance(ele, (int, decoded_instruction.decoded_instruction)) and isinstance(is_swizzled[idx], bool):
                    is_valid.append(bool(is_valid_instruction(ele, is_swizzled[idx])))
                else:
                    raise Exception(f"- error: type mismatch. Expected instruction to be of type either int or decoded_instruction.decoded_instruction, received: {type(ele)}. Expected is_swizzled flag to be bool, received {type(is_swizzled[idx])}")

            return is_valid
        else:

            if not isinstance(is_swizzled, bool):
                raise Exception(f"- error: expected is_swizzled flag to be bool, received {type(is_swizzled)}")

            is_valid = []
            for ele in instruction:
                if isinstance(ele, (int, decoded_instruction.decoded_instruction)):
                    is_valid.append(bool(is_valid_instruction(ele, is_swizzled)))
                else:
                    raise Exception(f"- error: expected instruction to be of type either int or decoded_instruction.decoded_instruction, received {type(ele)}")

            return is_valid

    else:
        raise Exception(f"- error: no method defined to determine the instruction kind for instruction of type {type(instruction)}")

def get_operands(arguments: dict[str, int]) -> decoded_instruction.operands:
    operands: decoded_instruction.operands = decoded_instruction.operands()
    operands.set_all(arguments)
    operands.set_attributes(dict[str, int | list[str]](copy.deepcopy(arguments)))

    return operands

def decode_instruction(instruction: int,
    kind: None | decoded_instruction.instruction_kind = None,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    is_swizzled: bool = True) -> decoded_instruction.decoded_instruction:
    def set_stall_res_attributes(de_instr: decoded_instruction.decoded_instruction, instruction_set: dict[str, typing.Any]) -> None:
        if not hasattr(de_instr, 'operands'):
            return

        if not hasattr(de_instr.operands, 'attributes'):
            return

        stall_res_str: str = "stall_res"
        bins: dict[int, str] = dict()
        if stall_res_str in de_instr.operands.attributes.keys():
            if 'arguments' in instruction_set[de_instr.mnemonic].keys():
                arguments: list[dict[str, typing.Any]] = instruction_set[de_instr.mnemonic]['arguments']
                for arg in arguments:
                    if stall_res_str == arg['name']:
                        fcov_point_bins: dict[str, typing.Any] = arg['fcov_point_bins']
                        if 'bins' in fcov_point_bins.keys():
                            for dict_ele in fcov_point_bins['bins']:
                                key: int = to_int(dict_ele['value'])
                                value: str = dict_ele['name'].removeprefix("stall_") # Python 3.9+.

                                if key in bins.keys():
                                    raise Exception(f"- key {hex(key)} exists in bins. bins = {bins}")

                                bins[key] = value
                            break

        if bins:
            if isinstance(de_instr.operands.attributes[stall_res_str], int):
                stall_res_value: int       = typing.cast(int, de_instr.operands.attributes[stall_res_str])
                stall_res_ids: list[int]   = [2**i for i in range(stall_res_value.bit_length()) if (stall_res_value >> i) & 1]
                stall_res_names: list[str] = [bins[ele] for ele in stall_res_ids] # we do not use bit positions as keys because if the stall_res value is 0, bit_length will return 0 and bit_length - 1 will be negative number.
                de_instr.operands.attributes[stall_res_str] = stall_res_names

        return

    arg_kind = kind
    kind = None
    opcode: None | int = None
    name: None | str   = None
    operands: dict[str, int] = dict()

    if is_valid_instruction(instruction, is_swizzled):

        kind = arg_kind

        instruction_set_ = get_instruction_set(kind, instruction_set)

        if instruction_set_ is None:
            raise Exception("- error: instruction set not known. please provide either instruction set or kind")

        # 0. set kind to instruction_kind
        unswzld_instr: int = decoded_instruction.unswizzle_instruction(instruction) if is_swizzled else instruction
        instr_opcode: int = (unswzld_instr >> 24) & 0b1111_1111

        for mnemonic, info in instruction_set_.items():
            if instr_opcode == info["op_binary"]:
                opcode = instr_opcode
                name = mnemonic

                if isinstance(info, dict) and ("arguments" in info.keys()) and isinstance(info["arguments"], list):
                    for arg in info["arguments"]:
                        oprd_value: int = (unswzld_instr >> arg["start_bit"]) & ((1 << arg["size"]) - 1)
                        oprd_name: str = arg["name"]
                        # oprd_name     = arg["name"] + "[" + str(arg["start_bit"] + arg["size"] - 1) + ":" + str(arg["start_bit"]) + "]"

                        if oprd_name.startswith("imm"):
                            oprd_value = decoded_instruction.extend_sign(oprd_value, arg["size"] - 1)

                        # if ((arg["max"] < oprd_value) if "max" in arg.keys() else False): # exceeds max
                        #     oprd_name += f"[exceeds max ({arg["max"]})]"
                        # elif ((oprd_value < arg["min"]) if "min" in arg.keys() else False): # lower than min
                        #     oprd_name += f"[less than min ({arg["min"]})]"

                        operands.update({oprd_name : oprd_value})

    # if isinstance(name, list):
    #     print(f"- could not determine the instruction, instruction: 0b{instruction:032b}, opcode = {instr_opcode}, unswizzled instruction: 0b{unswzld_instr:032b}", )

    decoded_instr: decoded_instruction.decoded_instruction = decoded_instruction.decoded_instruction()

    decoded_instr.set_word(instruction if is_swizzled else decoded_instruction.swizzle_instruction(instruction))
    if isinstance(kind, decoded_instruction.instruction_kind):
        decoded_instr.set_kind(kind)

    if isinstance(opcode, int):
        decoded_instr.set_opcode(opcode)

    if isinstance(name, str):
        decoded_instr.set_mnemonic(name)

    decoded_instr.set_operands(get_operands(operands))
    if 'instruction_set_' in locals():
        set_stall_res_attributes(decoded_instr, instruction_set_)

    return decoded_instr

def to_field_type_str(value: int, field_type: str) -> str:
    if "BIN" == field_type:
        return f"0b{value:b}"
    elif "HEX" == field_type:
        return f"0x{value:x}"
    else:
        return f"{value}"

@typing.overload
def instruction_to_str(
    instruction: decoded_instruction.decoded_instruction,
    instruction_set: None = None,
    base: int = 10) -> str | list[str]: ...

@typing.overload
def instruction_to_str(
    instruction: decoded_instruction.decoded_instruction,
    instruction_set: decoded_instruction.instruction_kind | str | dict[str, dict[str, typing.Any]] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]],
    base: int = 10) -> str: ...

@typing.overload
def instruction_to_str(
    instruction: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...],
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, dict[str, typing.Any]] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10) -> list[str | list[str]]: ...

def instruction_to_str(
    instruction: decoded_instruction.decoded_instruction | list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...],
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, dict[str, typing.Any]] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10) -> str | list[str] | list[str | list[str]]:
    if isinstance(instruction, decoded_instruction.decoded_instruction):
        msg = ""
        if decoded_instruction.get_num_bits_per_instruction() % 4:
            raise Exception("- error: num bits per instruction word are not multiple of 4")

        if hasattr(instruction, 'program_counter'):
            msg += help_utils.from_int_to_hex_str(instruction.program_counter, int(decoded_instruction.get_num_bits_per_instruction()/4)) + ": "
        msg += help_utils.from_int_to_hex_str(instruction.word, int(decoded_instruction.get_num_bits_per_instruction()/4)) + " "

        if not hasattr(instruction, 'kind'):
            return msg.rstrip()

        if not instruction.kind.is_tensix():
            raise Exception("- error: tensix.instruction_to_str is defined only for instructions of tensix kind")

        instruction_set = get_instruction_set(instruction.kind, instruction_set)

        if not hasattr(instruction, 'mnemonic'):
            return msg

        msg += instruction.mnemonic.lower() + " "

        info = instruction_set[instruction.mnemonic]

        if isinstance(info["arguments"], list):
            args = sorted(info["arguments"], key = lambda x : x["start_bit"])

            for idx in reversed(range(len(args))):
                oprd = args[idx]
                oprd_value = instruction.operands.all[oprd["name"]]

                oprd_name = oprd["name"] + "[" + str(oprd["start_bit"] + oprd["size"] - 1) + ":" + str(oprd["start_bit"]) + "]"

                msg += f"{oprd_name} = {to_field_type_str(oprd_value, oprd['field_type'])}"

                if idx:
                    msg += ", "

        return msg.rstrip()
    elif isinstance(instruction, (list, tuple)):
        kinds = sorted(set([ele.kind for ele in instruction if hasattr(ele, 'kind')]))
        kinds_instruction_sets = get_instruction_sets(kinds, instruction_sets = instruction_set)
        return [instruction_to_str(ele, instruction_set = kinds_instruction_sets[ele.kind] if hasattr(ele, 'kind') else None, base = base) for ele in instruction]

def print_instruction(instruction, instruction_set = None, end = '\n', print_offset = 2):
    msg: str = f"{print_offset * ' '}"
    instr_str = instruction_to_str(instruction, instruction_set = instruction_set)
    if isinstance(instr_str, str):
        msg += instr_str
    elif isinstance(instr_str, list):
        msg += '['
        for ele in instr_str:
            msg += ele + ", "
        msg += ']'
    print(msg, end = end)