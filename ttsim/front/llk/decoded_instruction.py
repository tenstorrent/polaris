#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import enum
import types
import typing

import yaml

import ttsim.front.llk.help.utils as help_utils


class instruction_kind (enum.Enum):
    rv32 = enum.auto()
    ttwh = enum.auto()
    ttbh = enum.auto()
    ttqs = enum.auto()

    def __str__(self: typing.Self) -> str:
        return self.name

    def __lt__(self: typing.Self, other: typing.Self) -> bool:
        if self.__class__ == other.__class__:
            return self.value < other.value
        else:
            raise NotImplementedError(f"- error: no method defined to compare {type(self)} and {type(other)}")

    def is_tensix(self: typing.Self) -> bool:
        match self:
            case instruction_kind.rv32 :
                return False
            case instruction_kind.ttwh :
                return True
            case instruction_kind.ttbh :
                return True
            case instruction_kind.ttqs :
                return True
            case _ :
                raise AssertionError(f"- error: tensix flag not defined for instruction kind {self}")

class registers:
    def __init__(self: typing.Self) -> None:
        pass

    def set_integers (self, integers: int | list[int] | tuple[int]) -> None:
        if m_list := self.to_list_of_non_negative_ints(integers):
            self.integers: list[int] = m_list

    def set_floats (self, floats) -> None:
        if m_list := self.to_list_of_non_negative_ints(floats):
            self.floats: list[int] = m_list

    def __str__(self: typing.Self) -> str:
        msg = ""
        if hasattr(self, 'integers'):
            msg = f"- integers: {self.integers}\n"
        if hasattr(self, 'floats'):
            msg = f"- floats: {self.floats}\n"

        return msg.rstrip()

    def __repr__(self: typing.Self) -> str:
        return self.__str__()

    def __eq__(self: typing.Self, other: object) -> bool:
        if not isinstance(other, registers):
            return NotImplemented

        attrs = help_utils.get_user_attrs(self)
        if attrs != help_utils.get_user_attrs(other):
            return False

        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    @staticmethod
    def to_list_of_ints(arg: typing.Any) -> list[int] | None:
        if isinstance(arg, int):
            return [arg]

        if not isinstance(arg, (list, tuple)):
            return None

        if arg and all(isinstance(ele, int) for ele in arg):
            return list(arg)

        return None
        # TODO: impose positivity condition, won't work for immediates.

    @staticmethod
    def to_list_of_non_negative_ints(arg: typing.Any) -> list[int] | None:
        if m_list := registers.to_list_of_ints(arg):
            return m_list if all(ele >= 0 for ele in m_list) else None

        return None

class operands:
    def __init__(self: typing.Self) -> None:
        pass

    def set_all(self: typing.Self, arg_all: dict[str, int], mode = "q") -> None:
        if arg_all and isinstance(arg_all, dict) and all(isinstance(key, str) for key in arg_all.keys()) and all(isinstance(value, int) for value in arg_all.values()):
            self.all:dict[str, int] = arg_all
        elif "v" == mode:
            print(f"- WARNING: attribute all not set, given argument: {arg_all}")

    def set_sources(self: typing.Self, sources: registers, mode = "q") -> None:
        if isinstance(sources, registers):
            if hasattr(sources, 'integers') or hasattr(sources, 'floats'):
                self.sources = sources
            elif "v" == mode:
                print("- WARNING: source not set")
        elif "v" == mode:
            print("- WARNING: source not set")

    def set_destinations(self: typing.Self, destinations: registers, mode = "q") -> None:
        if isinstance(destinations, registers):
            if hasattr(destinations, 'integers') or hasattr(destinations, 'floats'):
                self.destinations = destinations
            elif "v" == mode:
                print("- WARNING: destination not set")
        elif "v" == mode:
            print("- WARNING: destination not set")

    def set_integer_sources(self: typing.Self, integers: int | list[int] | tuple[int], mode = "q") -> None:
        # additional declaration because mypy doesn't like `if integers := registers.to_list_of_ints(integers)`
        if m_list := registers.to_list_of_non_negative_ints(integers):
            if not hasattr(self, 'sources'):
                self.sources = registers()

            self.sources.set_integers(m_list)
        elif "v" == mode:
            print("- WARNING: source integer registers not set")

    def set_float_sources(self: typing.Self, floats: int | list[int] | tuple[int], mode = "q") -> None:
        if m_list := registers.to_list_of_non_negative_ints(floats):
            if not hasattr(self, 'sources'):
                self.sources = registers()

            self.sources.set_floats(m_list)
        elif "v" == mode:
            print("- WARNING: source floating point registers not set")

    def set_integer_destinations(self: typing.Self, integers: int | list[int] | tuple[int], mode = "q") -> None:
        if m_list := registers.to_list_of_non_negative_ints(integers):
            if not hasattr(self, 'destinations'):
                self.destinations = registers()

            self.destinations.set_integers(m_list)
        elif "v" == mode:
            print("- WARNING: destination integer registers not set")

    def set_float_destinations(self: typing.Self, floats: int | list[int] | tuple[int], mode = "q") -> None:
        if m_list := registers.to_list_of_non_negative_ints(floats):
            if not hasattr(self, 'destinations'):
                self.destinations = registers()

            self.destinations.set_floats(m_list)
        elif "v" == mode:
            print("- WARNING: destination floating point registers not set")

    def set_immediates(self: typing.Self, immediates: int | list[int] | tuple[int], mode = "q") -> None:
        if imm := registers.to_list_of_ints(immediates):
            self.immediates = imm
        elif "v" == mode:
            print("- WARNING: immediates not set")

    def set_attributes(self: typing.Self, attributes: dict[str, int | list[str]], mode = "q") -> None:
        if attributes and isinstance(attributes, dict) and all(isinstance(key, str) for key in attributes.keys()):
            is_value_type_correct = True
            for value in attributes.values():
                is_value_type_correct == is_value_type_correct and (isinstance(value, int) or (isinstance(value, list) and all(isinstance(e, str) for e in value)))

            if is_value_type_correct:
                self.attributes = attributes
            elif "v" == mode:
                print(f"- WARNING: attributes not set, given argument: {attributes}")
        elif "v" == mode:
            print(f"- WARNING: attributes not set, given argument: {attributes}")


    def __str__(self: typing.Self) -> str:
        msg = ""
        if hasattr(self, 'all'):
            msg += f"  - all:                  {self.all}\n"

        if hasattr(self, 'sources'):
            if hasattr(self.sources, 'integers'):
                msg += f"  - integer sources:      {self.sources.integers}\n"

            if hasattr(self.sources, 'floats'):
                msg += f"  - float sources:        {self.sources.floats}\n"

        if hasattr(self, 'destinations'):
            if hasattr(self.destinations, 'integers'):
                msg += f"  - integer destinations: {self.destinations.integers}\n"

            if hasattr(self.destinations, 'floats'):
                msg += f"  - float destinations:   {self.destinations.floats}\n"

        if hasattr(self, 'immediates'):
            msg += f"  - immediates:           {self.immediates}\n"

        if hasattr(self, 'attributes'):
            msg += f"  - attributes:           {self.attributes}"

        if len(msg):
            msg = "- operands:\n" + msg

        return msg.rstrip()

    def __eq__(self: typing.Self, other: object) -> bool:
        if not isinstance(other, operands):
            return NotImplemented

        attrs = help_utils.get_user_attrs(self)
        if attrs != help_utils.get_user_attrs(other):
            return False

        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

    def __repr__(self: typing.Self) -> str:
        return self.__str__()

class decoded_instruction:
    def __init__(self: typing.Self) -> None:
        pass

    def set_word(self: typing.Self, word: int, mode = "q") -> None:
        if isinstance(word, int):
            self.word: int = word
        elif "v" == mode:
            print("- WARNING: word is not set")

    def set_program_counter(self: typing.Self, program_counter: int, mode = "q") -> None:
        if isinstance(program_counter, int):
            self.program_counter: int = program_counter
        elif "v" == mode:
            print("- WARNING: program counter is not set")

    def set_kind(self: typing.Self, kind: instruction_kind, mode = "q") -> None:
        if isinstance(kind, instruction_kind):
            self.kind: instruction_kind = kind
        elif "v" == mode:
            print("- WARNING: kind is not set")

    def set_opcode(self: typing.Self, opcode: int, mode = "q") -> None:
        if isinstance(opcode, int):
            self.opcode: int = opcode
        elif "v" == mode:
            print("- WARNING: opcode is not set")

    def set_mnemonic(self: typing.Self, mnemonic: str, mode = "q") -> None:
        if isinstance(mnemonic, str) and len(mnemonic):
            self.mnemonic: str = mnemonic
        elif "v" == mode:
            print("- WARNING: mnemonic not set")

    def set_operands(self: typing.Self, arg_operands: operands, mode = "q") -> None:
        if isinstance(arg_operands, operands) and (hasattr(arg_operands, 'all') or hasattr(arg_operands, 'sources') or hasattr(arg_operands, 'destinations') or hasattr(arg_operands, 'immediates') or hasattr(arg_operands, 'attributes')):
            self.operands: operands = arg_operands
        elif "v" == mode:
            print("- WARNING: operands not set")

    # def set_relative_address(self: typing.Self, relative_address: int) -> None:
    #     if isinstance(relative_address, int):
    #         self.relative_address: int = relative_address

    # def get_relative_address(self: typing.Self) -> int | None:
    #     return self.relative_address if hasattr(self, 'relative_address') else None

    def get_program_counter(self: typing.Self) -> int | None:
        return self.program_counter if hasattr(self, 'program_counter') else None

    def __str__(self: typing.Self) -> str:
        msg = ""
        if hasattr(self, 'program_counter'):
            msg += f"  - program counter: {hex(self.program_counter)}\n"

        if hasattr(self, 'word'):
            msg += f"  - {get_num_bits_per_instruction()} bit instruction:\n"
            msg += f"    - binary:        0b{self.word:0{get_num_bits_per_instruction()}b}\n"
            msg += f"    - hex:           0x{self.word:0{int(get_num_bits_per_instruction() / 4)}x}\n"

        if hasattr(self, 'kind'):
            msg += f"  - kind:            {self.kind}\n"

        if hasattr(self, 'opcode'):
            msg += "  - opcode: \n"
            if hasattr(self, 'kind'):
                msg += f"    - binary:        0b{self.opcode:0{get_max_num_bits_opcode(self.kind)}b}\n"
            else:
                msg += f"    - binary:        0b{self.opcode:b}\n"
            msg += f"    - hex:           0x{self.opcode:x}\n"

        if hasattr(self, 'mnemonic'):
            msg += f"  - mnemonic:        {self.mnemonic}\n"

        if hasattr(self, 'operands'):
            msg += "  - operands:\n"
            if hasattr(self.operands, 'all'):
                msg += f"    - all:                  {self.operands.all}\n"

            if hasattr(self.operands, 'sources'):
                if hasattr(self.operands.sources, 'integers'):
                    msg += f"    - integer sources:      {self.operands.sources.integers}\n"
                if hasattr(self.operands.sources, 'floats'):
                    msg += f"    - float sources:        {self.operands.sources.floats}\n"

            if hasattr(self.operands, 'destinations'):
                if hasattr(self.operands.destinations, 'integers'):
                    msg += f"    - integer destinations: {self.operands.destinations.integers}\n"
                if hasattr(self.operands.destinations, 'floats'):
                    msg += f"    - float destinations:   {self.operands.destinations.floats}\n"

            if hasattr(self.operands, 'immediates'):
                msg += f"    - immediates:           {self.operands.immediates}\n"

            if hasattr(self.operands, 'attributes'):
                msg += f"    - attributes:           {self.operands.attributes}\n"

        if len(msg):
            msg = "- decoded instruction: \n" + msg

        return msg.rstrip()

    def __repr__(self: typing.Self) -> str:
        return self.__str__()

    def __eq__(self: typing.Self, other: object) -> bool:
        if not isinstance(other, decoded_instruction):
            return NotImplemented

        attrs = help_utils.get_user_attrs(self)
        if attrs != help_utils.get_user_attrs(other):
            return False

        for attr in attrs:
            if getattr(self, attr) != getattr(other, attr):
                return False

        return True

def to_instruction_kind(arg: str) -> instruction_kind:
    for kind in instruction_kind:
        if f"{kind}" == arg:
            return kind

    raise Exception(f"- error: could not find instruction_kind associated with str: {arg}")

def extend_sign(value: int, bit_num_from_lsb: int) -> int:
    if value & (1 << bit_num_from_lsb):
        value |= ~((1 << (bit_num_from_lsb + 1)) - 1)  # Fill higher bits with 1

    return value

def get_instruction_kinds() -> set[instruction_kind]:
    kinds: set[instruction_kind] = set()
    for k in instruction_kind:
        kinds.add(k)

    return kinds

def get_max_num_bits_opcode(instr_type: instruction_kind) -> int:
    if instr_type == instruction_kind.rv32:
        return 7
    return 8

def get_num_bits_per_instruction() -> int:
    return 32

def get_num_bytes_per_instruction() -> int:
    num_bits_per_byte: int = 8

    if get_num_bits_per_instruction() % num_bits_per_byte:
        raise AssertionError(f"- error: can not get integer number of bytes from {get_num_bits_per_instruction()} bits")

    return int(get_num_bits_per_instruction() / num_bits_per_byte)

def get_default_instruction_set_file_name(instr_kind: instruction_kind) -> str:
    if instr_kind not in instruction_kind:
        msg:str = "- error: please provide correct instruction kind.\n"
        msg += f"- Given instruction kind: {instr_kind}\n"
        msg += "- Available kinds: \n"
        for k in instruction_kind:
            msg += f"  - {k}"

        raise Exception(msg)

    pwd                 = os.path.dirname(os.path.abspath(__file__)) # path of this file.
    file_name_incl_path = os.path.join(pwd, "..", "..", "config", "llk", "instruction_sets", f"{instr_kind}", "assembly.yaml")
    file_name_incl_path = os.path.normpath(file_name_incl_path)
    if not os.path.isfile(file_name_incl_path):
        raise AssertionError(f"- error: file {file_name_incl_path} does not exist.")

    return file_name_incl_path

def get_instruction_kinds_rv32_tensix_attributes_dict():
    rv32_ttwh_attr: list[str] = ["riscv_rv32i2p0_m2p0_xttwh1p0", "riscv_rv32i2p0_m2p0_zmmul1p0_xttwh1p0", "riscvrv32i2p0_m2p0_xttwh1p0"]
    rv32_ttbh_attr: list[str] = ["riscv_rv32i2p0_m2p0_xttbh1p0", "riscv#rv32i2p0_m2p0_xttbh1p0", "riscvrv32i2p0_m2p0_xttbh1p0"]
    rv32_ttqs_attr: list[str] = ["riscv_rv32i2p0_m2p0_a2p0_f2p0_v1p0_zfh0p1_zvamo1p0_zvlsseg1p0", "riscvDrv32i2p0_m2p0_a2p0_f2p0_v1p0_zfh0p1_zvamo1p0_zvlsseg1p0", "riscv@rv32i2p0_m2p0_a2p0_f2p0_v1p0_zfh0p1_zvamo1p0_zvlsseg1p0"]

    kinds_attrs: dict[tuple[instruction_kind, instruction_kind], list[str]] = dict()
    for kind in instruction_kind:
        match kind:
            case instruction_kind.ttwh:
                kinds_attrs[(instruction_kind.rv32, instruction_kind.ttwh)] = rv32_ttwh_attr
            case instruction_kind.ttbh:
                kinds_attrs[(instruction_kind.rv32, instruction_kind.ttbh)] = rv32_ttbh_attr
            case instruction_kind.ttqs:
                kinds_attrs[(instruction_kind.rv32, instruction_kind.ttqs)] = rv32_ttqs_attr
            case instruction_kind.rv32:
                pass
            case _:
                AssertionError(f"- error: no attributes defined for instruction kind {kind}")

    return kinds_attrs

def get_instruction_kinds_from_riscv_attribute(riscv_attribute: str) -> set[instruction_kind]:
    for instruction_kinds, attributes in get_instruction_kinds_rv32_tensix_attributes_dict().items():
        if riscv_attribute in attributes:
            return instruction_kinds

    msg = "- error: incorrect riscv attribute.\n"
    msg += f"- given riscv attribute: {riscv_attribute}\n"
    msg += "- accepted attributes:\n"
    for instruction_kinds, attributes in get_instruction_kinds_rv32_tensix_attributes_dict().items():
        msg += f"{instruction_kinds}: {attributes}\n"

    raise Exception(msg)

def get_instruction_set_from_file_name(file_name: str) -> dict[str, typing.Any]: #TODO: check the type of value.
    with open(file_name) as stream:
        instructions: dict[str, typing.Any] = yaml.safe_load(stream)

        for mnemonic, info in instructions.items():
            if not isinstance(info, dict):
                msg = f"- error: reading file: {file_name}, mnemonic: {mnemonic}\n"
                msg += f"- expected value/info type to be dict, received {type(info)}\n"
                msg += "- info:\n"
                msg += f"{info}"
                raise Exception(msg)

            if "arguments" not in info.keys():
                continue

            if isinstance(info["arguments"], list):
                arg_names = [arg["name"] for arg in info["arguments"]]
                if len(set(arg_names)) != len(arg_names):
                    raise AssertionError(f"- error: argument names are repeated for instruction {mnemonic}")

        return instructions

def get_default_instruction_set(kind: instruction_kind) -> dict[str, typing.Any]:
    file_name_incl_path = get_default_instruction_set_file_name(kind)
    return get_instruction_set_from_file_name(file_name_incl_path)

def is_instruction_set_dict_instance(instruction_set: typing.Any) -> typing.TypeGuard[dict[str, typing.Any]]:
    return isinstance(instruction_set, dict) and all(isinstance(key, str) for key in instruction_set.keys()) and all(isinstance(value, dict) for value in instruction_set.values())

def is_kinds_instruction_sets_dict_instance(instruction_set: typing.Any) -> typing.TypeGuard[dict[instruction_kind, dict[str, typing.Any]]]:
    return isinstance(instruction_set, dict) and all(isinstance(key, instruction_kind) for key in instruction_set.keys())

def is_instruction_kinds_modules_dict_instance(arg) -> typing.TypeGuard[dict[instruction_kind, types.ModuleType]]:
    return isinstance(arg, dict) and all(isinstance(key, instruction_kind) for key in arg.keys()) and all(isinstance(value, types.ModuleType) for value in arg.values())

def left_circular_shift(value: int, shift: int, num_bits: int) -> int:
    # return ((value << shift) | (value >> (num_bits - shift))) & ((1 << num_bits) - 1)
    return help_utils.left_circular_shift(value, shift, num_bits)

def right_circular_shift(value: int, shift: int, num_bits: int) -> int:
    # return ((value >> shift) | (value << (num_bits - shift))) & ((1 << num_bits) - 1)
    return help_utils.right_circular_shift(value, shift, num_bits)

def swizzle_instruction(instruction: int) -> int:
    return left_circular_shift(instruction, 2, get_num_bits_per_instruction())

def unswizzle_instruction(instruction: int) -> int:
    return right_circular_shift(instruction, 2, get_num_bits_per_instruction())