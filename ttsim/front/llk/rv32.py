#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import typing

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.help.utils as help_utils


def instruction_kind() -> decoded_instruction.instruction_kind:
    return decoded_instruction.instruction_kind.rv32

def get_default_instruction_set() -> dict[str, typing.Any]:
    return decoded_instruction.get_default_instruction_set(instruction_kind())

def get_instruction_set_from_file_name(file_name: str) -> dict[str, typing.Any]:
    return decoded_instruction.get_instruction_set_from_file_name(file_name)

def is_valid_instruction(instruction: int, is_swizzled: bool = True) -> bool:
    if isinstance(instruction, int) and isinstance(is_swizzled, bool):
        if is_swizzled:
            return instruction & 0b11 == 0b11
        else:
            return is_valid_instruction(decoded_instruction.swizzle_instruction(instruction), is_swizzled = True)
    else:
        raise Exception(f"- error: no method defined to calculate validity of given instruction. instruction_type: {type(instruction)}, is_swizzled_type: {type(is_swizzled)}")

# todo: get_opcode function

def get_operands(arguments: dict[str, int]) -> decoded_instruction.operands:
    def get_values_keys_as_list(arg_dict: dict[str, int], starts_with: str) -> tuple[list[int], list[str]]:
        keys: list[str] = sorted([key for key in arg_dict.keys() if key.startswith(starts_with)])
        values: list[int] = [arg_dict[key] for key in keys]

        return values, keys

    def get_integer_sources(arguments: dict[str, int]) -> tuple[list[int], list[str]]:
        return get_values_keys_as_list(arguments, "rs")

    def get_float_sources(arguments: dict[str, int]) -> tuple[list[int], list[str]]:
        return get_values_keys_as_list(arguments, "frs")

    def get_integer_destinations(arguments: dict[str, int]) -> tuple[list[int], list[str]]:
        return get_values_keys_as_list(arguments, "rd")

    def get_float_destinations(arguments: dict[str, int]) -> tuple[list[int], list[str]]:
        return get_values_keys_as_list(arguments, "frd")

    def get_immediates(arguments: dict[str, int]) -> tuple[list[int], list[str]]:
        return get_values_keys_as_list(arguments, "imm")

    def get_attributes(arguments: dict[str, int], keys_to_ignore: list[str]) -> dict[str, int]:
        attr: dict[str, int] = dict()
        for key, value in arguments.items():
            if key not in keys_to_ignore:
                attr.update({key : value})

        return attr

    operands: decoded_instruction.operands = decoded_instruction.operands()

    operands.set_all(arguments)

    attr_keys_to_ignore: list[str] = list()
    sources: list[int]
    keys_to_ignore: list[str]

    sources, keys_to_ignore = get_integer_sources(arguments)
    attr_keys_to_ignore.extend(keys_to_ignore)
    operands.set_integer_sources(sources)

    sources, keys_to_ignore = get_float_sources(arguments)
    attr_keys_to_ignore.extend(keys_to_ignore)
    operands.set_float_sources(sources)

    destinations: list[int]

    destinations, keys_to_ignore = get_integer_destinations(arguments)
    attr_keys_to_ignore.extend(keys_to_ignore)
    operands.set_integer_destinations(destinations)

    destinations, keys_to_ignore = get_float_destinations(arguments)
    attr_keys_to_ignore.extend(keys_to_ignore)
    operands.set_float_destinations(destinations)

    immediates: list[int]

    immediates, keys_to_ignore = get_immediates(arguments)
    attr_keys_to_ignore.extend(keys_to_ignore)
    operands.set_immediates(immediates)

    operands.set_attributes(dict[str, int | list[str]](get_attributes(arguments, attr_keys_to_ignore)))

    return operands

def get_instruction_set(instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None) -> dict[str, typing.Any]:
    if instruction_set is None:
        return get_default_instruction_set()

    elif isinstance(instruction_set, decoded_instruction.instruction_kind):
        if instruction_set != instruction_kind():
            raise Exception(f"- error: expected instruction kind to be {instruction_kind()}, received {instruction_set}")

        return get_instruction_set()

    elif isinstance(instruction_set, str): # we assume we have yaml file name
        return get_instruction_set_from_file_name(instruction_set)

    elif decoded_instruction.is_instruction_set_dict_instance(instruction_set):
        return instruction_set

    elif decoded_instruction.is_kinds_instruction_sets_dict_instance(instruction_set):
        if instruction_kind() in instruction_set.keys():
            # if isinstance(instruction_set[instruction_kind()], decoded_instruction.instruction_kind) and instruction_set[instruction_kind()] != instruction_kind():
            #     raise Exception("- error: expect instruction kind key and value to be identical for instruction_set of type kinds_instruction_sets_dict")
            return get_instruction_set(instruction_set[instruction_kind()])
        else:
            raise Exception(f"- error: could not find instruction set associated with instruction kind {instruction_kind()} in given instruction set argument.")
    else:
        raise Exception(f"- error: no method defined to read instruction set of data type {type(instruction_set)}")

@typing.overload
def decode_instruction(
    instruction: int,
    instruction_set: typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]],
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

@typing.overload
def decode_instruction(
    instruction: int,
    instruction_set: None,
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

@typing.overload
def decode_instruction(
    instruction: int,
    instruction_set: decoded_instruction.instruction_kind,
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

@typing.overload
def decode_instruction(
    instruction: int,
    instruction_set: str,
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

@typing.overload
def decode_instruction(
    instruction: int,
    instruction_set: dict[str, typing.Any],
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

@typing.overload
def decode_instruction(
    instruction: int,
    *, # instruction_set argument omitted
    is_swizzled: bool = ...) -> decoded_instruction.decoded_instruction: ...

def decode_instruction(
    instruction: int,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | typing.Mapping[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    is_swizzled: bool = True) -> decoded_instruction.decoded_instruction:
    m_instruction_set: dict[str, typing.Any] = get_instruction_set(instruction_set)

    kind: decoded_instruction.instruction_kind
    opcode: int
    mnemonic: str
    arguments: dict[str, int] = dict()

    instruction = instruction if is_swizzled else decoded_instruction.swizzle_instruction(instruction)

    if is_valid_instruction(instruction):
        # 0. set kind to riscv.
        kind = decoded_instruction.instruction_kind.rv32

        # 1. check if opcode matches. if yes, set opcode.
        instr_opcode: int = instruction & 0b0111_1111
        for key, value in m_instruction_set.items():
            if instr_opcode == value["opcode"]:
                opcode = instr_opcode

                # 2. check if encodings match. if yes, set name.
                encodings_match: bool = True
                if value["encodings"] is not None:

                    for encoding in value["encodings"]:
                        enc_opcode: int = encoding["opcode"]
                        mask = (instruction >> encoding["start_bit"]) & ((1 << encoding["size"]) - 1)
                        if mask != enc_opcode:
                            encodings_match = False
                            break

                # 3. operands.
                if encodings_match:
                    mnemonic = key

                    if value["arguments"]:
                        for argument in value["arguments"]:
                            arg_val: int
                            arg2 : dict[str, int | str] = dict() # imm[11:5]
                            imm11: int
                            if argument["name"] in ("csr", "frd", "frs1", "frs2", "frs3", "pred", "rd", "rm", "rs1", "rs2", "rs3", "shamt", "succ", "uimm", "uimm[31:12]"):
                                arg_val = (instruction >> argument["start_bit"]) & ((1 << argument["size"]) - 1)
                                if argument["name"] in ("shamt", "uimm", "uimm[31:12]"):
                                    arguments.update({"imm" : arg_val})
                                else:
                                    arguments.update({argument["name"] : arg_val})

                            elif argument["name"] in ("imm[11:0]"):
                                arg_val = (instruction >> argument["start_bit"]) & ((1 << argument["size"]) - 1)
                                arguments.update({"imm" : decoded_instruction.extend_sign(arg_val, argument["size"] - 1)})

                            elif argument["name"] == "imm[4:0]":
                                # arg2 : dict[str, int | str] = dict() # imm[11:5]
                                found_arg2 = False
                                for arg in value["arguments"]:
                                    if arg["name"] == "imm[11:5]":
                                        arg2 = arg
                                        found_arg2 = True
                                        break

                                if not found_arg2:
                                    raise Exception("- error could not find imm[11:5] for given imm[4:0]")

                                arg1_val: int = (instruction >> argument["start_bit"]) & ((1 << argument["size"]) - 1)
                                arg2_val: int = (instruction >> int(arg2["start_bit"]))     & ((1 << int(arg2["size"]))     - 1)
                                arg_val       = (arg2_val << argument["size"]) | arg1_val
                                arguments.update({"imm" : decoded_instruction.extend_sign(arg_val, (argument["size"] + arg2["size"] - 1))})
                            elif argument["name"] == "imm[11:5]":
                                pass

                            elif argument["name"] == "imm[4:1|11]":
                                # arg2: dict[str, int | str] = dict() # imm[12|10:5]
                                found_arg2 = False
                                for arg in value["arguments"]:
                                    if arg["name"] == "imm[12|10:5]":
                                        arg2 = arg
                                        found_arg2 = True
                                        break

                                if not found_arg2:
                                    raise Exception("- error could not find imm[12|10:5] for given imm[4:1|11]")

                                imm10_5: int = (instruction >> int(arg2["start_bit"]))           & 0b11_1111
                                imm12: int   = (instruction >> (int(arg2["start_bit"]) + 6))     & 0b1
                                imm11        = (instruction >> (argument["start_bit"]))          & 0b1
                                imm4_1: int  = (instruction >> (int(argument["start_bit"]) + 1)) & 0b1111
                                arg_val      = (imm12 << 12) | (imm11 << 11) | (imm10_5 << 5) | (imm4_1 << 1)

                                arguments.update({"imm" : decoded_instruction.extend_sign(arg_val, 12)})
                            elif argument["name"] == "imm[12|10:5]":
                                pass

                            elif argument["name"] == "imm[31:12]":
                                arg_val = (instruction >> argument["start_bit"]) & ((1 << argument["size"]) - 1)
                                arguments.update({"imm" : ((decoded_instruction.extend_sign(arg_val, argument["size"] - 1)))})
                                # no bit shifts, see: https://stackoverflow.com/questions/71882914/risc-v-u-format-instruction-immediate-confusion

                            elif argument["name"] == "imm[20|10:1|11|19:12]":
                                imm20: int    = (instruction >> 31) & 0b1
                                imm10_1: int  = (instruction >> 21) & 0b0011_1111_1111
                                imm11         = (instruction >> 20) & 0b1
                                imm19_12: int = (instruction >> 12) & 0b1111_1111
                                arg_val       = (imm20 << 20) | (imm19_12 << 12) | (imm11 << 11) | (imm10_1 << 1)

                                arguments.update({"imm" : decoded_instruction.extend_sign(arg_val, 20)})
                            elif argument["name"] == "fm":
                                pass

                            else:
                                raise Exception(f"- error: can not determine the argument type, argument: {argument}, key = {key}, value = {value}")

                    break
                    # TODO: With current approach, FENCE.TSO and PAUSE get categorised as FENCE (generic). We perhaps need to fix a sequence in which instructions are checked. Calculate opcode + encodings bit, instructions with least number of free bits are checked first.

    decoded_instr = decoded_instruction.decoded_instruction()
    decoded_instr.set_word(instruction)
    if 'kind' in locals():
        decoded_instr.set_kind(kind)

    if 'opcode' in locals():
        decoded_instr.set_opcode(opcode)

    if 'mnemonic' in locals():
        decoded_instr.set_mnemonic(mnemonic)

    if 'arguments' in locals():
        decoded_instr.set_operands(get_operands(arguments))

    return decoded_instr

def to_assembly(instruction: decoded_instruction.decoded_instruction, t_base: int = 10) -> str:

    if not isinstance(instruction, decoded_instruction.decoded_instruction):
        raise Exception(f"- error: no method defined to get assembly string from instruction of type {type(instruction)}")

    if int(t_base) != int(10):
        raise Exception("- error: only base 10 is supported at the moment")

    if decoded_instruction.get_num_bits_per_instruction() % 4:
        raise Exception("- error: num bits per instruction word are not multiple of 4")

    instr_hex_str: str = ""
    if hasattr(instruction, 'program_counter'):
        instr_hex_str += help_utils.from_int_to_hex_str(instruction.program_counter, int(decoded_instruction.get_num_bits_per_instruction()/4)) + ": "
    instr_hex_str += help_utils.from_int_to_hex_str(instruction.word, int(decoded_instruction.get_num_bits_per_instruction()/4))

    if not hasattr(instruction, 'kind'):
        return instr_hex_str

    if instruction.kind != instruction_kind():
        raise Exception(f"- error: instruction kind mismatch. expected instruction kind to be {instruction_kind()}, given decoded instruction has type: {instruction.kind}")

    if not hasattr(instruction, 'mnemonic'):
        return instr_hex_str

    mnemonic = instruction.mnemonic

    operands: dict[str, int] = dict()
    if hasattr(instruction, 'operands') and hasattr(instruction.operands, 'all'):
        operands = instruction.operands.all

    if mnemonic in ['EBREAK', 'ECALL', 'FENCE.I', 'FENCE.TSO', 'PAUSE']:
        #
        return f'{instr_hex_str} {mnemonic.lower()}'
    elif mnemonic in ['FCVT.D.Q', 'FCVT.D.S', 'FCVT.Q.D', 'FCVT.Q.S', 'FCVT.S.D', 'FCVT.S.Q', 'FSQRT.D', 'FSQRT.Q', 'FSQRT.S']:
        #frd, frs1
        return f'{instr_hex_str} {mnemonic.lower()} f{operands["frd"]}, f{operands["frs1"]}'
    elif mnemonic in ['FADD.D', 'FADD.Q', 'FADD.S', 'FDIV.D', 'FDIV.Q', 'FDIV.S', 'FMAX.D', 'FMAX.Q', 'FMAX.S', 'FMIN.D', 'FMIN.Q', 'FMIN.S', 'FMUL.D', 'FMUL.Q', 'FMUL.S', 'FSGNJ.D', 'FSGNJ.Q', 'FSGNJ.S', 'FSGNJN.D', 'FSGNJN.Q', 'FSGNJN.S', 'FSGNJX.D', 'FSGNJX.Q', 'FSGNJX.S', 'FSUB.D', 'FSUB.Q', 'FSUB.S']:
        #frd, frs1, frs2
        return f'{instr_hex_str} {mnemonic.lower()} f{operands["frd"]}, f{operands["frs1"]}, f{operands["frs2"]}'
    elif mnemonic in ['FMADD.D', 'FMADD.Q', 'FMADD.S', 'FMSUB.D', 'FMSUB.Q', 'FMSUB.S', 'FNMADD.D', 'FNMADD.Q', 'FNMADD.S', 'FNMSUB.D', 'FNMSUB.Q', 'FNMSUB.S']:
        #frd, frs1, frs2, frs3
        return f'{instr_hex_str} {mnemonic.lower()} f{operands["frd"]}, f{operands["frs1"]}, f{operands["frs2"]}, f{operands["frs3"]}'
    elif mnemonic in ['FLD', 'FLQ', 'FLW']:
        #frd, offset(rs1)
        return f'{instr_hex_str} {mnemonic.lower()} f{operands["frd"]}, {operands["imm"]}(x{operands["rs1"]})'
    elif mnemonic in ['FCVT.D.L', 'FCVT.D.LU', 'FCVT.D.W', 'FCVT.D.WU', 'FCVT.Q.L', 'FCVT.Q.LU', 'FCVT.Q.W', 'FCVT.Q.WU', 'FCVT.S.L', 'FCVT.S.LU', 'FCVT.S.W', 'FCVT.S.WU', 'FMV.D.X', 'FMV.W.X']:
        #frd, rs1
        return f'{instr_hex_str} {mnemonic.lower()} f{operands["frd"]}, x{operands["rs1"]}'
    elif mnemonic in ['FSD', 'FSQ', 'FSW']:
        #frs2, offset(rs1)
        return f'{instr_hex_str} {mnemonic.lower()} f{operands["frs2"]}, {operands["imm"]}(x{operands["rs1"]})'
    elif mnemonic in ['FENCE']:
        #iorw, iorw

        pred_str: str = ""
        pred_str += "i" if ((operands["pred"] >> 3) & 0b1) else ""
        pred_str += "o" if ((operands["pred"] >> 2) & 0b1) else ""
        pred_str += "r" if ((operands["pred"] >> 1) & 0b1) else ""
        pred_str += "w" if (operands["pred"]        & 0b1) else ""

        if not len(pred_str):
            raise Exception("- error: could not determine pred from the FENCE instruction, pred value in binary: {:04b}".format(operands["pred"]))

        succ_str: str = ""
        succ_str += "i" if ((operands["succ"] >> 3) & 0b1) else ""
        succ_str += "o" if ((operands["succ"] >> 2) & 0b1) else ""
        succ_str += "r" if ((operands["succ"] >> 1) & 0b1) else ""
        succ_str += "w" if (operands["succ"]        & 0b1) else ""

        if not len(succ_str):
            raise Exception("- error: could not determine succ from the FENCE instruction, succ value in binary: {:04b}".format(operands["succ"]))

        return f'{instr_hex_str} {mnemonic.lower()} {pred_str}, {succ_str}'
    elif mnemonic in ['LR.D', 'LR.W']:
        #rd, (rs1)
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, (x{operands["rs1"]})'
    elif mnemonic in ['CSRRC', 'CSRRS', 'CSRRW']:
        #rd, csr, rs1
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, {hex(operands["csr"])}, x{operands["rs1"]}'
    elif mnemonic in ['CSRRCI', 'CSRRSI', 'CSRRWI']:
        #rd, csr, uimm
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, {hex(operands["csr"])}, {operands["imm"]}'
    elif mnemonic in ['FCLASS.D', 'FCLASS.Q', 'FCLASS.S', 'FCVT.L.D', 'FCVT.L.Q', 'FCVT.L.S', 'FCVT.LU.D', 'FCVT.LU.Q', 'FCVT.LU.S', 'FCVT.W.D', 'FCVT.W.Q', 'FCVT.W.S', 'FCVT.WU.D', 'FCVT.WU.Q', 'FCVT.WU.S', 'FMV.X.D', 'FMV.X.W']:
        #rd, frs1
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, f{operands["frs1"]}'
    elif mnemonic in ['FEQ.D', 'FEQ.Q', 'FEQ.S', 'FLE.D', 'FLE.Q', 'FLE.S', 'FLT.D', 'FLT.Q', 'FLT.S']:
        #rd, frs1, frs2
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, f{operands["frs1"]}, f{operands["frs2"]}'
    elif mnemonic in ['AUIPC', 'LUI']:
        #rd, imm
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, {operands["imm"]}'
    elif mnemonic in ['JAL']:
        #rd, offset
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, {operands["imm"]}'
    elif mnemonic in ['JALR', 'LB', 'LBU', 'LD', 'LH', 'LHU', 'LW', 'LWU']:
        #rd, offset(rs1)
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, {operands["imm"]}(x{operands["rs1"]})'
    elif mnemonic in ['ADDI', 'ADDIW', 'ANDI', 'ORI', 'SLLI', 'SLLI', 'SLLIW', 'SLTI', 'SLTIU', 'SRAI', 'SRAI', 'SRAIW', 'SRLI', 'SRLI', 'SRLIW', 'XORI']:
        #rd, rs1, imm
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, x{operands["rs1"]}, {operands["imm"]}'
    elif mnemonic in ['ADD', 'ADDW', 'AND', 'DIV', 'DIVU', 'DIVUW', 'DIVW', 'MUL', 'MULH', 'MULHSU', 'MULHU', 'MULW', 'OR', 'REM', 'REMU', 'REMUW', 'REMW', 'SLL', 'SLLW', 'SLT', 'SLTU', 'SRA', 'SRAW', 'SRL', 'SRLW', 'SUB', 'SUBW', 'XOR']:
        #rd, rs1, rs2
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, x{operands["rs1"]}, x{operands["rs2"]}'
    elif mnemonic in ['AMOADD.D', 'AMOADD.W', 'AMOAND.D', 'AMOAND.W', 'AMOMAX.D', 'AMOMAX.W', 'AMOMAXU.D', 'AMOMAXU.W', 'AMOMIN.D', 'AMOMIN.W', 'AMOMINU.D', 'AMOMINU.W', 'AMOOR.D', 'AMOOR.W', 'AMOSWAP.D', 'AMOSWAP.W', 'AMOXOR.D', 'AMOXOR.W', 'SC.D', 'SC.W']:
        #rd, rs2, (rs1)
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rd"]}, x{operands["rs2"]}, (x{operands["rs1"]})'
    elif mnemonic in ['BEQ', 'BGE', 'BGEU', 'BLT', 'BLTU', 'BNE']:
        #rs1, rs2, offset
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rs1"]}, x{operands["rs2"]}, {operands["imm"]}'
    elif mnemonic in ['SB', 'SD', 'SH', 'SW']:
        #rs2, offset(rs1)
        return f'{instr_hex_str} {mnemonic.lower()} x{operands["rs2"]}, {operands["imm"]}(x{operands["rs1"]})'
    else:
        raise Exception(f"- error: no method defined to write instruction {mnemonic.lower()} to assembly")

@typing.overload
def instruction_to_str(instruction: decoded_instruction.decoded_instruction,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, dict[str, typing.Any]] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10) -> str: ...

@typing.overload
def instruction_to_str(instruction: list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...],
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, dict[str, typing.Any]] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10) -> list[str]: ...

def instruction_to_str(instruction: decoded_instruction.decoded_instruction | list[decoded_instruction.decoded_instruction] | tuple[decoded_instruction.decoded_instruction, ...],
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, dict[str, typing.Any]] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    base: int = 10) -> str | list[str]:
    if isinstance(instruction, decoded_instruction.decoded_instruction):
        return to_assembly(instruction, t_base = base)
    elif isinstance(instruction, (list, tuple)):
        return [instruction_to_str(ele, instruction_set, base) for ele in instruction]
    else:
        raise Exception(f"- error: instruction_to_str not defined for instruction of type {type(instruction)}")

def print_instruction(instruction: decoded_instruction.decoded_instruction, instruction_set = None, end = '\n', print_offset = 2) -> None:
    msg: str = f"{print_offset * ' '}"
    msg += instruction_to_str(instruction, instruction_set = instruction_set)
    print(msg, end = end)
