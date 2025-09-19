#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import typing

import pytest
import random

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.tensix as tensix


def test_to_int():
    max = int(1e9)
    bases = sorted([2, 8, 10, 16])
    nums = [random.randint(-max, max) for id in range(len(bases))]
    nums_str = ["" for id in range(len(nums))]
    for idx in range(len(bases)):
        if 2 == bases[idx]:
            nums_str[idx] = bin(nums[idx])
        elif 8 == bases[idx]:
            nums_str[idx] = oct(nums[idx])
        elif 10 == bases[idx]:
            nums_str[idx] = str(nums[idx])
        elif 16 == bases[idx]:
            nums_str[idx] = hex(nums[idx])
        else:
            raise Exception(f"- no method defined for base: {bases[idx]}")

    nums1 = [tensix.to_int(nums_str[idx]) for idx in range(len(nums_str))]

    assert nums1 == nums

def test_num_bits_opcode():
    assert 8 == tensix.num_bits_opcode()

def test_add_size():
    def test_add_size_correctness():
        for kind in decoded_instruction.instruction_kind:
            if not kind.is_tensix():
                continue
            instruction_set = decoded_instruction.get_default_instruction_set(kind)
            tensix.add_size(instruction_set)
            for mnemonic, info in instruction_set.items():
                if hasattr(info, "arguments"):
                    if info["arguments"]:
                        for arg in info["arguments"]:
                            assert hasattr(arg, "size")
                            assert arg["size"] <= (decoded_instruction.get_num_bits_per_instruction() - tensix.num_bits_opcode())
                            assert arg["size"] > 0
                            # TODO: more stricter checks on size.

    def test_size_exception():
        instruction_set = dict()
        instruction_set["ABC"] = dict({"arguments" : [{"name" : "a", "start_bit" : 0, "size" : 25}]})

        with pytest.raises(Exception) as exe_info:
            tensix.add_size(instruction_set)
        assert "error: size of operand" in str(exe_info.value)

    test_add_size_correctness()
    test_size_exception()

def test_get_default_instruction_set_from_kind():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue
        instruction_set = tensix.get_default_instruction_set_from_kind(kind)
        assert isinstance(instruction_set, dict)
        assert len(instruction_set)
        assert all(isinstance(key, str) for key in instruction_set.keys())
        assert all(isinstance(value, dict) for value in instruction_set.values())

def test_get_instruction_set_from_file_name():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue
        instruction_set = tensix.get_instruction_set_from_file_name(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "ttsim", "config", "llk", "instruction_sets", f"{kind}", "assembly.yaml")))
        assert isinstance(instruction_set, dict)
        assert len(instruction_set)
        assert all(isinstance(key, str) for key in instruction_set.keys())
        assert all(isinstance(value, dict) for value in instruction_set.values())

def test_get_instruction_set_0():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue
        instruction_set = tensix.get_instruction_set(kind)
        assert instruction_set == tensix.get_instruction_set(kind = None, instruction_set = kind)
        assert instruction_set == tensix.get_default_instruction_set_from_kind(kind)

def test_get_instruction_set_1():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue
        file_path = decoded_instruction.get_default_instruction_set_file_name(kind)
        instruction_set = tensix.get_instruction_set(kind = None, instruction_set = file_path)
        assert instruction_set == tensix.get_default_instruction_set_from_kind(kind)

def test_get_instruction_set_2():
    instruction_set: dict[str, typing.Any] = dict({"abc": dict()})
    assert instruction_set == tensix.get_instruction_set(kind = None, instruction_set = instruction_set)

def test_get_instruction_set_3_exceptions():
    with pytest.raises(Exception):
        tensix.get_instruction_set(kind = "ttwh") # type: ignore[arg-type]

    with pytest.raises(Exception) as exe_info:
        tensix.get_instruction_set(decoded_instruction.instruction_kind.rv32) # non-tensix instruction kind

    with pytest.raises(Exception):
        tensix.get_instruction_set(kind = None, instruction_set = typing.cast(typing.Any, []))

    with pytest.raises(Exception):
        ins_sets: dict[decoded_instruction.instruction_kind, typing.Any] = dict({
            decoded_instruction.instruction_kind.ttwh : None,
            decoded_instruction.instruction_kind.ttbh : dict()
        })
        tensix.get_instruction_set(kind = None, instruction_set = ins_sets)

    with pytest.raises(Exception):
        tensix.get_instruction_set(kind = decoded_instruction.instruction_kind.ttwh, instruction_set = decoded_instruction.instruction_kind.ttbh)

def test_get_instruction_set_4_exceptions():
    kind = decoded_instruction.instruction_kind.ttwh
    instruction_set: dict[decoded_instruction.instruction_kind, typing.Any] = dict({
        kind : dict()
    })
    with pytest.raises(Exception):
        tensix.get_instruction_set(kind = decoded_instruction.instruction_kind.ttbh, instruction_set = instruction_set)

def test_get_instruction_set_5():
    kind = decoded_instruction.instruction_kind.ttwh
    instruction_set: dict[decoded_instruction.instruction_kind, typing.Any]= dict({
        kind : dict()
    })
    assert instruction_set[kind] == tensix.get_instruction_set(kind = kind, instruction_set = instruction_set)

def test_get_instruction_set_6_exceptions():
    instruction_set: typing.Any = dict({
        decoded_instruction.instruction_kind.ttwh : dict(),
        "ttbh" : dict()
    })
    with pytest.raises(Exception):
        tensix.get_instruction_set(kind = decoded_instruction.instruction_kind.ttwh, instruction_set = instruction_set) # type: ignore[arg-type]

def test_get_instruction_sets_0():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue
        kinds_sets = tensix.get_instruction_sets(kind)
        assert sorted(kinds_sets.keys()) == sorted([kind])
        assert kinds_sets[kind] == tensix.get_instruction_set(kind = None, instruction_set = kind)
        assert kinds_sets[kind] == tensix.get_default_instruction_set_from_kind(kind)

def test_get_instruction_sets_1():
    kinds = [kind for kind in decoded_instruction.instruction_kind if kind.is_tensix()]
    kinds_sets = tensix.get_instruction_sets(kinds)
    assert sorted(kinds) == sorted(kinds_sets.keys())
    for kind in kinds:
        assert kinds_sets[kind] == tensix.get_instruction_set(kind)

def test_get_instruction_sets_2():
    kinds = [kind for kind in decoded_instruction.instruction_kind if kind.is_tensix()]
    kinds_sets = tensix.get_instruction_sets(kinds = None, instruction_sets = {kind : None for kind in kinds})
    assert sorted(kinds) == sorted(kinds_sets.keys())
    for kind in kinds:
        assert kinds_sets[kind] == tensix.get_instruction_set(kind)

def test_get_instruction_sets_3():
    kind = decoded_instruction.instruction_kind.ttwh
    ins_set: dict[str, typing.Any] = dict()
    kinds_sets = tensix.get_instruction_sets(kinds = kind, instruction_sets = ins_set)
    assert sorted(kinds_sets.keys()) == sorted([kind])
    assert kinds_sets[kind] == ins_set

    with pytest.raises(Exception):
        tensix.get_instruction_sets(kinds = None, instruction_sets = ins_set)

def test_get_instruction_sets_4():
    kind = decoded_instruction.instruction_kind.ttwh
    kinds_sets = tensix.get_instruction_sets(kinds = kind, instruction_sets = decoded_instruction.get_default_instruction_set_file_name(kind))
    assert sorted(kinds_sets.keys()) == sorted([kind])
    assert kinds_sets[kind] == tensix.get_instruction_set(kind)

    with pytest.raises(Exception):
        tensix.get_instruction_sets(kinds = None, instruction_sets = decoded_instruction.get_default_instruction_set_file_name(kind))

def test_get_instruction_sets_5():
    inp_kinds_sets: typing.Any = {
        decoded_instruction.instruction_kind.ttwh : dict(),
        decoded_instruction.instruction_kind.ttbh : decoded_instruction.get_default_instruction_set_file_name(decoded_instruction.instruction_kind.ttbh),
        decoded_instruction.instruction_kind.ttqs : decoded_instruction.instruction_kind.ttqs
    }
    kinds_sets = tensix.get_instruction_sets(kinds = None, instruction_sets = inp_kinds_sets)
    assert sorted(kinds_sets.keys()) == sorted(inp_kinds_sets.keys())
    assert kinds_sets[decoded_instruction.instruction_kind.ttwh] == inp_kinds_sets[decoded_instruction.instruction_kind.ttwh]
    assert kinds_sets[decoded_instruction.instruction_kind.ttbh] == tensix.get_instruction_set(decoded_instruction.instruction_kind.ttbh)
    assert kinds_sets[decoded_instruction.instruction_kind.ttqs] == tensix.get_instruction_set(decoded_instruction.instruction_kind.ttqs)

def test_get_instruction_sets_6():
    inp_kinds_sets: typing.Any = {
        decoded_instruction.instruction_kind.ttbh : decoded_instruction.get_default_instruction_set_file_name(decoded_instruction.instruction_kind.ttbh),
        decoded_instruction.instruction_kind.ttqs : decoded_instruction.instruction_kind.ttqs
    }
    kinds_sets = tensix.get_instruction_sets(kinds = decoded_instruction.instruction_kind.ttwh, instruction_sets = inp_kinds_sets)
    kinds = []
    kinds.append(decoded_instruction.instruction_kind.ttwh)
    for key in inp_kinds_sets.keys():
        kinds.append(key)
    assert sorted(kinds_sets.keys()) == sorted(kinds)
    assert kinds_sets[decoded_instruction.instruction_kind.ttwh] == tensix.get_instruction_set(decoded_instruction.instruction_kind.ttwh)
    assert kinds_sets[decoded_instruction.instruction_kind.ttbh] == tensix.get_instruction_set(decoded_instruction.instruction_kind.ttbh)
    assert kinds_sets[decoded_instruction.instruction_kind.ttqs] == tensix.get_instruction_set(decoded_instruction.instruction_kind.ttqs)

def test_get_instruction_sets_7():
    assert dict() == tensix.get_instruction_sets(kinds = None, instruction_sets = None)
    assert dict() == tensix.get_instruction_sets(kinds = typing.cast(typing.Any, dict()))
    assert dict() == tensix.get_instruction_sets(kinds = "", instruction_sets = None) # type: ignore[arg-type]
    assert dict() == tensix.get_instruction_sets(kinds = decoded_instruction.instruction_kind.rv32)

def test_get_instruction_sets_8():
    with pytest.raises(Exception):
        tensix.get_instruction_sets(kinds = None, instruction_sets = "")

    with pytest.raises(Exception):
        tensix.get_instruction_sets(kinds = None, instruction_sets = dict())

def test_get_execution_engines_and_instructions():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue
        instruction_set = tensix.get_default_instruction_set_from_kind(kind)
        engines = sorted(set([info["ex_resource"] for info in instruction_set.values()]))

        engines_instructions = tensix.get_execution_engines_and_instructions(instruction_set)
        engines_instructions1 = tensix.get_execution_engines_and_instructions(kind)
        engines_instructions2 = tensix.get_execution_engines_and_instructions(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "ttsim", "config", "llk", "instruction_sets", f"{kind}", "assembly.yaml")))
        assert engines_instructions == engines_instructions1
        assert engines_instructions == engines_instructions2
        assert isinstance(engines_instructions, dict)
        assert len(instruction_set) and len(engines_instructions)
        assert all(isinstance(engine, str) for engine in engines_instructions.keys())
        assert all(isinstance(instructions, list) for instructions in engines_instructions.values())
        assert engines == sorted(engines_instructions.keys())
        assert len(instruction_set) == sum([len(instructions) for instructions in engines_instructions.values()])
        for engine, instructions in engines_instructions.items():
            for mnemonic in instructions:
                assert mnemonic in instruction_set.keys()
                assert engine == instruction_set[mnemonic]["ex_resource"]

    with pytest.raises(Exception) as exe_info:
        tensix.get_execution_engines_and_instructions(decoded_instruction.instruction_kind.rv32) # type: ignore[arg-type]
    assert "execution engine not defined for instructions of kind" in str(exe_info.value)

    with pytest.raises(Exception) as exe_info:
        tensix.get_execution_engines_and_instructions(decoded_instruction.instruction_kind) # type: ignore[arg-type]
    assert "error: no method defined to get execution engines of instruction_set of type" in str(exe_info.value)

    incorrect_instruction_set: typing.Any = dict()
    incorrect_instruction_set['abc'] = dict()
    incorrect_instruction_set[1] = 0
    with pytest.raises(Exception) as exe_info:
        tensix.get_execution_engines_and_instructions(incorrect_instruction_set) # type: ignore[arg-type]
    assert "no method defined to get execution engines" in str(exe_info.value)

def test_get_opcode():
    unswizzled_word = 0x5160000b
    swizzled_word   = decoded_instruction.swizzle_instruction(unswizzled_word)
    expected_opcode = 0x51
    received0 = tensix.get_opcode(swizzled_word, is_swizzled = True)
    received1 = tensix.get_opcode(unswizzled_word, is_swizzled = False)
    assert received0 == expected_opcode
    assert received1 == expected_opcode

def test_is_valid_instruction():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue

        instruction_set = tensix.get_default_instruction_set_from_kind(kind)
        unswizzled_words: list[int] = list()
        swizzled_words: list[int] = list()
        unswizzled_instructions: list[decoded_instruction.decoded_instruction] = list()
        swizzled_instructions: list[decoded_instruction.decoded_instruction] = list()
        for info in instruction_set.values():
            unswizzled_word = decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32) # opcode is set correctly. rest of arguments are 0
            unswizzled_words.append(unswizzled_word)

            swizzled_word = decoded_instruction.swizzle_instruction(unswizzled_word)
            swizzled_words.append(swizzled_word)

            assert tensix.is_valid_instruction(unswizzled_word, is_swizzled = False)
            assert tensix.is_valid_instruction(swizzled_word, is_swizzled = True)

            di_unsw = tensix.decode_instruction(unswizzled_word, kind = kind, instruction_set = instruction_set, is_swizzled = False)
            assert di_unsw.kind.is_tensix()
            unswizzled_instructions.append(di_unsw)

            di_swzl = tensix.decode_instruction(swizzled_word, kind = kind, instruction_set = instruction_set, is_swizzled = True)
            assert di_swzl.kind.is_tensix()
            swizzled_instructions.append(di_swzl)

        assert len(unswizzled_words)        == sum(tensix.is_valid_instruction(unswizzled_words, is_swizzled = False))
        assert len(unswizzled_words)        == sum(tensix.is_valid_instruction(unswizzled_words, is_swizzled = [False for idx in range(len(unswizzled_instructions))]))
        assert len(swizzled_words)          == sum(tensix.is_valid_instruction(swizzled_words, is_swizzled = True))
        assert len(swizzled_words)          == sum(tensix.is_valid_instruction(swizzled_words, is_swizzled = [True for idx in range(len(unswizzled_instructions))]))
        assert len(unswizzled_instructions) == sum(tensix.is_valid_instruction(unswizzled_instructions, is_swizzled = False))
        assert len(unswizzled_instructions) == sum(tensix.is_valid_instruction(unswizzled_instructions, is_swizzled = True))
        assert len(swizzled_instructions)   == sum(tensix.is_valid_instruction(swizzled_instructions, is_swizzled = False))
        assert len(swizzled_instructions)   == sum(tensix.is_valid_instruction(swizzled_instructions, is_swizzled = True))

        with pytest.raises(Exception) as exe_info:
            tensix.is_valid_instruction(unswizzled_words, is_swizzled = [False, False])
        assert "error: swizzled flag is not defined for each instruction" in str(exe_info.value)

        with pytest.raises(Exception) as exe_info:
            tensix.is_valid_instruction(typing.cast(typing.Any, [0, 'a']), is_swizzled = [False, False])
        assert "error: type mismatch. Expected instruction to be of type either int or decoded_instruction.decoded_instruction" in str(exe_info.value)

def test_get_operands():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue

        instruction_set = tensix.get_default_instruction_set_from_kind(kind)
        # words = [decoded_instruction.swizzle_instruction(info["op_binary"]) for info in instruction_set.values()]

        for mnemonic, info in instruction_set.items():
            args = dict()
            if hasattr(info, 'arguments'):
                if info['arguments']:
                    for arg in info['arguments']:
                        args[arg['name']] = 0 # assigning value 0

            if args:
                operands = tensix.get_operands(args)
                assert hasattr(operands, 'all')
                assert hasattr(operands, 'attributes')
                assert not hasattr(operands, 'sources')
                assert not hasattr(operands, 'destinations')
                assert not hasattr(operands, 'immediates')
                assert all(0 == value for value in operands.all.values())
                assert all(0 == value for value in operands.attributes.values())
                assert len(args) == len(operands.all)
                assert len(operands.all) == len(operands.attributes)

@pytest.mark.slow
def test_decode_instruction():
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue

        instruction_set = tensix.get_default_instruction_set_from_kind(kind)
        mnemonics_words = {mnemonic : decoded_instruction.swizzle_instruction(decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32)) for mnemonic, info in instruction_set.items()}
        # right_circular_shift: takes opcode to MSBs,
        # we then swizzle instruction

        for mnemonic, word in mnemonics_words.items():
            di0 = tensix.decode_instruction(word, kind = kind, instruction_set = instruction_set, is_swizzled = True)
            di1 = tensix.decode_instruction(word, kind = kind, instruction_set = instruction_set)
            di2 = tensix.decode_instruction(word, kind = kind)
            di3 = tensix.decode_instruction(decoded_instruction.unswizzle_instruction(word), kind = kind, instruction_set = instruction_set, is_swizzled = False)
            di4 = tensix.decode_instruction(decoded_instruction.unswizzle_instruction(word), kind = kind, instruction_set = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "ttsim", "config", "llk", "instruction_sets", f"{kind}", "assembly.yaml")), is_swizzled = False)
            assert hasattr(di0, 'mnemonic')
            assert di0.mnemonic == mnemonic
            assert di0 == di1
            assert di0 == di2
            assert di0 == di3
            assert di0 == di4

            for attr in ['word', 'opcode', 'kind', 'mnemonic']:
                assert hasattr(di0, attr)

            # not list
            for attr in ['program_counter']:
                assert not hasattr(di0, attr)

            info = instruction_set[di0.mnemonic]
            if hasattr(info, 'arguments'):
                if info['arguments']:
                    assert hasattr(di0, 'operands')

                    for attr in ['all', 'attributes']:
                        assert hasattr(di0.operands, attr)
                        for key, value in getattr(di0, attr).items():
                            if isinstance(value, int):
                                assert 0 == value
                            elif isinstance(value, list):
                                assert not sum([ele for ele in value if 0 == ele])
                            else:
                                raise Exception(f"- error: no method defined to check values of type {type(value)}")

                    # not list
                    for attr in ['sources', 'destinations', 'immediates']:
                        assert not hasattr(di0.operands, attr)

        #TODO: asserts for non-zero argument values.

def test_to_field_type_str():
    value = 5
    assert tensix.to_field_type_str(value, 'BIN').startswith('0b')
    assert tensix.to_field_type_str(value, 'HEX').startswith('0x')
    assert tensix.to_field_type_str(value, '').startswith(f"{value}")

@pytest.mark.slow
def test_instruction_to_str():
    def test_one_instruction_per_call():
        for kind in decoded_instruction.instruction_kind:
            if not kind.is_tensix():
                continue

            instruction_set = tensix.get_default_instruction_set_from_kind(kind)
            mnemonics_words = {mnemonic : decoded_instruction.swizzle_instruction(decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32)) for mnemonic, info in instruction_set.items()}

            for mnemonic, word in mnemonics_words.items():
                di = tensix.decode_instruction(word, kind = kind, instruction_set = instruction_set, is_swizzled = True)
                assert hasattr(di, 'mnemonic')

                msg = tensix.instruction_to_str(di, instruction_set = instruction_set)
                assert msg.startswith(f"0x{word:08x} {mnemonic.lower()}")
                if instruction_set[mnemonic]["arguments"]:
                    assert all(arg["name"] in msg for arg in instruction_set[mnemonic]["arguments"])

    def test_instruction_list_one_kind():
        kind = decoded_instruction.instruction_kind.ttqs
        instruction_set = tensix.get_default_instruction_set_from_kind(kind)
        mnemonics_words = {mnemonic : decoded_instruction.swizzle_instruction(decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32)) for mnemonic, info in instruction_set.items()}
        dec_ins = [tensix.decode_instruction(word, kind = kind) for word in mnemonics_words.values()]
        ins_to_str0 = tensix.instruction_to_str(dec_ins)
        ins_to_str1 = tensix.instruction_to_str(dec_ins, instruction_set)
        ins_to_str2 = tensix.instruction_to_str(dec_ins, decoded_instruction.get_default_instruction_set_file_name(kind))
        assert ins_to_str0 == ins_to_str1
        assert ins_to_str0 == ins_to_str2
        for idx, dec_in in enumerate(dec_ins):
            assert hasattr(dec_in, 'kind')
            assert hasattr(dec_in, 'mnemonic')
            assert hasattr(dec_in, 'word')
            assert dec_in.kind == kind
            assert dec_in.word == mnemonics_words[dec_in.mnemonic]
            ins_to_str = ins_to_str0[idx]
            if isinstance(ins_to_str, str):
                assert dec_in.mnemonic.lower() in ins_to_str
                assert ins_to_str.startswith(f"0x{dec_in.word:08x} {dec_in.mnemonic.lower()}")
            elif isinstance(ins_to_str, list):
                for idx2 in range(len(ins_to_str)):
                    ins_to_str_ele = ins_to_str[idx]
                    if isinstance(ins_to_str_ele, str):
                        assert dec_in.mnemonic.lower() in ins_to_str_ele
                        assert ins_to_str_ele.startswith(f"0x{dec_in.word:08x} {dec_in.mnemonic.lower()}")
                    else:
                        raise Exception(f"- error: expected str type, received type {type(ins_to_str_ele)}")
            else:
                raise Exception(f"- error: expected str/list[str] type, received type {type(ins_to_str)}")

            if instruction_set[dec_in.mnemonic]["arguments"]:
                if isinstance(ins_to_str0[idx], str):
                    assert all(arg["name"] in ins_to_str0[idx] for arg in instruction_set[dec_in.mnemonic]["arguments"])
                else:
                    raise Exception(f"- error: expected str type, received type {type(ins_to_str0[idx])}")

    def test_multiple_kinds():
        num_instructions_per_kind = 6
        dec_ins: list[decoded_instruction.decoded_instruction] = list()
        mnemonics_kinds_words = dict()
        kinds_instruction_sets = dict()
        for kind in decoded_instruction.instruction_kind:
            if not kind.is_tensix():
                continue
            num_instrs = 0
            instruction_set = tensix.get_default_instruction_set_from_kind(kind)
            kinds_instruction_sets[kind] = instruction_set
            for mnemonic, info in instruction_set.items():
                if sum([mnemonic == ele.mnemonic for ele in dec_ins]):
                    continue
                else:
                    mnemonics_kinds_words[mnemonic] = (kind, decoded_instruction.swizzle_instruction(decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32)))
                    dec_ins.append(tensix.decode_instruction(mnemonics_kinds_words[mnemonic][1], kind = kind))
                    num_instrs += 1

                    if num_instrs > num_instructions_per_kind:
                        break

        msgs = tensix.instruction_to_str(dec_ins, instruction_set = kinds_instruction_sets)
        assert msgs == tensix.instruction_to_str(dec_ins)
        for idx, dec_in in enumerate(dec_ins):
            assert hasattr(dec_in, 'kind')
            assert hasattr(dec_in, 'mnemonic')
            assert hasattr(dec_in, 'word')
            assert dec_in.kind == mnemonics_kinds_words[dec_in.mnemonic][0]
            assert dec_in.word == mnemonics_kinds_words[dec_in.mnemonic][1]
            msg = msgs[idx]
            if isinstance(msg, str):
                assert dec_in.mnemonic.lower() in msg
                assert msg.startswith(f"0x{dec_in.word:08x} {dec_in.mnemonic.lower()}")
                if kinds_instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"]:
                    assert all(arg["name"] in msg for arg in kinds_instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"])
            elif isinstance(msg, list):
                for ele in msg:
                    if isinstance(ele, str):
                        assert dec_in.mnemonic.lower() in ele
                        assert ele.startswith(f"0x{dec_in.word:08x} {dec_in.mnemonic.lower()}")
                        if kinds_instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"]:
                            assert all(arg["name"] in ele for arg in kinds_instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"])


    test_one_instruction_per_call()
    test_instruction_list_one_kind()
    test_multiple_kinds()

def test_print_instruction(capsys):
    for kind in decoded_instruction.instruction_kind:
        if not kind.is_tensix():
            continue
        opcode = 0x2
        word = decoded_instruction.swizzle_instruction(decoded_instruction.right_circular_shift(opcode, shift = 8, num_bits = 32))
        di = tensix.decode_instruction(word, kind = kind)
        di.set_program_counter(0)
        capsys.readouterr()
        tensix.print_instruction(di)
        captured = capsys.readouterr()
        msg = "  0x00000000: 0x08000000 nop\n"
        assert captured.out == msg