#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import types
import typing

import textwrap

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.instructions as instructions
import ttsim.front.llk.rv32 as rv32
import ttsim.front.llk.tensix as tensix
import ttsim.front.llk.ttbh as ttbh
import ttsim.front.llk.ttqs as ttqs
import ttsim.front.llk.ttwh as ttwh

def test_get_kinds_from_modules_and_instruction_sets():
    assert set() == instructions.get_instruction_kinds(kinds = None, modules = None, instruction_sets = None)

    kinds_modules = instructions.get_modules([kind for kind in decoded_instruction.instruction_kind])
    kinds_instruction_sets = instructions.get_default_instruction_sets([kind for kind in decoded_instruction.instruction_kind])

    assert sorted([decoded_instruction.instruction_kind.ttwh, decoded_instruction.instruction_kind.rv32]) == sorted(instructions.get_instruction_kinds(kinds = None, modules = {
        decoded_instruction.instruction_kind.ttwh : ttwh,
        decoded_instruction.instruction_kind.rv32 : rv32},
        instruction_sets = None))

    assert sorted([decoded_instruction.instruction_kind.ttbh, decoded_instruction.instruction_kind.ttqs]) == sorted(instructions.get_instruction_kinds(kinds = None, modules = None,
        instruction_sets = {
        decoded_instruction.instruction_kind.ttbh : kinds_instruction_sets[decoded_instruction.instruction_kind.ttbh],
        decoded_instruction.instruction_kind.ttqs : kinds_instruction_sets[decoded_instruction.instruction_kind.ttqs]}))

    assert sorted([kind for kind in decoded_instruction.instruction_kind]) == sorted(instructions.get_instruction_kinds(
        kinds = None,
        modules = kinds_modules,
        instruction_sets = kinds_instruction_sets))

def test_get_instruction_kinds():
    def test_none():
        assert set() == instructions.get_instruction_kinds(kinds = None, modules = None, instruction_sets = None)

    def test_kinds():
        kinds = {
            decoded_instruction.instruction_kind.ttwh,
            decoded_instruction.instruction_kind.ttbh
        }

        assert sorted(kinds) == sorted(instructions.get_instruction_kinds(kinds = kinds, modules = None, instruction_sets = None))

    def test_modules():
        kinds_modules = {
            decoded_instruction.instruction_kind.ttbh : ttbh,
            decoded_instruction.instruction_kind.rv32 : rv32
        }

        kinds = sorted(instructions.get_instruction_kinds(kinds = None, modules = kinds_modules, instruction_sets = None))
        assert kinds == sorted(kinds_modules.keys())

    def test_instruction_sets():
        kinds_instruction_sets: typing.Any = {
            decoded_instruction.instruction_kind.ttwh : dict(),
            decoded_instruction.instruction_kind.ttqs : ""
        }

        kinds = sorted(instructions.get_instruction_kinds(kinds = None, modules = None, instruction_sets = kinds_instruction_sets))
        assert kinds == sorted(kinds_instruction_sets.keys())

    def test_multiple_arguments():
        kinds = set([decoded_instruction.instruction_kind.ttbh])

        kinds_modules: typing.Any = {
            decoded_instruction.instruction_kind.ttbh : ttbh,
            decoded_instruction.instruction_kind.rv32 : rv32
        }

        kinds_instruction_sets: typing.Any = {
            decoded_instruction.instruction_kind.ttbh : dict(),
            decoded_instruction.instruction_kind.ttqs : ""
        }

        exp_kinds_110: typing.Any = set(kinds)
        exp_kinds_110 = exp_kinds_110.union(kinds_modules.keys())
        exp_kinds_110 = sorted(exp_kinds_110)

        exp_kinds_101: typing.Any = set(kinds)
        exp_kinds_101 = exp_kinds_101.union(kinds_instruction_sets.keys())
        exp_kinds_101 = sorted(exp_kinds_101)

        exp_kinds_011: typing.Any = set(kinds_modules.keys())
        exp_kinds_011 = exp_kinds_011.union(kinds_instruction_sets.keys())
        exp_kinds_011 = sorted(exp_kinds_011)

        exp_kinds_111: typing.Any = set(kinds)
        exp_kinds_111 = exp_kinds_111.union(kinds_modules.keys())
        exp_kinds_111 = exp_kinds_111.union(kinds_instruction_sets.keys())
        exp_kinds_111 = sorted(exp_kinds_111)

        assert exp_kinds_110 == sorted(instructions.get_instruction_kinds(kinds = kinds, modules = kinds_modules, instruction_sets = None))
        assert exp_kinds_101 == sorted(instructions.get_instruction_kinds(kinds = kinds, modules = None, instruction_sets = kinds_instruction_sets))
        assert exp_kinds_011 == sorted(instructions.get_instruction_kinds(kinds = None,  modules = kinds_modules, instruction_sets = kinds_instruction_sets))
        assert exp_kinds_111 == sorted(instructions.get_instruction_kinds(kinds = kinds,  modules = kinds_modules, instruction_sets = kinds_instruction_sets))

    test_none()
    test_kinds()
    test_modules()
    test_instruction_sets()
    test_multiple_arguments()

def test_get_modules_0():
    kinds0 = {decoded_instruction.instruction_kind.ttqs, decoded_instruction.instruction_kind.rv32}
    kinds_modules0 = {
        decoded_instruction.instruction_kind.ttwh : ttwh,
        decoded_instruction.instruction_kind.ttbh : ttbh,
        }

    kinds_modules = {
        decoded_instruction.instruction_kind.ttwh : ttwh,
        decoded_instruction.instruction_kind.ttbh : ttbh,
        decoded_instruction.instruction_kind.ttqs : ttqs,
        decoded_instruction.instruction_kind.rv32 : rv32}

    assert kinds_modules == instructions.get_modules(kinds = kinds0, modules = kinds_modules0)

    kinds_modules1 = instructions.get_modules(kinds = kinds0, modules = None)
    assert sorted(kinds0) == sorted(kinds_modules1.keys())
    for kind, module in kinds_modules1.items():
        match kind:
            case decoded_instruction.instruction_kind.ttqs:
                assert module is ttqs
            case decoded_instruction.instruction_kind.rv32:
                assert module is rv32
            case _:
                raise Exception(f"- error: no comparison method defined for {kind} as it is not expected part of kinds_modules1 dict")

    assert {decoded_instruction.instruction_kind.rv32 : rv32} == instructions.get_modules(kinds = {decoded_instruction.instruction_kind.rv32}, modules = typing.cast(typing.Any, {"a" : 0}))
    assert {decoded_instruction.instruction_kind.rv32 : rv32} == instructions.get_modules(kinds = {decoded_instruction.instruction_kind.rv32}, modules = typing.cast(typing.Any, {decoded_instruction.instruction_kind.ttqs : 0}))

def test_get_modules_1(monkeypatch):
    def make_module(name: str, kind: decoded_instruction.instruction_kind) -> types.ModuleType:
        mod = types.ModuleType(name)
        return mod

    kind = decoded_instruction.instruction_kind.ttbh
    my_module = make_module(f"{kind}", kind)

    with pytest.raises(Exception):
        instructions.get_modules(kind, modules = my_module)

def test_get_modules_2(monkeypatch):
    def make_module(name: str, kind: decoded_instruction.instruction_kind) -> types.ModuleType:
        mod = types.ModuleType(name)
        def instruction_kind() -> decoded_instruction.instruction_kind:
            return kind
        # mod.instruction_kind = instruction_kind # type: ignore[attr-defined]
        setattr(mod, "instruction_kind", instruction_kind)

        return mod

    kind = decoded_instruction.instruction_kind.ttbh
    my_module = make_module(f"{kind}", kind)
    assert {kind : my_module} == instructions.get_modules(kind, modules = my_module)

def test_get_modules_3(monkeypatch):
    def make_module(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)
        return mod

    kind = decoded_instruction.instruction_kind.ttbh
    my_module = make_module(f"{kind}")
    {kind : my_module} == instructions.get_modules(modules = {kind : my_module})

def test_get_modules_4(monkeypatch):
    def make_module(name: str, kind: decoded_instruction.instruction_kind) -> types.ModuleType:
        mod = types.ModuleType(name)
        def instruction_kind() -> decoded_instruction.instruction_kind:
            return kind
        # mod.instruction_kind = instruction_kind # type: ignore[attr-defined]
        setattr(mod, "instruction_kind", instruction_kind)

        return mod

    kind = decoded_instruction.instruction_kind.ttbh
    my_module = make_module(f"{kind}", decoded_instruction.instruction_kind.ttwh)

    with pytest.raises(Exception):
        instructions.get_modules(modules = {kind : my_module})

def test_get_module_0():
    for kind in decoded_instruction.instruction_kind:
        module = instructions.get_module(kind)
        if decoded_instruction.instruction_kind.ttwh == kind:
            assert module is ttwh
        elif decoded_instruction.instruction_kind.ttbh == kind:
            assert module is ttbh
        elif decoded_instruction.instruction_kind.ttqs == kind:
            assert module is ttqs
        elif decoded_instruction.instruction_kind.rv32 == kind:
            assert module is rv32
        else:
            raise Exception(f"- error: no module comparison defined for instruction of kind {kind}")

    with pytest.raises(Exception):
        instructions.get_module(set([kind for kind in decoded_instruction.instruction_kind])) # type: ignore[arg-type]

def test_get_module_1(monkeypatch):
    def make_module(name: str, kind: decoded_instruction.instruction_kind) -> types.ModuleType:
        mod = types.ModuleType(name)
        def instruction_kind() -> decoded_instruction.instruction_kind:
            return kind
        # mod.instruction_kind = instruction_kind # type: ignore[attr-defined]
        setattr(mod, "instruction_kind", instruction_kind)

        return mod

    kinds_modules = dict()
    for kind in decoded_instruction.instruction_kind:
        kinds_modules[kind] = make_module(f"{kind}", kind)

    for kind in decoded_instruction.instruction_kind:
        module = instructions.get_module(kind, modules = kinds_modules)
        assert (kinds_modules[kind] is module) and module.instruction_kind() == kind

def test_get_module_2(monkeypatch):
    def make_module(name: str, kind: decoded_instruction.instruction_kind) -> types.ModuleType:
        mod = types.ModuleType(name)
        def instruction_kind() -> decoded_instruction.instruction_kind:
            return kind
        # mod.instruction_kind = instruction_kind # type: ignore[attr-defined]
        setattr(mod, "instruction_kind", instruction_kind)

        return mod

    kinds_modules = dict()

    kind = decoded_instruction.instruction_kind.ttbh
    kinds_modules[kind] = make_module(f"{kind}", kind)

    kind = decoded_instruction.instruction_kind.ttqs
    kinds_modules[kind] = make_module(f"{kind}", kind)

    kinds_modules[ttwh.instruction_kind()] = ttwh
    kinds_modules[rv32.instruction_kind()] = rv32

    for kind in decoded_instruction.instruction_kind:
        module = instructions.get_module(kind, modules = kinds_modules)
        assert (kinds_modules[kind] is module) and module.instruction_kind() == kind

def test_get_module_3():
    kinds_modules = dict()
    kinds_modules[ttwh.instruction_kind()] = rv32
    kinds_modules[rv32.instruction_kind()] = ttwh

    with pytest.raises(Exception):
        instructions.get_module(ttwh.instruction_kind(), modules = kinds_modules)

def test_get_module_4(monkeypatch):
    def make_module(name: str, kind: decoded_instruction.instruction_kind) -> types.ModuleType:
        mod = types.ModuleType(name)
        return mod

    kinds_modules = dict()

    kind = decoded_instruction.instruction_kind.ttbh
    kinds_modules[kind] = make_module(f"{kind}", kind)

    assert kinds_modules[kind] == instructions.get_module(kind, modules = kinds_modules)

def test_get_module_5(monkeypatch):
    def make_module(name: str, kind: decoded_instruction.instruction_kind) -> types.ModuleType:
        mod = types.ModuleType(name)
        return mod

    kind = decoded_instruction.instruction_kind.ttbh
    my_module = make_module(f"{kind}", kind)

    with pytest.raises(Exception):
        instructions.get_module(kind, modules = my_module)

def test_get_default_modules():
    kinds_modules = dict()
    kinds_modules[decoded_instruction.instruction_kind.rv32] = rv32
    kinds_modules[decoded_instruction.instruction_kind.ttbh] = ttbh
    kinds_modules[decoded_instruction.instruction_kind.ttqs] = ttqs
    kinds_modules[decoded_instruction.instruction_kind.ttwh] = ttwh

    for kind in decoded_instruction.instruction_kind:
        assert {kind : kinds_modules[kind]} == instructions.get_default_modules(kind)

    assert kinds_modules == instructions.get_default_modules(decoded_instruction.get_instruction_kinds())

def test_get_default_module():
    kinds_modules = dict()
    kinds_modules[decoded_instruction.instruction_kind.rv32] = rv32
    kinds_modules[decoded_instruction.instruction_kind.ttbh] = ttbh
    kinds_modules[decoded_instruction.instruction_kind.ttqs] = ttqs
    kinds_modules[decoded_instruction.instruction_kind.ttwh] = ttwh

    for kind in decoded_instruction.instruction_kind:
        assert kinds_modules[kind] == instructions.get_default_module(kind)

    with pytest.raises(Exception):
        instructions.get_default_module(decoded_instruction.get_instruction_kinds()) # type: ignore[arg-type]

def test_get_default_modules_from_riscv_attribute():
    default_kinds_modules = instructions.get_default_modules(decoded_instruction.get_instruction_kinds())

    for kinds, attrs in decoded_instruction.get_instruction_kinds_rv32_tensix_attributes_dict().items():
        for attr in attrs:
            kinds_modules = instructions.get_default_modules_from_riscv_attribute(attr)
            assert sorted(kinds) == sorted(kinds_modules.keys())
            for kind in kinds_modules.keys():
                assert kinds_modules[kind] == default_kinds_modules[kind]

@pytest.mark.slow
def test_get_instruction_sets_0():
    kinds_sets = instructions.get_instruction_sets(decoded_instruction.get_instruction_kinds())
    for kind in decoded_instruction.instruction_kind:
        instr_set = decoded_instruction.get_default_instruction_set(kind)
        assert sorted(instr_set.keys()) == sorted(kinds_sets[kind].keys())

    kinds_modules = instructions.get_modules(decoded_instruction.get_instruction_kinds())
    for kind, module in kinds_modules.items():
        assert module.get_instruction_set() == kinds_sets[kind]

def test_get_instruction_sets_1():
    kinds_set_strs = {kind : decoded_instruction.get_default_instruction_set_file_name(kind) for kind in decoded_instruction.get_instruction_kinds()}
    kinds_sets = instructions.get_instruction_sets(instruction_sets = kinds_set_strs)
    kinds_modules = instructions.get_modules(decoded_instruction.get_instruction_kinds())
    for kind, module in kinds_modules.items():
        assert module.get_instruction_set() == kinds_sets[kind]

def test_get_instruction_sets_2():
    with pytest.raises(Exception):
        instruction_sets = decoded_instruction.get_default_instruction_set_file_name(decoded_instruction.instruction_kind.ttwh)
        instructions.get_instruction_sets(instruction_sets = instruction_sets)

def test_get_instruction_sets_3():
    with pytest.raises(Exception):
        instructions.get_instruction_sets(instruction_sets = dict())

def test_get_instruction_sets_4():
    assert {ttwh.instruction_kind() : ttwh.get_instruction_set()} == instructions.get_instruction_sets(modules = ttwh)

def test_get_instruction_sets_5():
    kinds_modules = instructions.get_modules(decoded_instruction.get_instruction_kinds())
    kinds_sets = instructions.get_instruction_sets(modules = kinds_modules)
    for kind, module in kinds_modules.items():
        assert module.get_instruction_set() == kinds_sets[kind]

def test_get_instruction_sets_6():
    kind = decoded_instruction.instruction_kind.ttwh
    ins_set: dict[str, typing.Any] = dict()
    kinds_sets = instructions.get_instruction_sets(kinds = kind, instruction_sets = ins_set)
    assert sorted(kinds_sets.keys()) == [kind]
    assert kinds_sets[kind] == ins_set

def test_get_instruction_sets_7():
    ins_sets: dict[decoded_instruction.instruction_kind, None | str | decoded_instruction.instruction_kind | dict[str, typing.Any]] = dict()
    ins_sets[decoded_instruction.instruction_kind.ttwh] = None
    ins_sets[decoded_instruction.instruction_kind.ttbh] = decoded_instruction.get_default_instruction_set_file_name(decoded_instruction.instruction_kind.ttbh)
    ins_sets[decoded_instruction.instruction_kind.ttqs] = decoded_instruction.instruction_kind.ttqs
    ins_sets[decoded_instruction.instruction_kind.rv32] = dict({"a" : dict()})

    kinds_sets = instructions.get_instruction_sets(instruction_sets = ins_sets)
    assert sorted(kinds_sets.keys()) == sorted(ins_sets.keys())
    assert ttwh.get_instruction_set() == kinds_sets[decoded_instruction.instruction_kind.ttwh]
    assert ttbh.get_instruction_set() == kinds_sets[decoded_instruction.instruction_kind.ttbh]
    assert ttqs.get_instruction_set() == kinds_sets[decoded_instruction.instruction_kind.ttqs]
    assert ins_sets[decoded_instruction.instruction_kind.rv32] == kinds_sets[decoded_instruction.instruction_kind.rv32]

def test_get_instruction_set_0():
    kinds_modules = instructions.get_modules(decoded_instruction.get_instruction_kinds())
    for kind, module in kinds_modules.items():
        assert module.get_instruction_set() == instructions.get_instruction_set(kind = kind)

def test_get_instruction_set_1():
    ins_set: dict[str, typing.Any] = dict()
    assert ins_set == instructions.get_instruction_set(kind = decoded_instruction.instruction_kind.ttwh, instruction_sets = ins_set)

def test_get_instruction_set_2():
    with pytest.raises(Exception):
        instructions.get_instruction_set(kind = None) # type: ignore[arg-type]

    with pytest.raises(Exception):
        instructions.get_instruction_set(kind = "") # type: ignore[arg-type]

@pytest.mark.slow
def test_get_default_instruction_sets():
    kinds_modules = instructions.get_modules(decoded_instruction.get_instruction_kinds())
    kinds_sets = instructions.get_default_instruction_sets(decoded_instruction.get_instruction_kinds())
    for kind, module in kinds_modules.items():
        def_ins_set = module.get_instruction_set()
        assert def_ins_set == kinds_sets[kind]

def test_get_default_instruction_set_0():
    kinds_modules = instructions.get_modules(decoded_instruction.get_instruction_kinds())
    for kind, module in kinds_modules.items():
        assert module.get_instruction_set() == instructions.get_default_instruction_set(kind)

def test_get_default_instruction_set_1():
    with pytest.raises(Exception):
        instructions.get_default_instruction_set(kind = typing.cast(typing.Any, ""))

    with pytest.raises(Exception):
        instructions.get_default_instruction_set(kind = typing.cast(typing.Any, None))

@pytest.mark.slow
def test_get_default_instruction_sets_from_riscv_attribute():
    for kinds, attrs in decoded_instruction.get_instruction_kinds_rv32_tensix_attributes_dict().items():
        for attr in attrs:
            kinds_sets = instructions.get_default_instruction_sets_from_riscv_attribute(attr)
            kinds_modules = instructions.get_default_modules_from_riscv_attribute(attr)

            assert sorted(kinds) == sorted(kinds_modules.keys())
            assert sorted(kinds) == sorted(kinds_sets.keys())

            for kind in kinds_sets.keys():
                assert kinds_sets[kind] == kinds_modules[kind].get_instruction_set()

def test_get_valid_instruction_kinds():
    ttword = instructions.swizzle_instruction(0x5160000b)
    words = sorted(set([ttword, 0x0087a783]))

    for word in words:
        kinds = instructions.get_valid_instruction_kinds(word)
        if 0x0087a783 == word:
            expected: typing.Any = set([decoded_instruction.instruction_kind.rv32])
            assert expected == kinds
        elif ttword == word:
            expected = sorted([kind for kind in decoded_instruction.instruction_kind if kind.is_tensix()])
            assert expected == sorted(kinds)
        else:
            raise Exception(f"- error: no method described to compare given word {hex(word)}")

    word = 0x0087a783
    kinds = instructions.get_valid_instruction_kinds(word, kinds = [decoded_instruction.instruction_kind.ttwh], modules = {decoded_instruction.instruction_kind.ttbh : ttbh})
    assert not kinds

    with pytest.raises(Exception) as exe_info:
        word = 0x0087a783
        kinds = instructions.get_valid_instruction_kinds(word, kinds = [decoded_instruction.instruction_kind.ttwh], modules = dict({decoded_instruction.instruction_kind.ttbh : ttbh}), raise_exception = True)
    assert "could not determine kind for given instruction" in str(exe_info.value)

def test_get_valid_instruction_kind():
    ttword = instructions.swizzle_instruction(0x5160000b)
    words = sorted(set([ttword, 0x0087a783]))

    for word in words:
        kinds: typing.Any = instructions.get_valid_instruction_kind(word)
        if 0x0087a783 == word:
            assert isinstance(kinds, decoded_instruction.instruction_kind)
            assert decoded_instruction.instruction_kind.rv32 == kinds
        elif ttword == word:
            assert isinstance(kinds, set) and all(isinstance(ele, decoded_instruction.instruction_kind) for ele in kinds)
            assert sorted([kind for kind in decoded_instruction.instruction_kind if kind.is_tensix()]) == sorted(kinds)
        else:
            raise Exception(f"- error: no method described to compare given word {hex(word)}")

    word = 0x0087a783
    kinds = instructions.get_valid_instruction_kind(word, kinds = [decoded_instruction.instruction_kind.ttwh], modules = {decoded_instruction.instruction_kind.ttbh : ttbh})
    assert not kinds

    with pytest.raises(Exception) as exe_info:
        word = 0x0087a783
        kinds = instructions.get_valid_instruction_kind(word, kinds = [decoded_instruction.instruction_kind.ttwh], modules = dict({decoded_instruction.instruction_kind.ttbh : ttbh}), raise_exception = True)
    assert "could not determine kind for given instruction" in str(exe_info.value)

def test_get_instruction_kinds_from_decoded_instructions():
    def test_list():
        decoded_instructions = [
            rv32.decode_instruction(0x0087a783),
            tensix.decode_instruction(instructions.swizzle_instruction(0x5160000b), kind = decoded_instruction.instruction_kind.ttwh)
        ]

        kinds0 = sorted(instructions.get_instruction_kinds_from_decoded_instructions(decoded_instructions, flatten_dict = False))
        kinds1 = sorted(instructions.get_instruction_kinds_from_decoded_instructions(decoded_instructions, flatten_dict = True))
        assert kinds0 == kinds1
        assert kinds0 == sorted([decoded_instruction.instruction_kind.ttwh, decoded_instruction.instruction_kind.rv32])

    def test_dict():
        decoded_instructions = {
            'func1' : [rv32.decode_instruction(0x0087a783)],
            'func2' : [tensix.decode_instruction(instructions.swizzle_instruction(0x5160000b), kind = decoded_instruction.instruction_kind.ttbh)],
            'func3' : [tensix.decode_instruction(instructions.swizzle_instruction(0x5160000b), kind = decoded_instruction.instruction_kind.ttqs)]
        }

        expected_kinds0 = sorted([
            decoded_instruction.instruction_kind.rv32,
            decoded_instruction.instruction_kind.ttbh,
            decoded_instruction.instruction_kind.ttqs])

        expected_kinds1 = {
            'func1' : {decoded_instruction.instruction_kind.rv32},
            'func2' : {decoded_instruction.instruction_kind.ttbh},
            'func3' : {decoded_instruction.instruction_kind.ttqs}
        }

        assert expected_kinds0 == sorted(instructions.get_instruction_kinds_from_decoded_instructions(decoded_instructions, flatten_dict = True))
        assert expected_kinds1 == instructions.get_instruction_kinds_from_decoded_instructions(decoded_instructions, flatten_dict = False)

    test_list()
    test_dict()

@pytest.mark.slow
def test_decode_instruction():
    def test_tensix():
        word = instructions.swizzle_instruction(0x5160000b)
        unswizzled_word = instructions.unswizzle_instruction(word)
        tensix_kinds = sorted([kind for kind in decoded_instruction.instruction_kind if kind.is_tensix()])

        # 0,0,0,0: kind, module, instruction_set, swizzle
        di_0000 = instructions.decode_instruction(unswizzled_word, is_swizzled = False)
        assert isinstance(di_0000, list) and all(isinstance(ele, decoded_instruction.decoded_instruction) for ele in di_0000)
        assert sorted([ele.kind for ele in di_0000]) == tensix_kinds

        # 0,0,0,1: kind, module, instruction_set, swizzle
        di_0001 = instructions.decode_instruction(word, is_swizzled = True)
        assert isinstance(di_0001, list) and all(isinstance(ele, decoded_instruction.decoded_instruction) for ele in di_0001)
        assert sorted([ele.kind for ele in di_0001]) == tensix_kinds
        assert all(value == di_0000[idx] for idx, value in enumerate(di_0001))

        for kind in decoded_instruction.instruction_kind:
            if not kind.is_tensix():
                continue

            module = instructions.get_module(kind)
            instruction_set = instructions.get_instruction_set(kind)

            # 0,0,1,0: kind, module, instruction_set, swizzle
            with pytest.raises(Exception) as exe_info:
                instructions.decode_instruction(unswizzled_word, instruction_set = instruction_set, is_swizzled = False)
            # assert "please provide either instruction kind and/or module" in str(exe_info.value), "- error: error message mismatch"
            assert "expected only one instruction kind as argument" in str(exe_info.value)

            # 0,0,1,1: kind, module, instruction_set, swizzle
            with pytest.raises(Exception) as exe_info:
                instructions.decode_instruction(word, instruction_set = instruction_set, is_swizzled = True)
            assert "expected only one instruction kind as argument" in str(exe_info.value)

            # 0,1,0,0: kind, module, instruction_set, swizzle
            di_1000 = instructions.decode_instruction(unswizzled_word, kind = kind, is_swizzled = False)
            di_1001 = instructions.decode_instruction(word,            kind = kind, is_swizzled = True)
            di_1010 = instructions.decode_instruction(unswizzled_word, kind = kind, instruction_set = instruction_set, is_swizzled = False)
            di_1011 = instructions.decode_instruction(word,            kind = kind, instruction_set = instruction_set, is_swizzled = True)

            di_0100 = instructions.decode_instruction(unswizzled_word, module = module, is_swizzled = False)
            di_0101 = instructions.decode_instruction(word,            module = module, is_swizzled = True)
            di_0110 = instructions.decode_instruction(unswizzled_word, module = module, instruction_set = instruction_set, is_swizzled = False)
            di_0111 = instructions.decode_instruction(word,            module = module, instruction_set = instruction_set, is_swizzled = True)

            di_1100 = instructions.decode_instruction(unswizzled_word, kind = kind, module = module, is_swizzled = False)
            di_1101 = instructions.decode_instruction(word,            kind = kind, module = module, is_swizzled = True)
            di_1110 = instructions.decode_instruction(unswizzled_word, kind = kind, module = module, instruction_set = instruction_set, is_swizzled = False)
            di_1111 = instructions.decode_instruction(word,            kind = kind, module = module, instruction_set = instruction_set, is_swizzled = True)

            assert di_1000 == di_1001
            assert di_1000 == di_1010
            assert di_1000 == di_1011
            assert di_1000 == di_0100
            assert di_1000 == di_0101
            assert di_1000 == di_0110
            assert di_1000 == di_0111
            assert di_1000 == di_1100
            assert di_1000 == di_1101
            assert di_1000 == di_1110
            assert di_1000 == di_1111
            assert sum([di_1000 == ele for ele in di_0000 if ele.kind == kind])
            assert sum([di_1000 == ele for ele in di_0001 if ele.kind == kind])

    def test_riscv():
        word = 0x0087a783
        unswizzled_word = decoded_instruction.unswizzle_instruction(word)
        kind = decoded_instruction.instruction_kind.rv32
        module = rv32
        # instruction_set = instructions.get_instruction_set(kind)
        instruction_set = instructions.get_instruction_set(kind)

        # 0,0,0,0: kind, module, instruction_set, swizzle
        di_0000 = instructions.decode_instruction(unswizzled_word, is_swizzled = False)
        assert hasattr(di_0000, 'kind')
        assert di_0000.kind == kind

        # 0,0,0,1: kind, module, instruction_set, swizzle
        di_0001 = instructions.decode_instruction(word, is_swizzled = True)
        assert hasattr(di_0001, 'kind')
        assert di_0001.kind == kind
        assert di_0000 == di_0001

        # 0,0,1,0: kind, module, instruction_set, swizzle
        di_0010 = instructions.decode_instruction(unswizzled_word, instruction_set = instruction_set, is_swizzled = False)

        # 0,0,1,1: kind, module, instruction_set, swizzle
        di_0011 = instructions.decode_instruction(word, instruction_set = instruction_set, is_swizzled = True)

        # 0,1,0,0: kind, module, instruction_set, swizzle
        di_0100 = instructions.decode_instruction(unswizzled_word, module = module, is_swizzled = False)
        di_0101 = instructions.decode_instruction(word,            module = module, is_swizzled = True)
        di_0110 = instructions.decode_instruction(unswizzled_word, module = module, instruction_set = instruction_set, is_swizzled = False)
        di_0111 = instructions.decode_instruction(word,            module = module, instruction_set = instruction_set, is_swizzled = True)

        di_1000 = instructions.decode_instruction(unswizzled_word, kind = kind, is_swizzled = False)
        di_1001 = instructions.decode_instruction(word,            kind = kind, is_swizzled = True)
        di_1010 = instructions.decode_instruction(unswizzled_word, kind = kind, instruction_set = instruction_set, is_swizzled = False)
        di_1011 = instructions.decode_instruction(word,            kind = kind, instruction_set = instruction_set, is_swizzled = True)

        di_1100 = instructions.decode_instruction(unswizzled_word, kind = kind, module = module, is_swizzled = False)
        di_1101 = instructions.decode_instruction(word,            kind = kind, module = module, is_swizzled = True)
        di_1110 = instructions.decode_instruction(unswizzled_word, kind = kind, module = module, instruction_set = instruction_set, is_swizzled = False)
        di_1111 = instructions.decode_instruction(word,            kind = kind, module = module, instruction_set = instruction_set, is_swizzled = True)

        assert di_1000 == di_0010
        assert di_1000 == di_0011
        assert di_1000 == di_1001
        assert di_1000 == di_1010
        assert di_1000 == di_1011
        assert di_1000 == di_0100
        assert di_1000 == di_0101
        assert di_1000 == di_0110
        assert di_1000 == di_0111
        assert di_1000 == di_1100
        assert di_1000 == di_1101
        assert di_1000 == di_1110
        assert di_1000 == di_1111
        assert di_1000 == di_0000
        assert di_1000 == di_0001

    def test_module_mismatch():
        word = instructions.swizzle_instruction(0x5160000b)
        assert instructions.decode_instruction(word, kind = decoded_instruction.instruction_kind.ttwh, module = ttwh) == instructions.decode_instruction(word, kind = decoded_instruction.instruction_kind.ttwh, module = ttbh)

    def test_exceptions():
        with pytest.raises(Exception):
            instructions.decode_instruction(typing.cast(typing.Any, [0]))

    test_tensix()
    test_riscv()
    test_module_mismatch()
    test_exceptions()

def test_decode_instructions():
    words = [0x0087a783, instructions.swizzle_instruction(0x5160000b)]
    words_in_bytes: typing.Any = bytearray(len(words) * decoded_instruction.get_num_bytes_per_instruction())
    for idx, word in enumerate(words):
        words_in_bytes[(idx * decoded_instruction.get_num_bytes_per_instruction()):((idx * decoded_instruction.get_num_bytes_per_instruction())+4)] = word.to_bytes(decoded_instruction.get_num_bytes_per_instruction(), byteorder = 'little')
    words_in_bytes = bytes(words_in_bytes)

    decoded_instructions = instructions.decode_instructions(words_in_bytes, kinds = [decoded_instruction.instruction_kind.rv32, decoded_instruction.instruction_kind.ttqs])

    assert all(word == decoded_instructions[idx].word for idx, word in enumerate(words))

def test_get_statistics_from_list():
    words = [0x0087a783, instructions.swizzle_instruction(0x5160000b)]
    kinds = [decoded_instruction.instruction_kind.rv32, decoded_instruction.instruction_kind.ttwh]
    words_in_bytes: typing.Any = bytearray(len(words) * decoded_instruction.get_num_bytes_per_instruction())
    for idx, word in enumerate(words):
        words_in_bytes[(idx * decoded_instruction.get_num_bytes_per_instruction()):((idx * decoded_instruction.get_num_bytes_per_instruction())+4)] = word.to_bytes(decoded_instruction.get_num_bytes_per_instruction(), byteorder = 'little')
    words_in_bytes = bytes(words_in_bytes)
    decoded_instructions = instructions.decode_instructions(words_in_bytes, kinds = kinds)

    instruction_histogram0, kind_histogram0 = instructions.get_statistics_from_list(decoded_instructions)
    instruction_histogram1, kind_histogram1 = instructions.get_statistics_from_list(decoded_instructions, instructions.get_modules(kinds))

    assert instruction_histogram0 == instruction_histogram1
    assert kind_histogram0        == kind_histogram1
    assert sorted(instruction_histogram0.keys()) == sorted(kind_histogram0.keys())
    assert sorted(instruction_histogram0.keys()) == sorted(kinds)

    histogram: typing.Any = 0
    for kind, histogram in instruction_histogram0.items():
        for stats in histogram.values():
            assert stats == [1, 1.0, 0.5]

    for kind, histogram in kind_histogram0.items():
        assert histogram == {'decoded': 1, 'no_opcode': 0, 'no_mnemonic': 0}

def test_flatten_dict_and_get_statistics():
    functions_words = dict({
        'funct1' : [0x0087a783],
        'funct2' : [instructions.swizzle_instruction(0x5160000b)]})

    functions_kinds = dict({
        'funct1' : [decoded_instruction.instruction_kind.rv32],
        'funct2' : [decoded_instruction.instruction_kind.ttwh]})

    funcs_deins = dict()

    for function_name, words in functions_words.items():
        words_in_bytes: typing.Any = bytearray(len(words) * decoded_instruction.get_num_bytes_per_instruction())
        for idx, word in enumerate(words):
            words_in_bytes[(idx * decoded_instruction.get_num_bytes_per_instruction()):((idx * decoded_instruction.get_num_bytes_per_instruction())+4)] = word.to_bytes(decoded_instruction.get_num_bytes_per_instruction(), byteorder = 'little')
        words_in_bytes = bytes(words_in_bytes)
        decoded_instructions = instructions.decode_instructions(words_in_bytes, kinds = functions_kinds[function_name])
        funcs_deins[function_name] = decoded_instructions

    instruction_histogram0, kind_histogram0 = instructions.flatten_dict_and_get_statistics(funcs_deins)

    assert sorted(instruction_histogram0.keys()) == sorted(set([kind for kinds_in_function in functions_kinds.values() for kind in kinds_in_function]))
    assert sorted(instruction_histogram0.keys()) == sorted(kind_histogram0.keys())

    histogram: typing.Any = 0
    for kind, histogram in instruction_histogram0.items():
        for stats in histogram.values():
            assert stats == [1, 1.0, 0.5]

    for kind, histogram in kind_histogram0.items():
        assert histogram == {'decoded': 1, 'no_opcode': 0, 'no_mnemonic': 0}

def test_get_statistics_from_dict():
    def test_flatten_dict_is_true(funcs_deins):
        instruction_histogram, kind_histogram = instructions.get_statistics_from_dict(funcs_deins, flatten_dict = True)

        assert sorted(instruction_histogram.keys()) == sorted(set([kind for kinds_in_function in functions_kinds.values() for kind in kinds_in_function]))
        assert sorted(instruction_histogram.keys()) == sorted(kind_histogram.keys())

        histogram: typing.Any = 0
        for kind, histogram in instruction_histogram.items():
            for stats in histogram.values():
                assert stats == [1, 1.0, 0.5]

        for kind, histogram in kind_histogram.items():
            assert histogram == {'decoded': 1, 'no_opcode': 0, 'no_mnemonic': 0}

    def test_flatten_dict_is_false(funcs_deins):
        funcs_instruction_histogram, funcs_kind_histogram = instructions.get_statistics_from_dict(funcs_deins, flatten_dict = False)

        assert sorted(funcs_instruction_histogram.keys()) == sorted(funcs_deins.keys())
        assert sorted(funcs_instruction_histogram.keys()) == sorted(funcs_kind_histogram.keys())
        histogram: typing.Any = 0
        for _, instruction_histogram in funcs_instruction_histogram.items():
            for _, histogram in instruction_histogram.items():
                for stats in histogram.values():
                    assert stats == [1, 1.0, 1.0]

        for _, kind_histogram in funcs_kind_histogram.items():
            for _, histogram in kind_histogram.items():
                assert histogram == {'decoded': 1, 'no_opcode': 0, 'no_mnemonic': 0}

    functions_words = dict({
        'funct1' : [0x0087a783],
        'funct2' : [instructions.swizzle_instruction(0x5160000b)]})

    functions_kinds = dict({
        'funct1' : [decoded_instruction.instruction_kind.rv32],
        'funct2' : [decoded_instruction.instruction_kind.ttwh]})

    funcs_deins = dict()

    for function_name, words in functions_words.items():
        words_in_bytes: typing.Any = bytearray(len(words) * decoded_instruction.get_num_bytes_per_instruction())
        for idx, word in enumerate(words):
            words_in_bytes[(idx * decoded_instruction.get_num_bytes_per_instruction()):((idx * decoded_instruction.get_num_bytes_per_instruction())+4)] = word.to_bytes(decoded_instruction.get_num_bytes_per_instruction(), byteorder = 'little')
        words_in_bytes = bytes(words_in_bytes)
        decoded_instructions = instructions.decode_instructions(words_in_bytes, kinds = functions_kinds[function_name])
        funcs_deins[function_name] = decoded_instructions

    test_flatten_dict_is_true(funcs_deins)
    test_flatten_dict_is_false(funcs_deins)

def test_get_statistics():
    def test_list():
        words = [0x0087a783, instructions.swizzle_instruction(0x5160000b)]
        kinds = [decoded_instruction.instruction_kind.rv32, decoded_instruction.instruction_kind.ttwh]
        words_in_bytes: typing.Any = bytearray(len(words) * decoded_instruction.get_num_bytes_per_instruction())
        for idx, word in enumerate(words):
            words_in_bytes[(idx * decoded_instruction.get_num_bytes_per_instruction()):((idx * decoded_instruction.get_num_bytes_per_instruction())+4)] = word.to_bytes(decoded_instruction.get_num_bytes_per_instruction(), byteorder = 'little')
        words_in_bytes = bytes(words_in_bytes)
        decoded_instructions = instructions.decode_instructions(words_in_bytes, kinds = kinds)

        instruction_histogram, kind_histogram = instructions.get_statistics(decoded_instructions)

        assert sorted(instruction_histogram.keys()) == sorted(kind_histogram.keys())
        assert sorted(instruction_histogram.keys()) == sorted(kinds)
        histogram: typing.Any = 0
        for _, histogram in instruction_histogram.items():
            for stats in histogram.values():
                assert stats == [1, 1.0, 0.5]

        for _, histogram in kind_histogram.items():
            assert histogram == {'decoded': 1, 'no_opcode': 0, 'no_mnemonic': 0}

    def test_dict():
        def test_flatten_dict_is_true(funcs_deins):
            instruction_histogram, kind_histogram = instructions.get_statistics(funcs_deins, flatten_dict = True)

            assert sorted(instruction_histogram.keys()) == sorted(set([kind for kinds_in_function in functions_kinds.values() for kind in kinds_in_function]))
            assert sorted(instruction_histogram.keys()) == sorted(kind_histogram.keys())

            histogram: typing.Any = 0
            for _, histogram in instruction_histogram.items():
                for stats in histogram.values():
                    assert stats == [1, 1.0, 0.5]

            for _, histogram in kind_histogram.items():
                assert histogram == {'decoded': 1, 'no_opcode': 0, 'no_mnemonic': 0}

        def test_flatten_dict_is_false(funcs_deins):
            funcs_instruction_histogram, funcs_kind_histogram = instructions.get_statistics(funcs_deins, flatten_dict = False)

            assert sorted(funcs_instruction_histogram.keys()) == sorted(funcs_deins.keys())
            assert sorted(funcs_instruction_histogram.keys()) == sorted(funcs_kind_histogram.keys())

            histogram: typing.Any = 0
            for func_name, instruction_histogram in funcs_instruction_histogram.items():
                for kind, histogram in instruction_histogram.items():
                    for stats in histogram.values():
                        assert stats == [1, 1.0, 1.0]

            for _, kind_histogram in funcs_kind_histogram.items():
                for _, histogram in kind_histogram.items():
                    assert histogram == {'decoded': 1, 'no_opcode': 0, 'no_mnemonic': 0}

        functions_words = dict({
            'funct1' : [0x0087a783],
            'funct2' : [instructions.swizzle_instruction(0x5160000b)]})

        functions_kinds = dict({
            'funct1' : [decoded_instruction.instruction_kind.rv32],
            'funct2' : [decoded_instruction.instruction_kind.ttwh]})

        funcs_deins = dict()

        for function_name, words in functions_words.items():
            words_in_bytes: typing.Any = bytearray(len(words) * decoded_instruction.get_num_bytes_per_instruction())
            for idx, word in enumerate(words):
                words_in_bytes[(idx * decoded_instruction.get_num_bytes_per_instruction()):((idx * decoded_instruction.get_num_bytes_per_instruction())+4)] = word.to_bytes(decoded_instruction.get_num_bytes_per_instruction(), byteorder = 'little')
            words_in_bytes = bytes(words_in_bytes)
            decoded_instructions = instructions.decode_instructions(words_in_bytes, kinds = functions_kinds[function_name])
            funcs_deins[function_name] = decoded_instructions

        test_flatten_dict_is_true(funcs_deins)
        test_flatten_dict_is_false(funcs_deins)

    def test_exceptions():
        with pytest.raises(Exception) as exe_info:
            instructions.get_statistics(typing.cast(typing.Any, 0))
        assert "error: no method defined to get statistics from instructions of type" in str(exe_info.value)

    test_list()
    test_dict()
    test_exceptions()

def test_print_instruction(capsys):
    words_kinds_instruction_strs: dict[int, tuple[decoded_instruction.instruction_kind, str]] = {
        0x0087a783 : (decoded_instruction.instruction_kind.rv32, "0x0087a783 lw x15, 8(x15)"),
        instructions.swizzle_instruction(0x5160000b) : (decoded_instruction.instruction_kind.ttbh, "0x4580002d setadcxy CntSetMask[23:21] = 0b11, Ch1_Y[20:15] = 0, Ch1_X[14:12] = 0, Ch0_Y[11:9] = 0, Ch0_X[8:6] = 0, BitMask[5:0] = 0b1011")
    }

    for word, kind_ins_str in words_kinds_instruction_strs.items():
        kind = kind_ins_str[0]
        ins_str = kind_ins_str[1]
        di = instructions.decode_instruction(word, kind = kind)
        capsys.readouterr()
        instructions.print_instruction(di)
        captured = capsys.readouterr()
        msg = f"  {ins_str}\n"
        assert captured.out == msg

        capsys.readouterr()
        instructions.print_instruction({"abc" : di}) # dict input
        captured = capsys.readouterr()
        msg = f"  {ins_str}\n"
        assert captured.out == msg

    with pytest.raises(Exception) as exe_info:
        instructions.print_instruction(typing.cast(typing.Any, []))
    assert "no method defined to print instruction" in str(exe_info.value)

def test_print_instructions(capsys):
    def test_list(words_kinds_instruction_strs):
        dis = []
        instr_strs = []
        print_offset = 4
        for word, kind_instruction_str in words_kinds_instruction_strs.items():
            kind = kind_instruction_str[0]
            instr_str = kind_instruction_str[1]
            dis.append(instructions.decode_instruction(word, kind = kind))
            instr_strs.append(instr_str)

        capsys.readouterr()
        instructions.print_instructions(dis, print_offset = print_offset)
        captured = capsys.readouterr()
        expected_instr_strs = ""
        for ele in instr_strs:
            expected_instr_strs += f"{" " * print_offset}{ele}\n"

        assert captured.out == expected_instr_strs

    def test_dict(words_kinds_instruction_strs):
        funcs_dis: typing.Any = dict()
        funcs_instr_strs: typing.Any = dict()
        print_offset = 3

        funcs_dis['func1'] = []
        funcs_instr_strs['func1'] = []
        for word, kind_instruction_str in words_kinds_instruction_strs.items():
            kind = kind_instruction_str[0]
            instr_str = kind_instruction_str[1]
            funcs_dis['func1'].append(instructions.decode_instruction(word, kind = kind))
            funcs_instr_strs['func1'].append(instr_str)

        funcs_dis['func2'] = []
        funcs_instr_strs['func2'] = []
        for word, kind_instruction_str in reversed(words_kinds_instruction_strs.items()):
            kind = kind_instruction_str[0]
            instr_str = kind_instruction_str[1]
            funcs_dis['func2'].append(instructions.decode_instruction(word, kind = kind))
            funcs_instr_strs['func2'].append(instr_str)

        expected_str = ""
        for func_name, strs in funcs_instr_strs.items():
            expected_str += f"{' ' * print_offset}instructions from function: {func_name}\n"
            for ele in strs:
                expected_str += f"{' ' * (print_offset + 2)}{ele}\n"

        capsys.readouterr()
        instructions.print_instructions(funcs_dis, print_offset = print_offset)
        captured = capsys.readouterr()

        assert captured.out == expected_str

    words_kinds_instruction_strs = {
        0x0087a783 : [decoded_instruction.instruction_kind.rv32, "0x0087a783 lw x15, 8(x15)"],
        instructions.swizzle_instruction(0x5160000b) : [decoded_instruction.instruction_kind.ttbh, "0x4580002d setadcxy CntSetMask[23:21] = 0b11, Ch1_Y[20:15] = 0, Ch1_X[14:12] = 0, Ch0_Y[11:9] = 0, Ch0_X[8:6] = 0, BitMask[5:0] = 0b1011"]
    }

    test_list(words_kinds_instruction_strs)
    test_dict(words_kinds_instruction_strs)

def test_instruction_histogram_to_str():
    def test_dict0():
        arg: typing.Any = dict()
        for kind in decoded_instruction.instruction_kind:
            arg[kind] = dict()

        counter = 0
        for value in arg.values():
            for i in range(1,5):
                value[str(counter)] = [i, i / sum(range(1,5)), i / (sum(range(1,5)) * len(arg))]
                counter += 1

        hstr = instructions.instruction_histogram_to_str(arg, print_offset = 0)
        expected = textwrap.dedent("""\
            - Instructions for kind rv32
              - instruction: 3,  number of occurrences:    4, % (within kind): 40.00, % (overall): 10.00
              - instruction: 2,  number of occurrences:    3, % (within kind): 30.00, % (overall):  7.50
              - instruction: 1,  number of occurrences:    2, % (within kind): 20.00, % (overall):  5.00
              - instruction: 0,  number of occurrences:    1, % (within kind): 10.00, % (overall):  2.50
              - Instruction profile for kind rv32
                - number of instructions:                10, unique:    4
                - number of no_opcode instructions:       0, unique:    0
                - number of no_mnemonic instructions:     0, unique:    0
            - Instructions for kind ttwh
              - instruction: 7,  number of occurrences:    4, % (within kind): 40.00, % (overall): 10.00
              - instruction: 6,  number of occurrences:    3, % (within kind): 30.00, % (overall):  7.50
              - instruction: 5,  number of occurrences:    2, % (within kind): 20.00, % (overall):  5.00
              - instruction: 4,  number of occurrences:    1, % (within kind): 10.00, % (overall):  2.50
              - Instruction profile for kind ttwh
                - number of instructions:                10, unique:    4
                - number of no_opcode instructions:       0, unique:    0
                - number of no_mnemonic instructions:     0, unique:    0
            - Instructions for kind ttbh
              - instruction: 11, number of occurrences:    4, % (within kind): 40.00, % (overall): 10.00
              - instruction: 10, number of occurrences:    3, % (within kind): 30.00, % (overall):  7.50
              - instruction: 9,  number of occurrences:    2, % (within kind): 20.00, % (overall):  5.00
              - instruction: 8,  number of occurrences:    1, % (within kind): 10.00, % (overall):  2.50
              - Instruction profile for kind ttbh
                - number of instructions:                10, unique:    4
                - number of no_opcode instructions:       0, unique:    0
                - number of no_mnemonic instructions:     0, unique:    0
            - Instructions for kind ttqs
              - instruction: 15, number of occurrences:    4, % (within kind): 40.00, % (overall): 10.00
              - instruction: 14, number of occurrences:    3, % (within kind): 30.00, % (overall):  7.50
              - instruction: 13, number of occurrences:    2, % (within kind): 20.00, % (overall):  5.00
              - instruction: 12, number of occurrences:    1, % (within kind): 10.00, % (overall):  2.50
              - Instruction profile for kind ttqs
                - number of instructions:                10, unique:    4
                - number of no_opcode instructions:       0, unique:    0
                - number of no_mnemonic instructions:     0, unique:    0
            - Instruction profile:
              - number of instructions:                  40, unique:   16
              - number of no_opcode instructions:         0, unique:    0
              - number of no_mnemonic instructions:       0, unique:    0
            """)

        assert hstr == expected

    def test_dict1():
        arg: typing.Any = dict()
        for i in range(0, 2):
            arg["func" + str(i)] = dict()
            for kind in decoded_instruction.instruction_kind:
                arg["func" + str(i)][kind] = dict()

        counter = 0
        for kind_hist_dict in arg.values():
            for hist_dict in kind_hist_dict.values():
                for i in range(1,5):
                    hist_dict[str(counter)] = [i, i / sum(range(1,5)), i / (sum(range(1,5)) * len(arg))]
                    counter += 1

        hstr = instructions.instruction_histogram_to_str(arg, print_offset = 0)
        expected = textwrap.dedent("""\
        func0
        - Instructions for kind rv32
          - instruction: 3,  number of occurrences:    4, % (within kind): 40.00, % (overall): 20.00
          - instruction: 2,  number of occurrences:    3, % (within kind): 30.00, % (overall): 15.00
          - instruction: 1,  number of occurrences:    2, % (within kind): 20.00, % (overall): 10.00
          - instruction: 0,  number of occurrences:    1, % (within kind): 10.00, % (overall):  5.00
          - Instruction profile for kind rv32
            - number of instructions:                10, unique:    4
            - number of no_opcode instructions:       0, unique:    0
            - number of no_mnemonic instructions:     0, unique:    0
        - Instructions for kind ttwh
          - instruction: 7,  number of occurrences:    4, % (within kind): 40.00, % (overall): 20.00
          - instruction: 6,  number of occurrences:    3, % (within kind): 30.00, % (overall): 15.00
          - instruction: 5,  number of occurrences:    2, % (within kind): 20.00, % (overall): 10.00
          - instruction: 4,  number of occurrences:    1, % (within kind): 10.00, % (overall):  5.00
          - Instruction profile for kind ttwh
            - number of instructions:                10, unique:    4
            - number of no_opcode instructions:       0, unique:    0
            - number of no_mnemonic instructions:     0, unique:    0
        - Instructions for kind ttbh
          - instruction: 11, number of occurrences:    4, % (within kind): 40.00, % (overall): 20.00
          - instruction: 10, number of occurrences:    3, % (within kind): 30.00, % (overall): 15.00
          - instruction: 9,  number of occurrences:    2, % (within kind): 20.00, % (overall): 10.00
          - instruction: 8,  number of occurrences:    1, % (within kind): 10.00, % (overall):  5.00
          - Instruction profile for kind ttbh
            - number of instructions:                10, unique:    4
            - number of no_opcode instructions:       0, unique:    0
            - number of no_mnemonic instructions:     0, unique:    0
        - Instructions for kind ttqs
          - instruction: 15, number of occurrences:    4, % (within kind): 40.00, % (overall): 20.00
          - instruction: 14, number of occurrences:    3, % (within kind): 30.00, % (overall): 15.00
          - instruction: 13, number of occurrences:    2, % (within kind): 20.00, % (overall): 10.00
          - instruction: 12, number of occurrences:    1, % (within kind): 10.00, % (overall):  5.00
          - Instruction profile for kind ttqs
            - number of instructions:                10, unique:    4
            - number of no_opcode instructions:       0, unique:    0
            - number of no_mnemonic instructions:     0, unique:    0
        - Instruction profile:
          - number of instructions:                  40, unique:   16
          - number of no_opcode instructions:         0, unique:    0
          - number of no_mnemonic instructions:       0, unique:    0
        func1
        - Instructions for kind rv32
          - instruction: 19, number of occurrences:    4, % (within kind): 40.00, % (overall): 20.00
          - instruction: 18, number of occurrences:    3, % (within kind): 30.00, % (overall): 15.00
          - instruction: 17, number of occurrences:    2, % (within kind): 20.00, % (overall): 10.00
          - instruction: 16, number of occurrences:    1, % (within kind): 10.00, % (overall):  5.00
          - Instruction profile for kind rv32
            - number of instructions:                10, unique:    4
            - number of no_opcode instructions:       0, unique:    0
            - number of no_mnemonic instructions:     0, unique:    0
        - Instructions for kind ttwh
          - instruction: 23, number of occurrences:    4, % (within kind): 40.00, % (overall): 20.00
          - instruction: 22, number of occurrences:    3, % (within kind): 30.00, % (overall): 15.00
          - instruction: 21, number of occurrences:    2, % (within kind): 20.00, % (overall): 10.00
          - instruction: 20, number of occurrences:    1, % (within kind): 10.00, % (overall):  5.00
          - Instruction profile for kind ttwh
            - number of instructions:                10, unique:    4
            - number of no_opcode instructions:       0, unique:    0
            - number of no_mnemonic instructions:     0, unique:    0
        - Instructions for kind ttbh
          - instruction: 27, number of occurrences:    4, % (within kind): 40.00, % (overall): 20.00
          - instruction: 26, number of occurrences:    3, % (within kind): 30.00, % (overall): 15.00
          - instruction: 25, number of occurrences:    2, % (within kind): 20.00, % (overall): 10.00
          - instruction: 24, number of occurrences:    1, % (within kind): 10.00, % (overall):  5.00
          - Instruction profile for kind ttbh
            - number of instructions:                10, unique:    4
            - number of no_opcode instructions:       0, unique:    0
            - number of no_mnemonic instructions:     0, unique:    0
        - Instructions for kind ttqs
          - instruction: 31, number of occurrences:    4, % (within kind): 40.00, % (overall): 20.00
          - instruction: 30, number of occurrences:    3, % (within kind): 30.00, % (overall): 15.00
          - instruction: 29, number of occurrences:    2, % (within kind): 20.00, % (overall): 10.00
          - instruction: 28, number of occurrences:    1, % (within kind): 10.00, % (overall):  5.00
          - Instruction profile for kind ttqs
            - number of instructions:                10, unique:    4
            - number of no_opcode instructions:       0, unique:    0
            - number of no_mnemonic instructions:     0, unique:    0
        - Instruction profile:
          - number of instructions:                  40, unique:   16
          - number of no_opcode instructions:         0, unique:    0
          - number of no_mnemonic instructions:       0, unique:    0
        """)

        assert hstr == expected

    test_dict0()
    test_dict1()

def test_print_instruction_histogram(capsys):
    def test_dict0():
        arg: typing.Any = dict()
        for kind in decoded_instruction.instruction_kind:
            arg[kind] = dict()

        counter = 0
        for value in arg.values():
            for i in range(1,5):
                value[str(counter)] = [i, i / sum(range(1,5)), i / (sum(range(1,5)) * len(arg))]
                counter += 1

        capsys.readouterr()
        instructions.print_instruction_histogram(arg)
        captured = capsys.readouterr()
        assert captured.out == instructions.instruction_histogram_to_str(arg)

    def test_dict1():
        arg: typing.Any = dict()
        for i in range(0, 2):
            arg["func" + str(i)] = dict()
            for kind in decoded_instruction.instruction_kind:
                arg["func" + str(i)][kind] = dict()

        counter = 0
        for kind_hist_dict in arg.values():
            for hist_dict in kind_hist_dict.values():
                for i in range(1,5):
                    hist_dict[str(counter)] = [i, i / sum(range(1,5)), i / (sum(range(1,5)) * len(arg))]
                    counter += 1

        capsys.readouterr()
        instructions.print_instruction_histogram(arg)
        captured = capsys.readouterr()
        assert captured.out == instructions.instruction_histogram_to_str(arg)

    test_dict0()
    test_dict1()

def test_instruction_kind_histogram_to_str_0():
    arg: typing.Any = dict()
    for kind in decoded_instruction.instruction_kind:
        arg[kind] = dict()

    counter = 0
    for value in arg.values():
        for i in range(1,5):
            value[str(counter)] = i
            counter += 1

    expected = textwrap.dedent("""\
        - Number of instructions:             40
          - Number of rv32 instructions:      10
            - Number of 0 instructions:        1
            - Number of 1 instructions:        2
            - Number of 2 instructions:        3
            - Number of 3 instructions:        4
          - Number of ttwh instructions:      10
            - Number of 4 instructions:        1
            - Number of 5 instructions:        2
            - Number of 6 instructions:        3
            - Number of 7 instructions:        4
          - Number of ttbh instructions:      10
            - Number of 8 instructions:        1
            - Number of 9 instructions:        2
            - Number of 10 instructions:       3
            - Number of 11 instructions:       4
          - Number of ttqs instructions:      10
            - Number of 12 instructions:       1
            - Number of 13 instructions:       2
            - Number of 14 instructions:       3
            - Number of 15 instructions:       4
    """)
    assert expected == instructions.instruction_kind_histogram_to_str(arg, print_offset = 0)

def test_instruction_kind_histogram_to_str_1(capsys):
    arg: typing.Any = dict()
    for i in range(0, 2):
        arg["func" + str(i)] = dict()
        for kind in decoded_instruction.instruction_kind:
            arg["func" + str(i)][kind] = dict()

    counter = 0
    for kind_hist_dict in arg.values():
        for hist_dict in kind_hist_dict.values():
            for i in range(1,5):
                hist_dict[str(counter)] = i
                counter += 1

    capsys.readouterr()
    instructions.print_instruction_kind_histogram(arg)
    captured = capsys.readouterr()
    assert captured.out == instructions.instruction_kind_histogram_to_str(arg)

def test_get_coverage_0():
    dec_ins = []
    counter = 0
    for idx0, kind in enumerate(decoded_instruction.instruction_kind):
        for idx1 in range(0, max(idx0, 1)):
            dec_in = decoded_instruction.decoded_instruction()
            dec_in.set_mnemonic(str(counter))
            dec_in.set_kind(kind)
            dec_in.set_opcode(0)
            dec_ins.append(dec_in)
            counter += 1

    instruction_sets: typing.Any = dict()
    for idx0, kind in enumerate(decoded_instruction.instruction_kind):
        instruction_sets[kind] = dict()
        counter = 0
        for idx1 in range(0, 5):
            instruction_sets[kind][str(counter)] = dict()
            counter += 1

    expected = dict()
    for kind in decoded_instruction.instruction_kind:
        expected[kind] = sum([dec_in.kind == kind for dec_in in dec_ins]) / len(instruction_sets[kind])

    cov = instructions.get_coverage(dec_ins, sets = instruction_sets)
    assert expected == cov

def test_get_coverage_1():
    # flatten_dict
    func_dec_ins: typing.Any = dict()
    counter = 0
    for i in range(2):
        func_dec_ins[i] = list()
        for idx0, kind in enumerate(decoded_instruction.instruction_kind):
            for idx1 in range(0, max(idx0, 1)):
                dec_in = decoded_instruction.decoded_instruction()
                dec_in.set_mnemonic(str(counter))
                dec_in.set_kind(kind)
                dec_in.set_opcode(0)
                func_dec_ins[i].append(dec_in)
                counter += 1
    num_ins = counter

    instruction_sets: typing.Any = dict()
    for idx0, kind in enumerate(decoded_instruction.instruction_kind):
        instruction_sets[kind] = dict()
        counter = 0
        for idx1 in range(0, num_ins):
            instruction_sets[kind][str(counter)] = dict()
            counter += 1

    print(instruction_sets)

    expected = dict()
    for kind in decoded_instruction.instruction_kind:
        num = 0
        for dec_ins in func_dec_ins.values():
            for dec_in in dec_ins:
                if kind == dec_in.kind:
                    num += 1

        expected[kind] = num/len(instruction_sets[kind])

    cov = instructions.get_coverage(func_dec_ins, sets = instruction_sets, flatten_dict = True)
    assert expected == cov

def test_get_coverage_2():
    func_dec_ins = dict()
    counter = 0
    for i in range(2):
        dec_ins = list()
        for idx0, kind in enumerate(decoded_instruction.instruction_kind):
            for idx1 in range(0, max(idx0, 1)):
                dec_in = decoded_instruction.decoded_instruction()
                dec_in.set_mnemonic(str(counter))
                dec_in.set_kind(kind)
                dec_in.set_opcode(0)
                dec_ins.append(dec_in)
                counter += 1
        func_dec_ins['func' + str(i)] = dec_ins
    num_ins = counter

    instruction_sets: typing.Any = dict()
    for idx0, kind in enumerate(decoded_instruction.instruction_kind):
        instruction_sets[kind] = dict()
        counter = 0
        for idx1 in range(0, num_ins):
            instruction_sets[kind][str(counter)] = dict()
            counter += 1

    expected: typing.Any = dict()
    for key in func_dec_ins.keys():
        expected[key] = dict()

    for func in expected.keys():
        for kind in decoded_instruction.instruction_kind:
            num = 0
            for dec_in in func_dec_ins[func]:
                    if kind == dec_in.kind:
                        num += 1
            expected[func][kind] = num/len(instruction_sets[kind])

    cov = instructions.get_coverage(func_dec_ins, sets = instruction_sets, flatten_dict = False)
    assert expected == cov

@pytest.mark.slow
def test_instruction_to_str():
    def test_list():
        kinds = [kind for kind in decoded_instruction.instruction_kind]
        modules = instructions.get_modules(kinds = kinds, modules = None)
        instruction_sets = instructions.get_instruction_sets(kinds = kinds, modules = modules, instruction_sets = None)
        num_instructions_per_kind = 6
        dec_ins: typing.Any = list()
        mnemonics_kinds_words: typing.Any = dict()
        for kind in decoded_instruction.instruction_kind:
            num_instrs = 0
            for mnemonic, info in instruction_sets[kind].items():
                if sum([mnemonic == ele.mnemonic for ele in dec_ins]):
                    continue
                else:
                    if kind.is_tensix():
                        mnemonics_kinds_words[mnemonic] = (kind, decoded_instruction.swizzle_instruction(decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32)))
                    else:
                        mnemonics_kinds_words[mnemonic] = [kind, info["opcode"]]
                    dec_ins.append(instructions.decode_instruction(mnemonics_kinds_words[mnemonic][1], kind = kind, module = modules[kind], instruction_set = instruction_sets))
                    num_instrs += 1

                    if num_instrs > num_instructions_per_kind:
                        break

        msgs = instructions.instruction_to_str(dec_ins, module = modules, instruction_set = instruction_sets)
        assert msgs == instructions.instruction_to_str(dec_ins)
        for idx, dec_in in enumerate(dec_ins):
            assert hasattr(dec_in, 'kind')
            assert hasattr(dec_in, 'mnemonic')
            assert hasattr(dec_in, 'word')
            assert dec_in.kind == mnemonics_kinds_words[dec_in.mnemonic][0]
            assert dec_in.word == mnemonics_kinds_words[dec_in.mnemonic][1]
            assert dec_in.mnemonic.lower() in msgs[idx]
            assert msgs[idx].startswith(f"0x{dec_in.word:08x} {dec_in.mnemonic.lower()}")
            if dec_in.kind.is_tensix():
                if instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"]:
                        assert all(arg["name"] in msgs[idx] for arg in instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"])
            else:
                assert msgs[idx] == modules[dec_in.kind].to_assembly(dec_in)

    def test_dict():
        num_keys = 2
        kinds = [kind for kind in decoded_instruction.instruction_kind]
        modules = instructions.get_modules(kinds = kinds, modules = None)
        instruction_sets = instructions.get_instruction_sets(kinds = kinds, modules = modules, instruction_sets = None)
        num_instructions_per_kind = 6
        funcs_dec_ins = dict()
        funcs_mnemonics_kinds_words = dict()
        for key_id in range(num_keys):
            func_name = f"func_{key_id}"
            mnemonics_kinds_words: typing.Any = dict()
            dec_ins: typing.Any = list()
            for kind in decoded_instruction.instruction_kind:
                num_instrs = 0
                for mnemonic, info in instruction_sets[kind].items():
                    if sum([mnemonic == ele.mnemonic for ele in dec_ins]):
                        continue
                    else:
                        if kind.is_tensix():
                            mnemonics_kinds_words[mnemonic] = [kind, decoded_instruction.swizzle_instruction(decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32))]
                        else:
                            mnemonics_kinds_words[mnemonic] = [kind, info["opcode"]]
                        dec_ins.append(instructions.decode_instruction(mnemonics_kinds_words[mnemonic][1], kind = kind, module = modules[kind], instruction_set = instruction_sets))
                        num_instrs += 1

                        if num_instrs > num_instructions_per_kind:
                            break

            funcs_mnemonics_kinds_words[func_name] = mnemonics_kinds_words
            funcs_dec_ins[func_name] = dec_ins

        funcs_msgs = instructions.instruction_to_str(funcs_dec_ins, module = modules, instruction_set = instruction_sets)
        assert funcs_msgs == instructions.instruction_to_str(funcs_dec_ins)

        if isinstance(funcs_msgs, dict):
            for func_name, dec_ins in funcs_dec_ins.items():
                for idx, dec_in in enumerate(dec_ins):
                    assert hasattr(dec_in, 'kind')
                    assert hasattr(dec_in, 'mnemonic')
                    assert hasattr(dec_in, 'word')
                    assert dec_in.kind == mnemonics_kinds_words[dec_in.mnemonic][0]
                    assert dec_in.word == mnemonics_kinds_words[dec_in.mnemonic][1]

                    assert isinstance(funcs_msgs[func_name][idx], str)
                    assert isinstance(dec_in, decoded_instruction.decoded_instruction)
                    assert dec_in.mnemonic.lower() in funcs_msgs[func_name][idx]
                    assert funcs_msgs[func_name][idx].startswith(f"0x{dec_in.word:08x} {dec_in.mnemonic.lower()}")
                    if dec_in.kind.is_tensix():
                        if instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"]:
                                assert all(arg["name"] in funcs_msgs[func_name][idx] for arg in instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"])
                    else:
                        assert funcs_msgs[func_name][idx] == modules[dec_in.kind].to_assembly(dec_in)

    def test_flatten_dict():
        num_keys = 2
        kinds = [kind for kind in decoded_instruction.instruction_kind]
        modules = instructions.get_modules(kinds = kinds, modules = None)
        instruction_sets = instructions.get_instruction_sets(kinds = kinds, modules = modules, instruction_sets = None)
        num_instructions_per_kind = 6
        funcs_dec_ins = dict()
        funcs_mnemonics_kinds_words = dict()
        for key_id in range(num_keys):
            func_name = f"func_{key_id}"
            mnemonics_kinds_words: typing.Any = dict()
            dec_ins: typing.Any = list()
            for kind in decoded_instruction.instruction_kind:
                num_instrs = 0
                for mnemonic, info in instruction_sets[kind].items():
                    if sum([mnemonic == ele.mnemonic for ele in dec_ins]):
                        continue
                    else:
                        if kind.is_tensix():
                            mnemonics_kinds_words[mnemonic] = (kind, decoded_instruction.swizzle_instruction(decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32)))
                        else:
                            mnemonics_kinds_words[mnemonic] = (kind, info["opcode"])
                        dec_ins.append(instructions.decode_instruction(mnemonics_kinds_words[mnemonic][1], kind = kind, module = modules[kind], instruction_set = instruction_sets))
                        num_instrs += 1

                        if num_instrs > num_instructions_per_kind:
                            break

            funcs_mnemonics_kinds_words[func_name] = mnemonics_kinds_words
            funcs_dec_ins[func_name] = dec_ins

        funcs_msgs = instructions.instruction_to_str(funcs_dec_ins, module = modules, instruction_set = instruction_sets, flatten_dict = True)
        assert funcs_msgs == instructions.instruction_to_str(funcs_dec_ins, flatten_dict = True)
        counter = 0
        for func_name, dec_ins in funcs_dec_ins.items():
            for dec_in in dec_ins:
                assert hasattr(dec_in, 'kind')
                assert hasattr(dec_in, 'mnemonic')
                assert hasattr(dec_in, 'word')
                assert dec_in.kind == mnemonics_kinds_words[dec_in.mnemonic][0]
                assert dec_in.word == mnemonics_kinds_words[dec_in.mnemonic][1]
                assert dec_in.mnemonic.lower() in funcs_msgs[counter]
                assert funcs_msgs[counter].startswith(f"0x{dec_in.word:08x} {dec_in.mnemonic.lower()}")
                if dec_in.kind.is_tensix():
                    if instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"]:
                        assert all(arg["name"] in funcs_msgs[counter] for arg in instruction_sets[dec_in.kind][dec_in.mnemonic]["arguments"])
                else:
                    assert funcs_msgs[counter] == modules[dec_in.kind].to_assembly(dec_in)

                counter += 1

    test_list()
    test_dict()
    test_flatten_dict()