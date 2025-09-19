#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import typing

import pytest

import ttsim.front.llk.decoded_instruction as decoded_instruction


def test_tensix_instruction_kind():
    def test_kind():
        for kind in decoded_instruction.instruction_kind:
            if decoded_instruction.instruction_kind.rv32 == kind:
                assert not kind.is_tensix()
            else:
                assert kind.is_tensix()

        with pytest.raises(AttributeError) as exc_info:
            # decoded_instruction.instruction_kind.__invalid_instruction_kind__.is_tensix()
            getattr(decoded_instruction.instruction_kind, "__invalid_instruction_kind__").is_tensix()
        assert "has no attribute" in str(exc_info.value)

        with pytest.raises(AttributeError) as exc_info:
            decoded_instruction.instruction_kind.__invalid_instruction_kind__.is_tensix() # type: ignore[attr-defined]
        assert "has no attribute" in str(exc_info.value)

    def test_lt():
        kinds = [kind for kind in decoded_instruction.instruction_kind]

        for idx1 in range(len(kinds)):
            for idx2 in range(idx1 + 1, len(kinds)):
                kind1 = kinds[idx1]
                kind2 = kinds[idx2]
                assert kind1 < kind2

    test_kind()
    test_lt()

def test_registers_to_list_of_ints():
    def test_ints():
        ints = [0, -10]
        for ele in ints:
            int_list = decoded_instruction.registers.to_list_of_ints(ele)
            assert isinstance(int_list, list)
            assert 1 == len(int_list)
            assert ele == int_list[0]

    def test_int_list_tuple(int_elems: list[int] | tuple[int,...]):
        int_list = decoded_instruction.registers.to_list_of_ints(int_elems)
        assert isinstance(int_list, list)
        assert len(int_elems) == len(int_list)
        for idx in range(len(int_elems)):
            assert int_elems[idx] == int_list[idx]

    def test_floats():
        floats = [0., 2.0, -5.]
        for ele in floats:
            int_list = decoded_instruction.registers.to_list_of_ints(ele)
            assert int_list is None

    def test_dict():
        m_dict = {0 : 0, 1 : 1}
        int_list = decoded_instruction.registers.to_list_of_ints(m_dict.keys())
        assert int_list is None
        int_list = decoded_instruction.registers.to_list_of_ints(m_dict.values())
        assert int_list is None

    test_ints()
    test_int_list_tuple([-1,2,1])
    test_int_list_tuple((-0,-10,6,-5))
    test_floats()
    test_dict()

def test_registers_to_list_of_non_negative_ints():
    def test_ints():
        ints = [0, -10]
        for ele in ints:
            if ele >= 0:
                int_list = decoded_instruction.registers.to_list_of_non_negative_ints(ele)
                assert isinstance(int_list, list)
                assert 1 == len(int_list)
                assert ele == int_list[0]
            else:
                int_list = decoded_instruction.registers.to_list_of_non_negative_ints(ele)
                assert int_list is None

    def test_list_tuple(int_elems: list[int] | tuple[int, ...]):
        if all((ele >= 0) and isinstance(ele, int) for ele in int_elems) and int_elems and isinstance(int_elems, (list, tuple)):
            int_list = decoded_instruction.registers.to_list_of_non_negative_ints(int_elems)
            assert isinstance(int_list, list)
            assert len(int_elems) == len(int_list)
            for idx in range(len(int_elems)):
                assert int_elems[idx] == int_list[idx]
        else:
            int_list = decoded_instruction.registers.to_list_of_non_negative_ints(int_elems)
            assert int_list is None

    def test_floats():
        floats = [0., 2.0, -5.]
        for ele in floats:
            int_list = decoded_instruction.registers.to_list_of_non_negative_ints(ele)
            assert int_list is None

    def test_dict():
        m_dict = {0 : 0, 1 : 1}
        int_list = decoded_instruction.registers.to_list_of_non_negative_ints(m_dict.keys())
        assert int_list is None
        int_list = decoded_instruction.registers.to_list_of_non_negative_ints(m_dict.values())
        assert int_list is None

    test_ints()
    test_list_tuple([-1,2,1])
    test_list_tuple((-0,-10,6,-5))
    test_list_tuple((0,))
    test_list_tuple((1,2))
    test_list_tuple([5])
    test_list_tuple([7,6])
    test_floats()
    test_dict()

def test_class_registers(capsys):
    def test_init():
        reg = decoded_instruction.registers()
        assert isinstance(reg, decoded_instruction.registers)
        assert not hasattr(reg, 'integers')
        assert not hasattr(reg, 'floats')

    def test_integers_int():
        ints: list[typing.Any] = [-0, 0, 5, 1, 0., 1., -5.1] # -0 is not considered positive.
        for ele in ints:
            reg = decoded_instruction.registers()
            reg.set_integers(ele)
            if (ele >= 0) and isinstance(ele, int):
                assert hasattr(reg, 'integers')
                assert not hasattr(reg, 'floats')
                assert isinstance(reg.integers, list)
                assert 1 == len(reg.integers)
                assert ele == reg.integers[0]
            else:
                assert not hasattr(reg, 'integers')
                assert not hasattr(reg, 'floats')

    def test_integers(regs):
        reg = decoded_instruction.registers()
        reg.set_integers(typing.cast(typing.Any, regs))

        if regs and all((ele >= 0) and isinstance(ele, int) for ele in regs) and isinstance(regs, (list, tuple)):
            assert hasattr(reg, 'integers')
            assert not hasattr(reg, 'floats')
            assert isinstance(reg.integers, list)
            assert len(regs) == len(reg.integers)
            for idx in range(len(regs)):
                assert regs[idx] == reg.integers[idx]
        else:
            assert not hasattr(reg, 'integers')
            assert not hasattr(reg, 'floats')

    def test_floats_int():
        ints: list[typing.Any] = [-0, 0, 5, 1, 0., 1., -5.1] # -0 is not considered positive.
        for ele in ints:
            reg = decoded_instruction.registers()
            reg.set_floats(ele)
            if (ele >= 0) and isinstance(ele, int):
                assert not hasattr(reg, 'integers')
                assert hasattr(reg, 'floats')
                assert isinstance(reg.floats, list)
                assert 1 == len(reg.floats)
                assert ele == reg.floats[0]
            else:
                assert not hasattr(reg, 'integers')
                assert not hasattr(reg, 'floats')

    def test_floats(regs):
        reg = decoded_instruction.registers()
        reg.set_floats(regs)

        if regs and all((ele >= 0) and isinstance(ele, int) for ele in regs) and isinstance(regs, (list, tuple)):
            assert not hasattr(reg, 'integers')
            assert hasattr(reg, 'floats')
            assert isinstance(reg.floats, list)
            assert len(regs) == len(reg.floats)
            for idx in range(len(regs)):
                assert regs[idx] == reg.floats[idx]
        else:
            assert not hasattr(reg, 'integers')
            assert not hasattr(reg, 'floats')

    def test___str__():
        reg = decoded_instruction.registers()
        assert "" == f"{reg}"
        reg.set_integers([0])
        assert "integers" in f"{reg}"
        assert "floats" not in f"{reg}"
        reg.set_floats([0])
        assert "floats" in f"{reg}"

    test_init()
    test_integers_int()
    test_integers([0,1])
    test_integers((2,3,4))
    test_integers([-1,-2])
    test_integers([1.0, 2.0])
    test_floats_int()
    test_floats([0,1])
    test_floats((2,3,4))
    test_floats([-1,-2])
    test_floats([1.0, 2.0])
    test___str__()

def test_registers_eq_operator():
    def test_non_register_comparison():
        a: list[typing.Any] = []
        assert a != decoded_instruction.registers()
        assert decoded_instruction.registers() != a

        class registers:
            def __init__(self):
                pass

        assert registers() != decoded_instruction.registers()

    def test_initised_registers():
        assert decoded_instruction.registers() == decoded_instruction.registers()

    def test_integers():
        regs0 = decoded_instruction.registers()
        regs0.set_integers([0])
        regs1 = decoded_instruction.registers()
        regs1.set_integers([0])

        assert (regs0 == regs1)
        assert (regs1 == regs0)

        regs1.set_integers([0,1])
        assert (regs0 != regs1)
        assert (regs1 != regs0)

    def test_floats():
        regs0 = decoded_instruction.registers()
        regs0.set_floats([0])
        regs1 = decoded_instruction.registers()
        regs1.set_floats([0])

        assert (regs0 == regs1)
        assert (regs1 == regs0)

        regs1.set_floats([0,1])
        assert (regs0 != regs1)
        assert (regs1 != regs0)

    def test_integers_floats():
        regs0 = decoded_instruction.registers()
        regs0.set_integers([0])

        regs1 = decoded_instruction.registers()
        regs1.set_floats([0])

        assert (regs0 != regs1)
        assert (regs1 != regs0)

        regs1.set_integers([0])
        regs0.set_floats([1])
        regs1.set_floats([1])

        assert (regs0 == regs1)
        assert (regs1 == regs0)

        regs1.set_integers([1,2])
        assert (regs0 != regs1)
        assert (regs1 != regs0)

        regs1.set_integers([0])
        regs1.set_floats([1,2,3])
        assert (regs0 != regs1)
        assert (regs1 != regs0)

    test_non_register_comparison()
    test_initised_registers()
    test_integers()
    test_floats()
    test_integers_floats()

def test_class_operands(capsys):
    def test_init():
        op = decoded_instruction.operands()
        assert isinstance(op, decoded_instruction.operands)
        assert not hasattr(op, 'all')
        assert not hasattr(op, 'sources')
        assert not hasattr(op, 'destinations')
        assert not hasattr(op, 'immediates')
        assert not hasattr(op, 'attributes')

    def test_all(capsys):
        dict_list: list[typing.Any] = [
            dict({"a" : 0, "b" : +1}),
            dict({"a" : 0, "b" : -1}),
            dict({"a" : 0, "b" : 1.}),
            dict({1 : 0, 2: 0}),
            dict()
            ]

        for elem in dict_list:
            op = decoded_instruction.operands()
            op.set_all(elem)
            assert not hasattr(op, 'sources')
            assert not hasattr(op, 'destinations')
            assert not hasattr(op, 'immediates')
            assert not hasattr(op, 'attributes')

            if elem and isinstance(elem, dict) and all(isinstance(key, str) for key in elem.keys()) and all(isinstance(value, int) for value in elem.values()):
                assert hasattr(op, 'all')

                attr_all = op.all
                assert 0 != len(attr_all)
                assert isinstance(attr_all, dict)
                assert all(isinstance(key, str) for key in attr_all.keys())
                assert all(isinstance(value, int) for value in attr_all.values())
            else:
                assert not hasattr(op, 'all')
                capsys.readouterr()
                op.set_all(elem, mode = "v")
                capture = capsys.readouterr()
                assert not hasattr(op, 'all')
                assert capture.out.startswith("- WARNING")

    def test_sources(capsys):
        def test_empty_sources(capsys):
            reg = decoded_instruction.registers()
            op  = decoded_instruction.operands()
            op.set_sources(reg)
            assert not hasattr(op, 'sources')
            capsys.readouterr()
            op.set_sources(reg, mode = "v")
            capture = capsys.readouterr()
            assert not hasattr(op, 'sources')
            assert capture.out.startswith("- WARNING")

        def test_sources_with_integers():
            reg = decoded_instruction.registers()
            reg.set_integers([0,1])
            op  = decoded_instruction.operands()
            op.set_sources(reg)
            assert hasattr(op, 'sources')
            assert hasattr(op.sources, 'integers')
            assert not hasattr(op.sources, 'floats')

        def test_sources_with_floats():
            reg = decoded_instruction.registers()
            reg.set_floats([0,1])
            op  = decoded_instruction.operands()
            op.set_sources(reg)
            assert hasattr(op, 'sources')
            assert not hasattr(op.sources, 'integers')
            assert hasattr(op.sources, 'floats')

        def test_sources_with_integers_and_floats():
            reg = decoded_instruction.registers()
            reg.set_integers([2,3])
            reg.set_floats([0,1])
            op = decoded_instruction.operands()
            op.set_sources(reg)
            assert hasattr(op, 'sources')
            assert hasattr(op.sources, 'integers')
            assert hasattr(op.sources, 'floats')

        test_empty_sources(capsys)
        test_sources_with_integers()
        test_sources_with_floats()
        test_sources_with_integers_and_floats()

    def test_destinations(capsys):
        def test_empty_destinations(capsys):
            reg = decoded_instruction.registers()
            op  = decoded_instruction.operands()
            op.set_destinations(reg)
            assert not hasattr(op, 'destinations')
            capsys.readouterr()
            op.set_destinations(reg, mode = "v")
            capture = capsys.readouterr()
            assert not hasattr(op, 'destinations')
            assert capture.out.startswith("- WARNING")

        def test_destinations_with_integers():
            reg = decoded_instruction.registers()
            reg.set_integers([0,1])
            op  = decoded_instruction.operands()
            op.set_destinations(reg)
            assert hasattr(op, 'destinations')

        def test_destinations_with_floats():
            reg = decoded_instruction.registers()
            reg.set_floats([0,1])
            op  = decoded_instruction.operands()
            op.set_destinations(reg)
            assert hasattr(op, 'destinations')

        test_empty_destinations(capsys)
        test_destinations_with_integers()
        test_destinations_with_floats()

    def test_integer_sources(capsys):
        def test_regs():
            regs = [0,1]
            op = decoded_instruction.operands()
            assert not hasattr(op, 'sources')
            op.set_integer_sources(regs)
            assert hasattr(op, 'sources')
            assert hasattr(op.sources, 'integers')
            assert not hasattr(op, 'all')
            assert not hasattr(op, 'destinations')
            assert not hasattr(op, 'immediates')
            assert not hasattr(op, 'attributes')

        def test_rewrite():
            regs = [2,3]
            src = decoded_instruction.registers()
            src.set_integers(regs)

            op = decoded_instruction.operands()
            op.set_sources(src)

            assert hasattr(op, 'sources')
            assert hasattr(op.sources, 'integers')
            assert op.sources.integers == regs

            regs = [0,1]
            op.set_integer_sources(regs)
            assert hasattr(op, 'sources')
            assert hasattr(op.sources, 'integers')
            assert op.sources.integers == regs

        def test_mode_v(capsys):
            regs = [-1]
            op = decoded_instruction.operands()
            assert not hasattr(op, 'sources')
            capsys.readouterr()
            op.set_integer_sources(regs, mode = "v")
            capture = capsys.readouterr()
            assert not hasattr(op, 'sources')
            assert capture.out.startswith("- WARNING")

        test_regs()
        test_rewrite()
        test_mode_v(capsys)

    def test_float_sources(capsys):
        def test_regs():
            regs = [0,1]
            op = decoded_instruction.operands()
            assert not hasattr(op, 'sources')
            op.set_float_sources(regs)
            assert hasattr(op, 'sources')
            assert hasattr(op.sources, 'floats')
            assert not hasattr(op, 'all')
            assert not hasattr(op, 'destinations')
            assert not hasattr(op, 'immediates')
            assert not hasattr(op, 'attributes')

        def test_rewrite():
            regs = [2,3]
            src = decoded_instruction.registers()
            src.set_floats(regs)

            op = decoded_instruction.operands()
            op.set_sources(src)

            assert hasattr(op, 'sources')
            assert hasattr(op.sources, 'floats')
            assert op.sources.floats == regs

            regs = [0,1]
            op.set_float_sources(regs)
            assert hasattr(op, 'sources')
            assert hasattr(op.sources, 'floats')
            assert op.sources.floats == regs

        def test_mode_v(capsys):
            regs = [-1]
            op = decoded_instruction.operands()
            assert not hasattr(op, 'sources')
            capsys.readouterr()
            op.set_float_sources(regs, mode = "v")
            capture = capsys.readouterr()
            assert not hasattr(op, 'sources')
            assert capture.out.startswith("- WARNING")

        test_regs()
        test_rewrite()
        test_mode_v(capsys)

    def test_integer_destinations(capsys):
        def test_regs():
            regs = [0,1]
            op = decoded_instruction.operands()
            assert not hasattr(op, 'destinations')
            op.set_integer_destinations(regs)
            assert hasattr(op, 'destinations')
            assert hasattr(op.destinations, 'integers')
            assert not hasattr(op, 'all')
            assert not hasattr(op, 'sources')
            assert not hasattr(op, 'immediates')
            assert not hasattr(op, 'attributes')

        def test_rewrite():
            regs = [2,3]
            dst = decoded_instruction.registers()
            dst.set_integers(regs)

            op = decoded_instruction.operands()
            op.set_destinations(dst)

            assert hasattr(op, 'destinations')
            assert hasattr(op.destinations, 'integers')
            assert op.destinations.integers == regs

            regs = [0,1]
            op.set_integer_destinations(regs)
            assert hasattr(op, 'destinations')
            assert hasattr(op.destinations, 'integers')
            assert op.destinations.integers == regs

        def test_mode_v(capsys):
            regs = [-1]
            op = decoded_instruction.operands()
            assert not hasattr(op, 'destinations')
            capsys.readouterr()
            op.set_integer_destinations(regs, mode = "v")
            capture = capsys.readouterr()
            assert not hasattr(op, 'destinations')
            assert capture.out.startswith("- WARNING")

        test_regs()
        test_rewrite()
        test_mode_v(capsys)

    def test_float_destinations(capsys):
        def test_regs():
            regs = [0,1]
            op = decoded_instruction.operands()
            assert not hasattr(op, 'destinations')
            op.set_float_destinations(regs)
            assert hasattr(op, 'destinations')
            assert hasattr(op.destinations, 'floats')
            assert not hasattr(op, 'all')
            assert not hasattr(op, 'sources')
            assert not hasattr(op, 'immediates')
            assert not hasattr(op, 'attributes')

        def test_rewrite():
            regs = [2,3]
            dst = decoded_instruction.registers()
            dst.set_floats(regs)

            op = decoded_instruction.operands()
            op.set_destinations(dst)

            assert hasattr(op, 'destinations')
            assert hasattr(op.destinations, 'floats')
            assert op.destinations.floats == regs

            regs = [0,1]
            op.set_float_destinations(regs)
            assert hasattr(op, 'destinations')
            assert hasattr(op.destinations, 'floats')
            assert op.destinations.floats == regs

        def test_mode_v(capsys):
            regs = [-1]
            op = decoded_instruction.operands()
            assert not hasattr(op, 'destinations')
            capsys.readouterr()
            op.set_float_destinations(regs, mode = "v")
            capture = capsys.readouterr()
            assert not hasattr(op, 'destinations')
            assert capture.out.startswith("- WARNING")

        test_regs()
        test_rewrite()
        test_mode_v(capsys)

    def test_immediates(capsys):
        imms: list[typing.Any] = [0, 1, -5, -0, 1., 0., -5., 'a', {0 : "a", 1 : "b"}, [], [0,1], [-1,2], (2,3,4), (-2,-4), [1, "a"]]
        for ele in imms:
            flag_check_imm: bool = False
            op = decoded_instruction.operands()
            op.set_immediates(ele)
            flag_check_imm = isinstance(ele, int) or (isinstance(ele, (list, tuple)) and (0 != len(ele)) and all(isinstance(e, int) for e in ele))

            if flag_check_imm:
                assert hasattr(op, 'immediates')
                assert isinstance(op.immediates, list)
                assert len(op.immediates)
                assert all(isinstance(e, int) for e in op.immediates)
            else:
                assert not hasattr(op, 'immediates')
                capsys.readouterr()
                op.set_immediates(ele, mode = "v")
                capture = capsys.readouterr()
                assert not hasattr(op, 'immediates')
                assert capture.out.startswith("- WARNING")

    def test_attributes(capsys):
        attributes: list[typing.Any] = [0, 1, -1, 1., -0, {"a" : 0, "b" : ["c", "d"]}, {}, [], [0,1], [-1,2], [1, "a"], (0,1)]
        for ele in attributes:
            op = decoded_instruction.operands()
            op.set_attributes(ele)

            is_correct_attribute_type = True
            is_correct_attribute_type = is_correct_attribute_type and (isinstance(ele, dict) and (0 != len(ele)))
            is_correct_attribute_type = is_correct_attribute_type and (all(isinstance(key, str) for key in ele.keys()))
            is_correct_attribute_type = is_correct_attribute_type and (all(isinstance(value, (int, list)) for value in ele.values()))
            is_correct_attribute_type = is_correct_attribute_type and (all(all(isinstance(e, str) for e in value) for value in ele.values() if isinstance(value, list)))

            if is_correct_attribute_type:
                assert hasattr(op, 'attributes')
                assert isinstance(op.attributes, dict) and (0 != len(op.attributes))
                assert all(isinstance(key, str) for key in ele.keys())
                assert all(isinstance(value, (int, list)) for value in ele.values())
                assert all(all(isinstance(e, str) for e in value) for value in ele.values() if isinstance(value, list))
            else:
                assert not hasattr(op, 'attributes')
                capsys.readouterr()
                op.set_attributes(ele, mode = "v")
                capture = capsys.readouterr()
                assert not hasattr(op, 'attributes')
                assert capture.out.startswith("- WARNING")

            assert not hasattr(op, 'all')
            assert not hasattr(op, 'sources')
            assert not hasattr(op, 'destinations')
            assert not hasattr(op, 'immediates')

    def test_str():
        op = decoded_instruction.operands()
        assert 0 == len(f"{op}")
        regs = decoded_instruction.registers()
        regs.set_integers([0,1])
        regs.set_floats([2,3])
        op.set_sources(regs)
        regs.set_integers([4])
        regs.set_floats([5])
        op.set_destinations(regs)
        op.set_all({"a" : 1, "b" : 0})
        op.set_immediates([-1,2])
        op.set_attributes({"c" : 0, "d" : 1})
        msg = op.__str__()
        words = {"operands", "all", "sources", "destinations", "immediates", "attributes", "integer sources", "float sources", "integer destinations", "float destinations"}
        for word in words:
            assert word in msg

    def test_eq_operator():
        op1 = decoded_instruction.operands()
        op2 = decoded_instruction.operands()

        regs = decoded_instruction.registers()
        regs.set_integers([1,2])
        regs.set_floats([4,3])

        assert op1 != decoded_instruction.registers()
        assert op2 != decoded_instruction.registers()
        assert op1 == op2
        assert op2 == op1

        op1.set_all({'a' : 0, 'b' : 5})
        assert hasattr(op1, 'all')
        assert op1 != op2
        assert op2 != op1

        op2.set_all(op1.all)
        assert hasattr(op2, 'all')
        assert op1 == op2
        assert op2 == op1

        op1.set_sources(regs)
        assert hasattr(op1, 'sources')
        assert op1 != op2
        assert op2 != op1

        op2.set_sources(op1.sources)
        assert hasattr(op2, 'sources')
        assert op1 == op2
        assert op2 == op1

        op1.set_destinations(regs)
        assert hasattr(op1, 'destinations')
        assert op1 != op2
        assert op2 != op1

        op2.set_destinations(op1.destinations)
        assert hasattr(op2, 'destinations')
        assert op1 == op2
        assert op2 == op1

        op1.set_immediates([10, -10])
        assert hasattr(op1, 'immediates')
        assert op1 != op2
        assert op2 != op1

        op2.set_immediates(op1.immediates)
        assert hasattr(op2, 'immediates')
        assert op1 == op2
        assert op2 == op1

        op1.set_attributes({'a' : 10, 'b' : ['c', 'df', 'e'], 'g' : -10})
        assert hasattr(op1, 'attributes')
        assert op1 != op2
        assert op2 != op1

        op2.set_attributes(op1.attributes)
        assert hasattr(op2, 'attributes')
        assert op1 == op2
        assert op2 == op1

    test_init()
    test_all(capsys)
    test_sources(capsys)
    test_destinations(capsys)
    test_integer_sources(capsys)
    test_float_sources(capsys)
    test_integer_destinations(capsys)
    test_float_destinations(capsys)
    test_immediates(capsys)
    test_attributes(capsys)
    test_str()
    test_eq_operator()

def test_class_decoded_instruction():
    def test_init():
        di = decoded_instruction.decoded_instruction()
        assert isinstance(di, decoded_instruction.decoded_instruction)
        assert not hasattr(di, 'word')
        assert not hasattr(di, 'program_counter')
        assert not hasattr(di, 'opcode')
        assert not hasattr(di, 'kind')
        assert not hasattr(di, 'mnemonic')
        assert not hasattr(di, 'operands')

    def test_word():
        words: list[typing.Any] = [-1, "a", [], (0,), [-1,2], {}, {0 : 1, 2 : 3}]

        for word in words:
            di = decoded_instruction.decoded_instruction()
            di.set_word(word)
            if isinstance(word, int):
                assert hasattr(di, 'word')
                assert isinstance(di.word, int)
                assert word == di.word
            else:
                assert not hasattr(di, 'word')

    def test_opcode():
        opcodes: list[typing.Any] = [-1, "a", [], (0,), [-1,2], {}, {0 : 1, 2 : 3}]

        for opcode in opcodes:
            di = decoded_instruction.decoded_instruction()
            di.set_opcode(opcode)
            if isinstance(opcode, int):
                assert hasattr(di, 'opcode')
                assert isinstance(di.opcode, int)
                assert opcode == di.opcode
            else:
                assert not hasattr(di, 'opcode')

    def test_program_counter():
        program_counters: list[typing.Any] = [-1, "a", [], (0,), [-1,2], {}, {0 : 1, 2 : 3}]

        for program_counter in program_counters:
            di = decoded_instruction.decoded_instruction()
            di.set_program_counter(program_counter)
            if isinstance(program_counter, int):
                assert hasattr(di, 'program_counter')
                assert isinstance(di.program_counter, int)
                assert program_counter == di.get_program_counter()
            else:
                assert not hasattr(di, 'program_counter')

    def test_kind():
        for kind in decoded_instruction.instruction_kind:
            di = decoded_instruction.decoded_instruction()
            di.set_kind(kind)
            assert hasattr(di, 'kind')
            assert isinstance(di.kind, decoded_instruction.instruction_kind)

    def test_mnemonic():
        mnemonics: list[typing.Any] = [-1, "a", [], (0,), [-1,2], {}, {0 : 1, 2 : 3}]
        for mnemonic in mnemonics:
            di = decoded_instruction.decoded_instruction()
            di.set_mnemonic(mnemonic)
            if isinstance(mnemonic, str):
                if (0 != len(mnemonic)):
                    assert hasattr(di, 'mnemonic')
                    assert isinstance(di.mnemonic, str)
                    assert mnemonic == di.mnemonic
                else:
                    assert not hasattr(di, 'mnemonic')
            else:
                assert not hasattr(di, 'mnemonic')

    def test_operands():
        operands = []
        op = decoded_instruction.operands()
        op.set_integer_sources([0,1])
        operands.append(op)

        op = decoded_instruction.operands()
        op.set_float_sources([0,1])
        operands.append(op)

        op = decoded_instruction.operands()
        op.set_integer_destinations([0,1])
        operands.append(op)

        op = decoded_instruction.operands()
        op.set_float_destinations([0,1])
        operands.append(op)

        op = decoded_instruction.operands()
        op.set_immediates([0])
        operands.append(op)

        op = decoded_instruction.operands()
        op.set_all({"a" : 0})
        operands.append(op)

        op = decoded_instruction.operands()
        op.set_attributes({"a" : 0, "b" : ["c", "d"]})
        operands.append(op)

        for op in operands:
            di = decoded_instruction.decoded_instruction()
            di.set_operands(op)

            assert hasattr(di, 'operands')
            assert isinstance(di.operands, decoded_instruction.operands)

    def test_str():
        di = decoded_instruction.decoded_instruction()
        di.set_word(0)
        di.set_opcode(0)
        di.set_program_counter(0)
        di.set_mnemonic("a")
        di.set_kind(decoded_instruction.instruction_kind.ttqs)

        op = decoded_instruction.operands()
        op.set_integer_sources([0,1])
        di.set_operands(op)

        msg = di.__str__()
        words = {"decoded instruction", "program counter", "bit instruction", "kind", "mnemonic", "operands"}
        for word in words:
            assert word in msg

    def test_eq_operator():
        di0 = decoded_instruction.decoded_instruction()
        di1 = decoded_instruction.decoded_instruction()

        assert di0 != decoded_instruction.registers()
        assert di1 != decoded_instruction.registers()
        assert di0 == di1
        assert di1 == di0

        di0.set_program_counter(4)
        assert hasattr(di0, 'program_counter')
        assert di0 != di1
        assert di1 != di0

        di1.set_program_counter(di0.program_counter)
        assert hasattr(di1, 'program_counter')
        assert di0 == di1
        assert di1 == di0

        di0.set_word(0x51)
        assert hasattr(di0, 'word')
        assert di0 != di1
        assert di1 != di0

        di1.set_word(di0.word)
        assert hasattr(di1, 'word')
        assert di0 == di1
        assert di1 == di0

        di0.set_opcode(0x51)
        assert hasattr(di0, 'opcode')
        assert di0 != di1
        assert di1 != di0

        di1.set_opcode(di0.opcode)
        assert hasattr(di1, 'opcode')
        assert di0 == di1
        assert di1 == di0

        for kind in decoded_instruction.instruction_kind:
            di0.set_kind(kind)
            assert hasattr(di0, 'kind')
            assert di0 != di1
            assert di1 != di0

            di1.set_kind(di0.kind)
            assert hasattr(di1, 'kind')
            assert di0 == di1
            assert di1 == di0

        di0.set_mnemonic('abc')
        assert hasattr(di0, 'mnemonic')
        assert di0 != di1
        assert di1 != di0

        di1.set_mnemonic(di0.mnemonic)
        assert hasattr(di1, 'mnemonic')
        assert di0 == di1
        assert di1 == di0

        src = decoded_instruction.registers()
        src.set_integers([4,9])
        src.set_floats([0,21])
        op = decoded_instruction.operands()
        op.set_all({'a' : 10, 'b' : 56, 'c' : -100})
        op.set_sources(src)
        op.set_destinations(src)
        op.set_immediates([-4598, 6953])
        op.set_attributes({'a' : 100, 'b' : ['c', 'd', 'e'], 'f' : 0, 'g' : -11556})

        di0.set_operands(op)
        assert hasattr(di0, 'operands')
        assert di0 != di1
        assert di1 != di0

        di1.set_operands(di0.operands)
        assert hasattr(di1, 'operands')
        assert di0 == di1
        assert di1 == di0

    test_init()
    test_word()
    test_program_counter()
    test_opcode()
    test_kind()
    test_mnemonic()
    test_operands()
    test_str()
    test_eq_operator()

def test_to_instruction_kind_0():
    for kind in decoded_instruction.instruction_kind:
        assert kind == decoded_instruction.to_instruction_kind(f"{kind}")

def test_to_instruction_kind_1():
    with pytest.raises(Exception):
        decoded_instruction.to_instruction_kind(f"kind")

def test_extend_sign():
    value = 0b1
    assert -1    == decoded_instruction.extend_sign(value, 0)
    assert value == decoded_instruction.extend_sign(value, 1)

def test_get_instruction_kinds():
    kinds = set()
    for kind in decoded_instruction.instruction_kind:
        kinds.add(kind)

    assert kinds == decoded_instruction.get_instruction_kinds()

def test_get_max_num_bits_opcode():
    for kind in decoded_instruction.instruction_kind:
        if kind.is_tensix():
            assert 8 == decoded_instruction.get_max_num_bits_opcode(kind)
        elif decoded_instruction.instruction_kind.rv32 == kind:
            assert 7 == decoded_instruction.get_max_num_bits_opcode(kind)
        else:
            raise Exception(f"- error: no method defined to determine the max number of bits for opcode for instruction of kind {kind}")

def test_get_num_bits_per_instruction():
    assert 32 == decoded_instruction.get_num_bits_per_instruction()

def test_get_num_bytes_per_instruction():
    assert 4 == decoded_instruction.get_num_bytes_per_instruction()

def test_get_default_instruction_set_file_name():
    pwd = os.path.dirname(os.path.abspath(__file__)) # path of this file.
    for kind in decoded_instruction.instruction_kind:
        path = os.path.normpath(os.path.join(pwd, "..", "..", "ttsim", "config", "llk", "instruction_sets", f"{kind}", "assembly.yaml"))
        assert path == decoded_instruction.get_default_instruction_set_file_name(kind)

    with pytest.raises(Exception) as exc_info:
        decoded_instruction.get_default_instruction_set_file_name("") # type: ignore[arg-type]
    assert "please provide correct instruction kind" in str(exc_info.value)

def test_get_instruction_kinds_from_riscv_attribute():
    rv32_ttwh_attr: tuple[str] = ("riscvrv32i2p0_m2p0_xttwh1p0",)
    rv32_ttbh_attr: tuple[str, str] = ("riscv#rv32i2p0_m2p0_xttbh1p0", "riscvrv32i2p0_m2p0_xttbh1p0")
    rv32_ttqs_attr: tuple[str, str] = ("riscvDrv32i2p0_m2p0_a2p0_f2p0_v1p0_zfh0p1_zvamo1p0_zvlsseg1p0", "riscv@rv32i2p0_m2p0_a2p0_f2p0_v1p0_zfh0p1_zvamo1p0_zvlsseg1p0")

    expected_set = {decoded_instruction.instruction_kind.rv32, decoded_instruction.instruction_kind.ttwh}
    for ele in rv32_ttwh_attr:
        assert sorted(expected_set) == sorted(decoded_instruction.get_instruction_kinds_from_riscv_attribute(ele))

    expected_set = {decoded_instruction.instruction_kind.rv32, decoded_instruction.instruction_kind.ttbh}
    for ele in rv32_ttbh_attr:
        assert sorted(expected_set) == sorted(decoded_instruction.get_instruction_kinds_from_riscv_attribute(ele))

    expected_set = {decoded_instruction.instruction_kind.rv32, decoded_instruction.instruction_kind.ttqs}
    for ele in rv32_ttqs_attr:
        assert sorted(expected_set) == sorted(decoded_instruction.get_instruction_kinds_from_riscv_attribute(ele))

    with pytest.raises(Exception) as exc_info:
        decoded_instruction.get_instruction_kinds_from_riscv_attribute("")
    assert "error: incorrect riscv attribute." in str(exc_info.value)

# @pytest.mark.slow # TODO: add mark slow to ini/pytoml file
def test_get_instruction_set_from_file_name():
    for kind in decoded_instruction.instruction_kind:
        path = decoded_instruction.get_default_instruction_set_file_name(kind)
        instruction_set = decoded_instruction.get_instruction_set_from_file_name(path)
        assert isinstance(instruction_set, dict)

# @pytest.mark.slow
def test_get_default_instruction_set():
    for kind in decoded_instruction.instruction_kind:
        instruction_set = decoded_instruction.get_default_instruction_set(kind)
        assert isinstance(instruction_set, dict)

def test_left_circular_shift():
    word   = 0b1100_0000_0000_0000_0000_0000_0000_0000
    expect = 0b0000_0000_0000_0000_0000_0000_0000_0011
    received = decoded_instruction.left_circular_shift(word, 2, 32)
    assert received == expect

def test_right_circular_shift():
    word   = 0b1100_0000_0000_0000_0000_0000_0000_0000
    expect = 0b0011_0000_0000_0000_0000_0000_0000_0000
    received = decoded_instruction.right_circular_shift(word, 2, 32)
    assert received == expect

def test_swizzle_instruction():
    word   = 0b1100_0000_0000_0000_0000_0000_0000_0000
    expect = 0b0000_0000_0000_0000_0000_0000_0000_0011
    received = decoded_instruction.swizzle_instruction(word)
    assert received == expect

def test_unswizzle_instruction():
    word   = 0b1100_0000_0000_0000_0000_0000_0000_0000
    expect = 0b0011_0000_0000_0000_0000_0000_0000_0000
    received = decoded_instruction.unswizzle_instruction(word)
    assert received == expect