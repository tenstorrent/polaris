#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import elftools
import fabric # type: ignore [import-untyped]
import pytest
import textwrap
import typing

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.read_elf as read_elf
import ttsim.front.llk.rv32 as rv32
import ttsim.front.llk.ttbh as ttbh
import ttsim.front.llk.ttqs as ttqs
import ttsim.front.llk.ttwh as ttwh

TEST_WITH_ELF_FILES = True
LOCAL_ELF_TEST_DIR = "tests/__data_files/llk_elf_files"


def test_get_riscv_attribute_from_elf_object():
    if TEST_WITH_ELF_FILES:
        kinds_attributes = decoded_instruction.get_instruction_kinds_rv32_tensix_attributes_dict()
        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    with open(file_name_incl_path, 'rb') as file:
                        elf = elftools.elf.elffile.ELFFile(file)
                        attr = read_elf.get_riscv_attribute_from_elf_object(elf)
                        for kind in decoded_instruction.instruction_kind:
                            if kind.is_tensix() and (f"{kind}" in file_name_incl_path):
                                kinds = (decoded_instruction.instruction_kind.rv32, kind)
                                assert attr in kinds_attributes[kinds]

def test_get_riscv_attribute():
    if TEST_WITH_ELF_FILES:
        kinds_attributes = decoded_instruction.get_instruction_kinds_rv32_tensix_attributes_dict()
        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    with open(file_name_incl_path, 'rb') as file:
                        elf = elftools.elf.elffile.ELFFile(file)
                        attr = read_elf.get_riscv_attribute(elf)
                        for kind in decoded_instruction.instruction_kind:
                            if kind.is_tensix() and (f"{kind}" in file_name_incl_path):
                                kinds = (decoded_instruction.instruction_kind.rv32, kind)
                                assert attr in kinds_attributes[kinds]

        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    attr = read_elf.get_riscv_attribute(file_name_incl_path)
                    for kind in decoded_instruction.instruction_kind:
                        if kind.is_tensix() and (f"{kind}" in file_name_incl_path):
                            kinds = (decoded_instruction.instruction_kind.rv32, kind)
                            assert attr in kinds_attributes[kinds]

    with pytest.raises(Exception):
        read_elf.get_riscv_attribute(typing.cast(typing.Any, []))

def test_get_instruction_kinds_from_riscv_attribute():
    kinds_attributes = decoded_instruction.get_instruction_kinds_rv32_tensix_attributes_dict()
    attribute_list = []
    for value in kinds_attributes.values():
        attribute_list.extend(value)

    for attr in attribute_list:
        kinds = read_elf.get_instruction_kinds_from_riscv_attribute(attr)
        assert attr in kinds_attributes[kinds]

def test_get_instruction_kinds():
    if TEST_WITH_ELF_FILES:
        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    with open(file_name_incl_path, 'rb') as file:
                        elf = elftools.elf.elffile.ELFFile(file)
                        kinds = read_elf.get_instruction_kinds(elf)
                        expected_kinds = decoded_instruction.get_instruction_kinds_from_riscv_attribute(read_elf.get_riscv_attribute(elf))
                        assert kinds == expected_kinds

        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    kinds = read_elf.get_instruction_kinds(file_name_incl_path)
                    expected_kinds = decoded_instruction.get_instruction_kinds_from_riscv_attribute(read_elf.get_riscv_attribute(file_name_incl_path))
                    assert kinds == expected_kinds

    with pytest.raises(Exception):
        read_elf.get_instruction_kinds(typing.cast(typing.Any, []))

def test_get_instruction_modules_from_elf_object():
    if TEST_WITH_ELF_FILES:
        kinds_modules = dict()
        kinds_modules[decoded_instruction.instruction_kind.rv32] = rv32
        kinds_modules[decoded_instruction.instruction_kind.ttbh] = ttbh
        kinds_modules[decoded_instruction.instruction_kind.ttqs] = ttqs
        kinds_modules[decoded_instruction.instruction_kind.ttwh] = ttwh

        if undef_kinds := [kind for kind in decoded_instruction.instruction_kind if kind not in kinds_modules.keys()]:
            raise Exception(f"- error: no module defined for instruction kind(s): {undef_kinds}")

        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    with open(file_name_incl_path, 'rb') as file:
                        elf = elftools.elf.elffile.ELFFile(file)
                        modules = read_elf.get_instruction_modules_from_elf_object(elf)
                        for kind, module in modules.items():
                            assert module is kinds_modules[kind]
                        for kind in modules.keys():
                            if kind.is_tensix():
                                assert f"{kind}" in file_name_incl_path

                        assert decoded_instruction.instruction_kind.rv32 in modules.keys()

def test_get_instruction_modules():
    if TEST_WITH_ELF_FILES:
        kinds_modules = dict()
        kinds_modules[decoded_instruction.instruction_kind.rv32] = rv32
        kinds_modules[decoded_instruction.instruction_kind.ttbh] = ttbh
        kinds_modules[decoded_instruction.instruction_kind.ttqs] = ttqs
        kinds_modules[decoded_instruction.instruction_kind.ttwh] = ttwh

        if undef_kinds := [kind for kind in decoded_instruction.instruction_kind if kind not in kinds_modules.keys()]:
            raise Exception(f"- error: no module defined for instruction kind(s): {undef_kinds}")

        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    with open(file_name_incl_path, 'rb') as file:
                        elf = elftools.elf.elffile.ELFFile(file)
                        modules = read_elf.get_instruction_modules(elf)
                        for kind, module in modules.items():
                            assert module is kinds_modules[kind]
                        for kind in modules.keys():
                            if kind.is_tensix():
                                assert f"{kind}" in file_name_incl_path

                        assert decoded_instruction.instruction_kind.rv32 in modules.keys()

                        del elf

            for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
                for file_name in file_names:
                    if file_name.endswith(".elf"):
                        file_name_incl_path = os.path.join(pwd, file_name)
                        modules = read_elf.get_instruction_modules(file_name_incl_path)
                        for kind, module in modules.items():
                            assert module is kinds_modules[kind]
                        for kind in modules.keys():
                            if kind.is_tensix():
                                assert f"{kind}" in file_name_incl_path

                        assert decoded_instruction.instruction_kind.rv32 in modules.keys()

    with pytest.raises(Exception):
        read_elf.get_instruction_modules(typing.cast(typing.Any, []))

@pytest.mark.slow
def test_get_default_instruction_sets_from_riscv_attribute():
    kinds_attributes = decoded_instruction.get_instruction_kinds_rv32_tensix_attributes_dict()
    kinds_modules = dict()
    kinds_modules[decoded_instruction.instruction_kind.rv32] = rv32
    kinds_modules[decoded_instruction.instruction_kind.ttbh] = ttbh
    kinds_modules[decoded_instruction.instruction_kind.ttqs] = ttqs
    kinds_modules[decoded_instruction.instruction_kind.ttwh] = ttwh

    if undef_kinds := [kind for kind in decoded_instruction.instruction_kind if kind not in kinds_modules.keys()]:
        raise Exception(f"- error: no module defined for instruction kind(s): {undef_kinds}")

    kinds_instruction_sets = {kind : kinds_modules[kind].get_default_instruction_set() for kind in decoded_instruction.instruction_kind}
    for kinds, attributes in kinds_attributes.items():
        for attr in attributes:
            sets = read_elf.get_default_instruction_sets_from_riscv_attribute(attr)
            assert sorted(kinds) == sorted(sets.keys())
            for kind, ins_set in sets.items():
                assert ins_set == kinds_instruction_sets[kind]

@pytest.mark.slow
def test_get_default_instruction_sets_from_elf_object():
    if TEST_WITH_ELF_FILES:
        kinds_modules = dict()
        kinds_modules[decoded_instruction.instruction_kind.rv32] = rv32
        kinds_modules[decoded_instruction.instruction_kind.ttbh] = ttbh
        kinds_modules[decoded_instruction.instruction_kind.ttqs] = ttqs
        kinds_modules[decoded_instruction.instruction_kind.ttwh] = ttwh

        if undef_kinds := [kind for kind in decoded_instruction.instruction_kind if kind not in kinds_modules.keys()]:
            raise Exception(f"- error: no module defined for instruction kind(s): {undef_kinds}")

        kinds_instruction_sets = {kind : kinds_modules[kind].get_default_instruction_set() for kind in decoded_instruction.instruction_kind}
        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    with open(file_name_incl_path, 'rb') as file:
                        elf = elftools.elf.elffile.ELFFile(file)
                        sets = read_elf.get_default_instruction_sets_from_elf_object(elf)
                        for kind in sets.keys():
                            if kind.is_tensix():
                                assert f"{kind}" in file_name_incl_path
                        assert decoded_instruction.instruction_kind.rv32 in sets.keys()
                        for kind, ins_set in sets.items():
                            assert ins_set == kinds_instruction_sets[kind]

@pytest.mark.slow
def test_get_default_instruction_sets_0():
    if TEST_WITH_ELF_FILES:
        kinds_modules = dict()
        kinds_modules[decoded_instruction.instruction_kind.rv32] = rv32
        kinds_modules[decoded_instruction.instruction_kind.ttbh] = ttbh
        kinds_modules[decoded_instruction.instruction_kind.ttqs] = ttqs
        kinds_modules[decoded_instruction.instruction_kind.ttwh] = ttwh

        if undef_kinds := [kind for kind in decoded_instruction.instruction_kind if kind not in kinds_modules.keys()]:
            raise Exception(f"- error: no module defined for instruction kind(s): {undef_kinds}")

        kinds_instruction_sets = {kind : kinds_modules[kind].get_default_instruction_set() for kind in decoded_instruction.instruction_kind}
        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    with open(file_name_incl_path, 'rb') as file:
                        elf = elftools.elf.elffile.ELFFile(file)
                        sets = read_elf.get_default_instruction_sets(elf)
                        for kind in sets.keys():
                            if kind.is_tensix():
                                assert f"{kind}" in file_name_incl_path
                        assert decoded_instruction.instruction_kind.rv32 in sets.keys()
                        for kind, ins_set in sets.items():
                            assert ins_set == kinds_instruction_sets[kind]

@pytest.mark.slow
def test_get_default_instruction_sets_1():
    if TEST_WITH_ELF_FILES:
        kinds_modules = dict()
        kinds_modules[decoded_instruction.instruction_kind.rv32] = rv32
        kinds_modules[decoded_instruction.instruction_kind.ttbh] = ttbh
        kinds_modules[decoded_instruction.instruction_kind.ttqs] = ttqs
        kinds_modules[decoded_instruction.instruction_kind.ttwh] = ttwh

        if undef_kinds := [kind for kind in decoded_instruction.instruction_kind if kind not in kinds_modules.keys()]:
            raise Exception(f"- error: no module defined for instruction kind(s): {undef_kinds}")

        kinds_instruction_sets = {kind : kinds_modules[kind].get_default_instruction_set() for kind in decoded_instruction.instruction_kind}
        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            for file_name in file_names:
                if file_name.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file_name)
                    sets = read_elf.get_default_instruction_sets(file_name_incl_path)
                    for kind in sets.keys():
                        if kind.is_tensix():
                            assert f"{kind}" in file_name_incl_path
                    assert decoded_instruction.instruction_kind.rv32 in sets.keys()
                    for kind, ins_set in sets.items():
                        assert ins_set == kinds_instruction_sets[kind]

def test_get_default_instruction_sets_exception():
    with pytest.raises(Exception):
        read_elf.get_default_instruction_sets(typing.cast(typing.Any, []))

def test_get_all_function_ranges_from_elf_object():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf")
        expected = [
            ('_ZNK7ckernel10addr_mod_t7src_valEv', 42400, 42492),
            ('_ZNK7ckernel10addr_mod_t8dest_valEv', 42492, 42604),
            ('_GLOBAL__sub_I__ZN7ckernel16ckernel_templateC2Emmm', 42364, 42392),
            ('wzerorange', 42608, 42620),
            ('memcpy', 42624, 42908),
            ('_init', 41348, 41352),
            ('_start', 40960, 41348),
            ('memset', 42908, 43128),
            ('main', 41360, 42364),
            ('_fini', 41348, 41352),
            ('exit', 42604, 42608)
            ]
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            funcs_ranges = read_elf.get_all_function_ranges_from_elf_object(elf)
            assert isinstance(funcs_ranges, list)
            assert sorted(expected) == sorted(funcs_ranges)

def test_get_all_function_ranges():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162/trisc2/trisc2.elf")
        expected = sorted([('_start', 23408, 23412), ('_Z13kernel_launchm', 23412, 24640)])
        funcs_ranges = read_elf.get_all_function_ranges(file_name)
        assert isinstance(funcs_ranges, list)
        assert expected == sorted(funcs_ranges)

        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttwh/tt-metal/bmm/686715490291187641/trisc0/trisc0.elf")
        expected = sorted([('_start', 18784, 18788), ('_Z13kernel_launchm', 18788, 20016)])
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            funcs_ranges = read_elf.get_all_function_ranges(elf)
            assert isinstance(funcs_ranges, list)
            assert expected == sorted(funcs_ranges)

    with pytest.raises(Exception) as exe_info:
        read_elf.get_all_function_ranges(typing.cast(typing.Any, []))
    assert "expected argument to be either elf file name (including path) or ELFFile object" in str(exe_info.value)

def test_get_all_function_names_from_elf_object():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttwh/tt-metal/bmm/686715490291187641/trisc1/trisc1.elf")
        expected = sorted(['_Z24matmul_configure_addrmodILi4ELN7ckernel17DstTileFaceLayoutE0EEvbmmmmmmmb.part.0.constprop.0', '_start', '_Z13kernel_launchm'])
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            funcs = read_elf.get_all_function_names_from_elf_object(elf)
            assert expected == sorted(funcs)

def test_get_all_function_names():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162/trisc0/trisc0.elf")
        expected = sorted(['_start', '_Z13kernel_launchm'])
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            funcs = read_elf.get_all_function_names(elf)
            assert expected == sorted(funcs)

        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162/trisc1/trisc1.elf")
        expected = sorted(['_Z24matmul_configure_addrmodILi4ELN7ckernel17DstTileFaceLayoutE0EEvbmmmmmmmb.part.0.constprop.0', '_start', '_Z13kernel_launchm'])
        funcs = read_elf.get_all_function_names(file_name)
        assert expected == sorted(funcs)

    with pytest.raises(Exception):
        read_elf.get_all_function_names(typing.cast(typing.Any, []))

def test_get_function_range_from_elf_object():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttwh/tt-metal/bmm/686715490291187641/trisc2/trisc2.elf")
        funcs = [('_start', 21872, 21876), ('_Z13kernel_launchm', 21876, 23240)]
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            for func_name, start, end in funcs:
                func_start_end = read_elf.get_function_range_from_elf_object(func_name, elf)
                if func_start_end is None:
                    raise Exception("- error: expected func_start_end to be tuple (func_name, start, end), but is None")

                assert func_start_end[0] == func_name
                assert func_start_end[1] == start
                assert func_start_end[2] == end

def test_get_function_range():
    if TEST_WITH_ELF_FILES:

        def test_elf_object():
            file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf")
            funcs = [
                ('_ZN7ckernel5trisc26_configure_buf_desc_table_ERK17tdma_descriptor_t', 26208, 26256),
                ('_ZN7ckernel5triscL16SCALE_DATUM_SIZEEmm', 26256, 26316),
                ('_GLOBAL__sub_I__ZN6UPK_0119PARAM_SRCA_PER_ITERE', 26176, 26204),
                ('wzerorange', 26320, 26332),
                ('memcpy', 26336, 26620),
                ('_init', 24964, 24968),
                ('_start', 24576, 24964),
                ('memset', 26620, 26840),
                ('main', 24976, 26176),
                ('_fini', 24964, 24968), ('exit', 26316, 26320)]
            with open(file_name, 'rb') as file:
                elf = elftools.elf.elffile.ELFFile(file)
                for func_name, start, end in funcs:
                    func_start_end = read_elf.get_function_range_from_elf_object(func_name, elf)
                    if func_start_end is None:
                        raise Exception("- error: expected func_start_end to be tuple (func_name, start, end), but is None")

                    assert func_start_end[0] == func_name
                    assert func_start_end[1] == start
                    assert func_start_end[2] == end

        def test_elf_file():
            file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_2/out/thread_2.elf")
            funcs = [
                ('_GLOBAL__sub_I__ZN6PCK_0134PARAM_OUTPUT_TOTAL_NUM_TILES_C_DIME', 58564, 58592),
                ('wzerorange', 58608, 58620),
                ('memcpy', 58620, 58904),
                ('_init', 57732, 57736),
                ('_start', 57344, 57732),
                ('memset', 58904, 59124),
                ('main', 57744, 58564),
                ('_fini', 57732, 57736),
                ('exit', 58592, 58596)]

            for func_name, start, end in funcs:
                func_start_end = read_elf.get_function_range(func_name, file_name)
                if func_start_end is None:
                    raise Exception("- error: expected func_start_end to be tuple (func_name, start, end), but is None")

                assert func_start_end[0] == func_name
                assert func_start_end[1] == start
                assert func_start_end[2] == end

        test_elf_object()
        test_elf_file()

def test_get_function_range_exception():
    with pytest.raises(Exception):
        read_elf.get_function_range("main", typing.cast(typing.Any, []))

def test_get_instruction_bytes_of_function_from_elf_object():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_2/out/thread_2.elf")
        func_name = "main"
        expected_bytes = b'\x13\x01\x01\xfc#.\x11\x02#,\x81\x027\xf7\x80\x00\x83\'\x07\x08\x93\xf7\xf7\x0f\xe3\x8c\x07\xfe7\x04\x80\x00\x93\x07\xf0\xff#(\xf4\x10\xb7\x07\x04\x04\x93\x87G@#"\xf4\x04#$\xf4\x04#(\xf4\x04# \x04&\x13\x07\x10\x11\xb7\x07\x82\x00#\xae\xe7:\xb7WP\x00\x93\x87\x070#$\xf1\x00\xb7\x07\x00\x01#&\xf1\x00\x13\x06\x00\x01\x93\x07\x00A\x93\x05\x81\x00\x13\x05\x81\x01#(\xf1\x00\xef\x00\x80/\xb7f\x80\x00\xb7U\x00\x007\xe7\x00\x00\x13\x86\x06\x00\x13\x05\x00\x08\x13\x83\x058\x13\x08\x10\x00\x93\x85\x050# \xa6\x0e#$\xa6\x0e#"f\x0e#*\xb6\x0e#,\x06\x0e#&\x06\x0f\x93\x07\x87\x00\x93\x08\x80\x00#\xa4\x17\x01\x83\xa8\x87\x00\x03&\x06\x0f\xb7\x08\xe0p\x93\x88\x08\x05#$\x17\x01\x83(\x87\x00#\xa4\xb7\x00#\xa6\xc7\x00\x03\xa6\x87\x00\x03\xa6\xc7\x00\x93\x86\x06\x007\x06`\xab\x13\x06\x16\x05#$\xc7\x00\x03&\x87\x00#\xa4g\x00#\xa6\xa7\x00\x03\xa6\x87\x00\x03\xa6\xc7\x007\x06p\xc0\x13\x06&\x05#$\xc7\x00\x03&\x87\x00#\xa4\xa7\x00#\xa6\x07\x01\x03\xa6\x87\x00\x83\xa7\xc7\x00\x13\x060\x00\xb7\x070\x82\x93\x877\x05#$\xf7\x00\x83\'\x87\x00\x13\x05\x87\x00\xb7\xb7\x80\x00#\xa8\x07\x11#\xa2\x07\x10\xb7\xe7\x00\x00\x83\xa7\x87oct\xf6\x1c\xb7\x07p\x8c\x93\x87\x87\x10#$\xf7\x00\x83\'\x87\x00\x93\x07\x00\xfe#$\xf4&#&\xf4&#(\xf4&7VP\x00\xb7\x07\x82\x00\x13\x06\x060#\xa8\xc7"\xb7\x05\x00\x01#\xaa\xb7"\x13\x86G#\x83\'\x01\x027\x08\x00\x02#"\xf6\x00\xb7\x071\xb27\xe6\x80\x00\x93\x87W\xf0# \xf6\x00\xb7\xc7\xad\xde\x93\x87\xf7\xee#"\xf1\x00\x83%A\x00\xb7\xf7\x80\x00#\xa4\xb7\x00\x83\xa7\x87\x00\x93\x05 \x00#"\xf1\x00\xb7\xd7\x80\x00#\xa0\xb7\x08\x93\x05\x10\x00#\xa0\xb7\x00#\xa2\xb7\x00#\xa4\x07\x01#\xa6\x07\x01\xb7\x05\x00\x19#\xa8\x07\x01\x93\x85\x05\n#\xaa\xb7\x00#\xac\x07\x01#\xae\xb7\x00#\xa0\xb7\x02\xb7\x07\x02\xab\x93\x87\x87\x02# \xf6\x00#$\x05\x00#&\x05\x00\x83\'\x85\x00\x83\'\xc5\x00\xb7\x07\xb0\x17\x93\x877\x07#$\xf7\x00\x83\'\x87\x00\xb7\x07\x0c\x06# \xf6\x00\xb7\x07\x0c\r# \xf6\x00\x00\x00\x00\x06\xb7\x87\x01=\x93\x87\x87\x02# \xf6\x00\x03\xd7\xa6\x0f\x83\xa7\x86\x0e\x13\x07\x17\x00#\x9d\xe6\x0e\x03\xa7F\x0f\xb3\x87\xe7\x00\x03\xa7F\x0e#\xaa\xf6\x0ec\xe8\xe7\x00\x03\xa7\x06\x0e\xb3\x87\xe7@#\xaa\xf6\x0e\xa2\n\x80\x88\xb7\xe7\x00\x00\x83\xa6\xc7o7\x07\x10\x103\x07\xd7\x00\xb7\xe6\x80\x00#\xa0\xe6\x00\x80\x00\x00\xd8\x83\xa6\xc7o\x13\x07\x10\x003\x07\xd7@#\xae\xe7n7\xb7\x80\x00\x83\'\xc7\x10\xe3\x9e\x07\xfe\xb7\xc6\xce\xfa\x13\x07\xa0\x00\xb7U\x01\x00\x93\x86\xf6\xee\x83\xd7\xe5\xff\x13\x96\x07\x01\x83\xd7\x05\x00\x13V\x06\x01\x93\x97\x07\x01\xb3\xe7\xc7\x00c\x86\xd7\x00\x13\x07\xf7\xff\xe3\x10\x07\xfe7\xe7\x00\x00\x83\'G\x00\x83 \xc1\x03\x03$\x81\x03\x93\x87\x17\x00#"\xf7\x00\x13\x05\x00\x00\x13\x01\x01\x04g\x80\x00\x00\x93\x87\x87\t\x93\x97\'\x00\xb3\x07\xf4\x00\x13\x06\x00\xfe#\xa4\xc7\x00o\xf0\x9f\xe4'

        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            instruction_bytes = read_elf.get_instruction_bytes_of_function_from_elf_object(func_name, elf)
            assert expected_bytes == instruction_bytes

        with pytest.raises(Exception) as exe_info:
            with open(file_name, 'rb') as file:
                elf = elftools.elf.elffile.ELFFile(file)
                instruction_bytes = read_elf.get_instruction_bytes_of_function_from_elf_object("", elf)
        assert "not present in given elf file" in str(exe_info.value)

def test_get_instruction_bytes_of_function():
    if TEST_WITH_ELF_FILES:
        def test_elf_obj():
            file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf")
            func_name = "main"
            expected_bytes = b'\x13\x01\x01\xf6#.\x11\x08#,\x81\x08#*\x91\x08#(!\t#&1\t#$A\t#"Q\t# a\t#.q\x07#,\x81\x07#*\x91\x07#(\xa1\x07#&\xb1\x07\xb7\x07\x80\x00\x13\x07P\x00#\xa0\xe7&7G\t\x00\x13\x07\xc7\xd0#\xa8\xe7\x107w\x18:\x13\x07\xe7\xb0#\xa2\xe7\x047\x87IR\x13\x077\xe7#\xa4\xe7\x047\'nf\x13\x07w\xd2#\xa8\xe7\x04\x06\x00D\x8c\x0f\x00\xf0\x0f\xf3\'\x10\xbc\x93\xf7\x07\x08\xe3\x9a\x07\xfe\xb7\t\x80\x00\x93\x07\xf0\xff#\xa8\xf9\x10\xb7\x07\x04\x04\x93\x87G@#\xa2\xf9\x04#\xa4\xf9\x04#\xa8\xf9\x04#\xa0\t&\x93\x05\x00@7%\x05\x00\xef\x00\xc0D7WP\x00\x93\x07\x07 \x13\x04\x05\x00\xb7\x08\x00\x01\x13\x08\x00A\x93\x05\x01\x01\x13\x06\x00\x01\x13\x05\x01\x03#(\xf1\x00#*\x11\x01#,\x01\x017[\x00\x00\xef\x00\x80F\x13UD\x00\x93\x07\x0b \xb7-\x80\x00\xb7d\x00\x00\x13\x89\r\x003\x06\xf5\x00\x93\x06P\x00\x13\x84\x84\x00#"\xd1\x04#"\xc9\x00# \xa9\x00#$\xa9\x00# \x01\x04#(\xf9\x00#,\t\x00#$\x04\x00\x83%\x84\x007\r\x10\xcf\x93\x05\xed\x04#\xa4\xb4\x00\x83\xa5\x84\x00\x83%I\x01\xb7\x0c\x90o7\x0c\xa0r#$\xb4\x00#&\xf4\x00\x83\'\x84\x00\x83\'\xc4\x00\xb7\x0b`\xaa\x93\x87\xfc\x04#\xa4\xf4\x00\x83\xa7\x84\x00#$\xc4\x00#&\xa4\x00\x83\'\x84\x00\x83\'\xc4\x007\xba\x80\x00\x93\x07\x0c\x05#\xa4\xf4\x00\x83\xa7\x84\x00\x83\'\xc9\x00#$\xa4\x00\x93\n\x10\x00#&\xf4\x00\x83\'\x84\x00\x83\'\xc4\x007WP\x00\x93\x87\x1b\x05#\xa4\xf4\x00\x83\xa7\x84\x00#(Z\x01#"\n\x00\x13\x06\x07(#$Z\x01\xb7\x08\x00\x01\x13\x08\x00A# \xc1\x02\x93\x05\x01\x02\x13\x06\x00\x01\x13\x05\x81\x04#"\x11\x03#$\x01\x03\xef\x00\xc0675\x05\x00\x93\x06P\x00\x93\x05\x00@\x13\x05\x05\x80#.\xd1\x04#,Q\x05\xef\x00\x000\x13UE\x00\x13\x07\x0b(\xb3\x07\xe5\x00# \xf9\x02#&\xe9\x02#.\xa9\x00#*\t\x02#"\xa9\x02#$T\x01\x13\x06\x8d\x07\x83&\x84\x00#\xa4\xc4\x00\x83\xa6\x84\x00\x83&\t\x03\x13\x86\x9c\x07#$\xd4\x00#&\xe4\x00\x03\'\x84\x00\x03\'\xc4\x00#\xa4\xc4\x00\x03\xa7\x84\x00#$\xf4\x00#&\xa4\x00\x83\'\x84\x00\x93\x06\xac\x07\x83\'\xc4\x00#\xa4\xd4\x00\x83\xa7\x84\x00\x83\'\x89\x02#$\xa4\x00\x13\x87\xbb\x07#&\xf4\x00\x83\'\x84\x00\x83\'\xc4\x00#\xa4\xe4\x00\x83\xa7\x84\x00#(Z\x03#"\n\x02#$Z\x03\x1c\x00\x00D\xb7w\x00\x00\x83\xa7\xc7\x8d\x13\x070\x00\x93\x8d\r\x00cr\xf7\x1e\xb7\x07p\x8c\x93\x87\x87\x10#\xa4\xf4\x00\x83\xa7\x84\x00\x93\x07\x00\xfe#\xa4\xf9&#\xa6\xf9&#\xa8\xf9&\x13\x05\x01\x03\xef\x00@\x1f\x83GA\x047\x07\r\xb2\x13\x07\x07\xf0\xb7\xe4\x80\x00\xb3\x87\xe7\x00#\xa0\xf4\x00\x13\x05\x81\x04\xef\x00@\x1d\x83G\xc1\x057\x07\x19\xb2\x13\x07\x07\xf0\xb3\x87\xe7\x00#\xa0\xf4\x00\xb7\x07\x0c\xb4\x93\x87\x07\x10#\xa0\xf4\x00\xb7\xc7\xad\xde\x93\x87\xf7\xee#&\xf1\x00\x03\'\xc1\x00\xb7\xf7\x80\x00#\xa4\xe7\x00\x83\xa7\x87\x00\x13\x07 \x00#&\xf1\x00\xb7\xd7\x80\x00#\xa0\xe7\x08\x13\x07\x10\x00#\xa0\xe7\x00#\xa2\xe7\x007\x07\x00i\x13\x07\'\x00#\xa4\xe7\x007\x07\x00\x02#\xa6\xe7\x00#\xa8\xe7\x007\x07\x00D\x13\x07g\x00#\xaa\xe7\x007\x07\x00\x07\x13\x07\x17\x00#\xac\xe7\x00#\xae\xe7\x00#\xa0\xe7\x02\xb7\x07\x04\xa9\x13\x87\x07\x02#\xa0\xe4\x00\x93\x87\x17\x02#\xa0\xf4\x00\x00\x00\x104\xb7\x07\x04\x06#\xa0\xf4\x00\xb7\x07\x00\x06#\xa0\xf4\x00\x00\x00\x00\x06\xb7\x87\x03>\x93\x87\x07\x02#\xa0\xf4\x00\x83\xd7\x8d\x01\x03\xa7\r\x01\x93\x87\x17\x00#\x9c\xfd\x00\x83\xa7\x8d\x00\xb3\x87\xe7\x00\x03\xa7M\x00#\xa8\xfd\x00c\xe8\xe7\x00\x03\xa7\r\x00\xb3\x87\xe7@#\xa8\xfd\x00\xb7\x87\x03>7\xe7\x80\x00\x93\x87\x17\x02# \xf7\x00\x83\xd7M\x03\x03\xa7\xcd\x02\x93\x87\x17\x00#\x9a\xfd\x02\x83\xa7M\x02\xb3\x87\xe7\x00\x03\xa7\r\x02#\xa6\xfd\x02c\xe8\xe7\x00\x03\xa7\xcd\x01\xb3\x87\xe7@#\xa6\xfd\x02\x0f\x00\xf0\x0f\xf3\'\x10\xbc\x93\xf7\x07\x01\xe3\x9a\x07\xfe7g\x00\x00\x83\'G\x00\x83 \xc1\t\x03$\x81\t\x93\x87\x17\x00#"\xf7\x00\x83$A\t\x03)\x01\t\x83)\xc1\x08\x03*\x81\x08\x83*A\x08\x03+\x01\x08\x83+\xc1\x07\x03,\x81\x07\x83,A\x07\x03-\x01\x07\x83-\xc1\x06\x13\x05\x00\x00\x13\x01\x01\ng\x80\x00\x00\x93\x87\x87\t\x93\x97\'\x00\xb3\x89\xf9\x00\x93\x07\x00\xfe#\xa4\xf9\x00o\xf0\xdf\xe2'

            with open(file_name, 'rb') as file:
                elf = elftools.elf.elffile.ELFFile(file)
                instruction_bytes = read_elf.get_instruction_bytes_of_function(func_name, elf)
                assert expected_bytes == instruction_bytes

        def test_elf_file():
            file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf")
            func_name = "main"
            expected_bytes = b'\x13\x01\x01\xfd#&\x11\x02#$\x81\x02#"\x91\x02# !\x03#.1\x017\xf7\x80\x00\x83\'\x07\x08\x93\xf7\xf7\x0f\xe3\x8c\x07\xfe\xb7\x07\x80\x00\x13\x07\xf0\xff#\xa8\xe7\x107\x07\x04\x04\x13\x07G@#\xa2\xe7\x04#\xa4\xe7\x04#\xa8\xe7\x04#\xa0\x07&7\x07\x82\x00\x93\x06\xf0\x10#*\xd7:\xb7\xb4\x00\x00\x03\xa7\xc4\x8b\x93\x060\x00c\xf6\xe6$\xb7\x06p\x8c7\xa6\x00\x00\x93\x86\x86\x10#$\xd6\x00\x83&\x86\x00\x93\x06\x00\xfe#\xa4\xd7&#\xa6\xd7&#\xa8\xd7&\x13\x07G\x0f7\x04\x82\x00\xb7\x87\x82\x02\x13\x17\'\x00#"\xf1\x003\x07\xe4\x00\x13\t\x10\x00\xb7\x17\x02\x00# \'\x01\x93\x87W\xb0# \xf4\x00\x83\'A\x00\x13\x05\x01\x00#"\xf4\x00#$\x04\x00# \x01\x00#"\x01\x00#$\x01\x00#\x16\x01\x00\xef\x00\x803#*\xa4D\x13\x05\x01\x00\xef\x00\x808#*\xa4T\x00\x00e@\x93\x07\x80\x00# \x01\x00#"\x01\x00\x13\x05\x01\x00\xa3\x01\xf1\x00#\x13\xf1\x00#$\x01\x00#\x16\x01\x00\xef\x00\x000# \xa4D\x13\x05\x01\x00\xef\x00\x005\xb7\xb5\x00\x00\x93\x89\x85\x87\x13\x06\xe0\x00\x93\x85\x85\x87# \xa4T\x13\x05\x01\x00\xef\x00\x80;\x13\x05\x01\x00\xef\x00\x00-#*\xa4D\x13\x05\x01\x00\xef\x00\x002\x93\x85\t\x01\x13\x06\xe0\x00#*\xa4T\x13\x05\x01\x00\xef\x00\x009\x13\x05\x01\x00\xef\x00\x80*#"\xa4D\x13\x05\x01\x00\xef\x00\x80/\x93\x85\t\x02\x13\x06\xe0\x00#"\xa4T\x13\x05\x01\x00\xef\x00\x806\x13\x05\x01\x00\xef\x00\x00(#$\xa4D\x13\x05\x01\x00\xef\x00\x00-\x13\x06\xe0\x00\x93\x85\t\x03#$\xa4T\x13\x05\x01\x00\xef\x00\x004\x13\x05\x01\x00\xef\x00\x80%#(\xa4D\x13\x05\x01\x00\xef\x00\x80*#(\xa4T\x0f\x00\xf0\x0f\x13\x03\x10\x00\x13\x13#\x01s \x03|\x04\x04\x00\x10\x00\x00\x00\x98\x00\x00\x01\x98\x00\x00\x00\x98\x00\x00\x02\x98\x00\x00\x00\x98\x00\x00\x01\x98\x00\x00\x00\x98\x00\x00\x04\x98\x00\x00\x00\x98\x00\x00\x01\x98\x00\x00\x00\x98\x00\x00\x02\x98\x00\x00\x00\x98\x00\x00\x01\x98\x00\x00\x00\x98\x00\x00\x05\x99\x13\x03\x10\x00\x13\x13#\x01s0\x03|\xb7\xc7\xad\xde\x93\x87\xf7\xee# \xf1\x00\x03\'\x01\x00\xb7\xf7\x80\x00\x13\x06 \x00#\xa4\xe7\x00\x83\xa7\x87\x00\xb7\x06\x00\x027\x07\x00\x04# \xf1\x00\xb7\xd7\x80\x00#\xa0\xc7\x08#\xa0\'\x01#\xa2\'\x01#\xa4\xd7\x00#\xa6\xd7\x00#\xa8\xd7\x00\x13\x07\x07\x10#\xaa\xe7\x00#\xac\xd7\x00#\xae\xe7\x00#\xa0\xe7\x02<\x00\x00\xdc\xb7\x07\x80\x00\x03\xa7G\x0c#\xa0\'\x03\x83\xa7\xc4\x8bc\x80\xc7\x14cd\xf6\x02c\x98\x07\x12# \x04\x02o\x00@\x02\x93\x06\x87\t\x93\x96&\x00\xb3\x87\xd7\x00\x93\x06\x00\xfe#\xa4\xd7\x00o\xf0\x9f\xdc\x93\x060\x00c\x8c\xd7\x10\x00\x00\x00\x06<\x00\x00\xdeV\x00\x80\x88 \x00\x00\xd8\x0f\x00\xf0\x0f\xf3\'\x10\xbc\x93\xf7\'\x00\xe3\x9a\x07\xfe\x0f\x00\xf0\x0f\xf3\'\x10\xbc\x93\xf7\x07@\xe3\x9a\x07\xfe\xb7\x07\x80\x00\x93\x06 \x00#\xa0\xd7\x02\x83\xa5G\x0c\x03\xa6\x87\n\x03\xa5\xc7\n\xb3\x85\xe5@\x13\x07@\x06\xb3\x07\xe5\x02\x83 \xc1\x02\x03$\x81\x02\x83$A\x02\x03)\x01\x02\x83)\xc1\x01\xb3\xf6\xc7\x02\xb3\x86\xe6\x027\xa7\x00\x003\xd8\xc7\x02\x93\x07\x87\x00#\xa4\xb7\x00\x83\xa5\x87\x00\xb7\x05\xb0\xa8\x93\x85e\x05#$\xb7\x00\x83%\x87\x00#\xa4\xc7\x00\xb3\xd6\xc6\x02\x03\xa6\x87\x007\x06P\xea\x13\x06v\x05#$\xc7\x00\x03&\x87\x00#\xa4\xa7\x00\x03\xa6\x87\x00\x13\x05\x00\x007\x06\x00\x1d\x13\x06\x86\x05#$\xc7\x00\x03&\x87\x00#\xa4\x07\x01#\xa6\xd7\x00\x83\xa6\x87\x00\x83\xa7\xc7\x00\xb7\x07@z\x93\x87\x97\x05#$\xf7\x00\x83\'\x87\x007\xa7\x00\x00\x83\'G\x00\x93\x87\x17\x00#"\xf7\x00\x13\x01\x01\x03g\x80\x00\x00#"\x04\x02o\xf0\x9f\xef#$\x04\x02o\xf0\x1f\xef#&\x04\x02o\xf0\x9f\xee'
            instruction_bytes = read_elf.get_instruction_bytes_of_function(func_name, file_name)
            assert expected_bytes == instruction_bytes

        test_elf_obj()
        test_elf_file()

    def test_exception():
        func_name = "main"
        with pytest.raises(Exception) as exe_info:
            read_elf.get_instruction_bytes_of_function(func_name, typing.cast(typing.Any, []))
        assert "error: no method defined to get instruction from elf of type" in str(exe_info.value)

    test_exception()

def test_contains_function():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf")
        assert read_elf.contains_function("main", file_name)
        assert not read_elf.contains_function("**__ABCDEFGHIJK", file_name)

def test_add_program_counter():
    dec_ins = [decoded_instruction.decoded_instruction() for _ in range(5)]
    start = 0
    end = (len(dec_ins) * decoded_instruction.get_num_bytes_per_instruction()) + start

    read_elf.add_program_counter(start, end, dec_ins)

    assert all(hasattr(ele, 'program_counter') for ele in dec_ins)
    assert all((ele.program_counter == (idx * decoded_instruction.get_num_bytes_per_instruction())) for idx, ele in enumerate(dec_ins))

def test_add_program_counter_exception():
    dec_ins = [decoded_instruction.decoded_instruction() for _ in range(5)]
    start = 0
    end = (len(dec_ins) * decoded_instruction.get_num_bytes_per_instruction()) + start + 5

    with pytest.raises(Exception) as exe_info:
        read_elf.add_program_counter(start, end, dec_ins)
    assert "error: number of instructions mismatch" in str(exe_info.value)

def test_decode_function_from_elf_object_0():
    if TEST_WITH_ELF_FILES:
        function_name = "main"
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf")
        expected_msgs = [
            '0x0000a190: 0xfd010113 addi x2, x2, -48',
            '0x0000a194: 0x02112623 sw x1, 44(x2)',
            '0x0000a198: 0x02812423 sw x8, 40(x2)',
            '0x0000a19c: 0x02912223 sw x9, 36(x2)',
            '0x0000a1a0: 0x03212023 sw x18, 32(x2)',
            '0x0000a1a4: 0x01312e23 sw x19, 28(x2)',
            '0x0000a1a8: 0x0080f737 lui x14, 2063',
            '0x0000a1ac: 0x08072783 lw x15, 128(x14)',
            '0x0000a1b0: 0x0ff7f793 andi x15, x15, 255',
            '0x0000a1b4: 0xfe078ce3 beq x15, x0, -8',
            '0x0000a1b8: 0x008007b7 lui x15, 2048',
            '0x0000a1bc: 0xfff00713 addi x14, x0, -1',
            '0x0000a1c0: 0x10e7a823 sw x14, 272(x15)',
            '0x0000a1c4: 0x04040737 lui x14, 16448',
            '0x0000a1c8: 0x40470713 addi x14, x14, 1028',
            '0x0000a1cc: 0x04e7a223 sw x14, 68(x15)',
            '0x0000a1d0: 0x04e7a423 sw x14, 72(x15)',
            '0x0000a1d4: 0x04e7a823 sw x14, 80(x15)',
            '0x0000a1d8: 0x2607a023 sw x0, 608(x15)',
            '0x0000a1dc: 0x00820737 lui x14, 2080',
            '0x0000a1e0: 0x10f00693 addi x13, x0, 271',
            '0x0000a1e4: 0x3ad72a23 sw x13, 948(x14)',
            '0x0000a1e8: 0x0000b4b7 lui x9, 11',
            '0x0000a1ec: 0x8bc4a703 lw x14, -1860(x9)',
            '0x0000a1f0: 0x00300693 addi x13, x0, 3',
            '0x0000a1f4: 0x24e6f663 bgeu x13, x14, 588',
            '0x0000a1f8: 0x8c7006b7 lui x13, 575232',
            '0x0000a1fc: 0x0000a637 lui x12, 10',
            '0x0000a200: 0x10868693 addi x13, x13, 264',
            '0x0000a204: 0x00d62423 sw x13, 8(x12)',
            '0x0000a208: 0x00862683 lw x13, 8(x12)',
            '0x0000a20c: 0xfe000693 addi x13, x0, -32',
            '0x0000a210: 0x26d7a423 sw x13, 616(x15)',
            '0x0000a214: 0x26d7a623 sw x13, 620(x15)',
            '0x0000a218: 0x26d7a823 sw x13, 624(x15)',
            '0x0000a21c: 0x0f470713 addi x14, x14, 244',
            '0x0000a220: 0x00820437 lui x8, 2080',
            '0x0000a224: 0x028287b7 lui x15, 10280',
            '0x0000a228: 0x00271713 slli x14, x14, 2',
            '0x0000a22c: 0x00f12223 sw x15, 4(x2)',
            '0x0000a230: 0x00e40733 add x14, x8, x14',
            '0x0000a234: 0x00100913 addi x18, x0, 1',
            '0x0000a238: 0x000217b7 lui x15, 33',
            '0x0000a23c: 0x01272023 sw x18, 0(x14)',
            '0x0000a240: 0xb0578793 addi x15, x15, -1275',
            '0x0000a244: 0x00f42023 sw x15, 0(x8)',
            '0x0000a248: 0x00412783 lw x15, 4(x2)',
            '0x0000a24c: 0x00010513 addi x10, x2, 0',
            '0x0000a250: 0x00f42223 sw x15, 4(x8)',
            '0x0000a254: 0x00042423 sw x0, 8(x8)',
            '0x0000a258: 0x00012023 sw x0, 0(x2)',
            '0x0000a25c: 0x00012223 sw x0, 4(x2)',
            '0x0000a260: 0x00012423 sw x0, 8(x2)',
            '0x0000a264: 0x00011623 sh x0, 12(x2)',
            '0x0000a268: 0x338000ef jal x1, 824',
            '0x0000a26c: 0x44a42a23 sw x10, 1108(x8)',
            '0x0000a270: 0x00010513 addi x10, x2, 0',
            '0x0000a274: 0x388000ef jal x1, 904',
            '0x0000a278: 0x54a42a23 sw x10, 1364(x8)',
            '0x0000a27c: 0x40650000 zeroacc clear_mode[23:19] = 0b11, use_32_bit_mode[18:18] = 0b0, clear_zero_flags[17:17] = 0b0, addr_mode[16:14] = 0b101, where[13:0] = 0x0',
            '0x0000a280: 0x00800793 addi x15, x0, 8',
            '0x0000a284: 0x00012023 sw x0, 0(x2)',
            '0x0000a288: 0x00012223 sw x0, 4(x2)',
            '0x0000a28c: 0x00010513 addi x10, x2, 0',
            '0x0000a290: 0x00f101a3 sb x15, 3(x2)',
            '0x0000a294: 0x00f11323 sh x15, 6(x2)',
            '0x0000a298: 0x00012423 sw x0, 8(x2)',
            '0x0000a29c: 0x00011623 sh x0, 12(x2)',
            '0x0000a2a0: 0x300000ef jal x1, 768',
            '0x0000a2a4: 0x44a42023 sw x10, 1088(x8)',
            '0x0000a2a8: 0x00010513 addi x10, x2, 0',
            '0x0000a2ac: 0x350000ef jal x1, 848',
            '0x0000a2b0: 0x0000b5b7 lui x11, 11',
            '0x0000a2b4: 0x87858993 addi x19, x11, -1928',
            '0x0000a2b8: 0x00e00613 addi x12, x0, 14',
            '0x0000a2bc: 0x87858593 addi x11, x11, -1928',
            '0x0000a2c0: 0x54a42023 sw x10, 1344(x8)',
            '0x0000a2c4: 0x00010513 addi x10, x2, 0',
            '0x0000a2c8: 0x3b8000ef jal x1, 952',
            '0x0000a2cc: 0x00010513 addi x10, x2, 0',
            '0x0000a2d0: 0x2d0000ef jal x1, 720',
            '0x0000a2d4: 0x44a42a23 sw x10, 1108(x8)',
            '0x0000a2d8: 0x00010513 addi x10, x2, 0',
            '0x0000a2dc: 0x320000ef jal x1, 800',
            '0x0000a2e0: 0x01098593 addi x11, x19, 16',
            '0x0000a2e4: 0x00e00613 addi x12, x0, 14',
            '0x0000a2e8: 0x54a42a23 sw x10, 1364(x8)',
            '0x0000a2ec: 0x00010513 addi x10, x2, 0',
            '0x0000a2f0: 0x390000ef jal x1, 912',
            '0x0000a2f4: 0x00010513 addi x10, x2, 0',
            '0x0000a2f8: 0x2a8000ef jal x1, 680',
            '0x0000a2fc: 0x44a42223 sw x10, 1092(x8)',
            '0x0000a300: 0x00010513 addi x10, x2, 0',
            '0x0000a304: 0x2f8000ef jal x1, 760',
            '0x0000a308: 0x02098593 addi x11, x19, 32',
            '0x0000a30c: 0x00e00613 addi x12, x0, 14',
            '0x0000a310: 0x54a42223 sw x10, 1348(x8)',
            '0x0000a314: 0x00010513 addi x10, x2, 0',
            '0x0000a318: 0x368000ef jal x1, 872',
            '0x0000a31c: 0x00010513 addi x10, x2, 0',
            '0x0000a320: 0x280000ef jal x1, 640',
            '0x0000a324: 0x44a42423 sw x10, 1096(x8)',
            '0x0000a328: 0x00010513 addi x10, x2, 0',
            '0x0000a32c: 0x2d0000ef jal x1, 720',
            '0x0000a330: 0x00e00613 addi x12, x0, 14',
            '0x0000a334: 0x03098593 addi x11, x19, 48',
            '0x0000a338: 0x54a42423 sw x10, 1352(x8)',
            '0x0000a33c: 0x00010513 addi x10, x2, 0',
            '0x0000a340: 0x340000ef jal x1, 832',
            '0x0000a344: 0x00010513 addi x10, x2, 0',
            '0x0000a348: 0x258000ef jal x1, 600',
            '0x0000a34c: 0x44a42823 sw x10, 1104(x8)',
            '0x0000a350: 0x00010513 addi x10, x2, 0',
            '0x0000a354: 0x2a8000ef jal x1, 680',
            '0x0000a358: 0x54a42823 sw x10, 1360(x8)',
            '0x0000a35c: 0x0ff0000f fence iorw, iorw',
            '0x0000a360: 0x00100313 addi x6, x0, 1',
            '0x0000a364: 0x01231313 slli x6, x6, 18',
            '0x0000a368: 0x7c032073 csrrs x0, 0x7c0, x6',
            '0x0000a36c: 0x10000404 replay start_idx[23:14] = 0, len[13:4] = 16, last[3:3] = 0b0, set_mutex[2:2] = 0b0, execute_while_loading[1:1] = 0b0, load_mode[0:0] = 0b1',
            '0x0000a370: 0x98000000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b0, dst[13:0] = 0',
            '0x0000a374: 0x98010000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b1, dst[13:0] = 0',
            '0x0000a378: 0x98000000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b0, dst[13:0] = 0',
            '0x0000a37c: 0x98020000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b10, dst[13:0] = 0',
            '0x0000a380: 0x98000000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b0, dst[13:0] = 0',
            '0x0000a384: 0x98010000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b1, dst[13:0] = 0',
            '0x0000a388: 0x98000000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b0, dst[13:0] = 0',
            '0x0000a38c: 0x98040000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b100, dst[13:0] = 0',
            '0x0000a390: 0x98000000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b0, dst[13:0] = 0',
            '0x0000a394: 0x98010000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b1, dst[13:0] = 0',
            '0x0000a398: 0x98000000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b0, dst[13:0] = 0',
            '0x0000a39c: 0x98020000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b10, dst[13:0] = 0',
            '0x0000a3a0: 0x98000000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b0, dst[13:0] = 0',
            '0x0000a3a4: 0x98010000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b1, dst[13:0] = 0',
            '0x0000a3a8: 0x98000000 mvmul clear_dvalid[23:22] = 0b0, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b0, dst[13:0] = 0',
            '0x0000a3ac: 0x99050000 mvmul clear_dvalid[23:22] = 0b1, instr_mod19[21:19] = 0, addr_mode[18:14] = 0b101, dst[13:0] = 0',
            '0x0000a3b0: 0x00100313 addi x6, x0, 1',
            '0x0000a3b4: 0x01231313 slli x6, x6, 18',
            '0x0000a3b8: 0x7c033073 csrrc x0, 0x7c0, x6',
            '0x0000a3bc: 0xdeadc7b7 lui x15, 912092',
            '0x0000a3c0: 0xeef78793 addi x15, x15, -273',
            '0x0000a3c4: 0x00f12023 sw x15, 0(x2)',
            '0x0000a3c8: 0x00012703 lw x14, 0(x2)',
            '0x0000a3cc: 0x0080f7b7 lui x15, 2063',
            '0x0000a3d0: 0x00200613 addi x12, x0, 2',
            '0x0000a3d4: 0x00e7a423 sw x14, 8(x15)',
            '0x0000a3d8: 0x0087a783 lw x15, 8(x15)',
            '0x0000a3dc: 0x020006b7 lui x13, 8192',
            '0x0000a3e0: 0x04000737 lui x14, 16384',
            '0x0000a3e4: 0x00f12023 sw x15, 0(x2)',
            '0x0000a3e8: 0x0080d7b7 lui x15, 2061',
            '0x0000a3ec: 0x08c7a023 sw x12, 128(x15)',
            '0x0000a3f0: 0x0127a023 sw x18, 0(x15)',
            '0x0000a3f4: 0x0127a223 sw x18, 4(x15)',
            '0x0000a3f8: 0x00d7a423 sw x13, 8(x15)',
            '0x0000a3fc: 0x00d7a623 sw x13, 12(x15)',
            '0x0000a400: 0x00d7a823 sw x13, 16(x15)',
            '0x0000a404: 0x10070713 addi x14, x14, 256',
            '0x0000a408: 0x00e7aa23 sw x14, 20(x15)',
            '0x0000a40c: 0x00d7ac23 sw x13, 24(x15)',
            '0x0000a410: 0x00e7ae23 sw x14, 28(x15)',
            '0x0000a414: 0x02e7a023 sw x14, 32(x15)',
            '0x0000a418: 0xdc00003c setrwc clear_ab_vld[23:22] = 0b0, rwc_cr[21:18] = 0b0, rwc_val[17:6] = 0, BitMask[5:0] = 0b1111',
            '0x0000a41c: 0x008007b7 lui x15, 2048',
            '0x0000a420: 0x0c47a703 lw x14, 196(x15)',
            '0x0000a424: 0x0327a023 sw x18, 32(x15)',
            '0x0000a428: 0x8bc4a783 lw x15, -1860(x9)',
            '0x0000a42c: 0x14c78063 beq x15, x12, 320',
            '0x0000a430: 0x02f66463 bltu x12, x15, 40',
            '0x0000a434: 0x12079863 bne x15, x0, 304',
            '0x0000a438: 0x02042023 sw x0, 32(x8)',
            '0x0000a43c: 0x0240006f jal x0, 36',
            '0x0000a440: 0x09870693 addi x13, x14, 152',
            '0x0000a444: 0x00269693 slli x13, x13, 2',
            '0x0000a448: 0x00d787b3 add x15, x15, x13',
            '0x0000a44c: 0xfe000693 addi x13, x0, -32',
            '0x0000a450: 0x00d7a423 sw x13, 8(x15)',
            '0x0000a454: 0xdc9ff06f jal x0, -568',
            '0x0000a458: 0x00300693 addi x13, x0, 3',
            '0x0000a45c: 0x10d78c63 beq x15, x13, 280',
            '0x0000a460: 0x06000000 mop mop_type[23:23] = 0b1, done[22:22] = 0b0, loop_count[21:15] = 0x0, zmask_lo8_or_loop_count[14:0] = 0x0',
            '0x0000a464: 0xde00003c setrwc clear_ab_vld[23:22] = 0b10, rwc_cr[21:18] = 0b0, rwc_val[17:6] = 0, BitMask[5:0] = 0b1111',
            '0x0000a468: 0x88800056 stallwait stall_res[23:15] = 0x40, wait_res_idx_2[14:10] = 0, wait_res_idx_1[9:5] = 0, wait_res_idx_0[4:0] = 21',
            '0x0000a46c: 0xd8000020 cleardvalid cleardvalid[23:22] = 0b0, cleardvalid_S[21:20] = 0b0, dest_dvalid_reset[13:10] = 0b0, dest_dvalid_client_bank_reset[9:6] = 0b0, dest_pulse_last[5:2] = 0b10, reset[1:0] = 0b0',
            '0x0000a470: 0x0ff0000f fence iorw, iorw',
            '0x0000a474: 0xbc1027f3 csrrs x15, 0xbc1, x0',
            '0x0000a478: 0x0027f793 andi x15, x15, 2',
            '0x0000a47c: 0xfe079ae3 bne x15, x0, -12',
            '0x0000a480: 0x0ff0000f fence iorw, iorw',
            '0x0000a484: 0xbc1027f3 csrrs x15, 0xbc1, x0',
            '0x0000a488: 0x4007f793 andi x15, x15, 1024',
            '0x0000a48c: 0xfe079ae3 bne x15, x0, -12',
            '0x0000a490: 0x008007b7 lui x15, 2048',
            '0x0000a494: 0x00200693 addi x13, x0, 2',
            '0x0000a498: 0x02d7a023 sw x13, 32(x15)',
            '0x0000a49c: 0x0c47a583 lw x11, 196(x15)',
            '0x0000a4a0: 0x0a87a603 lw x12, 168(x15)',
            '0x0000a4a4: 0x0ac7a503 lw x10, 172(x15)',
            '0x0000a4a8: 0x40e585b3 sub x11, x11, x14',
            '0x0000a4ac: 0x06400713 addi x14, x0, 100',
            '0x0000a4b0: 0x02e507b3 mul x15, x10, x14',
            '0x0000a4b4: 0x02c12083 lw x1, 44(x2)',
            '0x0000a4b8: 0x02812403 lw x8, 40(x2)',
            '0x0000a4bc: 0x02412483 lw x9, 36(x2)',
            '0x0000a4c0: 0x02012903 lw x18, 32(x2)',
            '0x0000a4c4: 0x01c12983 lw x19, 28(x2)',
            '0x0000a4c8: 0x02c7f6b3 remu x13, x15, x12',
            '0x0000a4cc: 0x02e686b3 mul x13, x13, x14',
            '0x0000a4d0: 0x0000a737 lui x14, 10',
            '0x0000a4d4: 0x02c7d833 divu x16, x15, x12',
            '0x0000a4d8: 0x00870793 addi x15, x14, 8',
            '0x0000a4dc: 0x00b7a423 sw x11, 8(x15)',
            '0x0000a4e0: 0x0087a583 lw x11, 8(x15)',
            '0x0000a4e4: 0xa8b005b7 lui x11, 690944',
            '0x0000a4e8: 0x05658593 addi x11, x11, 86',
            '0x0000a4ec: 0x00b72423 sw x11, 8(x14)',
            '0x0000a4f0: 0x00872583 lw x11, 8(x14)',
            '0x0000a4f4: 0x00c7a423 sw x12, 8(x15)',
            '0x0000a4f8: 0x02c6d6b3 divu x13, x13, x12',
            '0x0000a4fc: 0x0087a603 lw x12, 8(x15)',
            '0x0000a500: 0xea500637 lui x12, 959744',
            '0x0000a504: 0x05760613 addi x12, x12, 87',
            '0x0000a508: 0x00c72423 sw x12, 8(x14)',
            '0x0000a50c: 0x00872603 lw x12, 8(x14)',
            '0x0000a510: 0x00a7a423 sw x10, 8(x15)',
            '0x0000a514: 0x0087a603 lw x12, 8(x15)',
            '0x0000a518: 0x00000513 addi x10, x0, 0',
            '0x0000a51c: 0x1d000637 lui x12, 118784',
            '0x0000a520: 0x05860613 addi x12, x12, 88',
            '0x0000a524: 0x00c72423 sw x12, 8(x14)',
            '0x0000a528: 0x00872603 lw x12, 8(x14)',
            '0x0000a52c: 0x0107a423 sw x16, 8(x15)',
            '0x0000a530: 0x00d7a623 sw x13, 12(x15)',
            '0x0000a534: 0x0087a683 lw x13, 8(x15)',
            '0x0000a538: 0x00c7a783 lw x15, 12(x15)',
            '0x0000a53c: 0x7a4007b7 lui x15, 500736',
            '0x0000a540: 0x05978793 addi x15, x15, 89',
            '0x0000a544: 0x00f72423 sw x15, 8(x14)',
            '0x0000a548: 0x00872783 lw x15, 8(x14)',
            '0x0000a54c: 0x0000a737 lui x14, 10',
            '0x0000a550: 0x00472783 lw x15, 4(x14)',
            '0x0000a554: 0x00178793 addi x15, x15, 1',
            '0x0000a558: 0x00f72223 sw x15, 4(x14)',
            '0x0000a55c: 0x03010113 addi x2, x2, 48',
            '0x0000a560: 0x00008067 jalr x0, 0(x1)',
            '0x0000a564: 0x02042223 sw x0, 36(x8)',
            '0x0000a568: 0xef9ff06f jal x0, -264',
            '0x0000a56c: 0x02042423 sw x0, 40(x8)',
            '0x0000a570: 0xef1ff06f jal x0, -272',
            '0x0000a574: 0x02042623 sw x0, 44(x8)',
            '0x0000a578: 0xee9ff06f jal x0, -280']

        with open(file_name,'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            dec_ins = read_elf.decode_function_from_elf_object(function_name, elf)
            msgs = read_elf.instructions.instruction_to_str(dec_ins)
            assert expected_msgs == msgs

def test_decode_function_from_elf_object_1():
    if TEST_WITH_ELF_FILES:
        function_name = "ABCDEF"
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf")
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            with pytest.raises(Exception) as exe_info:
                read_elf.decode_function_from_elf_object(function_name, elf) # incorrect function name

def test_decode_function_arg_file_name():
    if TEST_WITH_ELF_FILES:
        function_name = "main"
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf")
        expected_msgs = [
            '0x00006190: 0xf6010113 addi x2, x2, -160',
            '0x00006194: 0x08112e23 sw x1, 156(x2)',
            '0x00006198: 0x08812c23 sw x8, 152(x2)',
            '0x0000619c: 0x08912a23 sw x9, 148(x2)',
            '0x000061a0: 0x09212823 sw x18, 144(x2)',
            '0x000061a4: 0x09312623 sw x19, 140(x2)',
            '0x000061a8: 0x09412423 sw x20, 136(x2)',
            '0x000061ac: 0x09512223 sw x21, 132(x2)',
            '0x000061b0: 0x09612023 sw x22, 128(x2)',
            '0x000061b4: 0x07712e23 sw x23, 124(x2)',
            '0x000061b8: 0x07812c23 sw x24, 120(x2)',
            '0x000061bc: 0x07912a23 sw x25, 116(x2)',
            '0x000061c0: 0x07a12823 sw x26, 112(x2)',
            '0x000061c4: 0x07b12623 sw x27, 108(x2)',
            '0x000061c8: 0x008007b7 lui x15, 2048',
            '0x000061cc: 0x00500713 addi x14, x0, 5',
            '0x000061d0: 0x26e7a023 sw x14, 608(x15)',
            '0x000061d4: 0x00094737 lui x14, 148',
            '0x000061d8: 0xd0c70713 addi x14, x14, -756',
            '0x000061dc: 0x10e7a823 sw x14, 272(x15)',
            '0x000061e0: 0x3a187737 lui x14, 237959',
            '0x000061e4: 0xb0e70713 addi x14, x14, -1266',
            '0x000061e8: 0x04e7a223 sw x14, 68(x15)',
            '0x000061ec: 0x52498737 lui x14, 337048',
            '0x000061f0: 0xe7370713 addi x14, x14, -397',
            '0x000061f4: 0x04e7a423 sw x14, 72(x15)',
            '0x000061f8: 0x666e2737 lui x14, 419554',
            '0x000061fc: 0xd2770713 addi x14, x14, -729',
            '0x00006200: 0x04e7a823 sw x14, 80(x15)',
            '0x00006204: 0x8c440006 seminit max_value[23:20] = 1, init_value[19:16] = 1, sem_bank_sel[12:8] = 0b0, sem_sel[7:0] = 1',
            '0x00006208: 0x0ff0000f fence iorw, iorw',
            '0x0000620c: 0xbc1027f3 csrrs x15, 0xbc1, x0',
            '0x00006210: 0x0807f793 andi x15, x15, 128',
            '0x00006214: 0xfe079ae3 bne x15, x0, -12',
            '0x00006218: 0x008009b7 lui x19, 2048',
            '0x0000621c: 0xfff00793 addi x15, x0, -1',
            '0x00006220: 0x10f9a823 sw x15, 272(x19)',
            '0x00006224: 0x040407b7 lui x15, 16448',
            '0x00006228: 0x40478793 addi x15, x15, 1028',
            '0x0000622c: 0x04f9a223 sw x15, 68(x19)',
            '0x00006230: 0x04f9a423 sw x15, 72(x19)',
            '0x00006234: 0x04f9a823 sw x15, 80(x19)',
            '0x00006238: 0x2609a023 sw x0, 608(x19)',
            '0x0000623c: 0x40000593 addi x11, x0, 1024',
            '0x00006240: 0x00052537 lui x10, 82',
            '0x00006244: 0x44c000ef jal x1, 1100',
            '0x00006248: 0x00505737 lui x14, 1285',
            '0x0000624c: 0x20070793 addi x15, x14, 512',
            '0x00006250: 0x00050413 addi x8, x10, 0',
            '0x00006254: 0x010008b7 lui x17, 4096',
            '0x00006258: 0x41000813 addi x16, x0, 1040',
            '0x0000625c: 0x01010593 addi x11, x2, 16',
            '0x00006260: 0x01000613 addi x12, x0, 16',
            '0x00006264: 0x03010513 addi x10, x2, 48',
            '0x00006268: 0x00f12823 sw x15, 16(x2)',
            '0x0000626c: 0x01112a23 sw x17, 20(x2)',
            '0x00006270: 0x01012c23 sw x16, 24(x2)',
            '0x00006274: 0x00005b37 lui x22, 5',
            '0x00006278: 0x468000ef jal x1, 1128',
            '0x0000627c: 0x00445513 srli x10, x8, 4',
            '0x00006280: 0x200b0793 addi x15, x22, 512',
            '0x00006284: 0x00802db7 lui x27, 2050',
            '0x00006288: 0x000064b7 lui x9, 6',
            '0x0000628c: 0x000d8913 addi x18, x27, 0',
            '0x00006290: 0x00f50633 add x12, x10, x15',
            '0x00006294: 0x00500693 addi x13, x0, 5',
            '0x00006298: 0x00848413 addi x8, x9, 8',
            '0x0000629c: 0x04d12223 sw x13, 68(x2)',
            '0x000062a0: 0x00c92223 sw x12, 4(x18)',
            '0x000062a4: 0x00a92023 sw x10, 0(x18)',
            '0x000062a8: 0x00a92423 sw x10, 8(x18)',
            '0x000062ac: 0x04012023 sw x0, 64(x2)',
            '0x000062b0: 0x00f92823 sw x15, 16(x18)',
            '0x000062b4: 0x00092c23 sw x0, 24(x18)',
            '0x000062b8: 0x00042423 sw x0, 8(x8)',
            '0x000062bc: 0x00842583 lw x11, 8(x8)',
            '0x000062c0: 0xcf100d37 lui x26, 848128',
            '0x000062c4: 0x04ed0593 addi x11, x26, 78',
            '0x000062c8: 0x00b4a423 sw x11, 8(x9)',
            '0x000062cc: 0x0084a583 lw x11, 8(x9)',
            '0x000062d0: 0x01492583 lw x11, 20(x18)',
            '0x000062d4: 0x6f900cb7 lui x25, 456960',
            '0x000062d8: 0x72a00c37 lui x24, 469504',
            '0x000062dc: 0x00b42423 sw x11, 8(x8)',
            '0x000062e0: 0x00f42623 sw x15, 12(x8)',
            '0x000062e4: 0x00842783 lw x15, 8(x8)',
            '0x000062e8: 0x00c42783 lw x15, 12(x8)',
            '0x000062ec: 0xaa600bb7 lui x23, 697856',
            '0x000062f0: 0x04fc8793 addi x15, x25, 79',
            '0x000062f4: 0x00f4a423 sw x15, 8(x9)',
            '0x000062f8: 0x0084a783 lw x15, 8(x9)',
            '0x000062fc: 0x00c42423 sw x12, 8(x8)',
            '0x00006300: 0x00a42623 sw x10, 12(x8)',
            '0x00006304: 0x00842783 lw x15, 8(x8)',
            '0x00006308: 0x00c42783 lw x15, 12(x8)',
            '0x0000630c: 0x0080ba37 lui x20, 2059',
            '0x00006310: 0x050c0793 addi x15, x24, 80',
            '0x00006314: 0x00f4a423 sw x15, 8(x9)',
            '0x00006318: 0x0084a783 lw x15, 8(x9)',
            '0x0000631c: 0x00c92783 lw x15, 12(x18)',
            '0x00006320: 0x00a42423 sw x10, 8(x8)',
            '0x00006324: 0x00100a93 addi x21, x0, 1',
            '0x00006328: 0x00f42623 sw x15, 12(x8)',
            '0x0000632c: 0x00842783 lw x15, 8(x8)',
            '0x00006330: 0x00c42783 lw x15, 12(x8)',
            '0x00006334: 0x00505737 lui x14, 1285',
            '0x00006338: 0x051b8793 addi x15, x23, 81',
            '0x0000633c: 0x00f4a423 sw x15, 8(x9)',
            '0x00006340: 0x0084a783 lw x15, 8(x9)',
            '0x00006344: 0x015a2823 sw x21, 16(x20)',
            '0x00006348: 0x000a2223 sw x0, 4(x20)',
            '0x0000634c: 0x28070613 addi x12, x14, 640',
            '0x00006350: 0x015a2423 sw x21, 8(x20)',
            '0x00006354: 0x010008b7 lui x17, 4096',
            '0x00006358: 0x41000813 addi x16, x0, 1040',
            '0x0000635c: 0x02c12023 sw x12, 32(x2)',
            '0x00006360: 0x02010593 addi x11, x2, 32',
            '0x00006364: 0x01000613 addi x12, x0, 16',
            '0x00006368: 0x04810513 addi x10, x2, 72',
            '0x0000636c: 0x03112223 sw x17, 36(x2)',
            '0x00006370: 0x03012423 sw x16, 40(x2)',
            '0x00006374: 0x36c000ef jal x1, 876',
            '0x00006378: 0x00053537 lui x10, 83',
            '0x0000637c: 0x00500693 addi x13, x0, 5',
            '0x00006380: 0x40000593 addi x11, x0, 1024',
            '0x00006384: 0x80050513 addi x10, x10, -2048',
            '0x00006388: 0x04d12e23 sw x13, 92(x2)',
            '0x0000638c: 0x05512c23 sw x21, 88(x2)',
            '0x00006390: 0x300000ef jal x1, 768',
            '0x00006394: 0x00455513 srli x10, x10, 4',
            '0x00006398: 0x280b0713 addi x14, x22, 640',
            '0x0000639c: 0x00e507b3 add x15, x10, x14',
            '0x000063a0: 0x02f92023 sw x15, 32(x18)',
            '0x000063a4: 0x02e92623 sw x14, 44(x18)',
            '0x000063a8: 0x00a92e23 sw x10, 28(x18)',
            '0x000063ac: 0x02092a23 sw x0, 52(x18)',
            '0x000063b0: 0x02a92223 sw x10, 36(x18)',
            '0x000063b4: 0x01542423 sw x21, 8(x8)',
            '0x000063b8: 0x078d0613 addi x12, x26, 120',
            '0x000063bc: 0x00842683 lw x13, 8(x8)',
            '0x000063c0: 0x00c4a423 sw x12, 8(x9)',
            '0x000063c4: 0x0084a683 lw x13, 8(x9)',
            '0x000063c8: 0x03092683 lw x13, 48(x18)',
            '0x000063cc: 0x079c8613 addi x12, x25, 121',
            '0x000063d0: 0x00d42423 sw x13, 8(x8)',
            '0x000063d4: 0x00e42623 sw x14, 12(x8)',
            '0x000063d8: 0x00842703 lw x14, 8(x8)',
            '0x000063dc: 0x00c42703 lw x14, 12(x8)',
            '0x000063e0: 0x00c4a423 sw x12, 8(x9)',
            '0x000063e4: 0x0084a703 lw x14, 8(x9)',
            '0x000063e8: 0x00f42423 sw x15, 8(x8)',
            '0x000063ec: 0x00a42623 sw x10, 12(x8)',
            '0x000063f0: 0x00842783 lw x15, 8(x8)',
            '0x000063f4: 0x07ac0693 addi x13, x24, 122',
            '0x000063f8: 0x00c42783 lw x15, 12(x8)',
            '0x000063fc: 0x00d4a423 sw x13, 8(x9)',
            '0x00006400: 0x0084a783 lw x15, 8(x9)',
            '0x00006404: 0x02892783 lw x15, 40(x18)',
            '0x00006408: 0x00a42423 sw x10, 8(x8)',
            '0x0000640c: 0x07bb8713 addi x14, x23, 123',
            '0x00006410: 0x00f42623 sw x15, 12(x8)',
            '0x00006414: 0x00842783 lw x15, 8(x8)',
            '0x00006418: 0x00c42783 lw x15, 12(x8)',
            '0x0000641c: 0x00e4a423 sw x14, 8(x9)',
            '0x00006420: 0x0084a783 lw x15, 8(x9)',
            '0x00006424: 0x035a2823 sw x21, 48(x20)',
            '0x00006428: 0x020a2223 sw x0, 36(x20)',
            '0x0000642c: 0x035a2423 sw x21, 40(x20)',
            '0x00006430: 0x4400001c zerosrc zero_val[4:4] = 0b0, write_mode[3:3] = 0, bank_mask[2:2] = 1, src_mask[1:0] = 3',
            '0x00006434: 0x000077b7 lui x15, 7',
            '0x00006438: 0x8dc7a783 lw x15, -1828(x15)',
            '0x0000643c: 0x00300713 addi x14, x0, 3',
            '0x00006440: 0x000d8d93 addi x27, x27, 0',
            '0x00006444: 0x1ef77263 bgeu x14, x15, 484',
            '0x00006448: 0x8c7007b7 lui x15, 575232',
            '0x0000644c: 0x10878793 addi x15, x15, 264',
            '0x00006450: 0x00f4a423 sw x15, 8(x9)',
            '0x00006454: 0x0084a783 lw x15, 8(x9)',
            '0x00006458: 0xfe000793 addi x15, x0, -32',
            '0x0000645c: 0x26f9a423 sw x15, 616(x19)',
            '0x00006460: 0x26f9a623 sw x15, 620(x19)',
            '0x00006464: 0x26f9a823 sw x15, 624(x19)',
            '0x00006468: 0x03010513 addi x10, x2, 48',
            '0x0000646c: 0x1f4000ef jal x1, 500',
            '0x00006470: 0x04414783 lbu x15, 68(x2)',
            '0x00006474: 0xb20d0737 lui x14, 729296',
            '0x00006478: 0xf0070713 addi x14, x14, -256',
            '0x0000647c: 0x0080e4b7 lui x9, 2062',
            '0x00006480: 0x00e787b3 add x15, x15, x14',
            '0x00006484: 0x00f4a023 sw x15, 0(x9)',
            '0x00006488: 0x04810513 addi x10, x2, 72',
            '0x0000648c: 0x1d4000ef jal x1, 468',
            '0x00006490: 0x05c14783 lbu x15, 92(x2)',
            '0x00006494: 0xb2190737 lui x14, 729488',
            '0x00006498: 0xf0070713 addi x14, x14, -256',
            '0x0000649c: 0x00e787b3 add x15, x15, x14',
            '0x000064a0: 0x00f4a023 sw x15, 0(x9)',
            '0x000064a4: 0xb40c07b7 lui x15, 737472',
            '0x000064a8: 0x10078793 addi x15, x15, 256',
            '0x000064ac: 0x00f4a023 sw x15, 0(x9)',
            '0x000064b0: 0xdeadc7b7 lui x15, 912092',
            '0x000064b4: 0xeef78793 addi x15, x15, -273',
            '0x000064b8: 0x00f12623 sw x15, 12(x2)',
            '0x000064bc: 0x00c12703 lw x14, 12(x2)',
            '0x000064c0: 0x0080f7b7 lui x15, 2063',
            '0x000064c4: 0x00e7a423 sw x14, 8(x15)',
            '0x000064c8: 0x0087a783 lw x15, 8(x15)',
            '0x000064cc: 0x00200713 addi x14, x0, 2',
            '0x000064d0: 0x00f12623 sw x15, 12(x2)',
            '0x000064d4: 0x0080d7b7 lui x15, 2061',
            '0x000064d8: 0x08e7a023 sw x14, 128(x15)',
            '0x000064dc: 0x00100713 addi x14, x0, 1',
            '0x000064e0: 0x00e7a023 sw x14, 0(x15)',
            '0x000064e4: 0x00e7a223 sw x14, 4(x15)',
            '0x000064e8: 0x69000737 lui x14, 430080',
            '0x000064ec: 0x00270713 addi x14, x14, 2',
            '0x000064f0: 0x00e7a423 sw x14, 8(x15)',
            '0x000064f4: 0x02000737 lui x14, 8192',
            '0x000064f8: 0x00e7a623 sw x14, 12(x15)',
            '0x000064fc: 0x00e7a823 sw x14, 16(x15)',
            '0x00006500: 0x44000737 lui x14, 278528',
            '0x00006504: 0x00670713 addi x14, x14, 6',
            '0x00006508: 0x00e7aa23 sw x14, 20(x15)',
            '0x0000650c: 0x07000737 lui x14, 28672',
            '0x00006510: 0x00170713 addi x14, x14, 1',
            '0x00006514: 0x00e7ac23 sw x14, 24(x15)',
            '0x00006518: 0x00e7ae23 sw x14, 28(x15)',
            '0x0000651c: 0x02e7a023 sw x14, 32(x15)',
            '0x00006520: 0xa90407b7 lui x15, 692288',
            '0x00006524: 0x02078713 addi x14, x15, 32',
            '0x00006528: 0x00e4a023 sw x14, 0(x9)',
            '0x0000652c: 0x02178793 addi x15, x15, 33',
            '0x00006530: 0x00f4a023 sw x15, 0(x9)',
            '0x00006534: 0x34100000 set_dst_tile_face_row_idx Tile_Face_Row_Sel[22:21] = 0b0, EngineSel[20:18] = 0b1, Value[17:0] = 0',
            '0x00006538: 0x060407b7 lui x15, 24640',
            '0x0000653c: 0x00f4a023 sw x15, 0(x9)',
            '0x00006540: 0x060007b7 lui x15, 24576',
            '0x00006544: 0x00f4a023 sw x15, 0(x9)',
            '0x00006548: 0x06000000 mop mop_type[23:23] = 0b1, done[22:22] = 0b0, loop_count[21:15] = 0x0, zmask_lo8_or_loop_count[14:0] = 0x0',
            '0x0000654c: 0x3e0387b7 lui x15, 254008',
            '0x00006550: 0x02078793 addi x15, x15, 32',
            '0x00006554: 0x00f4a023 sw x15, 0(x9)',
            '0x00006558: 0x018dd783 lhu x15, 24(x27)',
            '0x0000655c: 0x010da703 lw x14, 16(x27)',
            '0x00006560: 0x00178793 addi x15, x15, 1',
            '0x00006564: 0x00fd9c23 sh x15, 24(x27)',
            '0x00006568: 0x008da783 lw x15, 8(x27)',
            '0x0000656c: 0x00e787b3 add x15, x15, x14',
            '0x00006570: 0x004da703 lw x14, 4(x27)',
            '0x00006574: 0x00fda823 sw x15, 16(x27)',
            '0x00006578: 0x00e7e863 bltu x15, x14, 16',
            '0x0000657c: 0x000da703 lw x14, 0(x27)',
            '0x00006580: 0x40e787b3 sub x15, x15, x14',
            '0x00006584: 0x00fda823 sw x15, 16(x27)',
            '0x00006588: 0x3e0387b7 lui x15, 254008',
            '0x0000658c: 0x0080e737 lui x14, 2062',
            '0x00006590: 0x02178793 addi x15, x15, 33',
            '0x00006594: 0x00f72023 sw x15, 0(x14)',
            '0x00006598: 0x034dd783 lhu x15, 52(x27)',
            '0x0000659c: 0x02cda703 lw x14, 44(x27)',
            '0x000065a0: 0x00178793 addi x15, x15, 1',
            '0x000065a4: 0x02fd9a23 sh x15, 52(x27)',
            '0x000065a8: 0x024da783 lw x15, 36(x27)',
            '0x000065ac: 0x00e787b3 add x15, x15, x14',
            '0x000065b0: 0x020da703 lw x14, 32(x27)',
            '0x000065b4: 0x02fda623 sw x15, 44(x27)',
            '0x000065b8: 0x00e7e863 bltu x15, x14, 16',
            '0x000065bc: 0x01cda703 lw x14, 28(x27)',
            '0x000065c0: 0x40e787b3 sub x15, x15, x14',
            '0x000065c4: 0x02fda623 sw x15, 44(x27)',
            '0x000065c8: 0x0ff0000f fence iorw, iorw',
            '0x000065cc: 0xbc1027f3 csrrs x15, 0xbc1, x0',
            '0x000065d0: 0x0107f793 andi x15, x15, 16',
            '0x000065d4: 0xfe079ae3 bne x15, x0, -12',
            '0x000065d8: 0x00006737 lui x14, 6',
            '0x000065dc: 0x00472783 lw x15, 4(x14)',
            '0x000065e0: 0x09c12083 lw x1, 156(x2)',
            '0x000065e4: 0x09812403 lw x8, 152(x2)',
            '0x000065e8: 0x00178793 addi x15, x15, 1',
            '0x000065ec: 0x00f72223 sw x15, 4(x14)',
            '0x000065f0: 0x09412483 lw x9, 148(x2)',
            '0x000065f4: 0x09012903 lw x18, 144(x2)',
            '0x000065f8: 0x08c12983 lw x19, 140(x2)',
            '0x000065fc: 0x08812a03 lw x20, 136(x2)',
            '0x00006600: 0x08412a83 lw x21, 132(x2)',
            '0x00006604: 0x08012b03 lw x22, 128(x2)',
            '0x00006608: 0x07c12b83 lw x23, 124(x2)',
            '0x0000660c: 0x07812c03 lw x24, 120(x2)',
            '0x00006610: 0x07412c83 lw x25, 116(x2)',
            '0x00006614: 0x07012d03 lw x26, 112(x2)',
            '0x00006618: 0x06c12d83 lw x27, 108(x2)',
            '0x0000661c: 0x00000513 addi x10, x0, 0',
            '0x00006620: 0x0a010113 addi x2, x2, 160',
            '0x00006624: 0x00008067 jalr x0, 0(x1)',
            '0x00006628: 0x09878793 addi x15, x15, 152',
            '0x0000662c: 0x00279793 slli x15, x15, 2',
            '0x00006630: 0x00f989b3 add x19, x19, x15',
            '0x00006634: 0xfe000793 addi x15, x0, -32',
            '0x00006638: 0x00f9a423 sw x15, 8(x19)',
            '0x0000663c: 0xe2dff06f jal x0, -468',
        ]
        instruction_sets: dict[decoded_instruction.instruction_kind, typing.Any] = dict()
        instruction_sets[decoded_instruction.instruction_kind.ttqs] = os.path.join(os.path.dirname(__file__), "../../ttsim/config/llk/instruction_sets/ttqs/assembly.mar18.yaml")
        assert expected_msgs == read_elf.instructions.instruction_to_str(read_elf.decode_function(function_name, file_name, sets = instruction_sets), instruction_set = instruction_sets)

@pytest.mark.slow
def test_decode_function_arg_elf():
    if TEST_WITH_ELF_FILES:
        function_name = "main"
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_2/out/thread_2.elf")

        expected_msgs = [
            '0x0000e190: 0xfc010113 addi x2, x2, -64',
            '0x0000e194: 0x02112e23 sw x1, 60(x2)',
            '0x0000e198: 0x02812c23 sw x8, 56(x2)',
            '0x0000e19c: 0x0080f737 lui x14, 2063',
            '0x0000e1a0: 0x08072783 lw x15, 128(x14)',
            '0x0000e1a4: 0x0ff7f793 andi x15, x15, 255',
            '0x0000e1a8: 0xfe078ce3 beq x15, x0, -8',
            '0x0000e1ac: 0x00800437 lui x8, 2048',
            '0x0000e1b0: 0xfff00793 addi x15, x0, -1',
            '0x0000e1b4: 0x10f42823 sw x15, 272(x8)',
            '0x0000e1b8: 0x040407b7 lui x15, 16448',
            '0x0000e1bc: 0x40478793 addi x15, x15, 1028',
            '0x0000e1c0: 0x04f42223 sw x15, 68(x8)',
            '0x0000e1c4: 0x04f42423 sw x15, 72(x8)',
            '0x0000e1c8: 0x04f42823 sw x15, 80(x8)',
            '0x0000e1cc: 0x26042023 sw x0, 608(x8)',
            '0x0000e1d0: 0x11100713 addi x14, x0, 273',
            '0x0000e1d4: 0x008207b7 lui x15, 2080',
            '0x0000e1d8: 0x3ae7ae23 sw x14, 956(x15)',
            '0x0000e1dc: 0x005057b7 lui x15, 1285',
            '0x0000e1e0: 0x30078793 addi x15, x15, 768',
            '0x0000e1e4: 0x00f12423 sw x15, 8(x2)',
            '0x0000e1e8: 0x010007b7 lui x15, 4096',
            '0x0000e1ec: 0x00f12623 sw x15, 12(x2)',
            '0x0000e1f0: 0x01000613 addi x12, x0, 16',
            '0x0000e1f4: 0x41000793 addi x15, x0, 1040',
            '0x0000e1f8: 0x00810593 addi x11, x2, 8',
            '0x0000e1fc: 0x01810513 addi x10, x2, 24',
            '0x0000e200: 0x00f12823 sw x15, 16(x2)',
            '0x0000e204: 0x2f8000ef jal x1, 760',
            '0x0000e208: 0x008066b7 lui x13, 2054',
            '0x0000e20c: 0x000055b7 lui x11, 5',
            '0x0000e210: 0x0000e737 lui x14, 14',
            '0x0000e214: 0x00068613 addi x12, x13, 0',
            '0x0000e218: 0x08000513 addi x10, x0, 128',
            '0x0000e21c: 0x38058313 addi x6, x11, 896',
            '0x0000e220: 0x00100813 addi x16, x0, 1',
            '0x0000e224: 0x30058593 addi x11, x11, 768',
            '0x0000e228: 0x0ea62023 sw x10, 224(x12)',
            '0x0000e22c: 0x0ea62423 sw x10, 232(x12)',
            '0x0000e230: 0x0e662223 sw x6, 228(x12)',
            '0x0000e234: 0x0eb62a23 sw x11, 244(x12)',
            '0x0000e238: 0x0e062c23 sw x0, 248(x12)',
            '0x0000e23c: 0x0f062623 sw x16, 236(x12)',
            '0x0000e240: 0x00870793 addi x15, x14, 8',
            '0x0000e244: 0x00800893 addi x17, x0, 8',
            '0x0000e248: 0x0117a423 sw x17, 8(x15)',
            '0x0000e24c: 0x0087a883 lw x17, 8(x15)',
            '0x0000e250: 0x0f062603 lw x12, 240(x12)',
            '0x0000e254: 0x70e008b7 lui x17, 462336',
            '0x0000e258: 0x05088893 addi x17, x17, 80',
            '0x0000e25c: 0x01172423 sw x17, 8(x14)',
            '0x0000e260: 0x00872883 lw x17, 8(x14)',
            '0x0000e264: 0x00b7a423 sw x11, 8(x15)',
            '0x0000e268: 0x00c7a623 sw x12, 12(x15)',
            '0x0000e26c: 0x0087a603 lw x12, 8(x15)',
            '0x0000e270: 0x00c7a603 lw x12, 12(x15)',
            '0x0000e274: 0x00068693 addi x13, x13, 0',
            '0x0000e278: 0xab600637 lui x12, 701952',
            '0x0000e27c: 0x05160613 addi x12, x12, 81',
            '0x0000e280: 0x00c72423 sw x12, 8(x14)',
            '0x0000e284: 0x00872603 lw x12, 8(x14)',
            '0x0000e288: 0x0067a423 sw x6, 8(x15)',
            '0x0000e28c: 0x00a7a623 sw x10, 12(x15)',
            '0x0000e290: 0x0087a603 lw x12, 8(x15)',
            '0x0000e294: 0x00c7a603 lw x12, 12(x15)',
            '0x0000e298: 0xc0700637 lui x12, 788224',
            '0x0000e29c: 0x05260613 addi x12, x12, 82',
            '0x0000e2a0: 0x00c72423 sw x12, 8(x14)',
            '0x0000e2a4: 0x00872603 lw x12, 8(x14)',
            '0x0000e2a8: 0x00a7a423 sw x10, 8(x15)',
            '0x0000e2ac: 0x0107a623 sw x16, 12(x15)',
            '0x0000e2b0: 0x0087a603 lw x12, 8(x15)',
            '0x0000e2b4: 0x00c7a783 lw x15, 12(x15)',
            '0x0000e2b8: 0x00300613 addi x12, x0, 3',
            '0x0000e2bc: 0x823007b7 lui x15, 533248',
            '0x0000e2c0: 0x05378793 addi x15, x15, 83',
            '0x0000e2c4: 0x00f72423 sw x15, 8(x14)',
            '0x0000e2c8: 0x00872783 lw x15, 8(x14)',
            '0x0000e2cc: 0x00870513 addi x10, x14, 8',
            '0x0000e2d0: 0x0080b7b7 lui x15, 2059',
            '0x0000e2d4: 0x1107a823 sw x16, 272(x15)',
            '0x0000e2d8: 0x1007a223 sw x0, 260(x15)',
            '0x0000e2dc: 0x0000e7b7 lui x15, 14',
            '0x0000e2e0: 0x6f87a783 lw x15, 1784(x15)',
            '0x0000e2e4: 0x1cf67463 bgeu x12, x15, 456',
            '0x0000e2e8: 0x8c7007b7 lui x15, 575232',
            '0x0000e2ec: 0x10878793 addi x15, x15, 264',
            '0x0000e2f0: 0x00f72423 sw x15, 8(x14)',
            '0x0000e2f4: 0x00872783 lw x15, 8(x14)',
            '0x0000e2f8: 0xfe000793 addi x15, x0, -32',
            '0x0000e2fc: 0x26f42423 sw x15, 616(x8)',
            '0x0000e300: 0x26f42623 sw x15, 620(x8)',
            '0x0000e304: 0x26f42823 sw x15, 624(x8)',
            '0x0000e308: 0x00505637 lui x12, 1285',
            '0x0000e30c: 0x008207b7 lui x15, 2080',
            '0x0000e310: 0x30060613 addi x12, x12, 768',
            '0x0000e314: 0x22c7a823 sw x12, 560(x15)',
            '0x0000e318: 0x010005b7 lui x11, 4096',
            '0x0000e31c: 0x22b7aa23 sw x11, 564(x15)',
            '0x0000e320: 0x23478613 addi x12, x15, 564',
            '0x0000e324: 0x02012783 lw x15, 32(x2)',
            '0x0000e328: 0x02000837 lui x16, 8192',
            '0x0000e32c: 0x00f62223 sw x15, 4(x12)',
            '0x0000e330: 0xb23107b7 lui x15, 729872',
            '0x0000e334: 0x0080e637 lui x12, 2062',
            '0x0000e338: 0xf0578793 addi x15, x15, -251',
            '0x0000e33c: 0x00f62023 sw x15, 0(x12)',
            '0x0000e340: 0xdeadc7b7 lui x15, 912092',
            '0x0000e344: 0xeef78793 addi x15, x15, -273',
            '0x0000e348: 0x00f12223 sw x15, 4(x2)',
            '0x0000e34c: 0x00412583 lw x11, 4(x2)',
            '0x0000e350: 0x0080f7b7 lui x15, 2063',
            '0x0000e354: 0x00b7a423 sw x11, 8(x15)',
            '0x0000e358: 0x0087a783 lw x15, 8(x15)',
            '0x0000e35c: 0x00200593 addi x11, x0, 2',
            '0x0000e360: 0x00f12223 sw x15, 4(x2)',
            '0x0000e364: 0x0080d7b7 lui x15, 2061',
            '0x0000e368: 0x08b7a023 sw x11, 128(x15)',
            '0x0000e36c: 0x00100593 addi x11, x0, 1',
            '0x0000e370: 0x00b7a023 sw x11, 0(x15)',
            '0x0000e374: 0x00b7a223 sw x11, 4(x15)',
            '0x0000e378: 0x0107a423 sw x16, 8(x15)',
            '0x0000e37c: 0x0107a623 sw x16, 12(x15)',
            '0x0000e380: 0x190005b7 lui x11, 102400',
            '0x0000e384: 0x0107a823 sw x16, 16(x15)',
            '0x0000e388: 0x0a058593 addi x11, x11, 160',
            '0x0000e38c: 0x00b7aa23 sw x11, 20(x15)',
            '0x0000e390: 0x0107ac23 sw x16, 24(x15)',
            '0x0000e394: 0x00b7ae23 sw x11, 28(x15)',
            '0x0000e398: 0x02b7a023 sw x11, 32(x15)',
            '0x0000e39c: 0xab0207b7 lui x15, 700448',
            '0x0000e3a0: 0x02878793 addi x15, x15, 40',
            '0x0000e3a4: 0x00f62023 sw x15, 0(x12)',
            '0x0000e3a8: 0x00052423 sw x0, 8(x10)',
            '0x0000e3ac: 0x00052623 sw x0, 12(x10)',
            '0x0000e3b0: 0x00852783 lw x15, 8(x10)',
            '0x0000e3b4: 0x00c52783 lw x15, 12(x10)',
            '0x0000e3b8: 0x17b007b7 lui x15, 97024',
            '0x0000e3bc: 0x07378793 addi x15, x15, 115',
            '0x0000e3c0: 0x00f72423 sw x15, 8(x14)',
            '0x0000e3c4: 0x00872783 lw x15, 8(x14)',
            '0x0000e3c8: 0x060c07b7 lui x15, 24768',
            '0x0000e3cc: 0x00f62023 sw x15, 0(x12)',
            '0x0000e3d0: 0x0d0c07b7 lui x15, 53440',
            '0x0000e3d4: 0x00f62023 sw x15, 0(x12)',
            '0x0000e3d8: 0x06000000 mop mop_type[23:23] = 0b1, done[22:22] = 0b0, loop_count[21:15] = 0x0, zmask_lo8_or_loop_count[14:0] = 0x0',
            '0x0000e3dc: 0x3d0187b7 lui x15, 249880',
            '0x0000e3e0: 0x02878793 addi x15, x15, 40',
            '0x0000e3e4: 0x00f62023 sw x15, 0(x12)',
            '0x0000e3e8: 0x0fa6d703 lhu x14, 250(x13)',
            '0x0000e3ec: 0x0e86a783 lw x15, 232(x13)',
            '0x0000e3f0: 0x00170713 addi x14, x14, 1',
            '0x0000e3f4: 0x0ee69d23 sh x14, 250(x13)',
            '0x0000e3f8: 0x0f46a703 lw x14, 244(x13)',
            '0x0000e3fc: 0x00e787b3 add x15, x15, x14',
            '0x0000e400: 0x0e46a703 lw x14, 228(x13)',
            '0x0000e404: 0x0ef6aa23 sw x15, 244(x13)',
            '0x0000e408: 0x00e7e863 bltu x15, x14, 16',
            '0x0000e40c: 0x0e06a703 lw x14, 224(x13)',
            '0x0000e410: 0x40e787b3 sub x15, x15, x14',
            '0x0000e414: 0x0ef6aa23 sw x15, 244(x13)',
            '0x0000e418: 0x88800aa2 stallwait stall_res[23:15] = 0x40, wait_res_idx_2[14:10] = 0, wait_res_idx_1[9:5] = 21, wait_res_idx_0[4:0] = 8',
            '0x0000e41c: 0x0000e7b7 lui x15, 14',
            '0x0000e420: 0x6fc7a683 lw x13, 1788(x15)',
            '0x0000e424: 0x10100737 lui x14, 65792',
            '0x0000e428: 0x00d70733 add x14, x14, x13',
            '0x0000e42c: 0x0080e6b7 lui x13, 2062',
            '0x0000e430: 0x00e6a023 sw x14, 0(x13)',
            '0x0000e434: 0xd8000080 cleardvalid cleardvalid[23:22] = 0b0, cleardvalid_S[21:20] = 0b0, dest_dvalid_reset[13:10] = 0b0, dest_dvalid_client_bank_reset[9:6] = 0b0, dest_pulse_last[5:2] = 0b1000, reset[1:0] = 0b0',
            '0x0000e438: 0x6fc7a683 lw x13, 1788(x15)',
            '0x0000e43c: 0x00100713 addi x14, x0, 1',
            '0x0000e440: 0x40d70733 sub x14, x14, x13',
            '0x0000e444: 0x6ee7ae23 sw x14, 1788(x15)',
            '0x0000e448: 0x0080b737 lui x14, 2059',
            '0x0000e44c: 0x10c72783 lw x15, 268(x14)',
            '0x0000e450: 0xfe079ee3 bne x15, x0, -4',
            '0x0000e454: 0xfacec6b7 lui x13, 1027308',
            '0x0000e458: 0x00a00713 addi x14, x0, 10',
            '0x0000e45c: 0x000155b7 lui x11, 21',
            '0x0000e460: 0xeef68693 addi x13, x13, -273',
            '0x0000e464: 0xffe5d783 lhu x15, -2(x11)',
            '0x0000e468: 0x01079613 slli x12, x15, 16',
            '0x0000e46c: 0x0005d783 lhu x15, 0(x11)',
            '0x0000e470: 0x01065613 srli x12, x12, 16',
            '0x0000e474: 0x01079793 slli x15, x15, 16',
            '0x0000e478: 0x00c7e7b3 or x15, x15, x12',
            '0x0000e47c: 0x00d78663 beq x15, x13, 12',
            '0x0000e480: 0xfff70713 addi x14, x14, -1',
            '0x0000e484: 0xfe0710e3 bne x14, x0, -32',
            '0x0000e488: 0x0000e737 lui x14, 14',
            '0x0000e48c: 0x00472783 lw x15, 4(x14)',
            '0x0000e490: 0x03c12083 lw x1, 60(x2)',
            '0x0000e494: 0x03812403 lw x8, 56(x2)',
            '0x0000e498: 0x00178793 addi x15, x15, 1',
            '0x0000e49c: 0x00f72223 sw x15, 4(x14)',
            '0x0000e4a0: 0x00000513 addi x10, x0, 0',
            '0x0000e4a4: 0x04010113 addi x2, x2, 64',
            '0x0000e4a8: 0x00008067 jalr x0, 0(x1)',
            '0x0000e4ac: 0x09878793 addi x15, x15, 152',
            '0x0000e4b0: 0x00279793 slli x15, x15, 2',
            '0x0000e4b4: 0x00f407b3 add x15, x8, x15',
            '0x0000e4b8: 0xfe000613 addi x12, x0, -32',
            '0x0000e4bc: 0x00c7a423 sw x12, 8(x15)',
            '0x0000e4c0: 0xe49ff06f jal x0, -440']

        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            msgs = read_elf.instructions.instruction_to_str(read_elf.decode_function(function_name, elf))
            assert expected_msgs == msgs
            assert read_elf.instructions.instruction_to_str(read_elf.decode_function_from_elf_object(function_name, elf)) == msgs

def test_decode_function_exception():
    with pytest.raises(Exception) as exe_info:
        read_elf.decode_function("main", typing.cast(typing.Any, ["abc"]))

def test_decode_functions_from_elf_object():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttwh/tt-metal/bmm/686715490291187641/trisc0/trisc0.elf")
        func_names = ['_start', '_Z13kernel_launchm']
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            funcs_str_ins = read_elf.instructions.instruction_to_str(read_elf.decode_functions_from_elf_object(func_names, elf))
            assert isinstance(funcs_str_ins, dict)
            for func_name, msgs in funcs_str_ins.items():
                assert read_elf.instructions.instruction_to_str(read_elf.decode_function(func_name, elf)) == msgs

@pytest.mark.slow
def test_decode_functions_from_file_name():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttwh/tt-metal/bmm/686715490291187641/trisc1/trisc1.elf")
        funcs_str_ins = read_elf.instructions.instruction_to_str(read_elf.decode_functions(read_elf.get_all_function_names(file_name), file_name))
        assert isinstance(funcs_str_ins, dict)
        for func_name, msgs in funcs_str_ins.items():
            assert read_elf.instructions.instruction_to_str(read_elf.decode_function(func_name, file_name)) == msgs

def test_decode_functions_from_elf():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttwh/tt-metal/bmm/686715490291187641/trisc2/trisc2.elf")
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            funcs_str_ins = read_elf.instructions.instruction_to_str(read_elf.decode_functions(read_elf.get_all_function_names(elf), elf))
            assert isinstance(funcs_str_ins, dict)
            for func_name, msgs in funcs_str_ins.items():
                assert read_elf.instructions.instruction_to_str(read_elf.decode_function(func_name, file_name)) == msgs

def test_decode_functions_exception():
    with pytest.raises(Exception) as exe_info:
        read_elf.decode_functions([""],typing.cast(typing.Any, []))

@pytest.mark.slow
def test_decode_all_functions_from_elf_object():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162/trisc2/trisc2.elf")
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            funcs_str_ins = read_elf.instructions.instruction_to_str(read_elf.decode_all_functions_from_elf_object(elf), flatten_dict = False)
            assert sorted(funcs_str_ins.keys()) == sorted(read_elf.get_all_function_names(elf))
            for func_name, msgs in funcs_str_ins.items():
                assert read_elf.instructions.instruction_to_str(read_elf.decode_function(func_name, elf)) == msgs

@pytest.mark.slow
def test_decode_all_functions_from_file_name():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162/trisc0/trisc0.elf")
        funcs_str_ins = read_elf.instructions.instruction_to_str(read_elf.decode_all_functions(file_name))
        assert isinstance(funcs_str_ins, dict)
        assert sorted(funcs_str_ins.keys()) == sorted(read_elf.get_all_function_names(file_name))
        for func_name, msgs in funcs_str_ins.items():
            assert read_elf.instructions.instruction_to_str(read_elf.decode_function(func_name, file_name)) == msgs

@pytest.mark.slow
def test_decode_all_functions_from_elf():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162/trisc1/trisc1.elf")
        with open(file_name, 'rb') as file:
            elf = elftools.elf.elffile.ELFFile(file)
            funcs_str_ins = read_elf.instructions.instruction_to_str(read_elf.decode_all_functions(elf))
            assert isinstance(funcs_str_ins, dict)
            assert sorted(funcs_str_ins.keys()) == sorted(read_elf.get_all_function_names(elf))
            for func_name, msgs in funcs_str_ins.items():
                assert read_elf.instructions.instruction_to_str(read_elf.decode_function(func_name, file_name)) == msgs

def test_decode_all_functions_exception():
    with pytest.raises(Exception) as exe_info:
        read_elf.decode_all_functions(typing.cast(typing.Any, dict()))

def test_get_instruction_profile_0():
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162/trisc0/trisc0.elf")
        func_names_dec_ins = read_elf.decode_all_functions(file_name)

        ref_ip:typing.Any = read_elf.instructions.get_statistics(func_names_dec_ins, flatten_dict = True)
        ip: typing.Any = read_elf.get_instruction_profile(file_name, flatten_dict = True)
        assert ref_ip == ip

        ref_ip = read_elf.instructions.get_statistics(func_names_dec_ins, flatten_dict = False)
        ip = read_elf.get_instruction_profile(file_name, flatten_dict = False)
        assert ref_ip == ip

@pytest.mark.slow
def test_get_instruction_profile_1():
    def get_reference_instruction_profile(path, flatten_dict):
        ref_ip = dict()
        for pwd, _, files in os.walk(path):
            for file in files:
                if file.endswith(".elf"):
                    file_name_incl_path = os.path.join(pwd, file)
                    ref_ip[file_name_incl_path] = read_elf.instructions.get_statistics(read_elf.decode_all_functions(file_name_incl_path), flatten_dict = flatten_dict)

        return ref_ip

    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162")

        flatten_dict = True
        ref_ip = get_reference_instruction_profile(file_name, flatten_dict = flatten_dict)
        ip = read_elf.get_instruction_profile(file_name, flatten_dict = flatten_dict)
        assert ip == ref_ip

        flatten_dict = False
        ref_ip = get_reference_instruction_profile(file_name, flatten_dict = flatten_dict)
        ip = read_elf.get_instruction_profile(file_name, flatten_dict = flatten_dict)
        assert ip == ref_ip

@pytest.mark.slow
def test_get_instruction_profile_2():
    def get_elf_files(file_names):
        elf_files = list()
        for file_name in file_names:
            if os.path.isfile(file_name):
                elf_files.append(file_name)
            elif os.path.isdir(file_name):
                for pwd, _, files in os.walk(file_name):
                    for file in files:
                        if file.endswith(".elf"):
                            file_name_incl_path = os.path.join(pwd, file)
                            elf_files.append(file_name_incl_path)

        return sorted(elf_files)

    def get_reference_instruction_profile(elf_files, flatten_dict):
        ref_ip = dict()
        for file in elf_files:
            ref_ip[file] = read_elf.instructions.get_statistics(read_elf.decode_all_functions(file), flatten_dict = flatten_dict)

        return ref_ip

    if TEST_WITH_ELF_FILES:
        file_names = [
            os.path.join(LOCAL_ELF_TEST_DIR, "ttwh/tt-metal/bmm/686715490291187641"),
            os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf")
        ]

        flatten_dict = True
        ref_ip = get_reference_instruction_profile(get_elf_files(file_names), flatten_dict = flatten_dict)
        ip = read_elf.get_instruction_profile(file_names, flatten_dict = flatten_dict)
        assert ip == ref_ip

        flatten_dict = False
        ref_ip = get_reference_instruction_profile(get_elf_files(file_names), flatten_dict = flatten_dict)
        ip = read_elf.get_instruction_profile(file_names, flatten_dict = flatten_dict)
        assert ip == ref_ip

def test_get_instruction_profile_3():
    with pytest.raises(Exception) as exe_info:
        read_elf.get_instruction_profile(typing.cast(typing.Any, dict()))

def test_get_instruction_profile_exception():
    with pytest.raises(Exception):
        read_elf.get_instruction_profile("abc")

    with pytest.raises(Exception):
        read_elf.get_instruction_profile(["abc"])

    with pytest.raises(Exception):
        read_elf.get_instruction_profile(typing.cast(typing.Any, dict()))

def test_print_instruction_profile_0(capsys):
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf")

        flatten_dict = True
        ip = read_elf.get_instruction_profile(file_name, flatten_dict = flatten_dict)
        print("--------------------------")
        for idx, ele in enumerate(ip):
            print("---- s idx = ", idx)
            print(ele)
            print("---- e idx = ", idx)
        print("--------------------------")
        capsys.readouterr()
        read_elf.print_instruction_profile(ip, flatten_dict = flatten_dict, print_offset = 0)
        captured = capsys.readouterr()
        ref_ip = read_elf.instructions.get_statistics(read_elf.decode_all_functions(file_name), flatten_dict = flatten_dict)
        assert 2 == len(ref_ip)
        assert read_elf.instructions.instruction_histogram_to_str(ref_ip[0]) in captured.out
        assert read_elf.instructions.instruction_kind_histogram_to_str(ref_ip[1]) in captured.out

def test_print_instruction_profile_1(capsys):
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf")

        flatten_dict = False
        ip = read_elf.get_instruction_profile(file_name, flatten_dict = flatten_dict)
        capsys.readouterr()
        read_elf.print_instruction_profile(ip, flatten_dict = flatten_dict, print_offset = 0)
        captured = capsys.readouterr()
        ref_ip = read_elf.instructions.get_statistics(read_elf.decode_all_functions(file_name), flatten_dict = flatten_dict)
        assert 2 == len(ref_ip)
        assert read_elf.instructions.instruction_histogram_to_str(ref_ip[0]) in captured.out
        assert read_elf.instructions.instruction_kind_histogram_to_str(ref_ip[1]) in captured.out

@pytest.mark.slow
def test_print_instruction_profile_2(capsys):
    # str leads to directory
    def get_elf_files(file_names):
        elf_files = list()
        for file_name in file_names:
            if os.path.isfile(file_name):
                elf_files.append(file_name)
            elif os.path.isdir(file_name):
                for pwd, _, files in os.walk(file_name):
                    for file in files:
                        if file.endswith(".elf"):
                            file_name_incl_path = os.path.join(pwd, file)
                            elf_files.append(file_name_incl_path)

        return sorted(elf_files)

    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0")

        for flatten_dict in [True, False]:
            capsys.readouterr()
            ip = read_elf.get_instruction_profile(file_name, flatten_dict = flatten_dict)
            read_elf.print_instruction_profile(ip, flatten_dict = flatten_dict, print_offset = 0)
            captured = capsys.readouterr()
            for elf_file in get_elf_files([file_name]): # square brackets are important
                ref_ip = read_elf.instructions.get_statistics(read_elf.decode_all_functions(elf_file), flatten_dict = flatten_dict)
                assert 2 == len(ref_ip)
                assert read_elf.instructions.instruction_histogram_to_str(ref_ip[0]) in captured.out
                assert read_elf.instructions.instruction_kind_histogram_to_str(ref_ip[1]) in captured.out

@pytest.mark.slow
def test_print_instruction_profile_3(capsys):
    # str leads to directory
    def get_elf_files(file_names):
        elf_files = list()
        for file_name in file_names:
            if os.path.isfile(file_name):
                elf_files.append(file_name)
            elif os.path.isdir(file_name):
                for pwd, _, files in os.walk(file_name):
                    for file in files:
                        if file.endswith(".elf"):
                            file_name_incl_path = os.path.join(pwd, file)
                            elf_files.append(file_name_incl_path)

        return sorted(elf_files)

    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0")

        for flatten_dict in [True, False]:
            capsys.readouterr()
            ip = read_elf.get_instruction_profile(file_name, flatten_dict = flatten_dict)
            read_elf.print_instruction_profile(ip, flatten_dict = flatten_dict, print_offset = 0)
            captured = capsys.readouterr()
            for elf_file in get_elf_files([file_name]): # square brackets are important
                ref_ip = read_elf.instructions.get_statistics(read_elf.decode_all_functions(elf_file), flatten_dict = flatten_dict)
                assert 2 == len(ref_ip)
                assert read_elf.instructions.instruction_histogram_to_str(ref_ip[0]) in captured.out
                assert read_elf.instructions.instruction_kind_histogram_to_str(ref_ip[1]) in captured.out

@pytest.mark.slow
def test_print_instruction_profile_4(capsys):
    # str leads to directory
    def get_elf_files(file_names):
        elf_files = list()
        for file_name in file_names:
            if os.path.isfile(file_name):
                elf_files.append(file_name)
            elif os.path.isdir(file_name):
                for pwd, _, files in os.walk(file_name):
                    for file in files:
                        if file.endswith(".elf"):
                            file_name_incl_path = os.path.join(pwd, file)
                            elf_files.append(file_name_incl_path)

        return sorted(elf_files)

    if TEST_WITH_ELF_FILES:
        file_names = [
            os.path.join(LOCAL_ELF_TEST_DIR, "ttbh/tt-metal/bmm/13730376427513645162"),
            os.path.join(LOCAL_ELF_TEST_DIR, "ttwh/tt-metal/bmm/686715490291187641")
        ] # TODO: make it work for directories.
        file_names = [
            os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf"),
            os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_2/out/thread_2.elf"),
        ]

        for flatten_dict in [True, False]: # TODO: resolve issues associated with False. Somewhoe function names seem to be different.
            capsys.readouterr()
            read_elf.print_instruction_profile(set(file_names), flatten_dict = flatten_dict, print_offset = 0)
            captured = capsys.readouterr()
            for elf_file in get_elf_files(file_names):
                ref_ip = read_elf.instructions.get_statistics(read_elf.decode_all_functions(elf_file), flatten_dict = flatten_dict)
                assert 2 == len(ref_ip)
                assert read_elf.instructions.instruction_histogram_to_str(ref_ip[0]) in captured.out
                assert read_elf.instructions.instruction_kind_histogram_to_str(ref_ip[1]) in captured.out

def test_print_instruction_profile_exception():
    with pytest.raises(TypeError) as exe_info:
        read_elf.print_instruction_profile(typing.cast(typing.Any, dict({"abc" : "cde"}))) # incorrect dict type
    assert "no method defined to print instruction profile of type" in str(exe_info.value)

    with pytest.raises(TypeError):
        read_elf.print_instruction_profile(typing.cast(typing.Any, [])) # incorrect file type

    with pytest.raises(TypeError):
        ele0: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
        ele1: dict[decoded_instruction.instruction_kind, dict[str, int]] = dict()
        ele0[decoded_instruction.instruction_kind.ttwh] = dict({"a" : list([0, 1.0, 2.0])})
        ele1[decoded_instruction.instruction_kind.ttwh] = dict({"a" : 5})
        arg: typing.Any = tuple([ele0, ele1, ele0])
        read_elf.print_instruction_profile(typing.cast(typing.Any, arg)) # tuple 3 elements.

    with pytest.raises(TypeError):
        arg = tuple([[5], ["a"]])
        read_elf.print_instruction_profile(typing.cast(typing.Any, arg))

    with pytest.raises(TypeError):
        ele0a: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
        ele0a[decoded_instruction.instruction_kind.ttwh] = dict({"a" : list([0, 1.0, 2.0])})
        arg = tuple([ele0a, ["a"]])
        read_elf.print_instruction_profile(arg)

    with pytest.raises(TypeError):
        ele0b: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
        ele1b: dict[decoded_instruction.instruction_kind, dict[str, int]] = dict()
        ele0b[decoded_instruction.instruction_kind.ttwh] = typing.cast(typing.Any, [5])
        ele1b[decoded_instruction.instruction_kind.ttwh] = typing.cast(typing.Any, [1])
        arg = tuple([ele0b, ele1b])
        read_elf.print_instruction_profile(typing.cast(typing.Any, arg))

    with pytest.raises(TypeError):
        ele0c: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
        ele1c: dict[decoded_instruction.instruction_kind, dict[str, int]] = dict()
        ele0c[typing.cast(typing.Any, "a")] = dict({"a" : list([0, 1.0, 2.0])})
        ele1c[decoded_instruction.instruction_kind.ttwh] = dict({"a" : 5})
        arg = tuple([ele0c, ele1c])
        read_elf.print_instruction_profile(typing.cast(typing.Any, arg))

    with pytest.raises(TypeError):
        ele0d: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
        ele1d: dict[decoded_instruction.instruction_kind, dict[str, int]] = dict()
        ele0d[decoded_instruction.instruction_kind.ttwh] = dict({typing.cast(typing.Any, 5) : list([0, 1.0, 2.0])})
        ele1d[decoded_instruction.instruction_kind.ttwh] = dict({"a" : 5})
        arg = tuple([ele0d, ele1d])
        read_elf.print_instruction_profile(arg)

    with pytest.raises(TypeError):
        ele0e: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
        ele1e: dict[decoded_instruction.instruction_kind, dict[str, int]] = dict()
        ele0e[decoded_instruction.instruction_kind.ttwh] = dict({"a" : list([0, 1.0, typing.cast(typing.Any, "a")])})
        ele1e[decoded_instruction.instruction_kind.ttwh] = dict({"a" : 5})
        arg = tuple([ele0e, ele1e])
        read_elf.print_instruction_profile(arg)

    with pytest.raises(TypeError):
        ele0f: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
        ele1f: dict[decoded_instruction.instruction_kind, dict[str, int]] = dict()
        ele0f[decoded_instruction.instruction_kind.ttwh] = dict({"a" : list([0, 1.0, 2.0])})
        ele1f[decoded_instruction.instruction_kind.ttwh] = dict({typing.cast(typing.Any, 5) : 5})
        arg = tuple([ele0f, ele1f])
        read_elf.print_instruction_profile(arg)

    with pytest.raises(TypeError):
        ele0g: dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]] = dict()
        ele1g: dict[decoded_instruction.instruction_kind, dict[str, int]] = dict()
        ele0g[decoded_instruction.instruction_kind.ttwh] = dict({"a" : list([0, 1.0, 2.0])})
        ele1g[decoded_instruction.instruction_kind.ttwh] = dict({"a" : typing.cast(typing.Any, [])})
        arg = tuple([ele0g, ele1g])
        read_elf.print_instruction_profile(arg)

    with pytest.raises(TypeError):
        ele0h: dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]] = dict()
        ele1h: dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]] = dict()
        ele0h[typing.cast(typing.Any, 11)] = dict({decoded_instruction.instruction_kind.ttwh : dict({"a" : list([0, 1.0, 2.0])})})
        ele1h["abc"] = dict({typing.cast(typing.Any, "a") : typing.cast(typing.Any, [])})
        arg = tuple([ele0h, ele1h])
        read_elf.print_instruction_profile(arg)

    with pytest.raises(TypeError):
        ele0i: dict[str, dict[decoded_instruction.instruction_kind, dict[str, list[int | float]]]] = dict()
        ele1i: dict[str, dict[decoded_instruction.instruction_kind, dict[str, int]]] = dict()
        ele0i["s"] = dict({decoded_instruction.instruction_kind.ttwh : dict({"a" : list([0, 1.0, 2.0])})})
        ele1i[typing.cast(typing.Any, decoded_instruction.instruction_kind.ttwh)] = dict({typing.cast(typing.Any, "a") : typing.cast(typing.Any, [])})
        arg = tuple([ele0i, ele1i])
        read_elf.print_instruction_profile(arg)

    with pytest.raises(Exception):
        read_elf.print_instruction_profile(set(["abc"])) # files do not exist

    with pytest.raises(Exception):
        read_elf.print_instruction_profile(typing.cast(typing.Any, set([0, 1]))) # set of ints is incorrect argument


@pytest.mark.slow
def test_print_instruction_profile_with_print_instructions(capsys):
    if TEST_WITH_ELF_FILES:
        file_name = os.path.join(LOCAL_ELF_TEST_DIR, "ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf")

        capsys.readouterr()
        read_elf.print_instruction_profile(file_name, print_instructions = True, print_statistics = False)
        captured = capsys.readouterr()
        funcs_msgs = read_elf.instructions.instruction_to_str(read_elf.decode_all_functions(file_name))
        assert isinstance(funcs_msgs, dict)
        for func_name, msgs in funcs_msgs.items():
            instr_strs = ""
            for msg in msgs:
                instr_strs += f"    {msg}\n"
            assert func_name in captured.out
            assert instr_strs in captured.out

@pytest.mark.slow
def test_any_unknown_instructions_in_elf_0_dir():
    if TEST_WITH_ELF_FILES:
        files_with_unknown_instructions = [
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf',
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf',
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_2/out/thread_2.elf']
        assert all(ele.startswith(LOCAL_ELF_TEST_DIR) for ele in files_with_unknown_instructions)
        assert read_elf.any_unknown_instructions_in_elf(LOCAL_ELF_TEST_DIR, print_instructions = False, print_statistics = False) == files_with_unknown_instructions

def test_any_unknown_instructions_in_elf_1_file():
    if TEST_WITH_ELF_FILES:
        found_file = False
        for pwd, _, file_names in os.walk(LOCAL_ELF_TEST_DIR):
            if found_file:
                break
            for file_name in file_names:
                file_name_incl_path = os.path.join(pwd, file_name)
                if file_name_incl_path.endswith("ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf"):
                    assert read_elf.any_unknown_instructions_in_elf(file_name_incl_path)
                    found_file = True
                    break

def test_any_unknown_instructions_in_elf_2_print_statistics(capsys):
    if TEST_WITH_ELF_FILES:
        files_with_unknown_instructions = [
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf',
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf',
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_2/out/thread_2.elf']
        assert all(ele.startswith(LOCAL_ELF_TEST_DIR) for ele in files_with_unknown_instructions)
        capsys.readouterr()
        read_elf.any_unknown_instructions_in_elf(files_with_unknown_instructions[0], print_statistics = True, print_instructions = False)
        capture = capsys.readouterr()
        assert "not in ISA" in capture.out

def test_any_unknown_instructions_in_elf_3_print_instructions(capsys):
    if TEST_WITH_ELF_FILES:
        files_with_unknown_instructions = [
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf',
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf',
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_2/out/thread_2.elf']
        assert all(ele.startswith(LOCAL_ELF_TEST_DIR) for ele in files_with_unknown_instructions)
        capsys.readouterr()
        read_elf.any_unknown_instructions_in_elf(files_with_unknown_instructions[0], print_statistics = False, print_instructions = True)
        capture = capsys.readouterr()
        assert "0x00000000" in capture.out

def test_any_unknown_instructions_in_elf_4_exceptions(capsys):
    with pytest.raises(Exception):
        read_elf.any_unknown_instructions_in_elf(typing.cast(typing.Any, []))

    with pytest.raises(Exception):
        read_elf.any_unknown_instructions_in_elf("")


@pytest.mark.slow
def test_get_coverage_0():
    if TEST_WITH_ELF_FILES:

        cov = dict({
            LOCAL_ELF_TEST_DIR + '/ttwh/tt-metal/bmm/686715490291187641/trisc2/trisc2.elf': {
                '_Z13kernel_launchm': {decoded_instruction.instruction_kind.rv32: 0.09482758620689655, decoded_instruction.instruction_kind.ttwh: 0.11023622047244094},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207}},
            LOCAL_ELF_TEST_DIR + '/ttwh/tt-metal/bmm/686715490291187641/trisc0/trisc0.elf': {
                '_Z13kernel_launchm': {decoded_instruction.instruction_kind.rv32: 0.08620689655172414, decoded_instruction.instruction_kind.ttwh: 0.11023622047244094},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207}},
            LOCAL_ELF_TEST_DIR + '/ttwh/tt-metal/bmm/686715490291187641/trisc1/trisc1.elf': {
                '_Z13kernel_launchm': {decoded_instruction.instruction_kind.rv32: 0.07327586206896551, decoded_instruction.instruction_kind.ttwh: 0.06299212598425197},
                '_Z24matmul_configure_addrmodILi4ELN7ckernel17DstTileFaceLayoutE0EEvbmmmmmmmb.part.0.constprop.0': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207, decoded_instruction.instruction_kind.ttwh: 0.007874015748031496},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207}},
            LOCAL_ELF_TEST_DIR + '/ttbh/tt-metal/bmm/13730376427513645162/trisc2/trisc2.elf': {
                '_Z13kernel_launchm': {decoded_instruction.instruction_kind.rv32: 0.09482758620689655, decoded_instruction.instruction_kind.ttbh: 0.08759124087591241},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207}},
            LOCAL_ELF_TEST_DIR + '/ttbh/tt-metal/bmm/13730376427513645162/trisc0/trisc0.elf': {
                '_Z13kernel_launchm': {decoded_instruction.instruction_kind.rv32: 0.09482758620689655, decoded_instruction.instruction_kind.ttbh: 0.10218978102189781},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207}},
            LOCAL_ELF_TEST_DIR + '/ttbh/tt-metal/bmm/13730376427513645162/trisc1/trisc1.elf': {
                '_Z13kernel_launchm': {decoded_instruction.instruction_kind.rv32: 0.08189655172413793, decoded_instruction.instruction_kind.ttbh: 0.06569343065693431},
                '_Z24matmul_configure_addrmodILi4ELN7ckernel17DstTileFaceLayoutE0EEvbmmmmmmmb.part.0.constprop.0': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207, decoded_instruction.instruction_kind.ttbh: 0.0072992700729927005},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207}},
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_2/out/thread_2.elf': {
                '_GLOBAL__sub_I__ZN6PCK_0134PARAM_OUTPUT_TOTAL_NUM_TILES_C_DIME': {decoded_instruction.instruction_kind.rv32: 0.021551724137931036},
                '_fini': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                '_init': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.03017241379310345, decoded_instruction.instruction_kind.ttqs : 0.005649717514124294},
                'exit': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                'main': {decoded_instruction.instruction_kind.rv32: 0.07758620689655173, decoded_instruction.instruction_kind.ttqs : 0.01694915254237288},
                'memcpy': {decoded_instruction.instruction_kind.rv32: 0.06465517241379311},
                'memset': {decoded_instruction.instruction_kind.rv32: 0.0603448275862069},
                'wzerorange': {decoded_instruction.instruction_kind.rv32: 0.01293103448275862}},
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_0/out/thread_0.elf': {
                '_GLOBAL__sub_I__ZN6UPK_0119PARAM_SRCA_PER_ITERE': {decoded_instruction.instruction_kind.rv32: 0.021551724137931036},
                '_ZN7ckernel5trisc26_configure_buf_desc_table_ERK17tdma_descriptor_t': {decoded_instruction.instruction_kind.rv32: 0.03017241379310345},
                '_ZN7ckernel5triscL16SCALE_DATUM_SIZEEmm': {decoded_instruction.instruction_kind.rv32: 0.034482758620689655},
                '_fini': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                '_init': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.03017241379310345, decoded_instruction.instruction_kind.ttqs : 0.005649717514124294},
                'exit': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                'main': {decoded_instruction.instruction_kind.rv32: 0.08189655172413793, decoded_instruction.instruction_kind.ttqs : 0.022598870056497175},
                'memcpy': {decoded_instruction.instruction_kind.rv32: 0.06465517241379311},
                'memset': {decoded_instruction.instruction_kind.rv32: 0.0603448275862069},
                'wzerorange': {decoded_instruction.instruction_kind.rv32: 0.01293103448275862}},
            LOCAL_ELF_TEST_DIR + '/ttqs/rtl/t6-quas-n1-ttx-matmul-block-in0-2-16-in1-16-1-fp16_b-llk_0/ttx/kernels/core_00_00/neo_0/thread_1/out/thread_1.elf': {
                '_GLOBAL__sub_I__ZN7ckernel16ckernel_templateC2Emmm': {decoded_instruction.instruction_kind.rv32: 0.021551724137931036},
                '_ZNK7ckernel10addr_mod_t7src_valEv': {decoded_instruction.instruction_kind.rv32: 0.021551724137931036},
                '_ZNK7ckernel10addr_mod_t8dest_valEv': {decoded_instruction.instruction_kind.rv32: 0.04310344827586207},
                '_fini': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                '_init': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                '_start': {decoded_instruction.instruction_kind.rv32: 0.03017241379310345, decoded_instruction.instruction_kind.ttqs : 0.005649717514124294},
                'exit': {decoded_instruction.instruction_kind.rv32: 0.004310344827586207},
                'main': {decoded_instruction.instruction_kind.rv32: 0.09482758620689655, decoded_instruction.instruction_kind.ttqs : 0.03954802259887006},
                'memcpy': {decoded_instruction.instruction_kind.rv32: 0.06465517241379311},
                'memset': {decoded_instruction.instruction_kind.rv32: 0.0603448275862069},
                'wzerorange': {decoded_instruction.instruction_kind.rv32: 0.01293103448275862}}})
        instruction_sets: dict[decoded_instruction.instruction_kind, str] = dict()
        instruction_sets[decoded_instruction.instruction_kind.ttqs] = os.path.join(os.path.dirname(__file__), "../../ttsim/config/llk/instruction_sets/ttqs/assembly.mar18.yaml")
        assert cov == read_elf.get_coverage(LOCAL_ELF_TEST_DIR, sets = instruction_sets)

@pytest.mark.slow
def test_get_coverage_1():
    if TEST_WITH_ELF_FILES:
        cov = {
            decoded_instruction.instruction_kind.rv32 : 0.14224137931034483,
            decoded_instruction.instruction_kind.ttwh : 0.18110236220472442,
            decoded_instruction.instruction_kind.ttbh : 0.15328467153284672,
            decoded_instruction.instruction_kind.ttqs : 0.062146892655367235
        }

        instruction_sets: dict[decoded_instruction.instruction_kind, str] = dict()
        instruction_sets[decoded_instruction.instruction_kind.ttqs] = os.path.join(os.path.dirname(__file__), "../../ttsim/config/llk/instruction_sets/ttqs/assembly.mar18.yaml")
        assert cov == read_elf.get_coverage(LOCAL_ELF_TEST_DIR, flatten_dict = True, sets = instruction_sets)
