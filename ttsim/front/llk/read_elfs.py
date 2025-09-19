#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import typing

import ttsim.front.llk.read_elf as read_elf

def get_architecture_from_tneo_sim_args_dict(inputcfg: dict[str, typing.Any]) -> str:
    key_input = "input"
    key_numTriscCores = "numTCores"

    for key in [var_value for var_name, var_value in locals().items() if var_name.startswith("key_")]:
        assert key in inputcfg.keys(), f"- error: {key} not found in given inputcfg dict"

    input = inputcfg[key_input]
    instruction_kinds = set()
        
    for core_id in range(inputcfg[key_numTriscCores]):
        key_tc = f"tc{core_id}"
        assert key_tc in input.keys(), f"- error: {key_tc} not found in given input dict"
        tc = input[key_tc]

        key_numThreads = f"numThreads"
        assert key_numThreads in tc.keys(), f"- error: {key_numThreads} not found in given input dict"
        num_threads = tc[key_numThreads]

        for thread_id in range(num_threads): # this loop works only if num_threads is 4 (max possible value).
            key_elf_file = f"th{thread_id}Elf"
            key_elf_path = f"th{thread_id}Path"
            assert key_elf_file in tc.keys(), f"- error: {key_elf_file} not found in given tc dict"
            assert key_elf_path in tc.keys(), f"- error: {key_elf_path} not found in given tc dict"
            elf_file_name = os.path.join(tc[key_elf_path], tc[key_elf_file])
            if not elf_file_name:
                continue
            if not os.path.exists(elf_file_name):
                raise FileNotFoundError(f"ELF file {elf_file_name} does not exist.")
            
            if not instruction_kinds:
                instruction_kinds = read_elf.get_instruction_kinds(elf_file_name)
            else:
                assert instruction_kinds == read_elf.get_instruction_kinds(elf_file_name), \
                    f"- error: instruction kinds mismatch in {elf_file_name} with previous ELF files. " \
                    f"previous instruction kinds: {instruction_kinds}, " \
                    f"current instruction kinds: {read_elf.get_instruction_kinds(elf_file_name)}"
                
    if not instruction_kinds:
        raise ValueError(f"No instruction kinds found from elfs in the given input dict. input = {input}")
    
    assert len(instruction_kinds) == 2, \
        f"- error: expected 2 instruction kinds, but got {len(instruction_kinds)}. " \
        f"instruction_kinds = {instruction_kinds}"
    
    assert sum([1 for kind in instruction_kinds if kind.is_tensix()]) == 1, \
        f"- error: expected exactly 1 Tensix instruction kind, but got {sum([1 for kind in instruction_kinds if kind.is_tensix()])}. " \
        f"instruction_kinds = {instruction_kinds}"
    
    assert sum([1 for kind in instruction_kinds if not kind.is_tensix()]) == 1, \
        f"- error: expected exactly 1 non-tensix instruction kind, but got {sum([1 for kind in instruction_kinds if not kind.is_tensix()])}. " \
        f"instruction_kinds = {instruction_kinds}"
    
    for kind in instruction_kinds:
        if kind.is_tensix():
            return f"{kind}"