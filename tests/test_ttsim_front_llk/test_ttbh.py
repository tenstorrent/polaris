#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import pytest

import ttsim.front.llk.ttbh as ttbh


def test_instruction_kind():
    assert ttbh.tensix.decoded_instruction.instruction_kind.ttbh == ttbh.instruction_kind()

def test_get_default_instruction_set():
    instruction_set0 = ttbh.get_default_instruction_set()
    instruction_set1 = ttbh.get_instruction_set_from_file_name(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "ttsim", "config", "llk", "instruction_sets", f"{ttbh.instruction_kind()}", "assembly.yaml")))
    assert instruction_set0 == instruction_set1

@pytest.mark.slow
def test_decode_instruction():
    instruction_set = ttbh.get_default_instruction_set()
    mnemonics_words = {mnemonic : ttbh.decoded_instruction.swizzle_instruction(ttbh.decoded_instruction.right_circular_shift(info["op_binary"], shift = 8, num_bits = 32)) for mnemonic, info in instruction_set.items()}

    for mnemonic, word in mnemonics_words.items():
        di0 = ttbh.decode_instruction(word)
        di1 = ttbh.decode_instruction(word, instruction_set)
        di2 = ttbh.decode_instruction(ttbh.decoded_instruction.unswizzle_instruction(word), instruction_set, is_swizzled = False)

        assert di0 == di1
        assert di0 == di2
        assert hasattr(di0, 'mnemonic')
        assert di0.mnemonic == mnemonic

        if instruction_set[mnemonic]["arguments"]:
            assert hasattr(di0, 'operands')
            assert hasattr(di0.operands, 'all')
            assert hasattr(di0.operands, 'attributes')
            assert all(0 == value for value in di0.operands.all.values())
            assert all(0 == value for value in di0.operands.attributes.values() if isinstance(value, int))
