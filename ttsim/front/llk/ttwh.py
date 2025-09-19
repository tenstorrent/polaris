#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import typing

import ttsim.front.llk.decoded_instruction as decoded_instruction
import ttsim.front.llk.tensix as tensix
from ttsim.front.llk.tensix import get_instruction_set_from_file_name as get_instruction_set_from_file_name
from ttsim.front.llk.tensix import get_opcode as get_opcode
from ttsim.front.llk.tensix import instruction_to_str as instruction_to_str
from ttsim.front.llk.tensix import is_valid_instruction as is_valid_instruction
from ttsim.front.llk.tensix import print_instruction as print_instruction


def instruction_kind() -> decoded_instruction.instruction_kind:
    return decoded_instruction.instruction_kind.ttwh

def get_default_instruction_set() -> dict[str, typing.Any]:
    return tensix.get_default_instruction_set_from_kind(instruction_kind())

def get_instruction_set(instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None):
    return tensix.get_instruction_set(kind = instruction_kind(), instruction_set = instruction_set)

def decode_instruction(
    instruction: int,
    instruction_set: None | decoded_instruction.instruction_kind | str | dict[str, typing.Any] | dict[decoded_instruction.instruction_kind, None | decoded_instruction.instruction_kind | str | dict[str, typing.Any]] = None,
    is_swizzled: bool = True):
    return tensix.decode_instruction(instruction, instruction_kind(), instruction_set, is_swizzled)