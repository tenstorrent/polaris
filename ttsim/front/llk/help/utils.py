#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import typing


def get_user_attrs(obj: object) -> list[str]:
    # Get only user-defined attributes (no methods, no __ attrs)
    return sorted({attr for attr in dir(obj) if not attr.startswith('__') and not callable(getattr(obj, attr))})

def from_ints_to_bytes(ints: list[int], num_bytes_per_int = 4, byteorder: str = 'little') -> bytes:
    if byteorder not in ('little', 'big'):
        byte_orders = set(['little', 'big'])
        raise Exception(f"- error: incorrect byteorder. Accepted arguments: {byte_orders}")

    if not isinstance(ints, (list, tuple)):
        raise Exception(f"- error: expected a list of int, received type {type(ints)}")

    ints_in_bytes = bytearray(len(ints) * num_bytes_per_int)
    for idx, ele in enumerate(ints):
        offset = idx * num_bytes_per_int
        ints_in_bytes[offset : (offset + num_bytes_per_int)] = ele.to_bytes(
            num_bytes_per_int,
            byteorder = typing.cast(typing.Literal['little', 'big'], byteorder),
            signed = ele < 0)

    return bytes(ints_in_bytes)

def from_bytes_to_ints(arg_bytes: bytes, num_bytes_per_int = 4, is_signed: bool | list[bool] = False, byteorder: str = 'little') -> list[int]:
    if byteorder not in ('little', 'big'):
        byte_orders = set(['little', 'big'])
        raise Exception(f"- error: incorrect byteorder. Accepted arguments: {byte_orders}")

    # num_ints = int(int(len(arg_bytes)/num_bytes_per_int) * num_bytes_per_int)
    num_ints = int(len(arg_bytes) // num_bytes_per_int)
    if len(arg_bytes) % num_bytes_per_int:
        print(f"- warning: number of bytes are not commensurate with num_bytes_per_int, last {len(arg_bytes) % num_bytes_per_int} bytes would be ignored, len(bytes) = {len(arg_bytes)}, num_bytes_per_int = {num_bytes_per_int}")

    if isinstance(is_signed, (list, tuple)):
        assert len(is_signed) == num_ints, f"- error: is_signed list size mismatch. num_ints = {num_ints}, len(arg_bytes) = {len(arg_bytes)}, num_bytes_per_int = {num_bytes_per_int},len(is_signed) = {len(is_signed)}"

    ints = [int.from_bytes(
        arg_bytes[i : (i + num_bytes_per_int)],
        byteorder = typing.cast(typing.Literal['little', 'big'], byteorder),
        signed = is_signed[i // num_bytes_per_int] if isinstance(is_signed, (list, tuple)) else is_signed)
        for i in range(0, (num_ints * num_bytes_per_int), num_bytes_per_int)]

    return ints

def left_circular_shift(value: int, shift: int, num_bits: int) -> int:
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

    shift %= num_bits # makes any shift value safe

    return ((value << shift) | (value >> (num_bits - shift))) & ((1 << num_bits) - 1)

def right_circular_shift(value: int, shift: int, num_bits: int) -> int:
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

    shift %= num_bits # makes any shift value safe

    return ((value >> shift) | (value << (num_bits - shift))) & ((1 << num_bits) - 1)

def from_int_to_hex_str(value: int, min_num_chars: int = 8):
    min_num_chars = int(max((value.bit_length()/4) + 1 if value.bit_length() % 4 else 0, min_num_chars))
    return f"0x{value:0{min_num_chars}x}"
    # return f"{value:#0{min_num_chars + 2}x}" #2 is for adding 0x