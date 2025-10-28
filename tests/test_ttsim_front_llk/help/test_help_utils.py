#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import typing

import pytest

import ttsim.front.llk.help.utils as help_utils

def test_get_user_attrs():
    class A:
        def __init__(self):
            pass

        def set_a(self, t_a):
            self.a = t_a

    objA = A()
    a = [idx for idx in range(5)]
    objA.set_a(a)

    attr = help_utils.get_user_attrs(objA)
    assert ['a'] == attr

def test_from_bytes_to_ints():
    def test_empty_list():
        # Test that an empty list returns empty bytes#
        result = help_utils.from_ints_to_bytes([])
        assert b'' == result

    def test_positive_integers_default_params():
        # Test with positive integers and default parameters#
        ints = [1, 2, 3]
        result = help_utils.from_ints_to_bytes(ints)
        # In little endian: 1 -> 01 00 00 00, 2 -> 02 00 00 00, 3 -> 03 00 00 00
        expected = b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00'
        assert result == expected

    def test_negative_integers():
        # Test with negative integers#
        ints = [-1, -256, -65536]
        result = help_utils.from_ints_to_bytes(ints)
        # In little endian with signed=True
        expected = b'\xff\xff\xff\xff\x00\xff\xff\xff\x00\x00\xff\xff'
        assert result == expected

    def test_mixed_integers():
        # Test with a mix of positive and negative integers#
        ints = [42, -42, 0]
        result = help_utils.from_ints_to_bytes(ints)
        expected = b'\x2a\x00\x00\x00\xd6\xff\xff\xff\x00\x00\x00\x00'
        assert result == expected

    def test_big_endian():
        # Test with big endian byte order#
        ints = [1, 2, 3]
        result = help_utils.from_ints_to_bytes(ints, byteorder = 'big')
        # In big endian: 1 -> 00 00 00 01, 2 -> 00 00 00 02, 3 -> 00 00 00 03
        expected = b'\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03'
        assert result == expected

    def test_different_bytes_per_int():
        # Test with different number of bytes per integer#
        ints = [255, 256, 65535]
        result = help_utils.from_ints_to_bytes(ints, num_bytes_per_int=2)
        # In little endian with 2 bytes: 255 -> FF 00, 256 -> 00 01, 65535 -> FF FF
        expected = b'\xff\x00\x00\x01\xff\xff'
        assert result == expected

    def test_large_integers():
        # Test with integers close to the limit for the given byte size#
        # Maximum value for 2 bytes unsigned is 65535
        ints = [65535]
        result = help_utils.from_ints_to_bytes(ints, num_bytes_per_int=2)
        expected = b'\xff\xff'
        assert result == expected

        # Test that larger values raise OverflowError
        with pytest.raises(OverflowError):
            help_utils.from_ints_to_bytes([65536], num_bytes_per_int=2)

    def test_zero():
        # Test with zero#
        ints = [0, 0, 0]
        result = help_utils.from_ints_to_bytes(ints, num_bytes_per_int=1)
        expected = b'\x00\x00\x00'
        assert result == expected

    def test_roundtrip_conversion():
        # Test roundtrip conversion using int.from_bytes
        original_ints = [42, -73, 10000]
        bytes_data = help_utils.from_ints_to_bytes(original_ints)

        # Convert back to integers
        reconstructed_ints = []
        for i in range(len(original_ints)):
            offset = i * 4
            int_bytes = bytes_data[offset:offset+4]
            reconstructed_ints.append(int.from_bytes(int_bytes, byteorder='little', signed=original_ints[i] < 0))

        assert original_ints == reconstructed_ints

    def test_exceptions():
        with pytest.raises(Exception):
            help_utils.from_bytes_to_ints(b'', byteorder = 'abc')

    test_empty_list()
    test_positive_integers_default_params()
    test_negative_integers()
    test_mixed_integers()
    test_big_endian()
    test_different_bytes_per_int()
    test_large_integers()
    test_zero()
    test_roundtrip_conversion()
    test_exceptions()

def test_from_ints_to_bytes(capsys):
    def test_empty_bytes():
        # Test with empty bytes
        result = help_utils.from_bytes_to_ints(b'')
        assert result == []

    def test_positive_integers_default_params():
        # Test with positive integers using default parameters
        # In little endian: 1 -> 01 00 00 00, 2 -> 02 00 00 00, 3 -> 03 00 00 00
        bytes_data = b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00'
        result = help_utils.from_bytes_to_ints(bytes_data)
        assert result == [1, 2, 3]

    def test_negative_integers():
        # Test with negative integers
        # In little endian: -1, -256, -65536
        bytes_data = b'\xff\xff\xff\xff\x00\xff\xff\xff\x00\x00\xff\xff'
        result = help_utils.from_bytes_to_ints(bytes_data, is_signed=True)
        assert result == [-1, -256, -65536]

    def test_mixed_signed_unsigned():
        # Test with mixed signed and unsigned values using a list for is_signed
        # 255 (unsigned), -1 (signed), 65535 (unsigned)
        bytes_data = b'\xff\x00\x00\x00\xff\xff\xff\xff\xff\xff\x00\x00'
        result = help_utils.from_bytes_to_ints(bytes_data, is_signed=[False, True, False])
        assert result == [255, -1, 65535]

    def test_big_endian():
        # Test with big endian byte order
        # In big endian: 1, 2, 3
        bytes_data = b'\x00\x00\x00\x01\x00\x00\x00\x02\x00\x00\x00\x03'
        result = help_utils.from_bytes_to_ints(bytes_data, byteorder='big')
        assert result == [1, 2, 3]

    def test_different_bytes_per_int():
        # Test with different number of bytes per integer
        # In little endian with 2 bytes: 255, 256, 65535
        bytes_data = b'\xff\x00\x00\x01\xff\xff'
        result = help_utils.from_bytes_to_ints(bytes_data, num_bytes_per_int=2)
        assert result == [255, 256, 65535]

    def test_non_commensurate_bytes(capsys):
        # Test with bytes that are not a multiple of num_bytes_per_int
        bytes_data = b'\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00'  # 10 bytes, not divisible by 4
        result = help_utils.from_bytes_to_ints(bytes_data)
        # Check that we get 2 integers and ignore the remaining 2 bytes
        assert result == [1, 2]
        # Check that a warning was printed
        captured = capsys.readouterr()
        assert "warning: number of bytes are not commensurate" in captured.out

    def test_is_signed_list_length_mismatch():
        # Test that an error is raised when is_signed list length doesn't match number of integers
        bytes_data = b'\x01\x00\x00\x00\x02\x00\x00\x00'  # 2 integers
        with pytest.raises(AssertionError) as excinfo:
            help_utils.from_bytes_to_ints(bytes_data, is_signed=[True, False, True])
        assert "is_signed list size mismatch" in str(excinfo.value)

    def test_roundtrip_conversion():
        # Test roundtrip conversion using from_ints_to_bytes
        # Define a helper function to convert ints to bytes
        def from_ints_to_bytes(ints, num_bytes_per_int=4, byteorder='little'):
            ints_in_bytes = bytearray(len(ints) * num_bytes_per_int)
            for idx, ele in enumerate(ints):
                offset = idx * num_bytes_per_int
                ints_in_bytes[offset : (offset + num_bytes_per_int)] = ele.to_bytes(
                    num_bytes_per_int, byteorder=byteorder, signed=ele < 0)
            return bytes(ints_in_bytes)

        # Original integers
        original_ints = [42, -73, 10000]
        # Convert to bytes
        bytes_data = from_ints_to_bytes(original_ints)
        # Convert back to integers with is_signed list
        result = help_utils.from_bytes_to_ints(bytes_data, is_signed=[False, True, False])
        assert result == original_ints

    def test_zero_values():
        # Test with zero values
        bytes_data = b'\x00\x00\x00\x00\x00\x00\x00\x00'
        result = help_utils.from_bytes_to_ints(bytes_data, num_bytes_per_int=2)
        assert result == [0, 0, 0, 0]

    def test_exceptions():
        with pytest.raises(Exception):
            help_utils.from_ints_to_bytes([5], byteorder = '')

        with pytest.raises(Exception):
            help_utils.from_ints_to_bytes(typing.cast(typing.Any, 5))

    test_empty_bytes()
    test_positive_integers_default_params()
    test_negative_integers()
    test_mixed_signed_unsigned()
    test_big_endian()
    test_different_bytes_per_int()
    test_non_commensurate_bytes(capsys)
    test_is_signed_list_length_mismatch()
    test_roundtrip_conversion()
    test_zero_values()
    test_exceptions()

class TestCircularShifts:

    @pytest.mark.parametrize("value, shift, num_bits, expected", [
        (0b1,        1,  8, 0b10),        # Basic left shift
        (0b10000000, 1,  8, 0b00000001),  # Bit rotates around
        (0b10101010, 2,  8, 0b10101010),  # Multi-bit value shift
        (0b1,        0,  8, 0b1),         # Zero shift
        (0b00000001, 8,  8, 0b00000001),  # Full rotation (no change)
        (0b00000001, 9,  8, 0b00000010),  # Shift > num_bits
        (0b00000001, 3,  4, 0b1000),      # Different bit length
        (0xF0F0F0F0, 4, 32, 0x0F0F0F0F),  # 32-bit rotation
        (0b101,      2,  3, 0b110),       # Small bit length
    ])
    def test_left_circular_shift(self, value, shift, num_bits, expected):
        result = help_utils.left_circular_shift(value, shift, num_bits)
        assert result == expected

    @pytest.mark.parametrize("value, shift, num_bits, expected", [
        (0b1,        1,  8, 0b10000000),  # Basic right shift
        (0b00000001, 1,  8, 0b10000000),  # Bit rotates around
        (0b10101010, 2,  8, 0b10101010),  # Multi-bit value shift
        (0b1,        0,  8, 0b1),         # Zero shift
        (0b00000001, 8,  8, 0b00000001),  # Full rotation (no change)
        (0b00000001, 9,  8, 0b10000000),  # Shift > num_bits
        (0b1000,     3,  4, 0b0001),      # Different bit length
        (0xF0F0F0F0, 4, 32, 0x0F0F0F0F),  # 32-bit rotation
        (0b101,      2,  3, 0b011),       # Small bit length # 0b101 -> 0b110 -> 0b011
    ])
    def test_right_circular_shift(self, value, shift, num_bits, expected):
        result = help_utils.right_circular_shift(value, shift, num_bits)
        assert result == expected

    def test_shift_modulo_behavior(self):
        # Test that shifts greater than num_bits wrap around correctly#
        value = 0b10101010
        num_bits = 8

        # Left shift by 2 and by 2+8 (10) should give the same result
        assert help_utils.left_circular_shift(value, 2, num_bits) == help_utils.left_circular_shift(value, 10, num_bits)

        # Right shift by 3 and by 3+8 (11) should give the same result
        assert help_utils.right_circular_shift(value, 3, num_bits) == help_utils.right_circular_shift(value, 11, num_bits)

    def test_circular_property(self):
        # Test that left shift followed by right shift returns the original value#
        value = 0b10110101
        num_bits = 8

        for shift in range(1, num_bits):
            # Left shift then right shift should give original value
            left_shifted = help_utils.left_circular_shift(value, shift, num_bits)
            restored = help_utils.right_circular_shift(left_shifted, shift, num_bits)
            assert restored == value

    def test_bitmask_application(self):
        # Test that values are properly masked to num_bits#
        # Using a value with bits set beyond num_bits
        value = 0b1111111111111111  # 16 bits set
        num_bits = 8

        # Result should only have 8 bits set
        result = help_utils.left_circular_shift(value, 4, num_bits)
        assert result < 256

        result = help_utils.right_circular_shift(value, 4, num_bits)
        assert result < 256

    def test_complementary_shifts(self):
        # Test that left shift by x is the same as right shift by (num_bits - x)#
        value = 0b10110101
        num_bits = 8

        for shift in range(num_bits):
            left_result = help_utils.left_circular_shift(value, shift, num_bits)
            right_result = help_utils.right_circular_shift(value, num_bits - shift, num_bits)
            assert left_result == right_result

    def test_exceptions(self):
        with pytest.raises(Exception) as exec_info:
            help_utils.left_circular_shift(5, 0, -1)

        with pytest.raises(Exception) as exec_info:
            help_utils.right_circular_shift(5, 0, 0)

def test_from_int_to_hex_str():
    word = 0x7fbea483
    assert f"0x{word:x}"   == help_utils.from_int_to_hex_str(word)

    word = 0x25b6383
    assert f"0x{word:08x}" == help_utils.from_int_to_hex_str(word)

    word = 0x123fabcde
    assert f"0x{word:x}"   == help_utils.from_int_to_hex_str(0x123fabcde)



