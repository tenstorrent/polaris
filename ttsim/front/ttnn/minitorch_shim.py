#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight PyTorch API shim for TT-NN compatibility.

This module provides a minimal subset of PyTorch APIs used in workloads
to enable code compatibility when running in POLARIS mode (IS_POLARIS=True).

The shim is intended to be imported as:
    import ttsim.front.ttnn.minitorch_shim as torch
"""

# Import dtype constants from __init__.py in the same directory
from . import bfloat16, float32, int64, uint32, bfloat8_b, bool, int32

# Re-export the constants at module level for torch API compatibility
# These are defined in ttsim.front.ttnn.__init__.py as:
#   bfloat16 = DataType.BFLOAT16
#   float32 = DataType.FLOAT32
#   int64 = DataType.INT64
#   uint32 = DataType.UINT32
#   bfloat8_b = DataType.BFLOAT8_B
#   bool = DataType.BOOL
#   int32 = DataType.INT32


def manual_seed(seed: int) -> None:
    """
    Sets the seed for generating random numbers.

    This is a no-op in the TT-NN shim.

    Args:
        seed: The desired seed. Must be convertible to int.

    Returns:
        None
    """
    pass


# Make functions and attributes available at module level for direct access
__all__ = [
    'manual_seed',
    'bfloat16',
    'float32',
    'int64',
    'uint32',
    'bfloat8_b',
    'bool',
    'int32',
]
