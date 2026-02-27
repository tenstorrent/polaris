#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np

def generate_test_data(shape, data_type):
    """Generate test data based on type
    Args:
        shape: tuple or list, shape of the array
        data_type: str, one of ['positive', 'negative', 'zeros', 'mixed', 'small', 'large']
    Returns:
        np.ndarray: generated test data
    """
    if data_type == "positive":
        return np.random.rand(*shape).astype(np.float32) + 1.0  # Range [1, 2]
    elif data_type == "negative":
        return -np.random.rand(*shape).astype(np.float32) - 1.0  # Range [-2, -1]
    elif data_type == "zeros":
        return np.zeros(shape, dtype=np.float32)
    elif data_type == "mixed":
        return (np.random.randn(*shape) * 2).astype(np.float32)  # Mixed pos/neg
    elif data_type == "small":
        return np.random.rand(*shape).astype(np.float32) * 1e-6  # Very small
    elif data_type == "large":
        return np.random.rand(*shape).astype(np.float32) * 1e6  # Very large
    else:
        return np.random.randn(*shape).astype(np.float32)
