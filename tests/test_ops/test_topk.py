#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import ttsim.front.functional.op as F


@pytest.mark.unit
@pytest.mark.opunit
@pytest.mark.parametrize(
    "shape,k,axis",
    [
        ([4], 2, 0),
        ([3, 5], 3, 1),
        ([2, 3, 4], 1, -1),
    ],
)
def test_topk_output_shapes_and_dtypes(shape, k, axis):
    input_tensor = F._from_shape("TopKInput", shape, np_dtype=np.float32)
    op_handle = F.topk("TopKOp", k=k, axis=axis, largest=True, sorted=True)

    values_tensor, indices_tensor = op_handle(input_tensor)

    axis_idx = axis if axis >= 0 else axis + len(shape)
    expected_shape = list(shape)
    expected_shape[axis_idx] = k

    assert values_tensor.shape == expected_shape
    assert indices_tensor.shape == expected_shape
    assert values_tensor.dtype == np.dtype(np.float32)
    assert indices_tensor.dtype == np.dtype(np.int64)


@pytest.mark.unit
@pytest.mark.opunit
def test_topk_invalid_k_raises():
    input_tensor = F._from_shape("TopKInvalidInput", [2, 3], np_dtype=np.float32)
    op_handle = F.topk("TopKInvalid", k=5, axis=1, largest=True, sorted=True)

    with pytest.raises(ValueError):
        op_handle(input_tensor)
