# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Shared helpers for the Gemma3 Polaris unit tests in this directory.

Polaris/ttsim is a shape- and op-graph-driven performance simulator (see
GEMMA3_POLARIS_AUDIT.md and ttsim/ops/tensor.py:SimTensor) -- it never computes or checks
real numeric values. So unlike the tt-metal reference tests this directory mirrors (which
compare PCC/allclose against an HF/torch reference model), every test here validates output
*shape* only. That's not a weaker test than PCC by accident: it's the only thing Polaris's
own execution model makes meaningful. `ModelArgs.reference_*()` stubs (model_config.py) all
return `None` or no-op passthrough wrappers in simulation mode for exactly this reason.
"""
import numpy as np
from loguru import logger

import ttsim.front.ttnn as ttnn
from ttsim.front.ttnn.device import Device


def get_device(name="test_device"):
    return Device(name=name)


def get_shape_tuple(tensor):
    """Get shape as a plain tuple from a ttnn.Tensor (or anything shape-like)."""
    if hasattr(tensor, "shape"):
        shape = tensor.shape
        if hasattr(shape, "__iter__"):
            return tuple(int(d) for d in shape)
        return shape
    if hasattr(tensor, "get_shape"):
        return tuple(int(d) for d in tensor.get_shape())
    return ()


def check_shape(actual_tensor, expected_shape, test_name):
    """
    Shape-only validation for ttsim simulation mode.

    Compares the trailing len(expected_shape) dimensions, ignoring any extra leading
    dimensions -- Polaris tensors frequently carry a leading (1, ...) multi-device/replication
    dim that the reference torch shape doesn't have.
    """
    actual = get_shape_tuple(actual_tensor)
    expected = tuple(int(d) for d in expected_shape)

    actual_core = actual[-len(expected):] if len(actual) >= len(expected) else actual
    passed = actual_core == expected

    icon = "\u2705" if passed else "\u274c"
    logger.info(f"{icon} {test_name}: expected core shape {expected}, got {actual} (core {actual_core})")
    return passed


def safe_deallocate(tensor):
    if hasattr(ttnn, "deallocate"):
        try:
            ttnn.deallocate(tensor)
        except Exception:
            pass
    elif hasattr(tensor, "deallocate"):
        try:
            tensor.deallocate()
        except Exception:
            pass


def numpy_to_ttnn_tensor(np_array, device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                         memory_config=None):
    """Wrap a numpy array's shape/dtype into a ttnn.Tensor (values are not retained/read)."""
    kwargs = dict(dtype=dtype, layout=layout, device=device)
    if memory_config is not None:
        kwargs["memory_config"] = memory_config
    return ttnn.Tensor(np_array, **kwargs)


def print_summary(results, title):
    logger.info("\n" + "=" * 60)
    logger.info(title)
    logger.info("=" * 60)
    all_passed = True
    for k, v in results.items():
        passed = isinstance(v, str) and v.startswith("PASSED")
        all_passed = all_passed and passed
        icon = "\u2705" if passed else "\u274c"
        logger.info(f" {icon} {k}: {v}")
    return all_passed
