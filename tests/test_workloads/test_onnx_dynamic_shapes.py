#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for YAML-driven dynamic shape support in onnx2nx.py."""

import os
import tempfile

import onnx
import pytest
from onnx import TensorProto, helper

from ttsim.front.onnx.onnx2nx import (
    _resolve_symbolic_dim,
    onnx2graph,
)

def _build_symbolic_onnx(path: str) -> None:
    """Build a tiny ONNX model with a symbolic batch dim.

    Graph: input X[batch_size, 4]  --MatMul-->  Y[batch_size, 2]
    where W is a (4, 2) initializer.
    """
    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, ["batch_size", 4]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, ["batch_size", 2]
    )
    W = helper.make_tensor(
        "W", TensorProto.FLOAT, [4, 2], [0.0] * 8,
    )
    node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm")
    graph = helper.make_graph([node], "g", [X], [Y], initializer=[W])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, path)

def _build_unmapped_symbolic_onnx(path: str) -> None:
    """Build a tiny ONNX model with a symbolic dim no override will cover."""
    X = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [1, "weird_axis_42"]
    )
    Y = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [1, "weird_axis_42"]
    )
    node = helper.make_node("Identity", ["X"], ["Y"], name="id")
    graph = helper.make_graph([node], "g", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, path)

def test_resolve_exact_name_match():
    assert _resolve_symbolic_dim("data_dynamic_axes_2", {"data_dynamic_axes_2": 128}) == 128

def test_resolve_alias_match():
    assert _resolve_symbolic_dim("batch_size", {"bs": 8}) == 8
    assert _resolve_symbolic_dim("sequence_length", {"seq_len": 1024}) == 1024
    assert _resolve_symbolic_dim("H", {"img_height": 224}) == 224

def test_resolve_axis0_batch_heuristic():
    assert _resolve_symbolic_dim("custom_name", {"bs": 4}, axis=0) == 4

def test_resolve_fallback_to_1():
    """Unmapped symbolic dim falls back to 1 (legacy behavior)."""
    assert _resolve_symbolic_dim(
        "unmapped_dim", {"bs": 8}, tensor_name="t", axis=2,
    ) == 1

def test_onnx2graph_resolves_symbolic_via_alias():
    """batch_size symbolic dim → resolved via 'bs' alias to concrete int."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sym.onnx")
        _build_symbolic_onnx(path)

        g = onnx2graph("sym_test", path, dim_overrides={"bs": 8})

        x_tensor = g._tensors["X"]
        assert x_tensor.shape == [8, 4], f"expected [8, 4], got {x_tensor.shape}"
        y_tensor = g._tensors["Y"]
        assert y_tensor.shape == [8, 2], f"expected [8, 2], got {y_tensor.shape}"

def test_onnx2graph_exact_name_override():
    """Direct symbolic-name override (no alias) works."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sym.onnx")
        _build_symbolic_onnx(path)
        g = onnx2graph("sym_test", path, dim_overrides={"batch_size": 16})
        assert g._tensors["X"].shape == [16, 4]
        assert g._tensors["Y"].shape == [16, 2]

def test_onnx2graph_no_overrides_falls_back_to_1():
    """Without dim_overrides, symbolic dims fall back to 1 (legacy behavior)."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sym.onnx")
        _build_symbolic_onnx(path)
        g = onnx2graph("sym_test", path)
        assert g._tensors["X"].shape == [1, 4]
        assert g._tensors["Y"].shape == [1, 2]

def test_onnx2graph_unmapped_symbolic_falls_back_to_1():
    """Unmapped symbolic dim with no matching override → 1 + warning."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "sym.onnx")
        _build_unmapped_symbolic_onnx(path)
        g = onnx2graph("sym_test", path, dim_overrides={"bs": 99})
        assert g._tensors["X"].shape == [1, 1]

def test_onnx2graph_fixed_shape_unaffected():
    """Models with no symbolic dims behave identically with or without overrides."""
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "fixed.onnx")
        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])
        W = helper.make_tensor("W", TensorProto.FLOAT, [4, 2], [0.0] * 8)
        node = helper.make_node("MatMul", ["X", "W"], ["Y"], name="mm")
        graph = helper.make_graph([node], "g", [X], [Y], initializer=[W])
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 14)]
        )
        model.ir_version = 8
        onnx.save(model, path)

        g1 = onnx2graph("fixed_test", path, dim_overrides={"bs": 99})
        assert g1._tensors["X"].shape == [3, 4]
        g2 = onnx2graph("fixed_test", path)
        assert g2._tensors["X"].shape == [3, 4]