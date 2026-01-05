# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for tt-perf master operator perf lookup."""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from tools.perf_lookup.tt_perf_master_schema import MASTER_SINGLE_STAT_KEYS
from ttsim.ops.tensor import SimTensor


def _flat_single_value(num_cores: int, **stat_overrides):
    """Valid ``entry_type: single`` payload (all ``MASTER_SINGLE_STAT_KEYS`` + ``num_cores``)."""
    out = {
        "entry_type": "single",
        "num_cores": num_cores,
        "msecs": 0.0,
        "memory_traffic": 0.0,
        "mem_util": 0.0,
        "noc_util": 0.0,
        "noc_multicast_util": 0.0,
        "npe_cong_impact_pct": 0.0,
        "vector_pipe_util": 0.0,
        "matrix_pipe_util": 0.0,
    }
    for k in MASTER_SINGLE_STAT_KEYS:
        assert k in out
    out.update(stat_overrides)
    return out


def _hybrid_single_branch(num_cores: int, **stat_overrides) -> dict:
    """Inner ``hybrid.single`` mapping (no ``entry_type``)."""
    inner = _flat_single_value(num_cores, **stat_overrides)
    del inner["entry_type"]
    return inner


def _curve_stat(a: float, b: float, r2: float = 1.0, equation: str = "x") -> dict:
    return {"a": a, "b": b, "r2": r2, "equation": equation}


@pytest.mark.unit
def test_build_master_key_tuple_rank3_and_rank2():
    from tools.perf_lookup.lookup_operator_perf import build_master_key_tuple_15

    t0 = SimTensor(
        {
            "name": "a",
            "shape": [8, 224, 768],
            "op_in": [],
            "op_out": [],
        }
    )
    t1 = SimTensor(
        {
            "name": "b",
            "shape": [768, 768],
            "op_in": [],
            "op_out": [],
        }
    )
    op = SimpleNamespace(optype="Matmul", precision="BF16", inList=["a", "b"])
    key = build_master_key_tuple_15(op, t0, t1)
    assert key[0] == "matmul"
    assert key[1:5] == (1, 8, 224, 768)
    assert key[5:8] == ("TILE", "BFLOAT16", "DEV_1_DRAM_INTERLEAVED")
    assert key[8:12] == (1, 1, 768, 768)
    assert key[12:15] == ("TILE", "BFLOAT16", "DEV_1_DRAM_INTERLEAVED")


@pytest.mark.unit
def test_build_master_key_tuple_22():
    from tools.perf_lookup.lookup_operator_perf import build_master_key_tuple_22

    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 3, 4], "op_in": [], "op_out": []})
    t2 = SimTensor({"name": "c", "shape": [1, 4, 5], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Add", precision="BF16", inList=["a", "b", "c"])
    key = build_master_key_tuple_22(op, t0, t1, t2)
    assert len(key) == 22
    assert key[0] == "add"
    assert key[1:5] == (1, 1, 2, 3)
    assert key[8:12] == (1, 1, 3, 4)
    assert key[15:19] == (1, 1, 4, 5)


@pytest.mark.unit
def test_build_master_key_tuple_ignores_padded_shape_for_keys():
    """LUT keys use logical ``shape`` even when a tensor exposes a different ``padded_shape()``."""
    from ttsim.ops.tensor import Shape
    from tools.perf_lookup.lookup_operator_perf import build_master_key_tuple_15

    class _TileTensor:
        def __init__(self) -> None:
            self.shape = Shape([8, 14, 14, 1024])
            self.dtype = np.dtype(np.float16)
            self.layout = SimpleNamespace(name="TILE_LAYOUT")

        def padded_shape(self) -> Shape:
            return Shape([8, 14, 32, 1024])

    t0 = _TileTensor()
    t1 = SimTensor(
        {
            "name": "b",
            "shape": [1024, 768],
            "op_in": [],
            "op_out": [],
        }
    )
    op = SimpleNamespace(optype="Matmul", precision="BF16", inList=["a", "b"])
    key = build_master_key_tuple_15(op, t0, t1)
    assert key[1:5] == (8, 14, 14, 1024)
    assert key[8:12] == (1, 1, 1024, 768)


@pytest.mark.unit
def test_build_master_key_tuple_8():
    from tools.perf_lookup.lookup_operator_perf import build_master_key_tuple_8

    t0 = SimTensor(
        {
            "name": "x",
            "shape": [1, 197, 768],
            "op_in": [],
            "op_out": [],
        }
    )
    op = SimpleNamespace(optype="Softmax", precision="BF16", inList=["x"])
    key = build_master_key_tuple_8(op, t0)
    assert len(key) == 8
    assert key[0] == "softmax"
    assert key[1:5] == (1, 1, 197, 768)
    assert key[5:8] == ("TILE", "BFLOAT16", "DEV_1_DRAM_INTERLEAVED")


@pytest.mark.unit
def test_reshape_master_key_packs_input0_wzyx():
    """LUT uses (1, 1, w*z*y, x) for reshape input_0 vs raw rank-4 logical shape."""
    from tools.perf_lookup.lookup_operator_perf import build_master_key_tuple_8, build_master_key_tuple_15

    t0 = SimTensor(
        {
            "name": "a",
            "shape": [1, 8, 224, 768],
            "dtype": np.dtype(np.float16),
            "op_in": [],
            "op_out": [],
        }
    )
    t0.layout = SimpleNamespace(name="ROW_MAJOR_LAYOUT")
    t1 = SimTensor(
        {
            "name": "b",
            "shape": [1, 1, 1, 4],
            "dtype": np.dtype(np.int8),
            "op_in": [],
            "op_out": [],
        }
    )
    op8 = SimpleNamespace(optype="Reshape", precision="BF16", inList=["a"])
    k8 = build_master_key_tuple_8(op8, t0)
    assert k8[0] == "reshape"
    assert k8[1:5] == (1, 1, 1 * 8 * 224, 768)
    op15 = SimpleNamespace(optype="Reshape", precision="BF16", inList=["a", "b"])
    k15 = build_master_key_tuple_15(op15, t0, t1)
    assert k15[0] == "reshape"
    assert k15[1:5] == (1, 1, 1792, 768)
    assert k15[8:12] == (1, 1, 1, 4)


@pytest.mark.unit
def test_numpy_float16_storage_maps_to_bfloat16_lut_with_bf16_precision():
    """TTNN stores logical BF16 as ``np.float16``; LUT keys use ``BFLOAT16``."""
    from tools.perf_lookup.lookup_operator_perf import build_master_key_tuple_8

    t0 = SimTensor(
        {
            "name": "x",
            "shape": [1, 1, 197, 768],
            "dtype": np.dtype(np.float16),
            "op_in": [],
            "op_out": [],
        }
    )
    op = SimpleNamespace(optype="Softmax", precision="bf16", inList=["x"])
    key = build_master_key_tuple_8(op, t0)
    assert key[6] == "BFLOAT16"


@pytest.mark.unit
def test_numpy_float16_storage_maps_to_float16_lut_with_fp16_precision():
    from tools.perf_lookup.lookup_operator_perf import build_master_key_tuple_8

    t0 = SimTensor(
        {
            "name": "x",
            "shape": [1, 1, 197, 768],
            "dtype": np.dtype(np.float16),
            "op_in": [],
            "op_out": [],
        }
    )
    op = SimpleNamespace(optype="Softmax", precision="fp16", inList=["x"])
    key = build_master_key_tuple_8(op, t0)
    assert key[6] == "FLOAT16"


@pytest.mark.unit
def test_unary_single_lookup_hit(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "softmax",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(8, msecs=0.03, mem_util=20.0),
            }
        ],
    }
    p = tmp_path / "unary.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Softmax", precision="BF16", inList=["a"])
    g = SimpleNamespace(_tensors={"a": t0})

    st = m.lookup(op, g, core_count=8)
    assert st is not None
    assert st.msecs == pytest.approx(0.03)
    assert st.mem_util == pytest.approx(20.0)


@pytest.mark.unit
def test_lut_yaml_missing_matrix_pipe_util_rejected_at_load(tmp_path: Path):
    """Loader requires full single payload including matrix/vector util keys."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    bad_value = {k: v for k, v in _flat_single_value(8).items() if k != "matrix_pipe_util"}
    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "softmax",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": bad_value,
            }
        ],
    }
    p = tmp_path / "bad_matrix.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="matrix_pipe_util"):
        OperatorPerfMap(p)


@pytest.mark.unit
def test_curve_hit_missing_vector_pipe_util_raises_at_lookup(tmp_path: Path):
    """Curve rows may omit stats in YAML; resolver returns null → OperatorPerfLUTValidationError."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfLUTValidationError, OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "add",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 1,
                    "input_0_x_pad_logical": 1,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 1,
                    "input_1_z_pad_logical": 1,
                    "input_1_y_pad_logical": 1,
                    "input_1_x_pad_logical": 1,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": {
                    "entry_type": "curve",
                    "curve_family": "linear",
                    "msecs": _curve_stat(0.01, 0.2),
                    "matrix_pipe_util": _curve_stat(0.5, 10.0),
                },
            }
        ],
    }
    p = tmp_path / "curve_no_vector.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)
    t0 = SimTensor({"name": "a", "shape": [1, 1, 1, 1], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 1, 1, 1], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="add", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})
    with pytest.raises(OperatorPerfLUTValidationError, match="vector_pipe_util"):
        m.lookup(op, g, core_count=10)


@pytest.mark.unit
def test_lut_util_percent_out_of_range_raises(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfLUTValidationError, OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "softmax",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(8, msecs=0.03, matrix_pipe_util=100.01),
            }
        ],
    }
    p = tmp_path / "bad_pct.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)
    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Softmax", precision="BF16", inList=["a"])
    g = SimpleNamespace(_tensors={"a": t0})
    with pytest.raises(OperatorPerfLUTValidationError, match="matrix_pipe_util"):
        m.lookup(op, g, core_count=8)


@pytest.mark.unit
def test_lut_optional_noc_util_out_of_range_raises(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfLUTValidationError, OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "softmax",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(8, msecs=0.03, noc_util=-0.1),
            }
        ],
    }
    p = tmp_path / "bad_noc.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)
    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Softmax", precision="BF16", inList=["a"])
    g = SimpleNamespace(_tensors={"a": t0})
    with pytest.raises(OperatorPerfLUTValidationError, match="noc_util"):
        m.lookup(op, g, core_count=8)


@pytest.mark.unit
def test_binary_mul_falls_back_to_unary_lut_key(tmp_path: Path):
    """Master YAML may only have 8-tuple ``mul`` (input_0); graph has two operands (15-tuple)."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "mul",
                    "input_0_w_pad_logical": 8,
                    "input_0_z_pad_logical": 12,
                    "input_0_y_pad_logical": 224,
                    "input_0_x_pad_logical": 224,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(64, msecs=0.095, vector_pipe_util=51.0),
            }
        ],
    }
    p = tmp_path / "mul_unary_only.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "a", "shape": [8, 12, 224, 224], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 1, 1, 1], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Mul", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})

    st = m.lookup(op, g, core_count=64)
    assert st is not None
    assert st.msecs == pytest.approx(0.095)
    assert st.vector_pipe_util == pytest.approx(51.0)


@pytest.mark.unit
def test_ternary_lookup_22_tuple_hit(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "add",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 1,
                    "input_1_z_pad_logical": 1,
                    "input_1_y_pad_logical": 3,
                    "input_1_x_pad_logical": 4,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_2_w_pad_logical": 1,
                    "input_2_z_pad_logical": 1,
                    "input_2_y_pad_logical": 4,
                    "input_2_x_pad_logical": 5,
                    "input_2_layout": "TILE",
                    "input_2_datatype": "BFLOAT16",
                    "input_2_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(
                    8, msecs=0.321, matrix_pipe_util=22.0, vector_pipe_util=33.0
                ),
            }
        ],
    }
    p = tmp_path / "ternary_add.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 3, 4], "op_in": [], "op_out": []})
    t2 = SimTensor({"name": "c", "shape": [1, 4, 5], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="add", precision="BF16", inList=["a", "b", "c"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1, "c": t2})

    st = m.lookup(op, g, core_count=8)
    assert st is not None
    assert st.msecs == pytest.approx(0.321)
    assert st.matrix_pipe_util == pytest.approx(22.0)
    assert st.vector_pipe_util == pytest.approx(33.0)


@pytest.mark.unit
def test_add_broadcast_fallback_second_operand_is_1_1_1_x(tmp_path: Path):
    """LUT row has both add inputs at full WZYX; graph has (1,1,1,X) as second operand."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "add",
                    "input_0_w_pad_logical": 8,
                    "input_0_z_pad_logical": 14,
                    "input_0_y_pad_logical": 14,
                    "input_0_x_pad_logical": 768,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 8,
                    "input_1_z_pad_logical": 14,
                    "input_1_y_pad_logical": 14,
                    "input_1_x_pad_logical": 768,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(64, msecs=0.042, matrix_pipe_util=1.0, vector_pipe_util=2.0),
            }
        ],
    }
    p = tmp_path / "add_dup_full_lut.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "a", "shape": [8, 14, 14, 768], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 1, 1, 768], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Add", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})

    st = m.lookup(op, g, core_count=64)
    assert st is not None
    assert st.msecs == pytest.approx(0.042)
    assert st.matrix_pipe_util == pytest.approx(1.0)
    assert st.vector_pipe_util == pytest.approx(2.0)


@pytest.mark.unit
def test_add_broadcast_fallback_first_operand_is_1_1_1_x(tmp_path: Path):
    """Same LUT as duplicate-full add; graph has (1,1,1,X) as first operand (option B)."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "add",
                    "input_0_w_pad_logical": 8,
                    "input_0_z_pad_logical": 14,
                    "input_0_y_pad_logical": 14,
                    "input_0_x_pad_logical": 768,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 8,
                    "input_1_z_pad_logical": 14,
                    "input_1_y_pad_logical": 14,
                    "input_1_x_pad_logical": 768,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(64, msecs=0.051, matrix_pipe_util=3.0, vector_pipe_util=4.0),
            }
        ],
    }
    p = tmp_path / "add_dup_full_lut_swap.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "bias", "shape": [1, 1, 1, 768], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "act", "shape": [8, 14, 14, 768], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Add", precision="BF16", inList=["bias", "act"])
    g = SimpleNamespace(_tensors={"bias": t0, "act": t1})

    st = m.lookup(op, g, core_count=64)
    assert st is not None
    assert st.msecs == pytest.approx(0.051)
    assert st.matrix_pipe_util == pytest.approx(3.0)
    assert st.vector_pipe_util == pytest.approx(4.0)


@pytest.mark.unit
def test_binary_reshape_falls_back_to_unary_lut_key(tmp_path: Path):
    """LUT has unary ``reshape``; graph pairs data tensor with a small shape constant (15-tuple miss)."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "reshape",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 1792,
                    "input_0_x_pad_logical": 768,
                    "input_0_layout": "ROW_MAJOR",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(64, msecs=0.012, vector_pipe_util=10.0),
            }
        ],
    }
    p = tmp_path / "reshape_unary_only.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor(
        {
            "name": "a",
            "shape": [1, 8, 224, 768],
            "dtype": np.dtype(np.float16),
            "op_in": [],
            "op_out": [],
        }
    )
    t0.layout = SimpleNamespace(name="ROW_MAJOR_LAYOUT")
    t1 = SimTensor(
        {
            "name": "shape_const",
            "shape": [1, 1, 1, 4],
            "dtype": np.dtype(np.int8),
            "op_in": [],
            "op_out": [],
        }
    )
    t1.layout = SimpleNamespace(name="ROW_MAJOR_LAYOUT")
    op = SimpleNamespace(optype="Reshape", precision="BF16", inList=["a", "shape_const"])
    g = SimpleNamespace(_tensors={"a": t0, "shape_const": t1})

    st = m.lookup(op, g, core_count=64)
    assert st is not None
    assert st.msecs == pytest.approx(0.012)
    assert st.vector_pipe_util == pytest.approx(10.0)


@pytest.mark.unit
def test_unary_lookup_miss_returns_none(tmp_path: Path):
    """No table row: lookup returns None without requiring binary-only path."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [],
    }
    p = tmp_path / "empty.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "a", "shape": [1, 1, 1, 1], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Relu", precision="BF16", inList=["a"])
    g = SimpleNamespace(_tensors={"a": t0})

    assert m.lookup(op, g, core_count=8) is None


@pytest.mark.unit
def test_arity_zero_skips_lookup(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [],
    }
    p = tmp_path / "e2.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    op = SimpleNamespace(optype="Const", precision="BF16", inList=[])
    g = SimpleNamespace(_tensors={})
    assert m.lookup(op, g, core_count=8) is None


@pytest.mark.unit
def test_operator_perf_map_loads_repo_sample():
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    root = Path(__file__).resolve().parents[2]
    yml = root / "__ext" / "perf_lookup" / "whb0_n150_lut.yaml"
    if not yml.is_file():
        pytest.skip(f"repo LUT not present: {yml}")
    m = OperatorPerfMap(yml)
    assert len(m) > 0


@pytest.mark.unit
def test_hybrid_matmul_lookup_mocked_master_load(tmp_path: Path):
    """Hybrid matmul resolves ``single`` branch (``use_hybrid_curve`` default) without reading ``__ext`` LUT files.

    Build a valid master YAML under ``tmp_path``, parse it once with the real ``load_existing_yaml``, then
    patch ``lookup_operator_perf.load_existing_yaml`` so :class:`OperatorPerfMap` uses that dict while opening
    only a sentinel path (stable if repo LUT YAML is regenerated).
    """
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap
    from tools.perf_lookup.tt_perf_master_loader import load_existing_yaml

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "matmul",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 8,
                    "input_0_y_pad_logical": 197,
                    "input_0_x_pad_logical": 768,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 1,
                    "input_1_z_pad_logical": 1,
                    "input_1_y_pad_logical": 768,
                    "input_1_x_pad_logical": 768,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": {
                    "entry_type": "hybrid",
                    "single": _hybrid_single_branch(
                        24,
                        msecs=0.313104,
                        matrix_pipe_util=14.27,
                        memory_traffic=8656699.39,
                        mem_util=9.6,
                        noc_util=3.3,
                    ),
                    "curve": {
                        "curve_family": "linear",
                        "msecs": _curve_stat(0.001, 0.2, equation="msecs"),
                        "matrix_pipe_util": _curve_stat(0.1, 5.0, equation="matrix"),
                        "vector_pipe_util": _curve_stat(0.0, 1.0, equation="vector"),
                    },
                },
            }
        ],
    }
    embedded = tmp_path / "synthetic_master.yaml"
    embedded.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    loaded = load_existing_yaml(embedded)

    sentinel = tmp_path / "operator_perf_sentinel.yaml"
    sentinel.write_text("", encoding="utf-8")

    with patch(
        "tools.perf_lookup.lookup_operator_perf.load_existing_yaml",
        return_value=loaded,
    ) as mock_load:
        m = OperatorPerfMap(sentinel)
        mock_load.assert_called_once()
        assert mock_load.call_args[0][0] == sentinel

    t0 = SimTensor(
        {
            "name": "a",
            "shape": [8, 197, 768],
            "op_in": [],
            "op_out": [],
        }
    )
    t1 = SimTensor(
        {
            "name": "b",
            "shape": [768, 768],
            "op_in": [],
            "op_out": [],
        }
    )
    op = SimpleNamespace(optype="matmul", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})
    st = m.lookup(op, g, core_count=24)
    assert st is not None
    assert st.msecs == pytest.approx(0.313104)


@pytest.mark.unit
def test_single_entry_multi_stat_lookup(tmp_path: Path):
    """Binary non-matmul: ``matmul`` rows must be ``hybrid`` in the master loader."""
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "add",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 1,
                    "input_1_z_pad_logical": 1,
                    "input_1_y_pad_logical": 3,
                    "input_1_x_pad_logical": 4,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(
                    8,
                    msecs=0.5,
                    memory_traffic=1024.0,
                    mem_util=40.0,
                    vector_pipe_util=10.0,
                    matrix_pipe_util=70.0,
                ),
            }
        ],
    }
    p = tmp_path / "m.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 3, 4], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="add", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})

    st = m.lookup(op, g, core_count=8)
    assert st is not None
    assert st.msecs == pytest.approx(0.5)
    assert st.memory_traffic == pytest.approx(1024.0)
    assert st.mem_util == pytest.approx(40.0)
    assert st.vector_pipe_util == pytest.approx(10.0)
    assert st.matrix_pipe_util == pytest.approx(70.0)


@pytest.mark.unit
def test_curve_linear_multi_stat(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = textwrap.dedent(
        """
        schema_name: correqn.tt-perf-master
        schema_version: 1
        entries:
          - key:
              op_code: add
              input_0_w_pad_logical: 1
              input_0_z_pad_logical: 1
              input_0_y_pad_logical: 2
              input_0_x_pad_logical: 3
              input_0_layout: TILE
              input_0_datatype: BFLOAT16
              input_0_memory: DEV_1_DRAM_INTERLEAVED
              input_1_w_pad_logical: 1
              input_1_z_pad_logical: 1
              input_1_y_pad_logical: 3
              input_1_x_pad_logical: 4
              input_1_layout: TILE
              input_1_datatype: BFLOAT16
              input_1_memory: DEV_1_DRAM_INTERLEAVED
            value:
              entry_type: curve
              curve_family: linear
              msecs:
                a: 0.01
                b: 0.2
                r2: 0.99
                equation: "msecs linear"
              matrix_pipe_util:
                a: 0.5
                b: 10.0
                r2: 0.95
                equation: "matrix linear"
              vector_pipe_util:
                a: 0.1
                b: 3.0
                r2: 1.0
                equation: "vector linear"
        """
    )
    p = tmp_path / "c.yaml"
    p.write_text(doc, encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 3, 4], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="add", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})

    st = m.lookup(op, g, core_count=10)
    assert st is not None
    assert st.msecs == pytest.approx(0.01 * 10 + 0.2)
    assert st.matrix_pipe_util == pytest.approx(0.5 * 10 + 10.0)
    assert st.vector_pipe_util == pytest.approx(0.1 * 10 + 3.0)
    assert st.memory_traffic is None


@pytest.mark.unit
def test_curve_power_msecs(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "add",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 1,
                    "input_0_x_pad_logical": 1,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 1,
                    "input_1_z_pad_logical": 1,
                    "input_1_y_pad_logical": 1,
                    "input_1_x_pad_logical": 1,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": {
                    "entry_type": "curve",
                    "curve_family": "power",
                    "msecs": _curve_stat(2.0, 0.5),
                    "matrix_pipe_util": _curve_stat(2.5, 0.5),
                    "vector_pipe_util": _curve_stat(3.0, 0.5),
                },
            }
        ],
    }
    p = tmp_path / "pow.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)

    t0 = SimTensor({"name": "a", "shape": [1, 1, 1, 1], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 1, 1, 1], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="add", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})

    st = m.lookup(op, g, core_count=4)
    assert st is not None
    assert st.msecs == pytest.approx(2.0 * (4**0.5))
    assert st.matrix_pipe_util == pytest.approx(2.5 * (4**0.5))
    assert st.vector_pipe_util == pytest.approx(3.0 * (4**0.5))


@pytest.mark.unit
def test_hybrid_matmul_uses_single_when_use_hybrid_curve_false(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "matmul",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 1,
                    "input_1_z_pad_logical": 1,
                    "input_1_y_pad_logical": 3,
                    "input_1_x_pad_logical": 4,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": {
                    "entry_type": "hybrid",
                    "single": _hybrid_single_branch(
                        8,
                        msecs=9.99,
                        matrix_pipe_util=99.0,
                    ),
                    "curve": {
                        "curve_family": "linear",
                        "msecs": _curve_stat(0.02, 0.1, equation="msecs"),
                        "matrix_pipe_util": _curve_stat(0.1, 5.0, equation="matrix"),
                    },
                },
            }
        ],
    }
    p = tmp_path / "hyb.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p, use_hybrid_curve=False)

    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 3, 4], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="matmul", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})

    core = 10
    st = m.lookup(op, g, core_count=core)
    assert st is not None
    assert st.msecs == pytest.approx(9.99)
    assert st.matrix_pipe_util == pytest.approx(99.0)


@pytest.mark.unit
def test_hybrid_matmul_uses_curve_when_use_hybrid_curve_true(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap, _eval_curve_value
    from tools.perf_lookup.tt_perf_master_schema import MASTER_CURVE_FAMILY_LINEAR

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "matmul",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                    "input_1_w_pad_logical": 1,
                    "input_1_z_pad_logical": 1,
                    "input_1_y_pad_logical": 3,
                    "input_1_x_pad_logical": 4,
                    "input_1_layout": "TILE",
                    "input_1_datatype": "BFLOAT16",
                    "input_1_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": {
                    "entry_type": "hybrid",
                    "single": _hybrid_single_branch(
                        8,
                        msecs=9.99,
                        matrix_pipe_util=99.0,
                    ),
                    "curve": {
                        "curve_family": "linear",
                        "msecs": _curve_stat(0.02, 0.1, equation="msecs"),
                        "matrix_pipe_util": _curve_stat(0.1, 5.0, equation="matrix"),
                        "vector_pipe_util": _curve_stat(0.05, 2.0, equation="vector"),
                    },
                },
            }
        ],
    }
    p = tmp_path / "hyb_curve.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p, use_hybrid_curve=True)

    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    t1 = SimTensor({"name": "b", "shape": [1, 3, 4], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="matmul", precision="BF16", inList=["a", "b"])
    g = SimpleNamespace(_tensors={"a": t0, "b": t1})

    core = 10
    st = m.lookup(op, g, core_count=core)
    assert st is not None
    expected_m = _eval_curve_value(MASTER_CURVE_FAMILY_LINEAR, 0.02, 0.1, core)
    assert st.msecs == pytest.approx(expected_m)
    expected_matrix = _eval_curve_value(MASTER_CURVE_FAMILY_LINEAR, 0.1, 5.0, core)
    assert st.matrix_pipe_util == pytest.approx(expected_matrix)
    expected_vector = _eval_curve_value(MASTER_CURVE_FAMILY_LINEAR, 0.05, 2.0, core)
    assert st.vector_pipe_util == pytest.approx(expected_vector)
    assert st.msecs != pytest.approx(9.99)


@pytest.mark.unit
def test_loader_normalizes_non_finite_util_to_zero(tmp_path: Path):
    from tools.perf_lookup.lookup_operator_perf import OperatorPerfMap

    doc = {
        "schema_name": "correqn.tt-perf-master",
        "schema_version": 1,
        "entries": [
            {
                "key": {
                    "op_code": "softmax",
                    "input_0_w_pad_logical": 1,
                    "input_0_z_pad_logical": 1,
                    "input_0_y_pad_logical": 2,
                    "input_0_x_pad_logical": 3,
                    "input_0_layout": "TILE",
                    "input_0_datatype": "BFLOAT16",
                    "input_0_memory": "DEV_1_DRAM_INTERLEAVED",
                },
                "value": _flat_single_value(
                    8,
                    msecs=0.03,
                    matrix_pipe_util=float("nan"),
                    vector_pipe_util=float("inf"),
                    mem_util=float("-inf"),
                ),
            }
        ],
    }
    p = tmp_path / "util_nan_inf.yaml"
    p.write_text(yaml.dump(doc, sort_keys=False), encoding="utf-8")
    m = OperatorPerfMap(p)
    t0 = SimTensor({"name": "a", "shape": [1, 2, 3], "op_in": [], "op_out": []})
    op = SimpleNamespace(optype="Softmax", precision="BF16", inList=["a"])
    g = SimpleNamespace(_tensors={"a": t0})

    st = m.lookup(op, g, core_count=8)
    assert st is not None
    assert st.matrix_pipe_util == pytest.approx(0.0)
    assert st.vector_pipe_util == pytest.approx(0.0)
    assert st.mem_util == pytest.approx(0.0)
