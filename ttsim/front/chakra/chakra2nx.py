#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load MLCommons Chakra execution traces (.et) into a :class:`WorkloadGraph`."""

from __future__ import annotations

import ast
from typing import Any

import numpy as np
from google.protobuf.json_format import MessageToDict
from loguru import logger

from ttsim.front.chakra.et_def_pb2 import (
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    GlobalMetadata,
    INVALID_NODE,
    METADATA_NODE,
    Node as ChakraNodeProto,
)
from ttsim.ops.desc.registry import get_opdesc_registry
from ttsim.front.chakra.protolib import decodeMessage, openFileRd
from ttsim.graph import WorkloadGraph
from ttsim.ops import SimOp, SimTensor

# Polaris domain for ET rows translated onto ONNX-class ops (distinct from legacy ``chakra.et``).
_CHAKRA_DOMAIN = "chakra"

# Use ``np.dtype`` instances so :meth:`SimTensor.nbytes` and shape inference accept them.
_TYPE_MAP: dict[str, np.dtype] = {
    "float": np.dtype(np.float32),
    "double": np.dtype(np.float64),
    "Half": np.dtype(np.float16),
    "Float": np.dtype(np.float32),
    "BFloat16": np.dtype(np.float32),  # approximate for sizing
    "Int": np.dtype(np.int32),
    "Long": np.dtype(np.int64),
    "Bool": np.dtype(np.bool_),
    "bool": np.dtype(np.bool_),
    "c10::Half": np.dtype(np.float16),
    "c10::BFloat16": np.dtype(np.float32),
    "Scalar": np.dtype(np.float32),
    "Tensor": np.dtype(np.float32),
    "": np.dtype(np.float32),
}


def _literal_list(s: str) -> list[Any]:
    if s is None or not str(s).strip():
        return []
    try:
        v = ast.literal_eval(str(s))
    except (SyntaxError, ValueError, TypeError):
        logger.warning("Could not parse Chakra IO field as Python literal: {!r}", s[:200])
        return []
    return v if isinstance(v, list) else []


def _coerce_shape(raw: Any) -> list[int]:
    if raw is None:
        return [1]
    if isinstance(raw, (list, tuple)):
        out: list[int] = []
        for d in raw:
            if isinstance(d, (int, np.integer)):
                out.append(int(d))
            else:
                out.append(1)
        return out if out else [1]
    return [1]


def _coerce_dtype(type_name: str) -> np.dtype:
    if not type_name:
        return np.dtype(np.float32)
    base = type_name.split(".")[-1]
    return _TYPE_MAP.get(type_name, _TYPE_MAP.get(base, np.dtype(np.float32)))


def _io_lists(io) -> tuple[list[Any], list[Any], list[Any]]:
    if io is None:
        return [], [], []
    return (
        _literal_list(getattr(io, "values", "") or ""),
        _literal_list(getattr(io, "shapes", "") or ""),
        _literal_list(getattr(io, "types", "") or ""),
    )


def _tensor_names_values_shapes_types(
    values: list[Any], shapes: list[Any], types: list[str]
) -> tuple[list[str], list[list[int]], list]:
    names: list[str] = []
    shape_list: list[list[int]] = []
    dtype_list: list = []
    n = max(len(values), len(shapes), len(types))
    for i in range(n):
        v = values[i] if i < len(values) else f"_t{i}"
        names.append(str(v))
        raw_shape = shapes[i] if i < len(shapes) else None
        shape_list.append(_coerce_shape(raw_shape))
        tstr = types[i] if i < len(types) else ""
        dtype_list.append(_coerce_dtype(str(tstr)))
    return names, shape_list, dtype_list


def _attrs_to_dict(node: ChakraNodeProto) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for attr in node.attr:
        d = MessageToDict(attr, preserving_proto_field_name=True)
        name = d.get("name")
        if not name:
            continue
        val = None
        for k, v in d.items():
            if k in ("name", "doc_string"):
                continue
            if v is not None and v != [] and v != {}:
                val = v
                break
        out[name] = val
    return out


_COMM_TYPES = (COMM_COLL_NODE, COMM_SEND_NODE, COMM_RECV_NODE)


def _skip_node_type(t: int) -> bool:
    """Skip metadata/invalid and communication (Polaris single-device scope)."""
    return t in (INVALID_NODE, METADATA_NODE) or t in _COMM_TYPES


def _attr_scalar(attrs: dict[str, Any], key: str) -> Any:
    val = attrs.get(key)
    if isinstance(val, dict):
        for k in ("int64_val", "string_val", "uint64_val", "bool_val"):
            if k in val:
                return val[k]
    return val


def _merge_ttsim_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    raw = _attr_scalar(attrs, "ttsim_attrs")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    attrs.setdefault(k, v)
        except (SyntaxError, ValueError, TypeError):
            pass
    return attrs


def _apply_ttsim_const_data(tensors: dict[str, dict[str, Any]], attrs: dict[str, Any]) -> None:
    raw = _attr_scalar(attrs, "ttsim_const_data")
    if not isinstance(raw, str) or not raw.strip():
        return
    try:
        const_data = ast.literal_eval(raw)
    except (SyntaxError, ValueError, TypeError):
        return
    if not isinstance(const_data, dict):
        return
    for tname, vals in const_data.items():
        if tname in tensors:
            tensors[tname]["data"] = np.asarray(vals)


def _optype_from_ttsim(
    attrs: dict[str, Any], trace_name: str, in_names: list[str], out_names: list[str]
) -> tuple[str, str, list[str] | None, dict[str, Any]]:
    raw = _attr_scalar(attrs, "ttsim_optype")
    if isinstance(raw, str) and raw.strip():
        optype = raw.strip()
        try:
            get_opdesc_registry().get_opdesc(optype)
            extra = {"chakra_map": f"chakra.{optype.lower()}"}
            return optype, _CHAKRA_DOMAIN, None, extra
        except KeyError:
            logger.debug("Unknown ttsim_optype {!r} on {}; using heuristics", optype, trace_name)
    return infer_chakra_polaris_op(trace_name, in_names, out_names)


def infer_chakra_polaris_op(
    trace_name: str, in_names: list[str], out_names: list[str]
) -> tuple[str, str, list[str] | None, dict[str, Any]]:
    """Map a Chakra trace row to a Polaris op for clearer perf accounting.

    Returns ``(optype, domain, in_list_override, extra_attrs)``.
    - Mapped ONNX-class ops use ``domain`` :data:`_CHAKRA_DOMAIN` (``"chakra"``) and set ``attrs["chakra_map"]``
      to a short slug (``chakra.gemm``, ``chakra.gather``, …).
    - Unmapped rows stay ``ChakraNode`` / ``chakra.et`` (generic Chakra ET path).

    Heuristics are name- and arity-driven; traces that do not match fall back to ``ChakraNode``.
    """
    tn = trace_name.lower()
    nin = len(in_names)
    nout = len(out_names)
    extra: dict[str, Any] = {}

    # PyTorch-style elementwise add (unit tests and generic traces)
    if ("aten::add" in tn or tn.endswith("::add")) and nin == 2 and nout == 1:
        extra["chakra_map"] = "chakra.add"
        return "Add", _CHAKRA_DOMAIN, None, extra

    # Token embedding table: ONNX Gather(data, indices) with axis on vocab dimension
    if "word_embeddings" in tn and nin == 2 and nout == 1:
        ids_name = None
        weight_name = None
        for nm in in_names:
            low = nm.lower()
            if "input" in low and "id" in low:
                ids_name = nm
            elif "weight" in low or "embed" in low:
                weight_name = nm
        if ids_name is not None and weight_name is not None:
            extra["chakra_map"] = "chakra.gather"
            extra["axis"] = 0
            return "Gather", _CHAKRA_DOMAIN, [weight_name, ids_name], extra

    # Linear / matmul-shaped layers suitable for Gemm (last-two-dim matmul)
    if nin == 2 and nout == 1:
        if "matmul" in tn:
            extra["chakra_map"] = "chakra.gemm"
            extra["transA"], extra["transB"] = 0, 0
            return "Gemm", _CHAKRA_DOMAIN, None, extra
        if "intermediate.dense" in tn or "pooler.dense" in tn:
            extra["chakra_map"] = "chakra.gemm"
            extra["transA"], extra["transB"] = 0, 0
            return "Gemm", _CHAKRA_DOMAIN, None, extra
        # FFN down-proj: ``...output.dense`` but not attention output projection
        if "output.dense" in tn and "attention" not in tn:
            extra["chakra_map"] = "chakra.gemm"
            extra["transA"], extra["transB"] = 0, 0
            return "Gemm", _CHAKRA_DOMAIN, None, extra

    if nin == 3 and nout == 1 and "layernorm" in tn:
        extra["chakra_map"] = "chakra.layernorm"
        extra["axis"] = -1
        extra["epsilon"] = 1e-5
        return "LayerNormalization", _CHAKRA_DOMAIN, None, extra

    if nin == 1 and nout == 1 and ("intermediate.act" in tn or "gelu" in tn):
        extra["chakra_map"] = "chakra.gelu"
        return "Gelu", _CHAKRA_DOMAIN, None, extra

    if nin == 1 and nout == 1 and "softmax" in tn:
        extra["chakra_map"] = "chakra.softmax"
        extra["axis"] = -1
        return "Softmax", _CHAKRA_DOMAIN, None, extra

    # ONNX-export ET rows: ``OpType:/onnx/...`` (e.g. ResNet traces from ``generate_resnet_chakra_et_from_onnx``)
    if ":" in trace_name:
        prefix = trace_name.split(":", 1)[0].strip().lower()
        if prefix == "conv":
            # Keep generic ``ChakraNode``: ``conv_sinf`` may rewrite output shapes and break
            # residual ``Add`` branches that rely on exact ET / ONNX value_info dimensions.
            pass
        elif prefix == "relu":
            extra["chakra_map"] = "chakra.relu"
            return "Relu", _CHAKRA_DOMAIN, None, extra
        elif prefix == "add":
            extra["chakra_map"] = "chakra.add"
            return "Add", _CHAKRA_DOMAIN, None, extra
        elif prefix == "maxpool":
            # ResNet ``MaxPool`` is 3×3, stride 2, padding 1 (matches torchvision export).
            extra["chakra_map"] = "chakra.maxpool"
            extra["kernel_shape"] = [3, 3]
            extra["strides"] = [2, 2]
            extra["pads"] = [1, 1, 1, 1]
            return "MaxPool", _CHAKRA_DOMAIN, None, extra
        elif prefix == "globalaveragepool":
            extra["chakra_map"] = "chakra.globalaveragepool"
            return "GlobalAveragePool", _CHAKRA_DOMAIN, None, extra
        elif prefix == "averagepool":
            extra["chakra_map"] = "chakra.averagepool"
            return "AveragePool", _CHAKRA_DOMAIN, None, extra
        elif prefix == "flatten":
            extra["chakra_map"] = "chakra.flatten"
            return "Flatten", _CHAKRA_DOMAIN, None, extra
        elif prefix in ("gemm", "matmul"):
            # Keep generic ``ChakraNode`` — ONNX ``Gemm`` layouts / transposes vary vs ``gemm_sinf``.
            pass

    return "ChakraNode", "chakra.et", None, {}


def parse_chakra_trace(wlname: str, path: str) -> tuple[dict[str, Any], list[ChakraNodeProto]]:
    hdr: dict[str, Any] = {"name": wlname, "framework_type": "CHAKRA_ET"}
    nodes: list[ChakraNodeProto] = []
    fh = openFileRd(path)
    try:
        meta = GlobalMetadata()
        if not decodeMessage(fh, meta):
            raise ValueError(f"Chakra trace missing GlobalMetadata: {path}")
        hdr["chakra_version"] = meta.version
        hdr["chakra_global_attr"] = [MessageToDict(a, preserving_proto_field_name=True) for a in meta.attr]
        node = ChakraNodeProto()
        while decodeMessage(fh, node):
            nodes.append(node)
            node = ChakraNodeProto()
    finally:
        fh.close()
    return hdr, nodes


def _build_tensor_table(nodes: list[ChakraNodeProto]) -> dict[str, dict[str, Any]]:
    tensors: dict[str, dict[str, Any]] = {}
    for n in nodes:
        if _skip_node_type(n.type):
            continue
        op_key = f"c_{n.id}"
        for direction, io in (("in", n.inputs), ("out", n.outputs)):
            vals, shapes, types = _io_lists(io)
            tnames, tshapes, tdtypes = _tensor_names_values_shapes_types(vals, shapes, types)
            for ti, tname in enumerate(tnames):
                shape = tshapes[ti] if ti < len(tshapes) else [1]
                dtype = tdtypes[ti] if ti < len(tdtypes) else np.dtype(np.float32)
                if tname not in tensors:
                    tensors[tname] = {
                        "name": tname,
                        "shape": shape,
                        "dtype": dtype,
                        "data": None,
                        "resolve": "K",
                        "op_in": [],
                        "op_out": [],
                    }
                else:
                    if tensors[tname]["shape"] != shape:
                        logger.warning(
                            "Chakra tensor {!r} shape mismatch: {} vs {}; keeping first.",
                            tname,
                            tensors[tname]["shape"],
                            shape,
                        )
                if direction == "in":
                    tensors[tname]["op_in"].append(op_key)
                else:
                    tensors[tname]["op_out"].append(op_key)
        _apply_ttsim_const_data(tensors, _merge_ttsim_attrs(_attrs_to_dict(n)))
    return tensors


def chakra2graph(wlname: str, wlpath: str) -> WorkloadGraph:
    """Parse a Chakra ``.et`` file (length-delimited ``GlobalMetadata`` + ``Node`` messages)."""
    hdr, nodes = parse_chakra_trace(wlname, wlpath)
    tensor_tbl = _build_tensor_table(nodes)
    gg = WorkloadGraph(hdr["name"])
    gg.add_hdr_info(hdr)

    for tinfo in tensor_tbl.values():
        gg.add_tensor(SimTensor(tinfo))

    for n in nodes:
        if _skip_node_type(n.type):
            continue
        in_vals, in_shapes, in_types = _io_lists(n.inputs)
        out_vals, out_shapes, out_types = _io_lists(n.outputs)
        in_names, _, _ = _tensor_names_values_shapes_types(in_vals, in_shapes, in_types)
        out_names, _, _ = _tensor_names_values_shapes_types(out_vals, out_shapes, out_types)
        attrs = _merge_ttsim_attrs(_attrs_to_dict(n))
        attrs["chakra_node_id"] = int(n.id)
        attrs["chakra_node_type"] = int(n.type)
        attrs["chakra_trace_name"] = n.name
        attrs["chakra_duration_micros"] = int(n.duration_micros)

        optype, domain, in_override, map_attrs = _optype_from_ttsim(attrs, n.name, in_names, out_names)
        attrs.update(map_attrs)
        use_in = in_override if in_override is not None else in_names

        op_info = {
            "name": f"c_{n.id}",
            "optype": optype,
            "attrs": attrs,
            "inList": use_in,
            "outList": out_names,
            "domain": domain,
            "docstr": n.name,
        }
        sim_op = SimOp(op_info)
        repeat = int(_attr_scalar(attrs, "ttsim_repeat_count") or 1)
        if repeat > 1:
            sim_op.repeat_count = repeat
        gg.add_op(sim_op)

    gg.construct_graph()
    return gg
