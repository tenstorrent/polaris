#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Export TTSIM :class:`WorkloadGraph` rows to Chakra ``.et`` (compute + optional collectives)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

from ttsim.front.chakra.et_def_pb2 import (
    ALL_REDUCE,
    AttributeProto,
    COMM_COLL_NODE,
    COMP_NODE,
    GlobalMetadata,
    METADATA_NODE,
    Node,
)
from ttsim.front.chakra.protolib import encodeMessage
from ttsim.graph import WorkloadGraph
from ttsim.ops import SimOp, SimTensor


def _dtype_to_chakra(dtype) -> str:
    if isinstance(dtype, np.dtype):
        if dtype == np.dtype(np.float16):
            return "Half"
        if dtype == np.dtype(np.float64):
            return "double"
        if dtype == np.dtype(np.int64):
            return "Long"
        if dtype == np.dtype(np.int32):
            return "Int"
        return "Float"
    return "Float"


def _shape_list(t: SimTensor) -> list[int]:
    shp = list(t.shape) if t.shape is not None else [1]
    return [int(d) for d in shp] if shp else [1]


def _lol(shapes: list[list[int]]) -> list[list[int]]:
    return [list(s) for s in shapes]


def _add_attr(node: Node, name: str, **fields: Any) -> None:
    attr = AttributeProto()
    attr.name = name
    if "string_val" in fields:
        attr.string_val = str(fields["string_val"])
    elif "int64_val" in fields:
        attr.int64_val = int(fields["int64_val"])
    elif "uint64_val" in fields:
        attr.uint64_val = int(fields["uint64_val"])
    elif "bool_val" in fields:
        attr.bool_val = bool(fields["bool_val"])
    node.attr.append(attr)


def _emit_comp_node(
    fh,
    *,
    nid: int,
    name: str,
    op: SimOp,
    wlgraph: WorkloadGraph,
    data_deps: list[int],
) -> int:
    in_tensors = [wlgraph._tensors[t] for t in op.inList]
    out_tensors = [wlgraph._tensors[t] for t in op.outList]

    node = Node()
    node.id = nid
    node.name = op.name
    node.type = COMP_NODE
    node.duration_micros = 8
    node.data_deps.extend(data_deps)

    in_names = [t.name for t in in_tensors]
    in_shapes = [_shape_list(t) for t in in_tensors]
    in_types = [_dtype_to_chakra(t.dtype) for t in in_tensors]
    out_names = [t.name for t in out_tensors]
    out_shapes = [_shape_list(t) for t in out_tensors]
    out_types = [_dtype_to_chakra(t.dtype) for t in out_tensors]

    node.inputs.values = str(in_names)
    node.inputs.shapes = str(_lol(in_shapes))
    node.inputs.types = str(in_types)
    node.outputs.values = str(out_names)
    node.outputs.shapes = str(_lol(out_shapes))
    node.outputs.types = str(out_types)

    _add_attr(node, "is_cpu_op", bool_val=False)
    _add_attr(node, "ttsim_optype", string_val=op.optype)
    _add_attr(node, "ttsim_op_name", string_val=op.name)
    if op.repeat_count and op.repeat_count != 1:
        _add_attr(node, "ttsim_repeat_count", int64_val=int(op.repeat_count))
    if op.attrs:
        _add_attr(node, "ttsim_attrs", string_val=str(op.attrs))

    const_data: dict[str, Any] = {}
    for t in in_tensors:
        if t.data is not None:
            const_data[t.name] = np.asarray(t.data).tolist()
    if const_data:
        _add_attr(node, "ttsim_const_data", string_val=str(const_data))

    encodeMessage(fh, node)
    return nid


def _emit_allreduce_node(
    fh,
    *,
    nid: int,
    name: str,
    comm_size_bytes: int,
    data_deps: list[int],
    tp_group_size: int,
) -> int:
    node = Node()
    node.id = nid
    node.name = name
    node.type = COMM_COLL_NODE
    node.duration_micros = 8
    node.data_deps.extend(data_deps)
    _add_attr(node, "is_cpu_op", bool_val=False)
    _add_attr(node, "comm_type", int64_val=int(ALL_REDUCE))
    _add_attr(node, "comm_size", int64_val=int(comm_size_bytes))
    _add_attr(node, "pg_name", string_val="tp_group")
    _add_attr(node, "pg_size", int64_val=int(tp_group_size))
    encodeMessage(fh, node)
    return nid


def ttsim_graph_to_et(
    wlgraph: WorkloadGraph,
    path: str | Path,
    *,
    trace_name: str | None = None,
    mesh_rows: int = 1,
    mesh_cols: int = 1,
    tp_size: int = 1,
) -> int:
    """Write a single-rank Chakra ET with TTSIM-native compute ops."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    trace_name = trace_name or wlgraph._name

    ordered = wlgraph.get_ordered_nodes()
    fh = open(path, "wb")
    nid = 1
    prev_id: int | None = None
    try:
        meta = GlobalMetadata(version="0.0.4")
        encodeMessage(fh, meta)
        meta_node = Node()
        meta_node.id = 0
        meta_node.name = "metadata"
        meta_node.type = METADATA_NODE
        _add_attr(meta_node, "source", string_val="polaris_ttsim2chakra")
        _add_attr(meta_node, "ttsim_graph", string_val=trace_name)
        _add_attr(meta_node, "mesh_rows", int64_val=mesh_rows)
        _add_attr(meta_node, "mesh_cols", int64_val=mesh_cols)
        _add_attr(meta_node, "tp_size", int64_val=tp_size)
        encodeMessage(fh, meta_node)

        for opname in ordered:
            op = wlgraph.get_op(opname)
            deps = [prev_id] if prev_id is not None else []
            _emit_comp_node(fh, nid=nid, name=op.name, op=op, wlgraph=wlgraph, data_deps=deps)
            prev_id = nid
            nid += 1
    finally:
        fh.close()

    count = nid - 1
    logger.info("Wrote Chakra ET {} ({} COMP_NODEs)", path, count)
    return count


def ttsim_llama_layer_template_to_mesh_et(
    wlgraph: WorkloadGraph,
    path: str | Path,
    *,
    n_layers: int,
    comm_size_bytes: int,
    tp_size: int = 32,
    mesh_rows: int = 4,
    mesh_cols: int = 8,
    layer_prefix: str = "llama70.layer_0",
    attn_comm_after: str = "attention.wo",
    ffn_comm_after: str = "feed_forward.w2",
) -> int:
    """Expand one TTSIM transformer block into ``n_layers`` with TP AllReduce hooks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ordered = wlgraph.get_ordered_nodes()
    prefix_ops: list[SimOp] = []
    layer_ops: list[SimOp] = []
    suffix_ops: list[SimOp] = []

    for name in ordered:
        op = wlgraph.get_op(name)
        if layer_prefix.split(".layer_")[0] + ".layer_0" in op.name or ".layer_0." in op.name:
            layer_ops.append(op)
        elif any(f".layer_{i}." in op.name for i in range(1, 99)):
            continue
        elif "layer_0" in op.name:
            layer_ops.append(op)
        elif not layer_ops:
            prefix_ops.append(op)
        else:
            suffix_ops.append(op)

    def _split_layer(ops: list[SimOp]) -> tuple[list[SimOp], list[SimOp], list[SimOp]]:
        pre, post_attn, post_ff = [], [], []
        seen_attn_comm = False
        for op in ops:
            if not seen_attn_comm:
                pre.append(op)
                if attn_comm_after in op.name:
                    seen_attn_comm = True
            elif attn_comm_after in op.name:
                post_attn.append(op)
                seen_attn_comm = True
            elif ffn_comm_after in op.name:
                post_attn.append(op)
                post_ff = ops[ops.index(op) + 1 :]
                break
            elif seen_attn_comm and not post_ff:
                post_attn.append(op)
        if not post_ff:
            # fallback: split at first feed_forward and after w2
            pre, mid, tail = [], [], []
            phase = "attn"
            for op in ops:
                if "feed_forward" in op.name and phase == "attn":
                    phase = "ffn"
                if phase == "attn":
                    pre.append(op)
                    if attn_comm_after in op.name:
                        mid_start = len(pre)
                elif phase == "ffn":
                    if ffn_comm_after in op.name:
                        mid = ops[len(pre) : ops.index(op) + 1]
                        tail = ops[ops.index(op) + 1 :]
                        return pre, mid, tail
            return ops, [], []
        return pre, post_attn, post_ff

    attn_block, between, ffn_tail = _split_layer(layer_ops)

    fh = open(path, "wb")
    nid = 1
    prev_id: int | None = None
    try:
        encodeMessage(fh, GlobalMetadata(version="0.0.4"))
        meta_node = Node()
        meta_node.id = 0
        meta_node.name = "metadata"
        meta_node.type = METADATA_NODE
        _add_attr(meta_node, "source", string_val="polaris_llama_mesh_ttsim2chakra")
        _add_attr(meta_node, "model", string_val="llama70B")
        _add_attr(meta_node, "mesh_rows", int64_val=mesh_rows)
        _add_attr(meta_node, "mesh_cols", int64_val=mesh_cols)
        _add_attr(meta_node, "tp_size", int64_val=tp_size)
        _add_attr(meta_node, "n_layers", int64_val=n_layers)
        encodeMessage(fh, meta_node)

        def emit_ops(ops: list[SimOp], name_suffix: str = "") -> None:
            nonlocal nid, prev_id
            for op in ops:
                node_name = op.name
                if name_suffix:
                    node_name = op.name.replace(".layer_0.", f".layer_{name_suffix}.")
                dup = SimOp(
                    {
                        "name": node_name,
                        "optype": op.optype,
                        "attrs": dict(op.attrs),
                        "inList": list(op.inList),
                        "outList": list(op.outList),
                        "domain": op.domain,
                        "docstr": node_name,
                    }
                )
                deps = [prev_id] if prev_id is not None else []
                _emit_comp_node(fh, nid=nid, name=node_name, op=dup, wlgraph=wlgraph, data_deps=deps)
                prev_id = nid
                nid += 1

        for op in prefix_ops:
            emit_ops([op])

        for layer_idx in range(n_layers):
            emit_ops(attn_block)
            deps = [prev_id] if prev_id is not None else []
            _emit_allreduce_node(
                fh,
                nid=nid,
                name=f"layer_{layer_idx}.attention.tp_allreduce",
                comm_size_bytes=comm_size_bytes,
                data_deps=deps,
                tp_group_size=tp_size,
            )
            prev_id = nid
            nid += 1
            emit_ops(between)
            deps = [prev_id] if prev_id is not None else []
            _emit_allreduce_node(
                fh,
                nid=nid,
                name=f"layer_{layer_idx}.mlp.tp_allreduce",
                comm_size_bytes=comm_size_bytes,
                data_deps=deps,
                tp_group_size=tp_size,
            )
            prev_id = nid
            nid += 1
            emit_ops(ffn_tail)

        for op in suffix_ops:
            emit_ops([op])
    finally:
        fh.close()

    count = nid - 1
    logger.info("Wrote mesh Chakra ET {} ({} nodes, {} layers)", path, count, n_layers)
    return count


def build_ttsim_workload_graph(module_path: str, instance_name: str, cfg: dict[str, Any]) -> WorkloadGraph:
    from ttsim.utils.common import get_ttsim_functional_instance

    wl = get_ttsim_functional_instance(module_path, instance_name, cfg)
    wl.create_input_tensors()
    wl()
    return wl.get_forward_graph()
