#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Operator descriptor for Chakra execution-trace nodes (generic IR)."""

from ttsim.ops.desc.registry import register_ops
from ttsim.utils.types import get_bpe, get_sim_dtype


def chakra_node_sinf(iTList, oTList, op, **kwargs):
    """Shape/perf inference for nodes loaded from Chakra traces.

    Tensor shapes are populated while parsing the trace; this path only
    validates them and derives byte/element counts for perf stats.
    """
    for t in iTList:
        if not t.check_shape():
            t.set_shape([1])
    for t in oTList:
        if not t.check_shape():
            t.set_shape([1])

    prec = op.precision if op.precision is not None else "int8"
    bpe = get_bpe(get_sim_dtype(prec))
    in_elems = max(1, sum(t.nelems() for t in iTList))
    out_elems = max(1, sum(t.nelems() for t in oTList))
    in_bytes = sum(t.nbytes(prec) for t in iTList)
    out_bytes = sum(t.nbytes(prec) for t in oTList)
    if in_bytes == 0:
        in_bytes = in_elems * bpe
    if out_bytes == 0:
        out_bytes = out_elems * bpe
    op.perf_stats = {
        "inElems": in_elems,
        "outElems": out_elems,
        "inBytes": in_bytes,
        "outBytes": out_bytes,
        "instrs": {"mov": max(in_elems, out_elems, 1)},
    }


def register_chakra_ops():
    register_ops(
        "chakra",
        [
            [
                "ChakraNode",
                "ARITY_VARIADIC[0-*]->VARIADIC[0-*]",
                "chakra.et",
                "COMMON",
                1,
                1,
                2147483647,
                0,
                2147483647,
                0,
                chakra_node_sinf,
                True,
                True,
                True,
                True,
                True,
            ],
        ],
    )
