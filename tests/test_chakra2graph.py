#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from ttsim.front.chakra.chakra2nx import chakra2graph
from ttsim.front.chakra.et_def_pb2 import COMP_NODE, GlobalMetadata, METADATA_NODE, Node
from ttsim.front.chakra.protolib import encodeMessage


@pytest.mark.unit
def test_chakra2graph_minimal_trace(tmp_path):
    et_path = tmp_path / "minimal.et"
    with open(et_path, "wb") as fh:
        encodeMessage(fh, GlobalMetadata(version="0.0.4"))
        meta = Node()
        meta.id = 0
        meta.name = "metadata"
        meta.type = METADATA_NODE
        encodeMessage(fh, meta)
        n = Node()
        n.id = 1
        n.name = "aten::add"
        n.type = COMP_NODE
        n.inputs.values = str(["x", "y"])
        n.inputs.shapes = str([[2, 3], [2, 3]])
        n.inputs.types = str(["float", "float"])
        n.outputs.values = str(["z"])
        n.outputs.shapes = str([[2, 3]])
        n.outputs.types = str(["float"])
        encodeMessage(fh, n)

    graph = chakra2graph("trace_wl", str(et_path))
    assert graph.get_node_count() == 1
    op = graph.get_op("c_1")
    assert op.optype == "Add"
    assert op.domain == "chakra"
    assert op.attrs.get("chakra_map") == "chakra.add"
    assert op.inList == ["x", "y"]
    assert op.outList == ["z"]


@pytest.mark.unit
@pytest.mark.tools_secondary
def test_chakra2graph_sample_from_astra_sim():
    from pathlib import Path

    here = Path(__file__).resolve()
    sample = here.parents[1].parent / "astra-sim-TT/examples/workload/microbenchmarks/all_reduce/4npus_1MB/all_reduce.0.et"
    if not sample.is_file():
        pytest.skip(f"Optional Chakra sample trace not found at {sample}")
    graph = chakra2graph("all_reduce", str(sample))
    assert graph.get_node_count() > 0
