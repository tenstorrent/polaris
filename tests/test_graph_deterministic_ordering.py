#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for deterministic operator ordering in WorkloadGraph.

Tests verify that:
- Sequence numbers are assigned correctly when operators are added to graphs
- Each graph starts with seqno=0
- Multiple graphs have independent counters
- Deterministic ordering works correctly (same graph produces same ordering)
- Lexicographical topological sort uses seqno correctly for parallel branches
"""
import pytest

from ttsim.graph.wl_graph import WorkloadGraph
from ttsim.ops.op import SimOp
from ttsim.ops.tensor import SimTensor


@pytest.mark.unit
def test_seqno_assigned_on_add():
    """Test that sequence numbers are assigned when operators are added to a graph."""
    graph = WorkloadGraph('test_graph')

    # Create tensors
    t1 = SimTensor({'name': 't1', 'shape': [10], 'dtype': 'float32'})
    t2 = SimTensor({'name': 't2', 'shape': [10], 'dtype': 'float32'})
    t1.op_in = ['op1']
    t2.op_out = ['op1']
    graph.add_tensor(t1)
    graph.add_tensor(t2)

    # Create operator - seqno should be None initially
    op1 = SimOp({
        'name': 'op1',
        'optype': 'Add',
        'inList': ['t1'],
        'outList': ['t2']
    })
    assert op1.seqno is None, "seqno should be None before adding to graph"

    # Add operator to graph - seqno should be assigned
    graph.add_op(op1)
    assert op1.seqno == 0, "First operator should have seqno=0"


@pytest.mark.unit
def test_seqno_increments_per_graph():
    """Test that sequence numbers increment correctly within a graph."""
    graph = WorkloadGraph('test_graph')

    # Create multiple tensors
    tensors = []
    for i in range(5):
        t_in = SimTensor({'name': f't{i}_in', 'shape': [10], 'dtype': 'float32'})
        t_out = SimTensor({'name': f't{i}_out', 'shape': [10], 'dtype': 'float32'})
        t_in.op_in = [f'op{i}']
        t_out.op_out = [f'op{i}']
        graph.add_tensor(t_in)
        graph.add_tensor(t_out)
        tensors.append((t_in, t_out))

    # Create and add operators
    ops = []
    for i in range(5):
        op = SimOp({
            'name': f'op{i}',
            'optype': 'Add',
            'inList': [f't{i}_in'],
            'outList': [f't{i}_out']
        })
        graph.add_op(op)
        ops.append(op)

    # Verify sequence numbers are 0, 1, 2, 3, 4
    for i, op in enumerate(ops):
        assert op.seqno == i, f"Operator {i} should have seqno={i}, got {op.seqno}"


@pytest.mark.unit
def test_each_graph_starts_at_zero():
    """Test that each graph starts with seqno=0 independently."""
    # Create first graph
    graph1 = WorkloadGraph('graph1')
    t1_in = SimTensor({'name': 't1_in', 'shape': [10], 'dtype': 'float32'})
    t1_out = SimTensor({'name': 't1_out', 'shape': [10], 'dtype': 'float32'})
    t1_in.op_in = ['op1']
    t1_out.op_out = ['op1']
    graph1.add_tensor(t1_in)
    graph1.add_tensor(t1_out)

    op1 = SimOp({
        'name': 'op1',
        'optype': 'Add',
        'inList': ['t1_in'],
        'outList': ['t1_out']
    })
    graph1.add_op(op1)
    assert op1.seqno == 0, "First graph's first operator should have seqno=0"

    # Create second graph
    graph2 = WorkloadGraph('graph2')
    t2_in = SimTensor({'name': 't2_in', 'shape': [10], 'dtype': 'float32'})
    t2_out = SimTensor({'name': 't2_out', 'shape': [10], 'dtype': 'float32'})
    t2_in.op_in = ['op2']
    t2_out.op_out = ['op2']
    graph2.add_tensor(t2_in)
    graph2.add_tensor(t2_out)

    op2 = SimOp({
        'name': 'op2',
        'optype': 'Add',
        'inList': ['t2_in'],
        'outList': ['t2_out']
    })
    graph2.add_op(op2)
    assert op2.seqno == 0, "Second graph's first operator should also have seqno=0"

    # Verify graphs are independent
    assert graph1._seqcounter == 1, "Graph1 should have counter=1"
    assert graph2._seqcounter == 1, "Graph2 should have counter=1"


@pytest.mark.unit
def test_deterministic_ordering_sequential():
    """Test that sequential graphs produce deterministic ordering."""
    graph = WorkloadGraph('test_graph')

    # Create a simple sequential graph: op0 -> op1 -> op2
    tensors = []
    for i in range(4):
        t = SimTensor({'name': f't{i}', 'shape': [10], 'dtype': 'float32'})
        if i == 0:
            t.op_in = ['op0']
        elif i == 3:
            t.op_out = ['op2']
        else:
            t.op_in = [f'op{i}']
            t.op_out = [f'op{i-1}']
        graph.add_tensor(t)
        tensors.append(t)

    # Add operators in order
    for i in range(3):
        op = SimOp({
            'name': f'op{i}',
            'optype': 'Add',
            'inList': [f't{i}'],
            'outList': [f't{i+1}']
        })
        graph.add_op(op)

    graph.construct_graph()

    # Get ordered nodes - should be deterministic
    ordered1 = graph.get_ordered_nodes()
    ordered2 = graph.get_ordered_nodes()

    assert ordered1 == ordered2, "Ordering should be deterministic"
    assert ordered1 == ['op0', 'op1', 'op2'], "Sequential graph should maintain order"


@pytest.mark.unit
def test_deterministic_ordering_parallel_branches():
    """Test that parallel branches use seqno for deterministic ordering."""
    graph = WorkloadGraph('test_graph')

    # Create a graph with parallel branches:
    #   op0 -> op1
    #   op0 -> op2
    #   op1 -> op3
    #   op2 -> op3

    # Create tensors
    t0 = SimTensor({'name': 't0', 'shape': [10], 'dtype': 'float32'})
    t1 = SimTensor({'name': 't1', 'shape': [10], 'dtype': 'float32'})
    t2 = SimTensor({'name': 't2', 'shape': [10], 'dtype': 'float32'})
    t3 = SimTensor({'name': 't3', 'shape': [10], 'dtype': 'float32'})
    t4 = SimTensor({'name': 't4', 'shape': [10], 'dtype': 'float32'})

    t0.op_in = ['op0']
    t1.op_in = ['op1']
    t1.op_out = ['op0']
    t2.op_in = ['op2']
    t2.op_out = ['op0']
    t3.op_in = ['op3']
    t3.op_out = ['op1', 'op2']
    t4.op_out = ['op3']

    for t in [t0, t1, t2, t3, t4]:
        graph.add_tensor(t)

    # Add operators - op1 before op2 to test ordering
    op0 = SimOp({'name': 'op0', 'optype': 'Add', 'inList': ['t0'], 'outList': ['t1', 't2']})
    op1 = SimOp({'name': 'op1', 'optype': 'Add', 'inList': ['t1'], 'outList': ['t3']})
    op2 = SimOp({'name': 'op2', 'optype': 'Add', 'inList': ['t2'], 'outList': ['t3']})
    op3 = SimOp({'name': 'op3', 'optype': 'Add', 'inList': ['t3'], 'outList': ['t4']})

    graph.add_op(op0)
    graph.add_op(op1)
    graph.add_op(op2)
    graph.add_op(op3)

    graph.construct_graph()

    # Get ordered nodes multiple times - should be identical
    ordered1 = graph.get_ordered_nodes()
    ordered2 = graph.get_ordered_nodes()
    ordered3 = graph.get_ordered_nodes()

    assert ordered1 == ordered2 == ordered3, "Ordering should be deterministic across multiple calls"

    # Verify topological constraints: op0 before op1 and op2, op1 and op2 before op3
    assert ordered1.index('op0') < ordered1.index('op1'), "op0 should come before op1"
    assert ordered1.index('op0') < ordered1.index('op2'), "op0 should come before op2"
    assert ordered1.index('op1') < ordered1.index('op3'), "op1 should come before op3"
    assert ordered1.index('op2') < ordered1.index('op3'), "op2 should come before op3"

    # Verify deterministic tie-breaking: op1 should come before op2 (lower seqno)
    assert ordered1.index('op1') < ordered1.index('op2'), "op1 (seqno=1) should come before op2 (seqno=2)"


@pytest.mark.unit
def test_deterministic_ordering_independent_of_add_order():
    """Test that ordering is deterministic even if operators are added in different orders."""
    # Create graph and add operators in one order
    graph1 = WorkloadGraph('graph1')
    tensors1 = []
    for i in range(4):
        t = SimTensor({'name': f't1_{i}', 'shape': [10], 'dtype': 'float32'})
        if i == 0:
            t.op_in = ['op0']
        elif i == 3:
            t.op_out = ['op2']
        else:
            t.op_in = [f'op{i}']
            t.op_out = [f'op{i-1}']
        graph1.add_tensor(t)
        tensors1.append(t)

    # Add in order: op0, op1, op2
    for i in range(3):
        op = SimOp({
            'name': f'op{i}',
            'optype': 'Add',
            'inList': [f't1_{i}'],
            'outList': [f't1_{i+1}']
        })
        graph1.add_op(op)
    graph1.construct_graph()
    ordered1 = graph1.get_ordered_nodes()

    # Create another graph with same structure but add operators in reverse order
    graph2 = WorkloadGraph('graph2')
    tensors2 = []
    for i in range(4):
        t = SimTensor({'name': f't2_{i}', 'shape': [10], 'dtype': 'float32'})
        if i == 0:
            t.op_in = ['op0']
        elif i == 3:
            t.op_out = ['op2']
        else:
            t.op_in = [f'op{i}']
            t.op_out = [f'op{i-1}']
        graph2.add_tensor(t)
        tensors2.append(t)

    # Add in reverse order: op2, op1, op0
    for i in [2, 1, 0]:
        op = SimOp({
            'name': f'op{i}',
            'optype': 'Add',
            'inList': [f't2_{i}'],
            'outList': [f't2_{i+1}']
        })
        graph2.add_op(op)
    graph2.construct_graph()
    ordered2 = graph2.get_ordered_nodes()

    # Both should produce the same topological order (op0, op1, op2)
    # but seqno values will differ based on addition order
    assert ordered1 == ['op0', 'op1', 'op2'], "Graph1 should have correct topological order"
    assert ordered2 == ['op0', 'op1', 'op2'], "Graph2 should have correct topological order"

    # Verify seqno reflects addition order
    assert graph1._ops['op0'].seqno == 0
    assert graph1._ops['op1'].seqno == 1
    assert graph1._ops['op2'].seqno == 2

    assert graph2._ops['op0'].seqno == 2  # Added last
    assert graph2._ops['op1'].seqno == 1  # Added second
    assert graph2._ops['op2'].seqno == 0  # Added first


@pytest.mark.unit
def test_multiple_graphs_independent_counters():
    """Test that multiple graphs maintain independent sequence counters."""
    graphs = []
    ops_per_graph = []

    # Create 3 graphs with different numbers of operators
    for graph_idx in range(3):
        graph = WorkloadGraph(f'graph_{graph_idx}')
        num_ops = graph_idx + 2  # 2, 3, 4 operators

        # Create tensors
        tensors = []
        for i in range(num_ops + 1):
            t = SimTensor({
                'name': f't{graph_idx}_{i}',
                'shape': [10],
                'dtype': 'float32'
            })
            if i == 0:
                t.op_in = [f'op{graph_idx}_0']
            elif i == num_ops:
                t.op_out = [f'op{graph_idx}_{num_ops-1}']
            else:
                t.op_in = [f'op{graph_idx}_{i}']
                t.op_out = [f'op{graph_idx}_{i-1}']
            graph.add_tensor(t)
            tensors.append(t)

        # Create and add operators
        ops = []
        for i in range(num_ops):
            op = SimOp({
                'name': f'op{graph_idx}_{i}',
                'optype': 'Add',
                'inList': [f't{graph_idx}_{i}'],
                'outList': [f't{graph_idx}_{i+1}']
            })
            graph.add_op(op)
            ops.append(op)

        graphs.append(graph)
        ops_per_graph.append(ops)

    # Verify each graph's operators start at seqno=0
    for graph_idx, ops in enumerate(ops_per_graph):
        for i, op in enumerate(ops):
            assert op.seqno == i, f"Graph {graph_idx}, op {i} should have seqno={i}"

    # Verify each graph has correct counter value
    assert graphs[0]._seqcounter == 2, "Graph 0 should have 2 operators"
    assert graphs[1]._seqcounter == 3, "Graph 1 should have 3 operators"
    assert graphs[2]._seqcounter == 4, "Graph 2 should have 4 operators"
