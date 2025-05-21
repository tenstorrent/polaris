#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
import logging
import math
import networkx as nx
from collections import defaultdict

#for graph2onnx
import onnx
from onnx import TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor
from onnx.checker import check_model
import numpy as np

from ttsim.ops import SimOp, SimTensor

LOG   = logging.getLogger(__name__)
INFO  = LOG.info
DEBUG = LOG.debug

G_FUSE_OP_OVERLAP_COST_CONSTANT = 0.10 #10% overlap overhead

class WorkloadGraph():
    # Type hint for instance attribute
    _graph: nx.MultiDiGraph
    def __init__(self, name):
        self._graph          = nx.MultiDiGraph() # NetworkX MultiDiGraph
        self._name           = name              # graph.name                  : Str
        self._hdr_info       = {}                # Workload Header Info        : Dict -- ONNX like
        self._ops            = {}                # All Ops in the Graph        : Dict[Str, SimOp]
        self._tensors        = {}                # All Tensors in the Graph    : Dict[Str, SimTensor]
        self._input_nodes    = []                # Input Nodes in the Graph    : List[Str] Op Names
        self._output_nodes   = []                # Output Nodes in the Graph   : List[Str] Op Names
        self._input_tensors  = []                # Input Tensors in the Graph  : List[Str] Op Names
        self._output_tensors = []                # Output Tensors in the Graph : List[Str] Op Names
        self._optype_hist    = defaultdict(int)  # Op Type Histogram           : Dict[Str, Int]

    def get_node_count(self)     : return self._graph.number_of_nodes()
    def get_edge_count(self)     : return self._graph.number_of_edges()
    def get_node(self, name)     : return self._graph[name]['data']
    def has_node(self, name)     : return self._graph.has_node(name)
    def get_input_nodes(self)    : return self._input_nodes
    def get_output_nodes(self)   : return self._output_nodes
    def get_input_tensors(self)  : return self._input_tensors
    def get_output_tensors(self) : return self._output_tensors
    def get_ordered_nodes(self)  : return list(nx.topological_sort(self._graph))

    def add_hdr_info(self, hdr_info):
        self._hdr_info = hdr_info

    def add_op(self, op: SimOp):
        for tensor_name in op.inList:
            assert tensor_name in self._tensors, \
                    f"Input SimTensor {tensor_name} for SimOp {op.name} not found in WorkloadGraph"
        for tensor_name in op.outList:
            assert tensor_name in self._tensors, \
                    f"Output SimTensor {tensor_name} for SimOp {op.name()} not found in WorkloadGraph"
        assert op.name not in self._ops, f"SimOp({op.name}) is not unique!!!"
        self._ops[op.name] = op

    def add_tensor(self, tensor: SimTensor):
        assert tensor.name not in self._tensors, f"SimTensor({tensor.name}) is not unique!!!"
        self._tensors[tensor.name] = tensor

    def construct_graph(self):
        #construct graph
        for op_count, (op_name, op_info) in enumerate(self._ops.items()):
            self._graph.add_node(op_name)
            self._optype_hist[op_info.optype] += 1
            for o in op_info.outList:
                for inode in self._tensors[o].op_in:
                    self._graph.add_edge(op_name, inode, name=o)
        #update graph summary data structures
        self._input_tensors  = list(set([tname for tname,tval in self._tensors.items() if tval.op_out == []]))
        self._output_tensors = list(set([tname for tname,tval in self._tensors.items() if tval.op_in == []]))
        self._input_nodes    = list(set([o for t in self._input_tensors for o in self._tensors[t].op_in]))
        self._output_nodes   = list(set([o for t in self._output_tensors for o in self._tensors[t].op_out]))

        return

    def __str__(self):
        indent = 4
        ostr = ""

        ostr += '\nSUMMARY:\n'
        ostr += f"{' '*indent}num_nodes         : {self.get_node_count()}\n"
        ostr += f"{' '*indent}num_edges         : {self.get_edge_count()}\n"
        ostr += f"{' '*indent}num_input_nodes   : {len(self.get_input_nodes())}\n"
        ostr += f"{' '*indent}num_output_nodes  : {len(self.get_output_nodes())}\n"
        ostr += f"{' '*indent}num_input_tensors : {len(self.get_input_tensors())}\n"
        ostr += f"{' '*indent}num_output_tensors: {len(self.get_output_tensors())}\n"
        ostr += f"{' '*indent}input_nodes       :\n"
        for op_in in self.get_input_nodes(): ostr += f"{' '*2*indent}{op_in}\n"
        ostr += f"{' '*indent}output_nodes      :\n"
        for op_out in self.get_output_nodes(): ostr += f"{' '*2*indent}{op_out}\n"
        ostr += f"{' '*indent}input_tensors     :\n"
        for t_in in self.get_input_tensors(): ostr += f"{' '*2*indent}{t_in}\n"
        ostr += f"{' '*indent}output_tensors    :\n"
        for t_out in self.get_output_tensors(): ostr += f"{' '*2*indent}{t_out}\n"


        ostr += '\nHDRINFO:\n'
        for k,v in self._hdr_info.items():
            ostr += f"{' '*indent}{k:25s}: {v}\n"

        ostr += '\nOPTYPE_HISTOGRAM:\n'
        for k,v in sorted(self._optype_hist.items(),
                          key=lambda item: item[1],
                          reverse=True):
            ostr += f"{' '*indent}{k:25s}: {v:5d}\n"

        ostr += '\nTENSORS:\n'
        for k,v in self._tensors.items():
            ostr += f"{' '*indent}{v}\n"

        ostr += '\nOPS:\n'
        for k,v in self._ops.items():
            ostr += f"{' '*indent}{v}\n"

        return ostr

    def execute(self, device):
        #find compute/memory cycles, given a executable device
        for opname in nx.topological_sort(self._graph):
            op = self._ops[opname]
            op.execute(device)

    def reverse_graph(self): pass
    def find_forks_and_joins(self): pass
    def create_gradient_tensors(self): pass
    def backward(self): pass

    def set_precision(self, o2dt):
        for op_type_x, data_type_x in o2dt:
            for opnum, opname in enumerate(nx.topological_sort(self._graph)):
                op_type = self._ops[opname].optype.upper()
                if op_type_x == '*':
                    #print("\t--->matched op_type:", op_type, 'optype_x:', op_type_x, data_type_x)
                    self._ops[opname].set_precision(data_type_x)
                elif op_type == op_type_x:
                    #print("\t--->matched op_type:", op_type, 'optype_x:', op_type_x, data_type_x)
                    self._ops[opname].set_precision(data_type_x)
                else:
                    #print("\t--->NOT matched op_type:", op_type, 'optype_x:', op_type_x, data_type_x)
                    pass
        for opname, op in self._ops.items():
            assert op.precision is not None, f'Precision for {opname} is unset'
        return

    def remove_nodes(self, op_type_list):
        for op_type_x in op_type_list:
            for opname in nx.topological_sort(self._graph):
                op_type = self._ops[opname].optype.upper()
                if op_type == op_type_x:
                    self._ops[opname].remove_in_optimization()
        return

    def fuse_nodes(self, op_type_fusion_patterns):
        #op_type_fusion_list is a list of lists specifying optype-patterns to search for
        # in the graph and then fuse. The patterns are specifed in order of priority
        # e.g. [ [Matmul, Gelu, Add], [Matmul, Gelu] ] will match Matmul-Gelu-Add
        #   first and only then match the next pattern

        #FIRST PASS: find all matched nodes in the graph
        fusion_candidates = []
        #track nodes that have already been matched
        already_matched_nodes_set = set()
        for pattern in op_type_fusion_patterns:
            pattern_len = len(pattern)
            assert pattern_len > 1, f"Illegal fusion pattern specification {pattern}!!"
            for node in nx.topological_sort(self._graph):
                if node in already_matched_nodes_set:
                    continue

                if self._ops[node].optype.upper() == pattern[0]:
                    current_node          = node
                    matched_nodes_list    = [current_node]
                    current_node_op_type  = self._ops[current_node].optype.upper()
                    for i in range(1, pattern_len):
                        successors = list(self._graph.successors(current_node))
                        if ( len(successors) == 1 and
                            self._ops[successors[0]].optype.upper() == pattern[i] and
                            successors[0] not in already_matched_nodes_set):
                            current_node = successors[0]
                            matched_nodes_list.append(current_node)
                        elif len(successors) == 1 and self._ops[successors[0]].removed_in_optimization:
                            #TODO: How do we handle removed nodes if they occur in the path of
                            # fusion pattern --> we can pass for now because for LLMs we don't
                            # run into this case
                            assert False, "WARNING: Found a remove_in_optimization node in fusion pattern {pattern}!!"
                        else:
                            #pattern does not match break
                            break
                    if len(matched_nodes_list) == pattern_len:
                        fusion_candidates.append(matched_nodes_list)
                        already_matched_nodes_set.update(matched_nodes_list)

        #SECOND PASS: now all out fusion candidates have been found, and we can apply the
        # op-fusion on the graph
        for fusion_nodes in fusion_candidates:
            #create a new fused node with combined operations:
            pattern_len   = len(fusion_nodes)
            first_op_name = fusion_nodes[0]
            last_op_name  = fusion_nodes[-1]
            first_op      = self._ops[first_op_name]
            last_op       = self._ops[last_op_name]

            #update fusion op cycles
            # TODO: add some checks to make sure that intermediate fused ops have
            #   only one input - one output

            #compute cycles = sum of all fused op compute cycles + overhead per operator overlap
            # TODO: should we add overlap cost only if COMPUTE PIPES CHANGE?
            # intermediate mem rd/wr are suppressed by fusion
            # mem rd cycles = first op mem rd cycles
            # mem wr cycles = last op mem rd cycles
            fused_compute_cycles = first_op.compute_cycles
            fused_mem_rd_cycles  = first_op.mem_rd_cycles
            fused_mem_wr_cycles  = last_op.mem_wr_cycles
            for i in range(1, pattern_len):
                matched_op_name  = fusion_nodes[i]
                matched_op       = self._ops[matched_op_name]
                fused_compute_cycles += math.ceil(matched_op.compute_cycles * (1.0 + G_FUSE_OP_OVERLAP_COST_CONSTANT))
                matched_op.fuse_op(first_op_name)
            first_op.fused_op_cycles = {
                    'compute_cycles': fused_compute_cycles,
                    'mem_rd_cycles': fused_mem_rd_cycles,
                    'mem_wr_cycles': fused_mem_wr_cycles,
                    }
        return


    def map_resources(self, op2rsrc_map):
        for node in nx.topological_sort(self._graph):
            optype    = self._ops[node].optype.upper()
            try:
                self._ops[node].uses_compute_pipe = op2rsrc_map[optype]
            except KeyError as e:
                raise RuntimeError(f'operator {self._ops[node].optype} not mapped on the device')
        return

    def graph2onnx(self, onnx_filename, /, producer_name="ttsim.functional.export", do_model_check=True):
        nptype_map = {
                np.float32: TensorProto.FLOAT,
                np.float64: TensorProto.FLOAT,
                np.int64:   TensorProto.INT64,
                np.bool_:   TensorProto.BOOL,
                }
        #create tensors....
        onnx_tensors = {}
        onnx_constants = {}
        onnx_params = {}
        for tname, tval in self._tensors.items():
            assert tval.shape is not None, "Illegal tensor shape for make_tensor_value_info: {tval}"

            #TODO: ensure that dims are always int (or str)
            # somewhere in ttsim.front.functional we are creating float dims!!!
            # thens this stupid line to create shape0 can be removed
            shape0 = tuple([int(d) for d in tval.shape])
            assert tval.dtype.type in nptype_map, f"dtype={tval.dtype} not yet supported. Pl. edit wl_graph accordingly!!"

            if shape0 == (): #rank-0 tensor
                if tval.data is None:
                    _data = np.random.randn(1).astype(tval.dtype)
                    tval.data = _data[0]
                val_list = [tval.data]
            else:
                if tval.data is None:
                    tval.data = np.random.randn(*shape0).astype(tval.dtype)
                val_list = tval.data.flatten().tolist()


            if tval.is_const:
                onnx_constants[tname] = make_tensor(name=tname,
                                                    data_type=nptype_map[tval.dtype.type],
                                                    dims=shape0,
                                                    vals=val_list
                                                    )
            elif tval.is_param:
                #print("onnx_params", tval)
                #print("onnx_params", tval.data.shape, tval.data.size)
                onnx_params[tname] = make_tensor(name=tname,
                                                 data_type=nptype_map[tval.dtype.type],
                                                 dims=shape0,
                                                 vals=val_list
                                                 )
            else:
                onnx_tensors[tname] = make_tensor_value_info(tname,
                                                             nptype_map[tval.dtype.type],
                                                             shape0)

        onnx_nodes = {}
        for oname, op in self._ops.items():
            onnx_nodes[oname] = make_node(op.optype, op.inList, op.outList, name=oname, **op.attrs)

        input_list        = [onnx_tensors[x] for x in self.get_input_tensors() if x in onnx_tensors]
        output_list       = [onnx_tensors[x] for x in self.get_output_tensors() if x in onnx_tensors]
        initializer_list  = [onnx_constants[tname] for tname in onnx_constants] + [onnx_params[tname] for tname in onnx_params]
        node_list         = [onnx_nodes[node] for node in nx.topological_sort(self._graph)]

        onnx_graph = make_graph(
                nodes       = node_list,
                name        = self._name,
                inputs      = input_list,
                outputs     = output_list,
                initializer = initializer_list, #TODO: constants can go here...
                )
        model_def = make_model(onnx_graph, producer_name=producer_name)
        if do_model_check:
            check_model(model_def)
        onnx.save(model_def, onnx_filename)
        return

