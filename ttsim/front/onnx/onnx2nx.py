#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
from typing import Any
from collections import defaultdict

import onnx
from onnx import helper, numpy_helper, shape_inference
import onnx.checker

from ttsim.graph import WorkloadGraph
from ttsim.ops import SimOpFactory, SimTensor

def get_io_errstr(xarg, xdim, xdim1, is_init, is_val, is_in, is_out, ginfo):
    errstr  = f"shape mismatch: {xarg} : {xdim} != {xdim1}\n"
    errstr += f"initializer: {ginfo['initializer'][xarg]}\n" if is_init else ""
    errstr += f"value_info : {ginfo['value_info'][xarg]}\n"  if is_val  else ""
    errstr += f"input      : {ginfo['input'][xarg]}\n"       if is_in   else ""
    errstr += f"output     : {ginfo['output'][xarg]}\n"      if is_out  else ""
    return errstr

def parse_onnx_model(wlname, wlpath):
    """
    #Follows spec at https://github.com/onnx/onnx/blob/main/docs/IR.md
    # ir_version        int64                The ONNX version assumed by the model.
    # opset_import      OperatorSetId        A collection of operator set identifiers made available to the model.
    #                                        An implementation must support all operators in the set or reject the model.
    # producer_name     string               The name of the tool used to generate the model.
    # producer_version  string               The version of the generating tool.
    # domain            string               A reverse-DNS name to indicate the model namespace or domain, for example, 'org.onnx'
    # model_version     int64                The version of the model itself, encoded in an integer.
    # doc_string        string               Human-readable documentation for this model. Markdown is allowed.
    # graph             Graph                The parameterized graph that is evaluated to execute the model.
    # metadata_props    map<string,string>   Named metadata values; keys should be distinct.
    # training_info     TrainingInfoProto[]  An optional extension that contains information for training.
    # functions         FunctionProto[]      An optional list of functions local to the model.
    """

    onnxhdrinfo = {'name': wlname, 'framework_type': 'ONNX'}
    modelpb = onnx.load(wlpath)
    onnx.checker.check_model(modelpb)
    modelpb_inferred = shape_inference.infer_shapes(modelpb)
    modelpb = modelpb_inferred

    #check mandatory_fields are present
    fields_int_type = ["ir_version", "model_version"]
    fields_str_type = ["producer_name", "producer_version", "domain", "doc_string"]
    fields_cls_type = ["opset_import", "graph", "metadata_props"]
    mandatory_fields = fields_int_type + fields_str_type
    for ff in mandatory_fields:
        assert hasattr(modelpb, ff), f"Error: ONNX model has no attribute {ff}"
        onnxhdrinfo[ff] = getattr(modelpb, ff)

    for xf in ['training_info', 'functions']:
        pxf = 'has_' + xf
        onnxhdrinfo[pxf] = True if hasattr(modelpb, xf) and len (getattr(modelpb, xf)) > 0 else False
        if onnxhdrinfo[pxf] == True:
            print(f"WARNING {xf} is present but not parsed!!")

    onnxhdrinfo['opset_import'] = []
    for attr in modelpb.opset_import:
        domain_val = str(attr.domain) if attr.HasField("domain") else '<missing domain>'
        assert attr.HasField("version"), 'version field missing in opset_import'
        onnxhdrinfo['opset_import'].append({'domain': str(attr.domain), 'version': int(attr.version)})

    #metadata_props is a map <str, str> where key is model_author and val is model_license
    onnxhdrinfo['metadata_props'] = {}
    for k in modelpb.metadata_props:
        onnxhdrinfo['metadata_props'][k] = modelpb.metadata_props[k]

    #Graph-Protobuf...
    onnxgraphinfo: dict[str, Any] = {}
    onnxgraphinfo['name'] = modelpb.graph.name
    onnxgraphinfo['doc_string'] = modelpb.graph.doc_string
    #model.graph.node         #List of nodes                  : Type Node
        #A list of nodes, forming a partially ordered computation graph based on input/output data dependencies.
        #It is in topological order.
    #model.graph.initializer  #List of named tensor values    : Type Tensor
        #A list of named tensor values. When an initializer has the same name as a graph input, it specifies a
        #default value for that input. When an initializer has a name different from all graph inputs, it
        #specifies a constant value. The order of the list is unspecified.
    #model.graph.input        #List of graph inputs           : Type ValueInfo
        #The input parameters of the graph, possibly initialized by a default value found in ‘initializer.’
    #model.graph.output       #List of graph outputs          : Type ValueInfo
        #The output parameters of the graph. Once all output parameters have been written to by a graph execution,
        #the execution is complete.
    #model.graph.value_info   #List of graph values not I/O   : Type ValueInfo
        #Used to store the type and shape information of values that are not inputs or outputs.

        #Value-Info Fields
            #name	    string	The name of the value/parameter.
            #type	    Type	The type of the value including shape information.
            #doc_string	string	Human-readable documentation for this value. Markdown is allowed.

    onnxgraphinfo['input']       = parse_value_info_list(modelpb.graph.input)
    onnxgraphinfo['output']      = parse_value_info_list(modelpb.graph.output)
    onnxgraphinfo['value_info']  = parse_value_info_list(modelpb.graph.value_info)
    onnxgraphinfo['initializer'] = {}
    for tensor in modelpb.graph.initializer:
        dims       = [int(dim) for dim in tensor.dims]
        dtype      = helper.tensor_dtype_to_np_dtype(tensor.data_type)
        data       = numpy_helper.to_array(tensor)
        assert tensor.name not in onnxgraphinfo['initializer'], f"Initializer Tensor {tensor.name} not unique"
        onnxgraphinfo['initializer'][tensor.name] = {
                "name": tensor.name,
                "dtype": dtype,
                "dims": dims,
                "data": data
                }

    #Node Fields
        #name	    string	    An optional name of the node, used for diagnostic purposes only.
        #op_type	string	    The symbolic identifier of the operator to invoke.
        #domain	    string	    The domain of the operator set that contains the operator named by the op_type.
        #doc_string	string	    Human-readable documentation for this value. Markdown is allowed.
        #attribute	Attribute[]	Named attributes, another form of operator parameterization, used for constant values rather than propagated values.
        #input	    string[]	Names of the values used by the node to propagate input values to the node operator. It must refer to either a graph input, a graph initializer or a node output.
        #output	    string[]	Names of the outputs used by the node to capture data from the operator invoked by the node. It either introduces a value in the graph or refers to a graph output.

    onnxgraphinfo['node'] = []
    for i,node in enumerate(modelpb.graph.node):
        assert hasattr(node, "input"),  f"input missing for node {node.name}"
        assert hasattr(node, "output"), f"output missing for node {node.name}"

        attrhash = {}
        if hasattr(node, "attribute"):
            attrhash = {attr.name : onnx_get_value_from_attrs(attr) for attr in node.attribute }

        nodeinfo = {}
        nodeinfo['name']    = node.name
        nodeinfo['optype']  = node.op_type
        nodeinfo['domain']  = node.domain
        nodeinfo['docstr']  = node.doc_string
        nodeinfo['inList']  = [inp for inp in node.input]
        nodeinfo['outList'] = [outp for outp in node.output]
        nodeinfo['attrs']   = attrhash
        onnxgraphinfo['node'].append(nodeinfo)

    #resolve graph tensors
    tensorinfo = get_resolved_tensors(onnxgraphinfo)
    #add_edges(onnxgraphinfo)

    ##############################################
    # End of Parse Onnx
    ##############################################
    del modelpb
    return onnxhdrinfo, onnxgraphinfo, tensorinfo

def parse_value_info_list(vlist):
    tbl = {}
    for vi in vlist:
        assert vi.name not in tbl, f"value info name {vi.name} not unique!!"
        tbl[vi.name] = parse_value_info(vi)
    return tbl

def parse_value_info(vinfo):
    assert vinfo.HasField("name"), 'name field missing in val-info'
    assert vinfo.HasField("type"), 'type field missing in val-info'
    #if inp.HasField("doc_string"): print('doc field present in val info')
    return {
            'name': vinfo.name,
            'type': get_tensor_type_info(vinfo.type)
            }

def get_tensor_type_info(t):
    dims: list[int] = []
    for dim in t.tensor_type.shape.dim:
        value = getattr(dim, dim.WhichOneof("value"))
        if dim.HasField("dim_param"):
            value = 1 if not dims else str(value)
        elif dim.HasField("dim_value"):
            value = int(value)
        else:
            raise ValueError(f"Unknown field type for dim {dim}")
        dims.append(value)
    dtype = helper.tensor_dtype_to_np_dtype(t.tensor_type.elem_type)
    return { 'dtype': dtype, 'dims': dims }

def onnx_get_value_from_attrs(attr, **kwargs):
    """
    #name	      string	       The name of the attribute. Must be unique among attributes, inputs, and outputs for any given operator and node.
    #doc_string	  string	       Human-readable documentation for this value. Markdown is allowed.
    #type	      AttributeType	   The type of the attribute, determining which of the remaining fields is used to hold the value of the attribute.
    #ints	      int64[]	       A list of 64-bit integer values.
    #floats	      float[]	       A list of 32-bit floating-point values.
    #strings	  byte[][]	       A list of UTF-8 strings.
    #tensors	  Tensor[]	       A list of tensor values.
    #graphs	      Graph[]	       A list of graphs.
    #i	          int64	           A 64-bit integer value.
    #f	          float	           A 32-bit floating-point value.
    #s	          byte[]	       UTF-8 string.
    #t	          Tensor	       A tensor value.
    #g	          Graph	           A graph.
    """

    if len(attr.ints) > 0:       return [int(value) for value in attr.ints]
    elif len(attr.floats) > 0:   return [float(value) for value in attr.floats]
    elif len(attr.strings) > 0:  return [str(value) for value in attr.strings]
    elif len(attr.tensors) > 0:  return [numpy_helper.to_array(t) for t in attr.tensors]
    #elif len(attr.graphs) > 0:   return [onnx_get_graph_from_attrs(g, **kwargs) for g in attr.graphs]

    elif attr.HasField("i"):     return int(attr.i)
    elif attr.HasField("f"):     return float(attr.f)
    elif attr.HasField("s"):     return str(attr.s)
    elif attr.HasField("t"):     return numpy_helper.to_array(attr.t)
    #elif attr.HasField("g"):     return onnx_get_graph_from_attrs(attr.g, **kwargs)

    #TODO: fix this!! hack...for now
    elif len(attr.graphs) > 0:
        print(f"WARNING Subgraphs Present for {attr.name} but not parsed!!")
        return "<GRAPHS>"
    elif attr.HasField("g"):
        print(f"WARNING Subgraph Present for {attr.name} but not parsed!!")
        return "<GRAPH>"
    else:
        raise ValueError(f"Found some unknown type of attribute for key {attr.name}\n{dir(attr)}\n{attr}")

def resolve_tensor(_tname, _info):
    INIT_TBL       = _info['initializer']
    VALINFO_TBL    = _info['value_info']
    IN_TBL         = _info['input']
    OUT_TBL        = _info['output']
    is_initializer = _tname in INIT_TBL
    is_value_info  = _tname in VALINFO_TBL
    is_input       = _tname in IN_TBL
    is_output      = _tname in OUT_TBL

    dims  = None
    data  = None
    dtype = None
    tresolve: str = 'None'
    if is_initializer:
        dims  = INIT_TBL[_tname]['dims'] if 'dims' in INIT_TBL[_tname] else None
        data  = INIT_TBL[_tname]['data'] if 'data' in INIT_TBL[_tname] else None
        dtype = INIT_TBL[_tname]['dtype'] if 'dtype' in INIT_TBL[_tname] else None
        tresolve = 'C'
    if is_value_info:
        v = VALINFO_TBL[_tname]
        dims1 = v['type']['dims'] if 'type' in v and 'dims' in v['type'] else None
        dtype1 = v['type']['dtype'] if 'type' in v and 'dtype' in v['type'] else None
        if dims is not None:
            assert dims == dims1, get_io_errstr(_tname, dims, dims1, is_initializer, is_value_info, is_input, is_output, _info)
            assert dtype == dtype1, f"data type mismatch: {dtype} != {dtype1}"
            tresolve += 'V'
        else:
            dtype = dtype1
            tresolve = 'V'
            dims = dims1
        data1 = v['type']['data'] if 'type' in v and 'data' in v['type'] else None
        assert data1 is None, f"data appears in Value Info: {data1}"
    if is_input:
        v = IN_TBL[_tname]
        dims1 = v['type']['dims'] if 'type' in v and 'dims' in v['type'] else None
        dtype1 = v['type']['dtype'] if 'type' in v and 'dtype' in v['type'] else None
        if dims is not None:
            assert dims == dims1, get_io_errstr(_tname, dims, dims1, is_initializer, is_value_info, is_input, is_output, _info)
            assert dtype == dtype1, f"data type mismatch: {dtype} != {dtype1}"
            tresolve += 'I'
        else:
            dtype = dtype1
            tresolve = 'I'
            dims = dims1
        data1 = v['type']['data'] if 'type' in v and 'data' in v['type'] else None
        assert data1 is None, f"data appears in Input: {data1}"
    if is_output:
        v = OUT_TBL[_tname]
        dims1 = v['type']['dims'] if 'type' in v and 'dims' in v['type'] else None
        dtype1 = v['type']['dtype'] if 'type' in v and 'dtype' in v['type'] else None
        if dims is not None:
            assert dims == dims1, get_io_errstr(_tname, dims, dims1, is_initializer, is_value_info, is_input, is_output, _info)
            assert dtype == dtype1, f"data type mismatch: {dtype} != {dtype1}"
            tresolve += 'O'
        else:
            dtype = dtype1
            tresolve = 'O'
            dims = dims1
        data1 = v['type']['data'] if 'type' in v and 'data' in v['type'] else None
        assert data1 is None, f"data appears in Output: {data1}"
    assert tresolve is not None, f"Unable to resolve tensor {_tname}"
    return {'name': _tname, 'shape': dims, 'dtype': dtype, 'data': data,
            'resolve': tresolve, 'op_in': [], 'op_out': []}

def get_resolved_tensors(G):
    """
    # The occurrence of a name as a graph input, a graph initializer, or as a node output is said to be a definition
    # and the occurrence of a name as a node input or as a graph output is said to be a use.

    # When a name appears in both the initializer list and the graph input list, a runtime MAY allow a caller to specify
    # a value for this (input) name overriding the value specified in the initializer and a runtime MAY allow users to
    # omit specifying a value for this (input) name, choosing the value specified in the initializer.

    # Names of constants that are not meant to be overridden by the caller should appear only in the initializer list
    # and not in the graph input list.

    # In models with IR version >= 4, in nested subgraphs used as attribute values, users MUST NOT use the same name as both
    # a subgraph initializer and subgraph input unless the corresponding op's specification explicitly allows it. In models
    # with IR version <= 3, users MAY use the same name as both a subgraph initializer and subgraph input, but this is restricted
    # to support constants via initializers that are not intended to correspond to any actual inputs passed from the node into
    # the subgraph. In particular, the control-flow operator semantics determines the set of inputs supplied to the execution of
    # the subgraph, and these input names MUST NOT appear as subgraph initializers. Subgraph initializer names must appear in
    # the graph input list after the actual inputs. This allows the actual inputs and formal inputs to be matched positionally.
    """

    TENSORS = {}
    for node in G['node']:
        for itensor in node['inList']:
            itensor_info = resolve_tensor(itensor, G)
            if itensor not in TENSORS:
                TENSORS[itensor] = itensor_info
            else:
                tshape0 = TENSORS[itensor]['shape']
                tshape1 = itensor_info['shape']
                assert tshape1 == tshape0, f"Tensor Mismatch!! shape OLD={tshape0} : NEW={tshape1}\n"
                #tdata0  = TENSORS[itensor]['data']
                #tdata1  = itensor_info['data']
                #assert tdata1 == tdata0, f"Tensor Mismatch!! data OLD={tdata0} : NEW={tdata1}\n"
            TENSORS[itensor]['op_in'].append(node['name'])

        for otensor in node['outList']:
            itensor_info = resolve_tensor(otensor, G)
            if otensor not in TENSORS:
                TENSORS[otensor] = itensor_info
            else:
                tshape0 = TENSORS[otensor]['shape']
                tshape1 = itensor_info['shape']
                assert tshape1 == tshape0, f"Tensor Mismatch!! shape OLD={tshape0} : NEW={tshape1}\n"
                #tdata0  = TENSORS[otensor]['data']
                #tdata1  = itensor_info['data']
                #assert tdata1 == tdata0, f"Tensor Mismatch!! data OLD={tdata0} : NEW={tdata1}\n"
            TENSORS[otensor]['op_out'].append(node['name'])
    return TENSORS


def onnx2graph(_wlname, _wlpath):
    H,G,T = parse_onnx_model(_wlname, _wlpath)

    gg = WorkloadGraph(H['name'])

    gg.add_hdr_info(H)

    for tensor_name,tensor_info in T.items():
        gg.add_tensor(SimTensor(tensor_info))

    for op_info in G['node']:
        op_cls = SimOpFactory(op_info['optype'])
        gg.add_op(op_cls(op_info))

    gg.construct_graph()

    return gg
