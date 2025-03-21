from ttsim.ops.op import SimOpFactory, build_tmp_fp32_tensor_from_shape, build_tmp_data_tensor
from ttsim.graph import CREATE_GRAD_TENSOR, WorkloadGraph, BackwardWorkloadGraph
import numpy as np
import os
from typing import Any

###############################################################################
#ConstantOp             #EltwiseUnaryOp   #EltwiseBinaryOp   #GatherOp
#LayerNormalizationOp   #ConvOp           #MaxPoolOp         #MatMulOp
#SplitOp                #ReshapeOp        #TransposeOp       #WhereOp
#SoftmaxOp              #PowOp            #UnsqueezeOp       #ConcatOp
#SliceOp                #DropoutOp        #EqualOp           #CastOp
#ShapeOp                #RangeOp          #GeluOp            #ReluOp
#ReluGradOp             #GeluGradOp
###############################################################################

OPTYPES: dict[str, dict] = {
    # TODO: enable LayerNormalization after backward pass implemented for it
    # 'LayerNormalization'  : {'in': [[4,7,3,5],[5],[5]], 'out': [[4,7,3,5],[4,7,3,1],[4,7,3,1]]},

    'MatMul'              : {'in': [[4,3,6], [4,6,10]], 'out': [[4,3,10]]},
    'Add'                 : {'in': [[4,3,2], [4,3,2]],  'out': [[4,3,2]]},
    'Mul'                 : {'in': [[4,3,2], [4,3,2]],  'out': [[4,3,2]]},
    'Transpose'           : {'in': [[4,3,2]], 'out': [[2,4,3]], 'attrs': {'perm': [2,0,1]}},

    #activations
    'Gelu'                : {'in': [[1,2,3,5]], 'out': [[1,2,3,5]]},
    'Relu'                : {'in': [[55,7,21]], 'out': [[55,7,21]]},
    'Softmax'             : {'in': [[2,4,7,5]], 'out': [[2,4,7,5]]},

    'Dropout'             : {'in': [[2,7,3], [0.25], [True]], 'out': [[2,7,3],[2,7,3]]},
    #'Split'               : {'in': [], 'out': []},
    #'Concat'              : {'in': [], 'out': []},
    #'Gather'              : {'in': [], 'out': []},
    #'Identity'            : {'in': [[55,7,21]], 'out': [[55,7,21]]},
    #'Tanh'                : {'in': [], 'out': []},
    #'Pow'                 : {'in': [], 'out': []},
    #'Constant'            : {'in': [], 'out': []},


    # SPECIAL HANDLING....2nd input defines a shape instead of data
    'Reshape'             : {'in': [[4,3,2,7], [4,3,14]], 'out': [[4,3,14]]},

    # 'Where'               : {'in': [], 'out': []},
    # 'Unsqueeze'           : {'in': [], 'out': []},
    # 'Slice'               : {'in': [], 'out': []},
    # 'Equal'               : {'in': [], 'out': []},
    # 'Cast'                : {'in': [], 'out': []},
    # 'Shape'               : {'in': [], 'out': []},
    # 'Range'               : {'in': [], 'out': []},
    # 'Conv'                : {'in': [], 'out': []},
    # 'MaxPool'             : {'in': [], 'out': []},
        }

make_shape = build_tmp_fp32_tensor_from_shape

def check_perf_counts(x,ot):
    for f in [ 'inElems', 'outElems', 'inBytes', 'outBytes', 'instrs']:
        assert f in x, f"Field {f} missing in get_perf_counts() for {ot}"
    return

def test_backward_ops(tmp_path_factory):
    odir = tmp_path_factory.mktemp('bwd_ops_test')
    os.makedirs(odir, exist_ok=True)
    for op_num, (op_type, op_spec) in enumerate(OPTYPES.items()):
        op_name   = f'OP_{op_num}_{op_type}'
        if op_type == 'Reshape':
            i_tensors = [
                    make_shape(op_spec['in'][0], f'{op_name}_ITensor_0'),
                    build_tmp_data_tensor(np.array(op_spec['in'][1]), f'{op_name}_ITensor_1')
                    ]
            i_tensors[1].is_const = True
        elif op_type == 'Dropout':
            x0 = make_shape(op_spec['in'][0], f'{op_name}_ITensor_0')
            x1 = build_tmp_data_tensor(np.array(op_spec['in'][1]), f'{op_name}_ITensor_1')
            x2 = build_tmp_data_tensor(np.array(op_spec['in'][2]), f'{op_name}_ITensor_2')
            x1.is_const = True
            x2.is_const = True
            i_tensors = [x0,x1,x2]
        elif op_type == 'LayerNormalization':
            pass
            # TODO: enable LayerNormalization after backward pass implemented for it
            # x0 = make_shape(op_spec['in'][0], f'{op_name}_ITensor_0')
            # x1 = make_shape(op_spec['in'][1], f'{op_name}_ITensor_1')
            # x2 = make_shape(op_spec['in'][2], f'{op_name}_ITensor_2')
            # x1.is_param = True
            # x2.is_param = True
            # i_tensors = [x0,x1,x2]
        else:
            i_tensors = [make_shape(x, f'{op_name}_ITensor_{i}') for i,x in enumerate(op_spec['in'])]

        if op_type == 'LayerNormalization':
            pass
            # TODO: enable LayerNormalization after backward pass implemented for it
            # y0 = make_shape(op_spec['out'][0], f'{op_name}_OTensor_0')
            # y1 = make_shape(op_spec['out'][1], f'{op_name}_OTensor_1')
            # y2 = make_shape(op_spec['out'][2], f'{op_name}_OTensor_2')
            # y1.has_grad = False
            # y2.has_grad = False
            # o_tensors = [y0,y1,y2]
        else:
            o_tensors = [make_shape(x, f'{op_name}_OTensor_{i}') for i,x in enumerate(op_spec['out'])]

        op_info = {
                'name'   : op_name,
                'optype' : op_type,
                'inList' : [x.name for x in i_tensors],
                'outList': [x.name for x in o_tensors]
        }
        if 'attrs' in op_spec:
            op_info.update({'attrs': op_spec['attrs']})

        op_cls = SimOpFactory(op_type)
        op_obj = op_cls(op_info)

        for x in i_tensors: x.op_in  = [op_name]
        for x in o_tensors: x.op_out = [op_name]

        gg = WorkloadGraph(op_name + '.graph')
        print(str(gg))

        for x in i_tensors + o_tensors:
            gg.add_tensor(x)
        gg.add_op(op_obj)
        gg.construct_graph()

        op_perf = op_obj.get_perf_counts(i_tensors, o_tensors)
        check_perf_counts(op_perf, op_type)

        fwd_onnx_filename = os.path.join(odir, op_name + '_fwd.onnx')
        gg.graph2onnx(fwd_onnx_filename)

        bgg = BackwardWorkloadGraph(gg)

        for _,bwd_op in bgg._bwd_graph._ops.items():
            itensors = [bgg._bwd_graph._tensors[x] for x in bwd_op.inList]
            otensors = [bgg._bwd_graph._tensors[x] for x in bwd_op.outList]

            bwd_op_perf = bwd_op.get_perf_counts(itensors, otensors, is_backprop=True, batch_axis=0)#, bias_axis=1)
            check_perf_counts(bwd_op_perf, bwd_op.optype)

        bwd_onnx_filename = os.path.join(odir, op_name + '_bwd.onnx')
        bgg._bwd_graph.graph2onnx(bwd_onnx_filename, do_model_check=False)

        del op_obj
        del gg
        del bgg
        del i_tensors
        del o_tensors