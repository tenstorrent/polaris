#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0
import logging
import numpy as np
from workloads.basicmlp import BasicMLP
from ttsim.graph import BackwardWorkloadGraph
import ttsim.front.functional.op as F

def _backwardpass(tmp_path):
    model_cfgs ={
            'M1': {'mm_dims': [32, 64, 16, 4, 1], 'bs': 1 },
            'M2': {'mm_dims': [32, 64, 16], 'bs': 1, 'with_mul': True},
            'M3': {'mm_dims': [32, 64, 16], 'bs': 1, 'with_bias': True},
            'M4': {'mm_dims': [32, 64, 16], 'bs': 1, 'with_relu': True},
            'M5': {'mm_dims': [32, 64, 16], 'bs': 1, 'with_transpose': True},
            'M6': {'mm_dims': [32, 64], 'bs': 1, 'with_softmax': True},
            'M7': {'mm_dims': [32, 64], 'bs': 1, 'with_gelu': True},
            }

    basedir = tmp_path / "onnxdumps"
    basedir.mkdir()
    for m_name, m_cfg in model_cfgs.items():
        logging.info('Processing Model: %s', m_name)
        m_obj = BasicMLP(m_name, m_cfg)
        m_obj.set_batch_size(3)
        m_obj.create_input_tensors()
        y = m_obj()
        z = F._from_shape('Z', y.shape, np_dtype=np.float32)
        loss = np.sum((y.data-z.data)**2)/y.data.size #euclidean distance...
        loss_gradient = (y.data - z.data)/y.data.size

        logging.info('  OUTPUT TENSOR      : %s', y.name)
        logging.info('  GROUND TRUTH       : %s', z.name)
        logging.info('  LOSS               : %.2f', loss)
        logging.info('  LOSS GRADIENT SHAPE: %s', loss_gradient.shape)

        g = m_obj.get_forward_graph()
        fwd_onnxfile = basedir / f'{m_name}_FWD.onnx'
        bwd_onnxfile = basedir / f'{m_name}_BWD.onnx'
        g.graph2onnx(fwd_onnxfile)

        bg = BackwardWorkloadGraph(g)

        # TODO: Call this in topological order!!
        for _,o in bg._bwd_graph._ops.items():
            itensors = [bg._bwd_graph._tensors[x] for x in o.inList]
            otensors = [bg._bwd_graph._tensors[x] for x in o.outList]
            o.get_perf_counts(itensors, otensors, is_backprop=True, batch_axis=0, bias_axis=1)

        bg._bwd_graph.graph2onnx(bwd_onnxfile, do_model_check=False)

        # TODO: What are the asserts to add here
        del m_obj
        del g
        del bg
