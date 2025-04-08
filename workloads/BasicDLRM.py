#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from typing import Any
import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
import numpy as np

class BasicDLRM(SimNN.Module):
    # Type hints for instance attributes
    input_tensors: dict[str, Any]

    def __init__(self, name, cfg):
        super().__init__()
        self.name                 = name
        self.bs                   = cfg['bs']
        self.cat_features         = [ (cat['size'], cat['dim']) for cat in cfg['cat_features'] ]
        self.bottom_mlp_layers    = cfg['bottom_mlp']['layers']
        self.bottom_mlp_input_dim = cfg['bottom_mlp']['input_dim']
        self.top_mlp_layers       = cfg['top_mlp']['layers']
        self.top_mlp_input_dim    = cfg['top_mlp']['input_dim']
        self.interaction          = cfg['interaction']

        assert self.interaction == 'dot', f"Interaction {self.interaction} not supported for now!!"

        #Operators

        # 1. Embeddings
        self.embedding_layers = F.SimOpHandleList([
            F.Embedding(f"embedding_{i}", s,d) for i,(s,d) in enumerate(self.cat_features)
            ])

        # 2. Bottom MLP
        btm_ops = []
        i_dim = self.bottom_mlp_input_dim
        for li, o_dim in enumerate(self.bottom_mlp_layers):
            btm_ops.append(F.Linear(f"bottom_mlp_{li}.Linear", i_dim, o_dim))
            btm_ops.append(F.Relu(f"bottom_mlp_{li}.Relu"))
            i_dim = o_dim
        self.bottom_mlp = F.SimOpHandleList(btm_ops)

        # 3. Feature Interaction
        self.concat          = F.ConcatX("feature_concat", axis=1)
        self.unsqueeze_1     = F.Unsqueeze("feature_unsqueeze_1")
        self.unsqueeze_2     = F.Unsqueeze("feature_unsqueeze_2")
        self.matmul          = F.MatMul("feature_dot_product")
        self.trilu           = F.TriluX("feature_trilu")
        self.concat2         = F.ConcatX("feature_concat2", axis=1)
        self.unsqueeze_dim_1 = F._from_data('const_1', np.array([1]), is_param=False, is_const=True)
        self.unsqueeze_dim_2 = F._from_data('const_2', np.array([2]), is_param=False, is_const=True)

        # 4. Top MLP
        top_ops = []
        i_dim = self.top_mlp_input_dim
        for li, o_dim in enumerate(self.top_mlp_layers):
            top_ops.append(F.Linear(f"top_mlp_{li}.Linear", i_dim, o_dim))
            top_ops.append(F.Relu(f"top_mlp_{li}.Relu"))
            i_dim = o_dim
        self.top_mlp = F.SimOpHandleList(top_ops)

        super().link_op2module()

    def set_batch_size(self, new_bs):
        self.bs = new_bs

    def create_input_tensors(self):
        self.input_tensors = {
                'dense_features': F._from_shape('dense_features',
                                                [self.bs, self.bottom_mlp_input_dim],
                                                is_param=False, np_dtype=np.float32),
                'sparse_indices': [
                    F._from_shape(f"emb_in_{i}",
                                  [self.bs], is_param=False, np_dtype=np.int64)
                    for i,_ in enumerate(self.cat_features)
                    ]
                }
        return

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG


    def __call__(self):
        assert len(self.input_tensors) == 2, f"input_tensors missing!! Need create_input_tensors() before __call__: {self.input_tensors}"

        dense_features: SimTensor = self.input_tensors['dense_features']
        sparse_indices: list[SimTensor] = self.input_tensors['sparse_indices']

        # Bottom MLP for dense features
        dense_out = self.bottom_mlp(dense_features)

        # Embedding lookups for sparse features
        sparse_out = [] #emb(sparse_indices[i]) for i, emb in enumerate(self.embedding_layers)]
        for i, emb in enumerate(self.embedding_layers):
            xo = emb(sparse_indices[i])
            sparse_out.append(xo)

        # Feature interaction
        features          = self.concat(dense_out, *sparse_out)
        unsqueeze_1_out   = self.unsqueeze_1(features, self.unsqueeze_dim_1)
        unsqueeze_2_out   = self.unsqueeze_2(features, self.unsqueeze_dim_2)
        interaction       = self.matmul(unsqueeze_2_out, unsqueeze_1_out)
        interaction2      = self.trilu(interaction)
        combined_features = self.concat2(dense_out, interaction2)

        # Top MLP for final prediction
        out = self.top_mlp(combined_features)

        return out

    def analytical_param_count(self):
        return 100 #dummy for now


def run_standalone(outdir: str='.') -> None:
    dlrm_cfg = {
            'bottom_mlp': {
                'input_dim': 64,
                'layers': [128, 64, 16]
                },
            'top_mlp': {
                'input_dim': 1144,
                'layers': [64, 32, 1]
                },
            'cat_features': [
                {'size': 10000, 'dim': 16},
                {'size': 20000, 'dim': 16},
                ],
            'interaction': 'dot',
            'bs': 1
            }
    dlrm_obj = BasicDLRM('dlrm', dlrm_cfg)
    dlrm_obj.set_batch_size(8)
    dlrm_obj.create_input_tensors()
    y = dlrm_obj()
    gg = dlrm_obj.get_forward_graph()
    gg.graph2onnx(f'{outdir}/mydlrm.onnx')


if __name__ == '__main__':
    run_standalone()