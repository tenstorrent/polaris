# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0
# """
# torch.fx-based op count for BasicLLM PyTorch port.
# Uses symbolic tracing to capture the computation graph at nn.Module granularity,
# which more closely matches TTSim's high-level op vocabulary than ONNX export.
# """
# import os, sys, json
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# from collections import Counter
# import torch
# from torch.fx import symbolic_trace

# from workloads.torch_basic_llm import BasicLLM

# FX_TO_TTSIM = {
#     'Linear':           'MatMul',
#     'LayerNorm':        'LayerNormalization',
#     'Embedding':        'Gather',
#     'GELU':             'Gelu',
#     'Dropout':          'Dropout',
#     'Softmax':          'Softmax',
#     'matmul':           'MatMul',
#     'softmax':          'Softmax',
#     'add':              'Add',
#     'mul':              'Mul',
#     'truediv':          'Div',
#     'reshape':          'Reshape',
#     'transpose':        'Transpose',
#     'chunk':            'Split',
#     'getitem':          None,
# }

# def count_fx_ops(model):
#     traced = symbolic_trace(model)
#     counts: Counter[str] = Counter()
#     for node in traced.graph.nodes:
#         if node.op == 'call_module':
#             mod = traced.get_submodule(node.target)
#             cls = type(mod).__name__
#             mapped = FX_TO_TTSIM.get(cls)
#             if mapped:
#                 counts[mapped] += 1
#         elif node.op == 'call_function':
#             fn_name = getattr(node.target, '__name__', str(node.target))
#             mapped = FX_TO_TTSIM.get(fn_name)
#             if mapped:
#                 counts[mapped] += 1
#         elif node.op == 'call_method':
#             mapped = FX_TO_TTSIM.get(node.target)
#             if mapped:
#                 counts[mapped] += 1
#     return counts

# if __name__ == '__main__':
#     INSTANCES = {
#         'nano':   dict(nL=2,  nH=4,  dE=128,  nW=64),
#         'micro':  dict(nL=4,  nH=4,  dE=256,  nW=128),
#         'small':  dict(nL=8,  nH=8,  dE=384,  nW=512),
#         'base':   dict(nL=12, nH=12, dE=768,  nW=1024),
#         'large':  dict(nL=24, nH=16, dE=1024, nW=2048),
#         'xlarge': dict(nL=24, nH=16, dE=1536, nW=2048),
#     }
#     vocab_sz = 256

#     print(f'{"instance":<10} fx_op_counts')
#     print('-'*70)

#     for name, cfg in INSTANCES.items():
#         model = BasicLLM(vocab_sz, cfg['nW'], cfg['dE'], cfg['nH'], cfg['nL'])
#         model.eval()
#         try:
#             counts = count_fx_ops(model)
#             print(f'{name:<10}', dict(sorted(counts.items())))
#             with open(f'/tmp/basic_llm_{name}_fx.json', 'w') as f:
#                 json.dump(dict(counts), f)
#         except Exception as e:
#             print(f'{name:<10}  FX trace failed: {e}')
