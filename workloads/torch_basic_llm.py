# #!/usr/bin/env python
# # SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# # SPDX-License-Identifier: Apache-2.0
# """
# PyTorch port of workloads/BasicLLM.py for graph validation against TTSim.

# """
# import torch
# import torch.nn as nn
# import math

# class ATTN(nn.Module):
#     def __init__(self, dE, nH, drop_prob=0.0):
#         super().__init__()
#         self.dE, self.nH = dE, nH
#         self.dH = dE // nH
#         self.wqkv_proj = nn.Linear(dE, 3*dE)
#         self.w0_proj   = nn.Linear(dE, dE)
#         self.drop_attn   = nn.Dropout(drop_prob)
#         self.resid_drop  = nn.Dropout(drop_prob)
#         self.scale = 1.0 / math.sqrt(self.dH)

#     def forward(self, x):
#         B, N, _ = x.shape
#         WQKV = self.wqkv_proj(x)
#         Q, K, V = WQKV.chunk(3, dim=-1)
#         Q = Q.reshape(B, N, self.nH, self.dH).transpose(1, 2)
#         K = K.reshape(B, N, self.nH, self.dH).transpose(1, 2).transpose(2, 3)
#         V = V.reshape(B, N, self.nH, self.dH).transpose(1, 2)
#         QK = torch.matmul(Q, K) * self.scale
#         QK = self.drop_attn(torch.softmax(QK, dim=-1))
#         QKV = torch.matmul(QK, V).transpose(1, 2).reshape(B, N, self.dE)
#         return self.resid_drop(self.w0_proj(QKV))

# class TransformerBlock(nn.Module):
#     def __init__(self, dE, nH, drop_prob=0.0):
#         super().__init__()
#         idE = 4 * dE
#         self.ln_attn_in = nn.LayerNorm(dE)
#         self.attn       = ATTN(dE, nH, drop_prob)
#         self.lnorm      = nn.LayerNorm(dE)
#         self.ff1        = nn.Linear(dE, idE)
#         self.ff2        = nn.Linear(idE, dE)
#         self.gelu       = nn.GELU()
#         self.mlp_drop   = nn.Dropout(drop_prob)

#     def forward(self, x):
#         y = x + self.attn(self.ln_attn_in(x))
#         y = self.lnorm(y)
#         y = self.ff1(y)
#         y = self.ff2(y)
#         y = self.gelu(y)
#         y = self.mlp_drop(y)
#         return y

# class BasicLLM(nn.Module):
#     def __init__(self, vocab_sz, nW, dE, nH, nL, drop_prob=0.0):
#         super().__init__()
#         self.wte = nn.Embedding(vocab_sz, dE)
#         self.wpe = nn.Embedding(nW, dE)
#         self.drop_emb = nn.Dropout(drop_prob)
#         self.blocks   = nn.ModuleList([TransformerBlock(dE, nH, drop_prob) for _ in range(nL)])

#     def forward(self, tokens, mask):
#         x = self.wte(tokens) + self.wpe(mask)
#         x = self.drop_emb(x)
#         for blk in self.blocks:
#             x = blk(x)
#         return x

# if __name__ == '__main__':
#     import sys, json
#     from collections import Counter

#     INSTANCES = {
#         'nano':   dict(nL=2,  nH=4,  dE=128,  nW=64),
#         'micro':  dict(nL=4,  nH=4,  dE=256,  nW=128),
#         'small':  dict(nL=8,  nH=8,  dE=384,  nW=512),
#         'base':   dict(nL=12, nH=12, dE=768,  nW=1024),
#         'large':  dict(nL=24, nH=16, dE=1024, nW=2048),
#         'xlarge': dict(nL=24, nH=16, dE=1536, nW=2048),
#     }
#     vocab_sz, bs = 256, 1

#     print(f'{"instance":<10} {"PyTorch ONNX op counts":<60}')
#     print('-'*70)

#     for name, cfg in INSTANCES.items():
#         model = BasicLLM(vocab_sz, cfg['nW'], cfg['dE'], cfg['nH'], cfg['nL'])
#         model.eval()
#         tokens = torch.randint(0, vocab_sz, (bs, cfg['nW']))
#         mask   = torch.arange(cfg['nW']).unsqueeze(0).expand(bs, -1)

#         onnx_path = f'/tmp/basic_llm_{name}_torch.onnx'
#         torch.onnx.export(
#             model, (tokens, mask), onnx_path,
#             input_names=['tokens', 'mask'],
#             opset_version=17,
#             do_constant_folding=False,
#         )

#         import onnx
#         m = onnx.load(onnx_path)
#         op_counts = Counter(n.op_type for n in m.graph.node)
#         print(f'{name:<10}', dict(op_counts))
