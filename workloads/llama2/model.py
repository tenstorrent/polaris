#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import math
from dataclasses import dataclass
from typing import Optional, Tuple

from loguru import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import numpy as np

import ttsim.front.functional.op as F
import ttsim.front.functional.sim_nn as SimNN
import ttsim.front.functional.tensor_op as T


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(SimNN.Module):
    def __init__(self, objname: str, dim: int, eps: float = 1e-6):
        super().__init__()
        self.name = objname
        self.eps = eps
        self.weight = F._from_shape(f'{self.name}_weight', shape=[dim], is_param=True)
        self.mulopx2 = F.Mul(f'{self.name}_mulopx2')
        self.meanop = F.Mean(f'{self.name}_meanop', dim=-1)
        self.sqrtop = F.Sqrt(f'{self.name}_sqrtop')
        self.reciprocalop = F.Reciprocal(f'{self.name}_reciprocalop')
        self.mulop = F.Mul(f'{self.name}_mulop')
        super().link_op2module()

    def _norm(self, x):
        x2 = self.mulopx2(x,x)
        mu = self.meanop(x2).unsqueeze(-1) ## y_flat.pow(2) substituted with mul ## unsqueeze for keepdim=True
        rmsnorm_eps_tensor = F._from_shape('rmsnorm_eps', shape=mu.shape)
        return x * self.reciprocalop(self.sqrtop(mu + rmsnorm_eps_tensor))

    def __call__(self, x):
        output = self._norm(x) #self._norm(x.float()).type_as(x)
        y = self.mulop(output, self.weight)
        return y

class Precompute(SimNN.Module):
    def __init__(self, objname, dim: int, end: int, theta: float = 10000.0):
        super().__init__()
        self.name = objname
        freqs_t = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)] / dim))
        t_t = np.arange(end)
        self.freqs = F._from_shape('freqs', shape=[1, len(freqs_t)])
        self.t = F._from_shape('t', shape=[len(t_t), 1])
        self.matmulop = F.MatMul('matmul_freqs')
        super().link_op2module()

    def __call__(self):
        matmulout = self.matmulop(self.t, self.freqs)
        freqs_cis = matmulout[0:1, :]
        freqs_cis_shape = freqs_cis.shape
        freqs_cis= F._from_shape('freqs_cis', [*freqs_cis_shape, 2])
        return freqs_cis

def reshape_for_broadcast(freqs_cis: SimNN.SimTensor, x: SimNN.SimTensor) -> SimNN.SimTensor:
    ndim = len(x.shape)
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == [x.shape[1], x.shape[-2], x.shape[-1]]
    shape = [d if i == 1 or i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape) # type: ignore

def apply_rotary_emb(
    xq: SimNN.SimTensor,
    xk: SimNN.SimTensor,
    freqs_cis: SimNN.SimTensor,
) -> Tuple[SimNN.SimTensor, SimNN.SimTensor]:

    xqr = xq.reshape(*xq.shape[:-1], -1, 2) # type: ignore
    xkr = xk.reshape(*xk.shape[:-1], -1, 2) # type: ignore
    freqs_cis = reshape_for_broadcast(freqs_cis, xqr)
    xq_out = (xqr * freqs_cis).flatten(3)
    xk_out = (xkr * freqs_cis).flatten(3)
    return xq_out, xk_out


def repeat_kv(x: SimNN.SimTensor, n_rep: int) -> SimNN.SimTensor:
    """repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    assert n_rep == 1, 'Implementation supports only n_rep == 1!'
    # return (
    #     x[:, :, :, None, :]
    #     .expand(bs, slen, n_kv_heads, n_rep, head_dim)
    #     .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    # )
    return x ## placeholder, should not reach here!


class Attention(SimNN.Module):
    """Multi-head attention module."""
    def __init__(self, objname: str, args: ModelArgs):
        super().__init__()
        self.name = objname
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1 #fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = F.Linear(
            f'{self.name}_wq',
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = F.Linear(
            f'{self.name}_wk',
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wv = F.Linear(
            f'{self.name}_wv',
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
        )
        self.wo = F.Linear(
            f'{self.name}_wo',
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )

        self.cache_k = F._from_shape(f'{self.name}_cache_k',
            [
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ]
        )
        self.cache_v = F._from_shape(f'{self.name}_cache_v',
            [
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            ]
        )
        self.matmulop = F.MatMul(f'{self.name}_matmul')
        self.softmaxop = F.Softmax(f'{self.name}_softmax', dim=-1)
        self.matmulop2 = F.MatMul(f'{self.name}_matmul2')
        super().link_op2module()

    def __call__(
        self,
        x: SimNN.SimTensor,
        start_pos: int,
        freqs_cis: SimNN.SimTensor,
        mask: Optional[SimNN.SimTensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)

        scores_mulout = self.matmulop(xq, keys.transpose(2, 3))
        scores = scores_mulout / F._from_data(f'{self.name}_divfactor', np.full(scores_mulout.shape, math.sqrt(self.head_dim)))
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = self.softmaxop(scores) #.type_as(xq)
        output = self.matmulop2(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(SimNN.Module):
    def __init__(
        self,
        objname: str,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        self.name = objname
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = F.Linear(
            f'{self.name}_w1',
            dim, hidden_dim, bias=False
        )
        self.w2 = F.Linear(
            f'{self.name}_w2',
            hidden_dim, dim, bias=False
        )
        self.w3 = F.Linear(
            f'{self.name}_w3',
            dim, hidden_dim, bias=False
        )
        self.reluop = F.Relu(f'{self.name}_reluop')
        super().link_op2module()

    def __call__(self, x):
        return self.w2(self.reluop(self.w1(x)) * self.w3(x)) # use Relu instead of Silu - for now


class TransformerBlock(SimNN.Module):
    def __init__(self, objname: str, layer_id: int, args: ModelArgs):
        super().__init__()
        self.name = objname
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(f'{self.name}_attention', args)
        self.feed_forward = FeedForward(
            f'{self.name}_feed_forward',
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(f'{self.name}_attention_norm', args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(f'{self.name}_ffn_norm', args.dim, eps=args.norm_eps)
        super().link_op2module()

    def __call__(
        self,
        x: SimNN.SimTensor,
        start_pos: int,
        freqs_cis: SimNN.SimTensor,
        mask: Optional[SimNN.SimTensor],
    ):
        h = x + self.attention(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(SimNN.Module):
    def __init__(self, objname: str, cfg):
        super().__init__()
        logger.info('Transformer config: {}', cfg)
        self.name = objname
        self.params = ModelArgs(
            dim=cfg['dim'],
            n_layers=cfg['n_layers'],
            n_heads=cfg['n_heads'],
            vocab_size=cfg['vocab_size'],
            max_batch_size=cfg['max_batch_size'],
            max_seq_len=cfg['max_seq_len'],
        )
        self.batch_size = cfg['bs']
        self.seq_len = cfg['seq_len']
        self.vocab_size = self.params.vocab_size
        self.n_layers = self.params.n_layers

        self.tok_embeddings = F.Embedding(f'{self.name}_tok_embeddings',
            self.params.vocab_size, self.params.dim
        )
        self.layers = SimNN.ModuleList([TransformerBlock(f'{self.name}_layer_{i}', i, self.params) for i in range(self.n_layers)])
        self.norm = RMSNorm(f'{self.name}_rmsnorm', self.params.dim, eps=self.params.norm_eps)
        self.output = F.Linear(f'{self.name}_output',
            self.params.dim, self.params.vocab_size, bias=False
        )
        precompute_obj = Precompute(f'{self.name}_precompute', self.params.dim // self.params.n_heads,
                                    self.params.max_seq_len * 2)
        self.freqs_cis = precompute_obj()
        super().link_op2module()

    def create_input_tensors(self):
        self.input_tensors = {
                'x_in': F._from_shape('input_tokens', [self.batch_size, self.seq_len]),
        }
        return

    def analytical_param_count(self):
        return 0

    def get_forward_graph(self):
        GG = super()._get_forward_graph(self.input_tensors)
        return GG

    def __call__(self, tokens: SimNN.SimTensor = None, start_pos: int = 0): # type: ignore[assignment]
        tokens = self.input_tensors['x_in'] if tokens is None else tokens
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis.set_module(self)
        freqs_cis = self.freqs_cis #[start_pos : start_pos + seqlen] ## assume seqlen = 1

        mask = None
        if seqlen > 1:
            raise NotImplementedError("Simulation for sequence length > 1 is not implemented yet!")

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h)
        return output
