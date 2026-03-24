#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
VectorInstanceMemory Comparison: PyTorch vs ttsim
Tests positional encoding and transformation matrix generation
"""

import os, sys

polaris_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
sys.path.insert(0, polaris_path)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from scipy.spatial.transform import Rotation as R
from einops import repeat
import torch
from torch import nn

import ttsim.front.functional.op as F

# ============================================================
# PyTorch Reference Implementation (Self-Contained, matching source structure)
# ============================================================


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1DPyTorch(nn.Module):
    """
    1D Positional Encoding (PyTorch version matching source)
    :param channels: The last dimension of the tensor you want to apply pos emb to.
    """

    def __init__(self, channels):
        super(PositionalEncoding1DPyTorch, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None)

        # Pre-compute emb_cache for compatibility (like in VectorInstanceMemory.__init__)
        # In source: fake_tensor = torch.zeros((1, 1000, dim_in))
        # self.cached_pe = p_enc_1d(fake_tensor)[0]
        fake_tensor = torch.zeros((1, 1000, self.org_channels))
        cached_result = self.forward(fake_tensor)
        self.emb_cache = (
            cached_result[0].cpu().numpy()
        )  # Store as numpy for compatibility

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class VectorInstanceMemoryPyTorch:
    """Memory bank for tracking instances (PyTorch reference)"""

    def __init__(self, dim_in, number_ins, bank_size, mem_len):
        self.dim_in = dim_in
        self.number_ins = number_ins
        self.max_number_ins = 3 * number_ins
        self.bank_size = bank_size
        self.mem_len = mem_len

        self.pos_encoder = PositionalEncoding1DPyTorch(dim_in)
        self.cached_pe = self.pos_encoder.emb_cache

        self.mem_bank = None
        self.mem_bank_seq_id = None
        self.mem_bank_trans = None
        self.mem_bank_rot = None

    def init_memory(self, bs):
        # Main memory banks (torch tensors matching source, CPU instead of CUDA)
        self.mem_bank = torch.zeros(
            (self.bank_size, bs, self.max_number_ins, self.dim_in), dtype=torch.float32
        )
        self.mem_bank_seq_id = torch.zeros(
            (self.bank_size, bs, self.max_number_ins), dtype=torch.long
        )
        self.mem_bank_trans = torch.zeros((self.bank_size, bs, 3), dtype=torch.float32)
        self.mem_bank_rot = torch.zeros((self.bank_size, bs, 3, 3), dtype=torch.float32)

        # Current pose (matching source: torch.zeros((bs,3,3)))
        self.curr_rot = torch.zeros((bs, 3, 3), dtype=torch.float32)
        self.curr_trans = torch.zeros((bs, 3), dtype=torch.float32)

        # Memory recording information (lists indexed by batch, matching source structure)
        self.instance2mem = [{} for _ in range(bs)]
        self.num_ins = [0 for _ in range(bs)]
        self.active_mem_ids = [None for _ in range(bs)]
        init_entry_length = torch.tensor([0] * self.max_number_ins).long()
        self.mem_entry_lengths = [init_entry_length.clone() for _ in range(bs)]

    def prepare_transformation_batch(
        self, history_e2g_trans, history_e2g_rot, curr_e2g_trans, curr_e2g_rot
    ):
        N = len(history_e2g_trans)

        # Convert numpy inputs to torch tensors
        history_e2g_trans = torch.from_numpy(history_e2g_trans)
        history_e2g_rot = torch.from_numpy(history_e2g_rot)
        curr_e2g_trans = torch.from_numpy(curr_e2g_trans)
        curr_e2g_rot = torch.from_numpy(curr_e2g_rot)

        # Historical global-to-ego matrices (matching source)
        history_g2e_matrix = torch.stack(
            [
                torch.eye(4, dtype=torch.float64),
            ]
            * len(history_e2g_trans),
            dim=0,
        )
        history_g2e_matrix[:, :3, :3] = torch.transpose(history_e2g_rot, 1, 2)
        history_g2e_matrix[:, :3, 3] = -torch.bmm(
            torch.transpose(history_e2g_rot, 1, 2), history_e2g_trans[..., None]
        ).squeeze(-1)

        curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

        curr_e2g_matrix = torch.eye(4, dtype=torch.float64)
        curr_e2g_matrix[:3, :3] = curr_e2g_rot
        curr_e2g_matrix[:3, 3] = curr_e2g_trans

        history_e2g_matrix = torch.stack(
            [
                torch.eye(4, dtype=torch.float64),
            ]
            * len(history_e2g_trans),
            dim=0,
        )
        history_e2g_matrix[:, :3, :3] = history_e2g_rot
        history_e2g_matrix[:, :3, 3] = history_e2g_trans

        history_curr2prev_matrix = torch.bmm(
            history_g2e_matrix,
            repeat(curr_e2g_matrix, "n1 n2 -> r n1 n2", r=len(history_g2e_matrix)),
        )
        history_prev2curr_matrix = torch.bmm(
            repeat(curr_g2e_matrix, "n1 n2 -> r n1 n2", r=len(history_e2g_matrix)),
            history_e2g_matrix,
        )

        # Convert back to numpy for comparison
        return history_curr2prev_matrix.numpy(), history_prev2curr_matrix.numpy()


# Import ttsim implementation
from workloads.MapTracker.plugin.models.maper.vector_memory import (
    PositionalEncoding1D as PositionalEncoding1DTtsim,
    VectorInstanceMemory as VectorInstanceMemoryTtsim,
)

print("=" * 70)
print("VectorInstanceMemory Comparison: PyTorch vs ttsim")
print("=" * 70)

# ============================================================
# TEST 1: PositionalEncoding1D
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: PositionalEncoding1D")
print("-" * 70)

channels = 256
batch_size = 2
seq_len = 10

print(f"\nConfiguration:")
print(f"  channels={channels}")
print(f"  batch_size={batch_size}, seq_len={seq_len}")

# PyTorch reference (using forward() like the source, with torch tensors)
pe_pytorch = PositionalEncoding1DPyTorch(channels)
dummy_input_pytorch = torch.zeros((batch_size, seq_len, channels), dtype=torch.float32)
output_pytorch_tensor = pe_pytorch.forward(dummy_input_pytorch)
output_pytorch = output_pytorch_tensor.cpu().numpy()  # Convert to numpy for comparison

print(f"\nPyTorch PositionalEncoding1D:")
print(f"  Output shape: {output_pytorch.shape}")
print(f"  Output stats:")
print(f"    Min:  {output_pytorch.min():.6f}")
print(f"    Max:  {output_pytorch.max():.6f}")
print(f"    Mean: {output_pytorch.mean():.6f}")
print(f"    Std:  {output_pytorch.std():.6f}")
print(f"  Sample [0, 0, :5]: {output_pytorch[0, 0, :5]}")

# ttsim
pe_ttsim = PositionalEncoding1DTtsim("pe_test", channels)
dummy_input = F._from_data(
    "dummy", np.zeros((batch_size, seq_len, channels), dtype=np.float32)
)
output_ttsim_tensor = pe_ttsim(dummy_input)

print(f"\nttsim PositionalEncoding1D:")
print(f"  Output shape: {output_ttsim_tensor.shape}")
print(f"  Output .data is None? {output_ttsim_tensor.data is None}")

if output_ttsim_tensor.data is not None:
    output_ttsim = output_ttsim_tensor.data
    print(f"  Output stats:")
    print(f"    Min:  {output_ttsim.min():.6f}")
    print(f"    Max:  {output_ttsim.max():.6f}")
    print(f"    Mean: {output_ttsim.mean():.6f}")
    print(f"    Std:  {output_ttsim.std():.6f}")
    print(f"  Sample [0, 0, :5]: {output_ttsim[0, 0, :5]}")

    # Numerical comparison
    diff = np.abs(output_pytorch - output_ttsim)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"\nNumerical Comparison:")
    print(f"  Max absolute difference:  {max_diff:.10e}")
    print(f"  Mean absolute difference: {mean_diff:.10e}")

    atol = 1e-6
    rtol = 1e-5
    matches = np.allclose(output_pytorch, output_ttsim, atol=atol, rtol=rtol)

    if matches:
        print(f"  [PASS] Outputs match within tolerance (atol={atol}, rtol={rtol})")
    else:
        print(f"  [FAIL] Outputs differ beyond tolerance")
else:
    print(f"  [WARN] [SKIP] Cannot compare - ttsim output.data is None")


# ============================================================
# TEST 2: Memory Initialization
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Memory Initialization")
print("-" * 70)

dim_in = 256
number_ins = 50
bank_size = 4
mem_len = 4
batch_size = 2

print(f"\nConfiguration:")
print(f"  dim_in={dim_in}, number_ins={number_ins}")
print(f"  bank_size={bank_size}, mem_len={mem_len}")
print(f"  batch_size={batch_size}")

# PyTorch reference
mem_pytorch = VectorInstanceMemoryPyTorch(dim_in, number_ins, bank_size, mem_len)
mem_pytorch.init_memory(batch_size)

print(f"\nPyTorch VectorInstanceMemory:")
print(f"  mem_bank shape: {mem_pytorch.mem_bank.shape}")
print(f"  mem_bank_seq_id shape: {mem_pytorch.mem_bank_seq_id.shape}")
print(f"  mem_bank_trans shape: {mem_pytorch.mem_bank_trans.shape}")
print(f"  mem_bank_rot shape: {mem_pytorch.mem_bank_rot.shape}")
print(f"  num_ins[0]: {mem_pytorch.num_ins[0]}")

# ttsim
mem_ttsim = VectorInstanceMemoryTtsim(
    "mem_test", dim_in, number_ins, bank_size, mem_len
)
mem_ttsim.init_memory(batch_size)

print(f"\nttsim VectorInstanceMemory:")
print(f"  mem_bank shape: {mem_ttsim.mem_bank.shape}")
print(f"  mem_bank_seq_id shape: {mem_ttsim.mem_bank_seq_id.shape}")
print(f"  mem_bank_trans shape: {mem_ttsim.mem_bank_trans.shape}")
print(f"  mem_bank_rot shape: {mem_ttsim.mem_bank_rot.shape}")
print(f"  num_ins[0]: {mem_ttsim.num_ins[0]}")

# Compare shapes
shapes_match = (
    mem_pytorch.mem_bank.shape == mem_ttsim.mem_bank.shape
    and mem_pytorch.mem_bank_seq_id.shape == mem_ttsim.mem_bank_seq_id.shape
    and mem_pytorch.mem_bank_trans.shape == mem_ttsim.mem_bank_trans.shape
    and mem_pytorch.mem_bank_rot.shape == mem_ttsim.mem_bank_rot.shape
)

# Compare values (should all be zeros)
# ttsim uses numpy arrays, pytorch uses torch tensors
values_match = np.allclose(mem_pytorch.mem_bank.numpy(), mem_ttsim.mem_bank)

print(f"\nComparison:")
if shapes_match and values_match:
    print(f"  [PASS] Memory initialization matches")
else:
    print(f"  [FAIL] Memory initialization mismatch")
    if not shapes_match:
        print(f"    Shapes don't match")
    if not values_match:
        print(f"    Values don't match")


# ============================================================
# TEST 3: Transformation Matrices
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: Transformation Matrices")
print("-" * 70)

# Create sample pose data
np.random.seed(42)
N = 3
history_trans = np.random.randn(N, 3).astype(np.float32)
history_rot_euler = np.random.randn(N, 3).astype(np.float32) * 0.1
history_rot = R.from_euler("zxy", history_rot_euler).as_matrix().astype(np.float32)

curr_trans = np.random.randn(3).astype(np.float32)
curr_rot_euler = np.random.randn(3).astype(np.float32) * 0.1
curr_rot = R.from_euler("zxy", curr_rot_euler).as_matrix().astype(np.float32)

print(f"\nConfiguration:")
print(f"  N={N} historical frames")
print(f"  history_trans shape: {history_trans.shape}")
print(f"  history_rot shape: {history_rot.shape}")
print(f"  curr_trans shape: {curr_trans.shape}")
print(f"  curr_rot shape: {curr_rot.shape}")

print(f"\nInput data:")
print(f"  history_trans:\n{history_trans}")
print(f"  curr_trans: {curr_trans}")

# PyTorch reference
mem_pytorch = VectorInstanceMemoryPyTorch(dim_in, number_ins, bank_size, mem_len)
curr2prev_pytorch, prev2curr_pytorch = mem_pytorch.prepare_transformation_batch(
    history_trans, history_rot, curr_trans, curr_rot
)

print(f"\nPyTorch prepare_transformation_batch:")
print(f"  curr2prev shape: {curr2prev_pytorch.shape}")
print(f"  prev2curr shape: {prev2curr_pytorch.shape}")
print(f"  curr2prev stats:")
print(f"    Min:  {curr2prev_pytorch.min():.6f}")
print(f"    Max:  {curr2prev_pytorch.max():.6f}")
print(f"    Mean: {curr2prev_pytorch.mean():.6f}")

# ttsim
mem_ttsim = VectorInstanceMemoryTtsim(
    "mem_test", dim_in, number_ins, bank_size, mem_len
)
curr2prev_ttsim, prev2curr_ttsim = mem_ttsim.prepare_transformation_batch(
    history_trans, history_rot, curr_trans, curr_rot
)

print(f"\nttsim prepare_transformation_batch:")
print(f"  curr2prev shape: {curr2prev_ttsim.shape}")
print(f"  prev2curr shape: {prev2curr_ttsim.shape}")
print(f"  curr2prev stats:")
print(f"    Min:  {curr2prev_ttsim.min():.6f}")
print(f"    Max:  {curr2prev_ttsim.max():.6f}")
print(f"    Mean: {curr2prev_ttsim.mean():.6f}")

# Numerical comparison
print(f"\nNumerical Comparison:")

curr2prev_diff = np.abs(curr2prev_pytorch - curr2prev_ttsim)
prev2curr_diff = np.abs(prev2curr_pytorch - prev2curr_ttsim)

print(f"  curr2prev:")
print(f"    Max diff:  {np.max(curr2prev_diff):.10e}")
print(f"    Mean diff: {np.mean(curr2prev_diff):.10e}")

print(f"  prev2curr:")
print(f"    Max diff:  {np.max(prev2curr_diff):.10e}")
print(f"    Mean diff: {np.mean(prev2curr_diff):.10e}")

atol = 1e-5
rtol = 1e-4
curr2prev_match = np.allclose(curr2prev_pytorch, curr2prev_ttsim, atol=atol, rtol=rtol)
prev2curr_match = np.allclose(prev2curr_pytorch, prev2curr_ttsim, atol=atol, rtol=rtol)

# Check that matrices are valid transformations (inverse check)
identity = np.tile(np.eye(4), (N, 1, 1))
identity_check_pytorch = np.allclose(
    np.matmul(curr2prev_pytorch, prev2curr_pytorch), identity, atol=1e-4, rtol=1e-4
)
identity_check_ttsim = np.allclose(
    np.matmul(curr2prev_ttsim, prev2curr_ttsim), identity, atol=1e-4, rtol=1e-4
)

print(f"\nTransformation validity:")
print(f"  PyTorch inverse check (curr2prev @ prev2curr = I): {identity_check_pytorch}")
print(f"  ttsim inverse check (curr2prev @ prev2curr = I):   {identity_check_ttsim}")

all_match = (
    curr2prev_match
    and prev2curr_match
    and identity_check_pytorch
    and identity_check_ttsim
)

if all_match:
    print(f"\n  [PASS] Transformation matrices match (atol={atol}, rtol={rtol})")
else:
    print(f"\n  [FAIL] Transformation matrices mismatch")
    if not curr2prev_match:
        print(f"    curr2prev doesn't match")
    if not prev2curr_match:
        print(f"    prev2curr doesn't match")
    if not identity_check_pytorch or not identity_check_ttsim:
        print(f"    Inverse check failed")


# ============================================================
# TEST 4: Positional Encoding Cache Consistency
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Positional Encoding Cache Consistency")
print("-" * 70)

channels = 128

# PyTorch reference
pe_pytorch = PositionalEncoding1DPyTorch(channels)

# ttsim
pe_ttsim = PositionalEncoding1DTtsim("pe_cache_test", channels)

print(f"\nConfiguration:")
print(f"  channels={channels}")

print(f"\nCache comparison:")
print(f"  PyTorch cache shape: {pe_pytorch.emb_cache.shape}")
print(f"  ttsim cache shape: {pe_ttsim.emb_cache.shape}")

cache_diff = np.abs(pe_pytorch.emb_cache - pe_ttsim.emb_cache)
cache_max_diff = np.max(cache_diff)
cache_mean_diff = np.mean(cache_diff)

print(f"  Max cache difference:  {cache_max_diff:.10e}")
print(f"  Mean cache difference: {cache_mean_diff:.10e}")

atol = 1e-6
rtol = 1e-5
cache_match = np.allclose(
    pe_pytorch.emb_cache, pe_ttsim.emb_cache, atol=atol, rtol=rtol
)

# Test different sequence lengths
seq_lengths = [5, 20, 100]
all_match = cache_match

print(f"\nSequence length tests:")
for seq_len in seq_lengths:
    pytorch_slice = pe_pytorch.emb_cache[:seq_len, :]
    ttsim_slice = pe_ttsim.emb_cache[:seq_len, :]
    match = np.allclose(pytorch_slice, ttsim_slice, atol=atol, rtol=rtol)
    status = "[PASS]" if match else "[FAIL]"
    print(f"  {status} seq_len={seq_len}: cache match")
    all_match = all_match and match

if all_match:
    print(f"\n  [PASS] Positional encoding cache consistent (atol={atol}, rtol={rtol})")
else:
    print(f"\n  [FAIL] Positional encoding cache mismatch")

print()
print("=" * 70)
print("Test Complete!")
print("=" * 70)
