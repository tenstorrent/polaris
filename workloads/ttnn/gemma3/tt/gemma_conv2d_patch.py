# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 Conv2D Patch Embedding

Reference: tt-metal models/demos/multimodal/gemma3/tt/gemma_conv2d_patch.py

Design note (why patch extraction never touches real pixel data):
Polaris/ttsim is a shape- and op-graph-driven performance simulator, not a numerics engine.
`ttsim.front.ttnn.Tensor` extends `SimTensor` (ttsim/ops/tensor.py), which tracks shape, dtype,
and graph linkage -- nothing downstream ever reads real values. The framework's own
`Tensor(some_ndarray, ...)` constructor path makes this explicit: it reads `.shape`/`.dtype`
off the array and then discards it (only `ttnn.as_tensor(...)` bothers to retain `.data`, and
even that is optional/unused by the cost model). So the correct, framework-idiomatic way to
build the patch-embedding input is to compute its shape directly from the incoming tensor's
`.shape` and construct a fresh shape/dtype-only ttnn.Tensor -- never to round-trip through a
real numpy array.

This also matches the reference architecturally: tt-metal runs its "unfold" (im2col) step on
the *host* with torch.nn.Unfold, strictly before the first `ttnn.as_tensor(...)` call, i.e. it
never touches the Tensix cores at all. Computing the output shape directly (rather than
tracking it as a ttnn.reshape/permute op) correctly mirrors that: Polaris attributes no
simulated device cost to it, exactly like the real hardware wouldn't. Everything from
`ttnn.Tensor(...)` / `ttnn.linear(...)` onward below *is* on-device and must go through ttnn
so Polaris's cost model can see it.
"""
import numpy as np
from typing import Optional

import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule
from workloads.ttnn.gemma3.common.gemma_utils import nearest_32, to_numpy


class TtGemmaConv2dPatch(LightweightModule):
    """Conv2D Patching layer (column-parallel over unfolded input).

    Input: any tensor (numpy array or ttnn.Tensor) with logical shape (bsz, in_channels, H, W).
    Output: on-device ttnn tensor of shape (1, bsz, num_patches, out_channels).
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix: str,
        dtype,
        in_channels: int = 3,
        out_channels: int = 1152,
        kernel_size: int = 14,
        stride: int = 14,
        bias: bool = True,
        image_size: int = 896,
    ):
        super().__init__()
        assert stride == kernel_size, (
            "TtGemmaConv2dPatch only implements the non-overlapping-patch case "
            "(stride == kernel_size), matching Gemma3's SigLIP patch embedding."
        )
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_bias = bias
        self.image_size = image_size

        self.patch_dim = in_channels * kernel_size * kernel_size
        self.num_patches_per_side = image_size // kernel_size
        self.num_patches = self.num_patches_per_side ** 2

        # Pad the flattened-patch (matmul K) dimension to a tile multiple, exactly like the
        # reference's `nearest_32(weight.shape[-1])` padding. This is not cosmetic: Polaris's
        # matmul cost model keys off the padded/tile shape, so an unpadded K here would make
        # the simulated linear cheaper than it would actually be on tile-based hardware.
        self.padded_patch_dim = nearest_32(self.patch_dim)
        pad_len = self.padded_patch_dim - self.patch_dim

        # ---- Weight preprocessing (host-side, one-time; mirrors the reference's torch prep) ----
        weight_key = f"{state_dict_prefix}weight"
        linear_weight_key = f"{state_dict_prefix}_linear.weight"
        bias_key = f"{state_dict_prefix}bias"
        linear_bias_key = f"{state_dict_prefix}_linear.bias"

        conv_weight = state_dict.get(weight_key, state_dict.get(linear_weight_key))
        if conv_weight is not None:
            weight_np = to_numpy(conv_weight)
            if weight_np.ndim == 4:
                # (out_channels, in_channels, kH, kW) -> (out_channels, patch_dim)
                weight_np = weight_np.reshape(out_channels, -1)
            weight_np = weight_np.astype(np.float32)
        else:
            weight_np = np.random.randn(out_channels, self.patch_dim).astype(np.float32) * 0.02

        if pad_len > 0:
            weight_np = np.concatenate(
                [weight_np, np.zeros((out_channels, pad_len), dtype=weight_np.dtype)], axis=-1
            )
        # (out_channels, padded_patch_dim) -> (1, 1, padded_patch_dim, out_channels), matching
        # the reference's `padded_weight.permute(1, 0).reshape(1, 1, -1, out_channels)`.
        weight_np = weight_np.transpose(1, 0).reshape(1, 1, self.padded_patch_dim, out_channels)

        self.weight = ttnn.Tensor(
            weight_np,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self.bias: Optional[ttnn.Tensor] = None
        if self.use_bias:
            bias_data = state_dict.get(bias_key, state_dict.get(linear_bias_key))
            if bias_data is not None:
                bias_np = to_numpy(bias_data).reshape(-1).astype(np.float32)
            else:
                bias_np = np.zeros(out_channels, dtype=np.float32)
            bias_np = bias_np.reshape(1, -1)
            self.bias = ttnn.Tensor(
                bias_np,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        # Match the reference's actual kernel config values (fp32_dest_acc_en / packer_l1_acc).
        # These select the hardware compute path Polaris looks up cost for -- getting them wrong
        # silently mis-simulates the matmul's cycle count, even though nothing crashes.
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _patch_output_shape(self, x) -> tuple[int, int]:
        """
        Compute (batch, num_patches) for an input of logical shape (B, in_channels, H, W).

        Deliberately shape-only: ``x`` may be a raw numpy array (production path, mirroring
        the reference's raw torch.Tensor input) or a ttnn.Tensor with no backing data at all
        (e.g. built via the framework's own ``ttnn.Tensor(array, ...)`` convenience path, which
        keeps shape/dtype but drops the array) -- both expose ``.shape``, and that's all this
        needs. We never fabricate a shape on mismatch; an invalid input raises immediately
        instead of silently desyncing Polaris's shape tracking.
        """
        x_shape = tuple(int(d) for d in x.shape)
        if len(x_shape) != 4:
            raise ValueError(f"TtGemmaConv2dPatch expected a 4D (B, C, H, W) input, got shape {x_shape}")

        B, C, H, W = x_shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {C} (shape {x_shape})")
        if H % self.kernel_size != 0 or W % self.kernel_size != 0:
            raise ValueError(f"Image size {(H, W)} is not divisible by kernel_size {self.kernel_size}")

        num_patches_h = H // self.kernel_size
        num_patches_w = W // self.kernel_size
        return B, num_patches_h * num_patches_w

    def forward(self, x):
        """Everything below is on-device: real ttnn ops so Polaris's cost model actually sees
        and prices the matmul (and bias add). The patch/im2col reshape itself is intentionally
        *not* expressed as a ttnn op -- see module docstring -- since it never runs on the
        simulated device in the reference either."""
        B, num_patches = self._patch_output_shape(x)

        patches = ttnn.Tensor(
            shape=(1, B, num_patches, self.padded_patch_dim),
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Bias applied outside ttnn.linear to avoid the FUSE_BIAS matmul kernel path -- same as
        # the reference, and it matters for Polaris because fused-bias vs separate-add can hit
        # different entries in the op-cost lookup.
        out = ttnn.linear(
            patches,
            self.weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        if self.bias is not None:
            out = ttnn.add(out, self.bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(patches)

        return out
