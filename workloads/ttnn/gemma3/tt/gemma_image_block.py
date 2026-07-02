"""
This is the ImageTransformer block for Gemma-3-4b-it.
We have reused the TtLlamaImageTransformerBlock with incorporating the
TtGemmaImageAttention and TtGemmaImageFeedForward
"""
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
import ttsim.front.ttnn as ttnn
from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule
from workloads.ttnn.gemma3.tt.gemma_image_attention import TtGemmaImageAttention
from workloads.ttnn.gemma3.tt.gemma_image_mlp import TtGemmaImageFeedForward
from workloads.ttnn.tt_transformers.llama_layernorm import TtLayerNorm


class TtGemmaImageTransformerBlock(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.vision_dim

        self.ln_1 = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_1.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.attn = TtGemmaImageAttention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}attn.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
            configuration=configuration,
        )

        self.ln_2 = TtLayerNorm(
            device=mesh_device,
            dim=configuration.vision_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_2.",
            weight_cache_path=weight_cache_path,
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

        self.mlp = TtGemmaImageFeedForward(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}mlp.",
            weight_cache_path=weight_cache_path,
            dtype=dtype,
        )

    def _get_shape_tuple(self, tensor):
        """Get shape as tuple from various tensor types."""
        if hasattr(tensor, 'shape'):
            shape = tensor.shape
            if hasattr(shape, '__iter__'):
                return tuple(shape)
            return shape
        if hasattr(tensor, 'get_shape'):
            return tuple(tensor.get_shape())
        return ()

    def forward(self, x_11SH, mask=None):
        # Get shape safely
        x_shape = self._get_shape_tuple(x_11SH)

        # Handle 4D tensor: [1, B, seq_len, hidden_dim] or [B, 1, seq_len, hidden_dim]
        if len(x_shape) == 4:
            if x_shape[0] == 1 and x_shape[1] > 1:
                # Shape is [1, B, seq_len, hidden_dim]
                batch_size = x_shape[1]
                seq_len = x_shape[2]
                hidden_dim = x_shape[3]
            elif x_shape[1] == 1:
                # Shape is [B, 1, seq_len, hidden_dim]
                batch_size = x_shape[0]
                seq_len = x_shape[2]
                hidden_dim = x_shape[3]
            else:
                batch_size = x_shape[0]
                seq_len = x_shape[2]
                hidden_dim = x_shape[3]
        elif len(x_shape) == 3:
            batch_size = x_shape[0]
            seq_len = x_shape[1]
            hidden_dim = x_shape[2]
        else:
            batch_size = 1
            seq_len = x_shape[-2] if len(x_shape) >= 2 else 1
            hidden_dim = x_shape[-1] if len(x_shape) >= 1 else self.hidden_size

        assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"

        # Run attention
        attn_out = self.attn(self.ln_1(x_11SH), mask=mask)

        # Get attn_out shape
        attn_shape = self._get_shape_tuple(attn_out)

        # Reshape both tensors to same shape for addition
        if len(attn_shape) == 3:
            # attn_out is 3D [B, seq, hidden], reshape x_11SH to 3D
            x_11SH = ttnn.reshape(x_11SH, (batch_size, seq_len, hidden_dim))
        elif len(attn_shape) == 4:
            # attn_out is 4D, reshape x_11SH to match
            x_11SH = ttnn.reshape(x_11SH, (batch_size, 1, seq_len, hidden_dim))
            # Also ensure attn_out has same shape
            if attn_shape != (batch_size, 1, seq_len, hidden_dim):
                attn_out = ttnn.reshape(attn_out, (batch_size, 1, seq_len, hidden_dim))
        else:
            # Default: reshape both to 4D
            x_11SH = ttnn.reshape(x_11SH, (batch_size, 1, seq_len, hidden_dim))
            attn_out = ttnn.reshape(attn_out, (batch_size, 1, seq_len, hidden_dim))

        # Residual connection
        res = ttnn.add(x_11SH, attn_out)

        # MLP
        mlp_out = self.mlp(self.ln_2(res))

        # Get shapes for final add
        mlp_shape = self._get_shape_tuple(mlp_out)
        res_shape = self._get_shape_tuple(res)

        # Align shapes if needed
        if mlp_shape != res_shape and len(mlp_shape) > 0 and len(res_shape) > 0:
            if len(mlp_shape) == 4 and len(res_shape) == 3:
                res = ttnn.reshape(res, mlp_shape)
            elif len(mlp_shape) == 3 and len(res_shape) == 4:
                mlp_out = ttnn.reshape(mlp_out, res_shape)

        # Final residual connection
        out = ttnn.add(res, mlp_out)

        # Cleanup
        ttnn.deallocate(mlp_out)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(res)

        return out