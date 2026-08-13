# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
This is the Entire ImageTransformer for Gemma-3-4b-it.
We have adapted the TtGemmaImageTransformerBlock from TtLlamaImageTransformerBlock
with changes incorporating the GemmaImageAttention and GemmaImageFeedForward
"""

from workloads.ttnn.gemma3.common.gemma_Lightweightmodule import LightweightModule
from workloads.ttnn.gemma3.tt.gemma_image_block import TtGemmaImageTransformerBlock


class TtGemmaImageTransformer(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        # tt_ccl,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        layers,
        block_key="resblocks",
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.resblocks = [
            TtGemmaImageTransformerBlock(
                mesh_device=mesh_device,
                state_dict=state_dict,
                # tt_ccl=tt_ccl,
                state_dict_prefix=f"{state_dict_prefix}{block_key}.{i}.",
                weight_cache_path=weight_cache_path,
                dtype=dtype,
                configuration=configuration,
            )
            for i in range(layers)
        ]

    def forward(self, x, mask=None):
        seq_len = x.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        for r in self.resblocks:
            x = r(x, mask=mask)

        return x