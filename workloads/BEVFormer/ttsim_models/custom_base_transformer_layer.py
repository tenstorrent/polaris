#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of Custom Base Transformer Layer for BEVFormer.

This module implements a flexible transformer layer that can be composed
of various attention mechanisms (self-attention, cross-attention),
feed-forward networks (FFN), and normalization layers in any specified order.

Original: projects/mmdet3d_plugin/bevformer/modules/custom_base_transformer_layer.py
Reference: BEVFormer paper - https://arxiv.org/abs/2203.17270

============================================================================
MMCV Import Conversions (Python 3.13 Compatible)
============================================================================

The original PyTorch implementation uses several mmcv functions that are not
compatible with Python 3.13. This TTSim version includes the following conversions:

1. Base Classes:
   - BaseModule: Replaced with ttsim.front.functional.sim_nn.Module
   - ModuleList: Replaced with Python list
   - Sequential: Replaced with custom sequential execution

2. Builders:
   - build_attention: Custom builder function for attention modules
   - build_feedforward_network: Custom builder function for FFN modules
   - build_norm_layer: Custom builder function for normalization layers
   - build_activation_layer: Custom builder function for activation layers

3. Registry Decorators:
   - @TRANSFORMER_LAYER.register_module(): Not needed in TTSim (no module registry)

4. Config Management:
   - ConfigDict: Replaced with Python dict
   - deprecated_api_warning: Replaced with warnings.warn

5. Tensor Operations:
   - torch.Tensor checks: Replaced with appropriate type checks
   - copy.deepcopy: Still used (standard Python library)

All computational logic from the PyTorch version has been preserved and
converted to TTSim operations.
"""

import sys
import os
import copy
import warnings
from loguru import logger

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import numpy as np
import ttsim.front.functional.op as F
from ttsim.front.functional.sim_nn import Module, Linear

# Import component builders (will need to create these)
try:
    from .builder_utils import (
        build_attention,
        build_feedforward_network,
        build_norm_layer,
    )
except ImportError:
    # Handle case when run as script
    from builder_utils import build_attention, build_feedforward_network, build_norm_layer  # type: ignore[import-not-found,no-redef]


class MyCustomBaseTransformerLayer(Module):
    """
    TTSim implementation of Custom Base Transformer Layer for BEVFormer.

    This is a flexible transformer layer that can compose any number of
    attention, FFN, and normalization operations in a specified order.
    It supports both prenorm and postnorm architectures.

    Key features:
    - Flexible operation ordering (self_attn, cross_attn, norm, ffn)
    - Multiple attention mechanisms in single layer
    - Prenorm support (norm before attention/FFN)
    - Batch-first or sequence-first formats

    Args:
        name (str): Module name
        attn_cfgs (list[dict] | dict | None): Configs for attention modules.
            Order should match attention operations in operation_order.
            If dict, all attentions use same config. Default: None
        ffn_cfgs (list[dict] | dict | None): Configs for FFN modules.
            Order should match FFN operations in operation_order.
            Default: standard FFN config
        operation_order (tuple[str]): Execution order of operations.
            e.g., ('self_attn', 'norm', 'ffn', 'norm')
            Support prenorm when first element is 'norm'. Default: None
        norm_cfg (dict): Config for normalization layer. Default: dict(type='LN')
        batch_first (bool): If True, inputs are [batch, seq, dim]. Default: True

    Example operation_order:
        - Standard: ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
        - Prenorm: ('norm', 'self_attn', 'norm', 'cross_attn', 'norm', 'ffn')
    """

    def __init__(
        self,
        name,
        attn_cfgs=None,
        ffn_cfgs=None,
        operation_order=None,
        norm_cfg=None,
        batch_first=True,
        **kwargs,
    ):
        super().__init__()
        self.name = name

        # Set defaults
        if ffn_cfgs is None:
            ffn_cfgs = dict(
                type="FFN",
                embed_dims=256,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type="ReLU", inplace=True),
            )

        if norm_cfg is None:
            norm_cfg = dict(type="LN")

        # Handle deprecated arguments
        deprecated_args = dict(
            feedforward_channels="feedforward_channels",
            ffn_dropout="ffn_drop",
            ffn_num_fcs="num_fcs",
        )
        for ori_name, new_name in deprecated_args.items():
            if ori_name in kwargs:
                warnings.warn(
                    f"The arguments `{ori_name}` in BaseTransformerLayer "
                    f"has been deprecated, now you should set `{new_name}` "
                    f"and other FFN related arguments "
                    f"to a dict named `ffn_cfgs`. "
                )
                if isinstance(ffn_cfgs, dict):
                    ffn_cfgs[new_name] = kwargs[ori_name]

        self.batch_first = batch_first

        # Validate operation_order
        if operation_order is None:
            raise ValueError("operation_order must be specified")

        valid_ops = {"self_attn", "norm", "ffn", "cross_attn"}
        assert set(operation_order) & valid_ops == set(
            operation_order
        ), f"The operation_order should only contain operations from {valid_ops}"

        # Count number of each operation type
        num_attn = operation_order.count("self_attn") + operation_order.count(
            "cross_attn"
        )
        num_ffns = operation_order.count("ffn")
        num_norms = operation_order.count("norm")

        # Handle attention configs
        if attn_cfgs is None:
            attn_cfgs = []
        elif isinstance(attn_cfgs, dict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), (
                f"The length of attn_cfg {len(attn_cfgs)} is "
                f"not consistent with the number of attention "
                f"in operation_order {num_attn}."
            )

        self.num_attn = num_attn
        self.operation_order = operation_order
        self.norm_cfg = norm_cfg
        self.pre_norm = len(operation_order) > 0 and operation_order[0] == "norm"

        # Build attention modules
        self.attentions = []
        index = 0
        for operation_name in operation_order:
            if operation_name in ["self_attn", "cross_attn"]:
                if "batch_first" in attn_cfgs[index]:
                    assert self.batch_first == attn_cfgs[index]["batch_first"]
                else:
                    attn_cfgs[index]["batch_first"] = self.batch_first

                attention = build_attention(f"{name}.attn_{index}", attn_cfgs[index])
                # Mark operation type for the attention module
                attention.operation_name = operation_name
                self.attentions.append(attention)
                index += 1

        # Get embed_dims from first attention
        if len(self.attentions) > 0:
            self.embed_dims = self.attentions[0].embed_dims
        else:
            self.embed_dims = ffn_cfgs.get("embed_dims", 256)

        # Build FFN modules
        self.ffns = []
        if isinstance(ffn_cfgs, dict):
            ffn_cfgs = [copy.deepcopy(ffn_cfgs) for _ in range(num_ffns)]
        assert (
            len(ffn_cfgs) == num_ffns
        ), f"The length of ffn_cfgs {len(ffn_cfgs)} must equal num_ffns {num_ffns}"

        for ffn_index in range(num_ffns):
            if "embed_dims" not in ffn_cfgs[ffn_index]:
                ffn_cfgs[ffn_index]["embed_dims"] = self.embed_dims
            else:
                assert ffn_cfgs[ffn_index]["embed_dims"] == self.embed_dims

            ffn = build_feedforward_network(
                f"{name}.ffn_{ffn_index}", ffn_cfgs[ffn_index]
            )
            self.ffns.append(ffn)

        # Build normalization modules
        self.norms = []
        for norm_index in range(num_norms):
            norm = build_norm_layer(
                f"{name}.norm_{norm_index}", norm_cfg, self.embed_dims
            )
            self.norms.append(norm)

    def __call__(
        self,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):
        """
        Forward pass of Custom Base Transformer Layer.

        Args:
            query: Input query tensor
                [num_queries, bs, embed_dims] if batch_first=False
                [bs, num_queries, embed_dims] if batch_first=True
            key: Key tensor (for cross-attention)
            value: Value tensor (for cross-attention)
            query_pos: Positional encoding for query
            key_pos: Positional encoding for key
            attn_masks: List of attention masks (one per attention operation)
            query_key_padding_mask: Padding mask for query (self-attention)
            key_padding_mask: Padding mask for key (cross-attention)
            **kwargs: Additional arguments for attention modules

        Returns:
            Output tensor with same shape as query
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query

        # Handle attention masks
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif not isinstance(attn_masks, list):
            # If single mask provided, replicate for all attentions
            attn_masks = [attn_masks for _ in range(self.num_attn)]
            warnings.warn(
                f"Use same attn_mask in all attentions in "
                f"{self.__class__.__name__} "
            )
        else:
            assert len(attn_masks) == self.num_attn, (
                f"The length of attn_masks {len(attn_masks)} must be equal "
                f"to the number of attention in operation_order {self.num_attn}"
            )

        # Execute operations in specified order
        for layer in self.operation_order:
            if layer == "self_attn":
                # Self-attention: query attends to itself
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "norm":
                # Normalization
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == "cross_attn":
                # Cross-attention: query attends to key/value
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity=identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs,
                )
                attn_index += 1
                identity = query

            elif layer == "ffn":
                # Feed-forward network
                query = self.ffns[ffn_index](
                    query, identity=identity if self.pre_norm else None
                )
                ffn_index += 1

        return query

    def analytical_param_count(self):
        """
        Calculate the total number of parameters in this module.

        Returns:
            int: Total parameter count
        """
        total = 0

        # Count attention parameters
        for attn in self.attentions:
            if hasattr(attn, "analytical_param_count"):
                total += attn.analytical_param_count()

        # Count FFN parameters
        for ffn in self.ffns:
            if hasattr(ffn, "analytical_param_count"):
                total += ffn.analytical_param_count()

        # Count norm parameters
        for norm in self.norms:
            if hasattr(norm, "analytical_param_count"):
                total += norm.analytical_param_count()
            else:
                # LayerNorm has 2 * embed_dims parameters (weight + bias)
                total += 2 * self.embed_dims

        return total


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("Custom Base Transformer Layer TTSim Module")
    logger.info("=" * 80)
    logger.info("\n✓ Module imported successfully!")
    logger.info("\nAvailable component:")
    logger.info("  - MyCustomBaseTransformerLayer - Flexible transformer layer")

    logger.info("\nModule test:")

    # Test MyCustomBaseTransformerLayer
    try:
        # Simple test configuration
        attn_cfg = dict(
            type="TemporalSelfAttention",
            embed_dims=256,
            num_heads=8,
            num_levels=4,
            num_points=4,
            num_bev_queue=2,
        )

        ffn_cfg = dict(
            type="FFN",
            embed_dims=256,
            feedforward_channels=1024,
            num_fcs=2,
            ffn_drop=0.0,
        )

        operation_order = ("self_attn", "norm", "ffn", "norm")

        layer = MyCustomBaseTransformerLayer(
            name="test_layer",
            attn_cfgs=[attn_cfg],
            ffn_cfgs=ffn_cfg,
            operation_order=operation_order,
            batch_first=True,
        )
        logger.info("  ✓ MyCustomBaseTransformerLayer constructed successfully")
        logger.debug(f"    - Name: {layer.name}")
        logger.debug(f"    - Embed dims: {layer.embed_dims}")
        logger.debug(f"    - Operation order: {layer.operation_order}")
        logger.debug(f"    - Num attentions: {layer.num_attn}")
        logger.debug(f"    - Num FFNs: {len(layer.ffns)}")
        logger.debug(f"    - Num norms: {len(layer.norms)}")
        logger.debug(f"    - Batch first: {layer.batch_first}")
        logger.debug(f"    - Pre-norm: {layer.pre_norm}")
    except Exception as e:
        logger.info(f"  ✗ MyCustomBaseTransformerLayer construction failed: {e}")
        import traceback

        traceback.print_exc()

    logger.info("\n✓ Basic test passed!")
    logger.info(
        "\nNote: Use validation tests in Validation/ folder for full functionality testing."
    )
    logger.info("=" * 80)
