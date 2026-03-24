#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTSim implementation of PlaceHolderEncoder for MapTracker.

This module provides a placeholder encoder that simply passes through
the input query unchanged. Used in MapTracker configs as a no-op encoder.

Original: maptracker/plugin/models/transformer_utils/base_transformer.py
"""

# -------------------------------PyTorch--------------------------------

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# from mmcv.cnn import xavier_init, constant_init
# from mmcv.cnn.bricks.registry import (ATTENTION,
#                                       TRANSFORMER_LAYER_SEQUENCE)
# from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
#                                          TransformerLayerSequence,
#                                          build_transformer_layer_sequence)
# from mmcv.runner.base_module import BaseModule
#
# from mmdet.models.utils.builder import TRANSFORMER
#
# @TRANSFORMER_LAYER_SEQUENCE.register_module()
# class PlaceHolderEncoder(nn.Module):
#
#     def __init__(self, *args, embed_dims=None, **kwargs):
#         super(PlaceHolderEncoder, self).__init__()
#         self.embed_dims = embed_dims
#
#     def forward(self, *args, query=None, **kwargs):
#
#         return query

# -------------------------------TTSIM-----------------------------------


import sys
import os

# Add ttsim to path
current_dir = os.path.dirname(os.path.abspath(__file__))
polaris_root = os.path.abspath(os.path.join(current_dir, "..", "..", "..", "..", ".."))
if polaris_root not in sys.path:
    sys.path.insert(0, polaris_root)

import ttsim.front.functional.sim_nn as SimNN


class PlaceHolderEncoder(SimNN.Module):
    """
    Placeholder encoder that returns input unchanged.

    This is a no-op module used as a placeholder in transformer architectures
    where an encoder component is required but no actual encoding is needed.

    Args:
        embed_dims: Embedding dimensions (stored but not used)
        *args: Additional positional arguments (ignored)
        **kwargs: Additional keyword arguments (ignored)
    """

    def __init__(self, *args, embed_dims=None, **kwargs):
        super().__init__()
        self.name = "placeholder_encoder"
        self.embed_dims = embed_dims

    def __call__(self, *args, query=None, **kwargs):
        """
        Forward pass - simply returns the query unchanged.

        Args:
            query: Input query tensor
            *args: Additional positional arguments (ignored)
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            query: The input query tensor unchanged
        """
        return query

    def analytical_param_count(self, lvl=0):
        """
        Calculate parameter count for this module.

        Args:
            lvl (int): Verbosity level (0=silent, 1=summary, 2=detailed)

        Returns:
            int: Total parameter count (always 0 for placeholder)
        """
        if lvl >= 1:
            indent = "  " * lvl
            print(f"{indent}PlaceHolderEncoder '{self.name}': 0 params")
        return 0
