#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Polaris-compatible CLIP Text Encoders for SDXL.

This module provides CLIP text encoding operations that integrate
with the Polaris TTSIM workload system for SDXL text-to-image generation.
"""

import os
import sys
import math
from pathlib import Path
import numpy as np
from typing import Any, Dict, List, Optional

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import Polaris TTSIM components
import ttsim.front.functional.op as F
import ttsim.front.functional.tensor_op as T
import ttsim.front.functional.sim_nn as SimNN
from ttsim.ops import SimTensor
from ttsim.graph.wl_graph import WorkloadGraph


class CLIPTextModelPolaris(SimNN.Module):
    """
    Polaris-compatible CLIP Text Model (Text Encoder 1).

    This class implements CLIP text encoding operations using
    TTSIM-compatible operations for SDXL text-to-image generation.
    """

    def __init__(self, name: str, cfg: Dict[str, Any]):
        """
        Initialize the CLIP text model.

        Args:
            name: Component name identifier
            cfg: Configuration dictionary
        """
        super().__init__()

        self.name = name
        self.cfg = cfg

        # Extract text encoder configuration
        self.vocab_size = cfg.get('vocab_size', 49408)
        self.max_seq_length = cfg.get('max_seq_length', 77)
        self.hidden_size = cfg.get('hidden_size', 768)
        self.intermediate_size = cfg.get('intermediate_size', 3072)
        self.num_attention_heads = cfg.get('num_attention_heads', 12)
        self.num_hidden_layers = cfg.get('num_hidden_layers', 6)
        self.max_position_embeddings = cfg.get('max_position_embeddings', 77)

        # Build text encoder operations
        self._build_text_encoder()

    def _build_text_encoder(self):
        """Build text encoder operations."""
        # Token embedding
        self.token_embedding = F.Embedding(f"{self.name}_token_embedding",
                                          self.vocab_size, self.hidden_size)
        self.token_embedding.set_module(self)

        # Position embedding
        self.position_embedding = F.Embedding(f"{self.name}_position_embedding",
                                             self.max_position_embeddings, self.hidden_size)
        self.position_embedding.set_module(self)

        # Transformer layers
        self.transformer_layers = []
        for layer_idx in range(self.num_hidden_layers):
            layer = {
                'attention': self._build_attention_layer(f"{self.name}_layer_{layer_idx}"),
                'mlp': self._build_mlp_layer(f"{self.name}_layer_{layer_idx}"),
                'ln_1': F.LayerNorm(f"{self.name}_layer_{layer_idx}_ln_1", self.hidden_size),
                'ln_2': F.LayerNorm(f"{self.name}_layer_{layer_idx}_ln_2", self.hidden_size),
            }
            layer['ln_1'].set_module(self)
            layer['ln_2'].set_module(self)
            self.transformer_layers.append(layer)

        # Final layer norm
        self.final_ln = F.LayerNorm(f"{self.name}_final_ln", self.hidden_size)
        self.final_ln.set_module(self)

    def _build_attention_layer(self, layer_name: str) -> Dict[str, Any]:
        """Build attention layer operations."""
        attn = {
            'query': F.Linear(f"{layer_name}_attn_q", self.hidden_size, self.hidden_size),
            'key': F.Linear(f"{layer_name}_attn_k", self.hidden_size, self.hidden_size),
            'value': F.Linear(f"{layer_name}_attn_v", self.hidden_size, self.hidden_size),
            'out': F.Linear(f"{layer_name}_attn_out", self.hidden_size, self.hidden_size),
            'dropout': F.Dropout(f"{layer_name}_attn_dropout", 0.1),
        }
        attn['query'].set_module(self)
        attn['key'].set_module(self)
        attn['value'].set_module(self)
        attn['out'].set_module(self)
        attn['dropout'].set_module(self)
        return attn

    def _build_mlp_layer(self, layer_name: str) -> Dict[str, Any]:
        """Build MLP layer operations."""
        mlp = {
            'fc1': F.Linear(f"{layer_name}_mlp_fc1", self.hidden_size, self.intermediate_size),
            'fc2': F.Linear(f"{layer_name}_mlp_fc2", self.intermediate_size, self.hidden_size),
            'dropout': F.Dropout(f"{layer_name}_mlp_dropout", 0.1),
        }
        mlp['fc1'].set_module(self)
        mlp['fc2'].set_module(self)
        mlp['dropout'].set_module(self)
        return mlp

    def add_call(self, workload_graph: WorkloadGraph, input_ids: SimTensor,
                 attention_mask: Optional[SimTensor] = None) -> SimTensor:
        """
        Add text encoding operations to the workload graph.

        Args:
            workload_graph: TTSIM workload graph
            input_ids: Token input IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Text embeddings [batch, seq_len, hidden_size]
        """
        # Ensure inputs are linked to this module for downstream ops
        if hasattr(input_ids, 'set_module'):
            input_ids.set_module(self)
        if attention_mask is not None and hasattr(attention_mask, 'set_module'):
            attention_mask.set_module(self)

        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Position embeddings
        seq_len = input_ids.shape[1]
        pos_ids_data = np.arange(seq_len, dtype=np.int64)[None, :]
        position_ids = F._from_data(f"{self.name}_position_ids", data=pos_ids_data, is_const=True)
        position_ids.set_module(self)
        position_embeds = self.position_embedding(position_ids)

        # Combine embeddings
        embeddings = token_embeds + position_embeds

        # Transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            # Self-attention
            attn_output = self._apply_attention(hidden_states, layer['attention'], attention_mask)
            hidden_states = hidden_states + attn_output  # Residual connection
            hidden_states = layer['ln_1'](hidden_states)

            # MLP
            mlp_output = self._apply_mlp(hidden_states, layer['mlp'])
            hidden_states = hidden_states + mlp_output  # Residual connection
            hidden_states = layer['ln_2'](hidden_states)

        # Final layer norm
        output = self.final_ln(hidden_states)

        return output

    def _apply_attention(self, hidden_states: SimTensor, attn_layer: Dict[str, Any],
                        attention_mask: Optional[SimTensor] = None) -> SimTensor:
        """Apply attention mechanism."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Linear projections
        query = attn_layer['query'](hidden_states)
        key = attn_layer['key'](hidden_states)
        value = attn_layer['value'](hidden_states)

        # Reshape for attention
        head_dim = hidden_size // self.num_attention_heads
        query = query.reshape(batch_size, seq_len, self.num_attention_heads, head_dim)
        key = key.reshape(batch_size, seq_len, self.num_attention_heads, head_dim)
        value = value.reshape(batch_size, seq_len, self.num_attention_heads, head_dim)

        # Transpose for attention computation
        query = query.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Attention computation
        scale = F._from_data(f"{self.name}_attention_scale", data=np.array(1.0 / math.sqrt(head_dim), dtype=np.float32), is_const=True)
        scale.set_module(self)
        attn_weights = T.matmul(query, key.transpose(2, 3)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Compute (1.0 - expanded_mask) * -10000.0 using functional ops
            one_const = F._from_data(f"{self.name}_one", data=np.array(1.0, dtype=np.float32), is_const=True)
            one_const.set_module(self)
            neg_large_const = F._from_data(f"{self.name}_neg_large", data=np.array(-10000.0, dtype=np.float32), is_const=True)
            neg_large_const.set_module(self)
            sub_op = F.Sub(f"{self.name}_mask_inv")
            sub_op.set_module(self)
            inv_mask = sub_op(one_const, expanded_mask)
            mul_op = F.Mul(f"{self.name}_mask_bias_scale")
            mul_op.set_module(self)
            mask_bias = mul_op(inv_mask, neg_large_const)
            add_op = F.Add(f"{self.name}_mask_add")
            add_op.set_module(self)
            attn_weights = add_op(attn_weights, mask_bias)

        # Softmax
        attn_weights = F.Softmax('attention_softmax')(attn_weights)

        # Apply dropout
        attn_weights = attn_layer['dropout'](attn_weights)

        # Compute attention output
        attn_output = T.matmul(attn_weights, value)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

        # Final linear projection
        output = attn_layer['out'](attn_output)

        return output

    def _apply_mlp(self, hidden_states: SimTensor, mlp_layer: Dict[str, Any]) -> SimTensor:
        """Apply MLP operations."""
        # First linear layer
        hidden_states = mlp_layer['fc1'](hidden_states)
        hidden_states = F.Gelu('mlp_gelu')(hidden_states)

        # Dropout
        hidden_states = mlp_layer['dropout'](hidden_states)

        # Second linear layer
        hidden_states = mlp_layer['fc2'](hidden_states)

        return hidden_states

    def analytical_param_count(self, lvl: int = 0) -> int:
        """Estimate parameter count for analytical purposes."""
        total_params = 0

        # Token and position embeddings
        total_params += self.vocab_size * self.hidden_size  # Token embedding
        total_params += self.max_position_embeddings * self.hidden_size  # Position embedding

        # Transformer layers
        for _ in range(self.num_hidden_layers):
            # Attention parameters
            total_params += 3 * self.hidden_size * self.hidden_size  # Q, K, V projections
            total_params += self.hidden_size * self.hidden_size  # Output projection

            # MLP parameters
            total_params += self.hidden_size * self.intermediate_size  # FC1
            total_params += self.intermediate_size * self.hidden_size  # FC2

        return total_params


class CLIPTextModelWithProjectionPolaris(SimNN.Module):
    """
    Polaris-compatible CLIP Text Model with Projection (Text Encoder 2).

    This class implements CLIP text encoding with projection operations using
    TTSIM-compatible operations for SDXL text-to-image generation.
    """

    def __init__(self, name: str, cfg: Dict[str, Any]):
        """
        Initialize the CLIP text model with projection.

        Args:
            name: Component name identifier
            cfg: Configuration dictionary
        """
        super().__init__()

        self.name = name
        self.cfg = cfg

        # Extract text encoder configuration
        self.vocab_size = cfg.get('vocab_size', 49408)
        self.max_seq_length = cfg.get('max_seq_length', 77)
        self.hidden_size_2 = cfg.get('hidden_size_2', 1280)
        self.intermediate_size_2 = cfg.get('intermediate_size_2', 5120)
        self.num_attention_heads_2 = cfg.get('num_attention_heads_2', 20)
        self.projection_dim = cfg.get('projection_dim', 1280)
        self.num_hidden_layers = cfg.get('num_hidden_layers', 6)
        self.max_position_embeddings = cfg.get('max_position_embeddings', 77)

        # Use the larger hidden size for encoder 2 to match attention heads
        self.hidden_size = cfg.get('hidden_size_2', 1280)

        # Build text encoder with projection operations
        self._build_text_encoder_with_projection()

    def _build_text_encoder_with_projection(self):
        """Build text encoder with projection operations."""
        # Base text encoder (similar to CLIPTextModelPolaris)
        self.token_embedding = F.Embedding(f"{self.name}_token_embedding",
                                          self.vocab_size, self.hidden_size)
        self.token_embedding.set_module(self)

        self.position_embedding = F.Embedding(f"{self.name}_position_embedding",
                                             self.max_position_embeddings, self.hidden_size)
        self.position_embedding.set_module(self)

        # Transformer layers (with larger hidden size)
        self.transformer_layers = []
        for layer_idx in range(self.num_hidden_layers):
            layer = {
                'attention': self._build_attention_layer(f"{self.name}_layer_{layer_idx}"),
                'mlp': self._build_mlp_layer(f"{self.name}_layer_{layer_idx}"),
                'ln_1': F.LayerNorm(f"{self.name}_layer_{layer_idx}_ln_1", self.hidden_size),
                'ln_2': F.LayerNorm(f"{self.name}_layer_{layer_idx}_ln_2", self.hidden_size),
            }
            layer['ln_1'].set_module(self)
            layer['ln_2'].set_module(self)
            self.transformer_layers.append(layer)

        # Final layer norm
        self.final_ln = F.LayerNorm(f"{self.name}_final_ln", self.hidden_size)
        self.final_ln.set_module(self)

        # Projection layer (to match projection_dim)
        self.text_projection = F.Linear(f"{self.name}_text_projection",
                                      self.hidden_size, self.projection_dim)
        self.text_projection.set_module(self)

    def _build_attention_layer(self, layer_name: str) -> Dict[str, Any]:
        """Build attention layer operations."""
        attn = {
            'query': F.Linear(f"{layer_name}_attn_q", self.hidden_size, self.hidden_size),
            'key': F.Linear(f"{layer_name}_attn_k", self.hidden_size, self.hidden_size),
            'value': F.Linear(f"{layer_name}_attn_v", self.hidden_size, self.hidden_size),
            'out': F.Linear(f"{layer_name}_attn_out", self.hidden_size, self.hidden_size),
            'dropout': F.Dropout(f"{layer_name}_attn_dropout", 0.1),
        }
        attn['query'].set_module(self)
        attn['key'].set_module(self)
        attn['value'].set_module(self)
        attn['out'].set_module(self)
        attn['dropout'].set_module(self)
        return attn

    def _build_mlp_layer(self, layer_name: str) -> Dict[str, Any]:
        """Build MLP layer operations."""
        mlp = {
            'fc1': F.Linear(f"{layer_name}_mlp_fc1", self.hidden_size, self.intermediate_size_2),
            'fc2': F.Linear(f"{layer_name}_mlp_fc2", self.intermediate_size_2, self.hidden_size),
            'dropout': F.Dropout(f"{layer_name}_mlp_dropout", 0.1),
        }
        mlp['fc1'].set_module(self)
        mlp['fc2'].set_module(self)
        mlp['dropout'].set_module(self)
        return mlp

    def add_call(self, workload_graph: WorkloadGraph, input_ids: SimTensor,
                 attention_mask: Optional[SimTensor] = None) -> Dict[str, SimTensor]:
        """
        Add text encoding operations to the workload graph.

        Args:
            workload_graph: TTSIM workload graph
            input_ids: Token input IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dictionary containing text embeddings and pooled output
        """
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)

        # Position embeddings
        seq_len = input_ids.shape[1]
        pos_ids_data = np.arange(seq_len, dtype=np.int64)[None, :]
        position_ids = F._from_data(f"{self.name}_position_ids", data=pos_ids_data, is_const=True)
        position_ids.set_module(self)
        position_embeds = self.position_embedding(position_ids)

        # Combine embeddings
        embeddings = token_embeds + position_embeds

        # Transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            # Self-attention
            attn_output = self._apply_attention(hidden_states, layer['attention'], attention_mask)
            hidden_states = hidden_states + attn_output  # Residual connection
            hidden_states = layer['ln_1'](hidden_states)

            # MLP
            mlp_output = self._apply_mlp(hidden_states, layer['mlp'])
            hidden_states = hidden_states + mlp_output  # Residual connection
            hidden_states = layer['ln_2'](hidden_states)

        # Final layer norm
        text_embeds = self.final_ln(hidden_states)

        # Text projection
        pooled_output = self.text_projection(text_embeds[:, 0, :])  # Use [CLS] token

        return {
            'text_embeddings': text_embeds,
            'pooled_output': pooled_output
        }

    def _apply_attention(self, hidden_states: SimTensor, attn_layer: Dict[str, Any],
                        attention_mask: Optional[SimTensor] = None) -> SimTensor:
        """Apply attention mechanism."""
        batch_size, seq_len, hidden_size = hidden_states.shape

        # Linear projections
        query = attn_layer['query'](hidden_states)
        key = attn_layer['key'](hidden_states)
        value = attn_layer['value'](hidden_states)

        # Reshape for attention
        head_dim = hidden_size // self.num_attention_heads_2
        query = query.reshape(batch_size, seq_len, self.num_attention_heads_2, head_dim)
        key = key.reshape(batch_size, seq_len, self.num_attention_heads_2, head_dim)
        value = value.reshape(batch_size, seq_len, self.num_attention_heads_2, head_dim)

        # Transpose for attention computation
        query = query.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Attention computation
        scale = F._from_data(f"{self.name}_attention_scale", data=np.array(1.0 / math.sqrt(head_dim), dtype=np.float32), is_const=True)
        scale.set_module(self)
        attn_weights = T.matmul(query, key.transpose(2, 3)) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask for multi-head attention
            expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Compute (1.0 - expanded_mask) * -10000.0 using functional ops
            one_const = F._from_data(f"{self.name}_one", data=np.array(1.0, dtype=np.float32), is_const=True)
            one_const.set_module(self)
            neg_large_const = F._from_data(f"{self.name}_neg_large", data=np.array(-10000.0, dtype=np.float32), is_const=True)
            neg_large_const.set_module(self)
            sub_op = F.Sub(f"{self.name}_mask_inv")
            sub_op.set_module(self)
            inv_mask = sub_op(one_const, expanded_mask)
            mul_op = F.Mul(f"{self.name}_mask_bias_scale")
            mul_op.set_module(self)
            mask_bias = mul_op(inv_mask, neg_large_const)
            add_op = F.Add(f"{self.name}_mask_add")
            add_op.set_module(self)
            attn_weights = add_op(attn_weights, mask_bias)

        # Softmax
        attn_weights = F.Softmax('attention_softmax')(attn_weights)

        # Apply dropout
        attn_weights = attn_layer['dropout'](attn_weights)

        # Compute attention output
        attn_output = T.matmul(attn_weights, value)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

        # Final linear projection
        output = attn_layer['out'](attn_output)

        return output

    def _apply_mlp(self, hidden_states: SimTensor, mlp_layer: Dict[str, Any]) -> SimTensor:
        """Apply MLP operations."""
        # First linear layer
        hidden_states = mlp_layer['fc1'](hidden_states)
        hidden_states = F.Gelu('mlp_gelu')(hidden_states)

        # Dropout
        hidden_states = mlp_layer['dropout'](hidden_states)

        # Second linear layer
        hidden_states = mlp_layer['fc2'](hidden_states)

        return hidden_states

    def analytical_param_count(self, lvl: int = 0) -> int:
        """Estimate parameter count for analytical purposes."""
        total_params = 0

        # Token and position embeddings
        total_params += self.vocab_size * self.hidden_size  # Token embedding
        total_params += self.max_position_embeddings * self.hidden_size  # Position embedding

        # Transformer layers
        for _ in range(self.num_hidden_layers):
            # Attention parameters
            total_params += 3 * self.hidden_size * self.hidden_size  # Q, K, V projections
            total_params += self.hidden_size * self.hidden_size  # Output projection

            # MLP parameters
            total_params += self.hidden_size * self.intermediate_size_2  # FC1
            total_params += self.intermediate_size_2 * self.hidden_size  # FC2

        # Text projection
        total_params += self.hidden_size * self.projection_dim

        return total_params


class CLIPTokenizerHost:
    """
    Host-side CLIP tokenizer for Polaris workloads.

    This tokenizer runs on the host and provides tokenization
    for CLIP text models in Polaris SDXL workloads.
    """

    def __init__(self, vocab_size: int = 49408, max_length: int = 77):
        """
        Initialize the CLIP tokenizer.

        Args:
            vocab_size: Vocabulary size
            max_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.max_length = max_length

        # Simple vocabulary mapping (placeholder for actual CLIP tokenizer)
        self._vocab = {f"token_{i}": i for i in range(min(1000, vocab_size))}
        self._vocab.update({
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4,
        })

    def __call__(self, text: str, return_tensors: str = "np") -> Dict[str, Any]:
        """
        Tokenize text input.

        Args:
            text: Input text string
            return_tensors: Return format ("np" for numpy arrays)

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Simple tokenization (placeholder for actual CLIP tokenizer)
        tokens = text.lower().split()[:self.max_length - 2]  # Reserve space for [CLS] and [SEP]

        # Add special tokens
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        # Convert to token IDs
        input_ids = []
        for token in tokens:
            if token in self._vocab:
                input_ids.append(self._vocab[token])
            else:
                input_ids.append(self._vocab["[UNK]"])

        # Pad to max_length
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_length:
            input_ids.append(self._vocab["[PAD]"])
            attention_mask.append(0)

        # Convert to numpy arrays
        import numpy as np
        result = {
            'input_ids': np.array([input_ids], dtype=np.int64),
            'attention_mask': np.array([attention_mask], dtype=np.int64)
        }

        return result


# Logging setup
import logging
LOG = logging.getLogger(__name__)
INFO = LOG.info
DEBUG = LOG.debug
ERROR = LOG.error
WARNING = LOG.warning