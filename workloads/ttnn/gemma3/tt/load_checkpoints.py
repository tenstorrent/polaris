# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Load checkpoint utilities for converting between HuggingFace and Meta state dict formats.
Gemma3-specific checkpoint conversion (vision + text).

Pure NumPy implementation - NO PyTorch dependency.
"""
import re
import numpy as np
from typing import Any, Dict, List, Tuple, Optional

__all__ = [
    # HF to Meta conversions
    "split_hf_keys",
    "convert_hf_qkv_to_meta_format",
    "map_hf_to_meta_keys",
    "map_hf_to_meta_keys_vision_only",
    "convert_hf_to_meta",
    "standardize_hf_keys",
    # Vision HF to Meta conversions
    "map_vision_hf_to_meta_keys_split_to_submodels",
    "map_vision_hf_to_meta_keys",
    "convert_vision_hf_to_meta",
    # Meta to HF conversions (reverse)
    "convert_meta_qkv_to_hf_format",
    "map_meta_to_hf_keys",
    "map_meta_to_hf_keys_vision_only",
    "convert_meta_to_hf",
    # Vision Meta to HF conversions (reverse)
    "map_vision_meta_to_hf_keys",
    "convert_vision_meta_to_hf",
]


# ============================================================================
# Utility Functions (Pure NumPy - No PyTorch)
# ============================================================================
def _split_dim0(tensor: np.ndarray, chunk_size: int) -> Tuple[np.ndarray, ...]:
    """
    Split numpy array along dim 0 into chunks.
    
    Args:
        tensor: Input numpy array
        chunk_size: Size of each chunk
        
    Returns:
        Tuple of numpy arrays
    """
    num_chunks = tensor.shape[0] // chunk_size
    return tuple(
        tensor[i * chunk_size : (i + 1) * chunk_size] 
        for i in range(num_chunks)
    )


def _concat_dim0(tensors: List[np.ndarray]) -> np.ndarray:
    """
    Concatenate numpy arrays along dim 0.
    
    Args:
        tensors: List of numpy arrays
        
    Returns:
        Concatenated numpy array
    """
    return np.concatenate(tensors, axis=0)


def _reverse_permute(
    tensor: np.ndarray, 
    n_heads: int, 
    dim1: int, 
    dim2: int
) -> np.ndarray:
    """
    Reverse permute Q/K weights from HF format to Meta format.
    
    This operation transforms the interleaved HF format to the 
    contiguous Meta format for rotary embeddings.
    
    Args:
        tensor: Input weight tensor [dim1, dim2]
        n_heads: Number of attention heads
        dim1: First dimension (usually n_heads * head_dim)
        dim2: Second dimension (usually hidden_dim)
        
    Returns:
        Permuted numpy array
    """
    # Reshape: [dim1, dim2] -> [n_heads, 2, dim1/n_heads/2, dim2]
    # Swap axes 1 and 2: [n_heads, dim1/n_heads/2, 2, dim2]
    # Reshape back: [dim1, dim2]
    return (
        tensor
        .reshape(n_heads, 2, dim1 // n_heads // 2, dim2)
        .swapaxes(1, 2)
        .reshape(dim1, dim2)
    )


def _forward_permute(
    tensor: np.ndarray, 
    n_heads: int, 
    dim1: int, 
    dim2: int
) -> np.ndarray:
    """
    Forward permute Q/K weights from Meta format to HF format.
    Inverse of _reverse_permute.
    
    Args:
        tensor: Input weight tensor [dim1, dim2]
        n_heads: Number of attention heads
        dim1: First dimension (usually n_heads * head_dim)
        dim2: Second dimension (usually hidden_dim)
        
    Returns:
        Permuted numpy array
    """
    # Reshape: [dim1, dim2] -> [n_heads, dim1/n_heads/2, 2, dim2]
    # Swap axes 1 and 2: [n_heads, 2, dim1/n_heads/2, dim2]
    # Reshape back: [dim1, dim2]
    return (
        tensor
        .reshape(n_heads, dim1 // n_heads // 2, 2, dim2)
        .swapaxes(1, 2)
        .reshape(dim1, dim2)
    )


def _replace_suffixes(key: str, replacements: List[Tuple[str, str]]) -> str:
    """
    Replace suffixes in key based on replacement list.
    
    Args:
        key: Input key string
        replacements: List of (old, new) replacement pairs
        
    Returns:
        Key with replacements applied
    """
    for old, new in replacements:
        key = re.sub(rf"(^|\.){re.escape(old)}($|\.)", rf"\1{new}\2", key)
    return key


# ============================================================================
# HuggingFace to Meta Conversion Functions
# ============================================================================
def split_hf_keys(state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Split HuggingFace fused keys into separate keys.
    
    Handles fused self_attn.qkv_proj -> q_proj/k_proj/v_proj when present.
    Otherwise acts as a pass-through.
    
    Args:
        state_dict: HuggingFace state dictionary
        
    Returns:
        State dictionary with split QKV projections
    """
    converted_weights: Dict[str, np.ndarray] = {}
    
    for key, tensor in state_dict.items():
        if "self_attn.qkv_proj" in key:
            # Split fused QKV into separate projections
            q_key = key.replace("self_attn.qkv_proj", "self_attn.q_proj")
            k_key = key.replace("self_attn.qkv_proj", "self_attn.k_proj")
            v_key = key.replace("self_attn.qkv_proj", "self_attn.v_proj")
            
            chunk_size = tensor.shape[0] // 3
            q_tensor, k_tensor, v_tensor = _split_dim0(tensor, chunk_size)
            
            converted_weights[q_key] = q_tensor
            converted_weights[k_key] = k_tensor
            converted_weights[v_key] = v_tensor
        else:
            converted_weights[key] = tensor
    
    return converted_weights


def convert_hf_qkv_to_meta_format(
    loaded_weights: Dict[str, np.ndarray], 
    head_dim: int
) -> Dict[str, np.ndarray]:
    """
    Convert HuggingFace Q/K projection weights to Meta format.
    
    HuggingFace uses interleaved format for rotary embeddings,
    Meta uses contiguous format. This function converts between them.
    
    Args:
        loaded_weights: State dictionary with HF format weights
        head_dim: Dimension per attention head
        
    Returns:
        State dictionary with Meta format Q/K weights
    """
    converted_weights: Dict[str, np.ndarray] = {}
    
    for key, tensor in loaded_weights.items():
        if "q_proj.weight" in key or "k_proj.weight" in key:
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = _reverse_permute(
                tensor, n_heads, tensor.shape[0], tensor.shape[1]
            )
        else:
            converted_weights[key] = tensor
    
    return converted_weights


def map_hf_to_meta_keys(
    loaded_weights: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Map HuggingFace key names to Meta key names.
    
    Converts naming conventions:
    - model.layers.X.self_attn.q_proj -> layers.X.attention.wq
    - model.layers.X.mlp.gate_proj -> layers.X.feed_forward.w1
    - etc.
    
    Args:
        loaded_weights: State dictionary with HF key names
        
    Returns:
        State dictionary with Meta key names
    """
    hf_to_meta = {
        # Top-level mappings
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "output.weight",
        # Layer-level mappings (templates)
        "model.layers.{layer}.input_layernorm.weight": "layers.{layer}.attention_norm.weight",
        "model.layers.{layer}.post_attention_layernorm.weight": "layers.{layer}.ffn_norm.weight",
        # Attention mappings
        "model.layers.{layer}.self_attn.q_proj.weight": "layers.{layer}.attention.wq.weight",
        "model.layers.{layer}.self_attn.k_proj.weight": "layers.{layer}.attention.wk.weight",
        "model.layers.{layer}.self_attn.v_proj.weight": "layers.{layer}.attention.wv.weight",
        "model.layers.{layer}.self_attn.o_proj.weight": "layers.{layer}.attention.wo.weight",
        "model.layers.{layer}.self_attn.q_proj.bias": "layers.{layer}.attention.wq.bias",
        "model.layers.{layer}.self_attn.k_proj.bias": "layers.{layer}.attention.wk.bias",
        "model.layers.{layer}.self_attn.v_proj.bias": "layers.{layer}.attention.wv.bias",
        "model.layers.{layer}.self_attn.q_norm.weight": "layers.{layer}.attention.q_norm.weight",
        "model.layers.{layer}.self_attn.k_norm.weight": "layers.{layer}.attention.k_norm.weight",
        # MLP mappings
        "model.layers.{layer}.mlp.gate_proj.weight": "layers.{layer}.feed_forward.w1.weight",
        "model.layers.{layer}.mlp.up_proj.weight": "layers.{layer}.feed_forward.w3.weight",
        "model.layers.{layer}.mlp.down_proj.weight": "layers.{layer}.feed_forward.w2.weight",
        # Pre/post feedforward layernorm (Gemma3 specific)
        "model.layers.{layer}.pre_feedforward_layernorm.weight": "layers.{layer}.ffn_norm.weight",
        "model.layers.{layer}.post_feedforward_layernorm.weight": "layers.{layer}.ffn_norm_2.weight",
    }
    
    meta_state_dict: Dict[str, np.ndarray] = {}
    
    for key, tensor in loaded_weights.items():
        if key in hf_to_meta:
            meta_state_dict[hf_to_meta[key]] = tensor
        elif "model.layers." in key:
            # Extract layer number and apply template mapping
            parts = key.split(".")
            layer_num = parts[2]
            template_key = "model.layers.{layer}." + ".".join(parts[3:])
            
            if template_key in hf_to_meta:
                meta_state_dict[hf_to_meta[template_key].format(layer=layer_num)] = tensor
            else:
                # Pass through unmapped keys
                meta_state_dict[key] = tensor
        else:
            # Pass through unmapped keys
            meta_state_dict[key] = tensor
    
    return meta_state_dict


def map_hf_to_meta_keys_vision_only(
    state_dict: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Map HuggingFace vision-model key names to Meta-style names.
    
    Args:
        state_dict: Vision model state dictionary with HF keys
        
    Returns:
        State dictionary with Meta-style keys
    """
    replacements = [
        ("self_attn", "attn"),
        ("q_proj", "wq"),
        ("k_proj", "wk"),
        ("v_proj", "wv"),
        ("o_proj", "wo"),
        ("out_proj", "wo"),
        ("layer_norm1", "ln_1"),
        ("layer_norm2", "ln_2"),
        ("post_layernorm", "ln_post"),
        ("patch_conv", "patch_conv._linear"),
    ]
    
    output: Dict[str, np.ndarray] = {}
    for k, v in state_dict.items():
        output[_replace_suffixes(k, replacements)] = v
    
    return output


def convert_hf_to_meta(
    state_dict: Dict[str, np.ndarray], 
    head_dim: int
) -> Dict[str, np.ndarray]:
    """
    Convert text-only HF state dict to Meta format.
    
    Pipeline:
    1. Split fused QKV keys
    2. Convert Q/K weight format
    3. Map key names
    
    Args:
        state_dict: HuggingFace state dictionary
        head_dim: Dimension per attention head
        
    Returns:
        Meta format state dictionary
    """
    state_dict = split_hf_keys(state_dict)
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_hf_to_meta_keys(state_dict)
    return state_dict


def standardize_hf_keys(
    state_dict: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Normalize HF checkpoint keys to the expected conversion format.
    
    Operations:
    - Collapses model.model.* -> model.*
    - Collapses model.language_model.* / language_model.* -> model.*
    - Backfills lm_head.weight from model.embed_tokens.weight when absent
    
    Args:
        state_dict: Raw HuggingFace state dictionary
        
    Returns:
        Standardized state dictionary
    """
    new_state_dict: Dict[str, np.ndarray] = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Collapse nested model prefixes
        if new_key.startswith("model.model."):
            new_key = "model." + new_key[len("model.model."):]
        
        if new_key.startswith("model.language_model."):
            new_key = "model." + new_key[len("model.language_model."):]
        elif new_key.startswith("language_model."):
            new_key = "model." + new_key[len("language_model."):]
        
        new_state_dict[new_key] = value
    
    # Backfill lm_head.weight if missing (tied embeddings)
    if "lm_head.weight" not in new_state_dict and "model.embed_tokens.weight" in new_state_dict:
        new_state_dict["lm_head.weight"] = new_state_dict["model.embed_tokens.weight"]
    
    return new_state_dict


# ============================================================================
# Vision HuggingFace to Meta Conversion Functions
# ============================================================================
def map_vision_hf_to_meta_keys_split_to_submodels(
    state_dict: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split state dict into vision, text, and other components.
    
    Args:
        state_dict: Combined multimodal state dictionary
        
    Returns:
        Tuple of (vision_state_dict, text_state_dict, other_state_dict)
    """
    vision_state_dict: Dict[str, np.ndarray] = {}
    text_state_dict: Dict[str, np.ndarray] = {}
    other_state_dict: Dict[str, np.ndarray] = {}
    
    for k, v in state_dict.items():
        if k.startswith("model.vision_tower"):
            vision_state_dict[k] = v
        elif k.startswith("model.language_model") or k.startswith("lm_head"):
            text_state_dict[k] = v
        else:
            other_state_dict[k] = v
    
    return vision_state_dict, text_state_dict, other_state_dict


def map_vision_hf_to_meta_keys(
    state_dict: Dict[str, np.ndarray], 
    head_dim: int
) -> Dict[str, np.ndarray]:
    """
    Map multimodal Gemma3 HF keys to Meta format.
    
    Splits into vision/text/other, converts each appropriately,
    then merges back together.
    
    Args:
        state_dict: Multimodal HF state dictionary
        head_dim: Dimension per attention head (for text model)
        
    Returns:
        Meta format state dictionary
    """
    vision_state_dict, text_state_dict, other_state_dict = (
        map_vision_hf_to_meta_keys_split_to_submodels(state_dict)
    )
    
    # Convert text model keys
    text_state_dict = convert_hf_qkv_to_meta_format(text_state_dict, head_dim)
    text_state_dict = map_hf_to_meta_keys(text_state_dict)
    
    # Convert vision model keys
    vision_state_dict = map_hf_to_meta_keys_vision_only(vision_state_dict)
    
    # Merge all components
    return {**vision_state_dict, **text_state_dict, **other_state_dict}


def convert_vision_hf_to_meta(
    state_dict: Dict[str, np.ndarray], 
    head_dim: int
) -> Dict[str, np.ndarray]:
    """
    Convert multimodal Gemma3 HF state dict to Meta format.
    
    Args:
        state_dict: HuggingFace multimodal state dictionary
        head_dim: Dimension per attention head
        
    Returns:
        Meta format state dictionary
    """
    state_dict = split_hf_keys(state_dict)
    state_dict = map_vision_hf_to_meta_keys(state_dict, head_dim)
    return state_dict


# ============================================================================
# Meta to HuggingFace Conversion Functions (Reverse)
# ============================================================================
def convert_meta_qkv_to_hf_format(
    loaded_weights: Dict[str, np.ndarray], 
    head_dim: int
) -> Dict[str, np.ndarray]:
    """
    Convert Meta Q/K projection weights to HuggingFace format.
    Inverse of convert_hf_qkv_to_meta_format.
    
    Args:
        loaded_weights: State dictionary with Meta format weights
        head_dim: Dimension per attention head
        
    Returns:
        State dictionary with HF format Q/K weights
    """
    converted_weights: Dict[str, np.ndarray] = {}
    
    for key, tensor in loaded_weights.items():
        if "wq.weight" in key or "wk.weight" in key:
            n_heads = tensor.shape[0] // head_dim
            converted_weights[key] = _forward_permute(
                tensor, n_heads, tensor.shape[0], tensor.shape[1]
            )
        else:
            converted_weights[key] = tensor
    
    return converted_weights


def map_meta_to_hf_keys(
    loaded_weights: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Map Meta key names to HuggingFace key names.
    Inverse of map_hf_to_meta_keys.
    
    Args:
        loaded_weights: State dictionary with Meta key names
        
    Returns:
        State dictionary with HF key names
    """
    meta_to_hf = {
        # Top-level mappings
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
        # Layer-level mappings (templates)
        "layers.{layer}.attention_norm.weight": "model.layers.{layer}.input_layernorm.weight",
        "layers.{layer}.ffn_norm.weight": "model.layers.{layer}.post_attention_layernorm.weight",
        # Attention mappings
        "layers.{layer}.attention.wq.weight": "model.layers.{layer}.self_attn.q_proj.weight",
        "layers.{layer}.attention.wk.weight": "model.layers.{layer}.self_attn.k_proj.weight",
        "layers.{layer}.attention.wv.weight": "model.layers.{layer}.self_attn.v_proj.weight",
        "layers.{layer}.attention.wo.weight": "model.layers.{layer}.self_attn.o_proj.weight",
        "layers.{layer}.attention.wq.bias": "model.layers.{layer}.self_attn.q_proj.bias",
        "layers.{layer}.attention.wk.bias": "model.layers.{layer}.self_attn.k_proj.bias",
        "layers.{layer}.attention.wv.bias": "model.layers.{layer}.self_attn.v_proj.bias",
        "layers.{layer}.attention.q_norm.weight": "model.layers.{layer}.self_attn.q_norm.weight",
        "layers.{layer}.attention.k_norm.weight": "model.layers.{layer}.self_attn.k_norm.weight",
        # MLP mappings
        "layers.{layer}.feed_forward.w1.weight": "model.layers.{layer}.mlp.gate_proj.weight",
        "layers.{layer}.feed_forward.w3.weight": "model.layers.{layer}.mlp.up_proj.weight",
        "layers.{layer}.feed_forward.w2.weight": "model.layers.{layer}.mlp.down_proj.weight",
        # Pre/post feedforward layernorm (Gemma3 specific)
        "layers.{layer}.ffn_norm_2.weight": "model.layers.{layer}.post_feedforward_layernorm.weight",
    }
    
    hf_state_dict: Dict[str, np.ndarray] = {}
    
    for key, tensor in loaded_weights.items():
        if key in meta_to_hf:
            hf_state_dict[meta_to_hf[key]] = tensor
        elif "layers." in key and (
            ".attention." in key or 
            ".feed_forward." in key or 
            ".attention_norm." in key or 
            ".ffn_norm" in key
        ):
            # Extract layer number and rebuild template
            parts = key.split(".")
            layer_num = parts[1]
            template_key = "layers.{layer}." + ".".join(parts[2:])
            
            if template_key in meta_to_hf:
                hf_state_dict[meta_to_hf[template_key].format(layer=layer_num)] = tensor
            else:
                hf_state_dict[key] = tensor
        else:
            hf_state_dict[key] = tensor
    
    return hf_state_dict


def map_meta_to_hf_keys_vision_only(
    state_dict: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Map Meta-style vision-model key names to HuggingFace names.
    Inverse of map_hf_to_meta_keys_vision_only.
    
    Args:
        state_dict: Vision model state dictionary with Meta keys
        
    Returns:
        State dictionary with HF keys
    """
    replacements = [
        ("attn", "self_attn"),
        ("wq", "q_proj"),
        ("wk", "k_proj"),
        ("wv", "v_proj"),
        ("wo", "out_proj"),
        ("ln_1", "layer_norm1"),
        ("ln_2", "layer_norm2"),
        ("ln_post", "post_layernorm"),
        ("patch_conv._linear", "patch_conv"),
    ]
    
    output: Dict[str, np.ndarray] = {}
    for k, v in state_dict.items():
        output[_replace_suffixes(k, replacements)] = v
    
    return output


def convert_meta_to_hf(
    state_dict: Dict[str, np.ndarray], 
    head_dim: int
) -> Dict[str, np.ndarray]:
    """
    Convert text-only Meta state dict to HuggingFace format.
    Inverse of convert_hf_to_meta.
    
    Args:
        state_dict: Meta format state dictionary
        head_dim: Dimension per attention head
        
    Returns:
        HuggingFace format state dictionary
    """
    state_dict = convert_meta_qkv_to_hf_format(state_dict, head_dim)
    state_dict = map_meta_to_hf_keys(state_dict)
    return state_dict


# ============================================================================
# Vision Meta to HuggingFace Conversion Functions (Reverse)
# ============================================================================
def map_vision_meta_to_hf_keys_split_to_submodels(
    state_dict: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split Meta state dict into vision, text, and other components.
    
    Args:
        state_dict: Combined Meta format state dictionary
        
    Returns:
        Tuple of (vision_state_dict, text_state_dict, other_state_dict)
    """
    vision_state_dict: Dict[str, np.ndarray] = {}
    text_state_dict: Dict[str, np.ndarray] = {}
    other_state_dict: Dict[str, np.ndarray] = {}
    
    for k, v in state_dict.items():
        if k.startswith("model.vision_tower"):
            vision_state_dict[k] = v
        elif "layers." in k or k in (
            "tok_embeddings.weight", 
            "norm.weight", 
            "output.weight"
        ):
            text_state_dict[k] = v
        else:
            other_state_dict[k] = v
    
    return vision_state_dict, text_state_dict, other_state_dict


def map_vision_meta_to_hf_keys(
    state_dict: Dict[str, np.ndarray], 
    head_dim: Optional[int] = None  # Fix: explicitly Optional[int]
) -> Dict[str, np.ndarray]:
    """
    Map multimodal Gemma3 Meta keys to HuggingFace format.
    Inverse of map_vision_hf_to_meta_keys.
    
    Args:
        state_dict: Meta format multimodal state dictionary
        head_dim: Dimension per attention head (optional, for Q/K conversion)
        
    Returns:
        HuggingFace format state dictionary
    """
    vision_state_dict, text_state_dict, other_state_dict = (
        map_vision_meta_to_hf_keys_split_to_submodels(state_dict)
    )
    if head_dim is not None:
        text_state_dict = convert_meta_qkv_to_hf_format(text_state_dict, head_dim)
    text_state_dict = map_meta_to_hf_keys(text_state_dict)
    vision_state_dict = map_meta_to_hf_keys_vision_only(vision_state_dict)
    return {**vision_state_dict, **text_state_dict, **other_state_dict}

def convert_vision_meta_to_hf(
    state_dict: Dict[str, np.ndarray], 
    head_dim: int
) -> Dict[str, np.ndarray]:
    """
    Convert multimodal Gemma3 Meta state dict to HuggingFace format.
    Inverse of convert_vision_hf_to_meta.
    
    Args:
        state_dict: Meta format multimodal state dictionary
        head_dim: Dimension per attention head
        
    Returns:
        HuggingFace format state dictionary
    """
    state_dict = map_vision_meta_to_hf_keys(state_dict, head_dim)
    return state_dict