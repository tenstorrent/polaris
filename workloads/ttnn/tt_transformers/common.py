#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import math
import re
from enum import Enum
from types import SimpleNamespace
from typing import List, Optional, Union

import numpy as np
from typing import List, Any
import ttsim.front.ttnn as ttnn
from loguru import logger


# ============================================================================
# Pydantic stubs
# ============================================================================
class BaseModel:
    """Stub for pydantic BaseModel."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    class Config:
        arbitrary_types_allowed = True


def Field(*args, **kwargs):
    """Stub for pydantic Field."""
    return kwargs.get('default', None)


def AliasChoices(*args):
    """Stub for pydantic AliasChoices."""
    return args


# ============================================================================
# PIL stub
# ============================================================================
class PIL_Image:
    """Stub for PIL Image."""
    class Image:
        pass


# ============================================================================
# URL and Media Classes
# ============================================================================
class URL(BaseModel):
    def __init__(self, uri: str = "", **kwargs):
        super().__init__(**kwargs)
        self.uri = uri

    def __str__(self) -> str:
        return self.uri


class ImageMedia(BaseModel):
    def __init__(self, image=None, **kwargs):
        super().__init__(**kwargs)
        self.image = image

    class Config:
        arbitrary_types_allowed = True


class Role(Enum):
    system = "system"
    user = "user"
    assistant = "assistant"
    ipython = "ipython"


InterleavedTextMedia = Union[
    str,
    ImageMedia,
    List[Union[str, ImageMedia]],
]


class InferencePhase(Enum):
    DECODE = "decode"
    PREFILL = "prefill"


# ============================================================================
# Host Embedding Classes (numpy-based)
# ============================================================================
class HostEmbedding:
    """Host embedding layer using numpy."""
    def __init__(self, model_args):
        self.vocab_size = model_args.vocab_size
        self.dim = model_args.dim
        # Initialize embedding weights randomly
        self.weight = np.random.randn(model_args.vocab_size, model_args.dim).astype(np.float32) * 0.02

    def forward(self, x):
        """
        x: numpy array of indices [batch, seq_len]
        returns: [batch, seq_len, dim]
        """
        if isinstance(x, np.ndarray):
            indices = x.astype(np.int64)
        else:
            indices = np.array(x, dtype=np.int64)

        return self.weight[indices.flatten()].reshape(list(indices.shape) + [self.dim])

    def __call__(self, x):
        return self.forward(x)


class HostScaledEmbedding(HostEmbedding):
    """Scaled host embedding layer."""
    def __init__(self, model_args):
        super().__init__(model_args)
        self.embed_scale = model_args.embed_scale

    def forward(self, x):
        return super().forward(x) * self.embed_scale


# ============================================================================
# Paged Attention Configuration
# ============================================================================
class PagedAttentionConfig:
    def __init__(self, block_size=32, max_num_blocks=1024):
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks


# ============================================================================
# RoPE Scaling Classes
# ============================================================================
class RopeScalingType(str, Enum):
    """Types of RoPE scaling."""
    LINEAR = "linear"
    YARN = "yarn"
    LLAMA3 = "llama3"
    PHI3 = "longrope"
    DEFAULT = "default"


class RopeScaling(BaseModel):
    """RoPE scaling configuration."""
    def __init__(
        self,
        rope_type=None,
        factor: Optional[float] = None,
        original_max_position_embeddings: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if rope_type is None:
            rope_type = kwargs.get('type', RopeScalingType.DEFAULT)
        self.rope_type = rope_type
        self.factor = factor
        self.original_max_position_embeddings = original_max_position_embeddings


class RopeScalingLinear(RopeScaling):
    """RoPE scaling configuration for linear."""
    pass


class RopeScalingLlama3(RopeScaling):
    """RoPE scaling configuration for Llama-3.x."""
    def __init__(
        self,
        low_freq_factor: Optional[float] = 1.0,
        high_freq_factor: Optional[float] = 4.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor


class RopeScalingYarn(RopeScaling):
    """RoPE scaling configuration for Yarn."""
    def __init__(
        self,
        beta_fast: Optional[float] = 32.0,
        beta_slow: Optional[float] = 1.0,
        mscale: Optional[float] = 1.0,
        mscale_all_dim: Optional[float] = 0.0,
        truncate: Optional[bool] = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        self.truncate = truncate


class RopeScalingPhi3(RopeScaling):
    """RoPE scaling configuration for Phi3."""
    def __init__(
        self,
        long_factor: Optional[list] = None,
        short_factor: Optional[list] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.long_factor = long_factor
        self.short_factor = short_factor


def rope_scaling_model_factory(
    rope_scaling_params: dict, original_max_context_len: Optional[int] = None
):
    rope_scaling_type = rope_scaling_params.get("rope_type") or rope_scaling_params.get("type")

    if rope_scaling_type == RopeScalingType.LINEAR or rope_scaling_type == "linear":
        return RopeScalingLinear(**rope_scaling_params)
    elif rope_scaling_type == RopeScalingType.LLAMA3 or rope_scaling_type == "llama3":
        return RopeScalingLlama3(**rope_scaling_params)
    elif rope_scaling_type == RopeScalingType.YARN or rope_scaling_type == "yarn":
        return RopeScalingYarn(**rope_scaling_params)
    elif rope_scaling_type == RopeScalingType.PHI3 or rope_scaling_type == "longrope":
        return RopeScalingPhi3(original_max_position_embeddings=original_max_context_len, **rope_scaling_params)
    elif rope_scaling_type in ["default", "mrope"]:
        logger.warning(
            f"Rope scaling type was set to {rope_scaling_type}, defaulting to no rope scaling"
        )
        return None
    else:
        raise ValueError(f"Unexpected RoPE scaling type: {rope_scaling_type}")


# ============================================================================
# Rotation Matrix Helper
# ============================================================================
def get_rot_transformation_mat_v2(dhead=32):
    """Creates a transformation matrix for rotary embeddings."""
    rot_emb_matrix = np.zeros((1, 1, dhead, dhead), dtype=np.float32)
    rot_emb_matrix[..., np.arange(0, dhead, 2), np.arange(1, dhead, 2)] = 1
    rot_emb_matrix[..., np.arange(1, dhead, 2), np.arange(0, dhead, 2)] = -1
    return rot_emb_matrix


# ============================================================================
# Mistral Vision Support
# ============================================================================
def position_ids_in_meshgrid_tt(tt_patch_embeds_list, max_width, device):
    position_ids_tt = []
    for tt_patch in tt_patch_embeds_list:
        shape = tt_patch.shape
        height, width = shape[-2], shape[-1]

        # Create meshgrid
        h_range = np.arange(height)
        w_range = np.arange(width)
        h_grid, w_grid = np.meshgrid(h_range, w_range, indexing="ij")

        # Stack and reshape
        mesh = np.stack([h_grid, w_grid], axis=-1).reshape(-1, 2)
        h_vals = mesh[:, 0:1]
        w_vals = mesh[:, 1:2]

        ids = h_vals * max_width + w_vals

        tt_ids = ttnn.from_numpy( # type: ignore[attr-defined]
            ids,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        position_ids_tt.append(tt_ids[:, 0])

    return ttnn.concat(position_ids_tt, dim=0)


# ============================================================================
# Prompt Encoding Functions
# ============================================================================
def encode_prompt_instruct(tokenizer, prompt_text, system_prompt_text=None):
    """Encode prompt for instruct format."""
    begin_of_text = [tokenizer.special_tokens["<|begin_of_text|>"]]
    start_header = [tokenizer.special_tokens["<|start_header_id|>"]]
    end_header = [tokenizer.special_tokens["<|end_header_id|>"]]
    end_turn = [tokenizer.special_tokens["<|eot_id|>"]]

    system = tokenizer.encode("system", bos=False, eos=False)
    user = tokenizer.encode("user", bos=False, eos=False)
    assistant = tokenizer.encode("assistant", bos=False, eos=False)
    prompt = tokenizer.encode(prompt_text, bos=False, eos=False)

    system_prompt = start_header + system + end_header + system_prompt_text + end_turn if system_prompt_text else []
    user_prompt = start_header + user + end_header + prompt + end_turn
    assistant_reply = start_header + assistant + end_header

    return begin_of_text + system_prompt + user_prompt + assistant_reply


def preprocess_inputs_prefill(
    input_prompts,
    tokenizer,
    model_args,
    instruct,
    max_generated_tokens,
    max_prefill_len=128 * 1024,
):
    """Run tokenizer on inputs, and create embeddings for the first token of each input."""
    for m_args in model_args:
        assert (
            max_prefill_len <= m_args.max_context_len
        ), f"max_prefill_len {max_prefill_len} cannot exceed max_context_len {m_args.max_context_len}"

    max_prefill_len -= max_generated_tokens
    assert (
        max_prefill_len > 0
    ), f"max_prefill_len must be greater than max_generated_tokens ({max_generated_tokens})"

    encoded_prompts = [
        model_args[idx % len(model_args)].encode_prompt(prompt, instruct=instruct)
        for idx, prompt in enumerate(input_prompts)
    ]

    logger.info("Encoded prompt lengths:" + ", ".join(str(len(prompt)) for prompt in encoded_prompts))

    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    if min_prompt_len > max_prefill_len:
        logger.info(f"Left-clipping prompts to {max_prefill_len}")
        if instruct:
            raw_prompts = [
                model_args[idx % len(model_args)].encode_prompt(prompt, instruct=False)
                for idx, prompt in enumerate(input_prompts)
            ]
            overhead = [len(e) - len(r) for e, r in zip(encoded_prompts, raw_prompts)]
            shortened = []
            for idx, (e, o) in enumerate(zip(raw_prompts, overhead)):
                if isinstance(tokenizer, list):
                    sp = tokenizer[idx % len(model_args)].decode(e[-(max_prefill_len - o):])
                else:
                    sp = tokenizer.decode(e[-(max_prefill_len - o):])
                shortened.append(sp)

            encoded_prompts = [
                model_args[idx % len(model_args)].encode_prompt(prompt, instruct=instruct)
                for idx, prompt in enumerate(shortened)
            ]
            assert all(
                len(e) == max_prefill_len for e in encoded_prompts
            ), "Clipped prompts are not of the correct length"
        else:
            encoded_prompts = [encod[-max_prefill_len:] for encod in encoded_prompts]

        prompt_lens = [len(x) for x in encoded_prompts]
        min_prompt_len = min(prompt_lens)
        max_prompt_len = max(prompt_lens)

    for m in model_args:
        assert (
            max_prompt_len <= m.max_seq_len
        ), f"Max prompt length {max_prompt_len} exceeds model max seq len {m.max_seq_len}"

    assert min_prompt_len > 0, "Minimum prompt length must be greater than 0"
    assert min_prompt_len <= max_prompt_len

    logger.info(f"# of users: {len(encoded_prompts)}")

    input_tokens_prefill = []
    decoding_pos = []
    prefill_lens = []

    for i, encoded in enumerate(encoded_prompts):
        input_tokens_prefill_i = np.full((1, max_prompt_len), 0, dtype=np.int32)
        input_tokens_prefill_i[0, :len(encoded[:])] = np.array(encoded[:], dtype=np.int32)
        input_tokens_prefill.append(input_tokens_prefill_i)
        decoding_pos.append(len(encoded))
        prefill_lens.append(max_prompt_len)

    return (
        input_tokens_prefill,
        encoded_prompts,
        decoding_pos,
        prefill_lens,
    )


def encode_prompt_hf(tokenizer, prompt_text, system_prompt_text=None):
    """See https://huggingface.co/docs/transformers/main/en/chat_templating"""
    chat = []

    if isinstance(prompt_text, str):
        if system_prompt_text:
            chat.append({"role": "system", "content": system_prompt_text})
        if prompt_text:
            chat.append({"role": "user", "content": prompt_text})
        return tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=True)
    else:
        return tokenizer.apply_chat_template(prompt_text, add_generation_prompt=True, tokenize=True)


# ============================================================================
# RoPE Scaling Functions
# ============================================================================
def compute_llama3_parameters(freqs: np.ndarray, scale_factor: float, orig_context_len: int):
    """Llama-3.x specific scaling for rotary embeddings."""
    low_freq_factor = 1
    high_freq_factor = 4
    low_freq_wavelen = orig_context_len / low_freq_factor
    high_freq_wavelen = orig_context_len / high_freq_factor

    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (orig_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)

    return np.array(new_freqs, dtype=freqs.dtype)


def compute_linear_parameters(freqs: np.ndarray, scale_factor: float, orig_context_len: int):
    """Linear scaling for rotary embeddings."""
    return freqs / scale_factor


def compute_default_parameters(freqs: np.ndarray, scale_factor: float, orig_context_len: int):
    """Default scaling for rotary embeddings."""
    return freqs


def apply_scaling(freqs: np.ndarray, scale_factor: float, orig_context_len: int, rope_type="llama3"):
    if rope_type == "default":
        freqs = compute_default_parameters(freqs, scale_factor, orig_context_len)
    elif rope_type == "linear":
        freqs = compute_linear_parameters(freqs, scale_factor, orig_context_len)
    elif rope_type == "llama3":
        freqs = compute_llama3_parameters(freqs, scale_factor, orig_context_len)
    return freqs


def apply_scaling_vision(freqs: np.ndarray, scale_factor: float, orig_context_len: int):
    """Minimal addition for Mistral vision RoPE support."""
    return freqs / scale_factor


def precompute_mistral_vision_freqs(
    dim: int, max_patches_per_side: int, theta: float, scale_factor=None, orig_context_len=None
):
    """Minimal addition for Mistral vision RoPE support."""
    base_freqs = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))

    if scale_factor is not None:
        base_freqs = apply_scaling_vision(base_freqs, scale_factor, orig_context_len)

    h_idx = np.arange(max_patches_per_side)
    w_idx = np.arange(max_patches_per_side)

    freqs_h = np.outer(h_idx, base_freqs[::2])
    freqs_w = np.outer(w_idx, base_freqs[1::2])

    inv_freq = np.concatenate(
        [
            np.tile(freqs_h[:, None, :], (1, max_patches_per_side, 1)),
            np.tile(freqs_w[None, :, :], (max_patches_per_side, 1, 1)),
        ],
        axis=-1,
    ).reshape(-1, dim // 2)

    full_freqs = np.concatenate([inv_freq, inv_freq], axis=-1)
    cos = np.cos(full_freqs)
    sin = np.sin(full_freqs)

    return cos, sin


def precompute_freqs(dim: int, end: int, theta, scale_factor, orig_context_len, rope_type="llama3"):
    """Precompute the frequency tensor for sine and cosine values."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[:(dim // 2)].astype(np.float32) / dim))
    t = np.arange(end)

    if scale_factor is not None:
        freqs = apply_scaling(freqs, scale_factor, orig_context_len, rope_type=rope_type)

    freqs = np.outer(t, freqs).astype(np.float32)

    return np.cos(freqs), np.sin(freqs)


def freqs_to_rotation_matrix(cos_freqs, sin_freqs):
    """Transform cos/sin frequencies to a rotation matrix."""
    emb_size, emb_dim = cos_freqs.shape
    dhead = emb_dim * 2

    rot_emb_matrix = np.zeros((emb_size, dhead, dhead), dtype=np.float32)
    rot_emb_matrix[..., np.arange(0, dhead, 2), np.arange(0, dhead, 2)] = cos_freqs.copy()
    rot_emb_matrix[..., np.arange(1, dhead, 2), np.arange(1, dhead, 2)] = cos_freqs.copy()
    rot_emb_matrix[..., np.arange(0, dhead, 2), np.arange(1, dhead, 2)] = -sin_freqs.copy()
    rot_emb_matrix[..., np.arange(1, dhead, 2), np.arange(0, dhead, 2)] = sin_freqs.copy()

    rot_emb_matrix = np.transpose(rot_emb_matrix, (0, 2, 1))

    return rot_emb_matrix


def gather_cos_sin(position_ids, cos, sin):
    position_id_expanded = np.expand_dims(position_ids, 1)
    position_id_expanded = np.broadcast_to(position_id_expanded, (position_ids.shape[0], cos.shape[-1]))

    cos_gathered = np.take_along_axis(cos, position_id_expanded, axis=0)
    sin_gathered = np.take_along_axis(sin, position_id_expanded, axis=0)

    cos_stacked = np.stack([cos_gathered, cos_gathered], axis=-1).reshape(cos_gathered.shape[0], -1)
    sin_stacked = np.stack([sin_gathered, sin_gathered], axis=-1).reshape(sin_gathered.shape[0], -1)

    cos_out = np.expand_dims(np.expand_dims(cos_stacked, 0), 0)
    sin_out = np.expand_dims(np.expand_dims(sin_stacked, 0), 0)

    return cos_out, sin_out


def get_prefill_rot_mat(head_dim, mesh_device, seq_len, theta, scale_factor, orig_context_len, start_pos=0):
    cos, sin = precompute_freqs(
        head_dim, seq_len * 2, theta=theta, scale_factor=scale_factor, orig_context_len=orig_context_len
    )
    cos_gathered, sin_gathered = gather_cos_sin(np.arange(start_pos, start_pos + seq_len), cos, sin)

    assert cos_gathered.shape == (1, 1, seq_len, head_dim)
    assert sin_gathered.shape == (1, 1, seq_len, head_dim)

    cos_gathereds = ttnn.from_numpy( # type: ignore[attr-defined]
        cos_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_gathereds = ttnn.from_numpy( # type: ignore[attr-defined]
        sin_gathered,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    rot_mats = [cos_gathereds, sin_gathereds]
    return rot_mats


def get_rot_transformation_mat(dhead=32):
    """Add-Multiply method of rotary embeddings for prefill."""
    dhead = 32
    return get_rot_transformation_mat_v2(dhead)


def get_single_rot_mat(
    dhead,
    mesh_device,
    num_devices,
    start_pos,
    theta,
    scale_factor,
    orig_context_len,
    on_host=False,
):
    freqs_unscaled = 1.0 / (theta ** (np.arange(0, dhead, 2)[:(dhead // 2)].astype(np.float32) / dhead))

    if scale_factor is not None:
        freqs = apply_scaling(freqs_unscaled, scale_factor, orig_context_len, rope_type="llama3")
    else:
        freqs = freqs_unscaled.copy()

    rot_matrix = np.zeros((dhead, dhead), dtype=np.float32)
    sin_freqs = np.sin(freqs).astype(np.float32)
    cos_freqs = np.cos(freqs).astype(np.float32)

    rot_matrix[np.arange(0, dhead, 2), np.arange(0, dhead, 2)] = cos_freqs.copy()
    rot_matrix[np.arange(1, dhead, 2), np.arange(1, dhead, 2)] = cos_freqs.copy()
    rot_matrix[np.arange(0, dhead, 2), np.arange(1, dhead, 2)] = -sin_freqs.copy()
    rot_matrix[np.arange(1, dhead, 2), np.arange(0, dhead, 2)] = sin_freqs.copy()
    rot_matrix = rot_matrix.T

    freqs_pos = start_pos * freqs_unscaled
    if scale_factor is not None:
        freqs_pos = apply_scaling(freqs_pos, scale_factor, orig_context_len, rope_type="llama3")

    current_rot_mat = np.zeros((dhead, dhead), dtype=np.float32)
    sin_freqs_pos = np.sin(freqs_pos).astype(np.float32)
    cos_freqs_pos = np.cos(freqs_pos).astype(np.float32)

    current_rot_mat[np.arange(0, dhead, 2), np.arange(0, dhead, 2)] = cos_freqs_pos.copy()
    current_rot_mat[np.arange(1, dhead, 2), np.arange(1, dhead, 2)] = cos_freqs_pos.copy()
    current_rot_mat[np.arange(0, dhead, 2), np.arange(1, dhead, 2)] = -sin_freqs_pos.copy()
    current_rot_mat[np.arange(1, dhead, 2), np.arange(0, dhead, 2)] = sin_freqs_pos.copy()

    current_rot_mat_expanded = np.expand_dims(np.expand_dims(current_rot_mat.T, 0), 0)
    rot_matrix_expanded = np.expand_dims(np.expand_dims(rot_matrix, 0), 0)

    return ttnn.from_numpy( # type: ignore[attr-defined]
        current_rot_mat_expanded,
        device=mesh_device if not on_host else None,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if num_devices > 1 or not on_host else None,
    ), ttnn.from_numpy( # type: ignore[attr-defined]
        rot_matrix_expanded,
        device=mesh_device if not on_host else None,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if num_devices > 1 or not on_host else None,
    )


# ============================================================================
# Core Range Helper
# ============================================================================
def num_to_core_range_set(x):
    assert x < 8 or x % 8 == 0
    num_x = min(x, 8)
    num_y = x // num_x
    assert num_x * num_y == x

    return ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(num_x - 1, num_y - 1),
            ),
        }
    )


# ============================================================================
# Host to Device Copy
# ============================================================================
def copy_host_to_device(
    host_tensors,
    device_tensors=None,
    mesh_device=None,
    shard_specs=None,
):
    """Helper function which copies host tensors to device tensors."""
    if device_tensors is None:
        assert mesh_device is not None, "mesh_device is required when device_tensors is None"
        ret: List[Any] = []
        for i in range(len(host_tensors)):
            if host_tensors[i] is None:
                ret.append(None)
            elif shard_specs and shard_specs[i] is not None:
                on_device = ttnn.from_numpy(host_tensors[i], device=mesh_device) # type: ignore[attr-defined]
                ret.append(on_device)
            else:
                on_device = ttnn.to_device(host_tensors[i], device=mesh_device)
                ret.append(on_device)
        return ret
    else:
        for i in range(len(host_tensors)):
            if host_tensors[i] is None:
                assert device_tensors[i] is None
                continue
            ttnn.copy_host_to_device_tensor(host_tensors[i], device_tensors[i])
        return device_tensors


# ============================================================================
# Model Configuration Helpers
# ============================================================================
def calculate_hidden_dim(dim, ffn_dim_multiplier, multiple_of):
    """Helper function based on logic used in reference model."""
    hidden_dim = int(2 * (4 * dim) / 3)
    if ffn_dim_multiplier is not None:
        hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


def get_out_subblock_w(per_core_N, out_subblock_h):
    """Helper function to calculate out_subblock_w."""
    out_subblock_w = 4
    while out_subblock_w > 1:
        if out_subblock_w * out_subblock_h <= 4 and per_core_N % out_subblock_w == 0:
            break
        out_subblock_w -= 1
    return out_subblock_w


# ============================================================================
# Debug Helpers
# ============================================================================
def first_five(tensor, mesh_device, start=0, end=5):
    """Helper function to return the first 5 elements of a tensor."""
    tensor_np = ttnn.to_torch(tensor).numpy()  # type: ignore[union-attr]
    return tensor_np[0, 0, 0, start:end]


def last_five(tensor, mesh_device):
    """Helper function to return the last 5 elements of a tensor."""
    tensor_np = ttnn.to_torch(tensor).numpy()  # type: ignore[union-attr]
    return tensor_np[0, 0, 0, -5:]


# ============================================================================
# Sampling Functions
# ============================================================================
def sample_top_p(probs: np.ndarray, p: float):
    assert 0 <= p <= 1

    probs_sort_idx = np.argsort(probs, axis=-1)[:, ::-1]
    probs_sort = np.take_along_axis(probs, probs_sort_idx, axis=-1)
    probs_sum = np.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort = probs_sort / probs_sort.sum(axis=-1, keepdims=True)

    # Multinomial sampling
    cumsum = np.cumsum(probs_sort, axis=-1)
    random_vals = np.random.rand(probs_sort.shape[0], 1)
    next_token_idx = np.argmax(cumsum >= random_vals, axis=-1, keepdims=True)
    next_token = np.take_along_axis(probs_sort_idx, next_token_idx, axis=-1)

    return next_token


def sample_host(tt_input, temperature=0.6, top_p=0.08, on_host=True):
    vocab_size = tt_input.shape[-1]
    pt_input = tt_input[..., :vocab_size]

    if temperature > 0:
        # Softmax
        pt_input_shifted = pt_input - np.max(pt_input, axis=-1, keepdims=True)
        exp_input = np.exp(pt_input_shifted / temperature)
        probs = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        pt_out = sample_top_p(probs.squeeze(), top_p)
    else:
        pt_out = np.argmax(pt_input, axis=-1)

    if pt_out.ndim == 1:
        pt_out = np.expand_dims(pt_out, 0)

    return None, pt_out


# ============================================================================
# Sequence Length Helpers
# ============================================================================
def get_padded_prefill_len(seq_len: int) -> int:
    """Get the padded prefill length for a given sequence length."""
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    else:
        return 2 ** (seq_len - 1).bit_length()


def get_all_padded_prefill_lengths(max_len):
    lengths = [128]
    k = 0
    while (v := (1 << k) * 1024) <= max_len:
        lengths.append(v)
        k += 1
    return lengths


def calculate_prefill_warmup_seq_lens(max_seq_len_to_warmup, trace_supported_seq_lens):
    to_warmup_seq_lens = get_all_padded_prefill_lengths(max_seq_len_to_warmup)

    for trace_supported_seq_len in trace_supported_seq_lens:
        if trace_supported_seq_len not in to_warmup_seq_lens:
            to_warmup_seq_lens.append(trace_supported_seq_len)

    to_warmup_seq_lens.sort()
    return to_warmup_seq_lens


def cap_seq_lens_to_max_prefill_chunk_size(seq_lens, cap):
    for seq_len in seq_lens:
        if seq_len > cap:
            seq_lens = seq_lens[:seq_lens.index(seq_len)]
            break
    return seq_lens


def get_block_size(kv_cache):
    if kv_cache is None:
        return 64
    if isinstance(kv_cache, list) and len(kv_cache) > 0:
        if isinstance(kv_cache[0], list) and len(kv_cache[0]) > 0:
            return kv_cache[0][0].shape[2]
        elif hasattr(kv_cache[0], 'shape'):
            return kv_cache[0].shape[2]
    return 64


def num_blocks_in_seq(seq_len, block_size):
    return math.ceil(seq_len / block_size)


def nearest_pow_2(x):
    return 2 ** math.ceil(math.log2(x))


def get_max_prefill_chunk_size(seq_len, max_prefill_seq_len):
    """Determine the largest multiple of 2048 that divides seq_len."""
    MIN_CHUNK_SIZE = 2048

    if not isinstance(seq_len, int) or not isinstance(max_prefill_seq_len, int):
        raise TypeError("Both seq_len and max_prefill_seq_len must be integers.")

    if seq_len <= 0 or max_prefill_seq_len <= 0:
        raise ValueError("Both seq_len and max_prefill_seq_len must be positive integers.")

    if seq_len % MIN_CHUNK_SIZE != 0:
        raise ValueError(f"seq_len ({seq_len}) must be a multiple of {MIN_CHUNK_SIZE}.")

    if max_prefill_seq_len % MIN_CHUNK_SIZE != 0:
        raise ValueError(f"max_prefill_seq_len ({max_prefill_seq_len}) must be a multiple of {MIN_CHUNK_SIZE}.")

    max_possible_chunk = min(max_prefill_seq_len, seq_len)

    for chunk_size in range(max_possible_chunk, 0, -MIN_CHUNK_SIZE):
        if seq_len % chunk_size == 0:
            return chunk_size

    raise ValueError("No valid chunk size found")


def nearest_multiple(x, multiple_of):
    return math.ceil(x / multiple_of) * multiple_of


def pad_to_size(x: np.ndarray, dim: int, size: int) -> np.ndarray:
    """Pads the specified dimension of the input array with zeros."""
    if dim < 0:
        dim = x.ndim + dim

    assert isinstance(x, np.ndarray), "Input must be a numpy array"
    assert -x.ndim <= dim < x.ndim, f"Dimension {dim} out of range"

    dim = x.ndim + dim if dim < 0 else dim
    current_size = x.shape[dim]
    pad_size = size - current_size

    if pad_size == 0:
        return x

    pad_widths = [(0, 0)] * x.ndim
    pad_widths[dim] = (0, pad_size)

    padded_x = np.pad(x, pad_widths, mode='constant', constant_values=0)
    return padded_x


# ============================================================================
# Model Name Helpers
# ============================================================================
def get_base_model_name(model_name: str) -> str:
    if "phi-4" in model_name.lower():
        return "Phi-4"

    match = re.search(r"(.*?\d+[bB])-", model_name)
    return match.group(1) if match else model_name


def get_hf_model_name(model_path: str) -> str:
    if model_path.count("/") == 1:
        return model_path

    pattern = r".*/?models--(?P<model_provider>[^/]+?)--(?P<model_name>[^/]+)/?"
    match = re.search(pattern, model_path)

    if match:
        model_provider = match.group("model_provider")
        model_name = match.group("model_name")
        return f"{model_provider}/{model_name}"

    raise ValueError(
        f"Unsupported '{model_path}', please use HF model name or follow HF format"
    )


def get_hf_tt_cache_path(model_path: str) -> str:
    tt_cache_home = os.getenv("TT_CACHE_HOME", "/mnt/MLPerf/huggingface/tt_cache/")

    if not os.path.exists(tt_cache_home):
        tt_cache_home = "model_cache"

    model_name = get_hf_model_name(model_path)
    tt_cache_path = os.path.join(tt_cache_home, model_name)

    if not os.path.exists(tt_cache_path):
        os.makedirs(tt_cache_path, exist_ok=True)

    return tt_cache_path


# ============================================================================
# Model Creation (Simulation stub)
# ============================================================================
def create_tt_model(
    mesh_device,
    instruct,
    max_batch_size,
    optimizations,
    max_seq_len,
    paged_attention_config=None,
    dtype=None,
    state_dict=None,
    num_layers=None,
    use_prefetcher=False,
    use_hf_rope=False,
):
    """Simulation stub for create_tt_model."""
    if dtype is None:
        dtype = ttnn.bfloat8_b

    class ModelArgsStub:
        def __init__(self):
            self.mesh_device = mesh_device
            self.instruct = instruct
            self.max_batch_size = max_batch_size
            self.max_seq_len = max_seq_len
            self.n_layers = num_layers if num_layers is not None else 32
            self.dim = 4096
            self.vocab_size = 128256

        def weight_cache_path(self, dtype):
            return "model_cache"

        def load_state_dict(self):
            return {}

    tt_model_args = ModelArgsStub()
    logger.warning("create_tt_model: Simulation mode - returning stub model args")

    return tt_model_args, None, None, state_dict or {}


# ============================================================================
# Multimodal Encoding
# ============================================================================
def hf_multimodal_encode(messages, processor):
    hf_messages = []

    for msg in messages:
        hf_content = []
        for item in msg.content:
            if isinstance(item, ImageMedia):
                hf_content.append({
                    "type": "image",
                    "image": item.image,
                })
            elif isinstance(item, str):
                hf_content.append({
                    "type": "text",
                    "text": item,
                })

        hf_messages.append({
            "role": msg.role,
            "content": hf_content,
        })

    encoded = processor.apply_chat_template(
        hf_messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
    )

    return SimpleNamespace(
        **encoded,
        tokens=encoded["input_ids"].squeeze(0),
        vision=SimpleNamespace(
            images=encoded.get("pixel_values", None),
            mask=None,
        ),
    )


# ============================================================================
# Attention Mask Helpers
# ============================================================================
def get_decode_mask(args, mesh_device, paged_attention_config=None):
    """Function to create a decoding mask for the attention mechanism."""
    if paged_attention_config is not None:
        max_seq_len = (paged_attention_config.max_num_blocks * paged_attention_config.block_size) // args.max_batch_size
    else:
        max_seq_len = args.max_seq_len

    n_heads = getattr(args, 'n_heads', 32)
    mesh_shape = getattr(mesh_device, 'shape', (1, 1))
    if isinstance(mesh_shape, tuple):
        num_devices_col = mesh_shape[1] if len(mesh_shape) > 1 else 1
    else:
        num_devices_col = 1

    mask = np.triu(
        np.full(
            (args.max_batch_size, n_heads // num_devices_col, max_seq_len, max_seq_len),
            -np.inf,
            dtype=np.float32,
        ),
        k=1,
    )

    sliding_window = getattr(args, 'sliding_window', 0)
    if sliding_window > 0:
        mask += np.tril(
            np.full(
                (args.max_batch_size, n_heads // num_devices_col, max_seq_len, max_seq_len),
                -np.inf,
                dtype=np.float32,
            ),
            k=-sliding_window,
        )

    return mask


def build_encoder_attention_mask(
    x: np.ndarray,
    ar: np.ndarray,
    ntok: int,
    num_chunks: int,
    n_heads: int,
):
    """Build vision encoder attention mask that omits padding tokens."""
    def get_negative_inf_value(dtype):
        return np.finfo(dtype).min

    masks = []
    for arx in ar:
        mask_i = np.ones((num_chunks, x.shape[2], 1), dtype=x.dtype)
        mask_i[:arx[0] * arx[1], :ntok] = 0
        mask_i = mask_i.reshape(num_chunks * x.shape[2], -1)
        mask_i = mask_i @ mask_i.T * get_negative_inf_value(x.dtype)
        mask_i = np.expand_dims(mask_i, 0)
        masks.append(mask_i)

    masks_stacked = np.stack(masks)
    masks_broadcast = np.broadcast_to(masks_stacked, (masks_stacked.shape[0], n_heads, masks_stacked.shape[2], masks_stacked.shape[3]))
    return masks_broadcast