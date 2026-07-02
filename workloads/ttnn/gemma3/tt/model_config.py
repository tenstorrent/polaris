#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Gemma3 Model Configuration for Polaris/ttsim.
"""
import gc
import json
import os
import sys

import numpy as np
from loguru import logger
from typing import Dict, List

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import ttsim.front.ttnn as ttnn

from workloads.ttnn.gemma3.common.gemma_utils import is_blackhole
from workloads.ttnn.gemma3.tt.load_checkpoints import (
    convert_hf_to_meta,
    convert_meta_to_hf,
    convert_vision_hf_to_meta,
    convert_vision_meta_to_hf,
    standardize_hf_keys,
)
from workloads.ttnn.tt_transformers.common import (
    Mode,
    calculate_prefill_warmup_seq_lens,
    cap_seq_lens_to_max_prefill_chunk_size,
)
from workloads.ttnn.tt_transformers.attention import OpGroup


# File names for performance and accuracy mode override files
PERFORMANCE_DECODER_CONFIG_FILENAME = "performance_decoder_config.json"
ACCURACY_DECODER_CONFIG_FILENAME = "accuracy_decoder_config.json"

# SDPA decode k_chunk constants
_GEMMA3_SDPA_DECODE_K_CHUNK_DEFAULT = 256
_GEMMA3_SDPA_DECODE_K_CHUNK_PROGRAM_TRACE = 32


# ============================================================================
# Optimization Configuration Classes
# ============================================================================

class MathFidelitySetting:
    """Math fidelity settings for compute operations."""
    HIFI4 = "HIFI4"
    HIFI3 = "HIFI3"
    HIFI2 = "HIFI2"
    HIFI2_FP16 = "HIFI2_FP16"
    HIFI2_NA = "HIFI2_NA"
    LOFI = "LOFI"


class ModelOptimizations:
    """Model optimizations configuration."""
    def __init__(self, config=None):
        self.config = config or {}
        self.tensor_dtype_settings = self.config.get("TensorPrecision", {})
        self.op_fidelity_settings = self.config.get("OpFidelity", {})
        self.__name__ = "ModelOptimizations"
        self.decoder_optimizations = {}

    def set_decoder_conf(self, decoder_id, conf):
        self.decoder_optimizations[decoder_id] = conf


# ============================================================================
# HuggingFace Model Wrappers (for reference/testing)
# ============================================================================

class HfAttentionWrapper:
    """Wrapper for HF attention layer (used for reference comparison)."""
    def __init__(self, layer, head_dim, rotary_emb=None):
        self.layer = layer
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        if self.layer is not None and hasattr(self.layer, "load_state_dict"):
            converted = convert_meta_to_hf(state_dict, self.head_dim)
            return self.layer.load_state_dict(converted)


class HfDecoderWrapper:
    """Wrapper for HF decoder layer (used for reference comparison)."""
    def __init__(self, layer, head_dim, rotary_emb=None, rotary_emb_local=None):
        self.layer = layer
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.rotary_emb_local = rotary_emb_local

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        if self.layer is not None and hasattr(self.layer, "load_state_dict"):
            converted = convert_meta_to_hf(state_dict, self.head_dim)
            return self.layer.load_state_dict(converted)


class HfModelWrapper:
    """Wrapper for HF model (used for reference comparison)."""
    def __init__(self, model, head_dim):
        self.model = model
        self.head_dim = head_dim

    def forward(self, x):
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        if self.model is not None and hasattr(self.model, "load_state_dict"):
            converted = convert_meta_to_hf(state_dict, self.head_dim)
            return self.model.load_state_dict(converted)


class HfGemmaDecoderWrapper:
    """Wrapper for HuggingFace Gemma decoder layer (used for reference comparison)."""
    def __init__(self, decoder, head_dim, rotary_emb, rotary_emb_local):
        self.decoder = decoder
        self.head_dim = head_dim
        self.rotary_emb = rotary_emb
        self.rotary_emb_local = rotary_emb_local
        self.past_key_values = None
        self._state_dict = None

    def forward(self, x, start_pos, freqs_cis_i, mask=None):
        if self.decoder is None:
            return x
        return x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def load_state_dict(self, state_dict):
        if self.decoder is not None and hasattr(self.decoder, "load_state_dict"):
            converted = convert_meta_to_hf(state_dict, self.head_dim)
            return self.decoder.load_state_dict(converted)
        self._state_dict = state_dict


# ============================================================================
# Base TTModelArgs Class
# ============================================================================

class TTModelArgs:
    """Minimal Polaris-oriented base class for model arguments."""

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,
    ):
        self.mesh_device = mesh_device
        self.instruct = instruct
        self.dummy_weights = dummy_weights
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.cache_hf_flag = cache_hf
        self.cached_hf_model = None

        # Default text model parameters
        self.dim = 2560
        self.n_layers = 32
        self.full_model_n_layers = 32
        self.n_heads = 32
        self.n_kv_heads = 8
        self.head_dim = self.dim // self.n_heads
        self.vocab_size = 128256
        self.hidden_dim = 10240
        self.rope_theta = 500000.0
        self.norm_eps = 1e-5
        self.sliding_window = 0

        # Device configuration
        self.num_devices = 1
        self.ccl_topology_value = None
        self.is_galaxy = False
        self.is_multichip = False
        self.cluster_shape = (1, 1)
        self.num_reduce_scatter_links = 1
        self.num_all_gather_links = 1

        # RoPE configuration
        self.rope_scaling = None
        self.rope_theta_local = None

        # Attention configuration
        self.attn_logit_softcapping = None
        self.final_logit_softcapping = None
        self.query_pre_attn_scalar = None
        self.attention_bias = False
        self.qkv_bias = False
        self.use_flash_attention = True
        self.use_scaled_dot_product_attention = True

        # KV cache sharding parameters
        self.min_kv_prefill_shard_seqlen = 128
        self.max_kv_prefill_shard_seqlen = 2048

        # CCL configuration
        self.ccl_dtype = ttnn.bfloat16

        # Matrix multiplication configuration
        self.MAX_QKV_MM_SEQ_LEN = 2048
        self.tile_size = 32

        # Architecture configuration
        try:
            self.arch_name = ttnn.get_arch_name()
        except Exception:
            self.arch_name = "wormhole_b0"
        self.max_grid_size = (8, 8)

        # Compute kernel configuration placeholders
        # self.compute_kernel_config_hifi2 = None
        # self.compute_kernel_config_hifi2_fp16 = None
        # self.compute_kernel_config_hifi4 = None

        # Compute kernel configurations
        self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # MLP configuration
        self.mlp_bias = False
        self.tie_word_embeddings = False

        # MoE configuration
        self.moe = False
        self.num_experts = 1
        self.num_experts_per_tok = 0
        self.expert_parallel_size = 1

        # Normalization configuration
        self.use_qk_norm = False
        self.use_post_attn_norm = False
        self.use_pre_attn_norm = True
        self.use_gated_mlp = True
        self.mlp_activation = "silu"
        self.rms_norm_add_unit_offset = False
        self.embed_scale = 1.0
        self.is_distributed_norm = False

        # Vision configuration
        self.vision_chunk_size = 896
        self.vision_max_num_chunks = 4
        self.vision_num_cross_attention_layers = 8
        self.vision_dim = 1152
        self.vision_mlp_ratio = 4
        self.vision_hidden_dim = self.vision_dim * self.vision_mlp_ratio
        self.vision_attn_n_heads = 16
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_n_layers = 27
        self.vision_patch_size = 14
        self.vision_in_channels = 3
        self.vision_dropout = 0.0
        self.mm_tokens_per_image = 256
        self.vision_act_layer = "gelu"
        self.vision_n_global_layers = 8

        # Device / model info
        self.device_name = "N150"
        self.base_model_name = "gemma-3-4b"
        self.model_name = "gemma-3-4b"
        self.is_multimodal = False

        # Paths
        self.CKPT_DIR = os.environ.get("HF_MODEL", "google/gemma-3-4b-it")

        # Tokenizer / HF config
        self.tokenizer = None
        self.trust_remote_code_hf = True
        self.model_config = {}
        self.hf_config = {}

        # Optimizations
        if optimizations is None:
            self.optimizations = ModelOptimizations()
        else:
            self.optimizations = optimizations

        # Warmup settings
        self.capped_warmup_seq_len = 8192
        self.trace_prefill_supported_seq_lens = []

        self._set_model_specific_params()
        self._compute_derived_values()

    def _compute_derived_values(self):
        if self.query_pre_attn_scalar is None:
            self.query_pre_attn_scalar = self.head_dim ** -0.5
        self.qkv_size = self.n_heads * self.head_dim + 2 * self.n_kv_heads * self.head_dim

    def _set_model_specific_params(self):
        pass

    def _set_params_from_dict(self, config):
        text_config = config.get("text_config", config)
        self.dim = text_config.get("hidden_size", self.dim)
        self.n_layers = text_config.get("num_hidden_layers", self.n_layers)
        self.n_heads = text_config.get("num_attention_heads", self.n_heads)
        self.n_kv_heads = text_config.get("num_key_value_heads", self.n_kv_heads)
        self.vocab_size = text_config.get("vocab_size", self.vocab_size)
        self.hidden_dim = text_config.get("intermediate_size", self.hidden_dim)
        self.rope_theta = text_config.get("rope_theta", self.rope_theta)
        self.norm_eps = text_config.get("rms_norm_eps", self.norm_eps)
        if self.n_heads > 0:
            self.head_dim = self.dim // self.n_heads

    def ccl_topology(self):
        return self.ccl_topology_value

    def create_tokenizer(self):
        return None

    def get_max_prefill_chunk_size(self):
        return 8192

    def can_enable_trace(self, seq_len, num_cached_tokens=0):
        """Check if trace can be enabled for a given sequence length."""
        return seq_len in self.trace_prefill_supported_seq_lens

    def get_attn_qkv_program_config(self, mode, seq_len=1, prefetcher=None):
        return None

    def get_attn_sdpa_decode_program_config(self, prefetcher=None):
        return None

    def reference_transformer(self, wrap=True):
        return None

    def weight_cache_path(self, dtype):
        return None

    def prepare_residual_tensor_prefill(self, x):
        return x

    def prepare_residual_tensor_decode(self, x):
        return x

    def get_model_config(self):
        """Return the model configuration dictionary."""
        if not self.model_config:
            self.model_config = self._build_default_model_config()
        return self.model_config

    def _build_default_model_config(self):
        """Build default model configuration for vision and text components."""
        config = {
            "LM_HEAD_OUTPUT_MEMCFG": ttnn.DRAM_MEMORY_CONFIG,
            "DECODERS_OPTIMIZATIONS": self.optimizations,
            # Vision MLP program configs (return None for basic compatibility)
            "IMAGE_MLP_FC_PROGCFG": lambda seq_len, max_seq_len: None,
            "IMAGE_MLP_PROJ_PROGCFG": lambda seq_len, max_seq_len: None,
            # Sharded norm configs (placeholders)
            "SHARDED_NORM_INPUT_MEMCFG": ttnn.DRAM_MEMORY_CONFIG,
            "SHARDED_NORM_PRGM_CFG": None,
            "SHARDED_NORM_OUTPUT_MEMCFG": ttnn.DRAM_MEMORY_CONFIG,
            "SHARDED_NORM_ATTN_PRGM_CFG": None,
            "SHARDED_ATTN_INPUT_MEMCFG": ttnn.DRAM_MEMORY_CONFIG,
        }
        return config
# ============================================================================
# ModelArgs Class (Gemma3-specific)
# ============================================================================

class ModelArgs(TTModelArgs):
    OP_KEYS = (
        "EMB_WEIGHTS",
        "MLP_WEIGHTS",
        "FF1_OUTPUT",
        "FF3_OUTPUT",
        "FF2_OUTPUT",
        "MLP_W_LAYOUT",
        "ATTN_WEIGHTS",
        "XQKV_MM_OUTPUT",
        "QKV_HEADS_OUTPUT",
        "QV_ROT_EMB_OUTPUT",
        "KV_UNPAD_OUTPUT",
        "QK_MM_OUTPUT",
        "QKV_MM_OUTPUT",
        "CONCAT_HEADS_OUTPUT",
        "ATTN_OUTPUT",
        "ATTN_W_LAYOUT",
        "DECODE_RESIDUAL",
        "OUTPUT_MM",
    )
    MAX_QKV_MM_SEQ_LEN = 2048

    def __init__(
        self,
        mesh_device,
        instruct=False,
        dummy_weights=False,
        max_batch_size=1,
        max_seq_len=1024 * 128,
        optimizations=None,
        cache_hf=False,
        enable_program_trace: bool = False,
    ):
        hf_model = os.environ.get("HF_MODEL", "")
        if hf_model and not os.path.isabs(hf_model):
            snapshot = ModelArgs._resolve_hf_snapshot(hf_model)
            if snapshot:
                logger.info(f"[Gemma3] Resolved HF model '{hf_model}' to snapshot: {snapshot}")
                os.environ["HF_MODEL"] = str(snapshot)

        self._enable_program_trace = enable_program_trace

        if enable_program_trace:
            self.force_fixed_decode_k_chunk = True
            self._gemma3_sdpa_decode_k_chunk_override = _GEMMA3_SDPA_DECODE_K_CHUNK_PROGRAM_TRACE

        super().__init__(
            mesh_device,
            instruct=instruct,
            dummy_weights=dummy_weights,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            optimizations=optimizations,
            cache_hf=cache_hf,
        )

        # Load HF config and update parameters
        self._set_hf_params(self.CKPT_DIR)

        # Recompute derived values after loading HF params
        self._compute_derived_values()

        # Gemma3-specific overrides
        self.attn_logit_softcapping = 50.0
        self.final_logit_softcapping = 30.0
        self.sliding_window = 4096
        self.rope_theta_local = 10000.0
        self.rms_norm_add_unit_offset = True
        self.embed_scale = self.dim ** 0.5

        if dummy_weights and self.tokenizer is None:
            self.tokenizer = self.create_tokenizer()

        self.use_qk_fused = False
        self.model_config["LM_HEAD_OUTPUT_MEMCFG"] = ttnn.DRAM_MEMORY_CONFIG
        self.padded_vocab_size = 262400
        self.device_sampling_max_per_device_vocab = 192 * 1024
        self.trace_prefill_supported_seq_lens = self.get_trace_prefill_supported_seq_lens()

        if enable_program_trace:
            self._relax_attention_ops_for_program_trace()

        if not enable_program_trace:
            self._force_sdpa_decode_hifi2_na()

    # ========================================================================
    # Hardware Optimization Methods
    # ========================================================================

    def _relax_attention_ops_for_program_trace(self):
        """Lower L1 for prefill+decode attention under program tracing."""
        trace_groups = (
            OpGroup.LI_QKV_PREFILL,
            OpGroup.LI_O_PREFILL,
            OpGroup.SDPA_PREFILL,
            OpGroup.LI_QKV_DECODE,
            OpGroup.LI_O_DECODE,
            OpGroup.SDPA_DECODE,
        )
        for decoder_id, conf in list(self.optimizations.decoder_optimizations.items()):
            tensor_precision = {k: v for k, v in conf.tensor_dtype_settings.items() if v is not None}
            op_fidelity = dict(conf.op_fidelity_settings)
            for grp in trace_groups:
                if grp in op_fidelity:
                    op_fidelity[grp] = MathFidelitySetting.HIFI2_FP16
            fixed_conf = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
            fixed_conf.__name__ = getattr(conf, "__name__", fixed_conf.__name__)
            self.optimizations.set_decoder_conf(decoder_id, fixed_conf)
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations

    def _force_sdpa_decode_hifi2_na(self):
        """Gemma3 decode SDPA requires no-accumulation HiFi2 for correctness."""
        for decoder_id, conf in list(self.optimizations.decoder_optimizations.items()):
            tensor_precision = {k: v for k, v in conf.tensor_dtype_settings.items() if v is not None}
            op_fidelity = dict(conf.op_fidelity_settings)
            op_fidelity[OpGroup.SDPA_DECODE] = MathFidelitySetting.HIFI2_NA
            fixed_conf = ModelOptimizations({"TensorPrecision": tensor_precision, "OpFidelity": op_fidelity})
            fixed_conf.__name__ = getattr(conf, "__name__", fixed_conf.__name__)
            self.optimizations.set_decoder_conf(decoder_id, fixed_conf)
        self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations

    # ========================================================================
    # Static Utility Methods
    # ========================================================================

    @staticmethod
    def _resolve_hf_snapshot(hf_model_name):
        hf_cache = os.path.normpath(
            os.environ.get("HF_HUB_CACHE")
            or os.path.join(
                os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub"
            )
        )
        model_slug = "models--" + hf_model_name.replace("/", "--")
        snapshots_dir = os.path.normpath(os.path.join(hf_cache, model_slug, "snapshots"))

        if not snapshots_dir.startswith(hf_cache + os.sep):
            return None

        if not os.path.isdir(snapshots_dir):
            return None

        snaps = [
            os.path.join(snapshots_dir, s)
            for s in os.listdir(snapshots_dir)
            if os.path.isdir(os.path.join(snapshots_dir, s))
        ]
        return max(snaps, key=os.path.getmtime) if snaps else None

    # ========================================================================
    # Configuration Methods
    # ========================================================================

    def get_max_prefill_chunk_size(self):
        model_overrides = {
            "gemma-3-4b": {"P150": 128},
            "medgemma-4b": {"P150": 128},
            "gemma-3-27b": {"P150": 128},
            "medgemma-27b": {"P150": 128},
        }
        model_name = self.base_model_name
        device_name = self.device_name
        if model_name in model_overrides and device_name in model_overrides[model_name]:
            return model_overrides[model_name][device_name] * 1024
        return super().get_max_prefill_chunk_size()

    def get_attn_qkv_program_config(self, mode, seq_len: int = 1, prefetcher=None):
        if self._enable_program_trace and mode == Mode.PREFILL and seq_len > 128:
            return ttnn.MinimalMatmulConfig( # type: ignore[attr-defined]
                M_block_size=4,
                K_block_size=4,
                N_block_size=4,
                compute_with_storage_grid_size=(
                    ttnn.CoreCoord(8, 10) if is_blackhole() else ttnn.CoreCoord(8, 8)
                ),
            )
        return super().get_attn_qkv_program_config(mode, seq_len, prefetcher)

    def get_attn_sdpa_decode_program_config(self, prefetcher=None):
        force_fixed_k_chunk = getattr(self, "force_fixed_decode_k_chunk", False)
        if not force_fixed_k_chunk:
            return super().get_attn_sdpa_decode_program_config(prefetcher)

        override = getattr(self, "_gemma3_sdpa_decode_k_chunk_override", None)
        k_chunk_tokens = _GEMMA3_SDPA_DECODE_K_CHUNK_DEFAULT if override is None else int(override)

        if prefetcher is not None and hasattr(prefetcher, "all_worker_cores_range_set"):
            sdpa_grid_size = (8, 8)
            start_core = ttnn.CoreCoord(1, 0)
            num_sdpa_cores = sdpa_grid_size[0] * sdpa_grid_size[1]
            return ttnn.SDPAProgramConfig( # type: ignore[attr-defined]
                compute_with_storage_grid_size=sdpa_grid_size,
                sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids( # type: ignore[attr-defined]
                    start_core,
                    num_sdpa_cores,
                    prefetcher.all_worker_cores_range_set,
                    row_wise=True,
                ),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=k_chunk_tokens,
            )

        return ttnn.SDPAProgramConfig( # type: ignore[attr-defined]
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=0,
            k_chunk_size=k_chunk_tokens,
        )

    def get_warmup_prefill_supported_seq_lens(self):
        default_value = self.capped_warmup_seq_len
        model_specific_ceil_warmup_lengths = {
            "gemma-3-4b": 2048,
            "gemma-3-27b": 2048,
        }
        max_seq_len_to_warmup = model_specific_ceil_warmup_lengths.get(
            self.base_model_name, default_value
        )
        if max_seq_len_to_warmup > self.capped_warmup_seq_len:
            max_seq_len_to_warmup = self.capped_warmup_seq_len

        to_warmup_seq_lens = calculate_prefill_warmup_seq_lens(
            max_seq_len_to_warmup, self.trace_prefill_supported_seq_lens
        )
        return self.filter_warmup_seq_lens(to_warmup_seq_lens)

    def filter_warmup_seq_lens(self, to_warmup_seq_lens):
        return to_warmup_seq_lens

    def get_trace_prefill_supported_seq_lens(self):
        default_supported_seq_lens: Dict[str, List[int]] = {
            "N150": [],
            "N300": [],
            "T3K": [],
            "TG": [],
            "P150": [],
        }
        model_specific_supported_seq_lens: Dict[str, Dict[str, List[int]]] = {}

        result = model_specific_supported_seq_lens.get(self.base_model_name, {}).get(
            self.device_name, default_supported_seq_lens.get(self.device_name)
        )
        if result is not None:
            return cap_seq_lens_to_max_prefill_chunk_size(result, self.capped_warmup_seq_len)
        return []

    def _set_model_specific_params(self):
        self.rms_norm_add_unit_offset = True
        self.embed_scale = self.dim ** 0.5

    def _set_vision_params(self, vision_config):
        self.vision_chunk_size = vision_config.get("vision_chunk_size", 896)
        self.vision_max_num_chunks = vision_config.get("vision_max_num_chunks", 4)
        self.vision_num_cross_attention_layers = vision_config.get(
            "vision_num_cross_attention_layers", 8
        )
        self.vision_dim = vision_config.get("hidden_size", 1152)
        intermediate_size = vision_config.get("intermediate_size", self.vision_dim * 4)
        self.vision_mlp_ratio = intermediate_size // self.vision_dim
        self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
        self.vision_attn_n_heads = vision_config.get("num_attention_heads", 16)
        self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
        self.vision_n_layers = vision_config.get("num_hidden_layers", 27)
        self.vision_patch_size = vision_config.get("patch_size", 14)
        self.vision_in_channels = vision_config.get("num_channels", 3)
        self.vision_dropout = vision_config.get("attention_dropout", 0.0)
        self.mm_tokens_per_image = vision_config.get("mm_tokens_per_image", 256)

        act_layer = vision_config.get("act_layer", "gelu").lower()
        self.vision_act_layer = {  # type: ignore[assignment]
            "gelu": ttnn.UnaryOpType.GELU, # type: ignore[attr-defined]
            "relu": ttnn.UnaryOpType.RELU, # type: ignore[attr-defined]
            "silu": ttnn.UnaryOpType.SILU, # type: ignore[attr-defined]
        }.get(act_layer, ttnn.UnaryOpType.GELU) # type: ignore[attr-defined]

        self.vision_n_global_layers = vision_config.get("n_global_layers", 8)
        self.is_multimodal = True

    def _set_hf_params(self, checkpoint_dir):
        def merge_vision_config(base_config):
            vision_config = base_config.get("vision_config", {})
            vision_config.update(
                {k: v for k, v in base_config.items() if k not in ["text_config", "vision_config"]}
            )
            return vision_config

        config_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.hf_config = json.load(f)
        else:
            self.hf_config = {}
            logger.warning(f"Config file not found at {config_path}, using defaults")

        # Store full model layers before potentially changing n_layers
        if "text_config" in self.hf_config:
            self.full_model_n_layers = self.hf_config["text_config"].get(
                "num_hidden_layers", self.full_model_n_layers
            )
        elif "num_hidden_layers" in self.hf_config:
            self.full_model_n_layers = self.hf_config.get(
                "num_hidden_layers", self.full_model_n_layers
            )

        if "text_config" in self.hf_config or "vision_config" in self.hf_config:
            self._set_params_from_dict(self.hf_config)
            if "vision_config" in self.hf_config:
                merged_vision_config = merge_vision_config(self.hf_config)
                self._set_vision_params(merged_vision_config)
        else:
            self._set_params_from_dict(self.hf_config)

    def get_state_dict_prefix(self, module_name, layer_num, is_vision=False):
        text_prefix = "model.vision_tower.vision_model.encoder." if is_vision else ""
        layer_prefix = f"layers.{layer_num}." if layer_num is not None else ""

        module_map = {
            "MLP": "feed_forward",
            "Attention": "attention",
            "TransformerBlock": "",
            "": "",
        }
        vision_module_map = {
            "MLP": "mlp.",
            "Attention": "self_attn.",
            "TransformerBlock": "",
            "": "",
        }
        active_module_map = vision_module_map if is_vision else module_map
        return text_prefix + layer_prefix + active_module_map[module_name]

    # ========================================================================
    # Dummy Model and State Dict Loading
    # ========================================================================
    def _gemma_dummy_hf_model(self):
        logger.info("Gemma3 ModelArgs: building dummy model (simulation mode)")

        class DummyModel:
            def __init__(self, dim, n_layers, vocab_size, n_heads, n_kv_heads, hidden_dim,
                         vision_dim, vision_n_layers, vision_attn_n_heads, vision_hidden_dim,
                         vision_patch_size, vision_in_channels, vision_chunk_size):
                self.dim = dim
                self.n_layers = n_layers
                self.vocab_size = vocab_size
                self.n_heads = n_heads
                self.n_kv_heads = n_kv_heads
                self.hidden_dim = hidden_dim
                self.head_dim = dim // n_heads
                
                # Vision parameters
                self.vision_dim = vision_dim
                self.vision_n_layers = vision_n_layers
                self.vision_attn_n_heads = vision_attn_n_heads
                self.vision_hidden_dim = vision_hidden_dim
                self.vision_patch_size = vision_patch_size
                self.vision_in_channels = vision_in_channels
                self.vision_chunk_size = vision_chunk_size
                self.vision_head_dim = vision_dim // vision_attn_n_heads
                self.num_patches = (vision_chunk_size // vision_patch_size) ** 2

            def state_dict(self):
                state = {}
                
                # ============================================================
                # Text model weights
                # ============================================================
                state["model.embed_tokens.weight"] = (
                    np.random.randn(self.vocab_size, self.dim).astype(np.float32) * 0.02
                )
                for i in range(self.n_layers):
                    prefix = f"model.layers.{i}."
                    q_dim = self.n_heads * self.head_dim
                    kv_dim = self.n_kv_heads * self.head_dim

                    state[f"{prefix}self_attn.q_proj.weight"] = (
                        np.random.randn(q_dim, self.dim).astype(np.float32) * 0.02
                    )
                    state[f"{prefix}self_attn.k_proj.weight"] = (
                        np.random.randn(kv_dim, self.dim).astype(np.float32) * 0.02
                    )
                    state[f"{prefix}self_attn.v_proj.weight"] = (
                        np.random.randn(kv_dim, self.dim).astype(np.float32) * 0.02
                    )
                    state[f"{prefix}self_attn.o_proj.weight"] = (
                        np.random.randn(self.dim, self.dim).astype(np.float32) * 0.02
                    )

                    state[f"{prefix}mlp.gate_proj.weight"] = (
                        np.random.randn(self.hidden_dim, self.dim).astype(np.float32) * 0.02
                    )
                    state[f"{prefix}mlp.up_proj.weight"] = (
                        np.random.randn(self.hidden_dim, self.dim).astype(np.float32) * 0.02
                    )
                    state[f"{prefix}mlp.down_proj.weight"] = (
                        np.random.randn(self.dim, self.hidden_dim).astype(np.float32) * 0.02
                    )

                    state[f"{prefix}input_layernorm.weight"] = np.ones(self.dim, dtype=np.float32)
                    state[f"{prefix}post_attention_layernorm.weight"] = np.ones(
                        self.dim, dtype=np.float32
                    )

                state["model.norm.weight"] = np.ones(self.dim, dtype=np.float32)
                state["lm_head.weight"] = (
                    np.random.randn(self.vocab_size, self.dim).astype(np.float32) * 0.02
                )

                # ============================================================
                # Vision model weights
                # ============================================================
                vision_prefix = "model.vision_tower.vision_model."
                
                # Patch embedding (Conv2D as linear)
                patch_input_dim = self.vision_in_channels * self.vision_patch_size * self.vision_patch_size
                state[f"{vision_prefix}embeddings.patch_embedding._linear.weight"] = (
                    np.random.randn(self.vision_dim, patch_input_dim).astype(np.float32) * 0.02
                )
                state[f"{vision_prefix}embeddings.patch_embedding._linear.bias"] = (
                    np.zeros(self.vision_dim, dtype=np.float32)
                )
                
                # Position embedding
                state[f"{vision_prefix}embeddings.position_embedding.positional_embedding"] = (
                    np.random.randn(self.num_patches, self.vision_dim).astype(np.float32) * 0.02
                )
                
                # Vision encoder layers
                for i in range(self.vision_n_layers):
                    layer_prefix = f"{vision_prefix}encoder.layers.{i}."
                    
                    # Layer norms
                    state[f"{layer_prefix}ln_1.weight"] = np.ones(self.vision_dim, dtype=np.float32)
                    state[f"{layer_prefix}ln_1.bias"] = np.zeros(self.vision_dim, dtype=np.float32)
                    state[f"{layer_prefix}ln_2.weight"] = np.ones(self.vision_dim, dtype=np.float32)
                    state[f"{layer_prefix}ln_2.bias"] = np.zeros(self.vision_dim, dtype=np.float32)
                    
                    # Attention weights (Q, K, V, O)
                    attn_prefix = f"{layer_prefix}attn."
                    state[f"{attn_prefix}wq.weight"] = (
                        np.random.randn(self.vision_dim, self.vision_dim).astype(np.float32) * 0.02
                    )
                    state[f"{attn_prefix}wk.weight"] = (
                        np.random.randn(self.vision_dim, self.vision_dim).astype(np.float32) * 0.02
                    )
                    state[f"{attn_prefix}wv.weight"] = (
                        np.random.randn(self.vision_dim, self.vision_dim).astype(np.float32) * 0.02
                    )
                    state[f"{attn_prefix}wo.weight"] = (
                        np.random.randn(self.vision_dim, self.vision_dim).astype(np.float32) * 0.02
                    )
                    
                    # Attention biases
                    state[f"{attn_prefix}wq.bias"] = np.zeros(self.vision_dim, dtype=np.float32)
                    state[f"{attn_prefix}wk.bias"] = np.zeros(self.vision_dim, dtype=np.float32)
                    state[f"{attn_prefix}wv.bias"] = np.zeros(self.vision_dim, dtype=np.float32)
                    state[f"{attn_prefix}wo.bias"] = np.zeros(self.vision_dim, dtype=np.float32)
                    
                    # MLP weights
                    mlp_prefix = f"{layer_prefix}mlp."
                    state[f"{mlp_prefix}c_fc.weight"] = (
                        np.random.randn(self.vision_hidden_dim, self.vision_dim).astype(np.float32) * 0.02
                    )
                    state[f"{mlp_prefix}c_fc.bias"] = np.zeros(self.vision_hidden_dim, dtype=np.float32)
                    state[f"{mlp_prefix}c_proj.weight"] = (
                        np.random.randn(self.vision_dim, self.vision_hidden_dim).astype(np.float32) * 0.02
                    )
                    state[f"{mlp_prefix}c_proj.bias"] = np.zeros(self.vision_dim, dtype=np.float32)
                
                # Post layer norm
                state[f"{vision_prefix}ln_post.weight"] = np.ones(self.vision_dim, dtype=np.float32)
                state[f"{vision_prefix}ln_post.bias"] = np.zeros(self.vision_dim, dtype=np.float32)
                
                # ============================================================
                # Multi-modal projector weights
                # ============================================================
                mm_prefix = "model.multi_modal_projector"
                state[f"{mm_prefix}.mm_input_projection_weight"] = (
                    np.random.randn(self.vision_dim, self.vision_dim).astype(np.float32) * 0.02
                )
                state[f"{mm_prefix}.mm_soft_emb_norm.weight"] = np.ones(self.vision_dim, dtype=np.float32)

                return state

        model = DummyModel(
            dim=self.dim,
            n_layers=self.n_layers,
            vocab_size=self.vocab_size,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            hidden_dim=self.hidden_dim,
            # Vision parameters
            vision_dim=self.vision_dim,
            vision_n_layers=self.vision_n_layers,
            vision_attn_n_heads=self.vision_attn_n_heads,
            vision_hidden_dim=self.vision_hidden_dim,
            vision_patch_size=self.vision_patch_size,
            vision_in_channels=self.vision_in_channels,
            vision_chunk_size=self.vision_chunk_size,
        )
        gc.collect()
        return model
    # def _gemma_dummy_hf_model(self):
    #     logger.info("Gemma3 ModelArgs: building dummy model (simulation mode)")

    #     class DummyModel:
    #         def __init__(self, dim, n_layers, vocab_size, n_heads, n_kv_heads, hidden_dim):
    #             self.dim = dim
    #             self.n_layers = n_layers
    #             self.vocab_size = vocab_size
    #             self.n_heads = n_heads
    #             self.n_kv_heads = n_kv_heads
    #             self.hidden_dim = hidden_dim
    #             self.head_dim = dim // n_heads

    #         def state_dict(self):
    #             state = {}
    #             state["model.embed_tokens.weight"] = (
    #                 np.random.randn(self.vocab_size, self.dim).astype(np.float32) * 0.02
    #             )
    #             for i in range(self.n_layers):
    #                 prefix = f"model.layers.{i}."
    #                 q_dim = self.n_heads * self.head_dim
    #                 kv_dim = self.n_kv_heads * self.head_dim

    #                 state[f"{prefix}self_attn.q_proj.weight"] = (
    #                     np.random.randn(q_dim, self.dim).astype(np.float32) * 0.02
    #                 )
    #                 state[f"{prefix}self_attn.k_proj.weight"] = (
    #                     np.random.randn(kv_dim, self.dim).astype(np.float32) * 0.02
    #                 )
    #                 state[f"{prefix}self_attn.v_proj.weight"] = (
    #                     np.random.randn(kv_dim, self.dim).astype(np.float32) * 0.02
    #                 )
    #                 state[f"{prefix}self_attn.o_proj.weight"] = (
    #                     np.random.randn(self.dim, self.dim).astype(np.float32) * 0.02
    #                 )
    #                 state[f"{prefix}mlp.gate_proj.weight"] = (
    #                     np.random.randn(self.hidden_dim, self.dim).astype(np.float32) * 0.02
    #                 )
    #                 state[f"{prefix}mlp.up_proj.weight"] = (
    #                     np.random.randn(self.hidden_dim, self.dim).astype(np.float32) * 0.02
    #                 )
    #                 state[f"{prefix}mlp.down_proj.weight"] = (
    #                     np.random.randn(self.dim, self.hidden_dim).astype(np.float32) * 0.02
    #                 )
    #                 state[f"{prefix}input_layernorm.weight"] = np.ones(self.dim, dtype=np.float32)
    #                 state[f"{prefix}post_attention_layernorm.weight"] = np.ones(
    #                     self.dim, dtype=np.float32
    #                 )

    #             state["model.norm.weight"] = np.ones(self.dim, dtype=np.float32)
    #             state["lm_head.weight"] = (
    #                 np.random.randn(self.vocab_size, self.dim).astype(np.float32) * 0.02
    #             )
    #             return state

    #     model = DummyModel(
    #         dim=self.dim,
    #         n_layers=self.n_layers,
    #         vocab_size=self.vocab_size,
    #         n_heads=self.n_heads,
    #         n_kv_heads=self.n_kv_heads,
    #         hidden_dim=self.hidden_dim,
    #     )
    #     gc.collect()
    #     return model

    def load_state_dict(self):
        if self.dummy_weights:
            logger.info("Gemma3 ModelArgs: using dummy_weights path; NOT loading checkpoints")
        else:
            logger.info("Gemma3 ModelArgs: simulation mode - using dummy weights")

        model = self._gemma_dummy_hf_model()
        state_dict = model.state_dict()
        del model
        gc.collect()

        # Apply appropriate conversion based on model type
        if self.is_multimodal:
            state_dict = convert_vision_hf_to_meta(state_dict, self.head_dim)
        else:
            state_dict = standardize_hf_keys(state_dict)
            state_dict = convert_hf_to_meta(state_dict, self.head_dim)

        # Remove layers beyond n_layers (when using a subset of the full model)
        if self.n_layers < self.full_model_n_layers:
            keys_to_remove = []
            for k in state_dict.keys():
                for layer_idx in range(self.n_layers, self.full_model_n_layers):
                    if f"layers.{layer_idx}." in k:
                        keys_to_remove.append(k)
                        break
            for k in keys_to_remove:
                state_dict.pop(k)

        return state_dict

    # ========================================================================
    # Reference Model Methods (stubs for simulation)
    # ========================================================================

    def reference_vision_multi_modal(self):
        model = self.reference_vision_transformer(wrap=False)
        if model is None:
            return None
        return getattr(model, "multi_modal_projector", None)

    def reference_vision_rms_norm(self):
        model = self.reference_vision_transformer(wrap=False)
        if model is None:
            return None
        projector = getattr(model, "multi_modal_projector", None)
        if projector is None:
            return None
        return getattr(projector, "mm_soft_emb_norm", None)

    def reference_rms_norm(self, i=0):
        model = self.reference_transformer(wrap=False)
        if model is None:
            return None
        try:
            layer = model.model.layers[i].self_attn.q_norm
            layer._load_state_dict = layer.load_state_dict
            layer.load_state_dict = lambda x: layer._load_state_dict(
                convert_meta_to_hf(x, self.head_dim)
            )
            return layer
        except (AttributeError, IndexError):
            return None

    def reference_rms_norm_text(self):
        model = self.reference_transformer(wrap=False)
        if model is None:
            return None
        try:
            layer = model.model.norm
            layer._load_state_dict = layer.load_state_dict
            layer.load_state_dict = lambda x: layer._load_state_dict(
                convert_meta_to_hf(x, self.head_dim)
            )
            return layer
        except AttributeError:
            return None

    def get_hf_model_cls(self):
        return None  # Simulation mode

    def reference_mlp(self):
        model = self.reference_transformer(wrap=False)
        if model is None:
            return None
        try:
            layer = model.model.layers[0].mlp
            layer._load_state_dict = layer.load_state_dict
            layer.load_state_dict = lambda x: layer._load_state_dict(
                convert_meta_to_hf(x, self.head_dim)
            )
            return layer
        except (AttributeError, IndexError):
            return None

    def reference_vision_transformer(self, wrap=True, load_checkpoint=False):
        # Simulation mode - return None
        logger.info("Simulation mode: reference_vision_transformer returns None")
        return None

    def reference_transformer(self, wrap=True):
        return self.reference_vision_transformer(wrap=wrap)

    def reference_gemma_model(self):
        model = self.reference_vision_transformer(wrap=False)
        if model is None:
            return None
        model._load_state_dict = model.load_state_dict
        model.load_state_dict = lambda x: model._load_state_dict(
            convert_vision_meta_to_hf(x, self.head_dim)
        )
        return model

    def reference_vision_model(self):
        model = self.reference_vision_transformer(wrap=False)
        if model is None:
            return None
        try:
            return model.vision_tower.vision_model
        except AttributeError:
            return None

    def reference_vision_mlp(self):
        return None

    def reference_siglip_patch_embed(self):
        return None

    def reference_vision_pos_embedding(self):
        return None

    def reference_vision_embedding(self):
        return None

    def reference_vision_layernorm(self, layer_name="layer_norm1"):
        return None

    def reference_vision_attention(self):
        return None

    def reference_vision_encoder_block(self):
        return None

    def reference_vision_encoder(self):
        return None

    def reference_decoder(self, i=0):
        return HfGemmaDecoderWrapper(None, self.head_dim, None, None)

    def reference_decoder_text(self, i=0):
        return HfDecoderWrapper(None, self.head_dim, None, None)

    def reference_attention(self, rope_embeddings="global"):
        return HfAttentionWrapper(None, self.head_dim, None)


# ============================================================================
# Test
# ============================================================================

if __name__ == "__main__":
    print("Testing Polaris ModelArgs...")
    mesh_device = None

    args = ModelArgs(
        mesh_device=mesh_device,
        instruct=True,
        dummy_weights=True,
        max_batch_size=1,
        max_seq_len=4096,
    )

    print(f"Model name: {args.model_name}")
    print(f"Dimensions: {args.dim}")
    print(f"Layers: {args.n_layers}")
    print(f"Heads: {args.n_heads}")
    print(f"Is multimodal: {args.is_multimodal}")

    state_dict = args.load_state_dict()
    print(f"State dict keys (first 10): {list(state_dict.keys())[:10]}")
    print(f"Total keys: {len(state_dict)}")

    print("\nPolaris ModelArgs test completed successfully!")