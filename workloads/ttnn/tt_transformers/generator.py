#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Generator module for Polaris - Pure NumPy + TTNN implementation (no PyTorch)
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union, Iterator, DefaultDict

import numpy as np

# Use ttsim.front.ttnn for Polaris (simulation environment)
import ttsim.front.ttnn as ttnn

# ============================================================================
# Constants
# ============================================================================
MAX_BATCHED_PREFILL_SEQ_LEN = 128 * 1024
SUPPORTED_PREFILL_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
DECODE_PAGE_TABLE_INPUT_IDX = 3

# ============================================================================
# Helper functions for tensor conversion (NO TORCH)
# ============================================================================
# Fix for line 57 - remove the unreachable code or restructure
# The issue is likely in tensor_to_numpy function. Here's the fix:

def tensor_to_numpy(tensor: Any) -> np.ndarray:
    """Convert ttnn tensor to numpy array - NO TORCH."""
    if tensor is None:
        return np.array([])
    if isinstance(tensor, np.ndarray):
        return tensor
    # Try ttnn's native numpy conversion
    if hasattr(ttnn, 'to_numpy'):
        result = ttnn.to_numpy(tensor)
        return np.array(result) if not isinstance(result, np.ndarray) else result
    # Fallback: try to extract data directly
    if hasattr(tensor, 'numpy'):
        result = tensor.numpy()
        return np.array(result) if not isinstance(result, np.ndarray) else result
    if hasattr(tensor, 'data'):
        return np.array(tensor.data)
    # Last resort
    return np.array(tensor)

def numpy_to_ttnn(
    arr: np.ndarray | None,
    device: Any = None,
    dtype: Any = None,
    layout: Any = None,
    mesh_mapper: Any = None
) -> Any:
    """Convert numpy array to ttnn tensor - NO TORCH."""
    if arr is None:
        return None
    if hasattr(ttnn, 'from_numpy'):
        kwargs: Dict[str, Any] = {}
        if device is not None:
            kwargs['device'] = device
        if dtype is not None:
            kwargs['dtype'] = dtype
        if layout is not None:
            kwargs['layout'] = layout
        if mesh_mapper is not None:
            kwargs['mesh_mapper'] = mesh_mapper
        return ttnn.from_numpy(arr, **kwargs)
    # Fallback for simulation
    return arr


# ============================================================================
# Utility Functions
# ============================================================================
def max_prefill_chunk_size_cutoff(sequence_length: int, max_prefill_chunk_size: int) -> bool:
    return sequence_length > max_prefill_chunk_size


def _deepseek_kvdbg_enabled() -> bool:
    return os.getenv("DEEPSEEK_KVDBG", "").lower() in ("1", "true", "yes", "y")


def _get_max_blocks_prefill(kv_cache: Any) -> int:
    """Get max blocks from KV cache."""
    if kv_cache is None:
        return 128
    first_cache_tensor = kv_cache[0][0]
    if hasattr(first_cache_tensor, 'shape'):
        return int(first_cache_tensor.shape[0])
    return 128


def _pad_or_create_page_table(
    table: Optional[np.ndarray],
    target_blocks: int
) -> np.ndarray:
    """Pad or create page table using numpy."""
    aligned_blocks = ((target_blocks + 7) // 8) * 8

    if table is not None:
        num_pad = aligned_blocks - table.shape[1]
        if num_pad > 0:
            padding = np.full((table.shape[0], num_pad), -1, dtype=np.int32)
            return np.concatenate([table, padding], axis=-1)
        return table

    return np.full((1, aligned_blocks), -1, dtype=np.int32)


# ============================================================================
# Mode Enum
# ============================================================================
class Mode:
    PREFILL = "prefill"
    DECODE = "decode"


# ============================================================================
# Sampling Parameters
# ============================================================================
class SamplingParams:
    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k


# ============================================================================
# Result Classes
# ============================================================================
class TokenResult:
    def __init__(self, token: int, text: str) -> None:
        self.token = token
        self.text = text


class StopReason:
    end_of_turn = "end_of_turn"
    end_of_message = "end_of_message"
    out_of_tokens = "out_of_tokens"


class CompletionMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class LogProbsResult:
    def __init__(
        self,
        topk_logprobs: Optional[np.ndarray] = None,
        topk_indices: Optional[np.ndarray] = None
    ) -> None:
        self.topk_logprobs = topk_logprobs
        self.topk_indices = topk_indices
        self.topk_logprobs_host: Optional[np.ndarray] = None
        self.topk_indices_host: Optional[np.ndarray] = None

    def cpu(self, blocking: bool = True) -> "LogProbsResult":
        return self

    def extract_user(self, idx: int) -> Optional[float]:
        if self.topk_logprobs is not None and idx < len(self.topk_logprobs.flatten()):
            return float(self.topk_logprobs.flatten()[idx])
        return None


# ============================================================================
# Common Functions
# ============================================================================
def copy_host_to_device(
    host_tensors: Union[List[Any], Tuple[Any, ...]],
    device_tensors: Optional[Union[List[Any], Tuple[Any, ...]]] = None,
    mesh_device: Any = None
) -> List[Any]:
    """Copy host tensors to device tensors."""
    if device_tensors is None:
        assert mesh_device is not None, "mesh_device required when device_tensors is None"
        ret: List[Any] = []
        for ht in host_tensors:
            if ht is None:
                ret.append(None)
            else:
                on_device = numpy_to_ttnn(ht, device=mesh_device)
                ret.append(on_device)
        return ret
    else:
        device_list = list(device_tensors)
        for i in range(len(host_tensors)):
            if host_tensors[i] is None:
                continue
            if hasattr(ttnn, 'copy_host_to_device_tensor'):
                ttnn.copy_host_to_device_tensor(host_tensors[i], device_list[i])
        return device_list


def get_block_size(kv_cache: Any) -> int:
    """Get block size from KV cache."""
    if kv_cache is None:
        return 64
    if isinstance(kv_cache, list) and len(kv_cache) > 0:
        if isinstance(kv_cache[0], list) and len(kv_cache[0]) > 0:
            if hasattr(kv_cache[0][0], 'shape'):
                return int(kv_cache[0][0].shape[2])
        elif hasattr(kv_cache[0], 'shape'):
            return int(kv_cache[0].shape[2])
    if hasattr(kv_cache, 'block_size'):
        return int(kv_cache.block_size)
    return 64


def get_max_prefill_chunk_size(seq_len: int, max_prefill_seq_len: int) -> int:
    """Determine largest chunk size that divides seq_len."""
    MIN_CHUNK_SIZE = 2048
    if seq_len <= 0 or max_prefill_seq_len <= 0:
        raise ValueError("seq_len and max_prefill_seq_len must be positive")
    if seq_len % MIN_CHUNK_SIZE != 0:
        raise ValueError(f"seq_len ({seq_len}) must be multiple of {MIN_CHUNK_SIZE}")
    if max_prefill_seq_len % MIN_CHUNK_SIZE != 0:
        raise ValueError(f"max_prefill_seq_len must be multiple of {MIN_CHUNK_SIZE}")

    max_possible_chunk = min(max_prefill_seq_len, seq_len)
    for chunk_size in range(max_possible_chunk, 0, -MIN_CHUNK_SIZE):
        if seq_len % chunk_size == 0:
            return chunk_size
    raise ValueError("No valid chunk size found")


def get_padded_prefill_len(seq_len: int) -> int:
    """Get padded prefill length."""
    if seq_len <= 128:
        return 128
    if seq_len <= 1024:
        return 1024
    # Return next power of 2
    return 2 ** (seq_len - 1).bit_length()


def num_blocks_in_seq(seq_len: int, block_size: int) -> int:
    """Calculate number of blocks needed for sequence."""
    return math.ceil(seq_len / block_size)


def sample_top_p(probs: np.ndarray, top_p: float) -> np.ndarray:
    """Sample using top-p (nucleus) sampling."""
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    probs_sort_idx = np.argsort(probs, axis=-1)[:, ::-1]
    probs_sort = np.take_along_axis(probs, probs_sort_idx, axis=-1)
    probs_sum = np.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > top_p
    probs_sort[mask] = 0.0
    probs_sort = probs_sort / (probs_sort.sum(axis=-1, keepdims=True) + 1e-10)

    cumsum = np.cumsum(probs_sort, axis=-1)
    random_vals = np.random.rand(probs_sort.shape[0], 1)
    next_token_idx = np.argmax(cumsum >= random_vals, axis=-1, keepdims=True)
    next_token = np.take_along_axis(probs_sort_idx, next_token_idx, axis=-1)
    return next_token


def broadcast_sampling_params(sampling_params: Any, idx: int, slot_len: int) -> Any:
    """Broadcast sampling params stub."""
    return sampling_params


def chunk_sampling_params(sampling_params: Any, num_chunks: int) -> List[Any]:
    """Chunk sampling params."""
    return [sampling_params] * num_chunks


def format_sampling_params(sampling_params: Any, batch_size: int) -> Any:
    """Format sampling params."""
    return sampling_params


def reformat_logprobs(log_probs: Any, batch_size: int) -> Any:
    """Reformat log probs."""
    return log_probs


def create_vision_mask(input_ids: Any, image_token_id: int) -> Optional[np.ndarray]:
    """Create vision mask stub."""
    return None


def encode_content(content: Any, vision_images: Any, image_token: Any) -> Any:
    """Encode content stub."""
    return content


def extract_images_from_messages(messages: Any) -> Optional[List[Any]]:
    """Extract images stub."""
    return None


# ============================================================================
# Mixin stubs
# ============================================================================
class ModelCapabilitiesMixin:
    """Stub for model capabilities mixin."""
    model_capabilities: Dict[str, bool] = {
        "supports_prefix_caching": True,
    }


class WarmupForwardMixin:
    """Stub for warmup forward mixin."""
    pass


# ============================================================================
# Logger
# ============================================================================
class LoggerStub:
    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        print(f"[INFO] {msg}")

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        print(f"[WARNING] {msg}")

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        pass  # Suppress debug by default


logger = LoggerStub()


# ============================================================================
# Main Generator Class
# ============================================================================
class Generator(ModelCapabilitiesMixin, WarmupForwardMixin):
    """
    Generator class for LLM inference - Pure NumPy + TTNN (no PyTorch).
    """

    model_capabilities: Dict[str, bool] = {
        "supports_prefix_caching": True,
    }

    def __init__(
        self,
        model: Any,
        model_args: Any,
        mesh_device: Any,
        processor: Any = None,
        tokenizer: Any = None
    ) -> None:
        """Initialize Generator."""
        self.model = model if isinstance(model, list) else [model]
        self.model_args = model_args if isinstance(model_args, list) else [model_args]
        self.mesh_device = mesh_device
        self.processor = processor
        self.tokenizer = tokenizer
        self.data_parallel = len(self.model)
        self.prev_page_table: Optional[Tuple[np.ndarray, ...]] = None

        # Initialize trace caches with proper types
        self.trace_id_prefill: DefaultDict[str, Optional[str]] = defaultdict(lambda: None)
        self.trace_inputs_prefill: DefaultDict[str, Optional[Tuple[Any, ...]]] = defaultdict(lambda: None)
        self.trace_output_prefill: DefaultDict[str, Any] = defaultdict(lambda: None)
        self.trace_id_prefill_sampling: DefaultDict[str, Optional[str]] = defaultdict(lambda: None)
        self.trace_input_prefill_sampling: DefaultDict[str, Any] = defaultdict(lambda: None)
        self.trace_output_prefill_sampling: DefaultDict[str, Any] = defaultdict(lambda: None)
        self.trace_ids_decode: DefaultDict[bool, Optional[Dict[int, str]]] = defaultdict(lambda: None)
        self.trace_inputs_decode: DefaultDict[bool, Optional[List[Any]]] = defaultdict(lambda: None)
        self.trace_output_decode: DefaultDict[bool, Optional[List[Any]]] = defaultdict(lambda: None)

        self.prefill_traces_warmup = False
        self.already_warmed_up_prefill = False
        self.enable_split_sampling = True
        self.mode: Optional[str] = None
        self._prev_sampling_on_device: Optional[bool] = None

    def _set_sampling_trace_mode(self, enabled: bool) -> None:
        """Set sampling trace mode on all models."""
        for model_instance in self.model:
            sampling_module = getattr(model_instance, "sampling", None)
            if sampling_module is not None:
                sampling_module.enable_internal_trace = enabled

    def _get_sampling_contract(
        self, model_id: int
    ) -> Tuple[Any, int, Optional[int], Optional[int]]:
        """Get sampling contract info."""
        sampling_module = getattr(self.model[model_id], "sampling", None)
        sampling_dp = getattr(self.model[model_id], "sampling_dp", 1)
        group_batch = (
            sampling_module.tt_sampling.max_batch_size
            if sampling_module is not None else None
        )
        total_sampling_batch = (
            group_batch * sampling_dp if group_batch is not None else None
        )
        return sampling_module, sampling_dp, group_batch, total_sampling_batch

    def _mock_tokens(
        self,
        batch_size: int,
        seq_len: int,
        kv_cache: Any,
        model_id: int
    ) -> Dict[str, Any]:
        """Create mock tokens for warmup."""
        ret: Dict[str, Any] = {}
        ret["tokens"] = np.zeros((batch_size, seq_len), dtype=np.int64)
        ret["prompt_lens"] = np.array([seq_len] * batch_size, dtype=np.int64)
        ret["empty_slots"] = list(range(batch_size))

        page_table_warmup: Optional[np.ndarray] = None
        if kv_cache is not None and kv_cache[model_id] is not None:
            block_size = get_block_size(kv_cache[model_id])
            num_blocks = num_blocks_in_seq(seq_len, block_size)
            page_table_warmup = np.zeros((batch_size, num_blocks), dtype=np.int32)
        ret["page_table"] = page_table_warmup
        return ret

    def _create_sampling_params(
        self,
        can_sample_on_device: bool,
        batch_size: int,
        greedy_only: bool = False
    ) -> List[Optional[SamplingParams]]:
        """Create sampling params for warmup."""
        if not can_sample_on_device:
            return [None]
        params: List[Optional[SamplingParams]] = [SamplingParams(temperature=0.0)]
        if not greedy_only:
            params.append(SamplingParams(temperature=0.7, top_p=0.9))
        return params

    def warmup_model_prefill(
        self,
        kv_cache: Any,
        enable_trace: bool,
        can_sample_on_device: bool,
        greedy_only: bool = False
    ) -> None:
        """Warmup model for prefill."""
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        logger.info("Simulation: Warmup model prefill")

        sequence_lengths_fn = getattr(
            self.model_args[0],
            'get_warmup_prefill_supported_seq_lens',
            lambda: [128, 256, 512, 1024, 2048]
        )
        sequence_lengths: List[int] = (
            sequence_lengths_fn() if callable(sequence_lengths_fn)
            else sequence_lengths_fn
        )

        warmup_batch_sizes = (1,)
        skip_sequence_lengths = False
        sampling_parameters_sweeped = False

        if enable_trace:
            logger.info("Using batch-1-only traced prefill warmup")

        for model_id in range(self.data_parallel):
            for supported_length in sequence_lengths:
                trace_supported = getattr(
                    self.model_args[0], 'trace_prefill_supported_seq_lens', []
                )
                if model_id != 0 and (
                    supported_length not in trace_supported or not enable_trace
                ):
                    continue

                for batch_size in warmup_batch_sizes:
                    if (batch_size > 1 and
                        batch_size * supported_length >= MAX_BATCHED_PREFILL_SEQ_LEN):
                        logger.info(
                            f"Skipping warmup batch_size={batch_size}, "
                            f"seq_len={supported_length}: exceeds token limit"
                        )
                        continue

                    warmup_args = self._mock_tokens(
                        batch_size, supported_length, kv_cache, model_id
                    )

                    max_prefill_chunk = getattr(
                        self.model_args[0], 'max_prefill_chunk_size', 8192
                    )
                    if (warmup_args["page_table"] is None and
                        max_prefill_chunk_size_cutoff(supported_length, max_prefill_chunk)):
                        logger.warning(
                            f"Skipping warmup for seq lengths after: {supported_length}"
                        )
                        skip_sequence_lengths = True
                        break

                    if not sampling_parameters_sweeped:
                        sampling_params_list = self._create_sampling_params(
                            can_sample_on_device=can_sample_on_device,
                            batch_size=batch_size,
                            greedy_only=greedy_only,
                        )
                    else:
                        sampling_params_list = [None]

                    for param in sampling_params_list:
                        logger.info(
                            f"Warming up prefill seq_len={supported_length}, "
                            f"batch_size={batch_size}"
                        )
                        self.prefill_forward_text(
                            **warmup_args,
                            kv_cache=kv_cache,
                            enable_trace=enable_trace,
                            model_id_warmup=model_id,
                            sampling_params=param,
                        )
                    sampling_parameters_sweeped = True

                if skip_sequence_lengths:
                    break

        # Vision warmup for multimodal models
        if getattr(self.model_args[0], "is_multimodal", False):
            vision_chunk_size = getattr(self.model_args[0], "vision_chunk_size", 896)
            vision_channels = getattr(self.model_args[0], "vision_in_channels", 3)

            warmup_pixel_values = [
                np.zeros((1, vision_channels, vision_chunk_size, vision_chunk_size),
                        dtype=np.float32)
            ]

            batch_size = 1
            prefill_args = self._mock_tokens(batch_size, 128, kv_cache, 0)

            logger.info(f"Warming up vision encoder {vision_chunk_size}x{vision_chunk_size}")
            self.prefill_forward_text(
                **prefill_args,
                kv_cache=kv_cache,
                enable_trace=False,
                model_id_warmup=0,
                sampling_params=None,
                pixel_values=warmup_pixel_values,
                image_sizes=[(vision_chunk_size, vision_chunk_size)],
            )
            logger.info("Vision encoder warmup completed")

    def _capture_trace_prefill(
        self,
        prefill_ids: np.ndarray,
        page_table: Optional[np.ndarray] = None,
        chunk_page_table: Optional[np.ndarray] = None,
        kv_cache: Any = None,
        model_id: int = -1,
        global_user_id: Any = None,
        batch_size: int = 1,
        user_id: int = 0,
        start_pos: int = 0,
    ) -> Tuple[str, Any, Tuple[Any, ...]]:
        """Capture trace for prefill."""
        prefill_kwargs: Dict[str, Any] = {
            "page_table": page_table,
            "chunk_page_table": chunk_page_table,
            "chunk_start_idx": start_pos,
            "batch_size": batch_size,
            "user_id": user_id,
        }
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id

        # Get host inputs
        if hasattr(self.model[model_id], 'prepare_prefill_inputs_trace'):
            host_inputs = self.model[model_id].prepare_prefill_inputs_trace(
                prefill_ids, **prefill_kwargs
            )
        else:
            host_inputs = self.model[model_id].prepare_inputs_prefill(
                prefill_ids, **prefill_kwargs
            )

        # Extract rotation matrices if present
        if isinstance(host_inputs, tuple) and len(host_inputs) > 3:
            tt_rot_mats_global = host_inputs[1] if len(host_inputs) > 1 else None
            tt_rot_mats_local = host_inputs[2] if len(host_inputs) > 2 else None
            device_inputs: Tuple[Any, ...] = (
                host_inputs[0],
                host_inputs[3] if len(host_inputs) > 3 else None,
                host_inputs[4] if len(host_inputs) > 4 else None,
                host_inputs[5] if len(host_inputs) > 5 else None,
            )
        else:
            tt_rot_mats_global = None
            tt_rot_mats_local = None
            device_inputs = host_inputs if isinstance(host_inputs, tuple) else (host_inputs,)

        # Copy to device
        mesh_dev = getattr(self.model_args[model_id], 'mesh_device', self.mesh_device)
        device_inputs_list = copy_host_to_device(
            list(device_inputs), mesh_device=mesh_dev
        )

        # Transform and forward
        if hasattr(self.model[model_id], 'transform_and_embed_prefill_inputs_device'):
            transformed = self.model[model_id].transform_and_embed_prefill_inputs_device(
                *device_inputs_list
            )
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=transformed[0],
                rot_mats_global=tt_rot_mats_global,
                rot_mats_local=tt_rot_mats_local,
                page_table=transformed[1] if len(transformed) > 1 else None,
                chunk_page_table=transformed[2] if len(transformed) > 2 else None,
                chunk_start_idx=transformed[3] if len(transformed) > 3 else start_pos,
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )
        else:
            tt_out_trace = self.model[model_id].ttnn_prefill_forward(
                x=device_inputs_list[0] if len(device_inputs_list) > 0 else prefill_ids,
                rot_mats_global=tt_rot_mats_global,
                rot_mats_local=tt_rot_mats_local,
                page_table=device_inputs_list[1] if len(device_inputs_list) > 1 else None,
                chunk_page_table=device_inputs_list[2] if len(device_inputs_list) > 2 else None,
                chunk_start_idx=start_pos,
                kv_cache=kv_cache,
                batch_size=batch_size,
                user_id=user_id,
            )

        # Synchronize
        if hasattr(ttnn, 'synchronize_device'):
            ttnn.synchronize_device(mesh_dev)

        logger.info("Done Compiling Model (Simulation)")

        # Capture trace (simulation)
        trace_id = f"trace_prefill_{model_id}_{batch_size}_{start_pos}"

        logger.info("Done Capturing Prefill Trace (Simulation)")

        return trace_id, tt_out_trace, tuple(device_inputs_list)

    def _capture_trace_prefill_sampling(
        self,
        model_id: int,
        sampling_batch: int
    ) -> Tuple[str, Tuple[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        """Capture trace for prefill sampling."""
        mesh_device = getattr(self.model_args[model_id], 'mesh_device', self.mesh_device)
        full_dim = getattr(self.model_args[model_id], 'dim', 4096)

        dummy_input = np.zeros((1, 1, sampling_batch, full_dim), dtype=np.float32)

        # Convert to device tensor
        dummy_on_device = numpy_to_ttnn(
            dummy_input,
            device=mesh_device,
            dtype=getattr(ttnn, 'bfloat16', None),
            layout=getattr(ttnn, 'TILE_LAYOUT', None),
        )

        # Apply norm and lm_head
        if hasattr(self.model[model_id], '_apply_norm_and_lm_head'):
            logits = self.model[model_id]._apply_norm_and_lm_head(dummy_on_device)
        else:
            vocab_size = getattr(self.model_args[model_id], 'vocab_size', 128256)
            logits = np.zeros((1, 1, sampling_batch, vocab_size), dtype=np.float32)

        # Sample
        tt_tokens: np.ndarray
        tt_log_probs: Optional[np.ndarray]
        if (hasattr(self.model[model_id], 'sampling') and
            self.model[model_id].sampling is not None):
            tt_tokens, tt_log_probs = self.model[model_id].sampling.sample(
                logits, enable_trace=False
            )
        else:
            tt_tokens = np.zeros(sampling_batch, dtype=np.int64)
            tt_log_probs = None

        if hasattr(ttnn, 'synchronize_device'):
            ttnn.synchronize_device(mesh_device)

        logger.info("Done compiling prefill sampling (Simulation)")

        trace_input = dummy_input
        trace_id = f"trace_prefill_sampling_{model_id}_{sampling_batch}"

        logger.info("Done capturing prefill sampling trace (Simulation)")

        return trace_id, (tt_tokens, tt_log_probs), trace_input

    def _row_sharded_batched_prefill(
        self,
        tokens: np.ndarray,
        page_table: Optional[np.ndarray],
        kv_cache: Any,
        prompt_lens: List[int],
        prefill_seq_lens: List[int],
        enable_trace: bool = True,
        sampling_params: Optional[SamplingParams] = None,
        empty_slots: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Dispatch to model's row-sharded batched prefill."""
        assert self.data_parallel == 1, "Row-sharded batched prefill requires data_parallel=1"

        if hasattr(self.model[0], 'row_sharded_batched_prefill'):
            return self.model[0].row_sharded_batched_prefill(
                tokens,
                page_table,
                kv_cache[0] if kv_cache else None,
                prompt_lens,
                prefill_seq_lens,
                enable_trace=enable_trace,
                sampling_params=sampling_params,
                model_args=self.model_args[0],
                trace_cache={
                    "ids": self.trace_id_prefill,
                    "inputs": self.trace_inputs_prefill,
                    "outputs": self.trace_output_prefill,
                },
                empty_slots=empty_slots,
            )
        else:
            batch_size = tokens.shape[0]
            vocab_size = getattr(self.model_args[0], 'vocab_size', 128256)
            return np.zeros((batch_size, 1, vocab_size), dtype=np.float32)

    def _easy_trace_prefill(
        self,
        prefill_ids: np.ndarray,
        page_table: Optional[np.ndarray] = None,
        full_page_table: Optional[np.ndarray] = None,
        user_id: int = 0,
        last_token_idx: Optional[Union[int, List[int]]] = None,
        kv_cache: Any = None,
        model_id: int = -1,
        prefill_seq_len: Optional[int] = None,
        batch_size: int = 1,
        num_cached_tokens: int = 0,
        **kwargs: Any,
    ) -> Any:
        """Easy trace prefill with caching."""
        global_user_id = kwargs.get("global_user_id", None)
        use_start_pos = "sp1" if num_cached_tokens > 0 else "sp0"
        trace_key = f"{prefill_seq_len}_{model_id}_{batch_size}_{use_start_pos}"

        use_prefix_caching = num_cached_tokens > 0
        chunk_start_idx = num_cached_tokens

        block_size = get_block_size(kv_cache)

        if page_table is not None and batch_size == 1:
            page_table = page_table[user_id:user_id + 1, :]
        if full_page_table is not None and batch_size == 1:
            full_page_table = full_page_table[user_id:user_id + 1, :]

        chunk_page_table: Optional[np.ndarray] = None
        max_blocks_prefill = _get_max_blocks_prefill(kv_cache)

        source_page_table = (
            full_page_table if full_page_table is not None else page_table
        )
        if source_page_table is None:
            raise ValueError("Traced prefill requires a page_table")

        page_table = _pad_or_create_page_table(source_page_table, max_blocks_prefill)

        if batch_size == 1 and use_prefix_caching and prefill_seq_len is not None:
            chunk_start_block = num_cached_tokens // block_size
            chunk_end_block = num_blocks_in_seq(
                num_cached_tokens + prefill_seq_len, block_size
            )
            chunk_page_table = source_page_table[:, chunk_start_block:chunk_end_block]
            chunk_blocks = num_blocks_in_seq(prefill_seq_len, block_size)
            chunk_page_table = _pad_or_create_page_table(chunk_page_table, chunk_blocks)

        if self.trace_id_prefill[trace_key] is None:
            trace_id, tt_out_trace, device_inputs = self._capture_trace_prefill(
                prefill_ids,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                kv_cache=kv_cache,
                model_id=model_id,
                global_user_id=global_user_id,
                batch_size=batch_size,
                user_id=user_id,
                start_pos=chunk_start_idx,
            )
            self.trace_id_prefill[trace_key] = trace_id
            self.trace_inputs_prefill[trace_key] = device_inputs
            self.trace_output_prefill[trace_key] = tt_out_trace

        tt_out_trace = self._prefill_forward_trace(
            self.trace_id_prefill[trace_key],
            self.trace_inputs_prefill[trace_key],
            self.trace_output_prefill[trace_key],
            prefill_ids,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            model_id=model_id,
            global_user_id=global_user_id,
            batch_size=batch_size,
            user_id=user_id,
            start_pos=chunk_start_idx,
        )

        return tt_out_trace

    def _prefill_forward_trace(
        self,
        trace_id: Optional[str],
        device_inputs: Optional[Tuple[Any, ...]],
        tt_out_trace: Any,
        prefill_ids: np.ndarray,
        user_id: int = 0,
        page_table: Optional[np.ndarray] = None,
        chunk_page_table: Optional[np.ndarray] = None,
        model_id: int = -1,
        global_user_id: Any = None,
        batch_size: int = 1,
        start_pos: int = 0,
    ) -> Any:
        """Execute prefill trace."""
        prefill_kwargs: Dict[str, Any] = {
            "page_table": page_table,
            "chunk_page_table": chunk_page_table,
            "chunk_start_idx": start_pos,
            "batch_size": batch_size,
            "user_id": user_id,
        }
        if global_user_id is not None:
            prefill_kwargs["global_user_id"] = global_user_id

        if hasattr(self.model[model_id], 'prepare_prefill_inputs_trace'):
            host_inputs = self.model[model_id].prepare_prefill_inputs_trace(
                prefill_ids, **prefill_kwargs
            )
        else:
            host_inputs = self.model[model_id].prepare_inputs_prefill(
                prefill_ids, **prefill_kwargs
            )

        if isinstance(host_inputs, tuple) and len(host_inputs) > 3:
            host_inputs_subset: Tuple[Any, ...] = (
                host_inputs[0],
                host_inputs[3] if len(host_inputs) > 3 else None,
                host_inputs[4] if len(host_inputs) > 4 else None,
                host_inputs[5] if len(host_inputs) > 5 else None,
            )
        else:
            host_inputs_subset = host_inputs if isinstance(host_inputs, tuple) else (host_inputs,)

        mesh_dev = getattr(self.model_args[model_id], 'mesh_device', self.mesh_device)
        if device_inputs is not None:
            copy_host_to_device(
                list(host_inputs_subset),
                device_tensors=list(device_inputs),
                mesh_device=mesh_dev
            )

        # Execute trace (simulation: just return cached output)
        if hasattr(ttnn, 'execute_trace'):
            ttnn.execute_trace(mesh_dev, trace_id, cq_id=0, blocking=False)

        return tt_out_trace

    def prefill_forward_text(
        self,
        tokens: np.ndarray,
        page_table: Optional[np.ndarray] = None,
        kv_cache: Any = None,
        prompt_lens: Optional[Union[np.ndarray, List[int]]] = None,
        empty_slots: Optional[List[int]] = None,
        enable_trace: bool = True,
        model_id_warmup: Optional[int] = None,
        sampling_params: Optional[SamplingParams] = None,
        start_pos: Optional[List[int]] = None,
        return_hidden_states: bool = False,
        warmup_prefill: bool = True,
        **kwargs: Any,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """Prefill forward for text."""
        self.mode = Mode.PREFILL

        if page_table is not None:
            assert isinstance(page_table, np.ndarray), "page_table must be numpy.ndarray"
        else:
            enable_trace = False

        sampling_on_device_requested = sampling_params is not None

        if warmup_prefill:
            sampling_on_device_enabled = (
                getattr(self.model[0], "_supports_on_device_sampling", False)
                and getattr(self.model[0], "sampling", None) is not None
            )
            self.warmup_model_prefill(
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                can_sample_on_device=sampling_on_device_enabled,
            )

        batch_size, batch_seq_len = tokens.shape
        max_batch_size_per_model = getattr(self.model_args[0], 'max_batch_size', 32)

        # Prepare output tensors
        output_tokens: np.ndarray
        output_log_probs: List[Any]

        if return_hidden_states:
            hidden_size = getattr(self.model_args[0], 'dim', 4096)
            output_tensor = np.zeros((batch_size, hidden_size), dtype=np.float32)
        else:
            vocab_size = getattr(self.model_args[0], 'vocab_size', 128256)
            output_tensor = np.zeros((batch_size, 1, vocab_size), dtype=np.float32)
            output_tokens = np.zeros((batch_size, 1), dtype=np.int64)
            output_log_probs = [None] * batch_size

        sampling_executed = False

        # Process prompt_lens
        prompt_lens_list: List[int]
        if prompt_lens is None:
            prompt_lens_list = [batch_seq_len] * batch_size
        elif isinstance(prompt_lens, np.ndarray):
            prompt_lens_list = [int(x) for x in prompt_lens.tolist()]
        else:
            prompt_lens_list = list(prompt_lens)

        if empty_slots is None:
            empty_slots = list(range(batch_size))

        local_batch_size = getattr(
            self.model_args[0], "max_local_batch_size", max_batch_size_per_model
        )

        # Process cached tokens
        num_cached_per_user: List[int] = (
            [int(n) for n in start_pos] if start_pos is not None
            else [0] * len(prompt_lens_list)
        )

        assert len(num_cached_per_user) == len(prompt_lens_list), \
            f"start_pos length mismatch: {len(num_cached_per_user)} vs {len(prompt_lens_list)}"

        for i, (seq_len_val, num_cached) in enumerate(zip(prompt_lens_list, num_cached_per_user)):
            assert 0 <= num_cached < seq_len_val, \
                f"user {i}: num_cached={num_cached} must be < seq_len={seq_len_val}"

        prefill_seq_lens = [
            get_padded_prefill_len(seq_len_val - num_cached)
            for seq_len_val, num_cached in zip(prompt_lens_list, num_cached_per_user)
        ]

        # Check for row-sharded batched prefill
        model_0 = self.model[0]
        is_harmony = tokens.shape[1] > 0 and int(tokens[0, 0]) == 200006

        if (
            getattr(model_0, "users_row_sharded", False)
            and batch_size > 1
            and sampling_params is not None
            and is_harmony
        ):
            return self._row_sharded_batched_prefill(
                tokens,
                page_table,
                kv_cache,
                prompt_lens_list,
                prefill_seq_lens=prefill_seq_lens,
                enable_trace=enable_trace,
                sampling_params=sampling_params,
                empty_slots=empty_slots,
            )

        # Check for batched prefill
        use_batched_prefill = (
            batch_size > 1
            and len(set(prefill_seq_lens)) == 1
            and self.data_parallel == 1
            and not getattr(self.model_args[0], "disable_batched_prefill", False)
            and all(n == 0 for n in num_cached_per_user)
        )

        max_prefill_chunk = getattr(self.model_args[0], 'max_prefill_chunk_size', 8192)

        if use_batched_prefill and any(s > max_prefill_chunk for s in prefill_seq_lens):
            logger.info(
                f"Batched prefill disabled: prefill len exceeds max_prefill_chunk_size"
            )
            use_batched_prefill = False

        if use_batched_prefill and sampling_on_device_requested:
            sampling_module, sampling_dp, _, _ = self._get_sampling_contract(0)
            if sampling_module is not None and sampling_dp > 1:
                use_batched_prefill = False

        if use_batched_prefill:
            padded_batch = next(
                (b for b in SUPPORTED_PREFILL_BATCH_SIZES if b >= batch_size),
                max_batch_size_per_model,
            )
            if padded_batch > max_batch_size_per_model:
                logger.info(f"Batched prefill disabled: padded_batch exceeds max_batch_size")
                use_batched_prefill = False
            elif padded_batch * prefill_seq_lens[0] >= MAX_BATCHED_PREFILL_SEQ_LEN:
                logger.info(f"Batched prefill disabled: token count exceeds limit")
                use_batched_prefill = False

        if not use_batched_prefill:
            padded_batch = max_batch_size_per_model

        all_users = [0] if use_batched_prefill else empty_slots
        prefill_results: List[Dict[str, Any]] = []

        for idx, user_id in enumerate(all_users):
            model_id = (
                user_id // max_batch_size_per_model
                if model_id_warmup is None else model_id_warmup
            )
            group_user_id = user_id % local_batch_size if page_table is None else 0

            # Declare variables with proper types
            batch_user_ids: Optional[List[int]]
            last_token_idx: Union[int, List[int]]
            prefill_seq_len: int
            current_seq_len: Union[int, List[int]]
            num_cached_tokens: int = 0

            if use_batched_prefill:
                batch_user_ids = empty_slots
                last_token_idx = [(sl - 1) for sl in prompt_lens_list]
                prefill_seq_len = prefill_seq_lens[0]
                current_seq_len = prompt_lens_list
            else:
                batch_user_ids = None
                current_seq_len = int(prompt_lens_list[idx])
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                last_token_idx = current_seq_len - 1
                prefill_seq_len = prefill_seq_lens[idx]
                logger.info(f"Prefilling User {user_id + 1} up to {current_seq_len} tokens")

            local_kwargs = kwargs.copy()

            if getattr(self.model[model_id], "users_row_sharded", False):
                local_kwargs["global_user_id"] = (
                    batch_user_ids if use_batched_prefill else user_id
                )

            sampling_enabled = (
                sampling_on_device_requested
                and getattr(self.model[model_id], "_supports_on_device_sampling", False)
                and getattr(self.model[model_id], "sampling", None) is not None
            )

            # Prepare prefill_ids
            if use_batched_prefill:
                prefill_ids = np.zeros((padded_batch, prefill_seq_len), dtype=np.int64)
                padded_last_token_idx: List[int] = [0] * padded_batch

                for local_idx, slot in enumerate(empty_slots):
                    seq_len_local = int(prompt_lens_list[local_idx])
                    padded_tokens = np.concatenate([
                        tokens[local_idx:local_idx + 1, :seq_len_local],
                        np.zeros((1, prefill_seq_len - seq_len_local), dtype=np.int64),
                    ], axis=-1)
                    prefill_ids[slot:slot + 1] = padded_tokens
                    if isinstance(last_token_idx, list):
                        padded_last_token_idx[slot] = last_token_idx[local_idx]

                last_token_idx = padded_last_token_idx
            else:
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0
                seq_len_int = current_seq_len if isinstance(current_seq_len, int) else current_seq_len[0]
                prefill_ids = np.concatenate([
                    tokens[idx:idx + 1, num_cached_tokens:seq_len_int],
                    np.zeros((1, prefill_seq_len - (seq_len_int - num_cached_tokens)),
                            dtype=np.int64),
                ], axis=-1)

            # Check trace enable
            can_enable_trace = getattr(
                self.model_args[model_id], 'can_enable_trace',
                lambda x, y: True
            )
            if callable(can_enable_trace):
                enable_trace_current = enable_trace and can_enable_trace(
                    prefill_seq_len,
                    num_cached_tokens if not use_batched_prefill else 0
                )
            else:
                enable_trace_current = enable_trace

            logger.info(
                f"Prefill seq_len={prefill_seq_len}, "
                f"max_chunk={max_prefill_chunk}, trace={enable_trace_current}"
            )

            # Get page table for user
            page_table_user: Optional[np.ndarray] = None
            full_page_table_user: Optional[np.ndarray] = None

            if page_table is not None:
                page_table_for_user = (
                    page_table if use_batched_prefill else page_table[idx:idx + 1]
                )
                page_table_user = self._get_prefill_user_page_table(
                    page_table_for_user,
                    kv_cache[model_id] if kv_cache else None,
                    current_seq_len,
                    trace_enabled=enable_trace_current,
                    prefill_seq_len=prefill_seq_len,
                    use_batched_prefill=use_batched_prefill,
                    user_id=batch_user_ids if use_batched_prefill else user_id,
                    padded_batch_size=padded_batch if use_batched_prefill else None,
                )
                if enable_trace_current and not use_batched_prefill:
                    full_page_table_user = self._get_prefill_user_page_table(
                        page_table_for_user,
                        kv_cache[model_id] if kv_cache else None,
                        current_seq_len,
                        trace_enabled=False,
                        prefill_seq_len=prefill_seq_len,
                        use_batched_prefill=False,
                        user_id=user_id,
                        padded_batch_size=None,
                    )

            model_kv_cache = kv_cache[model_id] if kv_cache is not None else None

            # Handle pixel_values for vision
            if local_kwargs.get("pixel_values") is not None:
                local_kwargs["pixel_values"] = local_kwargs["pixel_values"][idx]
                if "image_grid_thw" in local_kwargs:
                    local_kwargs["image_grid_thw"] = local_kwargs["image_grid_thw"][idx]
                if local_kwargs.get("image_sizes") is not None:
                    local_kwargs["image_sizes"] = local_kwargs["image_sizes"][idx]

            # Run prefill
            if enable_trace_current:
                logits = self._easy_trace_prefill(
                    prefill_ids,
                    page_table=page_table_user,
                    full_page_table=full_page_table_user,
                    user_id=(
                        batch_user_ids[0] if batch_user_ids else group_user_id
                    ),
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    prefill_seq_len=prefill_seq_len,
                    batch_size=padded_batch if use_batched_prefill else 1,
                    num_cached_tokens=0 if use_batched_prefill else num_cached_tokens,
                    **local_kwargs,
                )
            else:
                logits = self.prefill_forward_single_user_text(
                    prefill_ids,
                    page_table=page_table_user,
                    user_id=(
                        batch_user_ids[0] if batch_user_ids else group_user_id
                    ),
                    last_token_idx=last_token_idx,
                    kv_cache=model_kv_cache,
                    model_id=model_id,
                    num_cached_tokens=0 if use_batched_prefill else num_cached_tokens,
                    batch_size=padded_batch if use_batched_prefill else 1,
                    **local_kwargs,
                )

            # Collect results
            if not use_batched_prefill:
                prefill_results.append({
                    "idx": idx,
                    "model_id": model_id,
                    "last_token_idx": last_token_idx,
                    "logits": logits,
                    "sampling": sampling_enabled,
                })

        # Process results
        if len(prefill_results) > 0:
            for res in prefill_results:
                idx = res["idx"]
                last_token_idx_res = res["last_token_idx"]
                model_id = res["model_id"]
                num_cached_tokens = int(start_pos[idx]) if start_pos is not None else 0

                last_token_idx_int = (
                    last_token_idx_res[0] if isinstance(last_token_idx_res, list)
                    else last_token_idx_res
                )
                last_token_idx_relative = last_token_idx_int - num_cached_tokens

                logits = res["logits"]
                logits_np = tensor_to_numpy(logits)

                if return_hidden_states:
                    hidden_size = getattr(self.model_args[0], 'dim', 4096)
                    output_tensor[idx] = logits_np.reshape(-1)[:hidden_size]
                else:
                    vocab_size = getattr(self.model_args[0], 'vocab_size', 128256)
                    logits_flat = logits_np.reshape(-1, vocab_size)
                    pos = last_token_idx_relative % 32
                    output_tensor[idx] = logits_flat[pos:pos + 1, :vocab_size]

        logger.info(
            f"Finished prefill for all users up to {batch_seq_len} tokens, "
            f"Starting decode..."
        )

        if sampling_executed:
            return output_tokens, reformat_logprobs(output_log_probs, batch_size)
        else:
            return output_tensor

    def prefill_forward_single_user_text(
        self,
        tokens: np.ndarray,
        page_table: Optional[np.ndarray],
        user_id: Union[int, List[int]],
        last_token_idx: Union[int, List[int]],
        kv_cache: Any = None,
        model_id: int = -1,
        num_cached_tokens: int = 0,
        batch_size: int = 1,
        **kwargs: Any,
    ) -> Any:
        """Prefill forward for single user."""
        seq_len = tokens.shape[-1]
        max_prefill_chunk = getattr(
            self.model_args[model_id], 'max_prefill_chunk_size', 8192
        )
        use_chunked_prefill = seq_len > max_prefill_chunk
        use_prefix_caching = num_cached_tokens > 0

        user_id_int = user_id[0] if isinstance(user_id, list) else user_id
        last_token_idx_int = (
            last_token_idx[0] if isinstance(last_token_idx, list) else last_token_idx
        )

        if use_chunked_prefill or use_prefix_caching:
            assert page_table is not None, "page_table required for chunked prefill"
            assert kv_cache is not None, "kv_cache required for chunked prefill"

            if use_chunked_prefill:
                chunk_size = get_max_prefill_chunk_size(seq_len, max_prefill_chunk)
            else:
                chunk_size = seq_len

            last_token_idx_in_seq = last_token_idx_int - num_cached_tokens
            block_size = get_block_size(kv_cache)
            last_chunk_start = (last_token_idx_in_seq // chunk_size) * chunk_size

            page_table_user = page_table[user_id_int:user_id_int + 1, :]
            num_padding_blocks = (
                num_blocks_in_seq(seq_len + num_cached_tokens, block_size)
                - page_table_user.shape[1]
            )

            if num_padding_blocks > 0:
                page_table_user_padded = np.concatenate([
                    page_table_user,
                    np.zeros((1, num_padding_blocks), dtype=np.int32)
                ], axis=-1)
            else:
                page_table_user_padded = page_table_user

            CHUNK_USER_ID = 0

            for chunk_start in range(num_cached_tokens, num_cached_tokens + seq_len, chunk_size):
                chunk_end = chunk_start + chunk_size
                chunk_start_relative = chunk_start - num_cached_tokens
                chunk_end_relative = chunk_end - num_cached_tokens

                chunk_tokens = tokens[:, chunk_start_relative:chunk_end_relative]
                chunk_page_table = page_table_user_padded[
                    :, chunk_start // block_size:chunk_end // block_size
                ]

                chunk_inputs = self.model[model_id].prepare_inputs_prefill(
                    chunk_tokens,
                    start_pos=chunk_start,
                    page_table=page_table_user_padded,
                    chunk_page_table=chunk_page_table,
                    batch_size=batch_size,
                    user_id=CHUNK_USER_ID,
                    **kwargs,
                )

                if isinstance(chunk_inputs, tuple) and len(chunk_inputs) >= 5:
                    (
                        chunk_prefill_input,
                        chunk_rot_mats_global,
                        chunk_rot_mats_local,
                        page_table_tt,
                        chunk_page_table_tt,
                    ) = chunk_inputs[:5]
                else:
                    chunk_prefill_input = (
                        chunk_inputs[0] if isinstance(chunk_inputs, tuple)
                        else chunk_inputs
                    )
                    chunk_rot_mats_global = None
                    chunk_rot_mats_local = None
                    page_table_tt = None
                    chunk_page_table_tt = None

                tt_logits = self.model[model_id].ttnn_prefill_forward(
                    chunk_prefill_input,
                    rot_mats_global=chunk_rot_mats_global,
                    rot_mats_local=chunk_rot_mats_local,
                    user_id=CHUNK_USER_ID,
                    page_table=page_table_tt,
                    chunk_page_table=chunk_page_table_tt,
                    chunk_start_idx=chunk_start,
                    get_last_token=(last_token_idx_int // 32) * 32,
                    kv_cache=kv_cache,
                    batch_size=batch_size,
                    **kwargs,
                )

                if chunk_start_relative == last_chunk_start:
                    return tt_logits
                else:
                    del tt_logits

            # Fallback return
            vocab_size = getattr(self.model_args[model_id], 'vocab_size', 128256)
            return np.zeros((1, 1, vocab_size), dtype=np.float32)
        else:
            inputs = self.model[model_id].prepare_inputs_prefill(
                tokens,
                page_table=page_table,
                batch_size=batch_size,
                user_id=user_id,
                **kwargs,
            )

            if isinstance(inputs, tuple) and len(inputs) >= 4:
                prefill_input, rot_mats_global, rot_mats_local, page_table_tt = inputs[:4]
            else:
                prefill_input = inputs[0] if isinstance(inputs, tuple) else inputs
                rot_mats_global = None
                rot_mats_local = None
                page_table_tt = None

            tt_logits = self.model[model_id].ttnn_prefill_forward(
                prefill_input,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                page_table=page_table_tt,
                get_last_token=-1 if batch_size > 1 else (last_token_idx_int // 32) * 32,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
            return tt_logits

    def decode_forward(
        self,
        tokens: np.ndarray,
        start_pos: np.ndarray,
        page_table: Optional[np.ndarray] = None,
        kv_cache: Any = None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params: Optional[SamplingParams] = None,
        reset_batch: bool = False,
        prompt_tokens: Optional[np.ndarray] = None,
        output_tokens: Optional[np.ndarray] = None,
        slot_remap: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Any:
        """Decode forward."""
        mode_switched = False
        if self.mode != Mode.DECODE:
            self.mode = Mode.DECODE
            mode_switched = True

        for i in range(len(self.model)):
            if hasattr(self.model[i], 'switch_mode'):
                self.model[i].switch_mode(Mode.DECODE)

        sampling_on_device = sampling_params is not None
        split_sampling_enabled = bool(self.enable_split_sampling and sampling_on_device)
        self._set_sampling_trace_mode(split_sampling_enabled)

        B = tokens.shape[0]
        tokens_list = list(np.array_split(tokens, self.data_parallel, axis=0))
        start_pos_list = list(np.array_split(start_pos, self.data_parallel, axis=0))
        page_table_list: Optional[List[np.ndarray]] = (
            list(np.array_split(page_table, self.data_parallel, axis=0))
            if page_table is not None else None
        )

        # Handle sampling params
        if sampling_params is not None:
            sampling_dp_values = [
                getattr(self.model[i], "sampling_dp", 1)
                for i in range(self.data_parallel)
            ]
            sampling_dp = max(self.data_parallel, sampling_dp_values[0])
            sampling_params_list = chunk_sampling_params(sampling_params, sampling_dp)

            prompt_chunks: List[Optional[np.ndarray]] = (
                list(np.array_split(prompt_tokens, sampling_dp, axis=0))
                if prompt_tokens is not None else [None] * sampling_dp
            )
            output_chunks: List[Optional[np.ndarray]] = (
                list(np.array_split(output_tokens, sampling_dp, axis=0))
                if output_tokens is not None else [None] * sampling_dp
            )

            for i in range(self.data_parallel):
                sampling_module = getattr(self.model[i], "sampling", None)
                if sampling_module is not None:
                    cpm = sampling_dp // self.data_parallel
                    start = i * cpm
                    model_chunks = sampling_params_list[start:start + cpm]

                    model_prompt: Optional[np.ndarray] = None
                    if prompt_tokens is not None:
                        valid_chunks = [
                            c for c in prompt_chunks[start:start + cpm] if c is not None
                        ]
                        if valid_chunks:
                            model_prompt = np.concatenate(valid_chunks, axis=0)

                    model_output: Optional[np.ndarray] = None
                    if output_tokens is not None:
                        valid_chunks = [
                            c for c in output_chunks[start:start + cpm] if c is not None
                        ]
                        if valid_chunks:
                            model_output = np.concatenate(valid_chunks, axis=0)

                    if hasattr(sampling_module, 'apply_decode_state'):
                        sampling_module.apply_decode_state(
                            model_chunks,
                            reset_batch=reset_batch,
                            prompt_tokens=model_prompt,
                            output_tokens=model_output,
                        )

                    if slot_remap is not None:
                        sm_bs = sampling_module.seed_manager.max_batch_size
                        rank_remap = slot_remap[i * sm_bs:(i + 1) * sm_bs]
                        if hasattr(sampling_module.seed_manager, 'apply_slot_remap'):
                            sampling_module.seed_manager.apply_slot_remap(rank_remap)

                    if hasattr(sampling_module.seed_manager, 'get_new_values'):
                        sampling_module.seed_manager.get_new_values()

        decode_kwargs = {
            "current_pos": start_pos_list,
            "tokens": tokens_list,
            "page_table": page_table_list,
            "kv_cache": kv_cache,
            "sampling_on_device": sampling_on_device,
        }

        if enable_trace:
            tt_decode_output = self._decode_forward_trace_text(
                **decode_kwargs, reset_batch=reset_batch or mode_switched
            )
        else:
            tt_decode_output = self._decode_forward_no_trace_text(**decode_kwargs)

        if read_from_device:
            to_host = self.read_decode_output(tt_decode_output)
            return self.process_decode_output_host(
                to_host, is_tokens=(sampling_params is not None)
            )

        return tt_decode_output

    def _decode_forward_no_trace_text(
        self,
        tokens: List[np.ndarray],
        current_pos: List[np.ndarray],
        page_table: Optional[List[np.ndarray]] = None,
        kv_cache: Any = None,
        sampling_on_device: bool = False,
    ) -> List[Tuple[Any, Any]]:
        """Decode forward without trace."""
        tt_output: List[Tuple[Any, Any]] = []
        tt_tokens_list: List[Any] = []
        tt_current_pos_list: List[Any] = []
        tt_rot_mat_idxs_list: List[Any] = []
        tt_page_table_list: List[Any] = []

        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None
            model_i = self.model[i]

            decode_inputs = model_i.prepare_inputs_decode(
                tokens[i], current_pos[i], user_page_table
            )

            if isinstance(decode_inputs, tuple) and len(decode_inputs) >= 4:
                tt_tokens_i, tt_current_pos_i, tt_rot_mat_idxs_i, tt_page_table_i = decode_inputs[:4]
            else:
                tt_tokens_i = decode_inputs[0] if isinstance(decode_inputs, tuple) else decode_inputs
                tt_current_pos_i = None
                tt_rot_mat_idxs_i = None
                tt_page_table_i = None

            tt_tokens_list.append(tt_tokens_i)
            tt_current_pos_list.append(tt_current_pos_i)
            tt_rot_mat_idxs_list.append(tt_rot_mat_idxs_i)
            tt_page_table_list.append(tt_page_table_i)

        for i in range(self.data_parallel):
            user_kv_cache = kv_cache[i] if kv_cache is not None else None

            tt_logits_i, tt_log_probs_i = self.model[i].ttnn_decode_forward(
                tt_tokens_list[i],
                tt_current_pos_list[i],
                rot_mat_idxs=tt_rot_mat_idxs_list[i],
                page_table=tt_page_table_list[i],
                kv_cache=user_kv_cache,
                sampling_on_device=sampling_on_device,
            )
            tt_output.append((tt_logits_i, tt_log_probs_i))

        return tt_output

    def _capture_decode_trace_text(
        self,
        tokens: List[np.ndarray],
        current_pos: List[np.ndarray],
        page_table: Optional[List[np.ndarray]] = None,
        kv_cache: Any = None,
        sampling_on_device: bool = False,
    ) -> Tuple[Dict[int, str], List[Any], List[List[Any]]]:
        """Capture decode trace."""
        # Compile run
        self._decode_forward_no_trace_text(
            tokens, current_pos, page_table=page_table,
            kv_cache=kv_cache, sampling_on_device=sampling_on_device
        )
        logger.info("Done Compiling Model")

        device_inputs: List[List[Any]] = []
        tt_out_trace: List[Any] = []
        trace_ids: Dict[int, str] = {}

        for i in range(self.data_parallel):
            user_page_table = page_table[i] if page_table is not None else None

            if hasattr(self.model[i], 'prepare_decode_inputs_host'):
                host_inputs = self.model[i].prepare_decode_inputs_host(
                    tokens[i], current_pos[i], page_table=user_page_table
                )
            else:
                host_inputs = self.model[i].prepare_inputs_decode(
                    tokens[i], current_pos[i], user_page_table
                )

            mesh_dev = getattr(self.model_args[i], 'mesh_device', self.mesh_device)
            device_inputs_i = copy_host_to_device(
                list(host_inputs) if isinstance(host_inputs, tuple) else [host_inputs],
                mesh_device=mesh_dev
            )
            device_inputs.append(device_inputs_i)

        for i in range(self.data_parallel):
            sampling_module = getattr(self.model[i], "sampling", None)
            split_enabled = (
                sampling_on_device
                and sampling_module is not None
                and getattr(sampling_module, "enable_internal_trace", False)
            )

            trace_id = f"trace_decode_{i}_{sampling_on_device}"
            trace_ids[i] = trace_id

            user_kv_cache = kv_cache[i] if kv_cache is not None else None
            model_inputs = device_inputs[i][:4] if len(device_inputs[i]) > 4 else device_inputs[i]

            bind_trace_inputs = getattr(self.model[i], "bind_decode_trace_inputs", None)
            if bind_trace_inputs is not None:
                bind_trace_inputs(device_inputs[i])

            output = self.model[i].ttnn_decode_forward(
                *model_inputs,
                kv_cache=user_kv_cache,
                sampling_on_device=sampling_on_device,
                capture_sampling_trace=split_enabled,
            )
            tt_out_trace.append(output)

            if split_enabled and sampling_module is not None and hasattr(sampling_module, 'capture_trace'):
                sampling_module.capture_trace(
                    logits=output, tt_out_tok=device_inputs[i][0]
                )

        logger.info("Done Capturing Decode Trace")
        return trace_ids, tt_out_trace, device_inputs

    def _decode_forward_trace_text(
        self,
        tokens: List[np.ndarray],
        current_pos: List[np.ndarray],
        page_table: Optional[List[np.ndarray]] = None,
        kv_cache: Any = None,
        sampling_on_device: bool = False,
        reset_batch: bool = False,
    ) -> List[Any]:
        """Decode forward with trace."""
        if self.trace_ids_decode[sampling_on_device] is None:
            trace_ids, tt_out_trace, device_inputs = self._capture_decode_trace_text(
                tokens, current_pos, page_table=page_table,
                kv_cache=kv_cache, sampling_on_device=sampling_on_device
            )
            self.trace_ids_decode[sampling_on_device] = trace_ids
            self.trace_inputs_decode[sampling_on_device] = device_inputs
            self.trace_output_decode[sampling_on_device] = tt_out_trace

        prev_sampling = self._prev_sampling_on_device
        self._prev_sampling_on_device = sampling_on_device
        sampling_mode_changed = (
            prev_sampling is not None and prev_sampling != sampling_on_device
        )
        reset_inputs = reset_batch or not sampling_on_device or sampling_mode_changed

        # Check page table changes
        page_table_changed = page_table is not None and (
            self.prev_page_table is None
            or any(
                not np.array_equal(prev, curr)
                for prev, curr in zip(self.prev_page_table, page_table)
            )
        )

        trace_inputs = self.trace_inputs_decode[sampling_on_device]

        for i in range(self.data_parallel):
            refresh = reset_inputs or getattr(
                self.model[i], "_tt_vllm_always_refresh_decode_trace_inputs", False
            )
            user_page_table = page_table[i] if page_table is not None else None

            if refresh:
                if hasattr(self.model[i], 'prepare_decode_inputs_host'):
                    host_inputs_i = self.model[i].prepare_decode_inputs_host(
                        tokens[i], current_pos[i], user_page_table
                    )
                else:
                    host_inputs_i = self.model[i].prepare_inputs_decode(
                        tokens[i], current_pos[i], user_page_table
                    )

                if trace_inputs is not None:
                    copy_host_to_device(
                        list(host_inputs_i) if isinstance(host_inputs_i, tuple) else [host_inputs_i],
                        device_tensors=trace_inputs[i],
                    )
            elif page_table_changed:
                if hasattr(self.model[i], 'prepare_decode_inputs_host'):
                    host_inputs_i = self.model[i].prepare_decode_inputs_host(
                        tokens[i], current_pos[i], user_page_table
                    )
                else:
                    host_inputs_i = self.model[i].prepare_inputs_decode(
                        tokens[i], current_pos[i], user_page_table
                    )

                if (isinstance(host_inputs_i, tuple) and
                    len(host_inputs_i) > DECODE_PAGE_TABLE_INPUT_IDX and
                    trace_inputs is not None):
                    host_page_table = host_inputs_i[DECODE_PAGE_TABLE_INPUT_IDX]
                    device_page_table = trace_inputs[i][DECODE_PAGE_TABLE_INPUT_IDX]
                    if host_page_table is not None and hasattr(ttnn, 'copy_host_to_device_tensor'):
                        ttnn.copy_host_to_device_tensor(host_page_table, device_page_table)

        if page_table_changed:
            self.prev_page_table = tuple(pt.copy() for pt in page_table) # type: ignore[union-attr]

        # Execute traces
        trace_ids = self.trace_ids_decode[sampling_on_device] # type: ignore[assignment]
        if trace_ids is not None:
            for i, trace_id in trace_ids.items():
                mesh_dev = getattr(self.model_args[i], 'mesh_device', self.mesh_device)
                if hasattr(ttnn, 'execute_trace'):
                    ttnn.execute_trace(mesh_dev, trace_id, cq_id=0, blocking=False)

                _outputs = self.trace_output_decode[sampling_on_device]
                outputs: list[np.ndarray[Any, Any]] = _outputs if _outputs is not None else []

        if sampling_on_device:
            new_outputs: List[Any] = []
            for i in range(self.data_parallel):
                sampling_module = getattr(self.model[i], "sampling", None)
                if (sampling_module is None or
                    not getattr(sampling_module, "enable_internal_trace", False)):
                    new_outputs.append(outputs[i] if i < len(outputs) else None)
                    continue
                trace_input_i = trace_inputs[i] if trace_inputs is not None else None
                tt_out_tok = trace_input_i[0] if trace_input_i is not None else None
                new_outputs.append(
                    sampling_module.sample(
                        logits=outputs[i] if i < len(outputs) else None,
                        tt_out_tok=tt_out_tok,
                    )
                )
            return new_outputs

        return outputs

    def read_decode_output(
        self,
        tt_out: List[Any],
        async_read: bool = False
    ) -> List[Any]:
        """Read decode output from device."""
        def _read_logprobs(lp: Any, blocking: bool = True) -> Any:
            if lp is None:
                return None
            if isinstance(lp, np.ndarray):
                return lp
            if hasattr(lp, 'cpu'):
                return lp.cpu(blocking=blocking)
            return tensor_to_numpy(lp)

        results: List[Any] = []
        for out in tt_out:
            if isinstance(out, tuple):
                logits = tensor_to_numpy(out[0])
                log_probs = _read_logprobs(out[1])
                results.append((logits, log_probs))
            else:
                results.append(tensor_to_numpy(out))

        return results

    def process_decode_output_host(
        self,
        tt_out: List[Any],
        is_tokens: bool = False
    ) -> Tuple[np.ndarray, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """Process decode output on host."""
        max_batch_size_per_model = getattr(self.model_args[0], 'max_batch_size', 32)
        vocab_size = getattr(self.model_args[0], 'vocab_size', 128256)

        logits_list: List[np.ndarray] = []
        log_probs_list: List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]] = []

        for i in range(self.data_parallel):
            if isinstance(tt_out[i], tuple):
                logits_i = tensor_to_numpy(tt_out[i][0])
                logits_i = logits_i.reshape(max_batch_size_per_model, 1, -1)[:, :, :vocab_size]

                lp = tt_out[i][1]
                if isinstance(lp, LogProbsResult):
                    # Handle LogProbsResult
                    lp_tensor = (
                        lp.topk_logprobs_host if lp.topk_logprobs_host is not None
                        else lp.topk_logprobs
                    )
                    idx_tensor = (
                        lp.topk_indices_host if lp.topk_indices_host is not None
                        else lp.topk_indices
                    )

                    sampling_dp = getattr(self.model[i], "sampling_dp", 1)

                    if sampling_dp > 1 and hasattr(self.mesh_device, 'shape'):
                        rows, cols = self.mesh_device.shape
                        row_lps: List[np.ndarray] = []
                        row_idxs: List[np.ndarray] = []

                        if hasattr(ttnn, 'get_device_tensors') and lp_tensor is not None and idx_tensor is not None:
                            device_tensors_lp = ttnn.get_device_tensors(lp_tensor)
                            device_tensors_idx = ttnn.get_device_tensors(idx_tensor)

                            for row in range(rows):
                                dev_idx = row * cols
                                row_lp = tensor_to_numpy(device_tensors_lp[dev_idx])
                                row_lps.append(
                                    row_lp.reshape(-1, row_lp.shape[-1])[:max_batch_size_per_model]
                                )
                                row_idx = tensor_to_numpy(device_tensors_idx[dev_idx])
                                row_idxs.append(
                                    row_idx.reshape(-1, row_idx.shape[-1])[:max_batch_size_per_model]
                                )

                            topk_lp = np.concatenate(row_lps, axis=0).astype(np.float32)
                            topk_idx = np.concatenate(row_idxs, axis=0).astype(np.int32)
                        else:
                            topk_lp = tensor_to_numpy(lp_tensor).astype(np.float32) if lp_tensor is not None else np.zeros((max_batch_size_per_model, 32), dtype=np.float32)
                            topk_idx = tensor_to_numpy(idx_tensor).astype(np.int32) if idx_tensor is not None else np.zeros((max_batch_size_per_model, 32), dtype=np.int32)
                    else:
                        if lp_tensor is not None:
                            topk_lp = tensor_to_numpy(lp_tensor)
                            topk_lp = topk_lp.reshape(-1, topk_lp.shape[-1])[:max_batch_size_per_model].astype(np.float32)
                        else:
                            topk_lp = np.zeros((max_batch_size_per_model, 32), dtype=np.float32)

                        if idx_tensor is not None:
                            topk_idx = tensor_to_numpy(idx_tensor)
                            topk_idx = topk_idx.reshape(-1, topk_idx.shape[-1])[:max_batch_size_per_model].astype(np.int32)
                        else:
                            topk_idx = np.zeros((max_batch_size_per_model, 32), dtype=np.int32)

                    logits_list.append(logits_i)
                    log_probs_list.append((topk_lp, topk_idx))
                elif lp is not None:
                    lp_np = tensor_to_numpy(lp)
                    log_probs_i = lp_np.reshape(max_batch_size_per_model, 1, -1)
                    logits_list.append(logits_i)
                    log_probs_list.append(log_probs_i)
                else:
                    logits_list.append(logits_i)
                    log_probs_list.append(np.ones_like(logits_i))
            else:
                logits_i = tensor_to_numpy(tt_out[i])
                logits_i = logits_i.reshape(max_batch_size_per_model, 1, -1)[:, :, :vocab_size]
                logits_list.append(logits_i)
                log_probs_list.append(np.ones_like(logits_i))

        # Check for topk format
        has_topk = any(isinstance(lp, tuple) for lp in log_probs_list)

        if has_topk:
            normalized: List[Tuple[np.ndarray, np.ndarray]] = []
            for lp in log_probs_list:
                if isinstance(lp, tuple):
                    normalized.append(lp)
                else:
                    B = lp.shape[0]
                    normalized.append((
                        np.zeros((B, 32), dtype=np.float32),
                        np.zeros((B, 32), dtype=np.int32)
                    ))

            all_lp = np.concatenate([lp[0] for lp in normalized], axis=0)
            all_idx = np.concatenate([lp[1] for lp in normalized], axis=0)
            return (np.concatenate(logits_list, axis=0), (all_lp, all_idx))

        # All are ndarrays
        log_probs_arrays = [lp for lp in log_probs_list if isinstance(lp, np.ndarray)]
        return (np.concatenate(logits_list, axis=0), np.concatenate(log_probs_arrays, axis=0))

    def _get_prefill_user_page_table(
        self,
        page_table: np.ndarray,
        kv_cache: Any,
        prefill_len: Union[int, List[int]],
        trace_enabled: bool = False,
        prefill_seq_len: Optional[int] = None,
        use_batched_prefill: bool = False,
        user_id: Optional[Union[int, List[int]]] = None,
        padded_batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Get page table for prefill user."""
        block_size = get_block_size(kv_cache)

        if use_batched_prefill:
            batch_dim = (
                padded_batch_size if padded_batch_size is not None
                else getattr(self.model_args[0], 'max_batch_size', 32)
            )
            num_blocks = num_blocks_in_seq(prefill_seq_len or 0, block_size)
            page_table = page_table[:, :num_blocks]

            if trace_enabled and page_table.shape[1] < num_blocks:
                padding = np.full(
                    (page_table.shape[0], num_blocks - page_table.shape[1]),
                    -1, dtype=np.int32
                )
                page_table = np.concatenate([page_table, padding], axis=1)

            padded_page_table = np.full(
                (batch_dim, page_table.shape[1]), -1, dtype=np.int32
            )

            assert user_id is not None
            user_id_list = user_id if isinstance(user_id, list) else [user_id]
            for i, user in enumerate(user_id_list):
                if i < page_table.shape[0]:
                    padded_page_table[user, :] = page_table[i, :]

            return padded_page_table
        else:
            prefill_len_int = (
                prefill_len if isinstance(prefill_len, int) else prefill_len[0]
            )
            target_len = prefill_seq_len if prefill_seq_len is not None else prefill_len_int
            num_blocks = num_blocks_in_seq(target_len, block_size)

            if page_table.shape[1] < num_blocks:
                padding = np.full(
                    (1, num_blocks - page_table.shape[1]), -1, dtype=np.int32
                )
                page_table = np.concatenate([page_table, padding], axis=1)

            return page_table[:, :num_blocks]

    def generate(
        self,
        vision_images: Any,
        vision_mask: Any,
        prompt_tokens: List[int],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> Iterator[TokenResult]:
        """Generate tokens."""
        prefill_len = len(prompt_tokens)
        prompt_tokens_array = np.array(prompt_tokens, dtype=np.int64).reshape(1, -1)

        # Prefill
        prefill_result = self.prefill_forward_text(
            prompt_tokens_array,
            kv_cache=None,
            enable_trace=False,
        )

        if isinstance(prefill_result, tuple):
            logits = prefill_result[0]
        else:
            logits = prefill_result

        def sample(logits_arr: np.ndarray) -> Tuple[np.ndarray, str]:
            if temperature > 0:
                logits_shifted = logits_arr[:, -1] - np.max(logits_arr[:, -1], axis=-1, keepdims=True)
                exp_logits = np.exp(logits_shifted / temperature)
                probs = exp_logits / (np.sum(exp_logits, axis=-1, keepdims=True) + 1e-10)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = np.argmax(logits_arr[:, -1], axis=-1, keepdims=True)

            next_token = next_token.reshape(-1)

            decoder = self.tokenizer or self.processor
            if decoder is not None and hasattr(decoder, 'decode'):
                text = decoder.decode(next_token.tolist())
            else:
                text = f"<{next_token[0]}>"

            return next_token, text

        next_token, text = sample(logits)
        yield TokenResult(token=int(next_token[0]), text=text)

        for gen_idx in range(max_gen_len - 1):
            position_id = np.array([prefill_len + gen_idx])
            next_token_array = next_token.reshape(1, 1)

            decode_result = self.decode_forward(
                next_token_array,
                position_id,
                kv_cache=None,
                enable_trace=False,
            )

            if isinstance(decode_result, tuple):
                logits = decode_result[0]
            else:
                logits = decode_result

            next_token, text = sample(logits)
            yield TokenResult(token=int(next_token[0]), text=text)

            if text in ["<|eot_id|>", "<|end|>", "</s>"]:
                break

    def __del__(self) -> None:
        """Cleanup."""
        try:
            if hasattr(self, "trace_id_prefill"):
                self.trace_id_prefill.clear()
            if hasattr(self, "trace_id_prefill_sampling"):
                self.trace_id_prefill_sampling.clear()
            if hasattr(self, "trace_ids_decode"):
                self.trace_ids_decode.clear()
        except Exception:
            pass


# ============================================================================
# Utility Functions
# ============================================================================
def _mesh_shape_tuple(mesh_shape: Any) -> Tuple[int, ...]:
    """Convert mesh shape to tuple."""
    if hasattr(mesh_shape, '__iter__'):
        return tuple(int(dim) for dim in mesh_shape)
    return (int(mesh_shape),)


def _galaxy_data_parallel_submesh_shape(devices_per_group: int) -> Tuple[int, int]:
    """Get Galaxy DP submesh shape."""
    if devices_per_group >= 8 and devices_per_group % 8 == 0:
        return (devices_per_group // 8, 8)
    return (1, devices_per_group)


def create_submeshes(mesh_device: Any, data_parallel: int) -> List[Any]:
    """Create submeshes for data parallelism."""
    if data_parallel == 1:
        return [mesh_device]

    if hasattr(mesh_device, 'shape'):
        num_rows, num_cols = _mesh_shape_tuple(mesh_device.shape)
        num_devices = num_rows * num_cols

        assert num_devices % data_parallel == 0, \
            f"Unsupported split: {num_devices} devices, {data_parallel} groups"

        if num_devices == 32:
            if (num_rows, num_cols) != (4, 8):
                logger.info(f"Reshaping mesh from {(num_rows, num_cols)} to (4, 8)")
                if hasattr(mesh_device, 'reshape') and hasattr(ttnn, 'MeshShape'):
                    mesh_device.reshape(ttnn.MeshShape(4, 8))

            if hasattr(mesh_device, 'create_submeshes') and hasattr(ttnn, 'MeshShape'):
                submesh_shape = _galaxy_data_parallel_submesh_shape(
                    num_devices // data_parallel
                )
                return mesh_device.create_submeshes(
                    ttnn.MeshShape(*submesh_shape)
                )

        if hasattr(mesh_device, 'create_submeshes') and hasattr(ttnn, 'MeshShape'):
            return mesh_device.create_submeshes(
                ttnn.MeshShape(1, num_devices // data_parallel)
            )

    # Fallback
    return [mesh_device] * data_parallel